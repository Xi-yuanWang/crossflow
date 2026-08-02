# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math, os, pdb
import numpy as np
from typing import Iterable

import torch, torchvision
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from transformer_utils import RMSNorm
from torch.autograd import Function
class ClipGradFunction(Function):
    @staticmethod
    def forward(ctx, input, t):
        ctx.save_for_backward(t)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        t = ctx.saved_tensors[0]
        grad_output = grad_output * (t.square().reshape(-1, 1, 1, 1))
        return grad_output, None

DIFF_BASE=4096
EPS = 1e-5

def normalize(x, eps = EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def noisy_code(code, sigma, noise = None):
    return normalize(code + (sigma).reshape(-1,1,1) * noise)

def sample_time_steps(batch_size, device, fraction_equal = 0.75):
    """Sample time steps (r, t) according to the configured sampler"""
    # Step1: Sample two time points
    time_samples = torch.rand(batch_size, 2, device=device)
    # Step2: Ensure t > r by sorting
    sorted_samples, _ = torch.sort(time_samples, dim=1)
    r, t = sorted_samples[:, 0], sorted_samples[:, 1]
    # Create a mask for samples where r should equal t
    equal_mask = torch.rand(batch_size, device=device) < fraction_equal
    # Apply the mask: where equal_mask is True, set r=t (replace)
    r = torch.where(equal_mask, t, r)
    
    return r, t 
from lerp import renormalized_lerp_retime
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def beta(t):
    return 1 - t

def dotbeta(t):
    return -1

def alpha(t):
    return 1 - beta(t)

def dotalpha(t):
    return -dotbeta(t)

def dratio(t, r):
    ret = (beta(t)-beta(r))*beta(t)/dotbeta(t)/(torch.clamp_min(beta(r), 0.05))
    return ret

def lweight(t, r):
    t = torch.clamp_max(t, 0.95)
    ret = dotbeta(t)/beta(t)
    return ret

def tweight(t, r):
    ret = alpha(t) - beta(t)*dotalpha(t)/dotbeta(t)# for alpha(t)+beta(t)=1, it=1, for alpha(t)^2+beta(t)^2=1, it=(1-beta(t)^2)^{-0.5}
    return ret

def veloweight1(t):
    t = torch.clamp_max(t, 0.95)
    ret = dotalpha(t) - alpha(t)*dotbeta(t)/beta(t)# for alpha(t)+beta(t)=1, it=1, for alpha(t)^2+beta(t)^2=1, it=(1-beta(t)^2)^{-0.5}
    return ret

def veloweight2(t):
    t = torch.clamp_max(t, 0.95)
    ret = dotbeta(t)/beta(t)# for alpha(t)+beta(t)=1, it=1, for alpha(t)^2+beta(t)^2=1, it=(1-beta(t)^2)^{-0.5}
    return ret

def compute_dinov3_feature(x, dinov3model, return_recon=False):
    '''
    x: (B, C, H, W), in [-1, 1]
    '''
    inputx = x #(0.5 * x + 0.5 - IMAGENET_MEAN)/IMAGENET_STD
    ret = dinov3model(inputx, return_recon=return_recon)
    return ret

MAXTIME = 1.

def timesampler(shape, device, dtype):
    return MAXTIME * torch.rand(shape, device=device, dtype=dtype)#torch.where(torch.rand(shape, device=device, dtype=dtype) > 0.1, torch.sigmoid(torch.randn(shape, device=device, dtype=dtype)), torch.rand(shape, device=device, dtype=dtype))

def cfgsampler(batchsize, device, dtype):
    return MAXTIME * torch.rand((batchsize, 1, 1), device=device, dtype=dtype)

@torch.compile
def main_loss_backward(x, label, vae, vae_detach, ema_vae, perceptual_loss, frac_epoch, args, dinov3model, dino_disc=None, pretrain=False, ganweight=0.):
    f_l2loss = lambda x: x.square().mean()
    f_l1loss = lambda x: (x.square() + EPS).sqrt().mean() - EPS**0.5
    f_vl1loss = lambda x: (x.square().mean(dim=(-1, -2, -3)) + EPS).sqrt().mean(dim=0) - EPS**0.5
    f_vhuberloss = lambda x: (x.square().mean(dim=(-1, -2, -3)) + 1).sqrt().mean(dim=0) - 1**0.5

    f_zvl1loss = lambda x: (x.square().mean(dim=(-1, -2)) + EPS).sqrt().mean(dim=0) - EPS**0.5
    f_zvhuberloss = lambda x: (x.square().mean(dim=(-1, -2)) + 1).sqrt().mean(dim=0) - 1**0.5
    
    B = x.shape[0]
    t1, t2, t3 = timesampler((B, 1, 1), device=x.device, dtype=x.dtype), timesampler((B, 1, 1), device=x.device, dtype=x.dtype), timesampler((B, 1, 1), device=x.device, dtype=x.dtype)
    t1, t2 = torch.minimum(t1, t2), torch.maximum(t1, t2)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        rawx = compute_dinov3_feature(x, dinov3model)
        z = vae_detach.rep_reducer(rawx)
    
    xrecon, zrecon = vae(alpha(t3).reshape(-1, 1, 1) * z + beta(t3).reshape(-1, 1, 1) * torch.randn_like(z), t3, t3, label)
    zreconloss = f_l2loss(lweight(t3, t3).reshape(-1, 1, 1)*(zrecon - tweight(t3, t3).reshape(-1, 1, 1)*z.detach()))
    xreconloss = f_vhuberloss(lweight(t3, t3).reshape(-1, 1, 1, 1)*(xrecon - tweight(t3, t3).reshape(-1, 1, 1, 1)*x))
    if args.percloss:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            perc_recon = perceptual_loss(x, xrecon).flatten()
            xreconloss = perc_recon if args.percpure else xreconloss + perc_recon

    z_t = alpha(t1).reshape(-1, 1, 1) * z + beta(t1).reshape(-1, 1, 1) * torch.randn_like(z)
    z_t = z_t.detach()
    if True: # not pretrain:
        with torch.no_grad():
            if True:# pretrain:
                xtar, ztar = x, z.detach()
            else:
                xtar, ztar = ema_vae(z_t, t1, t1, label)
            zvelo = veloweight1(t1).reshape(-1, 1, 1) * ztar + veloweight2(t1).reshape(-1, 1, 1) * z_t
            def vae_decoder(z, t):
                return vae_detach(z, t, t2, label)[0]
            xcons, directional_derivative = torch.func.jvp(vae_decoder, (z_t, t1,), (zvelo, torch.ones_like(t1),))
        xcons = vae(z_t.detach(), t1, t2, label)[0]
        xconsloss = f_vhuberloss(lweight(t1, t2).reshape(-1, 1, 1, 1)*(xcons + dratio(t1, t2).reshape(-1, 1, 1, 1)*directional_derivative.detach() - tweight(t1, t2).reshape(-1, 1, 1, 1)*xtar))
        if args.percloss:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                perc_cons = perceptual_loss(tweight(t1, t2).reshape(-1, 1, 1, 1)*x, xcons + dratio(t1, t2).reshape(-1, 1, 1, 1)*directional_derivative.detach())
                xconsloss = perc_cons if args.percpure else xconsloss + perc_cons
    else:
        xconsloss = torch.zeros_like(xreconloss)
    # --- DinoDisc GAN loss ---
    if dino_disc is not None:
        # Generate fake images at t2=1 (full denoising) from z_t
        tgan = torch.rand_like(t1)# torch.sqrt(1/(1+GANEPOCH) + (GANEPOCH/(1+GANEPOCH))*torch.rand_like(t1))
        if not args.ganpure:
            fake_img = vae(alpha(tgan).reshape(-1, 1, 1) * z + beta(tgan).reshape(-1, 1, 1) * torch.randn_like(z), tgan, torch.ones_like(tgan), label)[0]
        else:
            fake_img = vae(torch.randn_like(z), torch.zeros_like(tgan), torch.ones_like(tgan), label)[0]
        leakratio = tgan.square().square().reshape(-1, 1, 1, 1)
        fake_img = leakratio * fake_img + (1 - leakratio) * fake_img.detach()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            g_loss, act_cache = dino_disc(real=x, fake=fake_img, class_labels=label, mode="g_loss")
    else:
        g_loss = torch.zeros(1, device=x.device)
        act_cache = None

    loss = 0.*zreconloss + xreconloss + xconsloss + ganweight * g_loss
    loss.backward()
    return zreconloss, xreconloss, xconsloss, g_loss.detach(), act_cache


def d_loss_backward(dino_disc, act_cache):
    d_loss = dino_disc(act_cache=act_cache, mode="d_loss")
    d_loss.backward()
    return d_loss

@torch.no_grad()
def pure_recon_sample(folder_path, samples, vae, dinov3model, number, valdataloader, device, batch_size=1024, compute_fid=True):
    from tqdm import tqdm
    from adm_evaluator_pytorch import Evaluator
    evaluator = Evaluator(device)
    # Pre-allocate the full image array (NHWC, uint8)
    all_images = np.empty((number, 256, 256, 3), dtype=np.uint8)
    offset = 0

    for samples in tqdm(valdataloader, desc='reconstruct samples'):
        if offset >= number:
            break
        samples, labels = samples["img"].to(device), samples["label"].to(device)
        code = compute_dinov3_feature(samples, dinov3model)
        code = vae.rep_reducer(code)
        xtar, ztar = vae(code, MAXTIME*torch.ones((samples.shape[0], 1, 1), device=device), MAXTIME*torch.ones((samples.shape[0], 1, 1), device=device), labels)
        rec =  xtar
        gens = 0.5*rec.clamp(min = -1, max = 1) + 0.5
        # Convert to NHWC uint8 numpy for ADM evaluator
        imgs_np = (gens * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        count = min(imgs_np.shape[0], number - offset)
        all_images[offset:offset + count] = imgs_np[:count]
        offset += count

    all_images = all_images[:offset]  # trim if fewer than `number` were produced

    if not os.path.exists(folder_path):   
        os.makedirs(folder_path)
    img_dir = os.path.join(folder_path, 'reconimages')
    os.makedirs(img_dir, exist_ok=True)
    num_save = min(10, gens.shape[0])
    indices = torch.randperm(gens.shape[0])[:num_save]
    for cnt, idx in enumerate(indices):
        img = gens[idx]  # (C, H, W)
        torchvision.utils.save_image(img, os.path.join(img_dir, f'{cnt:08d}.png'))

    # Compute FID score using ADM evaluator
    fid_stats_path = "./val_imagenet256_stats.npz"
    def batch_iterator(arr, bs=evaluator.batch_size):
        for i in range(0, len(arr), bs):
            yield arr[i:i+bs]
    sample_acts = evaluator.compute_activations(batch_iterator(all_images))
    sample_stats, sample_stats_spatial = tuple(evaluator.compute_statistics(x) for x in sample_acts)
    ref_obj = np.load(fid_stats_path)
    from adm_evaluator_pytorch import FIDStatistics
    ref_stats = FIDStatistics(ref_obj["mu"], ref_obj["sigma"])
    fid_score = sample_stats.frechet_distance(ref_stats)
    print(f'[pure_recon_sample] FID={fid_score:.4f}')
    if False: #"mu_s" in ref_obj:
        ref_stats_spatial = FIDStatistics(ref_obj["mu_s"], ref_obj["sigma_s"])
        sfid_score = sample_stats_spatial.frechet_distance(ref_stats_spatial)
        print(f'[pure_recon_sample] sFID={sfid_score:.4f}')
    return fid_score

@torch.no_grad()
def pure_gen_sample(folder_path, samples, vae, dinov3model, number, batch_size=1024,
                    image_size=256, num_classes=1000, num_step=1, compute_fid=True):
    from tqdm import tqdm
    out = []
    if samples.ndim < 4:
        samples = samples.unsqueeze(0)
    code = vae.rep_reducer(compute_dinov3_feature(samples[:1], dinov3model)).repeat(batch_size, 1, 1)
    ttable = torch.linspace(0, MAXTIME, num_step + 1, device = code.device)
    from adm_evaluator_pytorch import Evaluator
    evaluator = Evaluator(code.device)
    # Pre-allocate the full image array (NHWC, uint8)
    all_images = np.empty((number, image_size, image_size, 3), dtype=np.uint8)
    offset = 0

    label = torch.repeat_interleave(torch.arange(0, 1000, device = code.device), number//1000)
    for i in tqdm(range(0, number, batch_size), desc='Generating samples'):
        z = torch.randn_like(code)
        tlabel = label[i:i+batch_size]
        B = tlabel.shape[0]
        z = z[:B]
        for step in range(num_step):
            t1, t2 = ttable[step], ttable[step+1]
            xtar, ztar = vae(z, t1.expand(B, 1, 1), t2.expand(B, 1, 1), tlabel)
            z = (1 - beta(t2)/beta(t1)) * ztar + beta(t2)/beta(t1) * z
        rec =  xtar
        gens = 0.5*rec.clamp(min = -1, max = 1) + 0.5
        # Convert to NHWC uint8 numpy for ADM evaluator
        imgs_np = (gens * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        count = min(imgs_np.shape[0], number - offset)
        all_images[offset:offset + count] = imgs_np[:count]
        offset += count

    if not os.path.exists(folder_path):   
        os.makedirs(folder_path)
    img_dir = os.path.join(folder_path, 'images')
    os.makedirs(img_dir, exist_ok=True)
    num_save = min(10, gens.shape[0])
    indices = torch.randperm(gens.shape[0])[:num_save]
    for cnt, idx in enumerate(indices):
        img = gens[idx]  # (C, H, W)
        torchvision.utils.save_image(img, os.path.join(img_dir, f'{cnt:08d}.png'))

    # Compute FID score using ADM evaluator
    fid_stats_path = "./VIRTUAL_imagenet256_labeled.npz"
    def batch_iterator(arr, bs=evaluator.batch_size):
        for i in range(0, len(arr), bs):
            yield arr[i:i+bs]
    sample_acts = evaluator.compute_activations(batch_iterator(all_images))
    sample_stats, sample_stats_spatial = tuple(evaluator.compute_statistics(x) for x in sample_acts)
    ref_obj = np.load(fid_stats_path)
    from adm_evaluator_pytorch import FIDStatistics
    ref_stats = FIDStatistics(ref_obj["mu"], ref_obj["sigma"])
    fid_score = sample_stats.frechet_distance(ref_stats)
    print(f'[pure_gen_sample] FID={fid_score:.4f}')
    if "mu_s" in ref_obj:
        ref_stats_spatial = FIDStatistics(ref_obj["mu_s"], ref_obj["sigma_s"])
        sfid_score = sample_stats_spatial.frechet_distance(ref_stats_spatial)
        print(f'[pure_gen_sample] sFID={sfid_score:.4f}')
    return out, fid_score



@torch.no_grad()
def gen_sample(folder_path, img_name, samples, vae, dinov3model, noise = None):
    samples, label = samples
    samples = samples[:10]
    label = label[:10]
    if not os.path.exists(folder_path):   
        os.makedirs(folder_path)

    code = vae.rep_reducer(compute_dinov3_feature(samples, dinov3model))

    time_dist_shift = math.sqrt(code.shape[1] * code.shape[2] / DIFF_BASE) 
    t = torch.linspace(0, 1, 10).to(code)
    t = (time_dist_shift * t / (1 + (time_dist_shift - 1) * t))
    t = 1 - t
    t = MAXTIME * t

    out = [samples]
    if noise is None:
        noise = torch.randn_like(code)
    noise = noise[:, :, -code.shape[-1]:]
    for t_val in t:
        zt = code * alpha(t_val) + beta(t_val) * noise
        rec, ztar = vae(zt, t_val.expand(code.shape[0], 1, 1), MAXTIME*torch.ones_like(t_val).expand(code.shape[0], 1, 1), label)
        out.append(rec.clamp(min = -1, max = 1))

    out = torch.stack(out, dim = 1).clamp(min = -1, max = 1) * 0.5 + 0.5
    torchvision.utils.save_image(out.flatten(0,1), fp = os.path.join(folder_path, img_name), nrow = out.shape[1])


@torch.no_grad()
def multi_step_gen_sample(folder_path, img_name, samples, vae, noise = None):
    samples = samples[:10]
    if not os.path.exists(folder_path):   
        os.makedirs(folder_path)

    code, enc_cache = vae(x = samples, mode = 'encode')

    time_dist_shift = math.sqrt(code.shape[1] * code.shape[2] / DIFF_BASE) 
    t = torch.linspace(0, 1, 10).to(code)
    t = (time_dist_shift * t / (1 + (time_dist_shift - 1) * t))
    t = 1 - t
    t = t.flip(dims=(-1,))

    t = MAXTIME * t

    if noise is None:
        noise = torch.randn_like(code)
    z = noise[:, :, :code.shape[-1]]
    out = []
    for i in range(len(t) - 1):
        t1, t2 = t[i], t[i+1]
        with torch.cuda.amp.autocast():
            ztar, rec = vae(code = z, enc_cache = enc_cache, mode = 'decode', t1=t1.expand(code.shape[0], 1, 1), t2=torch.ones_like(t2.expand(code.shape[0], 1, 1)))
            z = (1 - beta(t2)/beta(t1)) * ztar + beta(t2)/beta(t1) * z
            out.append(rec.clamp(min = -1, max = 1))

    out = torch.stack(out, dim = 1).clamp(min = -1, max = 1) * 0.5 + 0.5
    torchvision.utils.save_image(out.flatten(0,1), fp = os.path.join(folder_path, img_name), nrow = out.shape[1])

@torch.no_grad()
def noise_step_gen_sample(folder_path, img_name, samples, vae, noise = None):
    samples = samples[:10]
    if not os.path.exists(folder_path):   
        os.makedirs(folder_path)

    code, enc_cache = vae(x = samples, mode = 'encode')

    time_dist_shift = math.sqrt(code.shape[1] * code.shape[2] / DIFF_BASE) 
    t = torch.linspace(0, 1, 10).to(code)
    t = (time_dist_shift * t / (1 + (time_dist_shift - 1) * t))
    t = 1 - t
    t = t.flip(dims=(-1,))
    t = MAXTIME * t

    if noise is None:
        noise = torch.randn_like(code)
    z = noise[:, :, :code.shape[-1]]
    out = []
    for i in range(len(t) - 1):
        t1, t2 = t[i], t[i+1]
        _, rec = vae(code = z, enc_cache = enc_cache, mode = 'decode', t1=torch.zeros_like(t2.expand(code.shape[0], 1, 1)), t2=t2.expand(code.shape[0], 1, 1))
        out.append(rec.clamp(min = -1, max = 1))

    out = torch.stack(out, dim = 1).clamp(min = -1, max = 1) * 0.5 + 0.5
    torchvision.utils.save_image(out.flatten(0,1), fp = os.path.join(folder_path, img_name), nrow = out.shape[1])


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def ema_update(ema_model, model, alpha = 0.999):    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)

def gen_one_epoch(gen_samples, vae_module, epoch, device, args, dinov3model, puregen=False):
    if gen_samples is None:
        gen_samples, gen_noise = torch.load('./noise.pth', map_location=device)
    else:
        gen_noise = None
    if puregen:
        pure_gen_sample(args.out_path, gen_samples, vae_module, dinov3model, 50000, batch_size=1024, image_size=256, num_classes=1000, compute_fid=True)
    else:
        gen_sample(args.out_path, f'{epoch:05d}_rec.png', gen_samples, vae_module, dinov3model, gen_noise)
    #multi_step_gen_sample(args.out_path, f'{epoch:05d}_nfe.png', gen_samples, vae_module, gen_noise)
    #noise_step_gen_sample(args.out_path, f'{epoch:05d}_n0r.png', gen_samples, vae_module, gen_noise)


@torch.no_grad()
def compute_dinov3_pca(dataloader, device, dinov3model, vae_type='vavae'):
    xmean = 0.
    xsq = 0.
    ALLB = 0
    from tqdm import tqdm
    for idx, data in tqdm(enumerate(dataloader)):
        samples = data["img"]
        samples = samples.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            px = compute_dinov3_feature(samples, dinov3model) # patchify, (B, C, H, W) -> (B, L, d)
        px = px.flatten(0, 1).to(torch.float)
        B = px.shape[0]

        running_xmean = px.mean(dim=0)
        px = B**(-0.5) * px
        running_xsq = px.T @ px

        ALLB = ALLB + B
        xmean = xmean + (running_xmean - xmean) * (B / ALLB)
        xsq = xsq + (running_xsq - xsq) * (B / ALLB)
    
    xsq = xsq - xmean.reshape(-1, 1) @ xmean.reshape(1, -1)
    eigval, eigvec = torch.linalg.eigh(xsq)
    pca_save_path = f'{vae_type}_pca.pth'
    torch.save((xmean, eigval, eigvec), pca_save_path)
    print(f'PCA saved to {pca_save_path}')


def train_one_epoch(ema_vaes, ema_rates, vae, perceptual_loss,
                    data_loader: Iterable, vae_optim,
                    device: torch.device, epoch: int, vae_scaler,
                    log_writer=None,
                    args=None, dinov3model=None, valdataloader=None,
                    dino_disc=None, disc_optim=None):
    # Use the first EMA model as the primary one for main_loss_backward
    vae.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    vae_optim.zero_grad()

    lendata = len(data_loader)

    genparams, disparams = [], []
    for n, p in vae.named_parameters():
        if 'discriminator' in n:
            disparams.append(p)
        else:
            genparams.append(p)
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        frac_epoch = epoch + data_iter_step / lendata

        samples = data["img"].to(device, non_blocking=True)
        label = data["label"].to(device, non_blocking=True)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(vae_optim, data_iter_step / len(data_loader) + epoch, args)
            if disc_optim is not None:
                lr_sched.adjust_learning_rate(disc_optim, data_iter_step / len(data_loader) + epoch, args)

        zreconloss, xrecon_loss, xconsloss, g_loss, act_cache = main_loss_backward(samples, label, vae, vae.module, ema_vaes[0], perceptual_loss, frac_epoch, args, dinov3model, dino_disc=dino_disc, pretrain=epoch<=10, ganweight=args.ganweight if frac_epoch >= args.gan_start_epoch else 0.)
        dis_grad_norm = torch.nn.utils.clip_grad_norm_(disparams, 1.0)
        gen_grad_norm = torch.nn.utils.clip_grad_norm_(genparams, 1.0)
        vae_optim.step()
        vae_optim.zero_grad()

        # --- DinoDisc discriminator step ---
        if dino_disc is not None and disc_optim is not None:
            disc_optim.zero_grad()
            d_loss = d_loss_backward(dino_disc, act_cache)
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(dino_disc.parameters(), 1.0)
            disc_optim.step()
        else:
            disc_grad_norm = torch.tensor(0.0)
            d_loss = torch.tensor(0.0)

        metric_logger.update(zrecons_loss=zreconloss)
        metric_logger.update(xrecon_loss=xrecon_loss)
        metric_logger.update(xcons_loss=xconsloss)
        metric_logger.update(g_loss=g_loss)
        metric_logger.update(d_loss=d_loss.detach() if torch.is_tensor(d_loss) else d_loss)
        metric_logger.update(disc_grad_norm=disc_grad_norm)
        metric_logger.update(dis_grad_norm = dis_grad_norm)
        metric_logger.update(gen_grad_norm = gen_grad_norm)
        lr = vae_optim.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        # Update all EMA models with their respective rates
        for ev, rate in zip(ema_vaes, ema_rates):
            ema_update(ev, vae.module, rate if epoch > 10 else 0.999)

    if args.rank == 0:
        gen_one_epoch((samples, label), ema_vaes[0], epoch, device, args, dinov3model)

    if (epoch + 1) % 10 == 0:
        # Build evaluation task list: (model, label_str, eval_type)
        # eval_type: 'gen' for generation FID, 'recon' for reconstruction FID
        eval_tasks = []
        for rate, ev in zip(ema_rates, ema_vaes):
            eval_tasks.append((ev, f'EMA {rate}', 'gen'))
        eval_tasks.append((vae.module, 'raw model (no EMA)', 'gen'))
        eval_tasks.append((ema_vaes[0], 'EMA 0.999', 'recon'))

        num_tasks = len(eval_tasks)
        # Each rank stores its FID result; non-participating ranks keep 0
        fid_results = torch.zeros(num_tasks, device=device)

        # Distribute tasks across ranks: rank i handles task i (if i < num_tasks)
        for task_idx, (model, label_str, eval_type) in enumerate(eval_tasks):
            if args.rank == task_idx:
                print(f"\n--- Epoch {epoch}, {label_str}, {eval_type} (rank {args.rank}) ---")
                if eval_type == 'gen':
                    _, fid_score = pure_gen_sample(args.out_path, samples, model, dinov3model, 50000, batch_size=1024, image_size=256, num_classes=1000, compute_fid=True)
                else:
                    fid_score = pure_recon_sample(args.out_path, samples, model, dinov3model, 50000, valdataloader, device, compute_fid=True)
                fid_results[task_idx] = fid_score

        # Gather all FID results to all ranks via all_reduce (SUM, since each slot is written by exactly one rank)
        torch.distributed.all_reduce(fid_results, op=torch.distributed.ReduceOp.SUM)

        # Rank 0 prints the summary table
        if args.rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} — FID Evaluation Summary")
            print(f"{'='*60}")
            for task_idx, (_, label_str, eval_type) in enumerate(eval_tasks):
                print(f"  [{label_str}] {eval_type} FID = {fid_results[task_idx].item():.4f}")
            print(f"{'='*60}")

        # All ranks synchronize after evaluation
        torch.distributed.barrier()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
