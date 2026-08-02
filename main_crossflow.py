# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from datasets import load_from_disk
import torch
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_crossflow import train_one_epoch, gen_one_epoch, compute_dinov3_pca, pure_recon_sample, pure_gen_sample
from PIL import Image
from perceptual.perceptual_loss import PerceptualLoss
from RAE_dinodisc import DinoDisc
import pickle
import os
from torchvision.datasets import ImageFolder

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--eval_epochs_interval', default=-1, type=int,
                        help='parameter_5')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--percloss', action='store_true',
                        help='Use perceptual loss')
    parser.add_argument('--percpure', action='store_true',
                        help='Purely use perceptual loss (no pixel-level loss component)')
    parser.add_argument('--percl1', action='store_true',
                        help='Use perceptual l1 loss')
    parser.add_argument('--perc_start_layer', type=int, default=1,
                        help='Start layer index for DINO perceptual loss feature extraction')
    parser.add_argument('--ganweight', type=float, default=0.1, help='Use GAN weight')
    parser.add_argument('--ganpure', action='store_true', help='Use GAN loss purely for generation')
    parser.add_argument('--gan_start_epoch', type=int, default=0,
                        help='Epoch from which g_loss is applied to the generator. '
                             'Before this epoch only the discriminator is trained.')
    parser.add_argument('--ganleak', type=float, default=1.0, help='ratio of gen/dis weight in GAN')
    parser.add_argument('--purerecon', action='store_true')
    parser.add_argument('--consdetach', action='store_true')


    # ViT architecture parameters
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size for ViT')
    parser.add_argument('--hiddim', default=768, type=int,
                        help='Hidden dimension of ViT')
    parser.add_argument('--code_dim', default=16, type=int,
                        help='Latent code dimension')
    parser.add_argument('--scale_shift', action='store_true')
    parser.add_argument('--twocond', action='store_true')
    parser.add_argument('--dec_depth', default=4, type=int,
                        help='Number of encoder transformer blocks')
    parser.add_argument('--enc_depth', default=4, type=int,
                        help='Number of encoder transformer blocks')
    parser.add_argument('--dis_depth', default=3, type=int,
                        help='Number of discriminator transformer blocks')
    parser.add_argument('--num_timetoken', default=8, type=int,
                        help='Number of time/global tokens')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--ganlr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--lowmomentum', action="store_true", help='low momentum for AdamW and Muon')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--out_path', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--gen_only', action='store_true',
                        help='whether to generate only')
    parser.add_argument('--computepca', action='store_true',
                        help='whether to compute PCA')

    parser.add_argument('--debug', action='store_true', help='flag for running debugging experiments')

    # VAE selection
    parser.add_argument('--vae_type', default='vavae', type=str, choices=['vavae', 'fluxv2'],
                        help='VAE model to use: "vavae" for Vavae.AutoencoderKL, "fluxv2" for flux2vae.Fluxv2vae_wrapper')

    return parser

def ema_update(ema_model, model, alpha = 0.999):    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)

def main(args):
    #handle_flash_attn()
    args.output_dir = f'{args.log_dir}/checkpoint'
    misc.init_distributed_mode(args)

    if misc.is_main_process():
        if not os.path.exists(args.output_dir):
            os.system('mkdir -p {}'.format(args.output_dir))
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    dataset_train = load_from_disk(args.data_path)
    dataset_val = load_from_disk(args.val_data_path)
    dataset_train.set_transform(lambda ex: {"img": [transform_train(img.convert("RGB")) for img in ex["image"]], "label": torch.tensor(ex["label"], dtype=torch.long)})
    dataset_val.set_transform(lambda ex: {"img": [transform_train(img.convert("RGB")) for img in ex["image"]], "label": torch.tensor(ex["label"], dtype=torch.long)})

    '''
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
    dinov3model = DINOv3ViTModel.from_pretrained("./dinov3").to(device)
    dinov3model.eval()
    for param in dinov3model.parameters():
        param.requires_grad = False
    dinov3model = torch.compile(dinov3model)
    '''
    if args.vae_type == 'vavae':
        from Vavae import AutoencoderKL
        vavae = AutoencoderKL(embed_dim=32, ch_mult=(1, 1, 2, 2, 4), ckpt_path='vavae-imagenet256-f16d32-dinov2.pt').to(device)
    elif args.vae_type == 'fluxv2':
        from flux2vae import Fluxv2vae_wrapper
        vavae = Fluxv2vae_wrapper(embed_dim=32, ch_mult=(1, 1, 2, 2, 4)).to(device)
    else:
        raise ValueError(f"Unknown vae_type: {args.vae_type}")
    vavae.eval()
    for param in vavae.parameters():
        param.requires_grad = False
    vavae = torch.compile(vavae)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    #if global_rank == 0 and args.log_dir is not None:
    #    os.makedirs(args.log_dir, exist_ok=True)
    #    log_writer = SummaryWriter(log_dir=args.log_dir)
    #else:
    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers = True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=250,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )

    if args.computepca:
        compute_dinov3_pca(data_loader_train, device, vavae, vae_type=args.vae_type)
        exit()

    from crossflow_withgan import ViT 
    pca_path = f'{args.vae_type}_pca.pth'
    vit_kwargs = dict(
        patch_size=args.patch_size,
        hidden_size=args.hiddim,
        code_dim=args.code_dim,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        num_heads=args.hiddim//64,
        num_timetoken=args.num_timetoken,
        scale_shift=args.scale_shift,
        discriminator_depth=args.dis_depth,
        twocond=args.twocond,
        pca_path=pca_path
    )
    vae = ViT(**vit_kwargs)
    vae.to(device)

    perceptual_loss = None if not args.percloss else PerceptualLoss(use_l1=args.percl1, start_layer=args.perc_start_layer).to(device)

    # --- DinoDisc discriminator (DDP-wrapped for gradient sync) ---
    if args.ganweight > 0:
        dino_disc = DinoDisc(device=device, dino_model_path="./dinov3", num_classes=1000).to(device)
        dino_disc = torch.nn.parallel.DistributedDataParallel(
            dino_disc, device_ids=[args.gpu], find_unused_parameters=False, broadcast_buffers=False
        )
        disc_optim = torch.optim.AdamW(
            [_ for _ in dino_disc.parameters() if _.requires_grad], lr=args.ganlr, weight_decay=0., betas=(0.5, 0.9)
        )
    else:
        dino_disc = None
        disc_optim = None
    
    # Multiple EMA models with different decay rates
    ema_rates = [0.999, 0.9995, 0.9999]
    ema_vaes = []
    for rate in ema_rates:
        ema_vae = ViT(**vit_kwargs)
        ema_vae.to(device)
        for param in ema_vae.parameters():
            param.requires_grad = False
        ema_vae.eval()
        ema_vaes.append(ema_vae)
    # Keep a reference to the first one for backward compatibility (used in main_loss_backward)
    ema_vae = ema_vaes[0]

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    vae = torch.nn.parallel.DistributedDataParallel(vae, device_ids=[args.gpu], find_unused_parameters=False, broadcast_buffers=False)
    from muon import MuonAdamWOptimizer
    vae_optim = MuonAdamWOptimizer(vae, lr=args.lr, weight_decay=args.weight_decay, adamw_kwargs={"betas": (0.9, 0.95)}, low_momentum_for_discriminator=args.lowmomentum)
    vae_scaler = torch.cuda.amp.GradScaler()
    for ev in ema_vaes:
        ema_update(ev, vae.module, 0)
    misc.load_model(args=args, model_without_ddp=vae, ema_models=ema_vaes, ema_rates=ema_rates, optimizer=vae_optim, loss_scaler=vae_scaler)
    # Load DinoDisc checkpoint if resuming
    if args.resume and dino_disc is not None:
        ckpt_path = args.resume
        if ckpt_path and ckpt_path != 'last' and not ckpt_path.startswith('https'):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'dino_disc' in ckpt:
                dino_disc.load_state_dict(ckpt['dino_disc'], strict=False)
                print("Loaded DinoDisc state from checkpoint")
    
    print(f"Start training at {args.start_epoch} for total {args.epochs} epochs")
    start_time = time.time()

    if args.gen_only:
        # Build evaluation task list: (model, label_str, eval_type)
        eval_tasks = []
        for rate, ev in zip(ema_rates, ema_vaes):
            eval_tasks.append((ev, f'EMA {rate}', 'gen'))
            eval_tasks.append((ev, f'EMA {rate}', 'recon'))
        eval_tasks.append((vae.module, 'raw model (no EMA)', 'gen'))
        eval_tasks.append((vae.module, 'raw model (no EMA)', 'recon'))

        num_tasks = len(eval_tasks)
        # Each rank stores its FID result; non-participating ranks keep 0
        fid_results = torch.zeros(num_tasks, device=device)

        train_sample = next(iter(dataset_train))["img"].to(device)
        # Distribute tasks across ranks: rank i handles task i (if i < num_tasks)
        for task_idx, (model, label_str, eval_type) in enumerate(eval_tasks):
            if args.rank == task_idx:
                print(f"\n[Rank {args.rank}] Evaluating {label_str} — {eval_type}")
                if eval_type == 'gen':
                    _, fid_score = pure_gen_sample(args.out_path, train_sample, model, vavae, 50000, batch_size=1024, image_size=256, num_classes=1000, compute_fid=True)
                else:
                    fid_score = pure_recon_sample(args.out_path, train_sample, model, vavae, 50000, data_loader_val, device, compute_fid=True)
                fid_results[task_idx] = fid_score

        # Gather all FID results to all ranks via all_reduce (SUM, since each slot is written by exactly one rank)
        torch.distributed.all_reduce(fid_results, op=torch.distributed.ReduceOp.SUM)

        # Rank 0 prints the summary table
        if args.rank == 0:
            print(f"\n{'='*60}")
            print(f"FID Evaluation Summary (gen_only)")
            print(f"{'='*60}")
            for task_idx, (_, label_str, eval_type) in enumerate(eval_tasks):
                print(f"  [{label_str}] {eval_type} FID = {fid_results[task_idx].item():.4f}")
            print(f"{'='*60}")

        # All ranks synchronize after evaluation
        torch.distributed.barrier()
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            ema_vaes, ema_rates, vae, perceptual_loss, data_loader_train,
            vae_optim, device, epoch, vae_scaler, 
            log_writer=log_writer,
            args=args, dinov3model=vavae, valdataloader = data_loader_val,
            dino_disc=dino_disc, disc_optim=disc_optim
        )

        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            output_dir = Path(args.log_dir)
            checkpoint_path = output_dir / ('checkpoint-{:05d}.pth'.format(epoch))
            to_save = {
                'vae': vae.state_dict(),
                'vae_optim': vae_optim.state_dict(),
                'vae_scaler': vae_scaler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if dino_disc is not None:
                to_save['dino_disc'] = dino_disc.state_dict()
                to_save['disc_optim'] = disc_optim.state_dict()
            # Save all EMA models keyed by their decay rate
            for rate, ev in zip(ema_rates, ema_vaes):
                to_save[f'ema_vae_{rate}'] = ev.state_dict()
            # Backward compatibility: also save the first one as 'ema_vae'
            to_save['ema_vae'] = ema_vaes[0].state_dict()
            misc.save_on_master(to_save, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        #if epoch % 100 == 0 and epoch != 0:
        #    #with torch.cuda.amp.autocast(enabled = bool(args.temp5)):
        #    misc.eval_model(args, epoch, model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
