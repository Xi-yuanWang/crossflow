import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder

import util.misc as misc
from Vavae import AutoencoderKL, center_crop_arr


def get_args_parser():
    parser = argparse.ArgumentParser("Train VAVAE with pixel L2 and KL losses only", add_help=False)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--embed_dim", default=32, type=int)
    parser.add_argument("--kl_weight", default=1e-6, type=float)
    parser.add_argument("--init_ckpt", default="", type=str,
                        help="Optional checkpoint used to initialize the VAVAE architecture")
    parser.add_argument("--model_type", default="vavae", choices=["vavae", "marvae"], type=str)

    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--lr", default=8e-5, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--data_path", default="../imagenet/train/", type=str)
    parser.add_argument("--val_data_path", default="../imagenet/validation/", type=str)
    parser.add_argument("--output_dir", default="./log_dir_vavae_l2kl/checkpoint", type=str)
    parser.add_argument("--log_dir", default="./log_dir_vavae_l2kl", type=str)
    parser.add_argument("--sample_dir", default="./work_dir_vavae_l2kl", type=str)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--num_val_batches", default=50, type=int)
    parser.add_argument("--num_sample_images", default=16, type=int)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--amp_dtype", default="bf16", choices=["none", "bf16", "fp16"], type=str)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)

    return parser


def build_transform(image_size, train=True):
    transform_list = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
    ]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(transform_list)


def build_dataset(path, transform):
    try:
        dataset = load_from_disk(path)
        dataset.set_transform(
            lambda ex: {
                "img": [transform(img.convert("RGB")) for img in ex["image"]],
                "label": torch.tensor(ex["label"], dtype=torch.long) if "label" in ex else torch.zeros(len(ex["image"]), dtype=torch.long),
            }
        )
        return dataset
    except Exception:
        return ImageFolder(path, transform=transform)


def get_images(batch, device):
    if isinstance(batch, dict):
        return batch["img"].to(device, non_blocking=True)
    return batch[0].to(device, non_blocking=True)


def get_autocast_context(device, amp_dtype):
    if device.type != "cuda" or amp_dtype == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype, enabled=True)


def adjust_learning_rate(optimizer, epoch_progress, args):
    if epoch_progress < args.warmup_epochs:
        lr = args.lr * epoch_progress / max(args.warmup_epochs, 1e-8)
    else:
        progress = (epoch_progress - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VAVAEL2KLTrainer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vae = AutoencoderKL(
            embed_dim=args.embed_dim,
            ch_mult=(1, 1, 2, 2, 4),
            use_variational=True,
            ckpt_path=args.init_ckpt if args.init_ckpt else None,
            model_type=args.model_type,
        )

    def forward(self, images, kl_weight):
        posterior = self.vae.encode(images)
        latents = posterior.sample()
        recon = self.vae.decode(latents)
        recon_loss = F.mse_loss(recon.float(), images.float(), reduction="mean")
        kl_loss = posterior.kl().float().mean()
        loss = recon_loss + kl_weight * kl_loss
        return loss, recon_loss.detach(), kl_loss.detach(), recon


def compute_losses(model, images, kl_weight):
    return model(images, kl_weight)


def save_reconstruction_grid(images, recon, sample_dir, epoch):
    os.makedirs(sample_dir, exist_ok=True)
    images = images[: recon.shape[0]].detach().float().clamp(-1, 1)
    recon = recon.detach().float().clamp(-1, 1)
    grid = torch.stack([images, recon], dim=1).flatten(0, 1)
    grid = grid * 0.5 + 0.5
    torchvision.utils.save_image(grid, os.path.join(sample_dir, f"recon_{epoch:05d}.png"), nrow=2)


def train_one_epoch(model, data_loader, optimizer, scaler, device, epoch, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    optimizer.zero_grad(set_to_none=True)
    last_recon = None
    last_images = None

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        epoch_progress = epoch + data_iter_step / len(data_loader)
        if data_iter_step % args.accum_iter == 0:
            lr = adjust_learning_rate(optimizer, epoch_progress, args)
        images = get_images(batch, device)

        is_last_step = data_iter_step + 1 == len(data_loader)
        sync_gradients = (data_iter_step + 1) % args.accum_iter == 0 or is_last_step
        ddp_no_sync = hasattr(model, "no_sync") and not sync_gradients
        sync_context = model.no_sync() if ddp_no_sync else torch.enable_grad()
        accum_steps = args.accum_iter if not is_last_step else ((data_iter_step % args.accum_iter) + 1)

        with sync_context:
            with get_autocast_context(device, args.amp_dtype):
                loss, recon_loss, kl_loss, recon = compute_losses(model, images, args.kl_weight)
            loss_value = loss.detach()
            scaled_loss = loss / accum_steps
            scaler.scale(scaled_loss).backward()

        if sync_gradients:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) if args.max_grad_norm > 0 else torch.tensor(0.0, device=device)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0, device=device)

        last_recon = recon.detach()
        last_images = images.detach()
        metric_logger.update(loss=loss_value)
        metric_logger.update(recon_loss=recon_loss)
        metric_logger.update(kl_loss=kl_loss)
        metric_logger.update(weighted_kl_loss=kl_loss * args.kl_weight)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(lr=lr)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, last_images, last_recon


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= args.num_val_batches:
            break
        images = get_images(batch, device)
        with get_autocast_context(device, args.amp_dtype):
            loss, recon_loss, kl_loss, recon = compute_losses(model, images, args.kl_weight)
        metric_logger.update(loss=loss.detach())
        metric_logger.update(recon_loss=recon_loss)
        metric_logger.update(kl_loss=kl_loss)
        metric_logger.update(weighted_kl_loss=kl_loss * args.kl_weight)

    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def load_checkpoint(args, model_without_ddp, optimizer, scaler):
    if not args.resume:
        return
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    args.start_epoch = checkpoint.get("epoch", -1) + 1
    print(f"Resume checkpoint {args.resume}")


def save_checkpoint(args, epoch, model_without_ddp, optimizer, scaler):
    checkpoint_path = Path(args.output_dir) / f"checkpoint-{epoch:05d}.pth"
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "args": args,
    }
    misc.save_on_master(to_save, checkpoint_path)


def main(args):
    args.output_dir = f"{args.log_dir}/checkpoint"
    misc.init_distributed_mode(args)

    if misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    train_transform = build_transform(args.input_size, train=True)
    val_transform = build_transform(args.input_size, train=False)
    dataset_train = build_dataset(args.data_path, train_transform)
    dataset_val = build_dataset(args.val_data_path, val_transform)

    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
        shuffle=True,
        drop_last=True,
    ) if args.distributed else None

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        shuffle=sampler_train is None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
    )

    model = VAVAEL2KLTrainer(args).to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    load_checkpoint(args, model_without_ddp, optimizer, scaler)

    print(f"Start VAVAE L2+KL training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)

        train_stats, last_images, last_recon = train_one_epoch(model, data_loader_train, optimizer, scaler, device, epoch, args)
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

        if args.eval_freq > 0 and ((epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs):
            val_stats = evaluate(model, data_loader_val, device, args)
            log_stats.update({f"val_{k}": v for k, v in val_stats.items()})

        if misc.is_main_process() and last_images is not None and last_recon is not None:
            save_reconstruction_grid(last_images[:args.num_sample_images], last_recon[:args.num_sample_images], args.sample_dir, epoch)

        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            save_checkpoint(args, epoch, model_without_ddp, optimizer, scaler)

        if misc.is_main_process():
            with open(os.path.join(args.log_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
