# MeanFlow Autoencoder

A flow-matching based autoencoder for image generation and reconstruction, trained with perceptual loss and GAN objectives.

## Overview

This codebase implements a cross-flow autoencoder that learns a latent space via flow matching. The encoder maps images to latent tokens, and the decoder reconstructs images from latent tokens using a learned flow. Training combines:

- **Perceptual loss** (DINOv3-based feature matching)
- **GAN loss** (DINOv3-based discriminator)
- **Flow matching** objective for the decoder

## Requirements

```bash
pip install torch torchvision timm einops datasets transformers accelerate diffusers rich matplotlib torchdiffeq torch_fidelity
```

## Data Preparation

The code expects ImageNet data in HuggingFace `datasets` format (loaded via `load_from_disk`). Prepare your data accordingly:

```python
from datasets import load_dataset
ds = load_dataset("imagenet-1k")
ds.save_to_disk("./data/imagenet/train/")
```

## Model Weights

Place the following in the project root:
- **VAE checkpoint**: `vavae-imagenet256-f16d32-dinov2.pt` (for `--vae_type vavae`)
- **DINOv3 model**: `./dinov3/` directory (HuggingFace format, used for perceptual loss and discriminator)
- **FID evaluation**: `weights-inception-2015-12-05-6726825d.pth` and `VIRTUAL_imagenet256_labeled.npz`

## Training

### Single-node training (8 GPUs)

```bash
bash scripts/train.sh
```

### Multi-node training

```bash
NNODES=4 NODE_RANK=0 MASTER_ADDR=<master_ip> bash scripts/train.sh
```

### Configuration

All hyperparameters can be set via environment variables (see `scripts/train.sh`) or passed directly to `main_crossflow.py`. Key options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VAE_TYPE` | `vavae` | Encoder type: `vavae`, `identity`, `sdvae`, `dinov3`, `vavae_l2kl` |
| `CODE_DIM` | `32` | Latent code dimension |
| `DEC_DEPTH` | `8` | Decoder transformer depth |
| `HIDDIM` | `768` | Hidden dimension (B=768, L=1024, XL=1152) |
| `GAN_WEIGHT` | `1` | GAN loss weight |
| `BATCH_SIZE` | `64` | Per-GPU batch size |

### Model sizes

- **Base (B)**: `--dec_depth 8 --hiddim 768`
- **Large (L)**: `--dec_depth 20 --hiddim 1024`
- **XL**: `--dec_depth 24 --hiddim 1152`

## Evaluation

```bash
# Evaluate a checkpoint (FID, Precision, Recall, PSNR, SSIM, LPIPS)
bash scripts/eval.sh --resume <path_to_checkpoint.pth>
```

## PCA Computation

Before training with certain VAE types, compute PCA statistics:

```bash
VAE_TYPE=vavae bash scripts/compute_pca.sh
```

## Project Structure

```
├── main_crossflow.py          # Main training entry point
├── main_vavae_l2kl.py         # VAE training with L2+KL loss
├── crossflow_withgan.py       # ViT model definition (encoder + decoder + discriminator)
├── engine_crossflow.py        # Training/evaluation loop
├── Vavae.py                   # VAE architecture (AutoencoderKL)
├── RAE_dinodisc.py            # DINOv3-based discriminator
├── transformer_utils.py       # Transformer building blocks
├── muon.py                    # Muon+AdamW hybrid optimizer
├── adm_evaluator_pytorch.py   # FID/IS evaluation
├── diffaug.py                 # Differentiable augmentation
├── lerp.py                    # Linear interpolation utilities
├── flux2vae.py                # Flux VAE wrapper
├── sdvae.py                   # Stable Diffusion VAE wrapper
├── perceptual/                # Perceptual loss modules
│   ├── dinoperc.py            # DINOv3 perceptual loss
│   ├── lpips.py               # LPIPS implementation
│   └── perceptual_loss.py     # Combined perceptual loss
├── util/
│   ├── misc.py                # Distributed training utilities
│   └── lr_sched.py            # Learning rate scheduler
└── scripts/
    ├── train.sh               # Training launch script
    ├── eval.sh                # Evaluation script
    └── compute_pca.sh         # PCA computation script
```

## License

This source code is licensed under the license found in the LICENSE file in the root directory of this source tree. Parts of this code are based on [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit).
