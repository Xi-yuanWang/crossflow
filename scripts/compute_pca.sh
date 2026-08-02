#!/bin/bash
# Compute PCA for a given VAE type.
# Usage: VAE_TYPE=vavae bash scripts/compute_pca.sh

set -euo pipefail

VAE_TYPE=${VAE_TYPE:-identity}
DATA_PATH=${DATA_PATH:-./data/imagenet/train/}
VAL_DATA_PATH=${VAL_DATA_PATH:-./data/imagenet/validation/}
OUT_PATH=${OUT_PATH:-./work_dir_pca_${VAE_TYPE}}
LOG_DIR=${LOG_DIR:-./log_dir_pca_${VAE_TYPE}}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}
PATCH_SIZE=${PATCH_SIZE:-16}

export INCEPTION_WEIGHTS=${INCEPTION_WEIGHTS:-./weights-inception-2015-12-05-6726825d.pth}
export IN256_FID_STATS=${IN256_FID_STATS:-./VIRTUAL_imagenet256_labeled.npz}

mkdir -p "$OUT_PATH" "$LOG_DIR"

torchrun --standalone --nproc_per_node=1 \
  main_crossflow.py \
  --data_path "$DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --out_path "$OUT_PATH" \
  --log_dir "$LOG_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --patch_size "$PATCH_SIZE" \
  --vae_type "$VAE_TYPE" \
  --computepca
