#!/bin/bash
# Evaluate a trained model (generation + reconstruction FID).
# Usage: bash scripts/eval.sh --resume <checkpoint_path>

set -euo pipefail

DATA_PATH=${DATA_PATH:-./data/imagenet/train/}
VAL_DATA_PATH=${VAL_DATA_PATH:-./data/imagenet/validation/}
OUT_PATH=${OUT_PATH:-./work_dir_eval}
LOG_DIR=${LOG_DIR:-./log_dir_eval}

export INCEPTION_WEIGHTS=${INCEPTION_WEIGHTS:-./weights-inception-2015-12-05-6726825d.pth}
export IN256_FID_STATS=${IN256_FID_STATS:-./VIRTUAL_imagenet256_labeled.npz}

# Model config (should match training)
VAE_TYPE=${VAE_TYPE:-vavae}
CODE_DIM=${CODE_DIM:-32}
ENC_DEPTH=${ENC_DEPTH:-4}
DEC_DEPTH=${DEC_DEPTH:-8}
HIDDIM=${HIDDIM:-768}
INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE:-128}

torchrun --standalone --nproc_per_node=1 \
  main_crossflow.py \
  --data_path "$DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --out_path "$OUT_PATH" \
  --log_dir "$LOG_DIR" \
  --batch_size 64 \
  --epochs 800 \
  --warmup_epochs 5 \
  --lr 8e-4 \
  --ganlr 8e-4 \
  --min_lr 8e-4 \
  --weight_decay 1e-6 \
  --code_dim $CODE_DIM \
  --enc_depth $ENC_DEPTH \
  --dec_depth $DEC_DEPTH \
  --hiddim $HIDDIM \
  --dis_depth 0 \
  --lowmomentum \
  --percloss \
  --percl1 \
  --percpure \
  --ganweight 1 \
  --gan_start_epoch 2 \
  --seed 0 \
  --gen_only \
  --inference_batch_size $INFERENCE_BATCH_SIZE \
  --vae_type $VAE_TYPE \
  "$@"
