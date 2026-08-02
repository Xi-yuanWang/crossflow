#!/bin/bash
# Example training script for MeanFlow Autoencoder
# Adjust paths and hyperparameters as needed.

set -euo pipefail

# --- Paths (modify these) ---
DATA_PATH=${DATA_PATH:-./data/imagenet/train/}
VAL_DATA_PATH=${VAL_DATA_PATH:-./data/imagenet/validation/}
OUT_PATH=${OUT_PATH:-./work_dir}
LOG_DIR=${LOG_DIR:-./log_dir}

# --- FID evaluation weights (download if needed) ---
export INCEPTION_WEIGHTS=${INCEPTION_WEIGHTS:-./weights-inception-2015-12-05-6726825d.pth}
export IN256_FID_STATS=${IN256_FID_STATS:-./VIRTUAL_imagenet256_labeled.npz}

# --- Distributed training settings ---
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12349}

# --- Model config ---
VAE_TYPE=${VAE_TYPE:-vavae}
CODE_DIM=${CODE_DIM:-32}
ENC_DEPTH=${ENC_DEPTH:-4}
DEC_DEPTH=${DEC_DEPTH:-8}
HIDDIM=${HIDDIM:-768}

# --- Training hyperparameters ---
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-800}
LR=${LR:-8e-4}
GAN_LR=${GAN_LR:-8e-4}
GAN_WEIGHT=${GAN_WEIGHT:-1}
GAN_START_EPOCH=${GAN_START_EPOCH:-2}
SEED=${SEED:-0}

torchrun \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main_crossflow.py \
  --data_path "$DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --out_path "$OUT_PATH" \
  --log_dir "$LOG_DIR" \
  --eval_epochs_interval -1 \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --warmup_epochs 5 \
  --lr $LR \
  --ganlr $GAN_LR \
  --min_lr $LR \
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
  --ganweight $GAN_WEIGHT \
  --gan_start_epoch $GAN_START_EPOCH \
  --seed $SEED \
  --vae_type $VAE_TYPE
