#!/bin/bash

# Define shared variables
MODEL="gpt2"
DATASET="Skylion007/openwebtext"
SPLIT="train"
CTX_LEN=64
BATCH_SIZE=256
GRAD_ACC_STEPS=1
MICRO_ACC_STEPS=4
LR_WARMUP_STEPS=500
LR=1e-4
AUXK_ALPHA=0.0
DEAD_FEATURE_THRESHOLD=100000
LOG_TO_WANDB=True
# MAX_EXAMPLES=10_000_000
# MAX_EXAMPLES=8192
MAX_EXAMPLES=-1
K=256
RUN_NAME="fvu-scaling-cross-test"
GPUS=8

export WANDB_PROJECT="sae-experiments"
export WANDB_ENTITY="michaelsklar"

# Cross-layer training
torchrun --nproc-per-node $GPUS -m sae \
    --multirun \
    model=$MODEL \
    dataset=$DATASET \
    enable_cross_layer_training=true \
    root_path=checkpoints/cross-layer-test \
    sae.scale_encoder_fvu=0.2 \
    sae.k=$K \
    'layers=[[4, 5, 6, 7, 8, 9]]' \
    split=$SPLIT \
    ctx_len=$CTX_LEN \
    max_examples=10000000 \
    batch_size=$BATCH_SIZE \
    grad_acc_steps=$GRAD_ACC_STEPS \
    micro_acc_steps=$MICRO_ACC_STEPS \
    lr_warmup_steps=$LR_WARMUP_STEPS \
    lr=$LR \
    auxk_alpha=$AUXK_ALPHA \
    dead_feature_threshold=$DEAD_FEATURE_THRESHOLD \
    run_name=$RUN_NAME \
    log_to_wandb=$LOG_TO_WANDB
