#!/bin/bash

# Define shared variables
MODEL="gpt2"
DATASET="togethercomputer/RedPajama-Data-1T-Sample"
SPLIT="train"
CTX_LEN=64
BATCH_SIZE=4096
GRAD_ACC_STEPS=4
MICRO_ACC_STEPS=4
LR_WARMUP_STEPS=500
LR=1e-4
AUXK_ALPHA=0.0
DEAD_FEATURE_THRESHOLD=100000
LOG_TO_WANDB=False
MAX_EXAMPLES=10000000
# MAX_EXAMPLES=8192
K=256
RUN_NAME="test-fvu-scaling"

export WANDB_PROJECT="sae-experiments"
export WANDB_ENTITY="michaelsklar"
export WANDB_RUN_GROUP="fvu-scale-microsweep"

python -m sae.train \
    --multirun \
    model=$MODEL \
    dataset=$DATASET \
    sae.k=$K \
    sae.scale_encoder_fvu=0.2 \
    layers=[8] \
    split=$SPLIT \
    ctx_len=$CTX_LEN \
    max_examples=$MAX_EXAMPLES \
    batch_size=$BATCH_SIZE \
    grad_acc_steps=$GRAD_ACC_STEPS \
    micro_acc_steps=$MICRO_ACC_STEPS \
    lr_warmup_steps=$LR_WARMUP_STEPS \
    lr=$LR \
    auxk_alpha=$AUXK_ALPHA \
    dead_feature_threshold=$DEAD_FEATURE_THRESHOLD \
    run_name=encoder_scale_fvu_test \
    log_to_wandb=$LOG_TO_WANDB \
    optimizer=adam_zero \
    stdout_log_frequency=1 \
    sae.signed=false
