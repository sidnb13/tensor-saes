#!/bin/bash

# Define shared variables
MODEL="EleutherAI/pythia-14m"
DATASET="togethercomputer/RedPajama-Data-1T-Sample"
SPLIT="train"
CTX_LEN=64
BATCH_SIZE=2048
GRAD_ACC_STEPS=1
MICRO_ACC_STEPS=4
LR_WARMUP_STEPS=20
LR=1e-3
AUXK_ALPHA=0.0
DEAD_FEATURE_THRESHOLD=100000
LOG_TO_WANDB=True
MAX_EXAMPLES=-1
K=128
SCALE_FVU=0.2
RUN_NAME="pythia14m-all-layers-rp1t-sample"
EXPANSION_FACTOR=4
WANDB_GROUP="pythia14m-sweeps-auxk-expansion"
SAVE_EVERY=50

SEEDS=42

export WANDB_PROJECT="sae-experiments"
export WANDB_ENTITY="michaelsklar"

# Distributed stuff
NPROC_PER_NODE=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
COMMA_COUNT=$(echo $CUDA_VISIBLE_DEVICES | grep -o ',' | tr -d " \n" | wc -c)

if [ $COMMA_COUNT -gt 0 ]; then
    NPROC_PER_NODE=$(($COMMA_COUNT + 1))
elif [[ ! -z $CUDA_VISIBLE_DEVICES ]]; then
    NPROC_PER_NODE=1
fi

# Cross-layer training
CMD="python -m sae.train \
    --multirun \
    model=$MODEL \
    seed=$SEEDS \
    dataset=$DATASET \
    enable_cross_layer_training=true \
    root_path=checkpoints/pythia14m-all-layers-rp1t \
    sae.scale_encoder_fvu=$SCALE_FVU \
    sae.k=$K \
    sae.expansion_factor=$EXPANSION_FACTOR \
    'layers=[[0, 1, 2, 3, 4, 5]]' \
    split=$SPLIT \
    ctx_len=$CTX_LEN \
    max_examples=$MAX_EXAMPLES \
    batch_size=$BATCH_SIZE \
    grad_acc_steps=$GRAD_ACC_STEPS \
    micro_acc_steps=$MICRO_ACC_STEPS \
    lr_warmup_steps=$LR_WARMUP_STEPS \
    save_every=$SAVE_EVERY \
    lr=$LR \
    auxk_alpha=$AUXK_ALPHA \
    dead_feature_threshold=$DEAD_FEATURE_THRESHOLD \
    run_name=$RUN_NAME \
    log_to_wandb=$LOG_TO_WANDB \
    wandb_group=$WANDB_GROUP \
    tp=false \
    ddp=true \
    optimizer=adam_zero"

eval $CMD
