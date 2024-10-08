#!/bin/bash

# Define shared variables
MODEL="gpt2"
DATASET="togethercomputer/RedPajama-Data-1T-Sample"
SPLIT="train"
CTX_LEN=64
BATCH_SIZE=128
GRAD_ACC_STEPS=1
MICRO_ACC_STEPS=4
LR_WARMUP_STEPS=100
LR=1e-3
AUXK_ALPHA=0.0
DEAD_FEATURE_THRESHOLD=100000
LOG_TO_WANDB=False
# MAX_EXAMPLES=10_000_000
# MAX_EXAMPLES=8192
MAX_EXAMPLES=-1
K=256
RUN_NAME="fvu-scaling-cross-test"
EXPANSION_FACTOR=8

export WANDB_PROJECT="sae-experiments"
export WANDB_ENTITY="michaelsklar"

# Distributed stuff
NPROC_PER_NODE=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
COMMA_COUNT=$(echo $CUDA_VISIBLE_DEVICES | grep -o ',' | tr -d " \n" | wc -c)
LAUNCHER="torchrun --nnodes 1 --nproc_per_node $NPROC_PER_NODE"

if [ $COMMA_COUNT -gt 0 ]; then
    NPROC_PER_NODE=$(($COMMA_COUNT + 1))
    LAUNCHER="torchrun --nnodes 1 --nproc_per_node $NPROC_PER_NODE"
elif [[ ! -z $CUDA_VISIBLE_DEVICES ]]; then
    NPROC_PER_NODE=1
    LAUNCHER="python"
fi

# Cross-layer training
CMD="-m sae.train \
    --multirun \
    model=$MODEL \
    dataset=$DATASET \
    enable_cross_layer_training=true \
    root_path=checkpoints/cross-layer-test \
    sae.scale_encoder_fvu=0.2 \
    sae.k=$K \
    sae.expansion_factor=$EXPANSION_FACTOR \
    'layers=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]' \
    split=$SPLIT \
    ctx_len=$CTX_LEN \
    max_examples=10000000 \
    batch_size=$BATCH_SIZE \
    grad_acc_steps=$GRAD_ACC_STEPS \
    micro_acc_steps=$MICRO_ACC_STEPS \
    lr_warmup_steps=$LR_WARMUP_STEPS \
    lr=$LR \
    auxk_alpha=$AUXK_ALPHA \
    save_every=10 \
    dead_feature_threshold=$DEAD_FEATURE_THRESHOLD \
    run_name=$RUN_NAME \
    log_to_wandb=$LOG_TO_WANDB \
    tp=true \
    ddp=false \
    optimizer=adam_zero"

eval $LAUNCHER $CMD
