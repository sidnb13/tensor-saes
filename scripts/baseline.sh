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
LOG_TO_WANDB=True
MAX_EXAMPLES=10000000
K=256
RUN_NAME="test-fvu-scaling"

export WANDB_PROJECT="sae-experiments"
export WANDB_ENTITY="michaelsklar"
export WANDB_RUN_GROUP="gpt2-init-sweeps-layer8"

# # Cross-layer training
# python -m sae \
#     model=$MODEL \
#     dataset=$DATASET \
#     enable_cross_layer_training=true \
#     sae.k=$K \
#     sae.scale_encoder_fvu=0.1 \
#     layers=[8, 9, 10, 11] \
#     split=$SPLIT \
#     ctx_len=$CTX_LEN \
#     max_examples=$MAX_EXAMPLES \
#     batch_size=$BATCH_SIZE \
#     grad_acc_steps=$GRAD_ACC_STEPS \
#     micro_acc_steps=$MICRO_ACC_STEPS \
#     lr_warmup_steps=$LR_WARMUP_STEPS \
#     lr=$LR \
#     auxk_alpha=$AUXK_ALPHA \
#     dead_feature_threshold=$DEAD_FEATURE_THRESHOLD \
#     run_name=$RUN_NAME \
#     log_to_wandb=$LOG_TO_WANDB

python -m sae \
    --multirun \
    model=$MODEL \
    dataset=$DATASET \
    sae.k=$K \
    sae.scale_encoder_k=true,false \
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
    run_name=encoder_scale_k_test \
    log_to_wandb=$LOG_TO_WANDB

python -m sae \
    --multirun \
    model=$MODEL \
    dataset=$DATASET \
    sae.k=$K \
    sae.scale_encoder_fvu=null,0.1,0.3,0.5,0.9 \
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
    log_to_wandb=$LOG_TO_WANDB
