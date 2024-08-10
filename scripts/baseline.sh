#!/bin/bash

# Define shared variables
MODEL="gpt2"
DATASET="togethercomputer/RedPajama-Data-1T-Sample"
SPLIT="train"
CTX_LEN=64
BATCH_SIZE=2048
GRAD_ACC_STEPS=4
MICRO_ACC_STEPS=4
LR_WARMUP_STEPS=10000
LR=1e-4
# AUXK_ALPHA=0.03125
AUXK_ALPHA=0.0
DEAD_FEATURE_THRESHOLD=100000
LOG_TO_WANDB=True
MAX_EXAMPLES=5000000
K=256

export WANDB_PROJECT="sae-experiments"
export WANDB_ENTITY="michaelsklar"

# Cross-layer training
python -m sae $MODEL $DATASET \
    --k $K \
    --enable_cross_layer_training True \
    --layers 8 9 10 11 \
    --split $SPLIT \
    --ctx_len $CTX_LEN \
    --max_examples $MAX_EXAMPLES \
    --batch_size $BATCH_SIZE \
    --grad_acc_steps $GRAD_ACC_STEPS \
    --micro_acc_steps $MICRO_ACC_STEPS \
    --lr_warmup_steps $LR_WARMUP_STEPS \
    --lr $LR \
    --auxk_alpha $AUXK_ALPHA \
    --dead_feature_threshold $DEAD_FEATURE_THRESHOLD \
    --run_name baseline_cross_layer \
    --log_to_wandb $LOG_TO_WANDB

Individual layer training
for LAYER in 8 9 10 11
do
    python -m sae $MODEL $DATASET \
        --k $K \
        --enable_cross_layer_training False \
        --layers $LAYER \
        --split $SPLIT \
        --ctx_len $CTX_LEN \
        --max_examples $MAX_EXAMPLES\
        --batch_size $BATCH_SIZE \
        --grad_acc_steps $GRAD_ACC_STEPS \
        --micro_acc_steps $MICRO_ACC_STEPS \
        --lr_warmup_steps $LR_WARMUP_STEPS \
        --lr $LR \
        --auxk_alpha $AUXK_ALPHA \
        --dead_feature_threshold $DEAD_FEATURE_THRESHOLD \
        --run_name baseline_layer_$LAYER \
        --log_to_wandb $LOG_TO_WANDB
done
