#!/bin/bash
set -e

# Configuration
CONTAINER_ID=${CONTAINER_ID:-$(hostname)}
RAY_HEAD_ADDRESS=${RAY_HEAD_ADDRESS:-0.0.0.0:6379}
RAY_JOIN_RETRIES=${RAY_JOIN_RETRIES:-10}
RAY_JOIN_RETRY_INTERVAL=${RAY_JOIN_RETRY_INTERVAL:-3}

# Check if already connected to Ray
if pgrep -f "ray::IDLE" >/dev/null || pgrep -f "raylet" >/dev/null; then
    echo "ℹ️ Ray worker already running in this container. Skipping registration."
    exit 0
fi

echo "🔍 Attempting to connect to Ray head node at ${RAY_HEAD_ADDRESS}"

# Detect number of GPUs available in the container
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "🔍 Detected ${NUM_GPUS} GPUs in container"
else
    NUM_GPUS=0
    echo "⚠️ nvidia-smi not found, assuming 0 GPUs"
fi

# Register container-specific resource with count equal to GPU count
RESOURCE_JSON="{\"${CONTAINER_ID}\": ${NUM_GPUS}}"

# Start Ray worker process
ray start --address=${RAY_HEAD_ADDRESS} \
    --resources="${RESOURCE_JSON}" \
    --num-gpus=${NUM_GPUS} \
    --node-ip-address=0.0.0.0

echo "🔗 Container ${CONTAINER_ID} registered with Ray cluster with ${NUM_GPUS} GPUs"
