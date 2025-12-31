#!/bin/bash
# ARC-AGI-3 training script for ls20 game
# Based on working ThetaEvolve training pattern
#
# Prerequisites:
# 1. Copy .env.example to .env and set your ARC_API_KEY from https://three.arcprize.org
# 2. Set WANDB credentials
#
# Usage:
#   export SAVE_PATH=/gpfs/scrubbed/rulins/save
#   bash scripts/run-arc-agi3-ls20.sh

########################### CONFIGURATION SECTION #############################

# Important: replace SAVE_PATH with your path with enough space
export SAVE_PATH=${SAVE_PATH:-/gpfs/scrubbed/rulins/save}

# Model configuration
MODEL_FAMILY="nvidia"
MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"

# Game configuration
# Game ID prefix (will be resolved to full ID automatically, e.g., "ls20" -> "ls20-fa137e247ce6")
GAME_ID="${GAME_ID:-ls20}"

# ARC-AGI-3 specific settings
NUM_PARALLEL_EPISODES=4
MAX_ACTIONS=80
HISTORY_WINDOW=0  # No history frames - only current frame (saves memory)

# Random seed
SEED=3407

# Run name (use short game name for readability)
SHORT_GAME_ID="${GAME_ID%%-*}"  # Extract "ls20" from "ls20-fa137e247ce6"
RUN_NAME="arc_agi3_${SHORT_GAME_ID}_${MODEL_NAME}_seed${SEED}"

# Replace with your own wandb settings
WANDB_API_KEY="${WANDB_API_KEY:-412bcc10b2150ae3dd49eb6963df7390e605efd0}"
WANDB_ENTITY="${WANDB_ENTITY:-Srl0310}"
WANDB_PROJECT="${WANDB_PROJECT:-arc-agi3-training}"

########################### PATH SETUP #############################

# Path configuration
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tmp
mkdir -p $SAVE_PATH/hf
mkdir -p $SAVE_PATH/wandb
mkdir -p $SAVE_PATH/shm
mkdir -p $SAVE_PATH/triton

# Setup paths
export TMPDIR=/tmp
export HF_HOME=$SAVE_PATH/hf
export HUGGINGFACE_HUB_CACHE=$SAVE_PATH/hf/hub
export TRANSFORMERS_CACHE=$SAVE_PATH/hf/hub
export HF_DATASETS_CACHE=$SAVE_PATH/hf/datasets
export SAVE_SHM_DIR=$SAVE_PATH/shm
export TRITON_CACHE_DIR=$SAVE_PATH/triton

# wandb
export WANDB_CACHE_DIR=$SAVE_PATH/wandb
export WANDB_DIR=$SAVE_PATH/wandb
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_ENTITY=$WANDB_ENTITY
export WANDB_PROJECT=$WANDB_PROJECT

# Checkpoint directory
CKPT_DIR="${SAVE_PATH}/${RUN_NAME}"
mkdir -p "${CKPT_DIR}"

########################### LOAD .env FILE #############################

# Load .env file if it exists (for ARC_API_KEY)
if [ -f ".env" ]; then
    echo "Loading environment from .env file..."
    set -a
    source .env
    set +a
elif [ -f ".env.example" ]; then
    echo "Warning: .env not found. Copy .env.example to .env and set your API keys."
fi

# Check for API key
if [ -z "$ARC_API_KEY" ]; then
    echo "Error: ARC_API_KEY environment variable is not set."
    echo "Get your API key from https://three.arcprize.org and run:"
    echo "  export ARC_API_KEY='your_api_key_here'"
    exit 1
fi

########################### MODEL SETUP #############################

FORCE_DOWNLOAD=0
# Check if model already exists
if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ] && [ -f "$SAVE_SHM_DIR/$MODEL_NAME/config.json" ] && [ $FORCE_DOWNLOAD -eq 0 ]; then
    echo "Model $MODEL_NAME already exists at $SAVE_SHM_DIR/$MODEL_NAME, skipping download"
else
    if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ]; then
        echo "Incomplete model directory found, deleting and re-downloading"
        rm -rf "$SAVE_SHM_DIR/$MODEL_NAME"
    fi
    mkdir -p $SAVE_SHM_DIR
    echo "Downloading model $MODEL_NAME..."
    huggingface-cli download $MODEL_FAMILY/$MODEL_NAME --local-dir $SAVE_SHM_DIR/$MODEL_NAME
    echo "Model download completed"
fi

source scripts/models/deepseek-r1-distill-qwen-1.5B.sh

if [ ! -d "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist" ] || [ $FORCE_DOWNLOAD -eq 1 ]; then
    echo "Converting HF model to torch dist format..."
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} --hf-checkpoint $SAVE_SHM_DIR/$MODEL_NAME --save $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist
    echo "Conversion completed"
else
    echo "Torch dist model already exists at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist"
fi

########################### CLEANUP #############################

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3

set -ex

export PYTHONBUFFERED=16
export TOKENIZERS_PARALLELISM=false

# Check for NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
echo "NVLINK_COUNT: $NVLINK_COUNT"
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

########################### ARGUMENTS #############################

CKPT_ARGS=(
   --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}"
   --ref-load "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
   --load "${CKPT_DIR}/"
   --save "${CKPT_DIR}/"
   --save-interval 10
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --arc-agi3-gym
  --arc-agi3-game-id $GAME_ID
  --arc-agi3-num-parallel-episodes $NUM_PARALLEL_EPISODES
  --arc-agi3-max-actions $MAX_ACTIONS
  --arc-agi3-history-window $HISTORY_WINDOW

  --apply-chat-template

  --rm-type arc-agi3
  --reward-key reward

  --num-rollout 1000
  --rollout-batch-size 4      # Reduced from 8
  --n-samples-per-prompt 2
  --rollout-max-response-len 256
  --rollout-temperature 0.7

  --over-sampling-batch-size 4  # Reduced from 8
  --partial-rollout

  --num-steps-per-rollout 1
  --wandb-always-use-train-step
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 28  # All 28 layers (saves activation memory)

  --use-dynamic-batch-size
  --max-tokens-per-gpu 512  # Reduced from 2048 to save memory
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-tis
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

WANDB_ARGS=(
  --use-wandb
  --wandb-team ${WANDB_ENTITY}
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${RUN_NAME}"
  --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.7  # Reduced from 0.8 to leave more room
  --sglang-server-concurrency 64    # Reduced from 256
)

MISC_ARGS=(
  --seed ${SEED}
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

########################### RUN TRAINING #############################

echo "=== ARC-AGI-3 Training Configuration ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "Game: ${GAME_ID}"
echo "Model: ${MODEL_NAME}"
echo "Checkpoint Dir: ${CKPT_DIR}"
echo "GPUs: 2 total (1 actor + 1 rollout)"
echo "========================================"

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Disable Triton
export TRITON_DISABLE=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export FAST_MOUNT=$SAVE_PATH/fast_mount
export HF_DATASETS_CACHE=$FAST_MOUNT/hf/datasets
export DATASETS_CACHE=$HF_DATASETS_CACHE
export DATASETS_TMPDIR=$FAST_MOUNT/tmp
export PYARROW_TMP_DIR=$FAST_MOUNT/tmp

mkdir -p "$HF_DATASETS_CACHE" "$DATASETS_TMPDIR" || true

# Build the runtime environment JSON
RUNTIME_ENV_JSON="$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "HF_HOME": "${HF_HOME}",
    "HUGGINGFACE_HUB_CACHE": "${HUGGINGFACE_HUB_CACHE}",
    "TRANSFORMERS_CACHE": "${TRANSFORMERS_CACHE}",
    "HF_DATASETS_CACHE": "${HF_DATASETS_CACHE}",
    "DATASETS_CACHE": "${DATASETS_CACHE}",
    "DATASETS_TMPDIR": "${DATASETS_TMPDIR}",
    "PYARROW_TMP_DIR": "${PYARROW_TMP_DIR}",
    "TMPDIR": "${TMPDIR}",
    "WANDB_CACHE_DIR": "${WANDB_CACHE_DIR}",
    "WANDB_DIR": "${WANDB_DIR}",
    "WANDB_GROUP": "${RUN_NAME}",
    "TRITON_DISABLE": "1",
    "ARC_API_KEY": "${ARC_API_KEY}",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}
JSON
)"

echo "Submitting Ray job..."

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]}
