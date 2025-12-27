#!/bin/bash
#SBATCH --job-name=theta_evolve_multi
#SBATCH --nodes=2                   # Number of nodes (adjust as needed)
#SBATCH --ntasks-per-node=1         
#SBATCH --hint=nomultithread   
#SBATCH --account comem
#SBATCH --qos h200_comem_high
#SBATCH --mem 400G
#SBATCH --gres=gpu:8                # 8 GPUs per node
#SBATCH --time 120:00:00      
#SBATCH --requeue
#SBATCH --chdir=/checkpoint/comem/rulin/rl_evolve
#SBATCH --output=/checkpoint/comem/rulin/cache/slurm/theta_evolve_multi-%j.out

set -ex

########################## ENVIRONMENT ACTIVATION #############################
# Activate the slime-evolve environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate slime-evolve

# Fix cuDNN library path - use system cuDNN instead of conda's
# This is needed because SLURM jobs don't inherit interactive shell paths
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
unset LD_PRELOAD

echo "Activated environment: $CONDA_DEFAULT_ENV"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
which python
python --version

########################### CONFIGURATION SECTION #############################

#### Important: replace SAVE_PATH with your path with enough space ####
export SAVE_PATH=/checkpoint/comem/rulin/cache/rl_evolve

#### Local environment paths (non-Docker) ####
export BASE_DIR="/checkpoint/comem/rulin/slime_env"
export SLIME_DIR="/checkpoint/comem/rulin/rl_evolve"
export MEGATRON_DIR="$BASE_DIR/Megatron-LM"

#### Model selection - Using the largest model from the paper (8B) ####
SMALL_MODEL_NAME="dpsk_distill_qwen3_8b"

#### Task configuration ####
TASK="circle_packing_modular"

#### CONFIG_POSTFIX options ####
CONFIG_POSTFIX="it_XL"

#### Training mode: True for training, False for inference-only ####
IS_TRAINING=True

#### Training parameters ####
REWARD_PROCESS_TYPE="original_reward"

#### Lazy output penalty ####
LAZY_OUTPUT_PENALTY=1

#### Random seed ####
SEED=3407

#### Different initial program ####
INITIAL_PROGRAM_POSTFIX=""

#### Additional note for file names ####
NOTE="_multinode${SLURM_NNODES}n"

#### Replace with your own wandb settings ####
WANDB_API_KEY=local-8a8a005b87a483480b22f6f6b990bd15a3cea399
WANDB_ENTITY=Rulin
WANDB_PROJECT=theta-evolve

#### Database size (set to paper default or larger) ####
DB_SIZE=10000

########################## END CONFIGURATION SECTION #############################

# Required for tensor parallelism
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

########################## MULTI-NODE SETUP #############################

# Get node information
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6379
export MASTER_ADDR
export MASTER_PORT

echo "=== Multi-Node Configuration ==="
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Current node: $(hostname)"
echo "================================"

# Calculate total GPUs
GPUS_PER_NODE=8
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))
echo "Total GPUs: $TOTAL_GPUS"

########################## PATH SETUP #############################

POSTFIX_STR="_seed${SEED}${INITIAL_PROGRAM_POSTFIX}${NOTE}"

if [ "$SMALL_MODEL_NAME" = "dpsk_prorl_v2_1.5b" ]; then
    MODEL_FAMILY="nvidia"
    MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"
    models_file_name="deepseek-r1-distill-qwen-1.5B.sh"
elif [ "$SMALL_MODEL_NAME" = "dpsk_distill_qwen3_8b" ]; then
    MODEL_FAMILY="deepseek-ai"
    MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
    models_file_name="qwen3-8B.sh"
else
    echo "Unknown SMALL_MODEL_NAME: $SMALL_MODEL_NAME"
    exit 1
fi
echo "Using model: $MODEL_NAME"

# Path configuration
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tmp
mkdir -p $SAVE_PATH/hf
mkdir -p $SAVE_PATH/wandb
mkdir -p $SAVE_PATH/shm
mkdir -p $SAVE_PATH/triton

# setup paths
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

########################## AUTO-GENERATED PATHS #############################

INITIAL_PROGRAM="openevolve_adapted/examples/${TASK}/initial_programs/initial_program${INITIAL_PROGRAM_POSTFIX}.py"
EVALUATOR_FILE="openevolve_adapted/examples/${TASK}/evaluators/evaluator_modular.py"
CONFIG_YAML="openevolve_adapted/examples/${TASK}/configs/config_${TASK}_${CONFIG_POSTFIX}.yaml"

# Reward suffix mapping
case "$REWARD_PROCESS_TYPE" in
    "original_reward") REWARD_SUFFIX="" ;;
    "rl_normalized_reward") REWARD_SUFFIX="_rlnorm" ;;
    *) REWARD_SUFFIX="_${REWARD_PROCESS_TYPE}" ;;
esac

# Generate RUN_NAME
RUN_NAME="${SMALL_MODEL_NAME}_tr${IS_TRAINING}_l${LAZY_OUTPUT_PENALTY}_${TASK}_${CONFIG_POSTFIX}${REWARD_SUFFIX}${POSTFIX_STR}"

########################## MODEL SETUP (run on all nodes) #############################

FORCE_DOWNLOAD=0
if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ] && [ -f "$SAVE_SHM_DIR/$MODEL_NAME/config.json" ] && [ $FORCE_DOWNLOAD -eq 0 ]; then
    echo "Model $MODEL_NAME already exists at $SAVE_SHM_DIR/$MODEL_NAME, skipping download"
else
    if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ]; then
        echo "Incomplete model directory found at $SAVE_SHM_DIR/$MODEL_NAME, deleting and re-downloading"
        rm -rf "$SAVE_SHM_DIR/$MODEL_NAME"
    fi
    
    mkdir -p $SAVE_SHM_DIR

    echo "Downloading model $MODEL_NAME directly to $SAVE_SHM_DIR/$MODEL_NAME..."
    huggingface-cli download $MODEL_FAMILY/$MODEL_NAME --local-dir $SAVE_SHM_DIR/$MODEL_NAME

    echo "Model download completed"
fi

source scripts/models/${models_file_name}

# Convert model (only needs to run once, shared storage handles it)
if [ ! -d "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist" ] || [ $FORCE_DOWNLOAD -eq 1 ]; then
    echo "Converting HF model to torch dist format..."
    PYTHONPATH=$SLIME_DIR:$MEGATRON_DIR python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} --hf-checkpoint $SAVE_SHM_DIR/$MODEL_NAME --save $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist
    echo "Conversion completed, torch dist model saved at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist"
else
    echo "Torch dist model already exists at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist, skipping conversion"
fi

########################## RAY CLUSTER SETUP #############################

# Kill any existing Ray processes
pkill -9 sglang || true
pkill -9 ray || true
sleep 3

export PYTHONBUFFERED=16
export TOKENIZERS_PARALLELISM=false

# Check for NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK"

# Start Ray cluster
if [ "$(hostname)" = "$MASTER_ADDR" ]; then
    echo "Starting Ray head on $MASTER_ADDR..."
    ray start --head --node-ip-address $MASTER_ADDR --port $MASTER_PORT --num-gpus $GPUS_PER_NODE --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
    echo "Starting Ray worker on $(hostname), connecting to $MASTER_ADDR:$MASTER_PORT..."
    sleep 10  # Wait for head to start
    ray start --address=$MASTER_ADDR:$MASTER_PORT --num-gpus $GPUS_PER_NODE
fi

# Wait for all nodes to join
sleep 15
echo "Ray cluster status:"
ray status

########################## MAIN EXECUTION (only on head node) #############################

if [ "$(hostname)" = "$MASTER_ADDR" ]; then
    echo "=== Experiment Configuration ==="
    echo "NODES: ${SLURM_NNODES}"
    echo "TOTAL_GPUS: ${TOTAL_GPUS}"
    echo "RUN_NAME: ${RUN_NAME}"
    echo "TASK: ${TASK}"
    echo "MODEL: ${MODEL_NAME}"
    echo "INITIAL_PROGRAM: ${INITIAL_PROGRAM}"
    echo "EVALUATOR_FILE: ${EVALUATOR_FILE}"
    echo "CONFIG_YAML: ${CONFIG_YAML}"
    echo "SAVE_PATH: ${SAVE_PATH}"
    echo "================================"

    mkdir -p "${SAVE_PATH}/${RUN_NAME}"

    # Disable Triton
    export TRITON_DISABLE=1

    export FAST_MOUNT=$SAVE_PATH/fast_mount
    export DATASETS_CACHE=$FAST_MOUNT/hf/datasets
    export DATASETS_TMPDIR=$FAST_MOUNT/tmp
    export PYARROW_TMP_DIR=$FAST_MOUNT/tmp
    mkdir -p "$DATASETS_CACHE" "$DATASETS_TMPDIR"

    SAVE_SHM_DIR="${SAVE_PATH}/shm"
    CKPT_DIR="${SAVE_PATH}/${RUN_NAME}"
    RECORD_PATH="${SAVE_PATH}/${RUN_NAME}/records"

    # Determine debug-rollout-only mode
    if [ "$IS_TRAINING" = "False" ] || [ "$IS_TRAINING" = "false" ]; then
        DEBUG_ROLLOUT_ONLY="--debug-rollout-only"
    else
        DEBUG_ROLLOUT_ONLY=""
    fi

    # Build runtime environment JSON
    RUNTIME_ENV_JSON="$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_DIR}",
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
    "TRITON_DISABLE": "1"
  }
}
JSON
)"

    # Submit the job to Ray cluster
    # For multi-node: increase actor-num-nodes and rollout-num-gpus
    ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train.py \
      --actor-num-nodes ${SLURM_NNODES} \
      --actor-num-gpus-per-node 8 \
      --rollout-num-gpus ${TOTAL_GPUS} \
      ${MODEL_ARGS[@]} \
      --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}" \
      --ref-load "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist" \
      --load "${CKPT_DIR}/" \
      --save "${CKPT_DIR}/" \
      --save-interval 5 \
      --disable-rollout-global-dataset \
      --evolving-gym \
      --evolving-gym-initial-program "${INITIAL_PROGRAM}" \
      --evolving-gym-evaluator-file "${EVALUATOR_FILE}" \
      --evolving-gym-config-path "${CONFIG_YAML}" \
      --evolving-gym-max-concurrent-evals 16 \
      --evolving-gym-log-prompts \
      --evolving-gym-record \
      --evolving-gym-record-dir "${RECORD_PATH}" \
      --evolving-gym-lazy-output-penalty-level "${LAZY_OUTPUT_PENALTY}" \
      --evolving-gym-seed ${SEED} \
      --evolving-gym-reward-process-type "${REWARD_PROCESS_TYPE}" \
      --apply-chat-template \
      --rm-type evolving-gym \
      --reward-key reward \
      --num-rollout 1000000 \
      --rollout-batch-size 32 \
      --n-samples-per-prompt 16 \
      --rollout-max-response-len 16384 \
      --rollout-temperature 1.0 \
      --over-sampling-batch-size 32 \
      --partial-rollout \
      --num-steps-per-rollout 1 \
      --wandb-always-use-train-step \
      --balance-data \
      --tensor-model-parallel-size 4 \
      --sequence-parallel \
      --pipeline-model-parallel-size 1 \
      --context-parallel-size 2 \
      --expert-model-parallel-size 1 \
      --expert-tensor-parallel-size 1 \
      --recompute-granularity full \
      --recompute-method uniform \
      --recompute-num-layers 1 \
      --use-dynamic-batch-size \
      --max-tokens-per-gpu 2048 \
      --advantage-estimator grpo \
      --entropy-coef 0.00 \
      --eps-clip 0.2 \
      --eps-clip-high 0.28 \
      --use-tis \
      --optimizer adam \
      --lr 1e-6 \
      --lr-decay-style constant \
      --weight-decay 0.1 \
      --adam-beta1 0.9 \
      --adam-beta2 0.98 \
      --optimizer-cpu-offload \
      --overlap-cpu-optimizer-d2h-h2d \
      --use-precision-aware-optimizer \
      --use-wandb \
      --wandb-team ${WANDB_ENTITY} \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-group "${RUN_NAME}" \
      --wandb-key "${WANDB_API_KEY}" \
      --rollout-num-gpus-per-engine 8 \
      --sglang-mem-fraction-static 0.5 \
      ${DEBUG_ROLLOUT_ONLY} \
      --seed ${SEED} \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0 \
      --accumulate-allreduce-grads-in-fp32 \
      --attention-softmax-in-fp32 \
      --attention-backend flash \
      2>&1 | tee -a "${SAVE_PATH}/${RUN_NAME}/train_log.txt"
else
    # Worker nodes just keep Ray running
    echo "Worker node $(hostname) waiting..."
    while true; do
        sleep 60
        if ! ray status > /dev/null 2>&1; then
            echo "Ray cluster disconnected, exiting"
            exit 0
        fi
    done
fi

