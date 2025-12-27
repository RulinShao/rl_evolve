#!/bin/bash
#SBATCH --job-name=theta_evolve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         
#SBATCH --hint=nomultithread   
#SBATCH --account comem
#SBATCH --qos h200_comem_high
#SBATCH --mem 400G
#SBATCH --gres=gpu:8           
#SBATCH --time 120:00:00      
#SBATCH --requeue
#SBATCH --chdir=/checkpoint/comem/rulin/rl_evolve
#SBATCH --output=/checkpoint/comem/rulin/cache/slurm/theta_evolve-%A_%a.out
#SBATCH --array=0-7

set -ex

########################## ENVIRONMENT ACTIVATION #############################
# Set library paths BEFORE activating environment to ensure system libs take precedence
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH}
unset LD_PRELOAD

# Activate the slime-evolve environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate slime-evolve

# Re-export to ensure they're not overwritten by conda activation
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

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
NOTE=""

#### Replace with your own wandb settings ####
WANDB_API_KEY=local-8a8a005b87a483480b22f6f6b990bd15a3cea399
WANDB_ENTITY=Rulin
WANDB_PROJECT=theta-evolve

########################## DATABASE SIZE SWEEP #############################
# Job array index maps to different database population sizes
# Start from paper default (10000) and scale up to study larger sizes

declare -a DB_SIZES=(10000 20000 50000 100000 200000 500000 1000000 2000000)
DB_SIZE=${DB_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "=== Job Array Task ID: $SLURM_ARRAY_TASK_ID ==="
echo "=== Database Population Size: $DB_SIZE ==="

# Add database size to the note for tracking
NOTE="_dbsize${DB_SIZE}"

########################## END CONFIGURATION SECTION #############################

# Required for tensor parallelism
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

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

######################################################

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

########################## CREATE MODIFIED CONFIG WITH DB_SIZE #############################

# Create a modified config file with the swept database size
CONFIG_DIR=$(dirname $CONFIG_YAML)
MODIFIED_CONFIG="${CONFIG_DIR}/config_${TASK}_${CONFIG_POSTFIX}_dbsize${DB_SIZE}.yaml"

# Copy original config and modify population_size
cp $CONFIG_YAML $MODIFIED_CONFIG
sed -i "s/population_size: [0-9]*/population_size: ${DB_SIZE}/" $MODIFIED_CONFIG

echo "Created modified config: $MODIFIED_CONFIG with population_size: $DB_SIZE"

# Use the modified config
CONFIG_YAML=$MODIFIED_CONFIG

########################## MODEL SETUP #############################

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

# Use local Megatron-LM path
if [ ! -d "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist" ] || [ $FORCE_DOWNLOAD -eq 1 ]; then
    echo "Converting HF model to torch dist format..."
    PYTHONPATH=$SLIME_DIR:$MEGATRON_DIR python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} --hf-checkpoint $SAVE_SHM_DIR/$MODEL_NAME --save $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist
    echo "Conversion completed, torch dist model saved at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist"
else
    echo "Torch dist model already exists at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist, skipping conversion"
fi

########################## MAIN EXECUTION #############################

echo "=== Experiment Configuration ==="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "DATABASE_SIZE: ${DB_SIZE}"
echo "RUN_NAME: ${RUN_NAME}"
echo "TASK: ${TASK}"
echo "MODEL: ${MODEL_NAME}"
echo "INITIAL_PROGRAM: ${INITIAL_PROGRAM}"
echo "EVALUATOR_FILE: ${EVALUATOR_FILE}"
echo "CONFIG_YAML: ${CONFIG_YAML}"
echo "SAVE_PATH: ${SAVE_PATH}"
echo "================================"

mkdir -p "${SAVE_PATH}/${RUN_NAME}"

bash scripts_evolve/${MODEL_NAME}/general.sh \
    "${WANDB_PROJECT}" \
    "${RUN_NAME}" \
    "${INITIAL_PROGRAM}" \
    "${EVALUATOR_FILE}" \
    "${CONFIG_YAML}" \
    "${SAVE_PATH}" \
    "${IS_TRAINING}" \
    "${LAZY_OUTPUT_PENALTY}" \
    "${REWARD_PROCESS_TYPE}" \
    "${SEED}" \
    2>&1 | tee -a "${SAVE_PATH}/${RUN_NAME}/train_log.txt"

