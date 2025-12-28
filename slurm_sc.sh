#!/bin/bash
#SBATCH --job-name=theta_evolve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         
#SBATCH --hint=nomultithread   
#SBATCH --account comem
#SBATCH --qos h200_comem_high
#SBATCH --mem 1000G
#SBATCH --gres=gpu:8           
#SBATCH --time 7-00:00:00      
#SBATCH --requeue
#SBATCH --chdir=/checkpoint/comem/rulin/rl_evolve
#SBATCH --output=/checkpoint/comem/rulin/cache/slurm/theta_evolve-%A_%a.out
#SBATCH --array=0-5

set -ex

########################## 2D SWEEP: DB_SIZE x SEED #############################
declare -a DB_SIZES=(10000 100000 500000)
declare -a SEEDS=(3407 42)

# Map array task ID to 2D indices
# Task 0: DB=10000,  SEED=3407
# Task 1: DB=10000,  SEED=42
# Task 2: DB=100000, SEED=3407
# Task 3: DB=100000, SEED=42
# Task 4: DB=500000, SEED=3407
# Task 5: DB=500000, SEED=42

NUM_SEEDS=${#SEEDS[@]}
DB_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

DB_SIZE=${DB_SIZES[$DB_IDX]}
SWEEP_SEED=${SEEDS[$SEED_IDX]}
JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Scale timeout linearly with datastore size
# Base: 1 hour, +30 min per 100k datastore size
TIMEOUT_SEC=$((3600 + DB_SIZE / 100000 * 1800))
TIMEOUT_MS=$((TIMEOUT_SEC * 1000))

export NCCL_TIMEOUT=$TIMEOUT_SEC
export GLOO_TIMEOUT=$TIMEOUT_MS

echo "=== Job: $JOB_ID, DB Size: $DB_SIZE, Seed: $SWEEP_SEED, Timeout: ${TIMEOUT_SEC}s ==="

########################## CREATE JOB-SPECIFIC CONFIG #############################
TASK="circle_packing_modular"
ORIG_CONFIG="openevolve_adapted/examples/${TASK}/configs/config_${TASK}_it_XL.yaml"
JOB_CONFIG="openevolve_adapted/examples/${TASK}/configs/config_${TASK}_it_XL_job${JOB_ID}.yaml"

# Create job-specific config (no interference between jobs)
cp $ORIG_CONFIG $JOB_CONFIG
sed -i "s/population_size: [0-9]*/population_size: ${DB_SIZE}/" $JOB_CONFIG

########################## CREATE JOB-SPECIFIC RUN SCRIPT #############################
JOB_SCRIPT="run_evolve_job${JOB_ID}.sh"
cp run_evolve.sh $JOB_SCRIPT

# Modify the job-specific script to use our config, seed, and note
# NOTE: SEED is already included in POSTFIX_STR as "_seed${SEED}", so only add dbsize to NOTE
sed -i 's/^NOTE=""/NOTE="_dbsize'${DB_SIZE}'"/' $JOB_SCRIPT
sed -i 's/^SEED=.*/SEED='${SWEEP_SEED}'/' $JOB_SCRIPT
sed -i 's/CONFIG_POSTFIX="it_XL"/CONFIG_POSTFIX="it_XL_job'${JOB_ID}'"/' $JOB_SCRIPT

########################## RUN #############################
bash $JOB_SCRIPT

########################## CLEANUP #############################
rm -f $JOB_CONFIG $JOB_SCRIPT
