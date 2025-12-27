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

########################## DATABASE SIZE SWEEP #############################
declare -a DB_SIZES=(10000 20000 50000 100000 200000 500000 1000000 2000000)
DB_SIZE=${DB_SIZES[$SLURM_ARRAY_TASK_ID]}
JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "=== Job: $JOB_ID, DB Size: $DB_SIZE ==="

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

# Modify the job-specific script to use our config and note
sed -i 's/^NOTE=""/NOTE="_dbsize'${DB_SIZE}'"/' $JOB_SCRIPT
sed -i 's/CONFIG_POSTFIX="it_XL"/CONFIG_POSTFIX="it_XL_job'${JOB_ID}'"/' $JOB_SCRIPT

########################## RUN #############################
bash $JOB_SCRIPT

########################## CLEANUP #############################
rm -f $JOB_CONFIG $JOB_SCRIPT
