#!/bin/bash
#SBATCH --job-name=theta_multi
#SBATCH --nodes=2                   # Adjust as needed
#SBATCH --ntasks-per-node=1         
#SBATCH --hint=nomultithread   
#SBATCH --account comem
#SBATCH --qos h200_comem_high
#SBATCH --mem 400G
#SBATCH --gres=gpu:8           
#SBATCH --time 120:00:00      
#SBATCH --requeue
#SBATCH --chdir=/checkpoint/comem/rulin/rl_evolve
#SBATCH --output=/checkpoint/comem/rulin/cache/slurm/theta_multi-%j.out

set -ex

########################## ENVIRONMENT ACTIVATION #############################
eval "$(micromamba shell hook --shell bash)"
micromamba activate slime-evolve

########################## MULTI-NODE RAY SETUP #############################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6379
GPUS_PER_NODE=8

echo "=== Multi-Node Setup ==="
echo "Nodes: $SLURM_NNODES, Master: $MASTER_ADDR"

# Start Ray cluster
if [ "$(hostname)" = "$MASTER_ADDR" ]; then
    ray start --head --node-ip-address $MASTER_ADDR --port $MASTER_PORT \
        --num-gpus $GPUS_PER_NODE --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port=8265
    
    sleep 15  # Wait for workers
    ray status
    
    # Run the main script from head node
    export MASTER_ADDR
    bash run_evolve.sh
else
    sleep 10  # Wait for head
    ray start --address=$MASTER_ADDR:$MASTER_PORT --num-gpus $GPUS_PER_NODE
    
    # Worker nodes just keep Ray running
    while ray status > /dev/null 2>&1; do sleep 60; done
fi
