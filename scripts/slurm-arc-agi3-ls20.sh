#!/bin/bash
#SBATCH --job-name=arc-agi3-ls20
#SBATCH --output=/gpfs/scrubbed/rulins/logs/arc-agi3-ls20_%j.out
#SBATCH --error=/gpfs/scrubbed/rulins/logs/arc-agi3-ls20_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --account=rulins

# ============================================================================
# ARC-AGI-3 Training on ThetaEvolve
# 
# Usage:
#   sbatch scripts/slurm-arc-agi3-ls20.sh
#
# Prerequisites:
#   1. .env file with ARC_API_KEY at /gpfs/projects/kohlab/rulins/ThetaEvolve/.env
#   2. Container at /gpfs/scrubbed/rulins/slime_v0.5.0rc0-cu126.sif
#   3. Model downloaded (will auto-download on first run)
# ============================================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# Create log directory
mkdir -p /gpfs/scrubbed/rulins/logs

# Set paths
WORKSPACE=/gpfs/projects/kohlab/rulins/ThetaEvolve
CONTAINER=/gpfs/scrubbed/rulins/slime_v0.5.0rc0-cu126.sif
SAVE_PATH=/gpfs/scrubbed/rulins/save

# Check container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    echo "Please run: apptainer pull $CONTAINER docker://slimerl/slime:v0.5.0rc0-cu126"
    exit 1
fi

# Check .env exists
if [ ! -f "$WORKSPACE/.env" ]; then
    echo "ERROR: .env file not found at $WORKSPACE/.env"
    echo "Please create it with your ARC_API_KEY"
    exit 1
fi

# Set Apptainer cache (avoid home quota issues)
export APPTAINER_CACHEDIR=/gpfs/scrubbed/rulins/cache/apptainer
mkdir -p $APPTAINER_CACHEDIR

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run training inside container
echo "Starting training..."
apptainer exec --nv \
    --bind $WORKSPACE:/workspace \
    --bind /gpfs/scrubbed/rulins:/gpfs/scrubbed/rulins \
    --bind /gpfs/projects/kohlab/rulins:/gpfs/projects/kohlab/rulins \
    --pwd /workspace \
    $CONTAINER \
    bash -c "
        set -e  # Exit on error
        
        # Install local packages (needed on first run or after changes)
        echo 'Installing ThetaEvolve...'
        cd /workspace
        pip install -e .
        
        echo 'Installing openevolve...'
        cd openevolve_adapted
        pip install --ignore-installed blinker
        rm -rf openevolve.egg-info && pip install -e .
        cd /workspace

        # Set environment
        export SAVE_PATH=$SAVE_PATH

        # Run training
        echo 'Starting training script...'
        bash scripts/run-arc-agi3-ls20.sh
    "

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "=============================================="

