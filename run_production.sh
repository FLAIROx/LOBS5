#!/bin/bash
#SBATCH --job-name=es-lobs5-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/production_%j.out
#SBATCH --error=logs/production_%j.err
#SBATCH --partition=gh200

# ==============================================================================
# ES-LOBS5 Production Training SLURM Script
# ==============================================================================
#
# Usage:
#   # Quick test (5 epochs)
#   sbatch run_production.sh quick_test
#
#   # Medium training (100 epochs)
#   sbatch run_production.sh medium
#
#   # Production (1000 epochs)
#   sbatch run_production.sh production
#
#   # Resume from checkpoint
#   RESUME_FROM=/path/to/checkpoint sbatch run_production.sh production
#
#   # Custom configuration
#   N_THREADS=256 N_EPOCHS=500 sbatch run_production.sh
#
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
echo "============================================================"
echo "ES-LOBS5 Production Training"
echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "------------------------------------------------------------"

# Create logs directory
mkdir -p logs

# Activate conda environment
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Set working directory
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# JAX configuration
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True"
export JAX_TRACEBACK_FILTERING=off

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PRESET="${1:-production}"
N_THREADS="${N_THREADS:-}"
N_EPOCHS="${N_EPOCHS:-}"
N_STEPS="${N_STEPS:-}"
SIGMA="${SIGMA:-}"
LR="${LR:-}"
RESUME_FROM="${RESUME_FROM:-}"
WANDB_NAME="${WANDB_NAME:-job_${SLURM_JOB_ID}}"

echo "Configuration:"
echo "  Preset:      $PRESET"
echo "  N_THREADS:   ${N_THREADS:-from preset}"
echo "  N_EPOCHS:    ${N_EPOCHS:-from preset}"
echo "  N_STEPS:     ${N_STEPS:-from preset}"
echo "  RESUME_FROM: ${RESUME_FROM:-none}"
echo "============================================================"
echo ""

# ------------------------------------------------------------------------------
# Build Command
# ------------------------------------------------------------------------------
CMD="python production_train.py"

# Add preset if specified and not "custom"
if [[ "$PRESET" != "custom" ]]; then
    CMD="$CMD --preset $PRESET"
fi

# Override with environment variables if set
[[ -n "$N_THREADS" ]] && CMD="$CMD --n_threads $N_THREADS"
[[ -n "$N_EPOCHS" ]] && CMD="$CMD --n_epochs $N_EPOCHS"
[[ -n "$N_STEPS" ]] && CMD="$CMD --n_steps $N_STEPS"
[[ -n "$SIGMA" ]] && CMD="$CMD --sigma $SIGMA"
[[ -n "$LR" ]] && CMD="$CMD --lr $LR"
[[ -n "$RESUME_FROM" ]] && CMD="$CMD --resume_from $RESUME_FROM"

# Add wandb name
CMD="$CMD --wandb_name $WANDB_NAME"

echo "Running: $CMD"
echo ""

# ------------------------------------------------------------------------------
# Run Training
# ------------------------------------------------------------------------------
$CMD

# ------------------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Training Complete!"
echo "End time: $(date)"
echo "============================================================"
