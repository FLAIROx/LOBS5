#!/bin/bash
#SBATCH --job-name=es-bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/es_bench_%j.out
#SBATCH --error=logs/es_bench_%j.err
#SBATCH --partition=workq

# =============================================================================
# ES Training Benchmark Script
# =============================================================================
# Purpose: Run short training for performance baseline measurement
#
# Usage:
#   sbatch es_lobs5/scripts/benchmark_es_training.sh
#
# Output: Timing information for N_EPOCHS epochs
# =============================================================================

echo "=============================================="
echo " ES Training - Benchmark"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4"
echo "Start time: $(date)"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/AlphaTrade:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Create Directories
# -----------------------------------------------------------------------------
mkdir -p logs
mkdir -p checkpoints/es_benchmark

# -----------------------------------------------------------------------------
# Benchmark Parameters (short run for timing)
# -----------------------------------------------------------------------------
CHECKPOINT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc"

# Short benchmark: 10 epochs
N_EPOCHS=10
N_PERTURBATIONS=128
N_STEPS=100
N_WARMUP=500
BG_MSGS=10

# ES hyperparameters
SIGMA=0.01
LR=0.001
NOISER=eggroll
LORA_RANK=4

# Task configuration
TASK=sell
TASK_SIZE=500
TICK_SIZE=100

# Disable wandb for benchmark
WANDB_PROJECT=""

# Fixed file index for reproducibility
FILE_IDX=0

echo ""
echo "Benchmark Configuration:"
echo "  N_EPOCHS: ${N_EPOCHS}"
echo "  N_PERTURBATIONS: ${N_PERTURBATIONS}"
echo "  N_STEPS: ${N_STEPS}"
echo "  FILE_IDX: ${FILE_IDX} (fixed for reproducibility)"
echo ""

# -----------------------------------------------------------------------------
# Timing wrapper
# -----------------------------------------------------------------------------
echo "=============================================="
echo " Starting Benchmark Run"
echo "=============================================="
START_TIME=$(date +%s.%N)

python es_lobs5/scripts/es_training.py \
    --lobs5_checkpoint "${CHECKPOINT}" \
    --replay_data_path "${DATA_DIR}" \
    --n_epochs ${N_EPOCHS} \
    --n_perturbations ${N_PERTURBATIONS} \
    --n_steps ${N_STEPS} \
    --n_warmup_msgs ${N_WARMUP} \
    --background_msgs_per_step ${BG_MSGS} \
    --sigma ${SIGMA} \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --noiser ${NOISER} \
    --background_mode historical_replay \
    --task ${TASK} \
    --task_size ${TASK_SIZE} \
    --tick_size ${TICK_SIZE} \
    --token_mode 24 \
    --checkpoint_every 100 \
    --checkpoint_dir "checkpoints/es_benchmark/${SLURM_JOB_ID}" \
    --file_idx ${FILE_IDX} 2>&1 | grep -v "sol_gpu_cost_model"

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
PER_EPOCH=$(echo "scale=2; $ELAPSED / $N_EPOCHS" | bc)

echo ""
echo "=============================================="
echo " Benchmark Results"
echo "=============================================="
echo "Total time: ${ELAPSED} seconds"
echo "Per-epoch time: ${PER_EPOCH} seconds"
echo "N_EPOCHS: ${N_EPOCHS}"
echo "N_PERTURBATIONS: ${N_PERTURBATIONS}"
echo "N_STEPS: ${N_STEPS}"
echo "End time: $(date)"
echo "=============================================="
