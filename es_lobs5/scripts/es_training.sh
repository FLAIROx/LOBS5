#!/bin/bash
#SBATCH --job-name=es-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/es_train_%j.out
#SBATCH --error=logs/es_train_%j.err
#SBATCH --partition=workq

# =============================================================================
# ES Training - Production Script
# =============================================================================
#
# Usage:
#   # Default configuration
#   sbatch es_lobs5/scripts/es_training.sh
#
#   # Custom parameters (via environment variables)
#   N_EPOCHS=2000 N_PERTURBATIONS=256 sbatch es_lobs5/scripts/es_training.sh
#
#   # Different wandb project
#   WANDB_PROJECT=my-experiment sbatch es_lobs5/scripts/es_training.sh
#
# Environment Variables:
#   CHECKPOINT        - Path to LOBS5 model checkpoint
#   DATA_DIR          - Path to preprocessed data (GOOG_24tok_preproc format)
#   N_EPOCHS          - Number of training epochs (default: 1000)
#   N_PERTURBATIONS   - Population size (default: 128, must divide by 4 GPUs)
#   N_STEPS           - Steps per episode (default: 100)
#   SIGMA             - Noise std (default: 0.01)
#   LR                - Learning rate (default: 0.001)
#   WANDB_PROJECT     - Wandb project name (default: es-lobs5-production)
#   WANDB_ENTITY      - Wandb entity (default: kang-oxford)
#
# =============================================================================

echo "=============================================="
echo " ES Training - Production"
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

# -----------------------------------------------------------------------------
# JAX Compilation Cache (must be set BEFORE Python/JAX starts)
# Caches XLA compilation results to disk for fast warm start (~20min → <2min)
# NOTE: Same config = cache hit; different config (N_STEPS, etc) = cache miss
# -----------------------------------------------------------------------------
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/es_lobs5_jax_compilation"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/AlphaTrade:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Create Directories
# -----------------------------------------------------------------------------
mkdir -p logs
mkdir -p checkpoints/es_runs

# -----------------------------------------------------------------------------
# Default Parameters (override via environment variables)
# -----------------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/}"
DATA_DIR="${DATA_DIR:-/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc}"

# Training scale
N_EPOCHS="${N_EPOCHS:-1000}"
N_PERTURBATIONS="${N_PERTURBATIONS:-128}"
N_STEPS="${N_STEPS:-100}"
N_WARMUP="${N_WARMUP:-500}"
BG_MSGS="${BG_MSGS:-10}"

# ES hyperparameters
SIGMA="${SIGMA:-0.01}"
LR="${LR:-0.001}"
NOISER="${NOISER:-eggroll}"
LORA_RANK="${LORA_RANK:-4}"

# Task configuration
TASK="${TASK:-sell}"
TASK_SIZE="${TASK_SIZE:-500}"
TICK_SIZE="${TICK_SIZE:-100}"

# Data window control (empty = random, number = fixed)
FILE_IDX="${FILE_IDX:-}"

# Wandb
WANDB_PROJECT="${WANDB_PROJECT:-es-lobs5}"
WANDB_ENTITY="${WANDB_ENTITY:-kang-oxford}"

# Checkpointing
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/es_runs/${SLURM_JOB_ID}}"

# Print configuration
echo ""
echo "Configuration:"
echo "  CHECKPOINT: ${CHECKPOINT}"
echo "  DATA_DIR: ${DATA_DIR}"
echo "  N_EPOCHS: ${N_EPOCHS}"
echo "  N_PERTURBATIONS: ${N_PERTURBATIONS}"
echo "  N_STEPS: ${N_STEPS}"
echo "  SIGMA: ${SIGMA}, LR: ${LR}, LORA_RANK: ${LORA_RANK}"
echo "  NOISER: ${NOISER}"
echo "  TASK: ${TASK} x ${TASK_SIZE}, TICK_SIZE: ${TICK_SIZE}"
echo "  WANDB: ${WANDB_PROJECT} / ${WANDB_ENTITY}"
echo "  CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "  FILE_IDX: ${FILE_IDX:-random}"
echo ""

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------
# Build optional file_idx argument
if [ -n "${FILE_IDX}" ]; then
    FILE_IDX_ARG="--file_idx ${FILE_IDX}"
else
    FILE_IDX_ARG=""
fi

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
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    ${FILE_IDX_ARG}

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
