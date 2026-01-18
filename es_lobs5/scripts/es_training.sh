#!/bin/bash
#SBATCH --job-name=es-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=logs/es_train_%j.out
#SBATCH --error=logs/es_train_%j.err
#SBATCH --partition=workq

# =============================================================================
# ES Training - Production Script
# =============================================================================
#
# PERGPU_PERTURBATIONS Scaling Test Results (2026-01-17):
# ------------------------------------------------------
# Max Stable:  14,336 (Total 57,344) - Job 1921017 - RUNNING
# First Fail:  16,384 (Total 65,536) - Job 1920937 - FAILED (OOM/Aborted)
#
# Jobs 18,432+ all fail with OOM (RESOURCE_EXHAUSTED ~48-80GB allocation)
# The "Aborted" status indicates XLA runtime forced abort to prevent deadlock
# after one replica hit OOM during distributed computation.
#
# Related files:
#   - es_lobs5/scripts/es_training.py     (entry point)
#   - es_lobs5/training/es_trainer.py     (core logic)
#
# Reference logs:
#   - logs/es_train_1921017.out/.err  (max stable run)
#   - logs/es_train_1920937.out/.err  (first OOM failure)
#   - logs/es_train_1921018.out/.err  (detailed OOM traceback)
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

# -----------------------------------------------------------------------------
# Git Version Info (captured at submission time)
# -----------------------------------------------------------------------------
GIT_BRANCH=$(git -C /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 branch --show-current 2>/dev/null || echo "unknown")
GIT_COMMIT_SHORT=$(git -C /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT_FULL=$(git -C /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT_MSG=$(git -C /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 log -1 --format='%s' 2>/dev/null || echo "unknown")

echo "=============================================="
echo " ES Training - Production"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4"
echo "Start time: $(date)"
echo "----------------------------------------------"
echo "Git Branch: ${GIT_BRANCH}"
echo "Git Commit: ${GIT_COMMIT_SHORT} (${GIT_COMMIT_FULL})"
echo "Commit Msg: ${GIT_COMMIT_MSG}"
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
PERGPU_PERTURBATIONS="${PERGPU_PERTURBATIONS:-56}"
# N_PERTURBATIONS is deprecated but kept for backward compatibility if set explicitly
N_PERTURBATIONS="${N_PERTURBATIONS:-}" 
N_STEPS="${N_STEPS:-10}"
N_WARMUP="${N_WARMUP:-10}"
BG_MSGS="${BG_MSGS:-50}"

# ES hyperparameters
SIGMA="${SIGMA:-0.01}"
LR="${LR:-0.001}"
NOISER="${NOISER:-eggroll}"
LORA_RANK="${LORA_RANK:-4}"

# Task configuration
TASK="${TASK:-sell}"
TASK_SIZE="${TASK_SIZE:-50}"
TICK_SIZE="${TICK_SIZE:-100}"

# Data window control (empty = random, number = fixed)
FILE_IDX="${FILE_IDX:-8}"
# FILE_IDX="${FILE_IDX:-}"  # Use this for random file selection

# Wandb
WANDB_PROJECT="${WANDB_PROJECT:-es-lobs5}"
WANDB_ENTITY="${WANDB_ENTITY:-kang-oxford}"

# Checkpointing
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/es_runs/${SLURM_JOB_ID}}"

# Random seed
SEED="${SEED:-2026}"

# Token mode
TOKEN_MODE="${TOKEN_MODE:-24}"

# =============================================================================
# Training Mode (single variable to control all LoRA settings)
# =============================================================================
# MODE options:
#   FULL     - Full parameter training (no LoRA)
#   LORA     - LoRA-only training (freeze non-LoRA params)
#   LORA+SSM - LoRA + SSM params training
#   LORA_V2  - LoRA v2: Expand LoRA to all projection matrices (ICLR 2025)
# =============================================================================
MODE="${MODE:-FULL}"

# Derive internal flags from MODE
case "${MODE}" in
    FULL)
        USE_LORA="False"
        FREEZE_NONLORA="False"
        LORA_V2="False"
        ;;
    LORA)
        USE_LORA="True"
        FREEZE_NONLORA="True"
        LORA_V2="False"
        ;;
    LORA+SSM)
        USE_LORA="True"
        FREEZE_NONLORA="False"
        LORA_V2="False"
        ;;
    LORA_V2)
        USE_LORA="True"
        FREEZE_NONLORA="False"
        LORA_V2="True"
        ;;
    *)
        echo "ERROR: Unknown MODE '${MODE}'. Valid options: FULL, LORA, LORA+SSM, LORA_V2"
        exit 1
        ;;
esac

# Print configuration table
echo ""
echo "┌────────────┬───────────────────────────────────┬────────────────┬─────────────────────────────────────────────────────────────┐"
echo "│  Category  │         Parameter / Flag          │     Value      │                         Description                         │"
echo "├────────────┼───────────────────────────────────┼────────────────┼─────────────────────────────────────────────────────────────┤"
printf "│ Task       │ TASK_SIZE                         │ %-14s │ %-59s │\n" "${TASK_SIZE}" "Total number of shares to sell/buy in the episode"
printf "│ Config     │ TASK                              │ %-14s │ %-59s │\n" "${TASK}" "Direction of the task (sell or buy)"
printf "│            │ TICK_SIZE                         │ %-14s │ %-59s │\n" "${TICK_SIZE}" "Tick size in cents (e.g., 100 = \$1.00)"
echo "├────────────┼───────────────────────────────────┼────────────────┼─────────────────────────────────────────────────────────────┤"
printf "│ Simulation │ N_STEPS                           │ %-14s │ %-59s │\n" "${N_STEPS}" "Number of simulation steps per episode"
printf "│            │ BG_MSGS (--background_msgs)       │ %-14s │ %-59s │\n" "${BG_MSGS}" "Background market messages processed per step"
printf "│            │ N_WARMUP (--n_warmup_msgs)        │ %-14s │ %-59s │\n" "${N_WARMUP}" "Warmup messages replayed before episode starts"
echo "├────────────┼───────────────────────────────────┼────────────────┼─────────────────────────────────────────────────────────────┤"
printf "│ Training   │ N_EPOCHS                          │ %-14s │ %-59s │\n" "${N_EPOCHS}" "Total number of training generations/epochs"
if [ -n "${N_PERTURBATIONS}" ]; then
printf "│            │ N_PERTURBATIONS                   │ %-14s │ %-59s │\n" "${N_PERTURBATIONS}" "Total population size (explicitly set)"
else
printf "│            │ PERGPU_PERTURBATIONS              │ %-14s │ %-59s │\n" "${PERGPU_PERTURBATIONS}" "Population per GPU (Total = ${PERGPU_PERTURBATIONS} x 4 GPUs = $((PERGPU_PERTURBATIONS * 4)))"
fi
printf "│            │ SIGMA                             │ %-14s │ %-59s │\n" "${SIGMA}" "Evolution Strategy noise standard deviation"
printf "│            │ LR                                │ %-14s │ %-59s │\n" "${LR}" "Learning rate"
printf "│            │ NOISER                            │ %-14s │ %-59s │\n" "${NOISER}" "Type of noise strategy used"
printf "│            │ LORA_RANK                         │ %-14s │ %-59s │\n" "${LORA_RANK}" "Rank for LoRA adapters"
printf "│            │ MODE                              │ %-14s │ %-59s │\n" "${MODE}" "Training mode: FULL / LORA / LORA+SSM / LORA_V2"
echo "├────────────┼───────────────────────────────────┼────────────────┼─────────────────────────────────────────────────────────────┤"
printf "│ System     │ CHECKPOINT_EVERY                  │ %-14s │ %-59s │\n" "${CHECKPOINT_EVERY}" "Epoch frequency to save checkpoints"
printf "│            │ TOKEN_MODE                        │ %-14s │ %-59s │\n" "${TOKEN_MODE}" "Token vocabulary mode (24 = base-100 encoding)"
printf "│            │ SEED                              │ %-14s │ %-59s │\n" "${SEED}" "Random seed for reproducibility"
echo "├────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤"
printf "│ Paths      │ CHECKPOINT                        │ %-78s │\n" "${CHECKPOINT}"
printf "│            │ DATA_DIR                          │ %-78s │\n" "${DATA_DIR}"
printf "│            │ CHECKPOINT_DIR                    │ %-78s │\n" "${CHECKPOINT_DIR}"
echo "├────────────┼───────────────────────────────────┼────────────────┬─────────────────────────────────────────────────────────────┤"
printf "│            │ FILE_IDX                          │ %-14s │ %-59s │\n" "${FILE_IDX}" "Data file index (random = random selection)"
echo "├────────────┼───────────────────────────────────┼────────────────┼─────────────────────────────────────────────────────────────┤"
printf "│ Logging    │ WANDB_PROJECT                     │ %-14s │ %-59s │\n" "${WANDB_PROJECT}" "Weights & Biases project name"
printf "│            │ WANDB_ENTITY                      │ %-14s │ %-59s │\n" "${WANDB_ENTITY}" "Weights & Biases entity/username"
echo "└────────────┴───────────────────────────────────┴────────────────┴─────────────────────────────────────────────────────────────┘"
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

# Determine perturbation argument
if [ -n "${N_PERTURBATIONS}" ]; then
    PERTURBATION_ARG="--n_perturbations ${N_PERTURBATIONS}"
else
    PERTURBATION_ARG="--pergpu_perturbations ${PERGPU_PERTURBATIONS}"
fi

python es_lobs5/scripts/es_training.py \
    --lobs5_checkpoint "${CHECKPOINT}" \
    --replay_data_path "${DATA_DIR}" \
    --n_epochs ${N_EPOCHS} \
    ${PERTURBATION_ARG} \
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
    --token_mode ${TOKEN_MODE} \
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --freeze_nonlora ${FREEZE_NONLORA} \
    --use_lora "${USE_LORA:-False}" \
    --lora_v2 "${LORA_V2}" \
    --seed ${SEED} \
    ${FILE_IDX_ARG} \
    "$@"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
