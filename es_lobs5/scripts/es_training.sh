#!/bin/bash
#SBATCH --job-name=es-train
#SBATCH --nodes=8                # Override with sbatch --nodes=N for multi-node
#SBATCH --ntasks-per-node=1      # One task per node (for multi-node srun)
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/es_train_%j.out
#SBATCH --error=logs/es_train_%j.err
#SBATCH --partition=workq
#SBATCH --contiguous             # Ensure contiguous node allocation for multi-node
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid011108

# =============================================================================
# ES Training - Production Script
# =============================================================================
#
# PERGPU_PERTURBATIONS Scaling Test Results (2026-01-17):
# ------------------------------------------------------
# Max Stable:  14,336 (Total 57,344) - Job 1921017 - RUNNING
# First Fail:  16,384 (Total 65,536) - Job 1920937 - FAILED (OOM/Aborted)
#
# PERGPU_PERTURBATIONS Scaling Test Results (2026-01-18, LORA_V2):
# ------------------------------------------------------
# Job ID    Per-GPU Perturbations  Total Population  Status     Result/Error
# 1927436   64                     256               COMPLETED  ✅ Success (Timed out after 40m)
# 1927437   128                    512               FAILED     ❌ OOM (Alloc 71GB)
#
# PERGPU_PERTURBATIONS Scaling Test Results (2026-01-21, LORA_V1.5):
# ------------------------------------------------------
# With dots_with_no_batch_dims_saveable checkpoint policy (commit 2a9e899)
# Job ID    Per-GPU Perturbations  Total Population  Status     Result/Error
# 1950636   256                    1,024             COMPLETED  ✅ Success (0.19h)
# 1950637   512                    2,048             COMPLETED  ✅ Success (0.20h) ★ MAX STABLE
# 1950638   1024                   4,096             FAILED     ❌ OOM (~17.7GB alloc)
#
# Summary: 4x improvement over default remat (128 -> 512 per-GPU)
# Commits: 9ec0d2d (sweep script), 987a042 (results docs)
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
#   # Default configuration (single node, 4 GPUs)
#   sbatch es_lobs5/scripts/es_training.sh
#
#   # Multi-node training (2 nodes, 8 GPUs total)
#   sbatch --nodes=2 es_lobs5/scripts/es_training.sh
#
#   # Multi-node training (4 nodes, 16 GPUs total)
#   sbatch --nodes=4 --time=02:00:00 es_lobs5/scripts/es_training.sh
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
# Note: BASE_DIR not available here (before script runs), use absolute path
REPO_PATH="/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5"
GIT_BRANCH=$(git -C ${REPO_PATH} branch --show-current 2>/dev/null || echo "unknown")
GIT_COMMIT_SHORT=$(git -C ${REPO_PATH} rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT_FULL=$(git -C ${REPO_PATH} rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT_MSG=$(git -C ${REPO_PATH} log -1 --format='%s' 2>/dev/null || echo "unknown")

echo "=============================================="
echo " ES Training - Production"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: 4"
echo "Total GPUs: $((${SLURM_NNODES:-1} * 4))"
echo "Start time: $(date)"
echo "----------------------------------------------"
echo "Git Branch: ${GIT_BRANCH}"
echo "Git Commit: ${GIT_COMMIT_SHORT} (${GIT_COMMIT_FULL})"
echo "Commit Msg: ${GIT_COMMIT_MSG}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Multi-Node Detection & Setup
# -----------------------------------------------------------------------------
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=4
TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))

if [ "$NNODES" -gt 1 ]; then
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
    MASTER_PORT=29500
    export JAX_COORDINATOR_ADDRESS="$MASTER_ADDR:$MASTER_PORT"
    export MASTER_ADDR
    export MASTER_PORT
    echo "[*] Multi-node mode: $NNODES nodes, $TOTAL_GPUS total GPUs"
    echo "[*] Coordinator: $JAX_COORDINATOR_ADDRESS"
else
    echo "[*] Single-node mode: $TOTAL_GPUS GPUs"
fi

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Base paths (adjust if needed)
BASE_DIR="/lus/lfs1aip2/projects/s5e/quant"
CONDA_PATH="/projects/s5e/quant/miniforge3"

cd ${BASE_DIR}/AlphaTrade/LOBS5

source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate lobs5

# WandB configuration
export WANDB_API_KEY="41f4ee88a220359a48d63a1a4239c83862288bb0"

# Debug: verify conda activate worked
echo "DEBUG: Python after conda activate: $(which python)"
echo "DEBUG: CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
${CONDA_PATH}/envs/lobs5/bin/python -c "import sys; print(f'DEBUG: sys.executable = {sys.executable}')"

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
export PYTHONPATH="${BASE_DIR}/AlphaTrade/AlphaTrade:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Create Directories
# -----------------------------------------------------------------------------
mkdir -p logs
mkdir -p checkpoints/es_runs

# -----------------------------------------------------------------------------
# Default Parameters (override via environment variables)
# -----------------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-${BASE_DIR}/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/}"
DATA_DIR="${DATA_DIR:-${BASE_DIR}/JAN2023/GOOG_24tok_preproc}"

# Training scale
N_EPOCHS="${N_EPOCHS:-1000000}"
PERGPU_PERTURBATIONS="${PERGPU_PERTURBATIONS:-2048}"
# PERGPU_PERTURBATIONS="${PERGPU_PERTURBATIONS:-56}" # for FULL mode
# N_PERTURBATIONS is deprecated but kept for backward compatibility if set explicitly
N_PERTURBATIONS="${N_PERTURBATIONS:-}" 
N_STEPS="${N_STEPS:-10}"
N_WARMUP="${N_WARMUP:-10}"
BG_MSGS="${BG_MSGS:-50}"

# ES hyperparameters
SIGMA="${SIGMA:-0.2}"             # Initial sigma (0.2 for exploration)
SIGMA_DECAY="${SIGMA_DECAY:-0.9997}"  # Per-epoch decay (0.9997: 0.2->0.01 over 10k epochs)
SIGMA_MIN="${SIGMA_MIN:-0.01}"     # Floor value (stop decaying at this value)
LR="${LR:-0.001}"
NOISER="${NOISER:-eggroll}"
LORA_RANK="${LORA_RANK:-4}"

# Task configuration
TASK="${TASK:-sell}"
TASK_SIZE="${TASK_SIZE:-30}"
TICK_SIZE="${TICK_SIZE:-100}"

# Data window control (empty = random, number = fixed)
FILE_IDX="${FILE_IDX:-8}"
# FILE_IDX="${FILE_IDX:-}"  # Use this for random file selection

# Wandb
WANDB_PROJECT="${WANDB_PROJECT:-es-lobs5}"
WANDB_ENTITY="${WANDB_ENTITY:-kang-oxford}"

# Checkpointing
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_DIR}/AlphaTrade/LOBS5/checkpoints/es_runs/${SLURM_JOB_ID}}"

# Resume training (optional)
RESUME_FROM="${RESUME_FROM:-}"

# Random seed
SEED="${SEED:-2026}"

# Token mode
TOKEN_MODE="${TOKEN_MODE:-24}"

# Rank transform (helps escape local optima like "no trading")
RANK_TRANSFORM="${RANK_TRANSFORM:-true}"

# =============================================================================
# Training Mode (single variable to control all LoRA settings)
# =============================================================================
# MODE options (handled by Python's training_modes.py):
#   FULL      - Full parameter training (no LoRA)
#   LORA      - LoRA-only training (freeze non-LoRA params)
#   LORA+SSM  - LoRA + SSM params training (legacy)
#   LORA_V1.5 - LoRA on all projections, freeze SSM, train norms
#   LORA_V1.6 - LoRA on all projections, freeze SSM/norms [High Capacity, Recommended]
#   LORA_V2   - LoRA on all projections + train SSM/norms
# =============================================================================
MODE="${MODE:-LORA_V1.6}"

# Validate MODE (Python will also validate, but fail fast here)
case "${MODE}" in
    FULL|LORA|LORA+SSM|LORA_V1.5|LORA_V1.6|LORA_V2)
        ;;  # Valid mode
    *)
        echo "ERROR: Unknown MODE '${MODE}'. Valid options: FULL, LORA, LORA+SSM, LORA_V1.5, LORA_V1.6, LORA_V2"
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
printf "│            │ SIGMA                             │ %-14s │ %-59s │\n" "${SIGMA}" "Initial noise std (decays to SIGMA_MIN over epochs)"
printf "│            │ SIGMA_DECAY                       │ %-14s │ %-59s │\n" "${SIGMA_DECAY}" "Per-epoch decay (0.9997: 0.2->0.01 over 10k epochs)"
printf "│            │ SIGMA_MIN                         │ %-14s │ %-59s │\n" "${SIGMA_MIN}" "Minimum sigma floor (stop decay at this value)"
printf "│            │ LR                                │ %-14s │ %-59s │\n" "${LR}" "Learning rate"
printf "│            │ NOISER                            │ %-14s │ %-59s │\n" "${NOISER}" "Type of noise strategy used"
printf "│            │ LORA_RANK                         │ %-14s │ %-59s │\n" "${LORA_RANK}" "Rank for LoRA adapters"
printf "│            │ MODE                              │ %-14s │ %-59s │\n" "${MODE}" "Training mode (V1.6=high capacity, V1.5=train norms)"
printf "│            │ RANK_TRANSFORM                    │ %-14s │ %-59s │\n" "${RANK_TRANSFORM}" "Rank-based fitness shaping (escape local optima)"
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

# Build optional resume_from argument
if [ -n "${RESUME_FROM}" ]; then
    RESUME_FROM_ARG="--resume_from '${RESUME_FROM}'"
else
    RESUME_FROM_ARG=""
fi

# Determine perturbation argument
if [ -n "${N_PERTURBATIONS}" ]; then
    PERTURBATION_ARG="--n_perturbations ${N_PERTURBATIONS}"
else
    PERTURBATION_ARG="--pergpu_perturbations ${PERGPU_PERTURBATIONS}"
fi

# Build the common Python command arguments
PYTHON_ARGS="es_lobs5/scripts/es_training.py \
    --lobs5_checkpoint '${CHECKPOINT}' \
    --replay_data_path '${DATA_DIR}' \
    --n_epochs ${N_EPOCHS} \
    ${PERTURBATION_ARG} \
    --n_steps ${N_STEPS} \
    --n_warmup_msgs ${N_WARMUP} \
    --background_msgs_per_step ${BG_MSGS} \
    --sigma ${SIGMA} \
    --sigma_decay ${SIGMA_DECAY} \
    --sigma_min ${SIGMA_MIN} \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --noiser ${NOISER} \
    --background_mode historical_replay \
    --task ${TASK} \
    --task_size ${TASK_SIZE} \
    --tick_size ${TICK_SIZE} \
    --token_mode ${TOKEN_MODE} \
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --checkpoint_dir '${CHECKPOINT_DIR}' \
    --wandb_project '${WANDB_PROJECT}' \
    --wandb_entity '${WANDB_ENTITY}' \
    --mode '${MODE}' \
    --rank_transform ${RANK_TRANSFORM} \
    --seed ${SEED} \
    ${FILE_IDX_ARG} \
    ${RESUME_FROM_ARG}"

if [ "$NNODES" -gt 1 ]; then
    # ==========================================================================
    # Multi-Node Execution via srun with inline bash wrapper
    # ==========================================================================
    # Each node needs its own environment setup (conda, CUDA, LD_LIBRARY_PATH)
    # The wrapper runs identically on all nodes but with different SLURM_PROCID
    echo "[*] Launching multi-node training with srun..."

    srun --nodes=$NNODES \
         --ntasks=$NNODES \
         --ntasks-per-node=1 \
         --gres=gpu:$GPUS_PER_NODE \
         --gpu-bind=map_gpu:0,1,2,3 \
         --output=logs/es_train_${SLURM_JOB_ID}_node%n.log \
         --export=ALL \
         bash -c '
# ========== Inline Wrapper (runs on each node) ==========
echo "========================================"
echo "[Wrapper] Running on node: $(hostname)"
echo "[Wrapper] SLURM_NODEID: ${SLURM_NODEID:-N/A}"
echo "[Wrapper] SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "[Wrapper] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "========================================"

# Source conda (must re-source in each srun process)
source '"${CONDA_PATH}"'/etc/profile.d/conda.sh
conda activate lobs5

# Load CUDA module for multi-node communication
module load cuda/12.6

# Set LD_LIBRARY_PATH for NVIDIA libraries (NCCL, cuDNN, etc.)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# JAX environment for multi-node
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COORDINATOR_TIMEOUT_MS=600000
export JAX_PLATFORMS="cuda"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# NCCL timeout: 1 hour (prevents infinite hang on communication deadlock)
export NCCL_TIMEOUT=3600

# Multi-node JAX config (critical for distributed mesh)
if [ -n "$JAX_COORDINATOR_ADDRESS" ]; then
    echo "[Wrapper] Multi-node coordinator: $JAX_COORDINATOR_ADDRESS"
    export JAX_PROCESS_COUNT=${SLURM_NNODES:-1}
    export JAX_PROCESS_INDEX=${SLURM_PROCID:-0}
    export JAX_LOCAL_PROCESS_COUNT=1
    export JAX_LOCAL_PROCESS_INDEX=0
    echo "[Wrapper] JAX_PROCESS_COUNT=${JAX_PROCESS_COUNT}"
    echo "[Wrapper] JAX_PROCESS_INDEX=${JAX_PROCESS_INDEX}"
fi

# CUDA config
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$HOME/.nv/ComputeCache" || true

echo "[Wrapper] Available GPUs:"
nvidia-smi --list-gpus | head -4

# Run training
cd '"${BASE_DIR}"'/AlphaTrade/LOBS5
export PYTHONPATH="'"${BASE_DIR}"'/AlphaTrade/AlphaTrade:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Use explicit path to lobs5 python (conda activate may not work in srun subshell)
'"${CONDA_PATH}"'/envs/lobs5/bin/python -u -B '"${PYTHON_ARGS}"'
'
else
    # ==========================================================================
    # Single-Node Execution (direct Python call, existing behavior)
    # ==========================================================================
    ${CONDA_PATH}/envs/lobs5/bin/python es_lobs5/scripts/es_training.py \
        --lobs5_checkpoint "${CHECKPOINT}" \
        --replay_data_path "${DATA_DIR}" \
        --n_epochs ${N_EPOCHS} \
        ${PERTURBATION_ARG} \
        --n_steps ${N_STEPS} \
        --n_warmup_msgs ${N_WARMUP} \
        --background_msgs_per_step ${BG_MSGS} \
        --sigma ${SIGMA} \
        --sigma_decay ${SIGMA_DECAY} \
        --sigma_min ${SIGMA_MIN} \
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
        --mode "${MODE}" \
        --rank_transform ${RANK_TRANSFORM} \
        --seed ${SEED} \
        ${FILE_IDX_ARG} \
        "$@"
fi

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
