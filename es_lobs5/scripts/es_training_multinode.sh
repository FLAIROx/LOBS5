#!/bin/bash
#SBATCH --job-name=es-multinode
#SBATCH --nodes=2             # Default, override with sbatch --nodes=N
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=02:00:00        # Short time for testing
#SBATCH --output=logs/es_multinode_%j.out
#SBATCH --error=logs/es_multinode_%j.err
#SBATCH --partition=workq
#SBATCH --contiguous
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid011108

# =============================================================================
# ES Multi-Node Training Script
# =============================================================================
# Based on working pattern from train_full_autoreg.batch
#
# Usage:
#   sbatch --nodes=4 es_lobs5/scripts/es_training_multinode.sh
# =============================================================================

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "============================================"

# Configuration
GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))

# Multi-Node Setup (EXACT PATTERN from train_full_autoreg.batch)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500
export JAX_COORDINATOR_ADDRESS="$MASTER_ADDR:$MASTER_PORT"
export MASTER_ADDR
export MASTER_PORT

echo "[*] Multi-node mode: $NNODES nodes"
echo "[*] Coordinator: $JAX_COORDINATOR_ADDRESS"

# Configuration
CHECKPOINT="${CHECKPOINT:-/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/}"
DATA_DIR="${DATA_DIR:-/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc}"

N_EPOCHS="${N_EPOCHS:-20}"
PERGPU_PERTURBATIONS="${PERGPU_PERTURBATIONS:-128}"
N_STEPS="${N_STEPS:-50}"
BG_MSGS="${BG_MSGS:-50}"
TASK_SIZE="${TASK_SIZE:-50}"

WANDB_PROJECT="${WANDB_PROJECT:-es-multinode-test}"
WANDB_ENTITY="${WANDB_ENTITY:-kang-oxford}"

CHECKPOINT_DIR="checkpoints/es_multinode/${SLURM_JOB_ID}"

echo "[*] Configuration:"
echo "    CHECKPOINT: $CHECKPOINT"
echo "    DATA_DIR: $DATA_DIR"
echo "    N_EPOCHS: $N_EPOCHS"
echo "    PERGPU_PERTURBATIONS: $PERGPU_PERTURBATIONS"
echo "    WANDB_PROJECT: $WANDB_PROJECT"

# Launch with srun - inline wrapper (EXACT PATTERN from train_full_autoreg.batch)
srun --nodes=$NNODES \
     --ntasks=$NNODES \
     --ntasks-per-node=1 \
     --gres=gpu:$GPUS_PER_NODE \
     --gpu-bind=map_gpu:0,1,2,3 \
     --output=logs/es_multinode_${SLURM_JOB_ID}_node%n.log \
     --export=ALL \
     bash -c '
# ========== Inline Wrapper ==========
echo "========================================"
echo "[Wrapper] Running on node: $(hostname)"
echo "[Wrapper] SLURM_NODEID: ${SLURM_NODEID:-N/A}"
echo "[Wrapper] SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "[Wrapper] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "========================================"

# Source conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Load CUDA module
module load cuda/12.6

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# JAX environment
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COORDINATOR_TIMEOUT_MS=600000
export JAX_PLATFORMS="cuda"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Multi-node JAX config
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
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/AlphaTrade:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python -u -B es_lobs5/scripts/es_training.py \
    --lobs5_checkpoint "'"${CHECKPOINT}"'" \
    --replay_data_path "'"${DATA_DIR}"'" \
    --background_mode historical_replay \
    --n_epochs '${N_EPOCHS}' \
    --pergpu_perturbations '${PERGPU_PERTURBATIONS}' \
    --n_steps '${N_STEPS}' \
    --n_warmup_msgs 500 \
    --background_msgs_per_step '${BG_MSGS}' \
    --task sell \
    --task_size '${TASK_SIZE}' \
    --tick_size 100 \
    --checkpoint_dir "'"${CHECKPOINT_DIR}"'" \
    --wandb_project "'"${WANDB_PROJECT}"'" \
    --wandb_entity "'"${WANDB_ENTITY}"'"
'

echo "============================================"
echo "Training completed at: $(date)"
echo "============================================"
