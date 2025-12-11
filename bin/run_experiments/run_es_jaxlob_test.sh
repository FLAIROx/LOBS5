#!/bin/bash
#
# Wrapper script for LOBS5 ES Training with Eggroll
# Tests ES training with JaxLOB environment
#

# Debug output
echo "========================================"
echo "[ES Wrapper] Running on node: $(hostname)"
echo "[ES Wrapper] SLURM_NODEID: ${SLURM_NODEID:-N/A}"
echo "[ES Wrapper] SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "[ES Wrapper] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "========================================"

# Source conda
source ~/miniforge3/etc/profile.d/conda.sh

# Activate environment
conda activate lobs5

# Load CUDA module
module load cuda/12.6

# Set LD_LIBRARY_PATH for JAX CUDA libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvshmem/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# JAX memory management
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Other optimizations
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

# Ensure CUDA cache/temp go to a writable path
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "$SLURM_TMPDIR" ]; then
    export TMPDIR="$SLURM_TMPDIR"
    export CUDA_CACHE_PATH="$SLURM_TMPDIR/.nv/ComputeCache"
else
    export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
fi
mkdir -p "$CUDA_CACHE_PATH" || true

# Check available GPUs
echo "[ES Wrapper] Available GPUs:"
nvidia-smi --list-gpus | head -4

# ============================================
# ES Training Configuration
# ============================================
CHECKPOINT_PATH="checkpoints/lobs5_d1024_l12_b16_bsz13x4_seed42_jid1684154_3zf0yp50"
OUTPUT_DIR="es_checkpoints/test_$(date +%Y%m%d_%H%M%S)"
LOGS_DIR="logs_es_test"

echo "[ES Wrapper] ============================================"
echo "[ES Wrapper] ES Training Configuration:"
echo "[ES Wrapper]   Checkpoint: $CHECKPOINT_PATH"
echo "[ES Wrapper]   Output: $OUTPUT_DIR"
echo "[ES Wrapper]   Noiser: eggroll"
echo "[ES Wrapper]   Population size: 128 (32 per GPU)"
echo "[ES Wrapper]   Test episodes: 10"
echo "[ES Wrapper]   Steps per episode: 50"
echo "[ES Wrapper] ============================================"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# ============================================
# Run ES Training with JaxLOB
# ============================================
# -u: unbuffered output, -B: don't write .pyc files (always use fresh code)
python -u -B -m es_lobs5.training.es_jaxlob_train \
    --lobs5_checkpoint="${CHECKPOINT_PATH}" \
    --noiser=eggroll \
    --sigma=0.01 \
    --lr=0.001 \
    --lora_rank=4 \
    --n_threads=128 \
    --n_epochs=10 \
    --n_steps=50 \
    --world_msgs_per_step=10 \
    --task=sell \
    --task_size=500 \
    --tick_size=100 \
    --seed=42 \
    --output_dir="${OUTPUT_DIR}"

echo ""
echo "[ES Wrapper] ============================================"
echo "[ES Wrapper] ES Training Test Completed!"
echo "[ES Wrapper] Results saved to: $OUTPUT_DIR"
echo "[ES Wrapper] ============================================"
