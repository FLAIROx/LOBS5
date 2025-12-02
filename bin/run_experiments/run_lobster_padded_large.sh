#!/bin/bash
#
# Wrapper script for AlphaTrade LOBS5 training
# Sets up environment and runs training
#

# Source JAX/XLA optimizations
# DISABLED: jax_xla_optimization.sh causes 2x slowdown due to:
#   - CUDA_DEVICE_MAX_CONNECTIONS=1 (serializes kernels)
#   - --xla_gpu_force_compilation_parallelism=1 (single-threaded)
# SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
# if [ -f "$SCRIPT_DIR/jax_xla_optimization.sh" ]; then
#     source "$SCRIPT_DIR/jax_xla_optimization.sh"
# else
#     echo "[WARNING] JAX optimization script not found, using default settings"
# fi
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

# Debug output
echo "========================================"
echo "[Wrapper] Running on node: $(hostname)"
echo "[Wrapper] SLURM_NODEID: ${SLURM_NODEID:-N/A}"
echo "[Wrapper] SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "[Wrapper] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "========================================"

# Source conda
source ~/miniforge3/etc/profile.d/conda.sh

# Activate environment
conda activate lobs5

# Load CUDA module
module load cuda/12.6

# Set LD_LIBRARY_PATH for JAX CUDA libraries (complete list from working environment)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvshmem/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# **************** BF16 vs FP32 梯度对比测试 ****************
# USE_BF16=1 (default): Enable BF16 mixed precision
# USE_BF16=0: Disable BF16, use FP32
# GRAD_STATS_PRECISION: Label for gradient stats JSON output (bf16/fp32)
export USE_BF16=${USE_BF16:-1}
export GRAD_STATS_PRECISION=${GRAD_STATS_PRECISION:-bf16_gpt_init}
echo "[Wrapper] Precision mode: USE_BF16=$USE_BF16, GRAD_STATS_PRECISION=$GRAD_STATS_PRECISION"
# **************** BF16 vs FP32 梯度对比测试 ****************

# **************** MULTI NODES COMMUNICATIONS OPTIMIZATION ****************
# ✓ Critical: Enable memory preallocate to eliminate 80ms overhead per batch
export XLA_PYTHON_CLIENT_PREALLOCATE=true   # Preallocate GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90    # Use 90% GPU memory 

# XLA optimization 
# export XLA_FLAGS="--xla_gpu_enable_analytical_sol_latency_estimator=false"

# NCCL multi-node optimization 
# Disable P2P and SHM for cross-node communication via Slingshot
# export NCCL_P2P_DISABLE=1
# export NCCL_P2P_DIRECT_DISABLE=1
# export NCCL_SHM_DISABLE=1

# export JAX_TRACEBACK_IN_LOCATIONS=1
# export TF_GPU_ALLOCATOR=cuda_malloc_async

# JAX distributed coordination service timeout configuration (10 minutes for multi-node)
export JAX_COORDINATOR_TIMEOUT_MS=600000  # 10 minutes in milliseconds
export TF_CPP_MIN_LOG_LEVEL=1  # Reduce TensorFlow spam but keep important messages

# export NCCL_IB_DISABLE=1       # Disable InfiniBand (Slingshot uses Ethernet)
# **************** MULTI NODES COMMUNICATIONS OPTIMIZATION ****************

# Multi-node communication configuration (passed from batch script)
if [ -n "$JAX_COORDINATOR_ADDRESS" ]; then
    echo "[Wrapper] Multi-node coordinator: $JAX_COORDINATOR_ADDRESS"
    # Provide JAX process topology from Slurm
    export JAX_PROCESS_COUNT=${SLURM_NNODES:-1}
    export JAX_PROCESS_INDEX=${SLURM_PROCID:-0}
    # Single task per node → one local process per node
    export JAX_LOCAL_PROCESS_COUNT=1
    export JAX_LOCAL_PROCESS_INDEX=0
    # Make JAX device visibility match CUDA
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        export JAX_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    fi
    echo "[Wrapper] JAX_PROCESS_COUNT=${JAX_PROCESS_COUNT}"
    echo "[Wrapper] JAX_PROCESS_INDEX=${JAX_PROCESS_INDEX}"
    echo "[Wrapper] JAX_LOCAL_PROCESS_COUNT=${JAX_LOCAL_PROCESS_COUNT}"
    echo "[Wrapper] JAX_LOCAL_PROCESS_INDEX=${JAX_LOCAL_PROCESS_INDEX}"
fi

# Ensure CUDA cache/temp go to a writable path (avoid ptxas tmp errors)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "$SLURM_TMPDIR" ]; then
    export TMPDIR="$SLURM_TMPDIR"
    export CUDA_CACHE_PATH="$SLURM_TMPDIR/.nv/ComputeCache"
else
    export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
fi
mkdir -p "$CUDA_CACHE_PATH" || true

# Check available GPUs
echo "[Wrapper] Available GPUs:"
nvidia-smi --list-gpus | head -4

# Run Python with all arguments passed through
# -u: unbuffered output for real-time logging
# -B: don't write .pyc files
# 3072 d_model x 32 layers x 48 blocks (from commit 12fe4a9)
python -u -B run_train.py \
        --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
        --blocks=48 --per_gpu_bsz=2 --d_model=3072 --dataset=lobster-prediction --merging=padded \
        --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG' \
        --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc' \
        --data_mode='preproc' \
        --clip_eigs=True --activation_fn=half_glu1 \
        --dt_global=False --epochs=20 --jax_seed=42 --lr_factor=1 --n_layers=32 \
        --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0001 --ssm_size_base=3072 \
        --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
        --use_book_data=True --use_simple_book=False --book_transform=True  \
        --masking=none \
        --num_devices=4 --n_data_workers=4 \
        --debug_loading=False \
        --enable_profiler=False \
        --random_offsets_train=True \
        --shuffle_train=True \
        --debug_overfit=False \
        --lr_patience=3 \
        --use_remat=True \
        --gradient_accumulation_steps=4 \
        --USE_WANDB=True \
        --wandb_project=lobs5-3072x32-tok24 \
        --wandb_entity=kang-oxford


        # --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOGL' \
        # --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_encoded' \
        # --data_mode='encoded' \
        # ························································
        # --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded' \
        # ============= 原始大模型配置 (3072x32) =================
        # --blocks=48 --d_model=3072 --n_layers=32 --ssm_size_base=3072 \
        # --epochs=20 --USE_WANDB=True --wandb_project=lobs5-3072x32-tok24 \
        # =========================================================


        # ============= Data choices =================
        # ············· choice A1 ·············
        # --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021' \
        # --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2' \
        # --data_mode='preproc' \


        # ············· choice A2 ·············
        # --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded24' \
        # --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2_encoded24' \
        # --data_mode='encoded' \
        # ============= Data choices =================


        # ============= Resume Training =================
        # Intra-epoch checkpoint is automatically saved every segment (~1.8 hours).
        # To resume training, simply re-run this script with the SAME wandb_project.
        # The training will automatically:
        #   1. Detect the latest checkpoint in checkpoints/{run_name}_{run_id}/
        #   2. Restore model state and training progress
        #   3. Recreate dataloader with saved seed (for random_offsets_train=True)
        #   4. Skip already trained batches and continue from the next segment
        #
        # Resume scenarios:
        #   - Mid-epoch (e.g., segment 2/5): Uses saved dataloader_seed, skips batches
        #   - New epoch (e.g., segment 5/5 -> epoch+1): Generates fresh seed, no skip
        #
        # Note: Checkpoint directory is named after wandb run (e.g., ruby-aardvark-62_98nov1i7)
        #       Make sure to use consistent wandb settings when resuming.
        #
        # Manual restore from specific checkpoint (optional):
        # --restore='checkpoints/ruby-aardvark-62_98nov1i7' \
        # --restore_step=37 \
        # ============= Resume Training =================