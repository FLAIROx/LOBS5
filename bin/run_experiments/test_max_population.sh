#!/bin/bash
#
# Test ES training with different population sizes to find maximum
# Usage: bash test_max_population.sh <n_threads>
#

N_THREADS=${1:-128}

echo "========================================"
echo "Testing ES with n_threads=$N_THREADS"
echo "========================================"

# Source conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Load CUDA
module load cuda/12.6

# Set environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvshmem/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_GPU_ALLOCATOR=cuda_malloc_async

CHECKPOINT_PATH="checkpoints/lobs5_d1024_l12_b16_bsz13x4_seed42_jid1684154_3zf0yp50"

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Run short test (only 1 epoch, 20 steps)
python -u -B -m es_lobs5.training.es_jaxlob_train \
    --lobs5_checkpoint="${CHECKPOINT_PATH}" \
    --noiser=eggroll \
    --sigma=0.01 \
    --lr=0.001 \
    --lora_rank=4 \
    --n_threads=$N_THREADS \
    --n_epochs=1 \
    --n_steps=20 \
    --world_msgs_per_step=5 \
    --task=sell \
    --task_size=500 \
    --tick_size=100 \
    --seed=42 \
    --output_dir="es_checkpoints/maxtest_${N_THREADS}"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: n_threads=$N_THREADS works!"
else
    echo "✗ FAILED: n_threads=$N_THREADS OOM or error"
fi
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
