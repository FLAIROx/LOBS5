#!/bin/bash
#SBATCH --job-name=single_opt_test
#SBATCH --output=logs/single_opt_%j.out
#SBATCH --error=logs/single_opt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=workq

mkdir -p logs

echo "========================================"
echo "Single Optimizer Test (BF16 + single AdamW)"
echo "========================================"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
module load cuda/12.6

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvshmem/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

export USE_BF16=1
export USE_SINGLE_OPTIMIZER=1  # Test single optimizer
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

python -u run_train.py \
    --model_preset=55M \
    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
    --dataset=lobster-prediction --merging=padded \
    --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG' \
    --data_mode='preproc' \
    --clip_eigs=True --activation_fn=half_glu1 \
    --dt_global=False --epochs=1 --jax_seed=42 \
    --opt_config=standard --p_dropout=0.0 \
    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
    --use_book_data=True --use_simple_book=False --book_transform=True \
    --masking=none \
    --num_devices=1 --n_data_workers=0 \
    --curtail_epochs=2 \
    --USE_WANDB=False

echo "========================================"
echo "Test completed: $(date)"
echo "========================================"
