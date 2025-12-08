#!/bin/bash
#SBATCH --job-name=es_lobs5
#SBATCH --output=logs/es_lobs5_%j.out
#SBATCH --error=logs/es_lobs5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# ES Training Script for LOBS5
# Based on HyperscaleES framework

set -e

# ==============================================================================
# Environment Setup
# ==============================================================================

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# CUDA setup
module load cuda/12.6 2>/dev/null || true

# JAX configuration
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Suppress warnings
export TF_CPP_MIN_LOG_LEVEL=2

# ==============================================================================
# Training Configuration
# ==============================================================================

# Data paths
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG"
OUTPUT_DIR="./es_checkpoints/$(date +%Y%m%d_%H%M%S)"

# Model architecture (smaller than gradient training for ES efficiency)
D_MODEL=512
D_OUTPUT=150  # Vocab size
D_BOOK=503
N_MESSAGE_LAYERS=2
N_FUSED_LAYERS=4
N_BOOK_PRE_LAYERS=1
N_BOOK_POST_LAYERS=1
SSM_SIZE=512
BLOCKS=16

# SSM configuration
C_INIT="trunc_standard_normal"
DISCRETIZATION="zoh"
DT_MIN=0.001
DT_MAX=0.1
ACTIVATION="half_glu1"

# ES configuration
NOISER="eggroll"  # Options: open_es, eggroll, eggrollbs, sparse
SIGMA=0.01        # Noise standard deviation
LR=0.001          # Learning rate
LORA_RANK=4       # LORA rank for eggroll
THREADS_PER_GPU=64  # Number of perturbations per GPU

# Training configuration
NUM_EPOCHS=1000
MSG_SEQ_LEN=500
SEED=42
VALIDATE_EVERY=50
SAVE_EVERY=100

# Precision
USE_BF16=true

# W&B logging (optional)
WANDB_PROJECT=""  # Set to enable W&B logging
WANDB_ENTITY=""

# Checkpoint initialization (optional)
# Set to gradient-trained checkpoint path to initialize from pretrained weights
INIT_CHECKPOINT=""  # e.g., "/path/to/checkpoints/lobs5_d3072_xxx/"

# ==============================================================================
# Create output directory
# ==============================================================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=============================================="
echo "ES Training for LOBS5"
echo "=============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Model: d_model=${D_MODEL}, ssm_size=${SSM_SIZE}, blocks=${BLOCKS}"
echo "ES: noiser=${NOISER}, sigma=${SIGMA}, lr=${LR}, rank=${LORA_RANK}"
echo "Threads per GPU: ${THREADS_PER_GPU}"
echo "=============================================="

# ==============================================================================
# Run Training
# ==============================================================================

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

python -m es_lobs5.training.es_train \
    --data_dir="${DATA_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --d_model=${D_MODEL} \
    --d_output=${D_OUTPUT} \
    --d_book=${D_BOOK} \
    --n_message_layers=${N_MESSAGE_LAYERS} \
    --n_fused_layers=${N_FUSED_LAYERS} \
    --n_book_pre_layers=${N_BOOK_PRE_LAYERS} \
    --n_book_post_layers=${N_BOOK_POST_LAYERS} \
    --ssm_size=${SSM_SIZE} \
    --blocks=${BLOCKS} \
    --C_init="${C_INIT}" \
    --discretization="${DISCRETIZATION}" \
    --dt_min=${DT_MIN} \
    --dt_max=${DT_MAX} \
    --activation="${ACTIVATION}" \
    --prenorm=True \
    --mode="none" \
    --noiser="${NOISER}" \
    --sigma=${SIGMA} \
    --lr=${LR} \
    --lora_rank=${LORA_RANK} \
    --threads_per_gpu=${THREADS_PER_GPU} \
    --num_epochs=${NUM_EPOCHS} \
    --msg_seq_len=${MSG_SEQ_LEN} \
    --seed=${SEED} \
    --validate_every=${VALIDATE_EVERY} \
    --save_every=${SAVE_EVERY} \
    --use_bf16=${USE_BF16} \
    --ignore_time_tokens=True \
    ${WANDB_PROJECT:+--wandb_project="${WANDB_PROJECT}"} \
    ${WANDB_ENTITY:+--wandb_entity="${WANDB_ENTITY}"} \
    ${INIT_CHECKPOINT:+--init_checkpoint="${INIT_CHECKPOINT}"}

echo "=============================================="
echo "Training Complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=============================================="
