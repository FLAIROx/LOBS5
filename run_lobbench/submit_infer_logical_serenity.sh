#!/bin/bash
###############################################################################
# Inference Job for logical-serenity-19 (360M model, d_model=2048, n_layers=24)
# Generates samples for lob_bench evaluation
###############################################################################

#SBATCH --job-name=infer_logical_serenity
#SBATCH --output=logs_lobs5/infer_logical_serenity_%j.out
#SBATCH --error=logs_lobs5/infer_logical_serenity_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5  # Changed from 'lob' to 'lobs5' for correct environment

export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export JAX_PLATFORMS=cuda

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

CHECKPOINT_PATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me"
CHECKPOINT_STEP=89410
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc"
SAVE_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output/logical-serenity-19"

echo "========================================="
echo "Running Inference for logical-serenity-19"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Step: $CHECKPOINT_STEP"
echo "Test Data: $DATA_DIR"
echo "Save Dir: $SAVE_DIR"
echo "========================================="

# Create save directory
mkdir -p "$SAVE_DIR"

python run_inference.py \
    --stock=GOOG \
    --checkpoint_step=$CHECKPOINT_STEP \
    --test_split=0.1 \
    --data_dir="$DATA_DIR" \
    --ckpt_path="$CHECKPOINT_PATH" \
    --save_dir="$SAVE_DIR"

echo "========================================="
echo "Inference completed"
echo "========================================="
