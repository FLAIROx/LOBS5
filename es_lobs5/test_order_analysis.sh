#!/bin/bash
#SBATCH --job-name=order-analysis
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/order_analysis_%j.out
#SBATCH --error=logs/order_analysis_%j.err
#SBATCH --partition=workq

# Order Quality Analysis Script - Real ES Training Version
# Extracts real policy orders from ES training for analysis

echo "=============================================="
echo " Order Quality Analysis - Real ES Training"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4"
echo "Start time: $(date)"
echo "=============================================="

# Setup environment
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Activate conda environment
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Create output directory
OUTPUT_DIR="./analysis_output"
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Default: Run with real ES training (not synthetic)
# Add --use_synthetic flag for quick testing without ES

echo ""
echo "[*] Running order analysis with REAL ES training..."
echo "[*] This will initialize the ESTrainer and run episodes to collect policy orders"
echo ""

python es_lobs5/test_order_analysis.py \
    --checkpoint "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/wandb/run-20241130_101652-logical-serenity-19/files/checkpoints" \
    --data_dir "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG/2021" \
    --n_historical 2000 \
    --n_episodes 5 \
    --n_steps 50 \
    --output_dir ${OUTPUT_DIR} \
    --token_mode 24

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
