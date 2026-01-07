#!/bin/bash
#SBATCH --job-name=order-analysis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/order_analysis_%j.out
#SBATCH --error=logs/order_analysis_%j.err
#SBATCH --partition=workq
#SBATCH --account=m100027

# Order Quality Analysis Script
# Compares historical vs policy-generated orders

echo "=============================================="
echo " Order Quality Analysis"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "=============================================="

# Setup environment
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Activate conda environment
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True"
export PYTHONDONTWRITEBYTECODE=1

# Create output directory
OUTPUT_DIR="./analysis_output"
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Run analysis
echo ""
echo "[*] Running order analysis..."
echo ""

python es_lobs5/test_order_analysis.py \
    --checkpoint "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/wandb/run-20241130_101652-logical-serenity-19/files/checkpoints" \
    --data_dir "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG/2021" \
    --n_historical 2000 \
    --n_episodes 10 \
    --n_steps 50 \
    --output_dir ${OUTPUT_DIR} \
    --token_mode 24

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
