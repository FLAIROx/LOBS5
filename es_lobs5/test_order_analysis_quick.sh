#!/bin/bash
#SBATCH --job-name=order-quick
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/order_quick_%j.out
#SBATCH --error=logs/order_quick_%j.err
#SBATCH --partition=workq

# Quick Order Analysis - Fast validation of ESTrainer token_mode fix
# Parameters: warmup=50, n_steps=10, episodes=3

echo "=============================================="
echo " Quick Order Analysis - Token Mode Fix Test"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "=============================================="

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p logs

CHECKPOINT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc"

python es_lobs5/test_order_analysis.py \
    --checkpoint "${CHECKPOINT}" \
    --data_dir "${DATA_DIR}" \
    --n_historical 500 \
    --n_episodes 3 \
    --n_steps 10 \
    --n_warmup_msgs 50 \
    --background_msgs_per_step 10 \
    --output_dir ./analysis_output \
    --token_mode 24

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
