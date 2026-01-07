#!/bin/bash
#SBATCH --job-name=verify-tokens
#SBATCH --output=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/logs/verify_tokens_%j.out
#SBATCH --error=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/logs/verify_tokens_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=workq
#SBATCH --gres=gpu:4

# Activate environment
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# PYTHONPATH
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5:$PYTHONPATH"

echo "=============================================="
echo " Token Verification Test"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=============================================="

python es_lobs5/verify_tokens.py

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
