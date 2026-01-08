#!/bin/bash
#SBATCH --job-name=token-debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=logs/token_debug_%j.out
#SBATCH --error=logs/token_debug_%j.err
#SBATCH --partition=workq

# Debug script to inspect raw policy token values

echo "=============================================="
echo " Token Debug - Inspecting Size Encoding"
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

python es_lobs5/test_policy_tokens.py

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
