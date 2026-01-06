#!/bin/bash
#SBATCH --job-name=es_e2_only
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=2:00:00
#SBATCH --output=logs_validation/e2_only_%j.out
#SBATCH --error=logs_validation/e2_only_%j.err

# E2 Only: Multi-GPU Utilization Test (standalone)

echo "=============================================="
echo "E2 Only: Multi-GPU Utilization Test"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/bin/activate es-lob
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
mkdir -p logs_validation

export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
PYTHON=/lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/envs/es-lob/bin/python

echo ""
echo "[E2] Multi-GPU Utilization (n_threads=256, n_steps=10, n_epochs=50)"
echo "----------------------------------------------"
# Use -u for unbuffered output to see progress in real-time
$PYTHON -u es_lobs5/scripts/validation/e2_multi_gpu_utilization.py
E2_STATUS=$?

echo ""
echo "=============================================="
echo "E2 Result: $([ $E2_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "=============================================="
