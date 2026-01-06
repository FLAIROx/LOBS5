#!/bin/bash
#SBATCH --job-name=es_layer_d
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=4:00:00
#SBATCH --output=logs_validation/layer_d_%j.out
#SBATCH --error=logs_validation/layer_d_%j.err

# Layer D: End-to-End Validation (Historical Replay ONLY)

echo "=============================================="
echo "ES-LOBS5 Layer D Validation (End-to-End)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Activate conda environment
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/bin/activate es-lob

# Set working directory
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Create logs directory
mkdir -p logs_validation

# Add JaxMARL-HFT (gymnax_exchange) to PYTHONPATH
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Python path
PYTHON=/lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/envs/es-lob/bin/python

echo ""
echo "[D1] Full Training Loop (5 epochs, checkpoint every 2)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/d1_full_training_loop.py
D1_STATUS=$?

if [ $D1_STATUS -ne 0 ]; then
    echo "❌ D1 FAILED with status $D1_STATUS"
    exit 1
fi

echo ""
echo "[D2] Training Resumption (checkpoint restore + continue)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/d2_training_resumption.py
D2_STATUS=$?

if [ $D2_STATUS -ne 0 ]; then
    echo "❌ D2 FAILED with status $D2_STATUS"
    exit 1
fi

echo ""
echo "[D3] Metrics Validation (10 epochs, detailed metrics)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/d3_metrics_validation.py
D3_STATUS=$?

if [ $D3_STATUS -ne 0 ]; then
    echo "❌ D3 FAILED with status $D3_STATUS"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ Layer D: All End-to-End tests PASSED"
echo "=============================================="
