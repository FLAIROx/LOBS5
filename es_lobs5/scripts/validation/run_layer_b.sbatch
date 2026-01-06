#!/bin/bash
#SBATCH --job-name=es_layer_b
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=2:00:00
#SBATCH --output=logs_validation/layer_b_%j.out
#SBATCH --error=logs_validation/layer_b_%j.err

# Layer B: Historical Replay 集成验证

echo "=============================================="
echo "ES-LOBS5 Layer B Validation (Historical Replay)"
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
echo "[B1] ESTrainer Initialization (Historical Replay)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/b1_trainer_init_hr.py
B1_STATUS=$?

if [ $B1_STATUS -ne 0 ]; then
    echo "❌ B1 FAILED with status $B1_STATUS"
    exit 1
fi

echo ""
echo "[B2] Single Epoch Training (8×10)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/b2_single_epoch_hr.py
B2_STATUS=$?

if [ $B2_STATUS -ne 0 ]; then
    echo "❌ B2 FAILED with status $B2_STATUS"
    exit 1
fi

echo ""
echo "[B3] Checkpoint Save"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/b3_checkpoint_save.py
B3_STATUS=$?

if [ $B3_STATUS -ne 0 ]; then
    echo "❌ B3 FAILED with status $B3_STATUS"
    exit 1
fi

echo ""
echo "[B4] Checkpoint Restore"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/b4_checkpoint_restore.py
B4_STATUS=$?

if [ $B4_STATUS -ne 0 ]; then
    echo "❌ B4 FAILED with status $B4_STATUS"
    exit 1
fi

echo ""
echo "[B5] Forward Pass Consistency"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/b5_forward_consistency.py
B5_STATUS=$?

if [ $B5_STATUS -ne 0 ]; then
    echo "❌ B5 FAILED with status $B5_STATUS"
    exit 1
fi

echo ""
echo "[B6] Medium Scale Training (32×50)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/b6_medium_scale_hr.py
B6_STATUS=$?

if [ $B6_STATUS -ne 0 ]; then
    echo "❌ B6 FAILED with status $B6_STATUS"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ Layer B: All Historical Replay tests PASSED"
echo "=============================================="
