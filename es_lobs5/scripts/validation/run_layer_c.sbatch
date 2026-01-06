#!/bin/bash
#SBATCH --job-name=es_layer_c
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=3:00:00
#SBATCH --output=logs_validation/layer_c_%j.out
#SBATCH --error=logs_validation/layer_c_%j.err

# Layer C: World Model 集成验证

echo "=============================================="
echo "ES-LOBS5 Layer C Validation (World Model)"
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
echo "[C1] ESTrainer Initialization (World Model)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/c1_trainer_init_wm.py
C1_STATUS=$?

if [ $C1_STATUS -ne 0 ]; then
    echo "❌ C1 FAILED with status $C1_STATUS"
    exit 1
fi

echo ""
echo "[C2] Single Epoch Training (World Model, 4×5)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/c2_single_epoch_wm.py
C2_STATUS=$?

if [ $C2_STATUS -ne 0 ]; then
    echo "❌ C2 FAILED with status $C2_STATUS"
    exit 1
fi

echo ""
echo "[C3] World Model Generation"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/c3_world_model_generation.py
C3_STATUS=$?

if [ $C3_STATUS -ne 0 ]; then
    echo "❌ C3 FAILED with status $C3_STATUS"
    exit 1
fi

echo ""
echo "[C4] Checkpoint Save/Restore (World Model)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/c4_checkpoint_wm.py
C4_STATUS=$?

if [ $C4_STATUS -ne 0 ]; then
    echo "❌ C4 FAILED with status $C4_STATUS"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ Layer C: All World Model tests PASSED"
echo "=============================================="
