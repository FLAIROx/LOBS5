#!/bin/bash
#SBATCH --job-name=es_layer_a
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=0:30:00
#SBATCH --output=logs_validation/layer_a_%j.out
#SBATCH --error=logs_validation/layer_a_%j.err

# ES-LOBS5 Layer A Validation Script
# 必须在 GPU 上运行

set -e

# Setup
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/bin/activate es-lob
mkdir -p logs_validation

echo "=============================================="
echo "ES-LOBS5 Layer A Validation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Run A1-A6 sequentially
echo ""
echo "[A1] Checkpoint Load Validation"
echo "----------------------------------------------"
python es_lobs5/scripts/validation/a1_checkpoint_load.py

echo ""
echo "[A2] Noiser Initialization Validation"
echo "----------------------------------------------"
python es_lobs5/scripts/validation/a2_noiser_init.py

echo ""
echo "[A3] CommonParams Build Validation"
echo "----------------------------------------------"
python es_lobs5/scripts/validation/a3_common_params.py

echo ""
echo "[A4] Forward Pass (Inference) Validation"
echo "----------------------------------------------"
python es_lobs5/scripts/validation/a4_forward_pass.py

echo ""
echo "[A5] Forward Pass (ES Noise) Validation"
echo "----------------------------------------------"
python es_lobs5/scripts/validation/a5_forward_es_noise.py

echo ""
echo "[A6] Historical Replay Data Load Validation"
echo "----------------------------------------------"
python es_lobs5/scripts/validation/a6_data_load.py

echo ""
echo "=============================================="
echo "Layer A Validation Complete!"
echo "=============================================="
