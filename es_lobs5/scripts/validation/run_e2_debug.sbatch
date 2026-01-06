#!/bin/bash
#SBATCH --job-name=e2_debug
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=0:30:00
#SBATCH --output=logs_validation/e2_debug_%j.out
#SBATCH --error=logs_validation/e2_debug_%j.err

# E2 Import Diagnostic - identify where hang occurs

echo "=============================================="
echo "E2 Import Diagnostic"
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
echo "Running import diagnostic..."
echo "----------------------------------------------"

# Use unbuffered Python output
$PYTHON -u es_lobs5/scripts/validation/e2_debug_imports.py

echo ""
echo "=============================================="
echo "Diagnostic complete"
echo "=============================================="
