#!/bin/bash
#SBATCH --job-name=es_layer_e
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --output=logs_validation/layer_e_%j.out
#SBATCH --error=logs_validation/layer_e_%j.err

# Layer E: Production Scale Validation (Historical Replay ONLY)
#
# Test order optimized to avoid GPU memory issues:
# E4 (quick inference) → E2 (multi-GPU) → E1 (high scale) → E3 (production) → E5 (OOM stress LAST)
# E5 runs LAST because it tests OOM conditions which can corrupt GPU state

echo "=============================================="
echo "ES-LOBS5 Layer E Validation (Production Scale)"
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

# Helper function to clear GPU memory between tests
clear_gpu_memory() {
    echo "  [Clearing GPU memory...]"
    $PYTHON -c "
import jax
import gc
try:
    jax.clear_caches()
except:
    pass
gc.collect()
print('  GPU memory cleared')
"
}

# ============================================
# E4: Inference Throughput (quick test first)
# ============================================
echo ""
echo "[E4] Inference Throughput (quick test first)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/e4_inference_throughput.py
E4_STATUS=$?
if [ $E4_STATUS -ne 0 ]; then
    echo "❌ E4 FAILED with status $E4_STATUS"
fi
clear_gpu_memory

# ============================================
# E2: Multi-GPU Utilization (before OOM tests)
# ============================================
echo ""
echo "[E2] Multi-GPU Utilization"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/e2_multi_gpu_utilization.py
E2_STATUS=$?
if [ $E2_STATUS -ne 0 ]; then
    echo "❌ E2 FAILED with status $E2_STATUS"
fi
clear_gpu_memory

# ============================================
# E1: High Scale Training (128x100x50)
# ============================================
echo ""
echo "[E1] High Scale Training (128x100x50)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/e1_high_scale_training.py
E1_STATUS=$?
if [ $E1_STATUS -ne 0 ]; then
    echo "❌ E1 FAILED with status $E1_STATUS"
fi
clear_gpu_memory

# ============================================
# E3: Production Run (128x100x100 + wandb)
# ============================================
echo ""
echo "[E3] Production Run (128x100x100 + wandb)"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/e3_production_run.py
E3_STATUS=$?
if [ $E3_STATUS -ne 0 ]; then
    echo "❌ E3 FAILED with status $E3_STATUS"
fi
clear_gpu_memory

# ============================================
# E5: Max Perturbations Scaling (OOM stress test - RUN LAST)
# ============================================
echo ""
echo "[E5] Max Perturbations Scaling (OOM stress test)"
echo "----------------------------------------------"
echo "  WARNING: This test intentionally pushes to OOM to find limits"
$PYTHON es_lobs5/scripts/validation/e5_max_perturbations.py
E5_STATUS=$?
if [ $E5_STATUS -ne 0 ]; then
    echo "❌ E5 FAILED with status $E5_STATUS"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Layer E Validation Summary:"
echo "  E1 (High Scale):     $([ $E1_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "  E2 (Multi-GPU):      $([ $E2_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "  E3 (Production):     $([ $E3_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "  E4 (Throughput):     $([ $E4_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "  E5 (Max Perturb):    $([ $E5_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "=============================================="
