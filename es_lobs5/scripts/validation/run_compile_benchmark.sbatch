#!/bin/bash
#SBATCH --job-name=es_compile_benchmark
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=logs_validation/compile_benchmark_%j.out
#SBATCH --error=logs_validation/compile_benchmark_%j.err

echo "=============================================="
echo "ES-LOBS5 Compilation Optimization Benchmarks"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Activate environment
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/bin/activate es-lob
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
mkdir -p logs_validation

export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
PYTHON="/lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/envs/es-lob/bin/python -u"

# Function to clear GPU memory between tests
clear_gpu_memory() {
    echo "[INFO] Clearing GPU memory..."
    $PYTHON -c "import jax; jax.clear_caches(); import gc; gc.collect()"
    sleep 2
}

# Show cache directory
CACHE_DIR="$HOME/.cache/es_lobs5_jax_compilation"
echo ""
echo "[INFO] JAX compilation cache: $CACHE_DIR"
if [ -d "$CACHE_DIR" ]; then
    echo "[INFO] Cache exists with $(find $CACHE_DIR -type f | wc -l) files"
    echo "[INFO] Cache size: $(du -sh $CACHE_DIR 2>/dev/null | cut -f1)"
else
    echo "[INFO] Cache directory does not exist yet"
fi
echo ""

# ==============================================================================
# I1: Compilation Time Benchmark
# ==============================================================================
echo ""
echo "[I1] Compilation Time Benchmark"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/i1_compilation_time.py
I1_STATUS=$?
echo "[I1] Exit status: $I1_STATUS"
clear_gpu_memory

# ==============================================================================
# I2: Recompilation Detection
# ==============================================================================
echo ""
echo "[I2] Recompilation Detection"
echo "----------------------------------------------"
# Run with JAX_LOG_COMPILES to see compilation logs
JAX_LOG_COMPILES=1 $PYTHON es_lobs5/scripts/validation/i2_recompilation_detection.py 2>&1
I2_STATUS=$?
echo "[I2] Exit status: $I2_STATUS"
clear_gpu_memory

# ==============================================================================
# I3: Persistent Cache Test
# ==============================================================================
echo ""
echo "[I3] Persistent Cache Test"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/i3_persistent_cache.py
I3_STATUS=$?
echo "[I3] Exit status: $I3_STATUS"
clear_gpu_memory

# ==============================================================================
# I4: Multi-GPU Utilization
# ==============================================================================
echo ""
echo "[I4] Multi-GPU Utilization Test"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/i4_multi_gpu_utilization.py
I4_STATUS=$?
echo "[I4] Exit status: $I4_STATUS"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================="
echo "Compilation Optimization Benchmark Summary"
echo "=============================================="
echo "  I1 (Compilation Time):  $([ $I1_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  I2 (Recompilation):     $([ $I2_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  I3 (Persistent Cache):  $([ $I3_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  I4 (Multi-GPU):         $([ $I4_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "=============================================="
echo ""

# Show final cache state
echo "[INFO] Final cache state:"
if [ -d "$CACHE_DIR" ]; then
    echo "[INFO] Cache files: $(find $CACHE_DIR -type f | wc -l)"
    echo "[INFO] Cache size: $(du -sh $CACHE_DIR 2>/dev/null | cut -f1)"
fi

# Exit with combined status
TOTAL_FAILURES=$((I1_STATUS + I2_STATUS + I3_STATUS + I4_STATUS))
if [ $TOTAL_FAILURES -eq 0 ]; then
    echo ""
    echo "ALL COMPILATION BENCHMARKS PASSED"
    exit 0
else
    echo ""
    echo "SOME BENCHMARKS FAILED ($TOTAL_FAILURES failures)"
    exit 1
fi
