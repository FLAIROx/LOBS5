#!/bin/bash
#SBATCH --job-name=es-benchmark-pert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/benchmark_perturbations_%j.out
#SBATCH --error=logs/benchmark_perturbations_%j.err
#SBATCH --partition=gh200

# ==============================================================================
# Benchmark: Find Maximum n_perturbations for Different n_steps
# ==============================================================================
#
# Usage:
#   # Test default (n_steps=10,100, threads 64-8192)
#   sbatch run_benchmark_perturbations.sh
#
#   # Test specific n_steps
#   N_STEPS="10 50 100" sbatch run_benchmark_perturbations.sh
#
#   # Custom thread range
#   MIN_THREADS=128 MAX_THREADS=4096 sbatch run_benchmark_perturbations.sh
#
# ==============================================================================

set -e

echo "============================================================"
echo "ES-LOBS5 Max Perturbations Benchmark"
echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "------------------------------------------------------------"

mkdir -p logs

# Activate environment
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# XLA flags
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True"
export JAX_TRACEBACK_FILTERING=off

# Show GPU info
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Configuration
N_STEPS="${N_STEPS:-10 100}"
MIN_THREADS="${MIN_THREADS:-64}"
MAX_THREADS="${MAX_THREADS:-8192}"
STEP_MULT="${STEP_MULT:-2.0}"
OUTPUT="benchmark_perturbations_${SLURM_JOB_ID}.json"

echo "Configuration:"
echo "  N_STEPS:     $N_STEPS"
echo "  MIN_THREADS: $MIN_THREADS"
echo "  MAX_THREADS: $MAX_THREADS"
echo "  STEP_MULT:   $STEP_MULT"
echo "  OUTPUT:      $OUTPUT"
echo "============================================================"

# Run benchmark
python benchmark_max_perturbations.py \
    --n_steps $N_STEPS \
    --min_threads $MIN_THREADS \
    --max_threads $MAX_THREADS \
    --step_multiplier $STEP_MULT \
    --output $OUTPUT

echo ""
echo "============================================================"
echo "Benchmark Complete!"
echo "End time: $(date)"
echo "============================================================"
