#!/bin/bash
# Submit jobs for PERGPU_PERTURBATIONS scaling test
# Values requested: 64 128 256 512 1024 2048 4096 8192

# Array of values to test
VALUES=(64 128 256 512 1024 2048 4096 8192)

echo "=================================================="
echo "Submitting PERGPU_PERTURBATIONS Benchmark Jobs"
echo "=================================================="

for VAL in "${VALUES[@]}"; do
    JOB_NAME="es_bench_${VAL}"
    
    # We use a short duration (N_EPOCHS=10) to quickly verify if it runs without OOM.
    # We assume OOM checks happen early (JIT compilation or first step).
    
    echo "Submitting job for PERGPU_PERTURBATIONS=$VAL..."
    
    # Submit using sbatch and capture the output
    # passing environment variables to override defaults and set job name
    OUTPUT=$(PERGPU_PERTURBATIONS=$VAL \
             N_EPOCHS=10 \
             WANDB_PROJECT="es-lobs5-benchmark" \
             sbatch --job-name="${JOB_NAME}" es_lobs5/scripts/es_training.sh)
             
    echo "  -> $OUTPUT"
    
    # Small delay between submissions to avoid slamming the scheduler?
    sleep 1
done

echo "=================================================="
echo "All jobs submitted. Monitor with 'squeue -u $USER'"
echo "Check logs/es_train_*.out for results."
echo "=================================================="
