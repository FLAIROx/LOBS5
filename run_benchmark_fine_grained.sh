#!/bin/bash
# Submit fine-grained benchmark jobs to find max PERGPU_PERTURBATIONS between 64 and 128
# Theoretical Max (80GB / 0.72GB per copy) ~= 110
# Values to test: 80 96 100 110

VALUES=(80 96 100 110)

echo "=================================================="
echo "Submitting Fine-Grained PERGPU_PERTURBATIONS Jobs"
echo "=================================================="

for VAL in "${VALUES[@]}"; do
    JOB_NAME="es_bench_${VAL}"
    
    echo "Submitting job for PERGPU_PERTURBATIONS=$VAL..."
    
    OUTPUT=$(PERGPU_PERTURBATIONS=$VAL \
             N_EPOCHS=10 \
             WANDB_PROJECT="es-lobs5-benchmark" \
             sbatch --job-name="${JOB_NAME}" es_lobs5/scripts/es_training.sh)
             
    echo "  -> $OUTPUT"
    sleep 1
done
