#!/bin/bash
# =============================================================================
# Sweep: PERGPU_PERTURBATIONS Exploration
# =============================================================================
# Explores memory/performance tradeoffs for different per-GPU perturbation counts.
#
# Usage:
#   ./es_lobs5/scripts/sweep_pergpu.sh              # Submit all jobs
#   ./es_lobs5/scripts/sweep_pergpu.sh --dry-run    # Preview without submitting
#
# Previous Results (2026-01-18):
#   LORA_V2 Mode:
#     64 per-GPU (256 total)  -> COMPLETED
#     128 per-GPU (512 total) -> OOM (71GB alloc)
#
#   LORA (frozen) Mode:
#     14,336 per-GPU (57,344 total) -> Max stable
#     16,384 per-GPU (65,536 total) -> OOM
# =============================================================================

set -e
cd /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5

# Configuration
MODE="${MODE:-LORA_V1.5}"
N_EPOCHS=5                    # Short runs for sweep exploration
CHECKPOINT_EVERY=10           # Don't checkpoint during sweep

# PERGPU values to test (will be multiplied by 4 GPUs for total population)
# Testing powers of 2 and some intermediate values
PERGPU_VALUES=(32 48 64 80 96 112 128)

echo "=============================================="
echo " PERGPU_PERTURBATIONS Sweep"
echo "=============================================="
echo "MODE: ${MODE}"
echo "N_EPOCHS: ${N_EPOCHS}"
echo "PERGPU values: ${PERGPU_VALUES[*]}"
echo "=============================================="
echo ""

# Check for dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN MODE] - Jobs will NOT be submitted"
    echo ""
fi

# Submit jobs
declare -a JOB_IDS
for pergpu in "${PERGPU_VALUES[@]}"; do
    total=$((pergpu * 4))
    job_name="sweep_pergpu${pergpu}_${MODE}"

    echo "----------------------------------------"
    echo "PERGPU_PERTURBATIONS: ${pergpu}"
    echo "Total Population: ${total}"
    echo "Job Name: ${job_name}"

    if $DRY_RUN; then
        echo "[DRY-RUN] Would submit: MODE=${MODE} PERGPU_PERTURBATIONS=${pergpu} N_EPOCHS=${N_EPOCHS}"
    else
        # Submit job and capture job ID
        JOB_ID=$(MODE=${MODE} \
                 PERGPU_PERTURBATIONS=${pergpu} \
                 N_EPOCHS=${N_EPOCHS} \
                 CHECKPOINT_EVERY=${CHECKPOINT_EVERY} \
                 sbatch --job-name=${job_name} \
                        --parsable \
                        es_lobs5/scripts/es_training.sh)

        JOB_IDS+=("${JOB_ID}")
        echo "Submitted: Job ID ${JOB_ID}"
    fi
done

echo ""
echo "=============================================="
echo " Sweep Summary"
echo "=============================================="
echo "MODE: ${MODE}"
echo "Total jobs: ${#PERGPU_VALUES[@]}"

if ! $DRY_RUN; then
    echo ""
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Check logs:"
    for i in "${!JOB_IDS[@]}"; do
        echo "  tail -f logs/es_train_${JOB_IDS[$i]}.out  # pergpu=${PERGPU_VALUES[$i]}"
    done
fi

echo "=============================================="
