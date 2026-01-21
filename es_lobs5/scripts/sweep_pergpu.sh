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
#
# Results (2026-01-20, LORA_V1.5 with dots_with_no_batch_dims_saveable):
#   160 per-GPU (640 total) -> COMPLETED (confirmed working)
#
# Results (2026-01-21, LORA_V1.5 exponential scaling test):
#   Checkpoint policy: dots_with_no_batch_dims_saveable (commit 2a9e899)
#   256 per-GPU (1,024 total)  -> COMPLETED (0.19h) Job 1950636
#   512 per-GPU (2,048 total)  -> COMPLETED (0.20h) Job 1950637 ★ NEW MAX STABLE
#   1024 per-GPU (4,096 total) -> OOM (~17.7GB alloc) Job 1950638
#   2048+ per-GPU              -> OOM (scales linearly) Jobs 1950639-1950642
#
# Summary: 4x improvement over default remat (128 -> 512 per-GPU)
# Commits: 9ec0d2d (sweep script), 987a042 (results docs)
# =============================================================================

set -e
cd /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5

# Configuration
MODE="${MODE:-LORA_V1.5}"
N_EPOCHS=3                    # Short runs for sweep exploration
CHECKPOINT_EVERY=10           # Don't checkpoint during sweep

# PERGPU values to test (will be multiplied by 4 GPUs for total population)
# Exponential scaling: 2x, 4x, 8x, 16x, 32x, 64x baseline, + historical max
# | PERGPU | Total   | Notes           |
# |--------|---------|-----------------|
# | 256    | 1,024   | 2x baseline     |
# | 512    | 2,048   | 4x baseline     |
# | 1024   | 4,096   | 8x baseline     |
# | 2048   | 8,192   | 16x baseline    |
# | 4096   | 16,384  | 32x baseline    |
# | 8192   | 32,768  | 64x baseline    |
# | 14336  | 57,344  | Historical max  |
PERGPU_VALUES=(256 512 1024 2048 4096 8192 14336)

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
