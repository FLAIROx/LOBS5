#!/bin/bash
# =============================================================================
# submit_with_priority_chain.sh — Pre-submit dependency chain for priority
# =============================================================================
#
# Problem: When a long-running job fails and auto-resume submits a NEW job,
# the new job has 0 accumulated age priority. Any large pending job that has
# been waiting will outrank it.
#
# Solution: Pre-submit N backup jobs with --dependency=afternotok:PREV at the
# same time as the main job. All jobs accumulate age priority from submission
# time. When the main job fails, the backup is already high-priority.
#
# Usage:
#   ./submit_with_priority_chain.sh [sbatch args...] train_full_autoreg.batch
#
# Examples:
#   LOCAL_STEPS_K=10 MSG_SEQ_LEN=4000 PER_GPU_BSZ=1 \
#     ./submit_with_priority_chain.sh --job-name=ctx4k-16n-12h --nodes=16 --time=12:00:00 train_full_autoreg.batch
#
#   # Custom chain length (default 3):
#   CHAIN_LENGTH=5 ./submit_with_priority_chain.sh --nodes=32 --time=24:00:00 train_full_autoreg.batch
#
# How it works:
#   1. Submits the main job normally
#   2. Submits CHAIN_LENGTH backup jobs, each with --dependency=afternotok:PREV
#   3. Backup jobs start accumulating priority immediately
#   4. If main job succeeds → backups auto-cancelled (afternotok not satisfied)
#   5. If main job fails → first backup released with full accumulated priority
#   6. Each backup discovers latest checkpoint at runtime (auto-resume logic)
#
# NOTE: This is complementary to the in-script auto-resume. The in-script
# auto-resume still works as fallback, but the dependency chain ensures
# priority preservation. Set NO_AUTO_RESUME=1 in the main script to disable
# the in-script auto-resume when using this wrapper.
# =============================================================================

set -euo pipefail

CHAIN_LENGTH=${CHAIN_LENGTH:-3}

# Submit the main job
MAIN_ID=$(sbatch --parsable "$@")
echo "[Chain] Main job: $MAIN_ID"

# Extract batch script path (last argument) and sbatch args (everything else)
BATCH_SCRIPT="${@: -1}"
SBATCH_ARGS="${@:1:$#-1}"

# Build dependency chain
PREV_ID=$MAIN_ID
for i in $(seq 1 $CHAIN_LENGTH); do
    # Each backup depends on the previous one failing (afternotok)
    # If previous succeeds, this job is automatically cancelled by SLURM
    BACKUP_ID=$(sbatch --parsable --dependency=afternotok:${PREV_ID} $SBATCH_ARGS "$BATCH_SCRIPT")
    echo "[Chain] Backup $i/$CHAIN_LENGTH: $BACKUP_ID (afternotok:$PREV_ID)"
    PREV_ID=$BACKUP_ID
done

echo ""
echo "[Chain] Submitted 1 main + $CHAIN_LENGTH backups. All accumulating priority now."
echo "[Chain] If main fails → backup 1 starts immediately with full priority."
echo "[Chain] If main succeeds → all backups auto-cancelled."
echo ""
squeue -u $(whoami) --sort=i -o "  %10i %15j %2t %10M %6D %R"
