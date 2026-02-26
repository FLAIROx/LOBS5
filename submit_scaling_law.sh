#!/bin/bash
# ============================================
# H1 Scaling Law — Batch Submission Script
# ============================================
# Submit 5 S5 models (10M → 120M) for scaling law experiments.
# Supports benchmark (CURTAIL=300) and train (15% data) modes.
# Jobs are chained sequentially via --dependency=afterok.
#
# Usage:
#   ./submit_scaling_law.sh benchmark              # 5 sequential benchmarks (30min each)
#   ./submit_scaling_law.sh train                   # 5 sequential training (24h each)
#   ./submit_scaling_law.sh single 10M benchmark    # single model benchmark
#   ./submit_scaling_law.sh single 10M train        # single model training
#
# Options:
#   NODES=32 ./submit_scaling_law.sh benchmark      # override node count (default 64)
#   DRY_RUN=1 ./submit_scaling_law.sh benchmark     # print commands without submitting

set -euo pipefail

# ── Configuration ──
NODES=${NODES:-64}
TOTAL_GPUS=$((NODES * 4))
TRAIN_SIZE=54264656   # 8 tickers, multi-ticker mode
DATA_FRACTION=0.15    # P = 15% of one epoch

WANDB_PROJECT="lobs5-scaling-law"
BATCH_SCRIPT="train_full_autoreg.batch"

# Benchmark settings
BENCH_CURTAIL=300
BENCH_TIME="00:30:00"

# Training settings
TRAIN_TIME="24:00:00"

# ── Model Configurations (block_size=64) ──
# Format: LABEL:D_MODEL:N_LAYERS:BLOCKS:SSM_SIZE:PER_GPU_BSZ
MODELS=(
    "10M:512:6:8:512:20"
    "22M:768:6:12:768:14"
    "55M:1024:12:16:1024:10"
    "85M:1280:12:20:1280:6"
    "120M:1536:12:24:1536:4"
)

# ── Helper Functions ──

parse_model() {
    local spec="$1"
    IFS=':' read -r LABEL D_MODEL N_LAYERS BLOCKS SSM_SIZE PER_GPU_BSZ <<< "$spec"
}

calc_curtail_15pct() {
    # CURTAIL = floor(TRAIN_SIZE * 0.15 / global_bsz) - 1
    local bsz=$1
    local global_bsz=$((bsz * TOTAL_GPUS))
    local steps_15pct=$(python3 -c "import math; print(math.floor($TRAIN_SIZE * $DATA_FRACTION / $global_bsz) - 1)")
    echo "$steps_15pct"
}

find_model() {
    local target="$1"
    for spec in "${MODELS[@]}"; do
        parse_model "$spec"
        if [[ "$LABEL" == "$target" ]]; then
            echo "$spec"
            return 0
        fi
    done
    echo "ERROR: Model '$target' not found. Available: 10M, 22M, 55M, 85M, 120M" >&2
    return 1
}

submit_job() {
    local label="$1"
    local d_model="$2"
    local n_layers="$3"
    local blocks="$4"
    local ssm_size="$5"
    local bsz="$6"
    local curtail="$7"
    local time_limit="$8"
    local dep_flag="${9:-}"

    local global_bsz=$((bsz * TOTAL_GPUS))

    # Informational output to stderr (so stdout is clean for job ID capture)
    echo "──────────────────────────────────────" >&2
    echo "Model: $label (d=$d_model, L=$n_layers, B=$blocks, SSM=$ssm_size)" >&2
    echo "BSZ: $bsz/GPU, global=$global_bsz, CURTAIL=$curtail" >&2
    echo "Nodes: $NODES, Time: $time_limit" >&2
    if [ -n "$dep_flag" ]; then
        echo "Dependency: $dep_flag" >&2
    fi

    local cmd="D_MODEL=$d_model N_LAYERS=$n_layers BLOCKS=$blocks SSM_SIZE_BASE=$ssm_size"
    cmd+=" PER_GPU_BSZ=$bsz CURTAIL_EPOCHS=$curtail"
    cmd+=" WANDB_PROJECT=$WANDB_PROJECT"
    cmd+=" NO_VALIDATION=1 NO_AUTO_RESUME=1"
    cmd+=" sbatch --contiguous --nodes=$NODES --time=$time_limit"
    if [ -n "$dep_flag" ]; then
        cmd+=" $dep_flag"
    fi
    cmd+=" $BATCH_SCRIPT"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] $cmd" >&2
        echo "DRY_${label}"  # stdout: fake job ID for chaining
        return 0
    fi

    echo ">> $cmd" >&2
    local output
    output=$(eval "$cmd" 2>&1)
    local job_id
    job_id=$(echo "$output" | grep -oP '\d+' | tail -1)

    if [ -z "$job_id" ]; then
        echo "ERROR: sbatch failed: $output" >&2
        return 1
    fi

    echo "Submitted: Job $job_id" >&2
    echo "Log: logs_lobs5/lobs5_${job_id}.out" >&2
    echo "$job_id"  # stdout: job ID for chaining
}

# ── Main Logic ──

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "Usage: $0 {benchmark|train|single <LABEL> {benchmark|train}}"
    echo ""
    echo "Models:"
    printf "  %-6s  d_model=%-5s  n_layers=%-3s  blocks=%-3s  ssm_size=%-5s  BSZ=%-3s\n" \
        "Label" "D" "L" "B" "SSM" "BSZ"
    echo "  ──────────────────────────────────────────────────────────────────"
    for spec in "${MODELS[@]}"; do
        parse_model "$spec"
        printf "  %-6s  d_model=%-5s  n_layers=%-3s  blocks=%-3s  ssm_size=%-5s  BSZ=%-3s\n" \
            "$LABEL" "$D_MODEL" "$N_LAYERS" "$BLOCKS" "$SSM_SIZE" "$PER_GPU_BSZ"
    done
    echo ""
    echo "Options:"
    echo "  NODES=32 $0 benchmark   # override node count (default 64)"
    echo "  DRY_RUN=1 $0 train      # preview commands without submitting"
    exit 1
fi

case "$MODE" in
    benchmark)
        echo "============================================"
        echo "H1 Scaling Law — Benchmark (CURTAIL=$BENCH_CURTAIL)"
        echo "Nodes: $NODES, Time: $BENCH_TIME per job"
        echo "============================================"
        echo ""

        PREV_JOB=""
        ALL_JOBS=()
        for spec in "${MODELS[@]}"; do
            parse_model "$spec"
            DEP=""
            if [ -n "$PREV_JOB" ]; then
                DEP="--dependency=afterok:$PREV_JOB"
            fi
            PREV_JOB=$(submit_job "$LABEL" "$D_MODEL" "$N_LAYERS" "$BLOCKS" "$SSM_SIZE" \
                       "$PER_GPU_BSZ" "$BENCH_CURTAIL" "$BENCH_TIME" "$DEP")
            ALL_JOBS+=("$LABEL:$PREV_JOB")
            echo ""
        done

        echo "============================================"
        echo "All benchmark jobs submitted:"
        for entry in "${ALL_JOBS[@]}"; do
            IFS=':' read -r label jid <<< "$entry"
            echo "  $label → Job $jid"
        done
        echo "============================================"
        ;;

    train)
        echo "============================================"
        echo "H1 Scaling Law — Training (P=${DATA_FRACTION}, 15% of 1 epoch)"
        echo "Nodes: $NODES, Time: $TRAIN_TIME per job"
        echo "============================================"
        echo ""

        PREV_JOB=""
        ALL_JOBS=()
        for spec in "${MODELS[@]}"; do
            parse_model "$spec"
            CURTAIL=$(calc_curtail_15pct "$PER_GPU_BSZ")
            DEP=""
            if [ -n "$PREV_JOB" ]; then
                DEP="--dependency=afterok:$PREV_JOB"
            fi
            PREV_JOB=$(submit_job "$LABEL" "$D_MODEL" "$N_LAYERS" "$BLOCKS" "$SSM_SIZE" \
                       "$PER_GPU_BSZ" "$CURTAIL" "$TRAIN_TIME" "$DEP")
            ALL_JOBS+=("$LABEL:$PREV_JOB:$CURTAIL")
            echo ""
        done

        echo "============================================"
        echo "All training jobs submitted:"
        for entry in "${ALL_JOBS[@]}"; do
            IFS=':' read -r label jid curtail <<< "$entry"
            echo "  $label → Job $jid (CURTAIL=$curtail)"
        done
        echo "============================================"
        ;;

    single)
        TARGET_LABEL="${2:-}"
        TARGET_MODE="${3:-benchmark}"
        if [ -z "$TARGET_LABEL" ]; then
            echo "Usage: $0 single <LABEL> {benchmark|train}"
            echo "Labels: 10M, 22M, 55M, 85M, 120M"
            exit 1
        fi

        SPEC=$(find_model "$TARGET_LABEL")
        parse_model "$SPEC"

        if [ "$TARGET_MODE" = "benchmark" ]; then
            CURTAIL=$BENCH_CURTAIL
            TIME=$BENCH_TIME
            echo "Single benchmark: $LABEL"
        elif [ "$TARGET_MODE" = "train" ]; then
            CURTAIL=$(calc_curtail_15pct "$PER_GPU_BSZ")
            TIME=$TRAIN_TIME
            echo "Single training: $LABEL (CURTAIL=$CURTAIL)"
        else
            echo "Unknown mode: $TARGET_MODE (use 'benchmark' or 'train')"
            exit 1
        fi

        JOB_ID=$(submit_job "$LABEL" "$D_MODEL" "$N_LAYERS" "$BLOCKS" "$SSM_SIZE" \
                 "$PER_GPU_BSZ" "$CURTAIL" "$TIME")
        echo ""
        echo "Job $JOB_ID submitted."
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {benchmark|train|single <LABEL> {benchmark|train}}"
        exit 1
        ;;
esac
