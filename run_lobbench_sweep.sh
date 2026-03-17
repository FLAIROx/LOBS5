#!/bin/bash
# LOBbench sweep: evaluate Muon vs AdamW checkpoints at ~10k step intervals
# Submits 17 independent SLURM jobs via the lob_pipeline.
#
# Usage: bash run_lobbench_sweep.sh

set -euo pipefail

PIPELINE="/projects/s5e/lob_pipeline/pipeline/run_lobbench_pipeline.sh"
CKPT_BASE="/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_J2_muon_optimizer/checkpoints"

# Format: "name:checkpoint_dir:step"
RUNS=(
    # Muon checkpoints (9 points)
    "muon-s10190:j2696686_d8d071ln_2696686:10190"
    "muon-s20496:j2696686_d8d071ln_2696686:20496"
    "muon-s32155:j2716524_lqs66551_2716524:32155"
    "muon-s40980:j2716524_lqs66551_2716524:40980"
    "muon-s49680:j2727131_d6gdaz35_2727131:49680"
    "muon-s63568:j2782925_4icb4jua_2782925:63568"
    "muon-s70903:j2782925_4icb4jua_2782925:70903"
    "muon-s80013:j2815248_4tkbytct_2815248:80013"
    "muon-s84432:j2815248_4tkbytct_2815248:84432"
    # AdamW checkpoints (8 points — no 30k due to checkpoint gap)
    "adamw-s9914:j2689747_ridsve7i_2689747:9914"
    "adamw-s19969:j2689747_ridsve7i_2689747:19969"
    "adamw-s39275:j2715665_ueew9sq2_2715665:39275"
    "adamw-s49871:j2715665_ueew9sq2_2715665:49871"
    "adamw-s60462:j2715665_ueew9sq2_2715665:60462"
    "adamw-s71307:j2782661_upsz4ijw_2782661:71307"
    "adamw-s80311:j2782661_upsz4ijw_2782661:80311"
    "adamw-s84249:j2782661_upsz4ijw_2782661:84249"
)

echo "=============================================="
echo "LOBbench Sweep: Muon vs AdamW"
echo "=============================================="
echo "Checkpoints: ${#RUNS[@]}"
echo "Pipeline:    ${PIPELINE}"
echo "Settings:    GOOG Jan 2026, L100, 1024 seq, skip_extended"
echo "Nodes/run:   1 (single-node scoring)"
echo "=============================================="
echo ""

SUBMITTED=0
FAILED=0

for ENTRY in "${RUNS[@]}"; do
    IFS=':' read -r NAME DIR STEP <<< "$ENTRY"
    CKPT_PATH="${CKPT_BASE}/${DIR}"

    if [ ! -d "${CKPT_PATH}/${STEP}" ]; then
        echo "SKIP: ${NAME} — checkpoint ${CKPT_PATH}/${STEP} not found"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "Submitting: ${NAME} (step ${STEP})..."
    $PIPELINE "$CKPT_PATH" \
        --name "$NAME" \
        --checkpoint_step "$STEP" \
        --stocks "GOOG" \
        --total_nodes 1 \
        --walltime "02:00:00"

    SUBMITTED=$((SUBMITTED + 1))
    echo ""
done

echo "=============================================="
echo "Sweep submitted: ${SUBMITTED}/${#RUNS[@]} jobs"
if [ "$FAILED" -gt 0 ]; then
    echo "Skipped: ${FAILED} (missing checkpoints)"
fi
echo ""
echo "Monitor: squeue --me | grep bench"
echo "Results: /projects/s5e/lob_pipeline/results_muon-s*/"
echo "         /projects/s5e/lob_pipeline/results_adamw-s*/"
echo "=============================================="
