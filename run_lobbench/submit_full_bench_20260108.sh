#!/bin/bash
#SBATCH --job-name=lob-bench-full
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/lob_bench_full_%j.out
#SBATCH --error=logs/lob_bench_full_%j.err
#SBATCH --partition=workq

# LOB Bench Full Pipeline: Inference → Scoring → Plotting
# After ESTrainer token_mode auto-detection fix

echo "=============================================="
echo " LOB Bench Full Pipeline"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4"
echo "Start time: $(date)"
echo "=============================================="

# Configuration
OUTPUT_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output/logical-serenity-19_20260108_071849"
CHECKPOINT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me"
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc"

echo "Output dir: ${OUTPUT_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data dir: ${DATA_DIR}"
echo ""

# Setup environment
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
mkdir -p logs

# Activate conda environment (lob for orbax 0.11.6 compatibility)
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lob

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "=============================================="
echo " Step 1: Inference"
echo "=============================================="
echo "Start: $(date)"

python run_inference.py \
    --stock=GOOG \
    --checkpoint_step=89410 \
    --data_dir="${DATA_DIR}" \
    --ckpt_path="${CHECKPOINT}" \
    --save_dir="${OUTPUT_DIR}"

INFERENCE_STATUS=$?
echo "Inference exit code: ${INFERENCE_STATUS}"
echo "End: $(date)"

if [ ${INFERENCE_STATUS} -ne 0 ]; then
    echo "ERROR: Inference failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo " Step 2: Scoring"
echo "=============================================="
echo "Start: $(date)"

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench

python run_bench.py \
    --data_dir="${OUTPUT_DIR}" \
    --save_dir="${OUTPUT_DIR}" \
    --model_name="s5" \
    --stock="GOOG" \
    --time_period="2023" \
    --uncond_only

SCORING_STATUS=$?
echo "Scoring exit code: ${SCORING_STATUS}"
echo "End: $(date)"

if [ ${SCORING_STATUS} -ne 0 ]; then
    echo "ERROR: Scoring failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo " Step 3: Plotting"
echo "=============================================="
echo "Start: $(date)"

PLOT_DIR="${OUTPUT_DIR}/plots_$(date +%Y%m%d)"

python run_plotting_matplotlib.py \
    --score_dir="${OUTPUT_DIR}/scores" \
    --plot_dir="${PLOT_DIR}"

PLOTTING_STATUS=$?
echo "Plotting exit code: ${PLOTTING_STATUS}"
echo "End: $(date)"

echo ""
echo "=============================================="
echo " Summary"
echo "=============================================="
echo "Inference: ${INFERENCE_STATUS}"
echo "Scoring: ${SCORING_STATUS}"
echo "Plotting: ${PLOTTING_STATUS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Plots directory: ${PLOT_DIR}"
echo "End time: $(date)"
echo "=============================================="
