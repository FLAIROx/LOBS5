#!/bin/bash
###############################################################################
# Full LOB Benchmark Pipeline: Inference → Scoring → Plotting
# For logical-serenity-19 checkpoint
###############################################################################

#SBATCH --job-name=full_lobbench
#SBATCH --output=logs_lobs5/full_lobbench_%j.out
#SBATCH --error=logs_lobs5/full_lobbench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4

###############################################################################
# Configuration
###############################################################################
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Paths
CHECKPOINT_PATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me"
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc"
OUTPUT_BASE="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output"
OUTPUT_DIR="${OUTPUT_BASE}/logical-serenity-19_${TIMESTAMP}"

# Model config
STOCK="GOOG"
TIME_PERIOD="2023"
MODEL_NAME="s5"
CHECKPOINT_STEP=89410

echo "============================================================"
echo " Full LOB Benchmark Pipeline"
echo "============================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

###############################################################################
# Environment Setup
###############################################################################
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lob

export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128

# Create directories
mkdir -p logs_lobs5
mkdir -p "${OUTPUT_DIR}"

###############################################################################
# STEP 1: INFERENCE
###############################################################################
echo ""
echo "[STEP 1/3] Running Inference..."
echo "============================================================"

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Run inference - saves to data_real/, data_gen/, data_cond/
python run_inference.py \
    --stock=${STOCK} \
    --checkpoint_step=${CHECKPOINT_STEP} \
    --test_split=0.1 \
    --data_dir="${DATA_DIR}" \
    --ckpt_path="${CHECKPOINT_PATH}" \
    --save_dir="${OUTPUT_DIR}"

# Check if inference succeeded
if [ $? -ne 0 ]; then
    echo "[ERROR] Inference failed!"
    exit 1
fi

# Reorganize data for lob_bench expected structure:
# {DATA_DIR}/{MODEL}/{STOCK}/{TIME_PERIOD}/data_*/
echo "[*] Reorganizing data for lob_bench..."
BENCH_DATA_DIR="${OUTPUT_DIR}/${MODEL_NAME}/${STOCK}/${TIME_PERIOD}"
mkdir -p "${BENCH_DATA_DIR}"
mv "${OUTPUT_DIR}/data_real" "${BENCH_DATA_DIR}/"
mv "${OUTPUT_DIR}/data_gen" "${BENCH_DATA_DIR}/"
mv "${OUTPUT_DIR}/data_cond" "${BENCH_DATA_DIR}/"

echo "[*] Data organized at: ${BENCH_DATA_DIR}"
ls -la "${BENCH_DATA_DIR}/"

###############################################################################
# STEP 2: SCORING
###############################################################################
echo ""
echo "[STEP 2/3] Running Scoring..."
echo "============================================================"

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench

# Run scoring - all metrics, saves to flat scores/ directory
python run_bench.py \
    --data_dir="${OUTPUT_DIR}" \
    --save_dir="${OUTPUT_DIR}" \
    --model_name="${MODEL_NAME}" \
    --stock="${STOCK}" \
    --time_period="${TIME_PERIOD}" \
    --uncond_only

# Check if scoring succeeded
if [ $? -ne 0 ]; then
    echo "[ERROR] Scoring failed!"
    exit 1
fi

echo "[*] Scoring completed. Files:"
ls -la "${OUTPUT_DIR}/scores/"

###############################################################################
# STEP 3: PLOTTING
###############################################################################
echo ""
echo "[STEP 3/3] Generating Plots..."
echo "============================================================"

PLOTS_DIR="${OUTPUT_DIR}/plots_${TIMESTAMP}"
mkdir -p "${PLOTS_DIR}"

python run_plotting.py \
    --score_dir="${OUTPUT_DIR}/scores" \
    --plot_dir="${PLOTS_DIR}" \
    --model_name="${MODEL_NAME}"

# Check if plotting succeeded
if [ $? -ne 0 ]; then
    echo "[ERROR] Plotting failed!"
    exit 1
fi

echo "[*] Plots generated at: ${PLOTS_DIR}"
ls -la "${PLOTS_DIR}/"

###############################################################################
# DONE
###############################################################################
echo ""
echo "============================================================"
echo " BENCHMARK COMPLETE"
echo "============================================================"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""
echo "Contents:"
echo "  - Data: ${BENCH_DATA_DIR}"
echo "  - Scores: ${OUTPUT_DIR}/scores/"
echo "  - Plots: ${PLOTS_DIR}/"
echo ""
echo "Generated plots:"
ls -1 "${PLOTS_DIR}/"
echo "============================================================"
