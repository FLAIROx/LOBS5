#!/bin/bash
#SBATCH --job-name=es_bench_4gpu
#SBATCH --output=logs_benchmark/bench_4gpu_%j.out
#SBATCH --error=logs_benchmark/bench_4gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=8:00:00

# =============================================================================
# ES LOBS5 Multi-GPU (4x) Benchmark Script
# Purpose: Test parameter limits with 4 GPUs using shard_map
# Note: n_threads must be divisible by 4 (n_devices)
# =============================================================================

# Configuration (override via environment)
TOKEN_MODE=${TOKEN_MODE:-24}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/amber-music-24_09502dqc}
WANDB_PROJECT=${WANDB_PROJECT:-es_lobs5_benchmark_4gpu}
WANDB_ENTITY=${WANDB_ENTITY:-}
REPLAY_DATA_PATH=${REPLAY_DATA_PATH:-/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022}

# Test mode: n_threads (default), n_steps, lora_rank
TEST_MODE=${TEST_MODE:-n_threads}

# Fixed parameters for each test mode
# Note: n_threads must be divisible by 4 for multi-GPU
N_THREADS_DEFAULT=128
N_STEPS_DEFAULT=100
LORA_RANK_DEFAULT=4
N_EPOCHS=5

# Create directories
mkdir -p logs_benchmark

# Setup environment
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# gymnax_exchange module path
# Location: /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:${PYTHONPATH}"

echo "========================================"
echo "ES LOBS5 Multi-GPU Benchmark (4x GPU)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Token Mode: ${TOKEN_MODE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "W&B Project: ${WANDB_PROJECT}"
echo "Test Mode: ${TEST_MODE}"
echo "========================================"

# Function to run single test
run_test() {
    local N_THREADS=$1
    local N_STEPS=$2
    local LORA_RANK=$3
    local TEST_NAME=$4

    echo ""
    echo "=== Test: ${TEST_NAME} ==="
    echo "  n_threads=${N_THREADS}, n_steps=${N_STEPS}, lora_rank=${LORA_RANK}"
    echo "  n_threads/4 = $((N_THREADS/4)) per GPU"

    # Record start time
    START_TIME=$(date +%s)

    python -u -B -m es_lobs5.run_es_train \
        --lobs5_checkpoint="${CHECKPOINT_PATH}" \
        --noiser=eggroll \
        --sigma=0.01 \
        --lr=0.001 \
        --lora_rank=${LORA_RANK} \
        --n_threads=${N_THREADS} \
        --n_epochs=${N_EPOCHS} \
        --n_steps=${N_STEPS} \
        --token_mode="${TOKEN_MODE}" \
        --background_mode=historical_replay \
        --replay_data_path="${REPLAY_DATA_PATH}" \
        --wandb_project="${WANDB_PROJECT}" \
        $([ -n "${WANDB_ENTITY}" ] && echo "--wandb_entity=${WANDB_ENTITY}") \
        --seed=42 \
        2>&1 | tee logs_benchmark/${TEST_NAME}_4gpu_${SLURM_JOB_ID}.log

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "  Status: SUCCESS (${DURATION}s)"
        echo "${TEST_NAME},${N_THREADS},${N_STEPS},${LORA_RANK},4,SUCCESS,${DURATION}" >> logs_benchmark/results_4gpu_${SLURM_JOB_ID}.csv
        return 0
    else
        echo "  Status: FAILED (exit code ${EXIT_CODE})"
        echo "${TEST_NAME},${N_THREADS},${N_STEPS},${LORA_RANK},4,FAILED,${DURATION}" >> logs_benchmark/results_4gpu_${SLURM_JOB_ID}.csv
        return 1
    fi
}

# Initialize results file
echo "test_name,n_threads,n_steps,lora_rank,n_gpus,status,duration_sec" > logs_benchmark/results_4gpu_${SLURM_JOB_ID}.csv

# Run tests based on mode
# Note: All n_threads values must be divisible by 4
case ${TEST_MODE} in
    n_threads)
        echo "Testing n_threads scaling (4 GPUs)..."
        # Values divisible by 4: 32, 64, 128, 256, 512, 1024, 2048
        for N in 32 64 128 256 512 1024 2048; do
            run_test ${N} ${N_STEPS_DEFAULT} ${LORA_RANK_DEFAULT} "n_threads_${N}"
            if [ $? -ne 0 ]; then
                echo "Stopping at n_threads=${N} due to failure"
                break
            fi
        done
        ;;

    n_steps)
        echo "Testing n_steps scaling (4 GPUs)..."
        for S in 50 100 200 500 1000; do
            run_test ${N_THREADS_DEFAULT} ${S} ${LORA_RANK_DEFAULT} "n_steps_${S}"
            if [ $? -ne 0 ]; then
                echo "Stopping at n_steps=${S} due to failure"
                break
            fi
        done
        ;;

    lora_rank)
        echo "Testing lora_rank scaling (4 GPUs)..."
        for R in 2 4 8 16 32; do
            run_test ${N_THREADS_DEFAULT} ${N_STEPS_DEFAULT} ${R} "lora_rank_${R}"
            if [ $? -ne 0 ]; then
                echo "Stopping at lora_rank=${R} due to failure"
                break
            fi
        done
        ;;

    *)
        echo "Unknown TEST_MODE: ${TEST_MODE}"
        echo "Valid modes: n_threads, n_steps, lora_rank"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Multi-GPU Benchmark Complete"
echo "Results saved to: logs_benchmark/results_4gpu_${SLURM_JOB_ID}.csv"
echo "========================================"
cat logs_benchmark/results_4gpu_${SLURM_JOB_ID}.csv
