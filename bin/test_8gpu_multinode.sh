#!/bin/bash
#SBATCH --job-name=test_8gpu_2node
#SBATCH --output=logs/test_8gpu_%j.out
#SBATCH --error=logs/test_8gpu_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=workq
#SBATCH --contiguous

mkdir -p logs

echo "========================================"
echo "8 GPU Multi-Node Test (2 nodes × 4 GPUs)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Node count: $SLURM_NNODES"
echo "========================================"

# Multi-node coordination
export JAX_COORDINATOR_ADDRESS=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):12345
export JAX_PROCESS_COUNT=$SLURM_NNODES
export JAX_COORDINATOR_TIMEOUT_MS=600000

# Run on all nodes
srun bash bin/run_experiments/run_lobster_padded_large.sh \
    --epochs=3 \
    --curtail_epochs=10 \
    --USE_WANDB=True \
    --wandb_project=lobs5-bf16-multinode-test

# Note: run_lobster_padded_large.sh will set USE_SINGLE_OPTIMIZER=1 via environment

echo "========================================"
echo "Test completed: $(date)"
echo "========================================"
