#!/bin/bash
#SBATCH --job-name=lobs5_3072x32
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/logs_lobs5_3072x32/lobs5_%j.out
#SBATCH --error=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/logs_lobs5_3072x32/lobs5_%j.err
#SBATCH --chdir=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Multi-node JAX training batch script
# Uses 2 nodes, 4 GPUs per node (8 total GPUs)

# Set working directory
WORK_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5"
cd "$WORK_DIR"

# Create logs directory if not exists
mkdir -p "$WORK_DIR/logs_lobs5_3072x32"

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on nodes: ${SLURM_JOB_NODELIST}"
echo "Number of nodes: ${SLURM_NNODES}"
echo "GPUs per node (requested): 4"
echo "Total GPUs (expected): $((SLURM_NNODES * 4))"
echo "============================================"
echo ""

# Multi-node communication setup
if [ "${SLURM_NNODES}" -gt 1 ]; then
    echo "[*] Multi-node mode detected (${SLURM_NNODES} nodes)"
    echo "[*] JAX will use jax.distributed.initialize()"

    # Get the first node as coordinator
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    MASTER_PORT=29500

    export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"

    echo ""
    echo "[*] Coordinator address: ${JAX_COORDINATOR_ADDRESS}"
    echo "[*] Master addr: ${MASTER_ADDR}"
    echo "[*] Master port: ${MASTER_PORT}"
fi

echo ""
echo "[*] Starting training at: $(date)"
echo "============================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run wrapper script on all nodes
srun --export=ALL bash "$WORK_DIR/bin/run_experiments/run_lobster_padded_large.sh"

# Calculate and display runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(((RUNTIME % 3600) / 60))
SECONDS=$((RUNTIME % 60))

echo ""
echo "============================================"
echo "Training completed at: $(date)"
echo "Total runtime: ${RUNTIME} seconds"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================"
