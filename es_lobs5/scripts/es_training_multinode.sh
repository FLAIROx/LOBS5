#!/bin/bash
#SBATCH --job-name=es-multinode
#SBATCH --nodes=2             # Default, override with sbatch --nodes=N
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00        # Short time for testing
#SBATCH --output=logs/es_multinode_%j.out
#SBATCH --error=logs/es_multinode_%j.err
#SBATCH --partition=workq
#SBATCH --exclusive           # Exclusive access is usually better for multi-node

# =============================================================================
# ES Multi-Node Training Script
# =============================================================================
# Usage:
#   sbatch --nodes=4 es_lobs5/scripts/es_training_multinode.sh
# =============================================================================

# -----------------------------------------------------------------------------
# JAX Distributed Setup
# -----------------------------------------------------------------------------
# get the first node name as coordinator
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect ipv6, we might need to use brackets, but hostname -I usually gives ipv4 first.
# actually `hostname --ip-address` gives the IP.

echo "Coordinator: $head_node ($head_node_ip)"
export COORD_ADDR="$head_node_ip:6000"

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/es_lobs5_jax_compilation"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/AlphaTrade:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/}"
DATA_DIR="${DATA_DIR:-/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc}"

N_EPOCHS="${N_EPOCHS:-20}" # Default short for testing
PERGPU_PERTURBATIONS="${PERGPU_PERTURBATIONS:-128}" # Safe default
N_STEPS="${N_STEPS:-50}"
BG_MSGS="${BG_MSGS:-50}"
TASK_SIZE="${TASK_SIZE:-50}"

WANDB_PROJECT="${WANDB_PROJECT:-es-multinode-test}"
WANDB_ENTITY="${WANDB_ENTITY:-kang-oxford}"

CHECKPOINT_DIR="checkpoints/es_multinode/${SLURM_JOB_ID}"

echo "=============================================="
echo " ES Multi-Node Training"
echo "=============================================="
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total Processes: ${SLURM_NTASKS}"
echo "Coordinator: ${COORD_ADDR}"
echo "Per-GPU Perturbations: ${PERGPU_PERTURBATIONS}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------
# We use srun to launch one process per GPU (ntasks-per-node=4)
# es_training.py handles distributed init using --coord_addr

srun python es_lobs5/scripts/es_training.py \
    --lobs5_checkpoint "${CHECKPOINT}" \
    --replay_data_path "${DATA_DIR}" \
    --n_epochs ${N_EPOCHS} \
    --pergpu_perturbations ${PERGPU_PERTURBATIONS} \
    --n_steps ${N_STEPS} \
    --n_warmup_msgs 500 \
    --background_msgs_per_step ${BG_MSGS} \
    --task sell \
    --task_size ${TASK_SIZE} \
    --tick_size 100 \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --coord_addr "${COORD_ADDR}" \
    --num_procs ${SLURM_NTASKS} \
    --proc_id -1 # Will be auto-detected from SLURM_PROCID
