#!/bin/bash
# =============================================================================
# ES Training Multi-Node Wrapper
# =============================================================================
#
# This script dynamically generates an SBATCH file with the specified number
# of nodes and submits it to the queue.
#
# Usage:
#   N_NODES=2 N_PERTURBATIONS=256 ./es_training_multinode.sh
#   N_NODES=4 N_EPOCHS=1000 ./es_training_multinode.sh
#
# Environment Variables:
#   N_NODES           - Number of nodes to use (default: 1)
#   N_PERTURBATIONS   - Population size (must be divisible by N_NODES * 4 GPUs)
#   ... all other variables from es_training.sh are supported
#
# =============================================================================

set -e

# Get number of nodes (default 1)
N_NODES="${N_NODES:-1}"
N_GPUS_PER_NODE=4
N_TOTAL_GPUS=$((N_NODES * N_GPUS_PER_NODE))

# Default parameters (same as es_training.sh)
CHECKPOINT="${CHECKPOINT:-/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/}"
DATA_DIR="${DATA_DIR:-/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc}"
N_EPOCHS="${N_EPOCHS:-1000}"
N_PERTURBATIONS="${N_PERTURBATIONS:-$((N_TOTAL_GPUS * 32))}"  # Default: 32 per GPU
N_STEPS="${N_STEPS:-100}"
N_WARMUP="${N_WARMUP:-500}"
BG_MSGS="${BG_MSGS:-10}"
SIGMA="${SIGMA:-0.01}"
LR="${LR:-0.001}"
NOISER="${NOISER:-eggroll}"
LORA_RANK="${LORA_RANK:-4}"
TASK="${TASK:-sell}"
TASK_SIZE="${TASK_SIZE:-500}"
TICK_SIZE="${TICK_SIZE:-100}"
FILE_IDX="${FILE_IDX:-}"
WANDB_PROJECT="${WANDB_PROJECT:-es-lobs5}"
WANDB_ENTITY="${WANDB_ENTITY:-kang-oxford}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"

# Validate n_perturbations is divisible by total GPUs
if [ $((N_PERTURBATIONS % N_TOTAL_GPUS)) -ne 0 ]; then
    echo "ERROR: N_PERTURBATIONS ($N_PERTURBATIONS) must be divisible by total GPUs ($N_TOTAL_GPUS)"
    echo "       Suggested: N_PERTURBATIONS=$((N_TOTAL_GPUS * (N_PERTURBATIONS / N_TOTAL_GPUS + 1)))"
    exit 1
fi

echo "=============================================="
echo " ES Training Multi-Node Setup"
echo "=============================================="
echo "N_NODES: ${N_NODES}"
echo "GPUs per node: ${N_GPUS_PER_NODE}"
echo "Total GPUs: ${N_TOTAL_GPUS}"
echo "N_PERTURBATIONS: ${N_PERTURBATIONS} (${N_PERTURBATIONS}/${N_TOTAL_GPUS} = $((N_PERTURBATIONS / N_TOTAL_GPUS)) per GPU)"
echo "=============================================="

# Create temporary SBATCH script
TEMP_SBATCH=$(mktemp /tmp/es_multinode_XXXXXX.sbatch)

cat > "${TEMP_SBATCH}" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=es-train-${N_NODES}n
#SBATCH --nodes=${N_NODES}
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/es_train_%j.out
#SBATCH --error=logs/es_train_%j.err
#SBATCH --partition=workq

echo "=============================================="
echo " ES Training Multi-Node - ${N_NODES} nodes"
echo "=============================================="
echo "Job ID: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_NODELIST}"
echo "Total GPUs: ${N_TOTAL_GPUS}"
echo "Start time: \$(date)"
echo "=============================================="

# Environment Setup
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/AlphaTrade:\$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Create directories
mkdir -p logs
mkdir -p checkpoints/es_runs

# Checkpoint directory for this job
CHECKPOINT_DIR="checkpoints/es_runs/\${SLURM_JOB_ID}"

# Get coordinator address (first node)
COORD_ADDR=\$(scontrol show hostnames \$SLURM_NODELIST | head -n 1)
COORD_PORT=12345

echo ""
echo "Distributed Configuration:"
echo "  Coordinator: \${COORD_ADDR}:\${COORD_PORT}"
echo "  Num processes: \${SLURM_NTASKS} (${N_NODES} nodes x 4 GPUs)"
echo ""

# Build optional file_idx argument
FILE_IDX_ARG=""
if [ -n "${FILE_IDX}" ]; then
    FILE_IDX_ARG="--file_idx ${FILE_IDX}"
fi

# Debug: Show SLURM environment variables
echo "DEBUG: SLURM_NODEID=\${SLURM_NODEID}, SLURM_PROCID=\${SLURM_PROCID}, SLURM_LOCALID=\${SLURM_LOCALID}"
echo "DEBUG: SLURM_NTASKS=\${SLURM_NTASKS}, SLURM_NNODES=\${SLURM_NNODES}"

# Launch with srun (4 processes per node, one per GPU)
srun bash -c 'echo "SRUN DEBUG: PROCID=\${SLURM_PROCID}, LOCALID=\${SLURM_LOCALID}, CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not_set} on \$(hostname)"'
srun python es_lobs5/scripts/es_training.py \\
    --lobs5_checkpoint "${CHECKPOINT}" \\
    --replay_data_path "${DATA_DIR}" \\
    --n_epochs ${N_EPOCHS} \\
    --n_perturbations ${N_PERTURBATIONS} \\
    --n_steps ${N_STEPS} \\
    --n_warmup_msgs ${N_WARMUP} \\
    --background_msgs_per_step ${BG_MSGS} \\
    --sigma ${SIGMA} \\
    --lr ${LR} \\
    --lora_rank ${LORA_RANK} \\
    --noiser ${NOISER} \\
    --background_mode historical_replay \\
    --task ${TASK} \\
    --task_size ${TASK_SIZE} \\
    --tick_size ${TICK_SIZE} \\
    --token_mode 24 \\
    --checkpoint_every ${CHECKPOINT_EVERY} \\
    --checkpoint_dir "\${CHECKPOINT_DIR}" \\
    --wandb_project "${WANDB_PROJECT}" \\
    --wandb_entity "${WANDB_ENTITY}" \\
    --coord_addr "\${COORD_ADDR}:\${COORD_PORT}" \\
    --num_procs \${SLURM_NTASKS} \\
    \${FILE_IDX_ARG}

EXIT_CODE=\$?

echo ""
echo "=============================================="
echo "End time: \$(date)"
echo "Exit code: \${EXIT_CODE}"
echo "=============================================="

exit \${EXIT_CODE}
SBATCH_EOF

echo ""
echo "Generated SBATCH script: ${TEMP_SBATCH}"
echo ""

# Submit the job
sbatch "${TEMP_SBATCH}"

# Clean up
rm -f "${TEMP_SBATCH}"
