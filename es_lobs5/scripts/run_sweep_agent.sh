#!/bin/bash
#SBATCH --job-name=es-sweep-agent
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_agent_%j.out
#SBATCH --error=logs/sweep_agent_%j.err
#SBATCH --partition=workq

# Usage: sbatch es_lobs5/scripts/run_sweep_agent.sh <SWEEP_ID> [PROJECT] [ENTITY]

SWEEP_ID=$1
PROJECT=${2:-es-lobs5-sweep}
ENTITY=${3:-kang-oxford}

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Sweep ID is required."
    echo "Usage: sbatch es_lobs5/scripts/run_sweep_agent.sh <SWEEP_ID>"
    exit 1
fi

echo "Starting Sweep Agent for ID: ${SWEEP_ID}"
echo "Project: ${PROJECT}"
echo "Entity: ${ENTITY}"

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# JAX Cache
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/es_lobs5_jax_compilation"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/AlphaTrade:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Configuration (Static values not in sweep)
# -----------------------------------------------------------------------------
# These will be passed to es_training.py alongside sweep parameters
# Note: wandb agent passes sweep params as command line arguments automatically
# We need to ensure other required arguments are present via defaults or cmd line.

# Since es_training.py requires --lobs5_checkpoint, we MUST provide it.
# We can provide it here or ensure it's in the command line constructed by wandb?
# WandB agent only passes the parameters defined in the sweep config.
# So we need to wrap the agent call or modify 'program' in sweep config to a wrapper script?
# OR we can just rely on argparse defaults if we change the script?
# But es_training.py requires --lobs5_checkpoint.

# Solution: We should probably modify setup_es_sweep.py to include 'command' field 
# OR use a wrapper script as 'program'.
# But simpler: Export environment variables that are picked up?
# es_training.py uses required=True for lobs5_checkpoint.

# Let's specify the constant arguments to the command by using the 'command' field in sweep config?
# But 'command' field overrides default python call.
# Actually, the easier way is to define CHECPOINT env var? No, argparse requires it.

# Let's check es_training.py again. It uses argparse.
# We can use ${args} in the 'command' section of sweep config.

# Let's update setup_es_sweep.py to include the full command structure first.
# Wait, I already wrote setup_es_sweep.py without 'command'.
# WandB default command is: ${program} ${args}
# This means it will run: python es_lobs5/scripts/es_training.py --sigma X --lr Y ...
# It won't have --lobs5_checkpoint.

# So I need to modify setup_es_sweep.py to explicitly define the command including the checkpoint.

# Default paths
CHECKPOINT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc"

# Clean up logs
mkdir -p logs

# Run Agent
# wandb agent will fetch one set of params and run the program.
# But we need to make sure the program gets the required static args.
# The robust way is to use a wrapper script OR define the command in sweep config.

# I will update setup_es_sweep.py to include the 'command' field.
# But for now, let's just make this script run the agent.
# The user will need to re-run setup_es_sweep.py.

wandb agent --count 1 ${ENTITY}/${PROJECT}/${SWEEP_ID}
