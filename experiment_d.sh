#!/bin/bash
#SBATCH --job-name=exp-d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/experiment_d_%j.out
#SBATCH --error=logs/experiment_d_%j.err
#SBATCH --partition=workq

# ==============================================================================
# Experiment D: n_perturbations Scaling Test
# ==============================================================================
# Fixed: background_msgs_per_step=10, n_steps=10
# Variable: n_perturbations = 64, 128, 256, 512, 1024, 2048, 4096
# ==============================================================================

set -e

echo "============================================================"
echo "Experiment D: n_perturbations Scaling Test"
echo "============================================================"
echo "Fixed: background_msgs_per_step=10, n_steps=10"
echo "Variable: n_perturbations = 64, 128, 256, 512, 1024, 2048, 4096"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "============================================================"

mkdir -p logs

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Add JaxMARL-HFT to PYTHONPATH
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Fixed parameters
N_STEPS=10
BACKGROUND_MSGS=10

# Variable: n_perturbations to test
PERTURBATIONS="64 128 256 512 1024 2048 4096"

echo "| n_perturbations | Epoch Time (s) | Fitness | Fitness Std | Status |"
echo "|-----------------|----------------|---------|-------------|--------|"

for N_PERT in $PERTURBATIONS; do
    echo ""
    echo "============================================================"
    echo "Testing: n_perturbations=$N_PERT"
    echo "============================================================"

    # Run test and capture output
    RESULT=$(python -c "
import sys
sys.path.insert(0, '.')

import jax
import time
from dataclasses import dataclass

@dataclass
class TestConfig:
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    data_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0
    n_perturbations: int = $N_PERT
    n_epochs: int = 1
    n_steps: int = $N_STEPS
    background_msgs_per_step: int = $BACKGROUND_MSGS
    token_mode: int = 24
    background_mode: str = 'historical_replay'
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    checkpoint_dir: str = '/tmp/exp_d_$N_PERT'
    checkpoint_every: int = 9999
    seed: int = 42
    output_dir: str = '/tmp/exp_d_$N_PERT'

from es_lobs5.training.es_trainer import ESTrainer
import jax.numpy as jnp

config = TestConfig()

try:
    trainer = ESTrainer(config)
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()

    key = jax.random.PRNGKey(config.seed)

    t0 = time.time()
    mean_fitness, fitnesses, info = trainer.train_epoch(
        key, epoch=0,
        initial_sim_state=initial_sim_state,
        initial_msg_history=initial_msg_history
    )
    mean_fitness.block_until_ready()
    epoch_time = time.time() - t0

    fitness_val = float(mean_fitness)
    fitness_std = float(jnp.std(fitnesses))

    print(f'SUCCESS,{epoch_time:.1f},{fitness_val:.4f},{fitness_std:.4f}')
except Exception as e:
    print(f'FAILED,0,0,0,{str(e)[:50]}')
" 2>&1 | tail -1)

    # Parse result
    STATUS=$(echo "$RESULT" | cut -d',' -f1)
    EPOCH_TIME=$(echo "$RESULT" | cut -d',' -f2)
    FITNESS=$(echo "$RESULT" | cut -d',' -f3)
    FITNESS_STD=$(echo "$RESULT" | cut -d',' -f4)

    if [ "$STATUS" = "SUCCESS" ]; then
        echo "| $N_PERT | $EPOCH_TIME | $FITNESS | $FITNESS_STD | OK |"
    else
        echo "| $N_PERT | - | - | - | FAILED |"
        # If OOM, stop testing larger sizes
        if echo "$RESULT" | grep -q "out of memory\|OOM\|RESOURCE_EXHAUSTED"; then
            echo ""
            echo "OOM detected at n_perturbations=$N_PERT, stopping further tests"
            break
        fi
    fi
done

echo ""
echo "============================================================"
echo "Experiment D Complete!"
echo "End: $(date)"
echo "============================================================"
