#!/bin/bash
#SBATCH --job-name=test-trade-markers
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_trade_markers_%j.out
#SBATCH --error=logs/test_trade_markers_%j.err
#SBATCH --partition=workq

set -e
echo "============================================================"
echo "Test: Trade Markers Visualization"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "============================================================"
echo "Testing:"
echo "  - market_data_trace (clean)"
echo "  - market_data_trace_with_trades (agent trades, colored by execution quality)"
echo "  - market_data_trace_historical_trades (LOBSTER event_type=4 trades)"
echo "============================================================"

mkdir -p logs checkpoints/test_trade_markers

# Use projects conda (same as es_training.sh)
CONDA_PATH="/projects/s5e/quant/miniforge3"
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate lobs5

cd /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5
export PYTHONPATH="/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/AlphaTrade:$PYTHONPATH"

# WandB environment
export WANDB_MODE=online
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_INSECURE_DISABLE_SSL=True

nvidia-smi --query-gpu=name,memory.total --format=csv

# Use explicit Python path (conda activate may not work in sbatch)
PYTHON="${CONDA_PATH}/envs/lobs5/bin/python"
export PYTHONUNBUFFERED=1
${PYTHON} -c "
import sys
sys.path.insert(0, '.')
import jax
from dataclasses import dataclass

@dataclass
class Config:
    # Model (using projects paths)
    lobs5_checkpoint: str = '/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    replay_data_path: str = '/lus/lfs1aip2/projects/s5e/quant/JAN2023/GOOG_24tok_preproc'
    data_dir: str = '/lus/lfs1aip2/projects/s5e/quant/JAN2023/GOOG_24tok_preproc'

    # ES parameters - short run for testing
    noiser: str = 'eggrollbs'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0
    n_perturbations: int = 256  # Smaller for faster test
    n_epochs: int = 30          # Short run
    n_steps: int = 10
    n_warmup_msgs: int = 50
    background_msgs_per_step: int = 50
    token_mode: int = 24
    background_mode: str = 'historical_replay'

    # Task
    task: str = 'sell'
    task_size: int = 50
    tick_size: int = 100

    # EggRollBS requires group_size
    group_size: int = 8

    # Checkpoint
    checkpoint_dir: str = '/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/test_trade_markers'
    checkpoint_every: int = 10
    seed: int = 42
    output_dir: str = '/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/test_trade_markers'

    # WandB
    wandb_project: str = 'ES-LOBS5'
    wandb_entity: str = 'kang-oxford'

print(f'JAX devices: {jax.devices()}')
print()
print('Testing trade markers visualization:')
print('  - market_data_trace (clean market data)')
print('  - market_data_trace_with_trades (agent trades, colored by execution quality)')
print('  - market_data_trace_historical_trades (LOBSTER event_type=4 trades)')
print()

from es_lobs5.training.es_trainer import ESTrainer

config = Config()
trainer = ESTrainer(config)
trainer.train(n_epochs=config.n_epochs)

print('Test complete!')
"

echo ""
echo "Test Complete! End: $(date)"
