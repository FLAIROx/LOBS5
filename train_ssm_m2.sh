#!/bin/bash
#SBATCH --job-name=es-ssm-m2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_ssm_m2_%j.out
#SBATCH --error=logs/train_ssm_m2_%j.err
#SBATCH --partition=workq

set -e
echo "============================================================"
echo "ES-LOBS5 Training M2: fitness = rank_transform(pnl)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "============================================================"
echo "Changes from M1:"
echo "  - Removed normalization: fitness = rank_transform(pnl) directly"
echo "  - WandB: pnl/* = raw cents, fitness/* = rank output [-0.5, 0.5]"
echo "  - Tracks best_pnl instead of best_fitness"
echo "============================================================"

mkdir -p logs checkpoints/es_ssm_m2

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
    # Model
    lobs5_checkpoint: str = '/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    data_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'

    # ES parameters
    noiser: str = 'eggrollbs'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0
    n_perturbations: int = 1024
    n_epochs: int = 5000
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
    checkpoint_dir: str = '/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/es_ssm_m2'
    checkpoint_every: int = 50
    seed: int = 42
    output_dir: str = '/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/es_ssm_m2'

    # WandB - enables trainer.train() integrated logging
    wandb_project: str = 'ES-LOBS5'
    wandb_entity: str = 'kang-oxford'

print(f'JAX devices: {jax.devices()}')
print()
print('Training with:')
print('  - fitness = rank_transform(pnl) directly (no intermediate normalization)')
print('  - PnL in raw cents (real economic value)')
print('  - Fitness in [-0.5, 0.5] range (rank-based)')
print()

from es_lobs5.training.es_trainer import ESTrainer

config = Config()
trainer = ESTrainer(config)

# Use trainer.train() for integrated wandb logging
trainer.train(n_epochs=config.n_epochs)

print('Training complete!')
"

echo ""
echo "M2 Training Complete! End: $(date)"
