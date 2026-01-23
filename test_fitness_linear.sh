#!/bin/bash
#SBATCH --job-name=test-fitness-linear
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_fitness_linear_%j.out
#SBATCH --error=logs/test_fitness_linear_%j.err
#SBATCH --partition=workq

set -e
echo "============================================================"
echo "Test: Linear Fitness (no tanh) + Improved X-axis"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "============================================================"

mkdir -p logs
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"

# WandB environment
export WANDB_MODE=online
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_INSECURE_DISABLE_SSL=True

nvidia-smi --query-gpu=name,memory.total --format=csv

python -c "
import sys
sys.path.insert(0, '.')
import jax
from dataclasses import dataclass

@dataclass
class Config:
    # Model
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    data_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'

    # ES parameters
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0
    n_perturbations: int = 1024
    n_epochs: int = 100  # Short test
    n_steps: int = 10
    n_warmup_msgs: int = 50
    background_msgs_per_step: int = 50
    token_mode: int = 24
    background_mode: str = 'historical_replay'

    # Task
    task: str = 'sell'
    task_size: int = 50
    tick_size: int = 100

    # Checkpoint
    checkpoint_dir: str = '/tmp/test_fitness_linear'
    checkpoint_every: int = 9999
    seed: int = 42
    output_dir: str = '/tmp/test_fitness_linear'

    # WandB
    wandb_project: str = 'ES-LOBS5'
    wandb_entity: str = 'kang-oxford'

print(f'JAX devices: {jax.devices()}')
print('Testing: Linear Fitness (no tanh) + Improved X-axis markings')
print('Expected: fitness values NOT bounded to [-1, +1]')
print('Expected: pnl shows raw cents value')

from es_lobs5.training.es_trainer import ESTrainer

config = Config()
trainer = ESTrainer(config)
trainer.train(n_epochs=config.n_epochs)

print('Test complete!')
"

echo ""
echo "Test Complete! End: $(date)"
