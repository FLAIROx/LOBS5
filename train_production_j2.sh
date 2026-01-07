#!/bin/bash
#SBATCH --job-name=es-prod-j2-24h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_production_j2_%j.out
#SBATCH --error=logs/train_production_j2_%j.err
#SBATCH --partition=workq

set -e
echo "============================================================"
echo "ES-LOBS5 Production Training J2 (24h) - task_size=10"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "============================================================"
echo "Config: G5 base + task_size=10 (n_steps=10, bg_msgs=100)"
echo "============================================================"

mkdir -p logs
mkdir -p checkpoints/es_production_j2

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"

nvidia-smi --query-gpu=name,memory.total --format=csv

export PYTHONUNBUFFERED=1
python -c "
import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
from dataclasses import dataclass

@dataclass
class Config:
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    data_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0
    n_perturbations: int = 1024
    n_epochs: int = 1000
    n_steps: int = 10                # G5 base
    n_warmup_msgs: int = 500
    background_msgs_per_step: int = 100  # G5 base
    token_mode: int = 24
    background_mode: str = 'historical_replay'
    task: str = 'sell'
    task_size: int = 10              # J2: 10 shares to execute
    tick_size: int = 100
    checkpoint_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/es_production_j2'
    checkpoint_every: int = 10
    seed: int = 42
    output_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/es_production_j2'

print(f'JAX devices: {jax.devices()}')
print(f'Config: J2 (G5 base, task_size=10)')

from es_lobs5.training.es_trainer import ESTrainer

config = Config()

print('[1/3] Initializing trainer...')
t0 = time.time()
trainer = ESTrainer(config)
print(f'  Init time: {time.time()-t0:.1f}s')

print('[2/3] Creating initial state...')
t0 = time.time()
initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
print(f'  State time: {time.time()-t0:.1f}s')

print('[3/3] Starting training loop...')
print('='*60)

key = jax.random.PRNGKey(config.seed)
training_start = time.time()
max_training_time = 23.5 * 3600

best_fitness = -float('inf')
fitness_history = []

for epoch in range(config.n_epochs):
    epoch_start = time.time()
    elapsed = time.time() - training_start
    if elapsed > max_training_time:
        print(f'Time limit reached ({elapsed/3600:.1f}h). Stopping.')
        break
    key, epoch_key = jax.random.split(key)
    mean_fitness, fitnesses, info = trainer.train_epoch(
        epoch_key, epoch=epoch,
        initial_sim_state=initial_sim_state,
        initial_msg_history=initial_msg_history
    )
    mean_fitness_val = float(mean_fitness.block_until_ready())
    std_fitness = float(jnp.std(fitnesses))
    epoch_time = time.time() - epoch_start
    total_elapsed = time.time() - training_start
    fitness_history.append(mean_fitness_val)
    if mean_fitness_val > best_fitness:
        best_fitness = mean_fitness_val
        best_marker = ' *BEST*'
    else:
        best_marker = ''
    print(f'Epoch {epoch+1:4d} | Fitness: {mean_fitness_val:8.2f} +/- {std_fitness:6.2f} | '
          f'Time: {epoch_time:5.1f}s | Total: {total_elapsed/3600:.2f}h{best_marker}')
    if (epoch + 1) % config.checkpoint_every == 0:
        np.save(os.path.join(config.checkpoint_dir, 'fitness_history.npy'), np.array(fitness_history))

print()
print('='*60)
print('TRAINING COMPLETE (J2: G5 base, task_size=10)')
print('='*60)
print(f'  Total epochs: {len(fitness_history)}')
print(f'  Best fitness: {best_fitness:.4f}')
np.save(os.path.join(config.checkpoint_dir, 'fitness_history.npy'), np.array(fitness_history))
"

echo "J2 Complete! End: $(date)"
