#!/bin/bash
#SBATCH --job-name=es-prod-24h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_production_%j.out
#SBATCH --error=logs/train_production_%j.err
#SBATCH --partition=workq

set -e
echo "============================================================"
echo "ES-LOBS5 Production Training (24h)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "============================================================"
echo "Config: n_perturbations=1024, n_steps=10, bg_msgs/step=100"
echo "============================================================"

mkdir -p logs
mkdir -p checkpoints/es_production

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
from datetime import datetime

@dataclass
class Config:
    # Model checkpoint
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'

    # Data paths
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    data_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'

    # ES parameters
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training scale (G5 config - optimal for 4x GH200)
    n_perturbations: int = 1024      # 256 per GPU
    n_epochs: int = 1000              # Run many epochs (will stop at 24h)
    n_steps: int = 10                # Steps per episode
    n_warmup_msgs: int = 500         # Warmup messages
    background_msgs_per_step: int = 100  # Background messages per step

    # Model config
    token_mode: int = 24
    background_mode: str = 'historical_replay'

    # Task config
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100

    # Checkpointing
    checkpoint_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/es_production'
    checkpoint_every: int = 10       # Save every 10 epochs

    seed: int = 42
    output_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/es_production'

print(f'JAX devices: {jax.devices()}')
print(f'Number of devices: {len(jax.devices())}')

from es_lobs5.training.es_trainer import ESTrainer

config = Config()

print()
print('[1/3] Initializing trainer...')
t0 = time.time()
trainer = ESTrainer(config)
init_time = time.time() - t0
print(f'  Init time: {init_time:.1f}s')

print()
print('[2/3] Creating initial state...')
t0 = time.time()
initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
state_time = time.time() - t0
print(f'  State time: {state_time:.1f}s')

print()
print('[3/3] Starting training loop...')
print('='*60)

key = jax.random.PRNGKey(config.seed)
training_start = time.time()
max_training_time = 23.5 * 3600  # Stop 30 min before 24h limit

best_fitness = -float('inf')
fitness_history = []

for epoch in range(config.n_epochs):
    epoch_start = time.time()

    # Check time limit
    elapsed = time.time() - training_start
    if elapsed > max_training_time:
        print(f'\\nTime limit reached ({elapsed/3600:.1f}h). Stopping training.')
        break

    # Split key for this epoch
    key, epoch_key = jax.random.split(key)

    # Train one epoch
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

    # Track best
    if mean_fitness_val > best_fitness:
        best_fitness = mean_fitness_val
        best_marker = ' *BEST*'
    else:
        best_marker = ''

    # Print progress
    print(f'Epoch {epoch+1:4d} | Fitness: {mean_fitness_val:8.2f} ± {std_fitness:6.2f} | '
          f'Time: {epoch_time:5.1f}s | Total: {total_elapsed/3600:.2f}h{best_marker}')

    # Checkpoint
    if (epoch + 1) % config.checkpoint_every == 0:
        ckpt_path = os.path.join(config.checkpoint_dir, f'epoch_{epoch+1:04d}')
        print(f'  -> Saving checkpoint to {ckpt_path}')
        # TODO: Implement checkpoint saving
        # For now just save fitness history
        np.save(os.path.join(config.checkpoint_dir, 'fitness_history.npy'),
                np.array(fitness_history))

total_time = time.time() - training_start
n_epochs_completed = len(fitness_history)

print()
print('='*60)
print('TRAINING COMPLETE')
print('='*60)
print(f'  Total epochs:     {n_epochs_completed}')
print(f'  Total time:       {total_time/3600:.2f}h')
print(f'  Avg epoch time:   {total_time/n_epochs_completed:.1f}s')
print(f'  Best fitness:     {best_fitness:.4f}')
print(f'  Final fitness:    {fitness_history[-1]:.4f}')
print(f'  Fitness std:      {np.std(fitness_history):.4f}')
print('='*60)

# Save final results
np.save(os.path.join(config.checkpoint_dir, 'fitness_history.npy'),
        np.array(fitness_history))

# Print GPU memory stats
print()
print('GPU Memory Usage:')
for i, device in enumerate(jax.devices()):
    try:
        mem = device.memory_stats()
        if mem:
            used_gb = mem.get('bytes_in_use', 0) / (1024**3)
            peak_gb = mem.get('peak_bytes_in_use', 0) / (1024**3)
            print(f'  GPU {i}: used={used_gb:.1f}GB, peak={peak_gb:.1f}GB')
    except: pass
"

echo ""
echo "============================================================"
echo "Production Training Complete! End: $(date)"
echo "============================================================"
