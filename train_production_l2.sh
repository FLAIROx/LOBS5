#!/bin/bash
#SBATCH --job-name=es-prod-l2-24h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_production_l2_%j.out
#SBATCH --error=logs/train_production_l2_%j.err
#SBATCH --partition=workq

set -e
echo "ES-LOBS5 Production Training L2 (24h) + WandB + EggRollBS"
echo "Config: G5 base + task_size=10 + file_idx=0 + EggRollBS (baseline subtraction)"

mkdir -p logs checkpoints/es_production_l2
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
nvidia-smi --query-gpu=name,memory.total --format=csv

export PYTHONUNBUFFERED=1
python -c "
import sys; sys.path.insert(0, '.')
import os
os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'True'
import wandb
import jax, jax.numpy as jnp, numpy as np, time
from dataclasses import dataclass, asdict

@dataclass
class Config:
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    data_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
    noiser: str = 'eggrollbs'; sigma: float = 0.01; lr: float = 0.001; lora_rank: int = 4; grad_clip: float = 1.0
    n_perturbations: int = 1024; n_epochs: int = 1000; n_steps: int = 10; n_warmup_msgs: int = 500
    background_msgs_per_step: int = 100; token_mode: int = 24; background_mode: str = 'historical_replay'
    task: str = 'sell'; task_size: int = 10; tick_size: int = 100
    file_idx: int = 0  # Fixed data window
    group_size: int = 8  # Required for EggRollBS
    checkpoint_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/es_production_l2'
    checkpoint_every: int = 10; seed: int = 42
    output_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/es_production_l2'

config = Config()
run = wandb.init(project='ES-LOBS5', name='L2_steps10_bg100_task10_eggrollbs', config=asdict(config), tags=['L2', 'production', '24h', 'task10', 'fixed_data', 'eggrollbs'])
print(f'JAX devices: {jax.devices()} | WandB: {run.name}')
from es_lobs5.training.es_trainer import ESTrainer
trainer = ESTrainer(config)
initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
key = jax.random.PRNGKey(config.seed)
training_start = time.time()
best_fitness = -float('inf')
fitness_history = []
for epoch in range(config.n_epochs):
    if time.time() - training_start > 23.5 * 3600: break
    epoch_start = time.time()
    key, epoch_key = jax.random.split(key)
    mean_fitness, fitnesses, info = trainer.train_epoch(epoch_key, epoch=epoch, initial_sim_state=initial_sim_state, initial_msg_history=initial_msg_history)
    mf = float(mean_fitness.block_until_ready()); sf = float(jnp.std(fitnesses))
    et = time.time() - epoch_start; te = time.time() - training_start
    fitness_history.append(mf)
    if mf > best_fitness: best_fitness = mf
    wandb.log({'epoch': epoch+1, 'fitness/mean': mf, 'fitness/std': sf, 'fitness/max': float(jnp.max(fitnesses)), 'fitness/min': float(jnp.min(fitnesses)), 'fitness/best': best_fitness, 'time/epoch_seconds': et, 'time/total_hours': te/3600}, step=epoch+1)
    print(f'Epoch {epoch+1:4d} | Fitness: {mf:8.2f} +/- {sf:6.2f} | Time: {et:5.1f}s | Total: {te/3600:.2f}h')
    if (epoch + 1) % config.checkpoint_every == 0: np.save(os.path.join(config.checkpoint_dir, 'fitness_history.npy'), np.array(fitness_history))
print(f'COMPLETE (L2) | Epochs: {len(fitness_history)} | Best: {best_fitness:.4f}')
np.save(os.path.join(config.checkpoint_dir, 'fitness_history.npy'), np.array(fitness_history))
wandb.finish()
"
echo "L2 Complete! End: \$(date)"
