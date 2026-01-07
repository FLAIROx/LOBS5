#!/bin/bash
#SBATCH --job-name=test-a1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_a1_%j.out
#SBATCH --error=logs/test_a1_%j.err
#SBATCH --partition=workq

# ==============================================================================
# TEST-A1: n_steps=10, n_perturbations=64, background_msgs_per_step=5
# ==============================================================================

set -e

echo "============================================================"
echo "TEST-A1: Minimal Benchmark Test"
echo "============================================================"
echo "Config: n_steps=10, n_perturbations=64, background_msgs_per_step=5"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "============================================================"

mkdir -p logs

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Add JaxMARL-HFT (gymnax_exchange) to PYTHONPATH - required for ESTrainer
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"

# XLA flags removed - incompatible with current JAX version
# export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True"

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "Running TEST-A1..."
echo ""

python -c "
import sys
sys.path.insert(0, '.')

import jax
print(f'JAX devices: {jax.devices()}')

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
    n_perturbations: int = 64
    n_epochs: int = 1
    n_steps: int = 10
    background_msgs_per_step: int = 5
    token_mode: int = 24
    background_mode: str = 'historical_replay'
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    checkpoint_dir: str = '/tmp/test_a1'
    checkpoint_every: int = 9999
    seed: int = 42
    output_dir: str = '/tmp/test_a1'

import time
from es_lobs5.training.es_trainer import ESTrainer

config = TestConfig()
print(f'Config: n_steps={config.n_steps}, n_perturbations={config.n_perturbations}, background_msgs_per_step={config.background_msgs_per_step}')

print('[1/3] Initializing trainer...')
t0 = time.time()
trainer = ESTrainer(config)
print(f'  Init time: {time.time()-t0:.1f}s')

print('[2/3] Creating initial state...')
t0 = time.time()
initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
print(f'  State time: {time.time()-t0:.1f}s')

print('[3/3] Running 1 epoch...')
import jax.numpy as jnp
key = jax.random.PRNGKey(config.seed)

t0 = time.time()
mean_fitness, fitnesses, info = trainer.train_epoch(
    key, epoch=0,
    initial_sim_state=initial_sim_state,
    initial_msg_history=initial_msg_history
)
mean_fitness.block_until_ready()
epoch_time = time.time() - t0

print(f'')
print(f'============================================================')
print(f'TEST-A1 RESULT: SUCCESS')
print(f'============================================================')
print(f'  Epoch time:  {epoch_time:.1f}s')
print(f'  Fitness:     {float(mean_fitness):.4f}')
print(f'  Fitness std: {float(jnp.std(fitnesses)):.4f}')
print(f'============================================================')

# Memory stats
for i, device in enumerate(jax.devices()):
    try:
        mem = device.memory_stats()
        if mem:
            used_gb = mem.get('bytes_in_use', 0) / (1024**3)
            peak_gb = mem.get('peak_bytes_in_use', 0) / (1024**3)
            print(f'  GPU {i}: used={used_gb:.1f}GB, peak={peak_gb:.1f}GB')
    except:
        pass
"

echo ""
echo "============================================================"
echo "TEST-A1 Complete!"
echo "End: $(date)"
echo "============================================================"
