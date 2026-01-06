#!/usr/bin/env python3
"""
C2: 单 Epoch 训练验证 (World Model, 最小规模)
目标: 验证 World Model 模式下能完成训练 (n_threads=4, n_steps=5)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import time


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
DATA_DIR = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """ES training configuration for World Model mode."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration (minimal for world model - slow due to AR generation)
    n_threads: int = 4
    n_epochs: int = 1
    n_steps: int = 5
    world_msgs_per_step: int = 2

    # Token mode (24 matches the checkpoint)
    token_mode: int = 24

    # Background mode: world_model
    background_mode: str = 'world_model'
    replay_data_path: str = None
    data_dir: str = DATA_DIR

    # Task configuration
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100

    # Other
    seed: int = 42
    output_dir: str = '/tmp/es_validation_c2'


def test_single_epoch_world_model():
    """验证 World Model 模式下能完成单 epoch 训练"""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 60)
    print("C2: 单 Epoch 训练验证 (World Model, 4×5)")
    print("=" * 60)

    # Create config
    print("\n[1/5] Creating ESConfig (World Model)...")
    config = ESConfig()
    print(f"  n_threads: {config.n_threads}")
    print(f"  n_steps: {config.n_steps}")
    print(f"  world_msgs_per_step: {config.world_msgs_per_step}")
    print(f"  background_mode: {config.background_mode}")
    print("  ✓ Config created")

    # Initialize trainer
    print("\n[2/5] Initializing ESTrainer...")
    trainer = ESTrainer(config)
    print("  ✓ ESTrainer initialized")

    # Get initial state
    print("\n[3/5] Creating initial simulation state...")
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  ✓ initial_msg_history shape: {initial_msg_history.shape}")

    # Run single epoch (World Model is slower due to AR generation)
    print("\n[4/5] Running single epoch (World Model - expect ~2-5 min)...")
    key = jax.random.PRNGKey(config.seed)

    start_time = time.time()
    mean_fitness, fitnesses, info = trainer.train_epoch(
        key, epoch=0,
        initial_sim_state=initial_sim_state,
        initial_msg_history=initial_msg_history
    )
    elapsed = time.time() - start_time

    print(f"  ✓ Epoch completed in {elapsed:.1f}s")
    print(f"    Mean fitness: {float(mean_fitness):.6f}")
    print(f"    Fitnesses shape: {fitnesses.shape}")
    print(f"    Fitness std: {float(jnp.std(fitnesses)):.6f}")

    # Verify results
    print("\n[5/5] Verifying results...")

    # 1. Check fitnesses shape
    assert fitnesses.shape == (config.n_threads,), \
        f"Expected shape ({config.n_threads},), got {fitnesses.shape}"
    print(f"  ✓ Fitnesses shape correct: {fitnesses.shape}")

    # 2. Check no NaN in fitnesses
    assert not jnp.any(jnp.isnan(fitnesses)), "Found NaN in fitnesses"
    print("  ✓ No NaN in fitnesses")

    # 3. Check mean fitness is finite
    assert jnp.isfinite(mean_fitness), f"Mean fitness {mean_fitness} is not finite"
    print(f"  ✓ Mean fitness is finite: {float(mean_fitness):.6f}")

    # 4. Check info dict
    assert 'pnl' in info, "Missing 'pnl' in info"
    print(f"  ✓ Info keys: {list(info.keys())}")
    print(f"    Mean PnL: {float(info['pnl']):.6f}")
    print(f"    Mean agent_quantity: {float(info['agent_quantity']):.2f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ C2: World Model 单 Epoch 训练通过")
    print(f"   mean_fitness={float(mean_fitness):.4f}, time={elapsed:.1f}s")
    print("=" * 60)

    return mean_fitness, fitnesses, info


def main():
    """Main entry point"""
    mean_fitness, fitnesses, info = test_single_epoch_world_model()
    return 0


if __name__ == "__main__":
    sys.exit(main())
