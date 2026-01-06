#!/usr/bin/env python3
"""
B2: 单 Epoch 训练验证 (Historical Replay, 最小规模)
目标: 验证能完成单 epoch ES 训练 (n_threads=8, n_steps=10)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
from dataclasses import dataclass


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """Minimal ES training configuration for testing."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration (minimal for testing)
    n_threads: int = 8
    n_epochs: int = 1
    n_steps: int = 10
    world_msgs_per_step: int = 5

    # Token mode (24 matches the checkpoint)
    token_mode: int = 24

    # Background mode: historical_replay
    background_mode: str = 'historical_replay'
    replay_data_path: str = REPLAY_DATA_PATH
    data_dir: str = REPLAY_DATA_PATH

    # Task configuration
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100

    # Other
    seed: int = 42
    output_dir: str = '/tmp/es_validation_b2'


def test_single_epoch_historical():
    """验证能完成单 epoch ES 训练"""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 60)
    print("B2: 单 Epoch 训练验证 (Historical Replay, 8×10)")
    print("=" * 60)

    # Create config
    print("\n[1/5] Creating ESConfig...")
    config = ESConfig()
    print(f"  n_threads: {config.n_threads}")
    print(f"  n_steps: {config.n_steps}")
    print(f"  world_msgs_per_step: {config.world_msgs_per_step}")
    print("  ✓ Config created")

    # Initialize trainer
    print("\n[2/5] Initializing ESTrainer...")
    trainer = ESTrainer(config)
    print("  ✓ ESTrainer initialized")

    # Get initial state
    print("\n[3/5] Creating initial simulation state...")
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  ✓ initial_msg_history shape: {initial_msg_history.shape}")

    # Run single epoch
    print("\n[4/5] Running single epoch...")
    key = jax.random.PRNGKey(config.seed)
    mean_fitness, fitnesses, info = trainer.train_epoch(
        key, epoch=0,
        initial_sim_state=initial_sim_state,
        initial_msg_history=initial_msg_history
    )
    print(f"  ✓ Epoch completed")
    print(f"    Mean fitness: {float(mean_fitness):.6f}")
    print(f"    Fitnesses shape: {fitnesses.shape}")
    print(f"    Fitness std: {float(jnp.std(fitnesses)):.6f}")
    print(f"    Fitness range: [{float(jnp.min(fitnesses)):.6f}, {float(jnp.max(fitnesses)):.6f}]")

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
    assert 'agent_quantity' in info, "Missing 'agent_quantity' in info"
    assert 'agent_trades' in info, "Missing 'agent_trades' in info"
    print(f"  ✓ Info keys: {list(info.keys())}")
    print(f"    Mean PnL: {float(info['pnl']):.6f}")
    print(f"    Mean agent_quantity: {float(info['agent_quantity']):.2f}")
    print(f"    Mean agent_trades: {float(info['agent_trades']):.2f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ B2: 单 Epoch 训练验证通过")
    print(f"   mean_fitness={float(mean_fitness):.4f}")
    print("=" * 60)

    return mean_fitness, fitnesses, info


def main():
    """Main entry point"""
    mean_fitness, fitnesses, info = test_single_epoch_historical()
    return 0


if __name__ == "__main__":
    sys.exit(main())
