#!/usr/bin/env python3
"""
B6: 中等规模训练验证 (Historical Replay, 32×50)
目标: 验证中等规模训练稳定性 (3 epochs)
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
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """Medium-scale ES training configuration."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration (medium scale)
    n_threads: int = 32
    n_epochs: int = 3
    n_steps: int = 50
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
    output_dir: str = '/tmp/es_validation_b6'


def test_medium_scale_historical():
    """验证中等规模训练稳定性"""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 60)
    print("B6: 中等规模训练验证 (Historical Replay, 32×50)")
    print("=" * 60)

    # Create config
    print("\n[1/4] Creating ESConfig (medium scale)...")
    config = ESConfig()
    print(f"  n_threads: {config.n_threads}")
    print(f"  n_steps: {config.n_steps}")
    print(f"  n_epochs: {config.n_epochs}")
    print("  ✓ Config created")

    # Initialize trainer
    print("\n[2/4] Initializing ESTrainer...")
    trainer = ESTrainer(config)
    print("  ✓ ESTrainer initialized")

    # Get initial state
    print("\n[3/4] Creating initial simulation state...")
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  ✓ initial_msg_history shape: {initial_msg_history.shape}")

    # Run multiple epochs
    print("\n[4/4] Running medium-scale training (3 epochs)...")
    key = jax.random.PRNGKey(config.seed)

    all_fitnesses = []
    all_infos = []

    for epoch in range(config.n_epochs):
        key, epoch_key = jax.random.split(key)

        start_time = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        elapsed = time.time() - start_time

        all_fitnesses.append(float(mean_fitness))
        all_infos.append(info)

        print(f"  Epoch {epoch}: mean_fitness={float(mean_fitness):.6f}, "
              f"std={float(jnp.std(fitnesses)):.6f}, "
              f"pnl={float(info['pnl']):.6f}, "
              f"time={elapsed:.2f}s")

    # Verify results
    print("\n  Verifying training stability...")

    # 1. Check no NaN in fitnesses
    assert not any(jnp.isnan(f) for f in all_fitnesses), "Found NaN in fitnesses"
    print("  ✓ No NaN in fitnesses across all epochs")

    # 2. Check fitness is finite
    assert all(jnp.isfinite(f) for f in all_fitnesses), "Found Inf in fitnesses"
    print("  ✓ All fitnesses are finite")

    # 3. Check fitness is not exploding
    fitness_range = max(all_fitnesses) - min(all_fitnesses)
    assert fitness_range < 100, f"Fitness range too large: {fitness_range}"
    print(f"  ✓ Fitness range reasonable: {fitness_range:.6f}")

    # 4. Check info metrics
    for i, info in enumerate(all_infos):
        assert not jnp.isnan(info['pnl']), f"Epoch {i}: NaN PnL"
        assert jnp.isfinite(info['agent_quantity']), f"Epoch {i}: Inf agent_quantity"
    print("  ✓ All epoch metrics are valid")

    # Summary statistics
    print("\n  Training Summary:")
    print(f"    Final fitness: {all_fitnesses[-1]:.6f}")
    print(f"    Fitness trend: {all_fitnesses[0]:.6f} → {all_fitnesses[-1]:.6f}")
    print(f"    Mean PnL (last epoch): {float(all_infos[-1]['pnl']):.6f}")
    print(f"    Mean agent_quantity (last epoch): {float(all_infos[-1]['agent_quantity']):.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ B6: 中等规模训练验证 (32×50) 通过")
    print("=" * 60)

    return all_fitnesses, all_infos


def main():
    """Main entry point"""
    all_fitnesses, all_infos = test_medium_scale_historical()
    return 0


if __name__ == "__main__":
    sys.exit(main())
