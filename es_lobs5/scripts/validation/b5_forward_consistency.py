#!/usr/bin/env python3
"""
B5: Forward Pass 一致性验证 (恢复后)
目标: 验证恢复后 forward pass 输出一致
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import shutil


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
SAVE_PATH = "/tmp/es_validation_b5_checkpoint"


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
    output_dir: str = '/tmp/es_validation_b5'


def test_forward_consistency():
    """验证恢复后 forward pass 输出一致"""
    from es_lobs5.training.es_trainer import ESTrainer
    from es_lobs5.models.lob_model import ES_PaddedLobPredModel

    print("=" * 60)
    print("B5: Forward Pass 一致性验证 (恢复后)")
    print("=" * 60)

    # Clean up previous test
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        print(f"  Cleaned up previous test directory: {SAVE_PATH}")

    # Create config
    print("\n[1/6] Creating ESConfig...")
    config = ESConfig()
    print("  ✓ Config created")

    # Initialize trainer 1
    print("\n[2/6] Initializing ESTrainer #1 and training...")
    trainer1 = ESTrainer(config)
    initial_sim_state, initial_msg_history = trainer1._create_initial_sim_state()
    key = jax.random.PRNGKey(config.seed)
    mean_fitness1, _, _ = trainer1.train_epoch(
        key, epoch=0,
        initial_sim_state=initial_sim_state,
        initial_msg_history=initial_msg_history
    )
    print(f"  ✓ ESTrainer #1 trained, mean_fitness={float(mean_fitness1):.6f}")

    # Save checkpoint
    trainer1.save_checkpoint(SAVE_PATH)
    print(f"  ✓ Checkpoint saved")

    # Create dummy input for forward pass
    print("\n[3/6] Creating test input...")
    fp = trainer1.lobs5_init.frozen_params
    msg_seq_len = fp.get('msg_seq_len', 500)
    book_depth = fp.get('book_depth', 500)
    d_book = 503

    x_m = jnp.zeros((msg_seq_len,), dtype=jnp.int32)
    x_b = jnp.zeros((book_depth, d_book), dtype=jnp.float32)
    print(f"  x_m shape: {x_m.shape}")
    print(f"  x_b shape: {x_b.shape}")

    # Forward pass with trainer 1 (before restore)
    print("\n[4/6] Running forward pass with trainer #1 (before restore)...")
    common_params_1 = trainer1.create_world_common_params()  # frozen, no noise
    logits_before = ES_PaddedLobPredModel._forward(common_params_1, x_m, x_b)
    print(f"  ✓ logits_before shape: {logits_before.shape}")
    print(f"  ✓ logits_before mean: {float(jnp.mean(logits_before)):.6f}")

    # Initialize trainer 2 and restore
    print("\n[5/6] Initializing ESTrainer #2 and restoring...")
    trainer2 = ESTrainer(config)
    trainer2.load_checkpoint(SAVE_PATH)
    print("  ✓ ESTrainer #2 restored from checkpoint")

    # Forward pass with trainer 2 (after restore)
    common_params_2 = trainer2.create_world_common_params()
    logits_after = ES_PaddedLobPredModel._forward(common_params_2, x_m, x_b)
    print(f"  ✓ logits_after shape: {logits_after.shape}")
    print(f"  ✓ logits_after mean: {float(jnp.mean(logits_after)):.6f}")

    # Verify consistency
    print("\n[6/6] Verifying forward pass consistency...")

    # Check shapes match
    assert logits_before.shape == logits_after.shape, \
        f"Shape mismatch: {logits_before.shape} vs {logits_after.shape}"
    print(f"  ✓ Shapes match: {logits_before.shape}")

    # Check values match
    max_diff = float(jnp.max(jnp.abs(logits_before - logits_after)))
    mean_diff = float(jnp.mean(jnp.abs(logits_before - logits_after)))
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")

    assert jnp.allclose(logits_before, logits_after, rtol=1e-5, atol=1e-6), \
        f"Logits don't match: max_diff={max_diff}"
    print(f"  ✓ Forward pass outputs match (max_diff={max_diff:.10f})")

    # Also test with ES noise (policy mode)
    print("\n  Testing with ES noise (policy mode)...")
    common_params_noise_1 = trainer1.create_policy_common_params(epoch=0, thread_id=0)
    common_params_noise_2 = trainer2.create_policy_common_params(epoch=0, thread_id=0)

    logits_noise_1 = ES_PaddedLobPredModel._forward(common_params_noise_1, x_m, x_b)
    logits_noise_2 = ES_PaddedLobPredModel._forward(common_params_noise_2, x_m, x_b)

    max_diff_noise = float(jnp.max(jnp.abs(logits_noise_1 - logits_noise_2)))
    assert jnp.allclose(logits_noise_1, logits_noise_2, rtol=1e-5, atol=1e-6), \
        f"Noisy logits don't match: max_diff={max_diff_noise}"
    print(f"  ✓ Forward pass with ES noise match (max_diff={max_diff_noise:.10f})")

    # Summary
    print("\n" + "=" * 60)
    print("✅ B5: Forward Pass 一致性验证通过")
    print("=" * 60)

    return logits_before, logits_after


def main():
    """Main entry point"""
    logits_before, logits_after = test_forward_consistency()
    return 0


if __name__ == "__main__":
    sys.exit(main())
