#!/usr/bin/env python3
"""
C4: Checkpoint 保存/恢复验证 (World Model)
目标: 验证 World Model 模式下 checkpoint 流程
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
DATA_DIR = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
SAVE_PATH = "/tmp/es_validation_c4_checkpoint"


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

    # Training configuration (minimal)
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
    output_dir: str = '/tmp/es_validation_c4'


def test_checkpoint_world_model():
    """验证 World Model 模式下 checkpoint 保存/恢复"""
    from es_lobs5.training.es_trainer import ESTrainer
    from es_lobs5.models.lob_model import ES_PaddedLobPredModel

    print("=" * 60)
    print("C4: Checkpoint 保存/恢复验证 (World Model)")
    print("=" * 60)

    # Clean up previous test
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        print(f"  Cleaned up previous test directory: {SAVE_PATH}")

    # Create config
    print("\n[1/6] Creating ESConfig (World Model)...")
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
    print(f"  ✓ Checkpoint saved to {SAVE_PATH}")

    # Create test input
    print("\n[3/6] Creating test input...")
    fp = trainer1.lobs5_init.frozen_params
    msg_seq_len = fp.get('msg_seq_len', 500)
    book_depth = fp.get('book_depth', 500)
    d_book = 503

    x_m = jnp.zeros((msg_seq_len,), dtype=jnp.int32)
    x_b = jnp.zeros((book_depth, d_book), dtype=jnp.float32)
    print(f"  x_m shape: {x_m.shape}")
    print(f"  x_b shape: {x_b.shape}")

    # Forward pass with trainer 1
    print("\n[4/6] Running forward pass with trainer #1...")
    common_params_1 = trainer1.create_world_common_params()
    logits_before = ES_PaddedLobPredModel._forward(common_params_1, x_m, x_b)
    print(f"  ✓ logits_before shape: {logits_before.shape}")

    # Initialize trainer 2 and restore
    print("\n[5/6] Initializing ESTrainer #2 (World Model) and restoring...")
    trainer2 = ESTrainer(config)
    trainer2.load_checkpoint(SAVE_PATH)
    print("  ✓ ESTrainer #2 restored from checkpoint")

    # Forward pass with trainer 2
    common_params_2 = trainer2.create_world_common_params()
    logits_after = ES_PaddedLobPredModel._forward(common_params_2, x_m, x_b)
    print(f"  ✓ logits_after shape: {logits_after.shape}")

    # Verify consistency
    print("\n[6/6] Verifying checkpoint restore consistency...")

    # Check shapes match
    assert logits_before.shape == logits_after.shape, \
        f"Shape mismatch: {logits_before.shape} vs {logits_after.shape}"
    print(f"  ✓ Shapes match: {logits_before.shape}")

    # Check values match
    max_diff = float(jnp.max(jnp.abs(logits_before - logits_after)))
    assert jnp.allclose(logits_before, logits_after, rtol=1e-5, atol=1e-6), \
        f"Logits don't match: max_diff={max_diff}"
    print(f"  ✓ Forward pass outputs match (max_diff={max_diff:.10f})")

    # Verify checkpoint file exists
    checkpoint_file = os.path.join(SAVE_PATH, 'es_checkpoint.pkl')
    assert os.path.exists(checkpoint_file), f"Missing {checkpoint_file}"
    file_size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024)
    print(f"  ✓ Checkpoint file size: {file_size_mb:.2f} MB")

    # Summary
    print("\n" + "=" * 60)
    print("✅ C4: World Model checkpoint 保存/恢复验证通过")
    print("=" * 60)

    return logits_before, logits_after


def main():
    """Main entry point"""
    logits_before, logits_after = test_checkpoint_world_model()
    return 0


if __name__ == "__main__":
    sys.exit(main())
