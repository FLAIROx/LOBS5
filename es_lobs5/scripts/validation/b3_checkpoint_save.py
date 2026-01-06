#!/usr/bin/env python3
"""
B3: Checkpoint 保存验证 (Historical Replay)
目标: 验证 ES checkpoint 能正确保存
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
SAVE_PATH = "/tmp/es_validation_b3_checkpoint"


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
    output_dir: str = '/tmp/es_validation_b3'


def test_checkpoint_save():
    """验证 ES checkpoint 能正确保存"""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 60)
    print("B3: Checkpoint 保存验证 (Historical Replay)")
    print("=" * 60)

    # Clean up previous test
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        print(f"  Cleaned up previous test directory: {SAVE_PATH}")

    # Create config
    print("\n[1/5] Creating ESConfig...")
    config = ESConfig()
    print("  ✓ Config created")

    # Initialize trainer
    print("\n[2/5] Initializing ESTrainer...")
    trainer = ESTrainer(config)
    print("  ✓ ESTrainer initialized")

    # Get initial state and run one epoch
    print("\n[3/5] Running one epoch before saving...")
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    key = jax.random.PRNGKey(config.seed)
    mean_fitness, fitnesses, info = trainer.train_epoch(
        key, epoch=0,
        initial_sim_state=initial_sim_state,
        initial_msg_history=initial_msg_history
    )
    print(f"  ✓ Epoch completed, mean_fitness={float(mean_fitness):.6f}")

    # Record params before save for later comparison
    params_before_save = jax.tree.map(lambda x: x.copy(), trainer.lobs5_init.params)

    # Save checkpoint
    print("\n[4/5] Saving checkpoint...")
    trainer.save_checkpoint(SAVE_PATH)
    print(f"  ✓ Checkpoint saved to {SAVE_PATH}")

    # Verify checkpoint files
    print("\n[5/5] Verifying checkpoint files...")

    # Check es_checkpoint.pkl
    checkpoint_file = os.path.join(SAVE_PATH, 'es_checkpoint.pkl')
    assert os.path.exists(checkpoint_file), f"Missing {checkpoint_file}"
    print(f"  ✓ Found: es_checkpoint.pkl")

    # Load and verify checkpoint contents
    import pickle
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)

    assert 'params' in checkpoint, "Missing 'params' in checkpoint"
    assert 'frozen_params' in checkpoint, "Missing 'frozen_params' in checkpoint"
    assert 'noiser_params' in checkpoint, "Missing 'noiser_params' in checkpoint"
    assert 'config' in checkpoint, "Missing 'config' in checkpoint"
    print(f"  ✓ Checkpoint contains: {list(checkpoint.keys())}")

    # Verify params match
    params_in_checkpoint = checkpoint['params']
    assert 'fused_encoder' in params_in_checkpoint, "Missing fused_encoder in saved params"
    assert 'decoder' in params_in_checkpoint, "Missing decoder in saved params"
    print(f"  ✓ Saved params keys: {list(params_in_checkpoint.keys())}")

    # Verify frozen_params
    frozen_params = checkpoint['frozen_params']
    assert 'd_model' in frozen_params, "Missing d_model in frozen_params"
    assert 'd_output' in frozen_params, "Missing d_output in frozen_params"
    print(f"  ✓ frozen_params d_model: {frozen_params['d_model']}")
    print(f"  ✓ frozen_params d_output: {frozen_params['d_output']}")

    # Verify config saved
    saved_config = checkpoint['config']
    assert saved_config['noiser'] == config.noiser
    assert saved_config['token_mode'] == config.token_mode
    print(f"  ✓ Config saved correctly (noiser={saved_config['noiser']}, token_mode={saved_config['token_mode']})")

    # Get file size
    file_size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024)
    print(f"  ✓ Checkpoint file size: {file_size_mb:.2f} MB")

    # Summary
    print("\n" + "=" * 60)
    print("✅ B3: Checkpoint 保存验证通过")
    print(f"   Checkpoint saved to: {SAVE_PATH}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print("=" * 60)

    return SAVE_PATH, params_before_save


def main():
    """Main entry point"""
    save_path, params_before_save = test_checkpoint_save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
