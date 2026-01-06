#!/usr/bin/env python3
"""
C1: ESTrainer 初始化验证 (World Model 模式)
目标: 验证 World Model 模式下 ESTrainer 初始化
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
# World model mode doesn't require replay_data_path, but needs data_dir for initial state
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

    # Training configuration (minimal for testing)
    n_threads: int = 4  # Smaller for world model mode (slower)
    n_epochs: int = 1
    n_steps: int = 5  # Fewer steps for world model (AR generation is slow)
    world_msgs_per_step: int = 2  # Fewer background messages

    # Token mode (24 matches the checkpoint)
    token_mode: int = 24

    # Background mode: world_model (autoregressive generation)
    background_mode: str = 'world_model'
    replay_data_path: str = None
    data_dir: str = DATA_DIR

    # Task configuration
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100

    # Other
    seed: int = 42
    output_dir: str = '/tmp/es_validation_c1'


def test_trainer_init_world_model():
    """验证 ESTrainer 能正确初始化 (World Model 模式)"""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 60)
    print("C1: ESTrainer 初始化验证 (World Model 模式)")
    print("=" * 60)

    # Create config
    print("\n[1/5] Creating ESConfig (World Model)...")
    config = ESConfig()
    print(f"  Checkpoint: {config.lobs5_checkpoint}")
    print(f"  Background mode: {config.background_mode}")
    print(f"  Data dir (for initial state): {config.data_dir}")
    print(f"  Token mode: {config.token_mode}")
    print(f"  n_threads: {config.n_threads}")
    print(f"  n_steps: {config.n_steps}")
    print("  ✓ Config created")

    # Initialize trainer
    print("\n[2/5] Initializing ESTrainer (World Model mode)...")
    trainer = ESTrainer(config)
    print("  ✓ ESTrainer initialized")

    # Verify LOBS5 checkpoint loaded
    print("\n[3/5] Verifying LOBS5 checkpoint...")
    assert trainer.lobs5_init is not None, "lobs5_init should not be None"
    assert trainer.lobs5_init.params is not None, "params should not be None"
    assert 'fused_encoder' in trainer.lobs5_init.params, "Missing fused_encoder"
    assert 'decoder' in trainer.lobs5_init.params, "Missing decoder"
    print(f"  ✓ Params keys: {list(trainer.lobs5_init.params.keys())}")

    # Verify noiser
    print("\n[4/5] Verifying noiser...")
    assert trainer.noiser_cls is not None, "noiser_cls should not be None"
    assert trainer.frozen_noiser_params is not None, "frozen_noiser_params should not be None"
    assert trainer.noiser_params is not None, "noiser_params should not be None"
    print(f"  ✓ Noiser configured")

    # Verify world model mode - no replay data
    print("\n[5/5] Verifying World Model mode specifics...")
    # In world model mode, replay_tokens should be None
    assert trainer.replay_tokens is None, "replay_tokens should be None in world_model mode"
    assert trainer.replay_data_raw is None, "replay_data_raw should be None in world_model mode"
    print("  ✓ No replay data (world model mode)")

    # Verify JaxLOB simulator
    assert trainer.sim is not None, "JaxLOB simulator should be initialized"
    print(f"  ✓ JaxLOB simulator initialized")

    # Verify CommonParams can be created
    world_common_params = trainer.create_world_common_params()
    assert world_common_params.iterinfo is None, "World model should have iterinfo=None"
    print("  ✓ World CommonParams created (iterinfo=None)")

    policy_common_params = trainer.create_policy_common_params(epoch=0, thread_id=0)
    assert policy_common_params.iterinfo is not None, "Policy should have iterinfo set"
    print("  ✓ Policy CommonParams created (iterinfo=(0, 0))")

    # Summary
    print("\n" + "=" * 60)
    print("✅ C1: ESTrainer 初始化验证 (World Model 模式) 通过")
    print("=" * 60)

    return trainer


def main():
    """Main entry point"""
    trainer = test_trainer_init_world_model()
    return 0


if __name__ == "__main__":
    sys.exit(main())
