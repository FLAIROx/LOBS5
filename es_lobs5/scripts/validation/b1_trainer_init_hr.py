#!/usr/bin/env python3
"""
B1: ESTrainer 初始化验证 (Historical Replay 模式)
目标: 验证 ESTrainer 能正确初始化并加载 historical replay 数据
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
    output_dir: str = '/tmp/es_validation_b1'


def test_trainer_init_historical():
    """验证 ESTrainer 能正确初始化 (Historical Replay 模式)"""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 60)
    print("B1: ESTrainer 初始化验证 (Historical Replay 模式)")
    print("=" * 60)

    # Create config
    print("\n[1/5] Creating ESConfig...")
    config = ESConfig()
    print(f"  Checkpoint: {config.lobs5_checkpoint}")
    print(f"  Background mode: {config.background_mode}")
    print(f"  Replay data path: {config.replay_data_path}")
    print(f"  Token mode: {config.token_mode}")
    print(f"  n_threads: {config.n_threads}")
    print(f"  n_steps: {config.n_steps}")
    print("  ✓ Config created")

    # Initialize trainer
    print("\n[2/5] Initializing ESTrainer...")
    trainer = ESTrainer(config)
    print("  ✓ ESTrainer initialized")

    # Verify LOBS5 checkpoint loaded
    print("\n[3/5] Verifying LOBS5 checkpoint...")
    assert trainer.lobs5_init is not None, "lobs5_init should not be None"
    assert trainer.lobs5_init.params is not None, "params should not be None"
    assert 'fused_encoder' in trainer.lobs5_init.params, "Missing fused_encoder"
    assert 'decoder' in trainer.lobs5_init.params, "Missing decoder"
    print(f"  ✓ Params keys: {list(trainer.lobs5_init.params.keys())}")
    print(f"  ✓ d_model: {trainer.lobs5_init.frozen_params.get('d_model', 'N/A')}")
    print(f"  ✓ d_output: {trainer.lobs5_init.frozen_params.get('d_output', 'N/A')}")

    # Verify noiser
    print("\n[4/5] Verifying noiser...")
    assert trainer.noiser_cls is not None, "noiser_cls should not be None"
    assert trainer.frozen_noiser_params is not None, "frozen_noiser_params should not be None"
    assert trainer.noiser_params is not None, "noiser_params should not be None"
    print(f"  ✓ Noiser class: {trainer.noiser_cls}")
    print(f"  ✓ frozen_noiser_params keys: {list(trainer.frozen_noiser_params.keys())[:5]}...")
    print(f"  ✓ noiser_params keys: {list(trainer.noiser_params.keys())[:5]}...")

    # Verify historical replay data
    print("\n[5/5] Verifying historical replay data...")
    assert trainer.replay_tokens is not None, "replay_tokens should not be None"
    assert trainer.replay_data_raw is not None, "replay_data_raw should not be None"
    print(f"  ✓ replay_tokens shape: {trainer.replay_tokens.shape}")
    print(f"  ✓ replay_data_raw shape: {trainer.replay_data_raw.shape}")

    # Verify token range
    token_min = int(jnp.min(trainer.replay_tokens))
    token_max = int(jnp.max(trainer.replay_tokens))
    d_output = trainer.lobs5_init.frozen_params.get('d_output', 2112)
    assert token_min >= 0, f"Token min {token_min} should be >= 0"
    assert token_max < d_output, f"Token max {token_max} should be < {d_output}"
    print(f"  ✓ Token range: [{token_min}, {token_max}] (vocab size: {d_output})")

    # Summary
    print("\n" + "=" * 60)
    print("✅ B1: ESTrainer 初始化验证 (Historical Replay 模式) 通过")
    print("=" * 60)

    return trainer


def main():
    """Main entry point"""
    trainer = test_trainer_init_historical()
    return 0


if __name__ == "__main__":
    sys.exit(main())
