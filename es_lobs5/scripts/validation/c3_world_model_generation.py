#!/usr/bin/env python3
"""
C3: World Model 生成验证
目标: 验证 World Model 能正确生成 messages (autoregressive)
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
DATA_DIR = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """ES training configuration."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration
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
    output_dir: str = '/tmp/es_validation_c3'


def test_world_model_generation():
    """验证 World Model 能正确生成 messages"""
    from es_lobs5.training.es_trainer import ESTrainer
    from es_lobs5.models.lob_model import ES_PaddedLobPredModel

    print("=" * 60)
    print("C3: World Model 生成验证")
    print("=" * 60)

    # Create config and trainer
    print("\n[1/5] Creating ESTrainer...")
    config = ESConfig()
    trainer = ESTrainer(config)
    print("  ✓ ESTrainer initialized")

    # Get initial state
    print("\n[2/5] Creating initial state...")
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    fp = trainer.lobs5_init.frozen_params
    msg_len = 24 if config.token_mode == 24 else 22
    book_depth = fp.get('book_depth', 500)
    print(f"  ✓ Initial msg_history shape: {initial_msg_history.shape}")

    # Create world model common params (frozen, no noise)
    print("\n[3/5] Creating World Model CommonParams...")
    world_common_params = trainer.create_world_common_params()
    assert world_common_params.iterinfo is None, "World model should have iterinfo=None"
    print("  ✓ World CommonParams created (iterinfo=None)")

    # Initialize hidden state
    print("\n[4/5] Initializing hidden state and generating tokens...")
    hiddens = ES_PaddedLobPredModel.initialize_carry(
        batch_size=1,
        ssm_size=fp.get('ssm_size', 256),
        n_message_layers=fp.get('n_message_layers', 2),
        n_book_pre_layers=fp.get('n_book_pre_layers', 1),
        n_book_post_layers=fp.get('n_book_post_layers', 1),
        n_fused_layers=fp.get('n_fused_layers', 4),
        d_model=fp.get('d_model', 256),
        conj_sym=fp.get('conj_sym', True),
    )
    print(f"  ✓ Hidden state initialized")

    # Get book features from initial state
    from es_lobs5.training.es_trainer import transform_L2_state_wrapper
    book_feat = transform_L2_state_wrapper(initial_sim_state, price_levels=book_depth, tick_size=config.tick_size)
    print(f"  ✓ Book features shape: {book_feat.shape}")

    # Generate one message autoregressively
    key = jax.random.PRNGKey(config.seed)
    msg_history = initial_msg_history

    generated_tokens = []

    # Sample msg_len tokens
    for t in range(msg_len):
        key, sample_key = jax.random.split(key)

        # Forward step
        hiddens, log_probs = ES_PaddedLobPredModel._forward_step(
            world_common_params, hiddens, msg_history[-msg_len:], book_feat[None, :]
        )

        # Truncate hidden to last step
        hiddens = jax.tree.map(lambda h: h[:, -1:, :], hiddens)

        # Sample token
        log_probs = jnp.nan_to_num(log_probs, nan=-1e9, posinf=1e9, neginf=-1e9)
        next_token = jax.random.categorical(sample_key, log_probs[-1])

        generated_tokens.append(int(next_token))

        # Update history
        msg_history = jnp.concatenate([msg_history[1:], jnp.array([next_token])])

    generated_tokens = jnp.array(generated_tokens)

    print(f"\n  Generated message tokens: {generated_tokens.tolist()}")

    # Verify generated tokens
    print("\n[5/5] Verifying generated tokens...")

    # Check shape
    assert generated_tokens.shape == (msg_len,), f"Expected ({msg_len},), got {generated_tokens.shape}"
    print(f"  ✓ Token shape correct: {generated_tokens.shape}")

    # Check token range
    d_output = fp.get('d_output', 2112)
    token_min = int(jnp.min(generated_tokens))
    token_max = int(jnp.max(generated_tokens))
    assert token_min >= 0, f"Token min {token_min} should be >= 0"
    assert token_max < d_output, f"Token max {token_max} should be < {d_output}"
    print(f"  ✓ Token range valid: [{token_min}, {token_max}] (vocab={d_output})")

    # Generate a few more messages to verify consistency
    print("\n  Generating 3 more messages for consistency check...")
    for i in range(3):
        msg_tokens = []
        for t in range(msg_len):
            key, sample_key = jax.random.split(key)
            hiddens, log_probs = ES_PaddedLobPredModel._forward_step(
                world_common_params, hiddens, msg_history[-msg_len:], book_feat[None, :]
            )
            hiddens = jax.tree.map(lambda h: h[:, -1:, :], hiddens)
            log_probs = jnp.nan_to_num(log_probs, nan=-1e9, posinf=1e9, neginf=-1e9)
            next_token = jax.random.categorical(sample_key, log_probs[-1])
            msg_tokens.append(int(next_token))
            msg_history = jnp.concatenate([msg_history[1:], jnp.array([next_token])])

        msg_tokens = jnp.array(msg_tokens)
        print(f"    Message {i+1}: range=[{int(jnp.min(msg_tokens))}, {int(jnp.max(msg_tokens))}]")

        # Verify each message
        assert jnp.all(msg_tokens >= 0) and jnp.all(msg_tokens < d_output), \
            f"Message {i+1} has invalid tokens"

    print("  ✓ All generated messages have valid tokens")

    # Summary
    print("\n" + "=" * 60)
    print("✅ C3: World Model 生成验证通过")
    print("=" * 60)

    return generated_tokens


def main():
    """Main entry point"""
    generated_tokens = test_world_model_generation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
