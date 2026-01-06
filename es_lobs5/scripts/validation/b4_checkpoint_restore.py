#!/usr/bin/env python3
"""
B4: Checkpoint 恢复验证 (Historical Replay)
目标: 验证 ES checkpoint 能正确恢复
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
SAVE_PATH = "/tmp/es_validation_b4_checkpoint"


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
    output_dir: str = '/tmp/es_validation_b4'


def test_checkpoint_restore():
    """验证 ES checkpoint 能正确恢复"""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 60)
    print("B4: Checkpoint 恢复验证 (Historical Replay)")
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
    print("\n[2/6] Initializing ESTrainer #1...")
    trainer1 = ESTrainer(config)
    print("  ✓ ESTrainer #1 initialized")

    # Get initial state and run one epoch
    print("\n[3/6] Running one epoch to modify params...")
    initial_sim_state, initial_msg_history = trainer1._create_initial_sim_state()
    key = jax.random.PRNGKey(config.seed)
    mean_fitness1, _, _ = trainer1.train_epoch(
        key, epoch=0,
        initial_sim_state=initial_sim_state,
        initial_msg_history=initial_msg_history
    )
    print(f"  ✓ Epoch completed, mean_fitness={float(mean_fitness1):.6f}")

    # Save params before checkpoint
    params_before_save = jax.tree.map(lambda x: jnp.array(x), trainer1.lobs5_init.params)

    # Save checkpoint
    print("\n[4/6] Saving checkpoint...")
    trainer1.save_checkpoint(SAVE_PATH)
    print(f"  ✓ Checkpoint saved to {SAVE_PATH}")

    # Initialize trainer 2 (fresh)
    print("\n[5/6] Initializing ESTrainer #2 and restoring...")
    trainer2 = ESTrainer(config)
    print("  ✓ ESTrainer #2 initialized (fresh)")

    # Verify params are different before restore
    # (trainer2 has fresh params from checkpoint, trainer1 has trained params)
    decoder_weight_1 = trainer1.lobs5_init.params['decoder']['weight']
    decoder_weight_2 = trainer2.lobs5_init.params['decoder']['weight']
    print(f"  Before restore - decoder weight diff: {float(jnp.max(jnp.abs(decoder_weight_1 - decoder_weight_2))):.8f}")

    # Restore checkpoint
    trainer2.load_checkpoint(SAVE_PATH)
    print("  ✓ Checkpoint restored to ESTrainer #2")

    # Verify params match
    print("\n[6/6] Verifying params match after restore...")

    def check_params_match(params1, params2, prefix=''):
        """Recursively check if params match."""
        if isinstance(params1, dict):
            for key in params1:
                if key not in params2:
                    return False, f"{prefix}{key} missing in params2"
                match, msg = check_params_match(params1[key], params2[key], f"{prefix}{key}.")
                if not match:
                    return False, msg
            return True, "All keys match"
        else:
            if not jnp.allclose(params1, params2, rtol=1e-5, atol=1e-8):
                max_diff = float(jnp.max(jnp.abs(params1 - params2)))
                return False, f"{prefix}max_diff={max_diff:.10f}"
            return True, "Match"

    match, msg = check_params_match(params_before_save, trainer2.lobs5_init.params)
    assert match, f"Params mismatch: {msg}"
    print(f"  ✓ All params match after restore")

    # Verify specific params
    decoder_weight_restored = trainer2.lobs5_init.params['decoder']['weight']
    assert jnp.allclose(decoder_weight_1, decoder_weight_restored, rtol=1e-5), \
        "Decoder weights don't match after restore"
    print(f"  ✓ Decoder weights match (shape: {decoder_weight_restored.shape})")

    # Verify noiser_params restored
    # Note: noiser_params may contain tuples (e.g., optax opt_state), so we use tree comparison
    assert trainer2.noiser_params is not None, "noiser_params should be restored"

    def compare_trees(tree1, tree2):
        """Recursively compare pytrees, handling tuples and arrays."""
        leaves1 = jax.tree.leaves(tree1)
        leaves2 = jax.tree.leaves(tree2)
        if len(leaves1) != len(leaves2):
            return False
        for l1, l2 in zip(leaves1, leaves2):
            if hasattr(l1, 'shape'):  # It's an array
                if not jnp.allclose(l1, l2, rtol=1e-5):
                    return False
            elif l1 != l2:  # Scalar comparison
                return False
        return True

    noiser_match = compare_trees(trainer1.noiser_params, trainer2.noiser_params)
    assert noiser_match, "noiser_params don't match after restore"
    print(f"  ✓ noiser_params match")

    # Summary
    print("\n" + "=" * 60)
    print("✅ B4: Checkpoint 恢复验证通过")
    print("=" * 60)

    return trainer1, trainer2


def main():
    """Main entry point"""
    trainer1, trainer2 = test_checkpoint_restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
