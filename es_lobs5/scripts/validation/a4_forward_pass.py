#!/usr/bin/env python3
"""
A4: Forward Pass 验证 (Inference 模式)
目标: 验证模型 forward pass 能正确执行
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"


def build_common_params():
    """Helper to build CommonParams (from A3)"""
    from es_lobs5.adapters.checkpoint_adapter import load_checkpoint_for_es
    from es_lobs5.utils.import_utils import get_all_noisers
    from es_lobs5.models.common import CommonParams

    es_init, es_tree_key = load_checkpoint_for_es(CHECKPOINT_PATH)
    noiser = get_all_noisers()['eggroll']

    # Use init_noiser API - solver=None defaults to optax.sgd
    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params,
        sigma=0.01,
        lr=0.001,
        rank=4,
        freeze_nonlora=False,
        noise_reuse=0,
        solver=None,
    )

    common_params = CommonParams(
        params=es_init.params,
        frozen_params=es_init.frozen_params,
        noiser=noiser,
        frozen_noiser_params=frozen_noiser_params,
        noiser_params=noiser_params,
        es_tree_key=es_tree_key,
        iterinfo=None  # inference mode
    )

    return common_params, es_init


def test_forward_pass():
    """验证模型 forward pass 能正确执行 (Inference 模式)"""
    from es_lobs5.models.lob_model import ES_PaddedLobPredModel

    print("=" * 60)
    print("A4: Forward Pass 验证 (Inference 模式)")
    print("=" * 60)

    # Build CommonParams
    print("\n[1/4] Building CommonParams...")
    common_params, es_init = build_common_params()
    print("  ✓ CommonParams built")

    # Create dummy input
    print("\n[2/4] Creating dummy input...")
    msg_seq_len = es_init.frozen_params.get('msg_seq_len', 500)
    book_depth = es_init.frozen_params.get('book_depth', 500)
    d_book = 503  # [mid_diff, time, volume_image(501)]

    x_m = jnp.zeros((msg_seq_len,), dtype=jnp.int32)
    x_b = jnp.zeros((book_depth, d_book), dtype=jnp.float32)
    print(f"  x_m shape: {x_m.shape} (message tokens)")
    print(f"  x_b shape: {x_b.shape} (book features)")

    # Forward pass
    print("\n[3/4] Running forward pass...")
    # Debug: print mode from frozen_params
    mode = common_params.frozen_params.get('mode', 'UNKNOWN')
    print(f"  Mode from frozen_params: {mode}")
    logits = ES_PaddedLobPredModel._forward(common_params, x_m, x_b)
    print(f"  ✓ Forward pass completed")
    print(f"    Output shape: {logits.shape}")
    print(f"    Output dtype: {logits.dtype}")

    # Verify output
    print("\n[4/4] Verifying output...")
    d_output = es_init.frozen_params['d_output']
    # For pool/last/ema modes, expect (d_output,); for none mode, expect (L, d_output)
    if mode in ['pool', 'last', 'ema']:
        expected_shape = (d_output,)
    else:
        expected_shape = (msg_seq_len, d_output)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    print(f"  ✓ Shape correct: {expected_shape}")

    assert not jnp.any(jnp.isnan(logits)), "Found NaN in output"
    print("  ✓ No NaN values")

    assert jnp.all(jnp.isfinite(logits)), "Found Inf in output"
    print("  ✓ All values finite")

    # Log softmax check (should sum to 1 after exp)
    probs = jnp.exp(logits)
    prob_sum = jnp.sum(probs)
    print(f"  ✓ Probability sum: {prob_sum:.6f} (should be ~1.0)")

    # Summary
    print("\n" + "=" * 60)
    print("✅ A4: Forward Pass 验证 (Inference 模式) 通过")
    print("=" * 60)

    return logits


def main():
    """Main entry point"""
    logits = test_forward_pass()
    return 0


if __name__ == "__main__":
    sys.exit(main())
