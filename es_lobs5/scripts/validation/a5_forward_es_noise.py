#!/usr/bin/env python3
"""
A5: Forward Pass 验证 (ES 噪声模式)
目标: 验证带 ES 扰动的 forward pass
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
    """Helper to build CommonParams"""
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
        iterinfo=None  # will be set per test
    )

    return common_params, es_init


def test_forward_with_noise():
    """验证带 ES 扰动的 forward pass"""
    from es_lobs5.models.lob_model import ES_PaddedLobPredModel

    print("=" * 60)
    print("A5: Forward Pass 验证 (ES 噪声模式)")
    print("=" * 60)

    # Build CommonParams
    print("\n[1/5] Building CommonParams...")
    common_params, es_init = build_common_params()
    print("  ✓ CommonParams built")

    # Create dummy input
    print("\n[2/5] Creating dummy input...")
    msg_seq_len = es_init.frozen_params.get('msg_seq_len', 500)
    book_depth = es_init.frozen_params.get('book_depth', 500)
    d_book = 503

    x_m = jnp.zeros((msg_seq_len,), dtype=jnp.int32)
    x_b = jnp.zeros((book_depth, d_book), dtype=jnp.float32)
    print(f"  x_m shape: {x_m.shape}")
    print(f"  x_b shape: {x_b.shape}")

    # Forward pass WITHOUT noise (inference mode)
    print("\n[3/5] Running forward pass WITHOUT noise (iterinfo=None)...")
    logits_no_noise = ES_PaddedLobPredModel._forward(common_params, x_m, x_b)
    print(f"  ✓ No-noise logits shape: {logits_no_noise.shape}")

    # Forward pass WITH noise (ES training mode)
    print("\n[4/5] Running forward pass WITH noise (iterinfo=(0, 0))...")
    common_params_noise = common_params._replace(
        iterinfo=(0, 0)  # epoch=0, thread_id=0
    )
    logits_with_noise = ES_PaddedLobPredModel._forward(common_params_noise, x_m, x_b)
    print(f"  ✓ With-noise logits shape: {logits_with_noise.shape}")

    # Verify noise changes output
    print("\n[5/5] Verifying ES noise affects output...")
    max_diff = jnp.max(jnp.abs(logits_with_noise - logits_no_noise))
    mean_diff = jnp.mean(jnp.abs(logits_with_noise - logits_no_noise))
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # ES noise should change output
    assert not jnp.allclose(logits_with_noise, logits_no_noise, rtol=1e-5), \
        "ES noise should change model output!"
    print("  ✓ ES noise successfully perturbs output")

    # Different iterinfo should give different noise
    print("\n  Testing different thread_ids...")
    common_params_noise_2 = common_params._replace(iterinfo=(0, 1))  # different thread
    logits_thread_1 = ES_PaddedLobPredModel._forward(common_params_noise_2, x_m, x_b)

    thread_diff = jnp.max(jnp.abs(logits_with_noise - logits_thread_1))
    print(f"  Max diff between thread 0 and 1: {thread_diff:.6f}")
    assert thread_diff > 0, "Different threads should have different noise"
    print("  ✓ Different threads have different perturbations")

    # Summary
    print("\n" + "=" * 60)
    print("✅ A5: Forward Pass 验证 (ES 噪声模式) 通过")
    print("=" * 60)

    return logits_no_noise, logits_with_noise


def main():
    """Main entry point"""
    logits_no_noise, logits_with_noise = test_forward_with_noise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
