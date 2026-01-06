#!/usr/bin/env python3
"""
A1: Checkpoint 加载验证
目标: 验证 Flax checkpoint 能正确转换为 ES 格式
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"


def test_checkpoint_load():
    """验证 Flax checkpoint 能正确加载并转换为 ES 格式"""
    from es_lobs5.adapters.checkpoint_adapter import load_checkpoint_for_es

    print("=" * 60)
    print("A1: Checkpoint 加载验证")
    print("=" * 60)
    print(f"Checkpoint path: {CHECKPOINT_PATH}")

    # Load checkpoint
    print("\n[1/4] Loading Flax checkpoint...")
    es_init, es_tree_key = load_checkpoint_for_es(CHECKPOINT_PATH)
    print("  ✓ Checkpoint loaded successfully")

    # Verify structure
    print("\n[2/4] Verifying ES params structure...")
    required_keys = ['fused_encoder', 'message_encoder', 'book_encoder', 'decoder']
    for key in required_keys:
        assert key in es_init.params, f"Missing key: {key}"
        print(f"  ✓ Found '{key}'")

    # Verify d_output
    print("\n[3/4] Verifying decoder dimensions...")
    d_output = es_init.params['decoder']['weight'].shape[0]
    d_model = es_init.params['decoder']['weight'].shape[1]
    print(f"  d_output (vocab size): {d_output}")
    print(f"  d_model: {d_model}")
    assert d_output == 2112, f"Expected d_output=2112 for token_mode=24, got {d_output}"
    print("  ✓ d_output matches token_mode=24 (vocab=2112)")

    # Verify frozen_params
    print("\n[4/4] Verifying frozen_params...")
    assert es_init.frozen_params is not None
    assert 'd_model' in es_init.frozen_params
    assert 'd_output' in es_init.frozen_params
    print(f"  d_model (frozen): {es_init.frozen_params['d_model']}")
    print(f"  d_output (frozen): {es_init.frozen_params['d_output']}")
    print(f"  n_layers (fused): {es_init.frozen_params.get('fused_encoder', {}).get('n_layers', 'N/A')}")
    print("  ✓ frozen_params structure valid")

    # Summary
    print("\n" + "=" * 60)
    print("✅ A1: Checkpoint 加载验证通过")
    print("=" * 60)

    # Return for use by subsequent tests
    return es_init, es_tree_key


def main():
    """Main entry point"""
    es_init, es_tree_key = test_checkpoint_load()
    return 0


if __name__ == "__main__":
    sys.exit(main())
