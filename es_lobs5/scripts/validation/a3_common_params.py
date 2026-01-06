#!/usr/bin/env python3
"""
A3: CommonParams 构建验证
目标: 验证 CommonParams 能正确构建
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"


def test_common_params():
    """验证 CommonParams 能正确构建"""
    from es_lobs5.adapters.checkpoint_adapter import load_checkpoint_for_es
    from es_lobs5.utils.import_utils import get_all_noisers
    from es_lobs5.models.common import CommonParams

    print("=" * 60)
    print("A3: CommonParams 构建验证")
    print("=" * 60)

    # Load checkpoint
    print("\n[1/4] Loading checkpoint...")
    es_init, es_tree_key = load_checkpoint_for_es(CHECKPOINT_PATH)
    print("  ✓ Checkpoint loaded")

    # Load noiser
    print("\n[2/4] Loading EGGROLL noiser...")
    noiser = get_all_noisers()['eggroll']
    print("  ✓ Noiser loaded")

    # Build frozen_noiser_params and noiser_params via init_noiser
    # Note: solver=None defaults to optax.sgd, called as solver(lr)
    print("\n[3/4] Building noiser params via init_noiser...")
    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params,
        sigma=0.01,
        lr=0.001,
        rank=4,
        freeze_nonlora=False,
        noise_reuse=0,
        solver=None,  # defaults to optax.sgd
    )
    print(f"  ✓ frozen_noiser_params built (type: {type(frozen_noiser_params).__name__})")
    print(f"  ✓ noiser_params initialized (type: {type(noiser_params).__name__})")

    # Build CommonParams
    print("\n[4/4] Building CommonParams...")
    common_params = CommonParams(
        params=es_init.params,
        frozen_params=es_init.frozen_params,
        noiser=noiser,
        frozen_noiser_params=frozen_noiser_params,
        noiser_params=noiser_params,
        es_tree_key=es_tree_key,
        iterinfo=None  # inference mode (no noise)
    )

    # Verify CommonParams
    assert common_params.params is not None
    assert common_params.frozen_params is not None
    assert common_params.noiser is not None
    assert common_params.iterinfo is None  # inference mode
    print("  ✓ CommonParams built successfully")
    print(f"    - params keys: {list(common_params.params.keys())}")
    print(f"    - iterinfo: {common_params.iterinfo} (inference mode)")

    # Summary
    print("\n" + "=" * 60)
    print("✅ A3: CommonParams 构建验证通过")
    print("=" * 60)

    return common_params, es_init, es_tree_key, noiser


def main():
    """Main entry point"""
    common_params, es_init, es_tree_key, noiser = test_common_params()
    return 0


if __name__ == "__main__":
    sys.exit(main())
