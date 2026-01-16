#!/usr/bin/env python3
"""
Count parameters for a single model configuration.

Usage:
    # Small models (fit in single GPU memory)
    python count_single_model.py --preset 125M
    python count_single_model.py --preset 1.9B-A153M

    # Large models (use eval_shape to avoid OOM)
    python count_single_model.py --preset 23B-A1.9B --use-eval-shape
    python count_single_model.py --preset 44B-A3.8B --use-eval-shape

The --use-eval-shape flag uses jax.eval_shape() to compute parameter shapes
without allocating any memory. This is essential for models that exceed
single GPU memory (like 23B-A1.9B which needs ~46GB, 44B-A3.8B needs ~88GB).
"""

import argparse
import os
import sys

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
MAXTEXT_PATH = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/maxtext/src'
if MAXTEXT_PATH not in sys.path:
    sys.path.insert(0, MAXTEXT_PATH)

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh

from lobmax.config import LOBMAXConfig
from lobmax.models import LOBMAXModel, count_parameters, count_active_parameters


# All configurations
CONFIGS = {
    # Dense presets
    "125M": dict(d_model=768, n_layers=12, num_heads=12, mlp_dim=3072, num_experts=1),
    "360M": dict(d_model=1024, n_layers=24, num_heads=16, mlp_dim=4096, num_experts=1),
    "1.3B": dict(d_model=2048, n_layers=24, num_heads=16, mlp_dim=8192, num_experts=1),
    "3B": dict(d_model=2560, n_layers=32, num_heads=20, mlp_dim=10240, num_experts=1),
    # MoE presets (based on Dense counterparts)
    "1.9B-A153M": dict(d_model=768, n_layers=12, num_heads=12, mlp_dim=3072, num_experts=16),
    "5.8B-A473M": dict(d_model=1024, n_layers=24, num_heads=16, mlp_dim=4096, num_experts=16),
    "23B-A1.9B": dict(d_model=2048, n_layers=24, num_heads=16, mlp_dim=8192, num_experts=16),
    "44B-A3.8B": dict(d_model=2560, n_layers=32, num_heads=20, mlp_dim=10240, num_experts=16),
}


def create_config(preset_name: str) -> LOBMAXConfig:
    """Create LOBMAXConfig from preset name."""
    cfg = CONFIGS[preset_name]
    head_dim = cfg["d_model"] // cfg["num_heads"]

    return LOBMAXConfig(
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_kv_heads=cfg["num_heads"],
        head_dim=head_dim,
        mlp_dim=cfg["mlp_dim"],
        n_layers=cfg["n_layers"],
        n_message_layers=2,
        n_book_pre_layers=1,
        n_book_post_layers=1,
        vocab_size=128,
        d_book=40,
        n_classes=128,
        max_target_length=12256,
        attention="dot_product",
        rope_type="llama3.1",
        dtype="float32",
        weight_dtype="float32",
        num_experts=cfg["num_experts"],
        num_experts_per_tok=1,
        megablox=False,
    )


def measure_parameters(preset_name: str, use_eval_shape: bool = False):
    """
    Measure parameters for a single preset.

    Args:
        preset_name: Model preset to measure
        use_eval_shape: If True, use jax.eval_shape to compute shapes without
                        allocating memory. Essential for large models (23B+).
    """
    import numpy as np
    cfg_dict = CONFIGS[preset_name]

    print("=" * 70)
    print(f"LOBMAX Parameter Counter - {preset_name}")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX platform: {jax.default_backend()}")
    print(f"Mode: {'eval_shape (zero memory)' if use_eval_shape else 'full init'}")
    print()
    print(f"Configuration:")
    print(f"  d_model    = {cfg_dict['d_model']}")
    print(f"  n_layers   = {cfg_dict['n_layers']}")
    print(f"  num_heads  = {cfg_dict['num_heads']}")
    print(f"  mlp_dim    = {cfg_dict['mlp_dim']}")
    print(f"  num_experts = {cfg_dict['num_experts']}")
    print()

    # Create config
    config = create_config(preset_name)

    # Create mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data',))

    # Create model
    model = LOBMAXModel(config=config, mesh=mesh, training=True)

    # Initialize with minimal dummy inputs
    key = random.PRNGKey(42)
    seq_len = 256
    batch_size = 1

    dummy_msg = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_book = jnp.zeros((batch_size, seq_len, config.d_book), dtype=jnp.float32)

    if use_eval_shape:
        # Use jax.eval_shape: computes shapes WITHOUT allocating memory
        # Returns ShapeDtypeStruct objects instead of actual arrays
        print("Computing parameter shapes (zero memory allocation)...")

        def init_fn():
            return model.init(key, x_m=dummy_msg, x_b=dummy_book)

        abstract_variables = jax.eval_shape(init_fn)
        params = abstract_variables['params']
        print("  -> Shape computation complete (0 FLOPs, 0 bytes allocated)")
    else:
        # Full initialization: allocates memory
        print("Initializing model (full memory allocation)...")
        variables = model.init(key, x_m=dummy_msg, x_b=dummy_book)
        params = variables['params']

    # Count parameters
    total_params = count_parameters(params)
    active_params = count_active_parameters(params, config.num_experts, 1)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Preset:        {preset_name}")
    print(f"Total params:  {total_params:,} ({total_params/1e9:.3f}B)")
    print(f"Active params: {active_params:,} ({active_params/1e9:.3f}B)")
    print(f"Ratio:         {total_params/active_params:.2f}x")
    print("=" * 70)

    # Output in easily parseable format
    print()
    print("### PARSEABLE OUTPUT ###")
    print(f"PRESET={preset_name}")
    print(f"TOTAL={total_params}")
    print(f"ACTIVE={active_params}")
    print(f"TOTAL_B={total_params/1e9:.3f}")
    print(f"ACTIVE_B={active_params/1e9:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Count parameters for a single LOBMAX model")
    parser.add_argument("--preset", type=str, required=True,
                        choices=list(CONFIGS.keys()),
                        help="Model preset to measure")
    parser.add_argument("--use-eval-shape", action="store_true",
                        help="Use jax.eval_shape for zero-memory parameter counting. "
                             "Essential for large models (23B+) that would OOM on single GPU.")
    args = parser.parse_args()

    measure_parameters(args.preset, use_eval_shape=args.use_eval_shape)


if __name__ == "__main__":
    main()
