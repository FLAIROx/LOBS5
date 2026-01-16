#!/usr/bin/env python3
"""
Count Model Parameters for LOBMAX Configurations

This script initializes each model configuration and measures the actual
parameter counts using count_parameters() and count_active_parameters().

Usage:
    python count_model_params.py

Or via sbatch:
    sbatch count_params.batch
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Add paths
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
MAXTEXT_PATH = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/maxtext/src'
if MAXTEXT_PATH not in sys.path:
    sys.path.insert(0, MAXTEXT_PATH)

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh

# Force CPU for parameter counting (no GPU needed)
# Comment out for GPU to verify
# os.environ['JAX_PLATFORMS'] = 'cpu'

from lobmax.config import LOBMAXConfig
from lobmax.models import LOBMAXModel, count_parameters, count_active_parameters, get_model_summary


@dataclass
class ModelPreset:
    """Model preset configuration."""
    name: str
    d_model: int
    n_layers: int
    num_heads: int
    mlp_dim: int
    num_experts: int = 1
    num_experts_per_tok: int = 1


# Define all configurations to measure
PRESETS: List[ModelPreset] = [
    # Dense presets
    ModelPreset("125M", d_model=768, n_layers=12, num_heads=12, mlp_dim=3072),
    ModelPreset("360M", d_model=1024, n_layers=24, num_heads=16, mlp_dim=4096),
    ModelPreset("1.3B", d_model=2048, n_layers=24, num_heads=16, mlp_dim=8192),
    ModelPreset("3B", d_model=2560, n_layers=32, num_heads=20, mlp_dim=10240),

    # MoE presets (16 experts, top-1 routing) - names match measured values
    ModelPreset("1.9B-A153M", d_model=768, n_layers=12, num_heads=12, mlp_dim=3072,
                num_experts=16, num_experts_per_tok=1),
    ModelPreset("5.8B-A473M", d_model=1024, n_layers=24, num_heads=16, mlp_dim=4096,
                num_experts=16, num_experts_per_tok=1),
    ModelPreset("20B-A1.9B", d_model=2048, n_layers=24, num_heads=16, mlp_dim=8192,
                num_experts=16, num_experts_per_tok=1),
    ModelPreset("30B-A3.6B", d_model=4096, n_layers=10, num_heads=32, mlp_dim=14336,
                num_experts=16, num_experts_per_tok=1),
]


def create_config_from_preset(preset: ModelPreset) -> LOBMAXConfig:
    """Create LOBMAXConfig from preset."""
    head_dim = preset.d_model // preset.num_heads

    return LOBMAXConfig(
        d_model=preset.d_model,
        num_heads=preset.num_heads,
        num_kv_heads=preset.num_heads,
        head_dim=head_dim,
        mlp_dim=preset.mlp_dim,
        n_layers=preset.n_layers,
        n_message_layers=2,
        n_book_pre_layers=1,
        n_book_post_layers=1,
        vocab_size=128,
        d_book=40,
        n_classes=128,
        max_target_length=12256,
        attention="dot_product",  # Use dot_product for CPU compatibility
        rope_type="llama3.1",
        dtype="float32",
        weight_dtype="float32",
        num_experts=preset.num_experts,
        num_experts_per_tok=preset.num_experts_per_tok,
        megablox=False,  # Disable megablox for param counting
    )


def measure_parameters(preset: ModelPreset) -> Tuple[int, int]:
    """
    Initialize model and measure actual parameter counts.

    Returns:
        (total_params, active_params)
    """
    print(f"\n{'='*60}")
    print(f"Measuring: {preset.name}")
    print(f"  d_model={preset.d_model}, n_layers={preset.n_layers}, "
          f"num_heads={preset.num_heads}, mlp_dim={preset.mlp_dim}")
    if preset.num_experts > 1:
        print(f"  MoE: {preset.num_experts} experts, top-{preset.num_experts_per_tok}")
    print(f"{'='*60}")

    # Create config
    config = create_config_from_preset(preset)

    # Create mesh (single device for parameter counting)
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data',))

    # Create model (training=True to avoid KV cache allocation)
    model = LOBMAXModel(config=config, mesh=mesh, training=True)

    # Initialize with minimal dummy inputs
    key = random.PRNGKey(42)
    seq_len = 256  # Small seq_len for fast init
    batch_size = 1

    dummy_msg = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_book = jnp.zeros((batch_size, seq_len, config.d_book), dtype=jnp.float32)

    print(f"  Initializing model...")
    variables = model.init(key, x_m=dummy_msg, x_b=dummy_book)
    params = variables['params']

    # Count parameters
    total_params = count_parameters(params)
    active_params = count_active_parameters(
        params,
        config.num_experts,
        config.num_experts_per_tok
    )

    print(f"  Total params:  {total_params:>15,} ({total_params/1e9:.3f}B)")
    print(f"  Active params: {active_params:>15,} ({active_params/1e9:.3f}B)")

    return total_params, active_params


def format_params(params: int) -> str:
    """Format parameter count as human-readable string."""
    if params >= 1e9:
        return f"{params/1e9:.2f}B"
    elif params >= 1e6:
        return f"{params/1e6:.0f}M"
    else:
        return f"{params/1e3:.0f}K"


def main():
    print("=" * 70)
    print("LOBMAX Model Parameter Counter")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX platform: {jax.default_backend()}")

    results = []

    for preset in PRESETS:
        total, active = measure_parameters(preset)
        results.append({
            'name': preset.name,
            'd_model': preset.d_model,
            'n_layers': preset.n_layers,
            'num_heads': preset.num_heads,
            'mlp_dim': preset.mlp_dim,
            'num_experts': preset.num_experts,
            'total_params': total,
            'active_params': active,
        })

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY: Measured Parameter Counts")
    print("=" * 90)

    # Dense models
    print("\nDense Models:")
    print("+---------------+---------+----------+-----------+---------+------------------+")
    print("| Preset        | d_model | n_layers | num_heads | mlp_dim | Measured Params  |")
    print("+---------------+---------+----------+-----------+---------+------------------+")

    for r in results:
        if r['num_experts'] == 1:
            total_str = format_params(r['total_params'])
            print(f"| {r['name']:<13} | {r['d_model']:>7} | {r['n_layers']:>8} | "
                  f"{r['num_heads']:>9} | {r['mlp_dim']:>7} | {total_str:>16} |")
    print("+---------------+---------+----------+-----------+---------+------------------+")

    # MoE models
    print("\nMoE Models (16 experts, top-1):")
    print("+---------------+---------+----------+-----------+---------+---------------------------+")
    print("| Preset        | d_model | n_layers | num_heads | mlp_dim | Params (Total/Activated)  |")
    print("+---------------+---------+----------+-----------+---------+---------------------------+")

    for r in results:
        if r['num_experts'] > 1:
            total_str = format_params(r['total_params'])
            active_str = format_params(r['active_params'])
            print(f"| {r['name']:<13} | {r['d_model']:>7} | {r['n_layers']:>8} | "
                  f"{r['num_heads']:>9} | {r['mlp_dim']:>7} | {total_str} / {active_str:<17} |")
    print("+---------------+---------+----------+-----------+---------+---------------------------+")

    # Print suggested preset renames
    print("\n" + "=" * 70)
    print("SUGGESTED PRESET RENAMES (based on measured values):")
    print("=" * 70)

    for r in results:
        if r['num_experts'] > 1:
            total_b = r['total_params'] / 1e9
            active_m = r['active_params'] / 1e6

            # Suggest new name
            if total_b >= 1:
                total_str = f"{total_b:.1f}B"
            else:
                total_str = f"{int(total_b*1000)}M"

            if active_m >= 1000:
                active_str = f"{active_m/1000:.1f}B"
            else:
                active_str = f"{int(active_m)}M"

            suggested = f"{total_str}-A{active_str}"
            print(f"  {r['name']:<15} -> {suggested}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
