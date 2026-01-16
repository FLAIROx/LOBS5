#!/usr/bin/env python3
"""
Test FSDP (Fully Sharded Data Parallel) initialization for large models.

This script verifies that large models (23B+) can be initialized across
multiple GPUs without OOM by sharding parameters during initialization.

Usage:
    # Test with 4 GPUs using FSDP
    python test_fsdp_init.py --preset 23B-A1.9B --num-gpus 4

    # Compare memory usage with standard init (will OOM for large models)
    python test_fsdp_init.py --preset 1.3B --num-gpus 4 --compare-standard
"""

import argparse
import os
import sys
import time

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
MAXTEXT_PATH = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/maxtext/src'
if MAXTEXT_PATH not in sys.path:
    sys.path.insert(0, MAXTEXT_PATH)

import jax
import jax.numpy as jnp
from jax import random

from lobmax.config import LOBMAXConfig
from lobmax.models import LOBMAXModel, count_parameters, count_active_parameters
from lob.sharding_utils import create_fsdp_mesh, sharded_init


# Model configurations (same as count_single_model.py)
CONFIGS = {
    "125M": dict(d_model=768, n_layers=12, num_heads=12, mlp_dim=3072, num_experts=1),
    "360M": dict(d_model=1024, n_layers=24, num_heads=16, mlp_dim=4096, num_experts=1),
    "1.3B": dict(d_model=2048, n_layers=24, num_heads=16, mlp_dim=8192, num_experts=1),
    "3B": dict(d_model=2560, n_layers=32, num_heads=20, mlp_dim=10240, num_experts=1),
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


def get_gpu_memory_usage():
    """Get current GPU memory usage for all devices."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        return [(int(line.split(',')[0]), int(line.split(',')[1])) for line in lines]
    except:
        return None


def test_fsdp_init(preset_name: str, num_gpus: int, compare_standard: bool = False):
    """Test FSDP initialization for a model preset."""
    cfg_dict = CONFIGS[preset_name]

    print("=" * 70)
    print(f"FSDP Initialization Test - {preset_name}")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX platform: {jax.default_backend()}")
    print(f"Requested GPUs: {num_gpus}")
    print()
    print(f"Configuration:")
    print(f"  d_model     = {cfg_dict['d_model']}")
    print(f"  n_layers    = {cfg_dict['n_layers']}")
    print(f"  num_heads   = {cfg_dict['num_heads']}")
    print(f"  mlp_dim     = {cfg_dict['mlp_dim']}")
    print(f"  num_experts = {cfg_dict['num_experts']}")
    print()

    # Show initial memory
    mem_before = get_gpu_memory_usage()
    if mem_before:
        print("GPU Memory (before init):")
        for i, (used, total) in enumerate(mem_before):
            print(f"  GPU {i}: {used} / {total} MB ({100*used/total:.1f}%)")
        print()

    # Create config and model
    config = create_config(preset_name)

    # Create FSDP mesh (all GPUs for FSDP)
    print(f"Creating FSDP mesh with {num_gpus} GPUs...")
    mesh = create_fsdp_mesh(num_gpus, fsdp_parallelism=num_gpus)
    print()

    # Create model
    model = LOBMAXModel(config=config, mesh=mesh, training=True)

    # Create dummy inputs
    key = random.PRNGKey(42)
    seq_len = 256
    batch_size = 1
    dummy_msg = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_book = jnp.zeros((batch_size, seq_len, config.d_book), dtype=jnp.float32)

    def init_fn():
        return model.init(key, x_m=dummy_msg, x_b=dummy_book)

    # Test FSDP init
    print("=" * 70)
    print("Testing FSDP Sharded Initialization")
    print("=" * 70)
    start_time = time.time()

    try:
        variables = sharded_init(init_fn, mesh)
        params = variables['params']
        fsdp_time = time.time() - start_time

        # Count parameters
        total_params = count_parameters(params)
        active_params = count_active_parameters(params, config.num_experts, 1)

        print()
        print(f"FSDP Init SUCCESS!")
        print(f"  Time: {fsdp_time:.2f}s")
        print(f"  Total params:  {total_params:,} ({total_params/1e9:.3f}B)")
        print(f"  Active params: {active_params:,} ({active_params/1e9:.3f}B)")
        print(f"  Ratio: {total_params/active_params:.2f}x")

        # Show memory after FSDP init
        mem_after_fsdp = get_gpu_memory_usage()
        if mem_after_fsdp:
            print()
            print("GPU Memory (after FSDP init):")
            for i, (used, total) in enumerate(mem_after_fsdp):
                delta = used - mem_before[i][0] if mem_before else 0
                print(f"  GPU {i}: {used} / {total} MB ({100*used/total:.1f}%) [+{delta} MB]")

        # Check that memory is distributed across GPUs
        if mem_after_fsdp and mem_before:
            deltas = [mem_after_fsdp[i][0] - mem_before[i][0] for i in range(len(mem_after_fsdp))]
            avg_delta = sum(deltas) / len(deltas)
            max_delta = max(deltas)
            print()
            print(f"Memory distribution:")
            print(f"  Average per GPU: {avg_delta:.0f} MB")
            print(f"  Max on single GPU: {max_delta:.0f} MB")
            print(f"  Distribution efficiency: {100*avg_delta/max_delta:.1f}%")

    except Exception as e:
        print(f"FSDP Init FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Optionally compare with standard init
    if compare_standard:
        print()
        print("=" * 70)
        print("Testing Standard Initialization (for comparison)")
        print("=" * 70)

        # Clear previous params to free memory
        del params, variables
        jax.clear_caches()

        start_time = time.time()
        try:
            variables_std = init_fn()
            std_time = time.time() - start_time
            print(f"Standard Init SUCCESS! Time: {std_time:.2f}s")

            mem_after_std = get_gpu_memory_usage()
            if mem_after_std:
                print("GPU Memory (after standard init):")
                for i, (used, total) in enumerate(mem_after_std):
                    print(f"  GPU {i}: {used} / {total} MB ({100*used/total:.1f}%)")

        except Exception as e:
            print(f"Standard Init FAILED (expected for large models): {e}")

    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test FSDP initialization for large models")
    parser.add_argument("--preset", type=str, required=True,
                        choices=list(CONFIGS.keys()),
                        help="Model preset to test")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs for FSDP (default: 4)")
    parser.add_argument("--compare-standard", action="store_true",
                        help="Also test standard init for comparison")
    args = parser.parse_args()

    test_fsdp_init(args.preset, args.num_gpus, args.compare_standard)


if __name__ == "__main__":
    main()
