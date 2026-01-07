#!/usr/bin/env python3
"""
Benchmark: Find Maximum n_perturbations (n_perturbations) for Different n_steps

Tests increasing n_perturbations until OOM to find the sweet spot for your hardware.

Usage:
    # Test both n_steps=10 and n_steps=100
    python benchmark_max_perturbations.py

    # Test specific n_steps
    python benchmark_max_perturbations.py --n_steps 10

    # Custom thread range
    python benchmark_max_perturbations.py --min_perturbations 64 --max_threads 4096

Results saved to: benchmark_max_perturbations_results.json
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAX config
_jax_cache_dir = os.path.expanduser("~/.cache/es_lobs5_jax_compilation")
os.makedirs(_jax_cache_dir, exist_ok=True)

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CHECKPOINT = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
DEFAULT_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    lobs5_checkpoint: str = DEFAULT_CHECKPOINT
    replay_data_path: str = DEFAULT_DATA_PATH
    data_dir: str = DEFAULT_DATA_PATH

    # ES config (fixed)
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Variable params
    n_perturbations: int = 64
    n_steps: int = 100
    n_epochs: int = 1  # Only 1 epoch for benchmark
    background_msgs_per_step: int = 5

    # Fixed params
    token_mode: int = 24
    background_mode: str = 'historical_replay'
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    checkpoint_dir: str = '/tmp/benchmark_perturbations'
    checkpoint_every: int = 9999  # Don't save
    seed: int = 42
    output_dir: str = '/tmp/benchmark_perturbations'


def get_memory_stats() -> Dict[str, Any]:
    """Get GPU memory stats."""
    stats = {}
    try:
        for i, device in enumerate(jax.devices()):
            try:
                mem = device.memory_stats()
                if mem:
                    stats[f"gpu_{i}"] = {
                        "used_gb": mem.get('bytes_in_use', 0) / (1024**3),
                        "peak_gb": mem.get('peak_bytes_in_use', 0) / (1024**3),
                        "limit_gb": mem.get('bytes_limit', 0) / (1024**3),
                    }
            except:
                pass
    except:
        pass
    return stats


def clear_memory():
    """Clear JAX memory and run GC."""
    gc.collect()
    try:
        for device in jax.devices():
            # Force sync to clear pending operations
            jax.device_put(jnp.array(0), device).block_until_ready()
    except:
        pass
    gc.collect()


def run_single_benchmark(n_perturbations: int, n_steps: int) -> Dict[str, Any]:
    """
    Run a single benchmark with given n_perturbations and n_steps.

    Returns:
        Dict with success, time, memory stats, or error info
    """
    from es_lobs5.training.es_trainer import ESTrainer

    result = {
        'n_perturbations': n_perturbations,
        'n_steps': n_steps,
        'success': False,
        'error': None,
        'init_time': 0,
        'epoch_time': 0,
        'total_time': 0,
        'memory_before': {},
        'memory_after': {},
        'peak_memory_gb': 0,
        'fitness': None,
    }

    # Adjust n_perturbations for multi-GPU divisibility
    n_devices = len(jax.devices())
    if n_perturbations % n_devices != 0:
        n_perturbations = (n_perturbations // n_devices + 1) * n_devices
        result['n_perturbations'] = n_perturbations

    print(f"\n{'='*60}")
    print(f"Testing: n_perturbations={n_perturbations}, n_steps={n_steps}")
    print(f"{'='*60}")

    try:
        # Record memory before
        result['memory_before'] = get_memory_stats()

        # Create config
        config = BenchmarkConfig(
            n_perturbations=n_perturbations,
            n_steps=n_steps,
        )

        # Initialize trainer
        init_start = time.time()
        trainer = ESTrainer(config)
        result['init_time'] = time.time() - init_start
        print(f"  Init time: {result['init_time']:.1f}s")

        # Create initial state
        initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()

        # Run one epoch
        key = jax.random.PRNGKey(config.seed)

        epoch_start = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            key, epoch=0,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        # Force sync
        mean_fitness.block_until_ready()
        result['epoch_time'] = time.time() - epoch_start

        result['success'] = True
        result['fitness'] = float(mean_fitness)
        result['total_time'] = result['init_time'] + result['epoch_time']

        # Record memory after
        result['memory_after'] = get_memory_stats()

        # Calculate peak memory
        peak_gb = 0
        for gpu_stats in result['memory_after'].values():
            peak_gb = max(peak_gb, gpu_stats.get('peak_gb', 0))
        result['peak_memory_gb'] = peak_gb

        print(f"  [SUCCESS] epoch_time={result['epoch_time']:.1f}s, "
              f"fitness={result['fitness']:.4f}, peak_mem={peak_gb:.1f}GB")

    except Exception as e:
        result['error'] = str(e)
        error_type = type(e).__name__

        # Check if OOM
        is_oom = 'out of memory' in str(e).lower() or 'OOM' in str(e) or 'RESOURCE_EXHAUSTED' in str(e)

        if is_oom:
            print(f"  [OOM] {error_type}: {str(e)[:100]}")
        else:
            print(f"  [ERROR] {error_type}: {str(e)[:200]}")

        result['is_oom'] = is_oom

    finally:
        # Cleanup
        try:
            del trainer
        except:
            pass
        clear_memory()

    return result


def find_max_perturbations(n_steps: int, min_perturbations: int, max_threads: int,
                           step_multiplier: float = 2.0) -> List[Dict[str, Any]]:
    """
    Binary search-like approach to find max n_perturbations before OOM.

    Args:
        n_steps: Number of steps per episode
        min_perturbations: Minimum threads to test
        max_threads: Maximum threads to test
        step_multiplier: Multiply threads by this each step (default 2x)

    Returns:
        List of benchmark results
    """
    results = []
    current_threads = min_perturbations
    last_success_threads = 0

    print(f"\n{'#'*60}")
    print(f"# Finding Max Perturbations for n_steps={n_steps}")
    print(f"# Range: {min_perturbations} - {max_threads}")
    print(f"{'#'*60}")

    while current_threads <= max_threads:
        result = run_single_benchmark(current_threads, n_steps)
        results.append(result)

        if result['success']:
            last_success_threads = result['n_perturbations']
            current_threads = int(current_threads * step_multiplier)
        else:
            # Hit OOM, stop testing this n_steps
            break

    print(f"\n[RESULT] n_steps={n_steps}: max_threads={last_success_threads}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark max perturbations')
    parser.add_argument('--n_steps', type=int, nargs='+', default=[10, 100],
                        help='n_steps values to test (default: 10 100)')
    parser.add_argument('--min_perturbations', type=int, default=64,
                        help='Minimum n_perturbations to test (default: 64)')
    parser.add_argument('--max_threads', type=int, default=8192,
                        help='Maximum n_perturbations to test (default: 8192)')
    parser.add_argument('--step_multiplier', type=float, default=2.0,
                        help='Multiply threads by this each step (default: 2.0)')
    parser.add_argument('--output', type=str, default='benchmark_max_perturbations_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("ES-LOBS5 Max Perturbations Benchmark")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"n_steps to test: {args.n_steps}")
    print(f"Thread range: {args.min_perturbations} - {args.max_threads}")
    print("=" * 60)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'devices': [str(d) for d in jax.devices()],
        'n_devices': len(jax.devices()),
        'args': vars(args),
        'results': {},
        'summary': {},
    }

    # Test each n_steps value
    for n_steps in args.n_steps:
        clear_memory()

        results = find_max_perturbations(
            n_steps=n_steps,
            min_perturbations=args.min_perturbations,
            max_threads=args.max_threads,
            step_multiplier=args.step_multiplier,
        )

        all_results['results'][f'n_steps_{n_steps}'] = results

        # Find max successful
        successful = [r for r in results if r['success']]
        if successful:
            max_result = max(successful, key=lambda x: x['n_perturbations'])
            all_results['summary'][f'n_steps_{n_steps}'] = {
                'max_threads': max_result['n_perturbations'],
                'epoch_time': max_result['epoch_time'],
                'peak_memory_gb': max_result['peak_memory_gb'],
            }

    # Print final summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\n{'n_steps':<10} {'max_threads':<15} {'epoch_time':<15} {'peak_mem':<15}")
    print("-" * 55)

    for n_steps in args.n_steps:
        key = f'n_steps_{n_steps}'
        if key in all_results['summary']:
            s = all_results['summary'][key]
            print(f"{n_steps:<10} {s['max_threads']:<15} {s['epoch_time']:.1f}s{'':<10} {s['peak_memory_gb']:.1f}GB")
        else:
            print(f"{n_steps:<10} {'FAILED':<15}")

    # Save results
    output_path = os.path.join(
        '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5',
        args.output
    )
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
