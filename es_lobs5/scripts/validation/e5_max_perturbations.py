#!/usr/bin/env python3
"""
E5: Max Perturbations Scaling Test

Purpose:
  Find the maximum n_threads before OOM and identify optimal throughput point.

Tests:
  - Test n_threads = 32, 64, 128, 256, 512, 1024 (stop at OOM)
  - For each: run 1 epoch, measure memory, throughput, fitness_std
  - Find optimal n_threads (max throughput)
  - Find max n_threads (before OOM)

Configuration:
  - n_steps=10 (short for quick testing)
  - n_epochs=1 per test
  - background_mode='historical_replay' ONLY
"""

import sys
import os
import gc
import time
import traceback
from dataclasses import dataclass, replace
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"

# n_threads values to test
N_THREADS_TO_TEST = [32, 64, 128, 256, 512, 1024]


@dataclass
class ESConfig:
    """ES training configuration for scaling test."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration (short for quick testing)
    n_threads: int = 32
    n_epochs: int = 1
    n_steps: int = 10
    world_msgs_per_step: int = 5

    # Token mode (24 matches the checkpoint)
    token_mode: int = 24

    # Background mode: historical_replay ONLY
    background_mode: str = 'historical_replay'
    replay_data_path: str = REPLAY_DATA_PATH
    data_dir: str = REPLAY_DATA_PATH

    # Task configuration
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100

    # Other
    seed: int = 42
    output_dir: str = '/tmp/es_validation_e5'


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information.

    Returns:
        Dict with 'used_gb', 'total_gb', 'free_gb' keys
    """
    try:
        # JAX memory stats
        backend = jax.lib.xla_bridge.get_backend()
        devices = backend.devices()

        total_used = 0.0
        total_available = 0.0

        for device in devices:
            try:
                # Get memory stats from JAX
                stats = device.memory_stats()
                if stats:
                    total_used += stats.get('bytes_in_use', 0)
                    total_available += stats.get('bytes_limit', 0)
            except Exception:
                pass

        # Convert to GB
        used_gb = total_used / (1024**3)
        total_gb = total_available / (1024**3) if total_available > 0 else 0
        free_gb = (total_available - total_used) / (1024**3) if total_available > 0 else 0

        return {
            'used_gb': used_gb,
            'total_gb': total_gb,
            'free_gb': free_gb
        }
    except Exception as e:
        return {
            'used_gb': -1.0,
            'total_gb': -1.0,
            'free_gb': -1.0,
            'error': str(e)
        }


def run_single_test(n_threads: int, base_config: ESConfig) -> Dict:
    """Run a single scaling test with given n_threads.

    Args:
        n_threads: Number of ES perturbations/threads
        base_config: Base ESConfig to modify

    Returns:
        Dict with test results:
        - success: bool
        - n_threads: int
        - memory_before_gb: float
        - memory_after_gb: float
        - memory_peak_gb: float (if available)
        - throughput_steps_per_sec: float
        - mean_fitness: float
        - fitness_std: float
        - elapsed_time_sec: float
        - error: str (if failed)
    """
    from es_lobs5.training.es_trainer import ESTrainer

    result = {
        'n_threads': n_threads,
        'success': False,
        'memory_before_gb': -1.0,
        'memory_after_gb': -1.0,
        'memory_peak_gb': -1.0,
        'throughput_steps_per_sec': -1.0,
        'mean_fitness': float('nan'),
        'fitness_std': float('nan'),
        'elapsed_time_sec': -1.0,
        'error': None,
    }

    # Force garbage collection before test
    gc.collect()
    jax.clear_caches()

    try:
        # Get memory before
        mem_before = get_gpu_memory_info()
        result['memory_before_gb'] = mem_before['used_gb']

        # Create config with modified n_threads
        config = replace(base_config, n_threads=n_threads)

        print(f"\n{'='*60}")
        print(f"Testing n_threads = {n_threads}")
        print(f"{'='*60}")
        print(f"  Memory before: {mem_before['used_gb']:.2f} GB")

        # Initialize trainer
        print("  [1/4] Initializing ESTrainer...")
        trainer = ESTrainer(config)
        print("        ESTrainer initialized")

        # Get initial state
        print("  [2/4] Creating initial simulation state...")
        initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
        print(f"        initial_msg_history shape: {initial_msg_history.shape}")

        # Run epoch with timing
        print("  [3/4] Running single epoch...")
        key = jax.random.PRNGKey(config.seed)

        start_time = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            key, epoch=0,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        # Block until computation is done
        jax.block_until_ready(mean_fitness)
        jax.block_until_ready(fitnesses)
        elapsed_time = time.time() - start_time

        # Get memory after
        mem_after = get_gpu_memory_info()
        result['memory_after_gb'] = mem_after['used_gb']

        # Calculate metrics
        fitness_std = float(jnp.std(fitnesses))
        mean_fitness_val = float(mean_fitness)

        # Throughput: steps per second across all threads
        total_steps = n_threads * config.n_steps
        throughput = total_steps / elapsed_time

        # Update result
        result['success'] = True
        result['mean_fitness'] = mean_fitness_val
        result['fitness_std'] = fitness_std
        result['elapsed_time_sec'] = elapsed_time
        result['throughput_steps_per_sec'] = throughput

        print("  [4/4] Results:")
        print(f"        Mean fitness: {mean_fitness_val:.6f}")
        print(f"        Fitness std: {fitness_std:.6f}")
        print(f"        Elapsed time: {elapsed_time:.2f}s")
        print(f"        Throughput: {throughput:.1f} steps/s")
        print(f"        Memory after: {mem_after['used_gb']:.2f} GB")
        print(f"        Memory delta: {mem_after['used_gb'] - mem_before['used_gb']:.2f} GB")

        # Cleanup
        del trainer
        del initial_sim_state
        del initial_msg_history
        del fitnesses
        gc.collect()
        jax.clear_caches()

        return result

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()

        # Check if it's an OOM error
        is_oom = any(term in error_msg.lower() or term in traceback_str.lower()
                     for term in ['oom', 'out of memory', 'resource exhausted', 'xla allocation'])

        result['error'] = f"{'OOM: ' if is_oom else ''}{error_msg}"

        mem_after = get_gpu_memory_info()
        result['memory_after_gb'] = mem_after['used_gb']

        print(f"  [ERROR] Test failed!")
        print(f"          Error: {result['error'][:100]}")
        if is_oom:
            print("          (OOM detected - stopping further tests)")

        # Cleanup on error
        gc.collect()
        jax.clear_caches()

        return result


def print_scaling_table(results: List[Dict]):
    """Print scaling results as a formatted table."""
    print("\n" + "="*100)
    print("SCALING TEST RESULTS")
    print("="*100)

    # Header
    header = f"{'n_threads':>10} | {'Status':>8} | {'Memory (GB)':>12} | {'Time (s)':>10} | {'Throughput':>12} | {'Fitness':>12} | {'Std':>10}"
    print(header)
    print("-" * 100)

    # Data rows
    for r in results:
        status = "OK" if r['success'] else "FAIL"
        mem = f"{r['memory_after_gb']:.2f}" if r['memory_after_gb'] >= 0 else "N/A"
        time_str = f"{r['elapsed_time_sec']:.2f}" if r['elapsed_time_sec'] >= 0 else "N/A"
        throughput = f"{r['throughput_steps_per_sec']:.1f}" if r['throughput_steps_per_sec'] >= 0 else "N/A"
        fitness = f"{r['mean_fitness']:.6f}" if not jnp.isnan(r['mean_fitness']) else "N/A"
        std = f"{r['fitness_std']:.6f}" if not jnp.isnan(r['fitness_std']) else "N/A"

        if not r['success'] and r['error']:
            error_short = r['error'][:20] + "..." if len(r['error']) > 20 else r['error']
            row = f"{r['n_threads']:>10} | {status:>8} | {mem:>12} | {'--':>10} | {'--':>12} | {error_short}"
        else:
            row = f"{r['n_threads']:>10} | {status:>8} | {mem:>12} | {time_str:>10} | {throughput:>12} | {fitness:>12} | {std:>10}"
        print(row)

    print("="*100)


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze scaling test results.

    Returns:
        Dict with:
        - max_n_threads: Maximum n_threads before OOM
        - optimal_n_threads: n_threads with highest throughput
        - optimal_throughput: Maximum throughput achieved
        - oom_n_threads: First n_threads that caused OOM (or None)
    """
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    analysis = {
        'max_n_threads': None,
        'optimal_n_threads': None,
        'optimal_throughput': 0.0,
        'oom_n_threads': None,
        'successful_tests': len(successful),
        'failed_tests': len(failed),
    }

    if not successful:
        print("\nWARNING: No successful tests!")
        return analysis

    # Max n_threads (largest successful)
    analysis['max_n_threads'] = max(r['n_threads'] for r in successful)

    # Optimal n_threads (highest throughput)
    best = max(successful, key=lambda x: x['throughput_steps_per_sec'])
    analysis['optimal_n_threads'] = best['n_threads']
    analysis['optimal_throughput'] = best['throughput_steps_per_sec']

    # First OOM
    oom_results = [r for r in failed if r['error'] and 'oom' in r['error'].lower()]
    if oom_results:
        analysis['oom_n_threads'] = min(r['n_threads'] for r in oom_results)

    return analysis


def main():
    """Main entry point for E5 scaling test."""
    print("="*80)
    print("E5: Max Perturbations Scaling Test")
    print("="*80)
    print(f"\nTest configuration:")
    print(f"  n_steps: 10 (short for quick testing)")
    print(f"  n_epochs: 1 per test")
    print(f"  background_mode: historical_replay")
    print(f"  n_threads to test: {N_THREADS_TO_TEST}")
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"Replay data: {REPLAY_DATA_PATH}")

    # Print GPU info
    print("\nGPU Information:")
    try:
        devices = jax.devices()
        print(f"  Number of devices: {len(devices)}")
        for i, d in enumerate(devices):
            print(f"  Device {i}: {d.device_kind}")
    except Exception as e:
        print(f"  Error getting device info: {e}")

    mem_info = get_gpu_memory_info()
    print(f"  Initial memory used: {mem_info['used_gb']:.2f} GB")
    print(f"  Total memory: {mem_info['total_gb']:.2f} GB")

    # Create base config
    base_config = ESConfig()

    # Run tests
    results = []
    stop_testing = False

    for n_threads in N_THREADS_TO_TEST:
        if stop_testing:
            print(f"\nSkipping n_threads={n_threads} (previous OOM)")
            results.append({
                'n_threads': n_threads,
                'success': False,
                'memory_before_gb': -1.0,
                'memory_after_gb': -1.0,
                'memory_peak_gb': -1.0,
                'throughput_steps_per_sec': -1.0,
                'mean_fitness': float('nan'),
                'fitness_std': float('nan'),
                'elapsed_time_sec': -1.0,
                'error': 'Skipped (previous OOM)',
            })
            continue

        result = run_single_test(n_threads, base_config)
        results.append(result)

        # Stop if OOM
        if not result['success'] and result['error'] and 'oom' in result['error'].lower():
            stop_testing = True

    # Print results table
    print_scaling_table(results)

    # Analyze results
    analysis = analyze_results(results)

    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"  Successful tests: {analysis['successful_tests']}/{len(results)}")
    print(f"  Max n_threads (before OOM): {analysis['max_n_threads']}")
    print(f"  Optimal n_threads (max throughput): {analysis['optimal_n_threads']}")
    print(f"  Optimal throughput: {analysis['optimal_throughput']:.1f} steps/s")
    if analysis['oom_n_threads']:
        print(f"  First OOM at n_threads: {analysis['oom_n_threads']}")
    else:
        print(f"  No OOM encountered (max tested: {N_THREADS_TO_TEST[-1]})")

    # Final summary
    print("\n" + "="*80)
    if analysis['successful_tests'] > 0:
        print(f"E5 PASSED: Max perturbations = {analysis['max_n_threads']}, "
              f"Optimal = {analysis['optimal_n_threads']} ({analysis['optimal_throughput']:.1f} steps/s)")
    else:
        print("E5 FAILED: No successful tests")
    print("="*80)

    return 0 if analysis['successful_tests'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
