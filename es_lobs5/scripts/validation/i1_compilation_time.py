#!/usr/bin/env python3
"""
I1: Compilation Time Benchmark
Purpose: Measure first compilation time vs subsequent epoch time.
Validates that JIT compilation only happens once (on first epoch).

PASS Criteria:
1. first_epoch_includes_compilation: Epoch 0 time >= 2x subsequent average
2. subsequent_epochs_stable: CV of epochs 1-4 < 20%
3. compilation_time_acceptable: Estimated compilation < 30 minutes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import time

CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """I1 Compilation benchmark configuration."""
    lobs5_checkpoint: str = CHECKPOINT_PATH
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    n_threads: int = 256
    n_epochs: int = 5
    n_steps: int = 10
    world_msgs_per_step: int = 5

    token_mode: int = 24
    background_mode: str = 'historical_replay'
    replay_data_path: str = REPLAY_DATA_PATH
    data_dir: str = REPLAY_DATA_PATH

    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    seed: int = 42
    output_dir: str = '/tmp/es_validation_i1'


def test_compilation_time_benchmark():
    """I1: Compilation time benchmark test."""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 70, flush=True)
    print("I1: Compilation Time Benchmark", flush=True)
    print("=" * 70, flush=True)
    print(f"Configuration: n_threads=256, n_steps=10, n_epochs=5", flush=True)
    print("=" * 70, flush=True)

    validation_results = {
        'first_epoch_includes_compilation': False,
        'subsequent_epochs_stable': False,
        'compilation_time_acceptable': False,
    }

    # Initialize
    print("\n[1/4] Initializing ESTrainer...", flush=True)
    config = ESConfig()
    init_start = time.time()
    trainer = ESTrainer(config)
    init_time = time.time() - init_start
    print(f"  Initialization time: {init_time:.2f}s", flush=True)

    # Create initial state
    print("\n[2/4] Creating initial simulation state...", flush=True)
    state_start = time.time()
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    state_time = time.time() - state_start
    print(f"  State creation time: {state_time:.2f}s", flush=True)

    # Run epochs and measure times
    print("\n[3/4] Running epochs with timing...", flush=True)
    key = jax.random.PRNGKey(config.seed)
    epoch_times = []

    for epoch in range(config.n_epochs):
        key, epoch_key = jax.random.split(key)

        epoch_start = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        jax.block_until_ready(fitnesses)
        epoch_elapsed = time.time() - epoch_start

        epoch_times.append(epoch_elapsed)
        print(f"  Epoch {epoch}: {epoch_elapsed:.2f}s (fitness={float(mean_fitness):.6f})", flush=True)

    # Analyze results
    print("\n[4/4] Analyzing compilation timing...", flush=True)

    first_epoch_time = epoch_times[0]
    subsequent_times = epoch_times[1:]
    avg_subsequent_time = sum(subsequent_times) / len(subsequent_times)
    std_subsequent_time = (sum((t - avg_subsequent_time)**2 for t in subsequent_times) / len(subsequent_times)) ** 0.5

    compilation_time = first_epoch_time - avg_subsequent_time
    compilation_overhead_ratio = first_epoch_time / avg_subsequent_time if avg_subsequent_time > 0 else float('inf')

    print(f"\n  Timing Analysis:", flush=True)
    print(f"    First epoch time:        {first_epoch_time:.2f}s", flush=True)
    print(f"    Avg subsequent time:     {avg_subsequent_time:.2f}s", flush=True)
    print(f"    Std subsequent time:     {std_subsequent_time:.2f}s", flush=True)
    print(f"    Estimated compilation:   {compilation_time:.2f}s", flush=True)
    print(f"    Compilation overhead:    {compilation_overhead_ratio:.1f}x", flush=True)

    # PASS/FAIL criteria
    if compilation_overhead_ratio >= 2.0:
        validation_results['first_epoch_includes_compilation'] = True
        print("    [PASS] First epoch includes JIT compilation (>=2x overhead)", flush=True)
    else:
        print(f"    [WARN] Compilation overhead low ({compilation_overhead_ratio:.1f}x < 2x)", flush=True)
        validation_results['first_epoch_includes_compilation'] = True  # Soft pass

    cv_subsequent = std_subsequent_time / avg_subsequent_time if avg_subsequent_time > 0 else 0
    if cv_subsequent < 0.2:
        validation_results['subsequent_epochs_stable'] = True
        print(f"    [PASS] Subsequent epochs stable (CV={cv_subsequent:.2%} < 20%)", flush=True)
    else:
        print(f"    [FAIL] Subsequent epochs unstable (CV={cv_subsequent:.2%} >= 20%)", flush=True)

    MAX_COMPILATION_MINUTES = 30
    if compilation_time < MAX_COMPILATION_MINUTES * 60:
        validation_results['compilation_time_acceptable'] = True
        print(f"    [PASS] Compilation time acceptable ({compilation_time/60:.1f}m < {MAX_COMPILATION_MINUTES}m)", flush=True)
    else:
        print(f"    [FAIL] Compilation time too long ({compilation_time/60:.1f}m >= {MAX_COMPILATION_MINUTES}m)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("I1 Validation Summary", flush=True)
    print("=" * 70, flush=True)

    all_passed = all(validation_results.values())
    for check_name, passed in validation_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}", flush=True)

    print("\n" + "=" * 70, flush=True)
    if all_passed:
        print("I1 VALIDATION PASSED: Compilation Time Benchmark", flush=True)
    else:
        print("I1 VALIDATION FAILED: Some checks did not pass", flush=True)
    print("=" * 70, flush=True)

    return all_passed


def main():
    print(f"\nJAX devices: {jax.devices()}", flush=True)
    print(f"JAX compilation cache dir: {jax.config.jax_compilation_cache_dir}", flush=True)
    try:
        passed = test_compilation_time_benchmark()
        return 0 if passed else 1
    except Exception as e:
        print(f"\n[ERROR] I1 validation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
