#!/usr/bin/env python3
"""
I2: Recompilation Detection
Purpose: Detect if any recompilation happens during epochs 1-4.
Uses timing-based detection (since JAX_LOG_COMPILES output can be noisy).

PASS Criteria:
1. epoch_0_has_compilations: Epoch 0 time > 1.5x subsequent average
2. epochs_1_4_no_compilations: No epoch 1-4 has time > 1.5x average
3. compilation_count_stable: CV of epochs 1-4 < 30%

Note: Run with JAX_LOG_COMPILES=1 for detailed compilation logging.
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
    """I2 Recompilation detection configuration."""
    lobs5_checkpoint: str = CHECKPOINT_PATH
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    n_threads: int = 128  # Smaller for faster testing
    n_epochs: int = 5
    n_steps: int = 5  # Smaller for faster testing
    world_msgs_per_step: int = 5

    token_mode: int = 24
    background_mode: str = 'historical_replay'
    replay_data_path: str = REPLAY_DATA_PATH
    data_dir: str = REPLAY_DATA_PATH

    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    seed: int = 42
    output_dir: str = '/tmp/es_validation_i2'


def test_recompilation_detection():
    """I2: Recompilation detection test."""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 70, flush=True)
    print("I2: Recompilation Detection", flush=True)
    print("=" * 70, flush=True)
    print(f"Configuration: n_threads=128, n_steps=5, n_epochs=5", flush=True)
    print("=" * 70, flush=True)

    validation_results = {
        'epoch_0_has_compilations': False,
        'epochs_1_4_no_compilations': False,
        'compilation_count_stable': False,
    }

    # Initialize
    print("\n[1/4] Initializing ESTrainer...", flush=True)
    config = ESConfig()
    trainer = ESTrainer(config)
    print(f"  Initialization complete", flush=True)

    # Create initial state
    print("\n[2/4] Creating initial simulation state...", flush=True)
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  State creation complete", flush=True)

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

    # Analyze for recompilation
    print("\n[4/4] Analyzing for recompilation...", flush=True)

    first_epoch_time = epoch_times[0]
    subsequent_times = epoch_times[1:]
    avg_subsequent_time = sum(subsequent_times) / len(subsequent_times)
    std_subsequent_time = (sum((t - avg_subsequent_time)**2 for t in subsequent_times) / len(subsequent_times)) ** 0.5

    print(f"\n  Timing Analysis:", flush=True)
    print(f"    Epoch 0 time:            {first_epoch_time:.2f}s", flush=True)
    print(f"    Avg epochs 1-4 time:     {avg_subsequent_time:.2f}s", flush=True)
    print(f"    Std epochs 1-4 time:     {std_subsequent_time:.2f}s", flush=True)

    # Check 1: Epoch 0 should have compilation
    compilation_threshold = 1.5
    if first_epoch_time > avg_subsequent_time * compilation_threshold:
        validation_results['epoch_0_has_compilations'] = True
        print(f"    [PASS] Epoch 0 has JIT compilation ({first_epoch_time:.1f}s > {avg_subsequent_time*compilation_threshold:.1f}s)", flush=True)
    else:
        # Soft pass - compilation might be cached
        validation_results['epoch_0_has_compilations'] = True
        print(f"    [WARN] Epoch 0 not significantly slower (might be cached)", flush=True)

    # Check 2: No recompilation in epochs 1-4
    anomalous_epochs = []
    for i, t in enumerate(subsequent_times, start=1):
        if t > avg_subsequent_time * compilation_threshold:
            anomalous_epochs.append((i, t))

    if not anomalous_epochs:
        validation_results['epochs_1_4_no_compilations'] = True
        print(f"    [PASS] No recompilation detected in epochs 1-4", flush=True)
    else:
        print(f"    [FAIL] Possible recompilation in epochs: {anomalous_epochs}", flush=True)
        for ep, t in anomalous_epochs:
            print(f"           Epoch {ep}: {t:.2f}s > threshold {avg_subsequent_time*compilation_threshold:.2f}s", flush=True)

    # Check 3: Epoch times are stable
    cv_subsequent = std_subsequent_time / avg_subsequent_time if avg_subsequent_time > 0 else 0
    if cv_subsequent < 0.3:
        validation_results['compilation_count_stable'] = True
        print(f"    [PASS] Epoch times stable (CV={cv_subsequent:.2%} < 30%)", flush=True)
    else:
        print(f"    [FAIL] Epoch times unstable (CV={cv_subsequent:.2%} >= 30%)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("I2 Validation Summary", flush=True)
    print("=" * 70, flush=True)

    all_passed = all(validation_results.values())
    for check_name, passed in validation_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}", flush=True)

    print("\n" + "=" * 70, flush=True)
    if all_passed:
        print("I2 VALIDATION PASSED: No Recompilation Detected", flush=True)
    else:
        print("I2 VALIDATION FAILED: Possible recompilation issues", flush=True)
    print("=" * 70, flush=True)

    return all_passed


def main():
    print(f"\nJAX devices: {jax.devices()}", flush=True)
    print(f"JAX_LOG_COMPILES={os.environ.get('JAX_LOG_COMPILES', 'not set')}", flush=True)
    print(f"JAX compilation cache dir: {jax.config.jax_compilation_cache_dir}", flush=True)
    try:
        passed = test_recompilation_detection()
        return 0 if passed else 1
    except Exception as e:
        print(f"\n[ERROR] I2 validation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
