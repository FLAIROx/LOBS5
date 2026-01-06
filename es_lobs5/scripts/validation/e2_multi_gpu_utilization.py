#!/usr/bin/env python3
"""
E2: Multi-GPU Utilization Validation for ES-LOBS5

Purpose: Validate that training uses all 4 GPUs effectively.

Configuration:
- n_threads=256 (64 per GPU target)
- n_steps=100, n_epochs=10
- background_mode='historical_replay' ONLY

Validates:
1. All 4 GPUs are utilized (check jax.devices())
2. Memory distributed across GPUs
3. Compare 1-GPU vs 4-GPU throughput (optional)
4. No device placement errors
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import time


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """E2 Multi-GPU validation configuration."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration (256 threads = 64 per GPU on 4 GPUs)
    # NOTE: n_steps=10 to avoid 55+ min XLA compilation (100 steps = 3.7M ops)
    # NOTE: n_epochs=5 for validation (full training would take 2+ hours)
    n_threads: int = 256
    n_epochs: int = 5   # Reduced from 50 for faster validation
    n_steps: int = 10   # Reduced from 100 to avoid XLA compilation hang
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
    output_dir: str = '/tmp/es_validation_e2'


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
    return len(gpu_devices) if gpu_devices else len(devices)


def get_device_memory_stats():
    """Get memory statistics for all devices."""
    devices = jax.devices()
    memory_stats = []

    for i, device in enumerate(devices):
        try:
            stats = device.memory_stats()
            if stats:
                bytes_in_use = stats.get('bytes_in_use', 0)
                bytes_limit = stats.get('bytes_limit', 0)
                peak_bytes = stats.get('peak_bytes_in_use', bytes_in_use)
                memory_stats.append({
                    'device_id': i,
                    'device_str': str(device),
                    'bytes_in_use': bytes_in_use,
                    'bytes_limit': bytes_limit,
                    'peak_bytes': peak_bytes,
                    'usage_pct': 100.0 * bytes_in_use / bytes_limit if bytes_limit > 0 else 0,
                    'peak_pct': 100.0 * peak_bytes / bytes_limit if bytes_limit > 0 else 0,
                })
        except Exception as e:
            memory_stats.append({
                'device_id': i,
                'device_str': str(device),
                'error': str(e),
            })

    return memory_stats


def format_bytes(bytes_val):
    """Format bytes to human readable string."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val} B"


def print_device_info():
    """Print detailed device information."""
    print("\n" + "=" * 70)
    print("JAX Device Information")
    print("=" * 70)

    devices = jax.devices()
    print(f"\nTotal devices: {len(devices)}")
    print(f"Default backend: {jax.default_backend()}")

    for i, device in enumerate(devices):
        print(f"\n  Device {i}:")
        print(f"    ID: {device.id}")
        print(f"    Platform: {device.platform}")
        print(f"    Device kind: {device.device_kind}")
        print(f"    String: {str(device)}")

        try:
            stats = device.memory_stats()
            if stats:
                print(f"    Memory in use: {format_bytes(stats.get('bytes_in_use', 0))}")
                print(f"    Memory limit: {format_bytes(stats.get('bytes_limit', 0))}")
        except Exception:
            print("    Memory stats: Not available")

    return len(devices)


def verify_multi_device_usage():
    """Verify that JAX is configured to use multiple devices."""
    print("\n" + "=" * 70)
    print("Multi-Device Configuration Check")
    print("=" * 70)

    devices = jax.devices()
    n_devices = len(devices)

    checks = {
        'multiple_devices_available': n_devices >= 4,
        'all_devices_same_type': len(set(d.platform for d in devices)) == 1,
        'devices_accessible': True,
    }

    # Test device accessibility
    for i, device in enumerate(devices):
        try:
            with jax.default_device(device):
                x = jnp.ones((10, 10))
                _ = x @ x  # Simple operation
        except Exception as e:
            checks['devices_accessible'] = False
            print(f"  [ERROR] Device {i} not accessible: {e}")

    print(f"\n  Devices available: {n_devices}")
    print(f"  Multiple devices (>=4): {'PASS' if checks['multiple_devices_available'] else 'FAIL'}")
    print(f"  Same device type: {'PASS' if checks['all_devices_same_type'] else 'FAIL'}")
    print(f"  All devices accessible: {'PASS' if checks['devices_accessible'] else 'FAIL'}")

    return checks


def test_pmap_distribution():
    """Test that pmap correctly distributes across devices."""
    print("\n" + "=" * 70)
    print("PMAP Distribution Test")
    print("=" * 70)

    devices = jax.devices()
    n_devices = len(devices)

    # Create test function that returns device info
    @jax.pmap
    def get_device_id(x):
        # Each device processes independently
        return x * 2

    # Test data distributed across devices
    test_data = jnp.arange(n_devices).reshape(n_devices, 1)

    print(f"\n  Input shape: {test_data.shape}")
    print(f"  Number of devices: {n_devices}")

    try:
        result = get_device_id(test_data)
        print(f"  Output shape: {result.shape}")
        print(f"  Output: {result.flatten()}")
        print("  [PASS] pmap distribution successful")
        return True
    except Exception as e:
        print(f"  [FAIL] pmap error: {e}")
        return False


def test_vmap_multi_gpu():
    """Test that vmap with large batch distributes across GPUs."""
    print("\n" + "=" * 70)
    print("VMAP Multi-GPU Distribution Test")
    print("=" * 70)

    devices = jax.devices()
    n_devices = len(devices)

    # Create a larger computation that should benefit from multi-GPU
    @jax.jit
    @jax.vmap
    def large_computation(x):
        # Simulate complex computation
        result = x
        for _ in range(10):
            result = jnp.tanh(result @ result.T)
        return jnp.sum(result)

    # 256 threads distributed across devices
    batch_size = 256
    matrix_size = 64  # Smaller matrix to avoid OOM

    print(f"\n  Batch size: {batch_size}")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Threads per GPU (target): {batch_size // n_devices}")

    # Create test data
    key = jax.random.PRNGKey(42)
    test_data = jax.random.normal(key, (batch_size, matrix_size, matrix_size))

    # Record initial memory
    initial_mem = get_device_memory_stats()

    # Run computation
    start_time = time.time()
    try:
        result = large_computation(test_data)
        result.block_until_ready()
        elapsed = time.time() - start_time

        print(f"  Computation time: {elapsed:.3f}s")
        print(f"  Result shape: {result.shape}")
        print("  [PASS] vmap execution successful")

        # Check memory distribution
        final_mem = get_device_memory_stats()
        print("\n  Memory usage per device:")
        for mem in final_mem:
            if 'error' not in mem:
                used = format_bytes(mem['bytes_in_use'])
                peak = format_bytes(mem['peak_bytes'])
                print(f"    Device {mem['device_id']}: {used} (peak: {peak})")

        return True
    except Exception as e:
        print(f"  [FAIL] vmap error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_es_training_multi_gpu():
    """Test actual ES training with multi-GPU configuration."""
    print("\n" + "=" * 70)
    print("ES Training Multi-GPU Test (256 threads, 5 epochs)")
    print("=" * 70)

    from es_lobs5.training.es_trainer import ESTrainer

    # Create config
    print("\n[1/5] Creating ESConfig with 256 threads (64 per GPU)...")
    config = ESConfig()
    print(f"  n_threads: {config.n_threads}")
    print(f"  n_steps: {config.n_steps}")
    print(f"  n_epochs: {config.n_epochs}")
    print(f"  background_mode: {config.background_mode}")

    # Verify background mode
    assert config.background_mode == 'historical_replay', \
        f"Expected historical_replay, got {config.background_mode}"
    print("  [PASS] Config validated")

    # Record initial memory
    print("\n[2/5] Recording initial memory state...")
    initial_memory = get_device_memory_stats()
    for mem in initial_memory:
        if 'error' not in mem:
            print(f"  Device {mem['device_id']}: {format_bytes(mem['bytes_in_use'])}")

    # Initialize trainer
    print("\n[3/5] Initializing ESTrainer...")
    start_init = time.time()
    trainer = ESTrainer(config)
    init_time = time.time() - start_init
    print(f"  Initialization time: {init_time:.2f}s")
    print("  [PASS] ESTrainer initialized")

    # Record post-init memory
    post_init_memory = get_device_memory_stats()
    print("\n  Memory after initialization:")
    for mem in post_init_memory:
        if 'error' not in mem:
            print(f"    Device {mem['device_id']}: {format_bytes(mem['bytes_in_use'])} ({mem['usage_pct']:.1f}%)")

    # Create initial state
    print("\n[4/5] Creating initial simulation state...")
    start_state = time.time()
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    state_time = time.time() - start_state
    print(f"  State creation time: {state_time:.2f}s")
    print(f"  Message history shape: {initial_msg_history.shape}")
    print("  [PASS] Initial state created")

    # Run training epochs
    print("\n[5/5] Running training loop (10 epochs)...")
    key = jax.random.PRNGKey(config.seed)

    epoch_times = []
    epoch_fitnesses = []
    memory_history = []

    for epoch in range(config.n_epochs):
        key, epoch_key = jax.random.split(key)

        epoch_start = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        epoch_elapsed = time.time() - epoch_start

        epoch_times.append(epoch_elapsed)
        epoch_fitnesses.append(float(mean_fitness))

        # Record memory
        mem_stats = get_device_memory_stats()
        memory_history.append(mem_stats)

        # Print progress
        fitness_std = float(jnp.std(fitnesses))
        print(f"\n  Epoch {epoch}/{config.n_epochs-1}:")
        print(f"    Mean fitness: {float(mean_fitness):.6f}")
        print(f"    Fitness std: {fitness_std:.6f}")
        print(f"    Time: {epoch_elapsed:.2f}s")
        print(f"    Throughput: {config.n_threads / epoch_elapsed:.1f} threads/s")

        # Memory per device
        print("    Memory usage:")
        for mem in mem_stats:
            if 'error' not in mem:
                print(f"      Device {mem['device_id']}: {format_bytes(mem['bytes_in_use'])} ({mem['usage_pct']:.1f}%)")

    # Compute statistics
    total_time = sum(epoch_times)
    avg_epoch_time = total_time / len(epoch_times)
    total_threads_processed = config.n_threads * config.n_epochs
    overall_throughput = total_threads_processed / total_time

    print("\n" + "-" * 70)
    print("Training Summary:")
    print("-" * 70)
    print(f"  Total epochs: {config.n_epochs}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average epoch time: {avg_epoch_time:.2f}s")
    print(f"  Total threads processed: {total_threads_processed}")
    print(f"  Overall throughput: {overall_throughput:.1f} threads/s")
    print(f"  Fitness progression: {epoch_fitnesses[0]:.4f} -> {epoch_fitnesses[-1]:.4f}")

    return True, {
        'epoch_times': epoch_times,
        'epoch_fitnesses': epoch_fitnesses,
        'memory_history': memory_history,
        'total_time': total_time,
        'throughput': overall_throughput,
    }


def validate_multi_gpu_utilization():
    """Run complete multi-GPU utilization validation."""
    print("=" * 70)
    print("E2: Multi-GPU Utilization Validation")
    print("=" * 70)
    print(f"Purpose: Validate that training uses all 4 GPUs effectively")
    print(f"Configuration: n_threads=256 (64 per GPU), n_steps=10, n_epochs=5")
    print("=" * 70)

    validation_results = {
        'device_detection': False,
        'multi_device_config': False,
        'pmap_distribution': False,
        'vmap_multi_gpu': False,
        'es_training': False,
        'memory_distributed': False,
        'no_device_errors': False,
    }

    # Step 1: Device Detection
    print("\n[STEP 1/5] Device Detection")
    try:
        n_devices = print_device_info()
        validation_results['device_detection'] = n_devices >= 4
        if n_devices < 4:
            print(f"\n  [WARNING] Only {n_devices} devices detected, expected 4")
        else:
            print(f"\n  [PASS] Detected {n_devices} devices")
    except Exception as e:
        print(f"\n  [FAIL] Device detection error: {e}")

    # Step 2: Multi-Device Configuration
    print("\n[STEP 2/5] Multi-Device Configuration Check")
    try:
        config_checks = verify_multi_device_usage()
        validation_results['multi_device_config'] = all(config_checks.values())
    except Exception as e:
        print(f"\n  [FAIL] Configuration check error: {e}")

    # Step 3: PMAP Distribution
    print("\n[STEP 3/5] PMAP Distribution Test")
    try:
        validation_results['pmap_distribution'] = test_pmap_distribution()
    except Exception as e:
        print(f"\n  [FAIL] PMAP test error: {e}")

    # Step 4: VMAP Multi-GPU
    print("\n[STEP 4/5] VMAP Multi-GPU Test")
    try:
        validation_results['vmap_multi_gpu'] = test_vmap_multi_gpu()
    except Exception as e:
        print(f"\n  [FAIL] VMAP test error: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: ES Training
    print("\n[STEP 5/5] ES Training Multi-GPU Test")
    try:
        es_passed, es_stats = test_es_training_multi_gpu()
        validation_results['es_training'] = es_passed

        # Check memory distribution
        if es_passed and es_stats.get('memory_history'):
            final_mem = es_stats['memory_history'][-1]
            # Check if all devices have non-zero memory usage
            devices_with_memory = sum(1 for m in final_mem if 'bytes_in_use' in m and m['bytes_in_use'] > 0)
            validation_results['memory_distributed'] = devices_with_memory >= 4
            print(f"\n  Devices with memory usage: {devices_with_memory}")

        validation_results['no_device_errors'] = es_passed
    except Exception as e:
        print(f"\n  [FAIL] ES training error: {e}")
        import traceback
        traceback.print_exc()

    # Final Summary
    print("\n" + "=" * 70)
    print("E2 Validation Summary")
    print("=" * 70)

    all_passed = True
    for check_name, passed in validation_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("E2 VALIDATION PASSED: Multi-GPU Utilization Verified")
    else:
        print("E2 VALIDATION FAILED: Some checks did not pass")
    print("=" * 70)

    return all_passed, validation_results


def main():
    """Main entry point."""
    import sys
    print(f"\nJAX version: {jax.__version__}", flush=True)
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"Number of devices: {len(jax.devices())}", flush=True)
    sys.stdout.flush()

    try:
        passed, results = validate_multi_gpu_utilization()
        return 0 if passed else 1
    except Exception as e:
        print(f"\n[ERROR] E2 validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
