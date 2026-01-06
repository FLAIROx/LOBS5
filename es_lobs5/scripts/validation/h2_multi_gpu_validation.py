#!/usr/bin/env python3
"""
H2: Multi-GPU shard_map Validation Test

PASS Criteria:
1. all_gpus_detected: >= 4 GPUs
2. memory_distributed: All GPUs have memory > 50MB
3. memory_balanced: Memory ratio < 2.0
4. training_success: Training completes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
import time
from dataclasses import dataclass

CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """H2 test configuration - 256 threads across 4 GPUs."""
    lobs5_checkpoint: str = CHECKPOINT_PATH
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    n_threads: int = 256  # 64 per GPU
    n_epochs: int = 3
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
    output_dir: str = '/tmp/es_validation_h2'


def get_gpu_memory():
    """Get memory usage for all GPUs."""
    memory = []
    for i, device in enumerate(jax.devices()):
        if device.platform == 'gpu':
            try:
                stats = device.memory_stats()
                memory.append(stats.get('bytes_in_use', 0) if stats else 0)
            except:
                memory.append(0)
    return memory


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


def test_multi_gpu_distribution():
    """H2: Verify shard_map distributes across GPUs."""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 70, flush=True)
    print("H2: Multi-GPU shard_map Validation", flush=True)
    print("=" * 70, flush=True)

    results = {
        'all_gpus_detected': False,
        'memory_distributed': False,
        'memory_balanced': False,
        'training_success': False,
    }

    # Check devices
    devices = jax.devices()
    n_gpus = len([d for d in devices if d.platform == 'gpu'])
    print(f"\n[1/4] GPU Detection: {n_gpus} GPUs", flush=True)
    for i, d in enumerate(devices):
        print(f"  Device {i}: {d.platform} - {d.device_kind}", flush=True)
    results['all_gpus_detected'] = n_gpus >= 4

    # Initialize trainer
    print("\n[2/4] Initializing ESTrainer...", flush=True)
    config = ESConfig()
    print(f"  n_threads: {config.n_threads}", flush=True)
    print(f"  n_epochs: {config.n_epochs}", flush=True)
    print(f"  n_steps: {config.n_steps}", flush=True)
    print(f"  background_mode: {config.background_mode}", flush=True)

    trainer = ESTrainer(config)

    # Check mesh exists
    if hasattr(trainer, '_mesh'):
        print(f"\n  Mesh Configuration:", flush=True)
        print(f"    Mesh: {trainer._mesh}", flush=True)
        print(f"    Axis names: {trainer._mesh.axis_names}", flush=True)
        print(f"    Devices in mesh: {trainer._n_devices}", flush=True)
    else:
        print("  [WARN] No mesh configured", flush=True)

    # Create initial state
    print("\n[3/4] Creating initial simulation state...", flush=True)
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  Message history shape: {initial_msg_history.shape}", flush=True)

    # Record initial memory
    print("\n  Initial memory state:", flush=True)
    initial_memory = get_gpu_memory()
    for i, mem in enumerate(initial_memory):
        print(f"    GPU {i}: {format_bytes(mem)}", flush=True)

    # Run training
    print(f"\n[4/4] Running {config.n_epochs} epochs...", flush=True)
    key = jax.random.PRNGKey(config.seed)

    epoch_times = []
    try:
        for epoch in range(config.n_epochs):
            key, epoch_key = jax.random.split(key)

            epoch_start = time.time()
            mean_fitness, fitnesses, _ = trainer.train_epoch(
                epoch_key, epoch, initial_sim_state, initial_msg_history
            )
            jax.block_until_ready(fitnesses)
            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(epoch_elapsed)

            # Get memory during training
            current_memory = get_gpu_memory()

            print(f"\n  Epoch {epoch}:", flush=True)
            print(f"    Mean fitness: {float(mean_fitness):.6f}", flush=True)
            print(f"    Time: {epoch_elapsed:.2f}s", flush=True)
            print(f"    Memory per GPU:", flush=True)
            for i, mem in enumerate(current_memory):
                print(f"      GPU {i}: {format_bytes(mem)}", flush=True)

        results['training_success'] = True
        print("\n  [PASS] Training completed successfully", flush=True)
    except Exception as e:
        print(f"\n  [ERROR] Training failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

    # Check memory distribution
    print("\n" + "=" * 70, flush=True)
    print("Memory Distribution Analysis", flush=True)
    print("=" * 70, flush=True)

    memory = get_gpu_memory()
    print("\n  Final memory per GPU:", flush=True)
    for i, mem in enumerate(memory):
        print(f"    GPU {i}: {format_bytes(mem)}", flush=True)

    # Check all GPUs have memory > 50MB
    min_threshold = 50 * 1e6  # 50 MB
    gpus_above_threshold = sum(1 for m in memory if m > min_threshold)
    results['memory_distributed'] = gpus_above_threshold >= 4

    if results['memory_distributed']:
        print(f"\n  [PASS] All {gpus_above_threshold} GPUs have memory > 50MB", flush=True)
    else:
        print(f"\n  [FAIL] Only {gpus_above_threshold} GPUs have memory > 50MB", flush=True)
        print(f"    (Required: 4 GPUs with > 50MB)", flush=True)

    # Check balance
    valid_memory = [m for m in memory if m > 0]
    if len(valid_memory) >= 2:
        ratio = max(valid_memory) / min(valid_memory)
        print(f"\n  Memory Balance Check:", flush=True)
        print(f"    Max memory: {format_bytes(max(valid_memory))}", flush=True)
        print(f"    Min memory: {format_bytes(min(valid_memory))}", flush=True)
        print(f"    Ratio (max/min): {ratio:.2f}", flush=True)
        results['memory_balanced'] = ratio < 2.0
        if results['memory_balanced']:
            print(f"    [PASS] Ratio {ratio:.2f} < 2.0 (balanced)", flush=True)
        else:
            print(f"    [FAIL] Ratio {ratio:.2f} >= 2.0 (imbalanced)", flush=True)
    else:
        print(f"\n  Memory Balance Check: SKIP (fewer than 2 GPUs with memory)", flush=True)
        results['memory_balanced'] = False

    # Training performance summary
    print("\n" + "=" * 70, flush=True)
    print("Training Performance Summary", flush=True)
    print("=" * 70, flush=True)
    if epoch_times:
        avg_time = sum(epoch_times) / len(epoch_times)
        # First epoch includes compilation
        compile_time = epoch_times[0] if len(epoch_times) > 0 else 0
        cached_time = sum(epoch_times[1:]) / len(epoch_times[1:]) if len(epoch_times) > 1 else 0
        print(f"  Total epochs: {len(epoch_times)}", flush=True)
        print(f"  Epoch 0 (w/ compile): {compile_time:.2f}s", flush=True)
        print(f"  Avg epoch (cached): {cached_time:.2f}s", flush=True)
        print(f"  Threads per epoch: {config.n_threads}", flush=True)
        if cached_time > 0:
            print(f"  Throughput (cached): {config.n_threads / cached_time:.1f} threads/s", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("H2 Validation Summary", flush=True)
    print("=" * 70, flush=True)

    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check}", flush=True)

    all_passed = all(results.values())
    print("\n" + "=" * 70, flush=True)
    print(f"H2 {'PASSED' if all_passed else 'FAILED'}", flush=True)
    print("=" * 70, flush=True)

    return all_passed


def main():
    print(f"JAX version: {jax.__version__}", flush=True)
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"Number of devices: {len(jax.devices())}", flush=True)
    sys.stdout.flush()

    try:
        passed = test_multi_gpu_distribution()
        return 0 if passed else 1
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
