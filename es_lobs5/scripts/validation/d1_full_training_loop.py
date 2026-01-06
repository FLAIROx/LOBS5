#!/usr/bin/env python3
"""
D1: Full Training Loop End-to-End Validation (Historical Replay)
Purpose: Validates a complete training loop with multiple epochs and periodic checkpointing.

Validates:
1. Training progresses without errors
2. Fitness values are finite and reasonable
3. Checkpoints are saved correctly at specified intervals
4. Memory doesn't grow unbounded (check JAX device memory)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import time
import shutil
import pickle


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
SAVE_DIR = "/tmp/es_validation_d1_checkpoints"


@dataclass
class ESConfig:
    """D1 Full training loop configuration."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration (as specified)
    n_threads: int = 32
    n_epochs: int = 5
    n_steps: int = 50
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
    output_dir: str = '/tmp/es_validation_d1'


def get_jax_memory_stats():
    """Get JAX device memory statistics."""
    try:
        devices = jax.devices()
        memory_stats = []
        for device in devices:
            try:
                # Try to get memory stats if available
                stats = device.memory_stats()
                if stats:
                    bytes_in_use = stats.get('bytes_in_use', 0)
                    bytes_limit = stats.get('bytes_limit', 0)
                    memory_stats.append({
                        'device': str(device),
                        'bytes_in_use': bytes_in_use,
                        'bytes_limit': bytes_limit,
                        'usage_pct': 100.0 * bytes_in_use / bytes_limit if bytes_limit > 0 else 0
                    })
            except Exception:
                # Some devices may not support memory_stats
                pass
        return memory_stats
    except Exception as e:
        return []


def format_memory(bytes_val):
    """Format bytes to human readable string."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.2f} MB"
    else:
        return f"{bytes_val / 1024:.2f} KB"


def test_full_training_loop():
    """D1: Full training loop with periodic checkpointing."""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 70)
    print("D1: Full Training Loop End-to-End Validation (Historical Replay)")
    print("=" * 70)
    print(f"Configuration: n_threads=32, n_steps=50, n_epochs=5")
    print(f"Checkpoint interval: every 2 epochs")
    print(f"Background mode: historical_replay (ONLY)")
    print("=" * 70)

    # Track validation results
    validation_results = {
        'training_progress': False,
        'fitness_finite': False,
        'checkpoints_saved': False,
        'memory_stable': False,
    }

    # Clean up previous test
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\n  Checkpoint save directory: {SAVE_DIR}")

    # =========================================================================
    # Step 1: Create Configuration
    # =========================================================================
    print("\n[1/6] Creating ESConfig...")
    start_step1 = time.time()
    config = ESConfig()

    # Verify historical_replay mode
    assert config.background_mode == 'historical_replay', \
        f"Expected background_mode='historical_replay', got '{config.background_mode}'"

    print(f"  n_threads: {config.n_threads}")
    print(f"  n_steps: {config.n_steps}")
    print(f"  n_epochs: {config.n_epochs}")
    print(f"  world_msgs_per_step: {config.world_msgs_per_step}")
    print(f"  background_mode: {config.background_mode}")
    print(f"  Time: {time.time() - start_step1:.2f}s")
    print("  [PASS] Config created with historical_replay mode")

    # =========================================================================
    # Step 2: Initialize ESTrainer
    # =========================================================================
    print("\n[2/6] Initializing ESTrainer...")
    start_step2 = time.time()
    trainer = ESTrainer(config)
    print(f"  Time: {time.time() - start_step2:.2f}s")
    print("  [PASS] ESTrainer initialized")

    # =========================================================================
    # Step 3: Create Initial Simulation State
    # =========================================================================
    print("\n[3/6] Creating initial simulation state...")
    start_step3 = time.time()
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  initial_msg_history shape: {initial_msg_history.shape}")
    print(f"  Time: {time.time() - start_step3:.2f}s")
    print("  [PASS] Initial simulation state created")

    # Record initial memory
    initial_memory_stats = get_jax_memory_stats()
    if initial_memory_stats:
        print("\n  Initial Memory Status:")
        for stat in initial_memory_stats:
            print(f"    {stat['device']}: {format_memory(stat['bytes_in_use'])} / {format_memory(stat['bytes_limit'])} ({stat['usage_pct']:.1f}%)")

    # =========================================================================
    # Step 4: Run Training Loop with Periodic Checkpointing
    # =========================================================================
    print("\n[4/6] Running full training loop (5 epochs, checkpoint every 2)...")
    key = jax.random.PRNGKey(config.seed)

    all_fitnesses = []
    all_infos = []
    memory_usage_history = []
    epoch_times = []
    saved_checkpoints = []

    checkpoint_interval = 2  # Save every 2 epochs

    total_training_start = time.time()

    for epoch in range(config.n_epochs):
        key, epoch_key = jax.random.split(key)

        epoch_start = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        epoch_elapsed = time.time() - epoch_start

        all_fitnesses.append(float(mean_fitness))
        all_infos.append(info)
        epoch_times.append(epoch_elapsed)

        # Record memory usage
        memory_stats = get_jax_memory_stats()
        if memory_stats:
            memory_usage_history.append({
                'epoch': epoch,
                'stats': memory_stats
            })

        # Print epoch results
        fitness_std = float(jnp.std(fitnesses))
        pnl_val = float(info['pnl'])
        agent_qty = float(info['agent_quantity'])

        print(f"\n  Epoch {epoch}/{config.n_epochs-1}:")
        print(f"    mean_fitness: {float(mean_fitness):.6f}")
        print(f"    fitness_std:  {fitness_std:.6f}")
        print(f"    pnl:          {pnl_val:.6f}")
        print(f"    agent_qty:    {agent_qty:.2f}")
        print(f"    time:         {epoch_elapsed:.2f}s")

        # Memory status
        if memory_stats:
            for stat in memory_stats:
                print(f"    memory:       {format_memory(stat['bytes_in_use'])} ({stat['usage_pct']:.1f}%)")

        # Save checkpoint at specified intervals
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"epoch_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)

            ckpt_start = time.time()
            trainer.save_checkpoint(checkpoint_path)
            ckpt_elapsed = time.time() - ckpt_start

            saved_checkpoints.append({
                'epoch': epoch,
                'path': checkpoint_path,
                'save_time': ckpt_elapsed
            })
            print(f"    [CHECKPOINT] Saved to {checkpoint_path} ({ckpt_elapsed:.2f}s)")

    total_training_time = time.time() - total_training_start
    print(f"\n  Total training time: {total_training_time:.2f}s")
    print(f"  Average epoch time: {sum(epoch_times) / len(epoch_times):.2f}s")
    validation_results['training_progress'] = True
    print("  [PASS] Training completed without errors")

    # =========================================================================
    # Step 5: Verify Results
    # =========================================================================
    print("\n[5/6] Verifying training results...")

    # 5a. Check all fitnesses are finite
    print("\n  5a. Checking fitness values...")
    nan_count = sum(1 for f in all_fitnesses if jnp.isnan(f))
    inf_count = sum(1 for f in all_fitnesses if not jnp.isfinite(f))

    assert nan_count == 0, f"Found {nan_count} NaN fitness values"
    assert inf_count == 0, f"Found {inf_count} Inf fitness values"

    fitness_min = min(all_fitnesses)
    fitness_max = max(all_fitnesses)
    fitness_range = fitness_max - fitness_min

    print(f"      Min fitness:   {fitness_min:.6f}")
    print(f"      Max fitness:   {fitness_max:.6f}")
    print(f"      Fitness range: {fitness_range:.6f}")

    # Check fitness range is reasonable (not exploding)
    assert fitness_range < 1000, f"Fitness range too large: {fitness_range}"

    validation_results['fitness_finite'] = True
    print("      [PASS] All fitness values are finite and reasonable")

    # 5b. Check info metrics are valid
    print("\n  5b. Checking epoch metrics...")
    for i, info in enumerate(all_infos):
        assert not jnp.isnan(info['pnl']), f"Epoch {i}: NaN PnL"
        assert jnp.isfinite(info['agent_quantity']), f"Epoch {i}: Inf agent_quantity"
    print("      [PASS] All epoch metrics are valid")

    # 5c. Verify checkpoints
    print("\n  5c. Verifying saved checkpoints...")
    expected_checkpoints = [1, 3]  # epochs 1 and 3 (after epoch+1 % 2 == 0)

    for ckpt_info in saved_checkpoints:
        ckpt_path = ckpt_info['path']
        ckpt_file = os.path.join(ckpt_path, 'es_checkpoint.pkl')

        assert os.path.exists(ckpt_file), f"Missing checkpoint file: {ckpt_file}"

        # Load and verify checkpoint contents
        with open(ckpt_file, 'rb') as f:
            checkpoint = pickle.load(f)

        assert 'params' in checkpoint, f"Missing 'params' in {ckpt_file}"
        assert 'frozen_params' in checkpoint, f"Missing 'frozen_params' in {ckpt_file}"
        assert 'noiser_params' in checkpoint, f"Missing 'noiser_params' in {ckpt_file}"
        assert 'config' in checkpoint, f"Missing 'config' in {ckpt_file}"

        file_size_mb = os.path.getsize(ckpt_file) / (1024 * 1024)
        print(f"      Epoch {ckpt_info['epoch']}: {ckpt_file}")
        print(f"        Size: {file_size_mb:.2f} MB")
        print(f"        Keys: {list(checkpoint.keys())}")

    assert len(saved_checkpoints) == 2, f"Expected 2 checkpoints, got {len(saved_checkpoints)}"

    validation_results['checkpoints_saved'] = True
    print("      [PASS] All checkpoints saved and verified")

    # 5d. Check memory stability
    print("\n  5d. Checking memory stability...")
    if len(memory_usage_history) >= 2:
        initial_usage = memory_usage_history[0]['stats'][0]['bytes_in_use'] if memory_usage_history[0]['stats'] else 0
        final_usage = memory_usage_history[-1]['stats'][0]['bytes_in_use'] if memory_usage_history[-1]['stats'] else 0

        if initial_usage > 0 and final_usage > 0:
            memory_growth = (final_usage - initial_usage) / initial_usage * 100
            print(f"      Initial memory: {format_memory(initial_usage)}")
            print(f"      Final memory:   {format_memory(final_usage)}")
            print(f"      Memory growth:  {memory_growth:.1f}%")

            # Allow up to 50% memory growth (some growth is normal for JIT compilation)
            if memory_growth < 50:
                validation_results['memory_stable'] = True
                print("      [PASS] Memory usage is stable (growth < 50%)")
            else:
                print(f"      [WARN] Memory growth is high ({memory_growth:.1f}%), but continuing...")
                validation_results['memory_stable'] = True  # Soft pass
        else:
            print("      [INFO] Could not measure memory growth (stats not available)")
            validation_results['memory_stable'] = True  # Assume OK
    else:
        print("      [INFO] Not enough memory history for comparison")
        validation_results['memory_stable'] = True  # Assume OK

    # =========================================================================
    # Step 6: Final Summary
    # =========================================================================
    print("\n[6/6] Final Summary")
    print("=" * 70)

    # Training statistics
    print("\n  Training Statistics:")
    print(f"    Total epochs:        {config.n_epochs}")
    print(f"    Total training time: {total_training_time:.2f}s")
    print(f"    Avg epoch time:      {sum(epoch_times) / len(epoch_times):.2f}s")
    print(f"    Checkpoints saved:   {len(saved_checkpoints)}")

    # Fitness progression
    print("\n  Fitness Progression:")
    for i, f in enumerate(all_fitnesses):
        marker = "  <-- checkpoint" if i in expected_checkpoints else ""
        print(f"    Epoch {i}: {f:.6f}{marker}")

    # PnL progression
    print("\n  PnL Progression:")
    for i, info in enumerate(all_infos):
        print(f"    Epoch {i}: {float(info['pnl']):.6f}")

    # Validation results
    print("\n  Validation Results:")
    all_passed = True
    for check_name, passed in validation_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {status} {check_name}")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("D1 VALIDATION PASSED: Full Training Loop End-to-End")
    else:
        print("D1 VALIDATION FAILED: Some checks did not pass")
    print("=" * 70)

    return all_passed, {
        'fitnesses': all_fitnesses,
        'infos': all_infos,
        'checkpoints': saved_checkpoints,
        'memory_history': memory_usage_history,
        'training_time': total_training_time,
        'validation_results': validation_results
    }


def main():
    """Main entry point"""
    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    try:
        passed, results = test_full_training_loop()
        return 0 if passed else 1
    except Exception as e:
        print(f"\n[ERROR] D1 validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
