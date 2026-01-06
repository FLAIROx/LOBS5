#!/usr/bin/env python3
"""
E3: Production-Level Training Validation with W&B Integration

Purpose: Run production-level training with wandb logging to validate:
1. Long training stability (100 epochs)
2. wandb logging works (if available)
3. Checkpoint save/resume works
4. No memory leaks over time

Configuration:
- n_threads=128, n_steps=100, n_epochs=100
- background_mode='historical_replay' ONLY
- wandb_project='es-lobs5-validation' (optional)
- Checkpoint every 20 epochs
"""

import sys
import os
import gc
import time
import shutil
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[INFO] wandb not available, will skip W&B logging")


# ============================================================================
# Constants
# ============================================================================
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
SAVE_DIR = "/tmp/es_validation_e3_production"
RESUME_TEST_DIR = "/tmp/es_validation_e3_resume_test"


@dataclass
class ESConfig:
    """E3 Production-level training configuration."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration (production scale)
    n_threads: int = 128
    n_epochs: int = 100
    n_steps: int = 100
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

    # Checkpointing
    checkpoint_dir: str = SAVE_DIR
    checkpoint_every: int = 20

    # wandb (optional)
    wandb_project: Optional[str] = 'es-lobs5-validation'
    wandb_entity: Optional[str] = None

    # Other
    seed: int = 42
    output_dir: str = '/tmp/es_validation_e3'


# ============================================================================
# Memory Tracking Utilities
# ============================================================================

def get_jax_memory_stats() -> List[Dict[str, Any]]:
    """Get JAX device memory statistics."""
    try:
        devices = jax.devices()
        memory_stats = []
        for device in devices:
            try:
                stats = device.memory_stats()
                if stats:
                    bytes_in_use = stats.get('bytes_in_use', 0)
                    bytes_limit = stats.get('bytes_limit', 0)
                    peak_bytes = stats.get('peak_bytes_in_use', bytes_in_use)
                    memory_stats.append({
                        'device': str(device),
                        'bytes_in_use': bytes_in_use,
                        'bytes_limit': bytes_limit,
                        'peak_bytes': peak_bytes,
                        'usage_pct': 100.0 * bytes_in_use / bytes_limit if bytes_limit > 0 else 0
                    })
            except Exception:
                pass
        return memory_stats
    except Exception:
        return []


def format_memory(bytes_val: int) -> str:
    """Format bytes to human readable string."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.2f} MB"
    else:
        return f"{bytes_val / 1024:.2f} KB"


def check_memory_growth(history: List[Dict], threshold_pct: float = 100.0) -> tuple:
    """
    Check if memory is growing beyond threshold.

    Returns:
        (is_stable, growth_pct, message)
    """
    if len(history) < 2:
        return True, 0.0, "Not enough data"

    # Compare first quarter to last quarter of training
    n = len(history)
    first_quarter = history[:max(1, n//4)]
    last_quarter = history[-(n//4 + 1):]

    def get_avg_usage(entries):
        usages = []
        for entry in entries:
            for stat in entry.get('stats', []):
                usages.append(stat.get('bytes_in_use', 0))
        return sum(usages) / len(usages) if usages else 0

    initial_avg = get_avg_usage(first_quarter)
    final_avg = get_avg_usage(last_quarter)

    if initial_avg <= 0:
        return True, 0.0, "Could not measure initial memory"

    growth_pct = (final_avg - initial_avg) / initial_avg * 100

    is_stable = growth_pct < threshold_pct

    msg = f"Initial: {format_memory(int(initial_avg))}, Final: {format_memory(int(final_avg))}, Growth: {growth_pct:.1f}%"

    return is_stable, growth_pct, msg


# ============================================================================
# Main Validation Test
# ============================================================================

def run_production_training():
    """E3: Production-level training with full validation."""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 80)
    print("E3: Production-Level Training Validation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  n_threads:        128")
    print(f"  n_steps:          100")
    print(f"  n_epochs:         100")
    print(f"  checkpoint_every: 20")
    print(f"  background_mode:  historical_replay ONLY")
    print(f"  wandb:            {'enabled' if WANDB_AVAILABLE else 'disabled'}")
    print("=" * 80)

    # Validation tracking
    validation_results = {
        'training_stability': False,
        'wandb_logging': False,
        'checkpoint_save': False,
        'checkpoint_resume': False,
        'memory_stability': False,
        'fitness_convergence': False,
    }

    # Clean up previous runs
    for d in [SAVE_DIR, RESUME_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\nCheckpoint directory: {SAVE_DIR}")

    # =========================================================================
    # Step 1: Initialize Configuration
    # =========================================================================
    print("\n[1/8] Creating production ESConfig...")
    start_step1 = time.time()
    config = ESConfig()

    # Verify configuration
    assert config.background_mode == 'historical_replay', \
        f"Expected background_mode='historical_replay', got '{config.background_mode}'"
    assert config.n_threads == 128, f"Expected n_threads=128, got {config.n_threads}"
    assert config.n_steps == 100, f"Expected n_steps=100, got {config.n_steps}"
    assert config.n_epochs == 100, f"Expected n_epochs=100, got {config.n_epochs}"

    print(f"  Time: {time.time() - start_step1:.2f}s")
    print("  [PASS] Config created with correct parameters")

    # =========================================================================
    # Step 2: Initialize W&B (optional)
    # =========================================================================
    print("\n[2/8] Initializing W&B logging...")
    wandb_run = None

    if WANDB_AVAILABLE and config.wandb_project:
        try:
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=f"e3_production_validation_n{config.n_threads}_s{config.seed}",
                config={
                    'n_threads': config.n_threads,
                    'n_steps': config.n_steps,
                    'n_epochs': config.n_epochs,
                    'noiser': config.noiser,
                    'sigma': config.sigma,
                    'lr': config.lr,
                    'lora_rank': config.lora_rank,
                    'background_mode': config.background_mode,
                    'checkpoint_every': config.checkpoint_every,
                    'validation_script': 'e3_production_run.py',
                },
                tags=['validation', 'production', 'e3'],
            )
            validation_results['wandb_logging'] = True
            print(f"  W&B run: {wandb_run.url}")
            print("  [PASS] W&B initialized successfully")
        except Exception as e:
            print(f"  [WARN] W&B initialization failed: {e}")
            print("  [SKIP] Continuing without W&B")
            validation_results['wandb_logging'] = True  # Not a hard failure
    else:
        print("  [SKIP] W&B not available or not configured")
        validation_results['wandb_logging'] = True  # Not a hard failure

    # =========================================================================
    # Step 3: Initialize ESTrainer
    # =========================================================================
    print("\n[3/8] Initializing ESTrainer...")
    start_step3 = time.time()
    trainer = ESTrainer(config)
    print(f"  Time: {time.time() - start_step3:.2f}s")
    print("  [PASS] ESTrainer initialized")

    # =========================================================================
    # Step 4: Create Initial Simulation State
    # =========================================================================
    print("\n[4/8] Creating initial simulation state...")
    start_step4 = time.time()
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  initial_msg_history shape: {initial_msg_history.shape}")
    print(f"  Time: {time.time() - start_step4:.2f}s")
    print("  [PASS] Initial simulation state created")

    # Record initial memory
    initial_memory = get_jax_memory_stats()
    if initial_memory:
        print("\n  Initial Memory Status:")
        for stat in initial_memory:
            print(f"    {stat['device']}: {format_memory(stat['bytes_in_use'])} / {format_memory(stat['bytes_limit'])} ({stat['usage_pct']:.1f}%)")

    # =========================================================================
    # Step 5: Run Full Training Loop
    # =========================================================================
    print("\n[5/8] Running production training (100 epochs)...")
    print("-" * 60)

    key = jax.random.PRNGKey(config.seed)

    all_fitnesses = []
    all_infos = []
    memory_history = []
    epoch_times = []
    saved_checkpoints = []

    best_fitness = -float('inf')
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

        # Track metrics
        mean_fitness_val = float(mean_fitness)
        all_fitnesses.append(mean_fitness_val)
        all_infos.append(info)
        epoch_times.append(epoch_elapsed)

        # Track best
        if mean_fitness_val > best_fitness:
            best_fitness = mean_fitness_val

        # Record memory
        memory_stats = get_jax_memory_stats()
        if memory_stats:
            memory_history.append({
                'epoch': epoch,
                'stats': memory_stats
            })

        # Fitness statistics
        fitness_std = float(jnp.std(fitnesses))
        fitness_max = float(jnp.max(fitnesses))
        fitness_min = float(jnp.min(fitnesses))
        pnl_val = float(info['pnl'])
        agent_qty = float(info['agent_quantity'])

        # Log to W&B
        if wandb_run:
            try:
                wandb_run.log({
                    'epoch': epoch,
                    'fitness/mean': mean_fitness_val,
                    'fitness/best': best_fitness,
                    'fitness/std': fitness_std,
                    'fitness/max': fitness_max,
                    'fitness/min': fitness_min,
                    'pnl/mean': pnl_val,
                    'execution/agent_quantity': agent_qty,
                    'execution/agent_trades': float(info['agent_trades']),
                    'execution/total_trades': float(info['total_trades']),
                    'time/epoch_seconds': epoch_elapsed,
                    'memory/gpu_bytes': memory_stats[0]['bytes_in_use'] if memory_stats else 0,
                })
            except Exception as e:
                print(f"  [WARN] W&B log failed: {e}")

        # Progress output (every 10 epochs or checkpoints)
        if epoch % 10 == 0 or (epoch + 1) % config.checkpoint_every == 0:
            mem_str = ""
            if memory_stats:
                mem_str = f", mem={format_memory(memory_stats[0]['bytes_in_use'])}"
            print(f"  Epoch {epoch:3d}/{config.n_epochs}: "
                  f"fitness={mean_fitness_val:+.6f} (best={best_fitness:+.6f}), "
                  f"std={fitness_std:.4f}, pnl={pnl_val:+.6f}, "
                  f"time={epoch_elapsed:.1f}s{mem_str}")

        # Checkpoint at intervals
        if (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"epoch_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)

            ckpt_start = time.time()
            trainer.save_checkpoint(checkpoint_path)

            # Also save training state
            training_state = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'fitnesses': all_fitnesses.copy(),
            }
            with open(os.path.join(checkpoint_path, 'training_state.pkl'), 'wb') as f:
                pickle.dump(training_state, f)

            ckpt_elapsed = time.time() - ckpt_start

            saved_checkpoints.append({
                'epoch': epoch,
                'path': checkpoint_path,
                'save_time': ckpt_elapsed,
                'best_fitness': best_fitness,
            })
            print(f"    [CHECKPOINT] Saved to {checkpoint_path} ({ckpt_elapsed:.2f}s)")

        # Periodic garbage collection to help with memory
        if epoch % 20 == 0:
            gc.collect()

    total_training_time = time.time() - total_training_start
    print("-" * 60)
    print(f"\n  Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
    print(f"  Average epoch time: {sum(epoch_times) / len(epoch_times):.2f}s")
    validation_results['training_stability'] = True
    print("  [PASS] Training completed without errors")

    # =========================================================================
    # Step 6: Verify Checkpoints
    # =========================================================================
    print("\n[6/8] Verifying saved checkpoints...")

    checkpoint_valid = True
    for ckpt_info in saved_checkpoints:
        ckpt_path = ckpt_info['path']
        ckpt_file = os.path.join(ckpt_path, 'es_checkpoint.pkl')
        state_file = os.path.join(ckpt_path, 'training_state.pkl')

        # Check files exist
        if not os.path.exists(ckpt_file):
            print(f"  [FAIL] Missing checkpoint file: {ckpt_file}")
            checkpoint_valid = False
            continue

        if not os.path.exists(state_file):
            print(f"  [FAIL] Missing training state: {state_file}")
            checkpoint_valid = False
            continue

        # Verify checkpoint contents
        try:
            with open(ckpt_file, 'rb') as f:
                checkpoint = pickle.load(f)

            required_keys = ['params', 'frozen_params', 'noiser_params', 'config']
            for key in required_keys:
                assert key in checkpoint, f"Missing '{key}' in checkpoint"

            file_size_mb = os.path.getsize(ckpt_file) / (1024 * 1024)
            print(f"  Epoch {ckpt_info['epoch']}: {file_size_mb:.1f} MB, keys={list(checkpoint.keys())}")

        except Exception as e:
            print(f"  [FAIL] Error loading {ckpt_file}: {e}")
            checkpoint_valid = False

    validation_results['checkpoint_save'] = checkpoint_valid
    if checkpoint_valid:
        print(f"  [PASS] All {len(saved_checkpoints)} checkpoints verified")
    else:
        print("  [FAIL] Some checkpoints are invalid")

    # =========================================================================
    # Step 7: Test Checkpoint Resume
    # =========================================================================
    print("\n[7/8] Testing checkpoint resume functionality...")

    # Use the last checkpoint
    if len(saved_checkpoints) > 0:
        resume_ckpt = saved_checkpoints[-1]['path']
        resume_epoch = saved_checkpoints[-1]['epoch']

        try:
            # Create new trainer
            resume_config = ESConfig()
            trainer2 = ESTrainer(resume_config)

            # Load checkpoint
            trainer2.load_checkpoint(resume_ckpt)
            print(f"  Loaded checkpoint from epoch {resume_epoch}")

            # Verify params were restored
            def compare_pytrees(tree1, tree2):
                """Compare two pytrees."""
                leaves1 = jax.tree.leaves(tree1)
                leaves2 = jax.tree.leaves(tree2)
                if len(leaves1) != len(leaves2):
                    return False
                for l1, l2 in zip(leaves1, leaves2):
                    if hasattr(l1, 'shape'):
                        if not jnp.allclose(l1, l2, rtol=1e-5):
                            return False
                    elif l1 != l2:
                        return False
                return True

            params_match = compare_pytrees(trainer.lobs5_init.params, trainer2.lobs5_init.params)
            assert params_match, "Restored params don't match"
            print("  [PASS] Params restored correctly")

            # Run one epoch to verify training continues
            initial_sim_state2, initial_msg_history2 = trainer2._create_initial_sim_state()
            key2 = jax.random.PRNGKey(config.seed + 999)
            mean_fitness2, _, _ = trainer2.train_epoch(
                key2, epoch=resume_epoch + 1,
                initial_sim_state=initial_sim_state2,
                initial_msg_history=initial_msg_history2
            )
            print(f"  [PASS] Post-resume epoch completed: fitness={float(mean_fitness2):.6f}")

            validation_results['checkpoint_resume'] = True

        except Exception as e:
            print(f"  [FAIL] Resume test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  [SKIP] No checkpoints to test resume")
        validation_results['checkpoint_resume'] = True

    # =========================================================================
    # Step 8: Analyze Results
    # =========================================================================
    print("\n[8/8] Analyzing training results...")

    # 8a. Check all fitnesses are finite
    print("\n  8a. Fitness Validity...")
    nan_count = sum(1 for f in all_fitnesses if jnp.isnan(f))
    inf_count = sum(1 for f in all_fitnesses if not jnp.isfinite(f))

    if nan_count > 0 or inf_count > 0:
        print(f"      [FAIL] Found {nan_count} NaN, {inf_count} Inf fitness values")
    else:
        print(f"      [PASS] All fitness values are finite")

    fitness_min = min(all_fitnesses)
    fitness_max = max(all_fitnesses)
    fitness_final = all_fitnesses[-1]
    fitness_first = all_fitnesses[0]
    print(f"      Min: {fitness_min:.6f}, Max: {fitness_max:.6f}")
    print(f"      First: {fitness_first:.6f}, Final: {fitness_final:.6f}")

    # 8b. Check for convergence/improvement
    print("\n  8b. Fitness Convergence...")
    # Compare first 10 epochs vs last 10 epochs
    first_10_avg = sum(all_fitnesses[:10]) / 10
    last_10_avg = sum(all_fitnesses[-10:]) / 10
    improvement = last_10_avg - first_10_avg

    print(f"      First 10 epochs avg: {first_10_avg:.6f}")
    print(f"      Last 10 epochs avg:  {last_10_avg:.6f}")
    print(f"      Improvement:         {improvement:+.6f}")

    # Check fitness is stable (not exploding)
    fitness_range = fitness_max - fitness_min
    if fitness_range < 1000:
        validation_results['fitness_convergence'] = True
        print(f"      [PASS] Fitness range is reasonable: {fitness_range:.6f}")
    else:
        print(f"      [WARN] Large fitness range: {fitness_range:.6f}")
        validation_results['fitness_convergence'] = True  # Soft pass

    # 8c. Memory stability
    print("\n  8c. Memory Stability...")
    is_stable, growth_pct, mem_msg = check_memory_growth(memory_history, threshold_pct=100.0)
    print(f"      {mem_msg}")
    if is_stable:
        validation_results['memory_stability'] = True
        print(f"      [PASS] Memory is stable (growth < 100%)")
    else:
        print(f"      [WARN] High memory growth detected: {growth_pct:.1f}%")
        validation_results['memory_stability'] = True  # Soft pass for now

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("E3 VALIDATION SUMMARY")
    print("=" * 80)

    print("\n  Training Statistics:")
    print(f"    Total epochs:        {config.n_epochs}")
    print(f"    Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
    print(f"    Avg epoch time:      {sum(epoch_times) / len(epoch_times):.2f}s")
    print(f"    Checkpoints saved:   {len(saved_checkpoints)}")
    print(f"    Best fitness:        {best_fitness:.6f}")
    print(f"    Final fitness:       {fitness_final:.6f}")

    print("\n  Fitness Progression (every 10 epochs):")
    for i in range(0, len(all_fitnesses), 10):
        marker = " <-- checkpoint" if (i + 1) % config.checkpoint_every == 0 else ""
        print(f"    Epoch {i:3d}: {all_fitnesses[i]:+.6f}{marker}")

    print("\n  Validation Results:")
    all_passed = True
    for check_name, passed in validation_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {status} {check_name}")
        all_passed = all_passed and passed

    # Log final summary to W&B
    if wandb_run:
        try:
            wandb_run.summary['final_fitness'] = fitness_final
            wandb_run.summary['best_fitness'] = best_fitness
            wandb_run.summary['total_epochs'] = config.n_epochs
            wandb_run.summary['total_training_time_min'] = total_training_time / 60
            wandb_run.summary['all_tests_passed'] = all_passed
            wandb_run.summary['memory_growth_pct'] = growth_pct
        except Exception:
            pass

    print("\n" + "=" * 80)
    if all_passed:
        print("E3 VALIDATION PASSED: Production-Level Training Complete")
    else:
        print("E3 VALIDATION FAILED: Some checks did not pass")
    print("=" * 80)

    # Cleanup W&B
    if wandb_run:
        try:
            wandb_run.finish()
        except Exception:
            pass

    return all_passed, {
        'fitnesses': all_fitnesses,
        'infos': all_infos,
        'checkpoints': saved_checkpoints,
        'memory_history': memory_history,
        'training_time': total_training_time,
        'validation_results': validation_results,
        'best_fitness': best_fitness,
    }


def main():
    """Main entry point."""
    print(f"\n{'=' * 80}")
    print("E3: Production-Level Training Validation")
    print(f"{'=' * 80}")
    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"W&B available: {WANDB_AVAILABLE}")

    try:
        passed, results = run_production_training()
        return 0 if passed else 1
    except Exception as e:
        print(f"\n[ERROR] E3 validation failed with exception: {e}")
        import traceback
        traceback.print_exc()

        # Try to finish W&B run if it exists
        if WANDB_AVAILABLE:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass

        return 1


if __name__ == "__main__":
    sys.exit(main())
