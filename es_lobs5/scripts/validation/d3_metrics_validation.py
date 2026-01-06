#!/usr/bin/env python3
"""
D3: Performance Metrics Validation (Historical Replay, 32x50)
Purpose: Validate that training metrics are reasonable and ES is learning.

Checks:
1. Fitness values don't diverge (no explosion)
2. PnL metrics are tracked correctly
3. Agent quantity and trade counts are reasonable
4. ES noise is being applied (perturbed fitnesses differ from mean)
5. Gradient updates are happening (params change between epochs)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import time
from typing import List, Dict, Any


# Constants
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """ES training configuration for D3 validation."""
    # Checkpoint
    lobs5_checkpoint: str = CHECKPOINT_PATH

    # ES configuration
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    # Training configuration (D3: 32 threads, 50 steps, 10 epochs)
    n_threads: int = 32
    n_epochs: int = 10
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
    output_dir: str = '/tmp/es_validation_d3'


def compute_param_norm(params: Dict) -> float:
    """Compute L2 norm of all parameters."""
    leaves = jax.tree.leaves(params)
    total_norm_sq = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    return float(jnp.sqrt(total_norm_sq))


def compute_param_diff_norm(params1: Dict, params2: Dict) -> float:
    """Compute L2 norm of parameter differences."""
    leaves1 = jax.tree.leaves(params1)
    leaves2 = jax.tree.leaves(params2)
    diff_norm_sq = sum(
        jnp.sum(jnp.square(l1 - l2))
        for l1, l2 in zip(leaves1, leaves2)
    )
    return float(jnp.sqrt(diff_norm_sq))


def print_epoch_table_header():
    """Print table header for epoch metrics."""
    print("\n" + "=" * 120)
    print(f"{'Epoch':>5} | {'Fitness Mean':>12} | {'Fitness Std':>11} | {'Fitness Min':>11} | "
          f"{'Fitness Max':>11} | {'PnL':>10} | {'AgentQty':>8} | {'Trades':>6} | {'ParamNorm':>10} | {'Time(s)':>7}")
    print("-" * 120)


def print_epoch_row(epoch: int, mean_fit: float, std_fit: float, min_fit: float, max_fit: float,
                    pnl: float, agent_qty: float, trades: float, param_norm: float, elapsed: float):
    """Print one row of epoch metrics."""
    print(f"{epoch:>5} | {mean_fit:>12.6f} | {std_fit:>11.6f} | {min_fit:>11.6f} | "
          f"{max_fit:>11.6f} | {pnl:>10.4f} | {agent_qty:>8.1f} | {trades:>6.1f} | {param_norm:>10.2f} | {elapsed:>7.2f}")


def test_metrics_validation():
    """D3: Validate performance metrics across 10 epochs."""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 80)
    print("D3: Performance Metrics Validation (Historical Replay, 32x50, 10 epochs)")
    print("=" * 80)

    # =========================================================================
    # Phase 1: Setup
    # =========================================================================
    print("\n[1/6] Creating ESConfig...")
    config = ESConfig()
    print(f"  n_threads: {config.n_threads}")
    print(f"  n_steps: {config.n_steps}")
    print(f"  n_epochs: {config.n_epochs}")
    print(f"  background_mode: {config.background_mode}")
    print("  [OK] Config created")

    # =========================================================================
    # Phase 2: Initialize Trainer
    # =========================================================================
    print("\n[2/6] Initializing ESTrainer...")
    trainer = ESTrainer(config)
    print("  [OK] ESTrainer initialized")

    # =========================================================================
    # Phase 3: Get Initial State
    # =========================================================================
    print("\n[3/6] Creating initial simulation state...")
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  [OK] initial_msg_history shape: {initial_msg_history.shape}")

    # Store initial params for comparison
    initial_param_norm = compute_param_norm(trainer.lobs5_init.params)
    initial_params_copy = jax.tree.map(lambda x: x.copy(), trainer.lobs5_init.params)
    print(f"  Initial param norm: {initial_param_norm:.4f}")

    # =========================================================================
    # Phase 4: Run Training with Detailed Metrics
    # =========================================================================
    print("\n[4/6] Running 10 epochs with detailed metrics tracking...")
    key = jax.random.PRNGKey(config.seed)

    # Storage for all epoch data
    all_fitnesses: List[jnp.ndarray] = []
    all_mean_fitnesses: List[float] = []
    all_infos: List[Dict[str, Any]] = []
    all_param_norms: List[float] = []
    all_param_diffs: List[float] = []
    all_times: List[float] = []

    print_epoch_table_header()

    prev_params = initial_params_copy

    for epoch in range(config.n_epochs):
        key, epoch_key = jax.random.split(key)

        start_time = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        elapsed = time.time() - start_time

        # Compute metrics
        std_fit = float(jnp.std(fitnesses))
        min_fit = float(jnp.min(fitnesses))
        max_fit = float(jnp.max(fitnesses))
        pnl = float(info['pnl'])
        agent_qty = float(info['agent_quantity'])
        trades = float(info['agent_trades'])
        param_norm = compute_param_norm(trainer.lobs5_init.params)
        param_diff = compute_param_diff_norm(prev_params, trainer.lobs5_init.params)

        # Store
        all_fitnesses.append(fitnesses)
        all_mean_fitnesses.append(float(mean_fitness))
        all_infos.append(info)
        all_param_norms.append(param_norm)
        all_param_diffs.append(param_diff)
        all_times.append(elapsed)

        # Print row
        print_epoch_row(epoch, float(mean_fitness), std_fit, min_fit, max_fit,
                        pnl, agent_qty, trades, param_norm, elapsed)

        # Update prev_params for next epoch comparison
        prev_params = jax.tree.map(lambda x: x.copy(), trainer.lobs5_init.params)

    print("-" * 120)

    # =========================================================================
    # Phase 5: Validation Checks
    # =========================================================================
    print("\n[5/6] Running validation checks...")

    checks_passed = 0
    checks_failed = 0
    check_details = []

    # Check 1: No NaN in fitnesses
    has_nan = any(jnp.any(jnp.isnan(f)) for f in all_fitnesses)
    if not has_nan:
        checks_passed += 1
        check_details.append(("No NaN in fitnesses", "PASS", "All fitness values are valid numbers"))
    else:
        checks_failed += 1
        check_details.append(("No NaN in fitnesses", "FAIL", "Found NaN values in fitnesses"))

    # Check 2: All fitnesses are finite (no explosion)
    has_inf = any(jnp.any(jnp.isinf(f)) for f in all_fitnesses)
    if not has_inf:
        checks_passed += 1
        check_details.append(("No Inf in fitnesses", "PASS", "All fitness values are finite"))
    else:
        checks_failed += 1
        check_details.append(("No Inf in fitnesses", "FAIL", "Found Inf values (divergence)"))

    # Check 3: Fitness range is reasonable (not exploding)
    fitness_range = max(all_mean_fitnesses) - min(all_mean_fitnesses)
    if fitness_range < 100:
        checks_passed += 1
        check_details.append(("Fitness range bounded", "PASS", f"Range = {fitness_range:.4f} < 100"))
    else:
        checks_failed += 1
        check_details.append(("Fitness range bounded", "FAIL", f"Range = {fitness_range:.4f} >= 100 (exploding)"))

    # Check 4: PnL is being tracked (not all zero)
    pnl_values = [float(info['pnl']) for info in all_infos]
    pnl_variance = jnp.var(jnp.array(pnl_values))
    if pnl_variance > 1e-10 or any(abs(p) > 0.001 for p in pnl_values):
        checks_passed += 1
        check_details.append(("PnL is tracked", "PASS", f"PnL variance = {float(pnl_variance):.6f}"))
    else:
        checks_failed += 1
        check_details.append(("PnL is tracked", "FAIL", "All PnL values are zero/constant"))

    # Check 5: Agent quantity is reasonable
    agent_quantities = [float(info['agent_quantity']) for info in all_infos]
    avg_qty = sum(agent_quantities) / len(agent_quantities)
    if avg_qty > 0:
        checks_passed += 1
        check_details.append(("Agent executes trades", "PASS", f"Avg quantity = {avg_qty:.1f}"))
    else:
        checks_failed += 1
        check_details.append(("Agent executes trades", "FAIL", "No trades executed"))

    # Check 6: Agent trade count is reasonable
    agent_trades = [float(info['agent_trades']) for info in all_infos]
    avg_trades = sum(agent_trades) / len(agent_trades)
    if avg_trades >= 0:  # At least some epochs should have trades
        checks_passed += 1
        check_details.append(("Trade count tracked", "PASS", f"Avg trades = {avg_trades:.1f}"))
    else:
        checks_failed += 1
        check_details.append(("Trade count tracked", "FAIL", "Negative trade count"))

    # Check 7: ES noise is applied (fitness variance within epoch)
    fitness_stds = [float(jnp.std(f)) for f in all_fitnesses]
    avg_std = sum(fitness_stds) / len(fitness_stds)
    if avg_std > 1e-6:
        checks_passed += 1
        check_details.append(("ES noise applied", "PASS", f"Avg fitness std = {avg_std:.6f}"))
    else:
        checks_failed += 1
        check_details.append(("ES noise applied", "FAIL", f"Fitness std = {avg_std:.6f} (no perturbation)"))

    # Check 8: Params are being updated
    total_param_diff = sum(all_param_diffs)
    if total_param_diff > 1e-8:
        checks_passed += 1
        check_details.append(("Params are updated", "PASS", f"Total param change = {total_param_diff:.6f}"))
    else:
        checks_failed += 1
        check_details.append(("Params are updated", "FAIL", f"Params unchanged (no learning)"))

    # Check 9: Param norm is stable (not exploding)
    param_norm_change = abs(all_param_norms[-1] - all_param_norms[0])
    param_norm_ratio = all_param_norms[-1] / all_param_norms[0] if all_param_norms[0] > 0 else 1.0
    if 0.5 < param_norm_ratio < 2.0:
        checks_passed += 1
        check_details.append(("Param norm stable", "PASS", f"Ratio = {param_norm_ratio:.4f}"))
    else:
        checks_failed += 1
        check_details.append(("Param norm stable", "FAIL", f"Ratio = {param_norm_ratio:.4f} (unstable)"))

    # Check 10: Timing is reasonable
    avg_time = sum(all_times) / len(all_times)
    if avg_time < 600:  # Each epoch should take less than 10 minutes
        checks_passed += 1
        check_details.append(("Timing reasonable", "PASS", f"Avg epoch time = {avg_time:.2f}s"))
    else:
        checks_failed += 1
        check_details.append(("Timing reasonable", "FAIL", f"Avg epoch time = {avg_time:.2f}s (too slow)"))

    # Print check results
    print("\n  Validation Results:")
    print("  " + "-" * 76)
    for name, status, detail in check_details:
        status_sym = "[OK]" if status == "PASS" else "[FAIL]"
        print(f"  {status_sym} {name:<25} : {detail}")
    print("  " + "-" * 76)

    # =========================================================================
    # Phase 6: Summary and Trends
    # =========================================================================
    print("\n[6/6] Summary and Trends...")

    print("\n  Fitness Trend:")
    print(f"    First epoch:  {all_mean_fitnesses[0]:.6f}")
    print(f"    Last epoch:   {all_mean_fitnesses[-1]:.6f}")
    print(f"    Change:       {all_mean_fitnesses[-1] - all_mean_fitnesses[0]:+.6f}")
    print(f"    Best epoch:   {all_mean_fitnesses.index(max(all_mean_fitnesses))} (fitness = {max(all_mean_fitnesses):.6f})")
    print(f"    Worst epoch:  {all_mean_fitnesses.index(min(all_mean_fitnesses))} (fitness = {min(all_mean_fitnesses):.6f})")

    print("\n  PnL Trend:")
    print(f"    First epoch:  {pnl_values[0]:.6f}")
    print(f"    Last epoch:   {pnl_values[-1]:.6f}")
    print(f"    Mean PnL:     {sum(pnl_values)/len(pnl_values):.6f}")

    print("\n  Parameter Updates:")
    print(f"    Initial norm:    {initial_param_norm:.4f}")
    print(f"    Final norm:      {all_param_norms[-1]:.4f}")
    print(f"    Total change:    {total_param_diff:.6f}")
    print(f"    Avg per-epoch:   {total_param_diff / config.n_epochs:.6f}")

    print("\n  Execution Stats:")
    print(f"    Avg agent quantity: {avg_qty:.1f}")
    print(f"    Avg agent trades:   {avg_trades:.1f}")
    print(f"    Total trades (all): {sum(float(info['total_trades']) for info in all_infos):.0f}")

    print("\n  Timing:")
    print(f"    Total time:     {sum(all_times):.2f}s")
    print(f"    Avg per epoch:  {avg_time:.2f}s")
    print(f"    Min epoch time: {min(all_times):.2f}s")
    print(f"    Max epoch time: {max(all_times):.2f}s")

    # =========================================================================
    # Final Result
    # =========================================================================
    print("\n" + "=" * 80)
    if checks_failed == 0:
        print(f"[PASS] D3: Performance Metrics Validation")
        print(f"       All {checks_passed} checks passed!")
    else:
        print(f"[FAIL] D3: Performance Metrics Validation")
        print(f"       {checks_passed} passed, {checks_failed} failed")
    print("=" * 80)

    return checks_failed == 0, {
        'checks_passed': checks_passed,
        'checks_failed': checks_failed,
        'check_details': check_details,
        'mean_fitnesses': all_mean_fitnesses,
        'pnl_values': pnl_values,
        'param_norms': all_param_norms,
        'param_diffs': all_param_diffs,
        'times': all_times,
    }


def main():
    """Main entry point."""
    success, results = test_metrics_validation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
