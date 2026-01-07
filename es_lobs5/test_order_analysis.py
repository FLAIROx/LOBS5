"""
Order Quality Analysis Script - Real ES Training Version

Extracts real policy-generated orders from ES training (info['policy_msgs'])
and compares with historical orders from replay data.

Usage: sbatch test_order_analysis.sh
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# Setup paths
LOBS5_ROOT = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5'
sys.path.insert(0, LOBS5_ROOT)
sys.path.insert(0, os.path.join(LOBS5_ROOT, 'HyperscaleES', 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import argparse

# Local imports
from lob.encoding import decode_msgs, Vocab
from es_lobs5.analysis import (
    compute_order_stats,
    print_stats_comparison,
    get_raw_order_data,
)
from es_lobs5.analysis.visualize import plot_comparison, plot_detailed_analysis


@dataclass
class ESConfig:
    """Minimal config for ES Trainer order analysis."""
    # Checkpoint
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/wandb/run-20241130_101652-logical-serenity-19/files/checkpoints'

    # ES configuration (minimal - just for episode collection)
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    group_size: int = 0  # No baseline subtraction for testing

    # Training configuration (minimal)
    n_perturbations: int = 4  # Small for testing
    n_epochs: int = 1
    n_steps: int = 50  # Steps per episode
    background_msgs_per_step: int = 10

    # Token mode
    token_mode: int = 24

    # Background mode - use historical replay for realistic testing
    background_mode: str = 'historical_replay'
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG/2021'

    # Task
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100

    # Training stability
    grad_clip: float = 1.0

    # Other
    seed: int = 42
    n_warmup_msgs: int = 500


def collect_policy_orders_from_es(config: ESConfig, n_episodes: int = 5) -> np.ndarray:
    """
    Run ES episodes and collect real policy-generated orders.

    Args:
        config: ESConfig with trainer settings
        n_episodes: Number of episodes to run

    Returns:
        Array of shape (n_episodes * n_steps, msg_len) containing policy tokens
    """
    from es_lobs5.training.es_trainer import ESTrainer

    print(f"[*] Initializing ESTrainer for order collection...")
    trainer = ESTrainer(config)

    all_policy_msgs = []
    all_fitnesses = []

    for ep in range(n_episodes):
        key = jax.random.PRNGKey(config.seed + ep * 100)

        # Get initial state from replay data
        initial_state, initial_msg_history = trainer.get_episode_initial_state(
            key,
            start_idx=ep * 1000  # Different starting points in replay data
        )

        # Run episode with thread_id=0 (baseline, no noise perturbation)
        # This gives us the policy's "mean" behavior
        fitness, info = trainer.eval_single_thread(
            key,
            thread_id=0,  # Baseline thread
            epoch=0,
            initial_sim_state=initial_state,
            initial_msg_history=initial_msg_history,
        )

        # Extract policy messages
        if 'policy_msgs' in info:
            policy_msgs = np.array(info['policy_msgs'])
            all_policy_msgs.append(policy_msgs)
            all_fitnesses.append(float(fitness))
            print(f"[*] Episode {ep}: collected {len(policy_msgs)} msgs, "
                  f"fitness={float(fitness):.4f}, pnl_raw={float(info.get('pnl_raw', 0)):.2f}")
        else:
            print(f"[!] Episode {ep}: no policy_msgs in info dict")
            print(f"[!] info keys: {list(info.keys())}")

    if all_policy_msgs:
        print(f"\n[*] Mean fitness across {len(all_fitnesses)} episodes: {np.mean(all_fitnesses):.4f}")
        return np.concatenate(all_policy_msgs, axis=0)
    else:
        raise ValueError("No policy messages collected from ES training")


def get_historical_orders_from_replay(data_path: str, n_samples: int = 1000, token_mode: int = 24) -> np.ndarray:
    """
    Load historical orders from encoded replay data.

    Args:
        data_path: Path to encoded data directory
        n_samples: Number of orders to extract
        token_mode: 22 or 24 token format

    Returns:
        Array of shape (n_samples, msg_len) containing historical tokens
    """
    import glob

    data_files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
    if not data_files:
        raise FileNotFoundError(f"No .npy files found in {data_path}")

    # Load first file
    print(f"[*] Loading historical data from: {data_files[0]}")
    encoded_data = np.load(data_files[0])
    print(f"[*] Encoded data shape: {encoded_data.shape}")

    # Extract samples
    if encoded_data.ndim == 2:
        historical_tokens = encoded_data[:n_samples, :]
    else:
        historical_tokens = encoded_data.reshape(-1, encoded_data.shape[-1])[:n_samples, :]

    return historical_tokens


def main():
    parser = argparse.ArgumentParser(description='Order Quality Analysis with Real ES Training')
    parser.add_argument('--checkpoint', type=str,
                       default='/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/wandb/run-20241130_101652-logical-serenity-19/files/checkpoints',
                       help='Path to LOBS5 checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG/2021',
                       help='Path to encoded data directory')
    parser.add_argument('--n_historical', type=int, default=2000,
                       help='Number of historical orders to analyze')
    parser.add_argument('--n_episodes', type=int, default=5,
                       help='Number of ES episodes to collect policy orders')
    parser.add_argument('--n_steps', type=int, default=50,
                       help='Steps per episode')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                       help='Output directory for reports and charts')
    parser.add_argument('--token_mode', type=int, default=24,
                       help='Token mode (22 or 24)')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic data instead of real ES training (for quick testing)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 60)
    print(" Order Quality Analysis - Real ES Training")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Token mode: {args.token_mode}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Steps/episode: {args.n_steps}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode: {'Synthetic' if args.use_synthetic else 'Real ES Training'}")
    print("=" * 60)

    # Initialize vocab for decoding
    v = Vocab(token_mode=args.token_mode)

    # ========================================
    # Step 1: Load historical orders
    # ========================================
    print("\n[Step 1] Loading historical orders...")
    historical_tokens = get_historical_orders_from_replay(
        args.data_dir,
        n_samples=args.n_historical,
        token_mode=args.token_mode
    )
    print(f"[*] Historical tokens shape: {historical_tokens.shape}")

    # Decode historical orders
    print("[*] Decoding historical orders...")
    historical_decoded = np.array(decode_msgs(historical_tokens, v.ENCODING, token_mode=args.token_mode))
    print(f"[*] Historical decoded shape: {historical_decoded.shape}")

    # Compute historical statistics
    hist_stats = compute_order_stats(historical_decoded)
    print(f"[*] Historical orders analyzed: {hist_stats['n_orders']}")

    # ========================================
    # Step 2: Collect policy orders from ES training
    # ========================================
    if args.use_synthetic:
        print("\n[Step 2] Generating synthetic policy orders (--use_synthetic flag)...")
        # Synthetic mode for quick testing
        np.random.seed(42)
        n_policy = min(args.n_historical, args.n_episodes * args.n_steps)
        policy_tokens = historical_tokens[:n_policy].copy()

        for i in range(len(policy_tokens)):
            if np.random.random() < 0.8:
                policy_tokens[i, 1] = 1  # event_type = new
            if np.random.random() < 0.3:
                policy_tokens[i, 5] = max(1, policy_tokens[i, 5] // 2)
    else:
        print("\n[Step 2] Running ES training to collect real policy orders...")

        # Create ES config
        config = ESConfig(
            lobs5_checkpoint=args.checkpoint,
            replay_data_path=args.data_dir,
            token_mode=args.token_mode,
            n_steps=args.n_steps,
            n_perturbations=4,  # Minimal for order collection
        )

        # Collect real policy orders
        policy_tokens = collect_policy_orders_from_es(config, n_episodes=args.n_episodes)

    print(f"[*] Policy tokens shape: {policy_tokens.shape}")

    # Decode policy orders
    print("[*] Decoding policy orders...")
    policy_decoded = np.array(decode_msgs(policy_tokens, v.ENCODING, token_mode=args.token_mode))
    print(f"[*] Policy decoded shape: {policy_decoded.shape}")

    # Compute policy statistics
    policy_stats = compute_order_stats(policy_decoded)
    print(f"[*] Policy orders analyzed: {policy_stats['n_orders']}")

    # ========================================
    # Step 3: Generate comparison report
    # ========================================
    print("\n[Step 3] Generating comparison report...")

    report_path = os.path.join(args.output_dir, f'order_analysis_report_{timestamp}.txt')
    mode_str = "Synthetic" if args.use_synthetic else "Real ES"
    report = print_stats_comparison(hist_stats, policy_stats, title=f"Historical vs Policy Orders ({mode_str})")

    # Save report to file
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"[*] Report saved to: {report_path}")

    # ========================================
    # Step 4: Generate visualization
    # ========================================
    print("\n[Step 4] Generating visualizations...")

    # Get raw data for visualization
    hist_raw = get_raw_order_data(historical_decoded)
    policy_raw = get_raw_order_data(policy_decoded)

    # Basic comparison chart
    chart_path = os.path.join(args.output_dir, f'order_comparison_{timestamp}.png')
    plot_comparison(
        hist_raw, policy_raw,
        hist_stats, policy_stats,
        save_path=chart_path,
        title=f'Historical vs Policy Orders ({mode_str})'
    )

    # Detailed analysis chart
    detailed_path = os.path.join(args.output_dir, f'order_detailed_{timestamp}.png')
    plot_detailed_analysis(
        hist_raw, policy_raw,
        hist_stats, policy_stats,
        save_path=detailed_path
    )

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print(" Analysis Complete!")
    print("=" * 60)
    print(f"Mode: {mode_str}")
    print(f"Report: {report_path}")
    print(f"Comparison chart: {chart_path}")
    print(f"Detailed chart: {detailed_path}")
    print("=" * 60)

    # Print key metrics summary
    print("\n[Key Metrics Summary]")
    print(f"  Historical orders: {hist_stats['n_orders']}")
    print(f"  Policy orders: {policy_stats['n_orders']}")
    print(f"  Historical validity: {hist_stats['validity']['valid_ratio']:.2%}")
    print(f"  Policy validity: {policy_stats['validity']['valid_ratio']:.2%}")

    if 'price' in hist_stats and 'price' in policy_stats:
        print(f"  Historical aggressive ratio: {hist_stats['price'].get('aggressive_ratio', 0):.2%}")
        print(f"  Policy aggressive ratio: {policy_stats['price'].get('aggressive_ratio', 0):.2%}")

    # Print event type comparison
    print("\n[Event Type Comparison]")
    for et in ['new', 'cancel', 'delete', 'execute']:
        h_val = hist_stats.get('event_type', {}).get(et, 0) * 100
        p_val = policy_stats.get('event_type', {}).get(et, 0) * 100
        print(f"  {et:10s}: Historical {h_val:5.1f}% | Policy {p_val:5.1f}%")


if __name__ == '__main__':
    main()
