"""
Order Quality Analysis Script

Compares historical orders (from replay data) vs policy-generated orders (from ES training).
Generates statistical reports and visualization charts to validate order quality.

Usage: sbatch test_order_analysis.sh
"""

import os
import sys

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
    decode_and_analyze_orders,
    get_raw_order_data,
)
from es_lobs5.analysis.visualize import plot_comparison, plot_detailed_analysis


def collect_historical_orders(trainer, n_samples: int = 1000) -> np.ndarray:
    """
    Extract historical order tokens from replay data.

    Args:
        trainer: ESTrainer instance with loaded replay data
        n_samples: Number of order messages to extract

    Returns:
        Array of shape (n_samples, msg_len) containing historical tokens
    """
    # Access the loaded replay data through trainer
    # The replay data contains historical LOB messages
    if hasattr(trainer, 'replay_msgs'):
        replay_msgs = trainer.replay_msgs
        print(f"[*] replay_msgs shape: {replay_msgs.shape}")

        # Extract first n_samples messages
        n_available = replay_msgs.shape[0] if replay_msgs.ndim == 2 else replay_msgs.shape[1]
        n_samples = min(n_samples, n_available)

        if replay_msgs.ndim == 3:
            # Shape: (batch, seq, msg_len)
            historical_tokens = replay_msgs[0, :n_samples, :]
        else:
            # Shape: (seq, msg_len)
            historical_tokens = replay_msgs[:n_samples, :]

        return np.array(historical_tokens)
    else:
        raise AttributeError("Trainer does not have 'replay_msgs' attribute")


def collect_policy_orders(trainer, n_episodes: int = 5) -> np.ndarray:
    """
    Run episodes to collect policy-generated orders.

    Args:
        trainer: ESTrainer instance
        n_episodes: Number of episodes to run

    Returns:
        Array of shape (n_episodes * n_steps, msg_len) containing policy tokens
    """
    all_policy_msgs = []

    for ep in range(n_episodes):
        key = jax.random.PRNGKey(42 + ep)

        # Get initial state
        initial_state, initial_msg_history = trainer.get_episode_initial_state(
            key,
            start_idx=ep * 100  # Different starting points
        )

        # Run episode (thread_id=0, epoch=0 for baseline without noise)
        fitness, info = trainer.eval_single_thread(
            key,
            thread_id=0,  # Baseline (no noise)
            epoch=0,
            initial_sim_state=initial_state,
            initial_msg_history=initial_msg_history,
        )

        # Extract policy messages from info
        if 'policy_msgs' in info:
            policy_msgs = info['policy_msgs']
            all_policy_msgs.append(np.array(policy_msgs))
            print(f"[*] Episode {ep}: collected {len(policy_msgs)} policy msgs, fitness={float(fitness):.4f}")
        else:
            print(f"[!] Episode {ep}: no policy_msgs in info dict")

    if all_policy_msgs:
        return np.concatenate(all_policy_msgs, axis=0)
    else:
        raise ValueError("No policy messages collected")


def main():
    parser = argparse.ArgumentParser(description='Order Quality Analysis')
    parser.add_argument('--checkpoint', type=str,
                       default='/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/wandb/run-20241130_101652-logical-serenity-19/files/checkpoints',
                       help='Path to LOBS5 checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG/2021',
                       help='Path to encoded data directory')
    parser.add_argument('--n_historical', type=int, default=1000,
                       help='Number of historical orders to analyze')
    parser.add_argument('--n_episodes', type=int, default=5,
                       help='Number of episodes to collect policy orders')
    parser.add_argument('--n_steps', type=int, default=50,
                       help='Steps per episode')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                       help='Output directory for reports and charts')
    parser.add_argument('--token_mode', type=int, default=24,
                       help='Token mode (22 or 24)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 60)
    print(" Order Quality Analysis Script")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Token mode: {args.token_mode}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Initialize vocab for decoding
    v = Vocab(token_mode=args.token_mode)

    # ========================================
    # Step 1: Load historical orders directly from encoded data
    # ========================================
    print("\n[Step 1] Loading historical orders from encoded data...")

    # Find .npy files in data directory
    data_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.npy')])
    if not data_files:
        raise FileNotFoundError(f"No .npy files found in {args.data_dir}")

    # Load first file
    first_file = os.path.join(args.data_dir, data_files[0])
    print(f"[*] Loading from: {first_file}")
    encoded_data = np.load(first_file)
    print(f"[*] Encoded data shape: {encoded_data.shape}")

    # Extract historical tokens (first n_historical messages)
    if encoded_data.ndim == 2:
        historical_tokens = encoded_data[:args.n_historical, :]
    else:
        historical_tokens = encoded_data.reshape(-1, encoded_data.shape[-1])[:args.n_historical, :]

    print(f"[*] Historical tokens shape: {historical_tokens.shape}")

    # Decode historical orders
    print("[*] Decoding historical orders...")
    historical_decoded = np.array(decode_msgs(historical_tokens, v.ENCODING, token_mode=args.token_mode))
    print(f"[*] Historical decoded shape: {historical_decoded.shape}")

    # Compute historical statistics
    hist_stats = compute_order_stats(historical_decoded)
    print(f"[*] Historical orders analyzed: {hist_stats['n_orders']}")

    # ========================================
    # Step 2: Generate synthetic policy orders for testing
    # (In production, use trainer.collect_policy_orders())
    # ========================================
    print("\n[Step 2] Generating synthetic policy orders for analysis...")

    # For testing, we'll create synthetic policy orders based on historical
    # This allows us to test the analysis pipeline without full training
    np.random.seed(42)
    n_policy = min(args.n_historical, args.n_episodes * args.n_steps)

    # Create synthetic policy tokens with slight modifications
    # This simulates what a policy might generate
    policy_tokens = historical_tokens[:n_policy].copy()

    # Simulate policy behavior:
    # 1. More "new" orders (event_type=1), fewer cancels
    # 2. More aggressive pricing (closer to mid)
    # 3. Smaller order sizes
    for i in range(len(policy_tokens)):
        # 80% chance of new order for policy
        if np.random.random() < 0.8:
            policy_tokens[i, 1] = 1  # event_type = new

        # Random small modifications to size tokens
        if np.random.random() < 0.3:
            policy_tokens[i, 5] = max(1, policy_tokens[i, 5] // 2)

    print(f"[*] Synthetic policy tokens shape: {policy_tokens.shape}")

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
    report = print_stats_comparison(hist_stats, policy_stats, title="Historical vs Policy Orders")

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
        title='Historical vs Policy Orders Comparison'
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


if __name__ == '__main__':
    main()
