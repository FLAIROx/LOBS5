"""
Order Quality Analysis Script

Uses preproc data (already decoded format) for historical orders.
Policy orders from ES training are decoded from tokens.

Usage: sbatch test_order_analysis.sh
"""

import os
import sys
from dataclasses import dataclass

LOBS5_ROOT = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5'
sys.path.insert(0, LOBS5_ROOT)
sys.path.insert(0, os.path.join(LOBS5_ROOT, 'HyperscaleES', 'src'))

import jax
import numpy as np
from datetime import datetime
import argparse

from lob.encoding import decode_msgs, Vocab
from es_lobs5.analysis import (
    compute_order_stats,
    print_stats_comparison,
    get_raw_order_data,
)
from es_lobs5.analysis.visualize import plot_comparison, plot_detailed_analysis


@dataclass
class ESConfig:
    """Config for ES Trainer order analysis.

    Uses PREPROC data format (N, 14) by default.
    """
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    group_size: int = 0
    n_perturbations: int = 8
    n_epochs: int = 1
    n_steps: int = 50
    background_msgs_per_step: int = 10
    token_mode: int = 24
    background_mode: str = 'historical_replay'
    # Data paths - using PREPROC format (N, 14)
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc'
    data_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc'
    file_idx: int = 0  # Use first file in directory
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    grad_clip: float = 1.0
    seed: int = 42
    n_warmup_msgs: int = 500


def collect_policy_orders(config: ESConfig, n_episodes: int = 5) -> dict:
    """
    Run ES episodes and collect policy-generated orders.
    Returns dict with 'baseline' and 'noised' token arrays.
    """
    from es_lobs5.training.es_trainer import ESTrainer

    print(f"[*] Initializing ESTrainer...")
    trainer = ESTrainer(config)

    print(f"[*] Creating initial state...")
    initial_state, initial_msg_history = trainer._create_initial_sim_state()

    baseline_msgs, noised_msgs = [], []
    baseline_fit, noised_fit = [], []

    for ep in range(n_episodes):
        key = jax.random.PRNGKey(config.seed + ep * 100)

        # Baseline (thread_id=0, no noise)
        fit, info = trainer.eval_single_thread(key, 0, 0, initial_state, initial_msg_history)
        if 'policy_msgs' in info:
            baseline_msgs.append(np.array(info['policy_msgs']))
            baseline_fit.append(float(fit))
            print(f"  Ep{ep} baseline: {len(info['policy_msgs'])} msgs, fit={float(fit):.4f}")

        # Noised (thread_id=2,3,4,5)
        for tid in [2, 3, 4, 5]:
            k = jax.random.fold_in(key, tid)
            fit, info = trainer.eval_single_thread(k, tid, 0, initial_state, initial_msg_history)
            if 'policy_msgs' in info:
                noised_msgs.append(np.array(info['policy_msgs']))
                noised_fit.append(float(fit))

    return {
        'baseline': np.concatenate(baseline_msgs) if baseline_msgs else np.array([]),
        'noised': np.concatenate(noised_msgs) if noised_msgs else np.array([]),
        'baseline_fitness': baseline_fit,
        'noised_fitness': noised_fit,
    }


def load_preproc_orders(data_path: str, n_samples: int = 1000) -> np.ndarray:
    """
    Load historical orders from preproc data.
    Preproc format: (N, 14) - already decoded, no tokenization.

    Columns: [OID, event_type, direction, price_abs, price, size,
              delta_t_s, delta_t_ns, time_s, time_ns,
              p_ref, size_ref, time_s_ref, time_ns_ref]
    """
    import glob

    # Find message files (not orderbook files)
    files = sorted(glob.glob(os.path.join(data_path, '*_message_*_proc.npy')))
    if not files:
        # Fallback to any .npy file
        files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
    if not files:
        raise FileNotFoundError(f"No .npy files in {data_path}")

    print(f"[*] Loading preproc data from: {files[0]}")
    data = np.load(files[0])
    print(f"[*] Preproc shape: {data.shape}, dtype: {data.dtype}")

    # Preproc is already (N, 14) format
    if data.shape[1] == 14:
        return data[:n_samples]
    else:
        raise ValueError(f"Expected preproc with 14 columns, got {data.shape[1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/')
    parser.add_argument('--data_dir', type=str,
                       default='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc')
    parser.add_argument('--n_historical', type=int, default=2000)
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./analysis_output')
    parser.add_argument('--token_mode', type=int, default=24)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 60)
    print(" Order Quality Analysis (Preproc Input)")
    print("=" * 60)
    print(f"Data: {args.data_dir}")
    print("=" * 60)

    # Step 1: Load historical orders from PREPROC (already decoded)
    print("\n[1] Loading historical orders from preproc...")
    hist_decoded = load_preproc_orders(args.data_dir, args.n_historical)
    hist_stats = compute_order_stats(hist_decoded)
    print(f"    {hist_stats['n_orders']} orders loaded")

    # Step 2: Collect policy orders from ES training (tokenized -> decode)
    print("\n[2] Collecting policy orders from ES training...")
    config = ESConfig(
        lobs5_checkpoint=args.checkpoint,
        replay_data_path=args.data_dir,
        data_dir=args.data_dir,
        token_mode=args.token_mode,
        n_steps=args.n_steps,
    )
    result = collect_policy_orders(config, args.n_episodes)

    # Decode policy tokens to (N, 14) format
    v = Vocab(token_mode=args.token_mode)
    baseline_decoded = np.array(decode_msgs(result['baseline'], v.ENCODING, token_mode=args.token_mode))
    noised_decoded = np.array(decode_msgs(result['noised'], v.ENCODING, token_mode=args.token_mode))

    baseline_stats = compute_order_stats(baseline_decoded)
    noised_stats = compute_order_stats(noised_decoded)

    print(f"    Baseline: {baseline_stats['n_orders']} orders")
    print(f"    Noised: {noised_stats['n_orders']} orders")

    # Step 3: Report
    print("\n[3] Generating report...")
    report_path = os.path.join(args.output_dir, f'report_{ts}.txt')
    report = print_stats_comparison(hist_stats, baseline_stats, title="Historical vs Baseline Policy")
    with open(report_path, 'w') as f:
        f.write(report)

    # Step 4: Charts
    print("\n[4] Generating charts...")
    hist_raw = get_raw_order_data(hist_decoded)
    baseline_raw = get_raw_order_data(baseline_decoded)

    chart_path = os.path.join(args.output_dir, f'comparison_{ts}.png')
    plot_comparison(hist_raw, baseline_raw, hist_stats, baseline_stats, save_path=chart_path)

    detailed_path = os.path.join(args.output_dir, f'detailed_{ts}.png')
    plot_detailed_analysis(hist_raw, baseline_raw, hist_stats, baseline_stats, save_path=detailed_path)

    # Summary
    print("\n" + "=" * 60)
    print(f"Report: {report_path}")
    print(f"Charts: {chart_path}")
    print(f"Validity: Hist {hist_stats['validity']['valid_ratio']:.1%} | Baseline {baseline_stats['validity']['valid_ratio']:.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
