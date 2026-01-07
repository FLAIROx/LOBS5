#!/usr/bin/env python3
"""
Plot LOB benchmark results from scores_*.pkl files
Handles multiple models, stocks, and score types (cond, uncond, div)

Usage:
    python plot_lobbench_from_scores.py /path/to/scores/directory

Example:
    python plot_lobbench_from_scores.py /homes/groups/finance/data/lobbench_scores/scores
"""

import sys
import pickle
import gzip
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt


def parse_score_filename(filename: str) -> Dict[str, str]:
    """
    Parse score filename to extract metadata
    Format: scores_{type}_{STOCK}_{MODEL}_{N}_{TIMESTAMP}.pkl
    Examples:
        scores_uncond_GOOG_s5v1_None_20260104_184010.pkl
        scores_cond_INTC_s5v2N100_20260104_200446.pkl
        scores_div_GOOG_s5v2N5_100_20260105_100928.pkl
    """
    # Remove .pkl extension
    name = filename.replace('.pkl', '')
    parts = name.split('_')

    if len(parts) < 4:
        return None

    score_type = parts[1]  # cond, uncond, div
    stock = parts[2]       # GOOG, INTC

    # Model name might contain underscores, need to find it
    # Looking for known patterns: s5v1, s5v2N100, s5v2N5, s5v2uncond, s5_main, s5
    model_patterns = ['s5v2uncond', 's5v2N100', 's5v2N5', 's5v1', 's5_main', 's5']

    model = None
    remaining_start = 3
    for pattern in model_patterns:
        # Check if pattern appears in the remaining parts
        remaining = '_'.join(parts[3:])
        if remaining.startswith(pattern):
            model = pattern
            # Calculate where model name ends
            model_parts = pattern.split('_')
            remaining_start = 3 + len(model_parts)
            break

    if model is None:
        model = parts[3]
        remaining_start = 4

    return {
        'type': score_type,
        'stock': stock,
        'model': model,
        'filename': filename
    }


def load_scores(pkl_path: Path) -> Dict[str, Any]:
    """Load scores from pkl file (supports both gzip and regular pickle)"""
    try:
        # Try gzip first
        with gzip.open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except (gzip.BadGzipFile, OSError):
        # Fall back to regular pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

    # Handle different data formats
    if isinstance(data, tuple) and len(data) >= 1:
        return data[0]  # First element contains scores dict
    elif isinstance(data, dict):
        return data
    else:
        print(f"Warning: Unexpected data format in {pkl_path}")
        return {}


def extract_metric_values(scores: Dict) -> Dict[str, Dict[str, float]]:
    """
    Extract L1 and Wasserstein values from scores
    Returns: {metric_name: {'l1': value, 'wasserstein': value}}
    """
    result = {}

    for metric_name, metric_data in scores.items():
        if not isinstance(metric_data, dict):
            continue

        l1_data = metric_data.get('l1', None)
        wass_data = metric_data.get('wasserstein', None)

        # Handle tuple format (mean, ci, bootstrap_samples)
        l1_val = None
        wass_val = None

        if l1_data is not None:
            if isinstance(l1_data, tuple):
                l1_val = float(l1_data[0])
            elif isinstance(l1_data, (int, float)):
                l1_val = float(l1_data)

        if wass_data is not None:
            if isinstance(wass_data, tuple):
                wass_val = float(wass_data[0])
            elif isinstance(wass_data, (int, float)):
                wass_val = float(wass_data)

        if l1_val is not None or wass_val is not None:
            result[metric_name] = {
                'l1': l1_val if l1_val is not None else 0,
                'wasserstein': wass_val if wass_val is not None else 0
            }

    return result


def compute_summary_stats(values: List[float]) -> Dict[str, float]:
    """Compute IQM, median, mean from a list of values"""
    if not values:
        return {'iqm': 0, 'median': 0, 'mean': 0}

    arr = np.array([v for v in values if v > 0])  # Filter out zeros
    if len(arr) == 0:
        return {'iqm': 0, 'median': 0, 'mean': 0}

    # IQM: mean of values between 25th and 75th percentile
    q25, q75 = np.percentile(arr, [25, 75])
    iqm_mask = (arr >= q25) & (arr <= q75)
    iqm = np.mean(arr[iqm_mask]) if iqm_mask.any() else np.mean(arr)

    return {
        'iqm': float(iqm),
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr))
    }


def plot_individual_model(model_name: str, stock: str, metric_values: Dict[str, Dict[str, float]],
                          output_dir: Path, score_type: str = 'uncond'):
    """Create bar plots for a single model showing L1 and Wasserstein distances by metric"""
    if not metric_values:
        print(f"  No data for {model_name}/{stock}")
        return

    metrics = list(metric_values.keys())
    l1_values = [metric_values[m]['l1'] for m in metrics]
    wass_values = [metric_values[m]['wasserstein'] for m in metrics]

    # Sort by L1
    sorted_indices = np.argsort(l1_values)
    metrics_sorted = [metrics[i] for i in sorted_indices]
    l1_sorted = [l1_values[i] for i in sorted_indices]
    wass_sorted = [wass_values[i] for i in sorted_indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # L1 Distance
    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(metrics_sorted)))
    ax1.barh(metrics_sorted, l1_sorted, color=colors)
    ax1.set_xlabel('L1 Distance (lower is better)', fontsize=12)
    ax1.set_title(f'L1 Distance - {model_name} / {stock}\n({score_type})', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)

    # Wasserstein Distance
    ax2 = axes[1]
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(metrics_sorted)))
    # Sort by Wasserstein for this panel
    wass_sorted_indices = np.argsort(wass_values)
    metrics_wass = [metrics[i] for i in wass_sorted_indices]
    wass_final = [wass_values[i] for i in wass_sorted_indices]
    ax2.barh(metrics_wass, wass_final, color=colors)
    ax2.set_xlabel('Wasserstein Distance (lower is better)', fontsize=12)
    ax2.set_title(f'Wasserstein Distance - {model_name} / {stock}\n({score_type})', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f'{score_type}_{stock}_{model_name}_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_summary_comparison(all_data: Dict, output_dir: Path, score_type: str = 'uncond'):
    """
    Create paper-style summary plot with IQM/median/mean X markers
    all_data: {stock: {model: {'l1': [values], 'wasserstein': [values]}}}
    """
    # Collect unique stocks and models
    stocks = sorted(all_data.keys())
    models = set()
    for stock_data in all_data.values():
        models.update(stock_data.keys())
    models = sorted(models)

    if not models or not stocks:
        print("  No data for summary plot")
        return

    # Create figure with 2 panels (L1 and Wasserstein)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.subplots_adjust(hspace=0.35)

    stat_types = ['IQM', 'median', 'mean']
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # Y positions
    y_positions = {}
    y_current = 0
    for stock in stocks:
        y_positions[stock] = {}
        for i, stat in enumerate(stat_types):
            y_positions[stock][stat] = y_current + (len(stat_types) - 1 - i) * 0.25
        y_current += 1.2

    for panel_idx, (ax, dist_type) in enumerate(zip(axes, ['l1', 'wasserstein'])):
        ax.set_title(dist_type.upper() if dist_type == 'l1' else 'Wasserstein',
                    fontsize=14, fontweight='bold', loc='left')

        for model_idx, model in enumerate(models):
            for stock in stocks:
                if stock not in all_data or model not in all_data[stock]:
                    continue

                values = all_data[stock][model].get(dist_type, [])
                if not values:
                    continue

                stats = compute_summary_stats(values)

                for stat_name in stat_types:
                    y = y_positions[stock][stat_name]
                    x = stats[stat_name.lower()]
                    ax.scatter(x, y, marker='x', s=100, c=[colors[model_idx]],
                              linewidths=2, zorder=10, label=model if stat_name == 'IQM' else '')

        # Y-axis labels
        y_labels = []
        y_ticks = []
        for stock in reversed(stocks):
            for stat in stat_types:
                y_labels.append(stat)
                y_ticks.append(y_positions[stock][stat])

        # Stock labels
        for stock in stocks:
            y_mid = np.mean([y_positions[stock][s] for s in stat_types])
            ax.text(-0.02, y_mid, stock, transform=ax.get_yaxis_transform(),
                   fontsize=12, fontweight='bold', ha='right', va='center')

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels * len(stocks), fontsize=10)
        ax.set_xlim(left=0)
        ax.grid(axis='x', alpha=0.3, linestyle='-')

        # Separator between stocks
        if len(stocks) > 1:
            for i in range(len(stocks) - 1):
                sep_y = (y_positions[stocks[i]]['mean'] + y_positions[stocks[i+1]]['IQM']) / 2
                ax.axhline(y=sep_y, color='black', linewidth=0.8)

    # Legend
    handles, labels = [], []
    seen = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen and label:
                handles.append(handle)
                labels.append(label)
                seen.add(label)

    fig.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)),
              fontsize=11, frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    output_path = output_dir / f'summary_stats_comp_{score_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_lobbench_from_scores.py /path/to/scores/directory")
        print("\nThis script will:")
        print("  1. Find all scores_*.pkl files in the directory")
        print("  2. Group them by type (cond/uncond/div), stock, and model")
        print("  3. Create individual metric plots for each model")
        print("  4. Create summary comparison plots")
        sys.exit(1)

    scores_dir = Path(sys.argv[1])
    if not scores_dir.exists():
        print(f"Error: Directory not found: {scores_dir}")
        sys.exit(1)

    output_dir = scores_dir / 'plots'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("LOB Benchmark Plotting from Scores Files")
    print("=" * 60)
    print(f"Scores directory: {scores_dir}")
    print(f"Output directory: {output_dir}")

    # Find all score files (recursively search subdirectories)
    pkl_files = list(scores_dir.glob('*/scores/scores_*.pkl'))
    print(f"\nFound {len(pkl_files)} score files")

    if not pkl_files:
        print("No scores_*.pkl files found!")
        sys.exit(1)

    # Group by type, stock, model
    # Structure: {score_type: {stock: {model: {filename: data}}}}
    grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for pkl_file in pkl_files:
        info = parse_score_filename(pkl_file.name)
        if info is None:
            print(f"  Skipping unrecognized file: {pkl_file.name}")
            continue

        print(f"  Loading: {pkl_file.name}")
        scores = load_scores(pkl_file)
        metric_values = extract_metric_values(scores)

        if metric_values:
            grouped_data[info['type']][info['stock']][info['model']][pkl_file.name] = metric_values

    # Print summary
    print("\n" + "-" * 40)
    print("Data Summary:")
    for score_type, stocks in grouped_data.items():
        print(f"\n  [{score_type}]")
        for stock, models in stocks.items():
            model_list = ', '.join(models.keys())
            print(f"    {stock}: {model_list}")

    # Generate plots for each score type
    for score_type, stocks in grouped_data.items():
        print(f"\n{'='*40}")
        print(f"Generating plots for: {score_type}")
        print("=" * 40)

        # Aggregate data for summary plot
        # {stock: {model: {'l1': [values], 'wasserstein': [values]}}}
        summary_data = defaultdict(lambda: defaultdict(lambda: {'l1': [], 'wasserstein': []}))

        for stock, models in stocks.items():
            for model, files in models.items():
                # Use the latest file (by timestamp in filename)
                latest_file = sorted(files.keys())[-1]
                metric_values = files[latest_file]

                # Individual model plot
                plot_individual_model(model, stock, metric_values, output_dir, score_type)

                # Collect for summary
                for metric, values in metric_values.items():
                    if values['l1'] > 0:
                        summary_data[stock][model]['l1'].append(values['l1'])
                    if values['wasserstein'] > 0:
                        summary_data[stock][model]['wasserstein'].append(values['wasserstein'])

        # Summary comparison plot
        print(f"\nGenerating summary comparison plot...")
        plot_summary_comparison(dict(summary_data), output_dir, score_type)

    print(f"\n{'='*60}")
    print(f"📊 All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
