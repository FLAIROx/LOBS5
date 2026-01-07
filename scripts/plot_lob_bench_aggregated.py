#!/usr/bin/env python3
"""
Create aggregated LOB benchmark plot in paper style:
- Shows IQM (interquartile mean), median, mean for each model-stock combination
- X markers with different colors for each model
- Two panels: L1 and Wasserstein distances
"""

import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str           # Display name for legend
    path: Path          # Path to results directory
    color: str          # Plot color
    marker: str = 'x'   # Marker style


# Define models to compare (add more as they become available)
MODELS = [
    ModelConfig(
        name='s5',
        path=Path('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output/brisk-violet-111/results'),
        color='#1f77b4',  # blue
    ),
    # Add more models here as they become available:
    # ModelConfig(
    #     name='s5v2N5',
    #     path=Path('/path/to/s5v2N5/results'),
    #     color='#2ca02c',  # green
    # ),
    # ModelConfig(
    #     name='s5v2uncond',
    #     path=Path('/path/to/s5v2uncond/results'),
    #     color='#d62728',  # red
    # ),
    # ModelConfig(
    #     name='s5v2N100',
    #     path=Path('/path/to/s5v2N100/results'),
    #     color='#ff7f0e',  # orange
    # ),
]

STOCKS = ['GOOG']  # Add 'INTC' when available

ALL_METRICS = [
    'vol_per_min', 'ofi_down', 'ofi_up', 'ask_cancellation_levels',
    'limit_ask_order_levels', 'limit_bid_order_levels', 'ask_cancellation_depth',
    'limit_ask_order_depth', 'bid_cancellation_levels', 'bid_cancellation_depth',
    'limit_bid_order_depth', 'spread', 'log_time_to_cancel', 'ofi_stay',
    'ofi', 'ask_volume', 'bid_volume', 'orderbook_imbalance',
    'ask_volume_touch', 'bid_volume_touch', 'log_inter_arrival_time'
]


# ============================================================================
# Data Loading
# ============================================================================

def load_metric_scores(results_dir: Path, metric_name: str) -> Optional[Dict]:
    """Load scores for a given metric"""
    metric_dir = results_dir / metric_name / 'scores'

    if not metric_dir.exists():
        return None

    pkl_files = list(metric_dir.glob('*.pkl'))
    if not pkl_files:
        return None

    pkl_file = sorted(pkl_files)[-1]  # Use most recent

    try:
        with gzip.open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        return data[0].get(metric_name, {})
    except Exception as e:
        print(f"Warning: Could not load {metric_name}: {e}")
        return None


def collect_model_scores(model: ModelConfig) -> Dict[str, List[float]]:
    """Collect L1 and Wasserstein scores for all metrics of a model"""
    l1_scores = []
    wass_scores = []

    for metric in ALL_METRICS:
        scores = load_metric_scores(model.path, metric)
        if scores is None:
            continue

        l1_data = scores.get('l1', (0,))
        wass_data = scores.get('wasserstein', (0,))

        l1_mean = float(l1_data[0]) if isinstance(l1_data, tuple) else float(l1_data)
        wass_mean = float(wass_data[0]) if isinstance(wass_data, tuple) else float(wass_data)

        # Skip zero values (likely missing data)
        if l1_mean > 0:
            l1_scores.append(l1_mean)
        if wass_mean > 0:
            wass_scores.append(wass_mean)

    return {'l1': l1_scores, 'wasserstein': wass_scores}


def compute_aggregated_stats(values: List[float]) -> Dict[str, float]:
    """Compute IQM, median, mean from a list of values"""
    if not values:
        return {'iqm': 0, 'median': 0, 'mean': 0}

    arr = np.array(values)

    # IQM: mean of values between 25th and 75th percentile
    q25, q75 = np.percentile(arr, [25, 75])
    iqm_mask = (arr >= q25) & (arr <= q75)
    iqm = np.mean(arr[iqm_mask]) if iqm_mask.any() else np.mean(arr)

    return {
        'iqm': float(iqm),
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr))
    }


# ============================================================================
# Plotting
# ============================================================================

def create_aggregated_plot(output_dir: Path):
    """Create the paper-style aggregated plot"""

    # Collect data for all models and stocks
    all_data = {}  # {stock: {model_name: {'l1': stats, 'wass': stats}}}

    for stock in STOCKS:
        all_data[stock] = {}
        for model in MODELS:
            scores = collect_model_scores(model)
            all_data[stock][model.name] = {
                'l1': compute_aggregated_stats(scores['l1']),
                'wasserstein': compute_aggregated_stats(scores['wasserstein'])
            }
            print(f"Loaded {model.name}/{stock}: {len(scores['l1'])} metrics")

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.subplots_adjust(hspace=0.3)

    stat_types = ['IQM', 'median', 'mean']
    stat_keys = ['iqm', 'median', 'mean']

    # Y positions for each stock and stat combination
    y_positions = {}
    y_current = 0
    for stock in STOCKS:
        y_positions[stock] = {}
        for i, stat in enumerate(stat_types):
            y_positions[stock][stat] = y_current + (len(stat_types) - 1 - i) * 0.25
        y_current += 1.2  # Gap between stocks

    # ========== L1 Panel ==========
    ax1 = axes[0]
    ax1.set_title('L1', fontsize=14, fontweight='bold', loc='left')

    for stock in STOCKS:
        for model in MODELS:
            data = all_data[stock][model.name]['l1']
            for stat_name, stat_key in zip(stat_types, stat_keys):
                y = y_positions[stock][stat_name]
                x = data[stat_key]
                ax1.scatter(x, y, marker='x', s=100, c=model.color,
                           linewidths=2, zorder=10)

    # Y-axis labels
    y_labels = []
    y_ticks = []
    for stock in reversed(STOCKS):
        for stat in stat_types:
            y_labels.append(stat)
            y_ticks.append(y_positions[stock][stat])

    # Add stock labels on the left
    for stock in STOCKS:
        y_mid = np.mean([y_positions[stock][s] for s in stat_types])
        ax1.text(-0.02, y_mid, stock, transform=ax1.get_yaxis_transform(),
                fontsize=12, fontweight='bold', ha='right', va='center')

    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels * len(STOCKS), fontsize=10)
    ax1.set_xlim(left=0)
    ax1.grid(axis='x', alpha=0.3, linestyle='-')
    ax1.axhline(y=0.6, color='black', linewidth=0.8)  # Separator between stocks

    # ========== Wasserstein Panel ==========
    ax2 = axes[1]
    ax2.set_title('Wasserstein', fontsize=14, fontweight='bold', loc='left')

    for stock in STOCKS:
        for model in MODELS:
            data = all_data[stock][model.name]['wasserstein']
            for stat_name, stat_key in zip(stat_types, stat_keys):
                y = y_positions[stock][stat_name]
                x = data[stat_key]
                ax2.scatter(x, y, marker='x', s=100, c=model.color,
                           linewidths=2, zorder=10)

    # Add stock labels on the left
    for stock in STOCKS:
        y_mid = np.mean([y_positions[stock][s] for s in stat_types])
        ax2.text(-0.02, y_mid, stock, transform=ax2.get_yaxis_transform(),
                fontsize=12, fontweight='bold', ha='right', va='center')

    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_labels * len(STOCKS), fontsize=10)
    ax2.set_xlim(left=0)
    ax2.grid(axis='x', alpha=0.3, linestyle='-')
    ax2.axhline(y=0.6, color='black', linewidth=0.8)

    # ========== Legend ==========
    legend_handles = [plt.Line2D([0], [0], marker='x', color='w',
                                  markerfacecolor=m.color, markeredgecolor=m.color,
                                  markersize=10, markeredgewidth=2, label=m.name)
                      for m in MODELS]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(MODELS),
               fontsize=11, frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Save
    output_path = output_dir / 'lob_bench_aggregated.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    return output_path


def create_detailed_summary_plot(output_dir: Path):
    """Create a more detailed summary showing all statistics"""

    # Collect data
    model = MODELS[0]  # Use first model for detailed view
    scores = collect_model_scores(model)

    l1_stats = compute_aggregated_stats(scores['l1'])
    wass_stats = compute_aggregated_stats(scores['wasserstein'])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ========== L1 Distribution ==========
    ax1 = axes[0]
    l1_arr = np.array(scores['l1'])

    # Histogram
    ax1.hist(l1_arr, bins=15, alpha=0.7, color='steelblue', edgecolor='navy')

    # Add vertical lines for statistics
    ax1.axvline(l1_stats['mean'], color='red', linestyle='-', linewidth=2, label=f"Mean: {l1_stats['mean']:.4f}")
    ax1.axvline(l1_stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {l1_stats['median']:.4f}")
    ax1.axvline(l1_stats['iqm'], color='orange', linestyle=':', linewidth=2, label=f"IQM: {l1_stats['iqm']:.4f}")

    ax1.set_xlabel('L1 Distance', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'L1 Distance Distribution\nModel: {model.name} | Stock: GOOG', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)

    # ========== Wasserstein Distribution ==========
    ax2 = axes[1]
    wass_arr = np.array(scores['wasserstein'])

    ax2.hist(wass_arr, bins=15, alpha=0.7, color='coral', edgecolor='darkred')

    ax2.axvline(wass_stats['mean'], color='red', linestyle='-', linewidth=2, label=f"Mean: {wass_stats['mean']:.4f}")
    ax2.axvline(wass_stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {wass_stats['median']:.4f}")
    ax2.axvline(wass_stats['iqm'], color='orange', linestyle=':', linewidth=2, label=f"IQM: {wass_stats['iqm']:.4f}")

    ax2.set_xlabel('Wasserstein Distance', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Wasserstein Distance Distribution\nModel: {model.name} | Stock: GOOG', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'lob_bench_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return output_path


def print_summary():
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("LOB Benchmark Aggregated Statistics")
    print("=" * 70)

    for stock in STOCKS:
        print(f"\n📈 Stock: {stock}")
        print("-" * 50)
        print(f"{'Model':<15} {'Metric':<12} {'IQM':<10} {'Median':<10} {'Mean':<10}")
        print("-" * 50)

        for model in MODELS:
            scores = collect_model_scores(model)
            l1_stats = compute_aggregated_stats(scores['l1'])
            wass_stats = compute_aggregated_stats(scores['wasserstein'])

            print(f"{model.name:<15} {'L1':<12} {l1_stats['iqm']:<10.4f} {l1_stats['median']:<10.4f} {l1_stats['mean']:<10.4f}")
            print(f"{'':<15} {'Wasserstein':<12} {wass_stats['iqm']:<10.4f} {wass_stats['median']:<10.4f} {wass_stats['mean']:<10.4f}")

    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = Path('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/benchmark_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Creating aggregated LOB benchmark plots...")

    # Print summary
    print_summary()

    # Create plots
    print("\n🎨 Creating visualizations...")
    create_aggregated_plot(output_dir)
    create_detailed_summary_plot(output_dir)

    print(f"\n📊 All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
