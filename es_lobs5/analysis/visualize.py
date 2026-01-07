"""
Visualization Module for Order Analysis

Creates comparison charts between historical and policy-generated orders.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


def plot_comparison(
    hist_data: Dict[str, np.ndarray],
    policy_data: Dict[str, np.ndarray],
    hist_stats: Dict,
    policy_stats: Dict,
    save_path: str = 'order_analysis.png',
    title: str = 'Historical vs Policy Orders Comparison'
):
    """
    Generate 2x2 comparison chart.

    Args:
        hist_data: Raw historical order data from get_raw_order_data()
        policy_data: Raw policy order data
        hist_stats: Statistics dictionary for historical orders
        policy_stats: Statistics dictionary for policy orders
        save_path: Path to save the figure
        title: Figure title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. Event Type Distribution (top-left)
    ax = axes[0, 0]
    plot_event_type_distribution(ax, hist_stats, policy_stats)

    # 2. Price Distribution (top-right)
    ax = axes[0, 1]
    plot_price_distribution(ax, hist_data, policy_data)

    # 3. Size Distribution (bottom-left)
    ax = axes[1, 0]
    plot_size_distribution(ax, hist_data, policy_data)

    # 4. Direction Distribution (bottom-right)
    ax = axes[1, 1]
    plot_direction_distribution(ax, hist_stats, policy_stats)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[*] Saved comparison chart to: {save_path}")
    plt.close()


def plot_event_type_distribution(
    ax: plt.Axes,
    hist_stats: Dict,
    policy_stats: Dict
):
    """Plot event type distribution comparison."""
    event_types = ['new', 'cancel', 'delete', 'execute']
    x = np.arange(len(event_types))
    width = 0.35

    hist_vals = [hist_stats.get('event_type', {}).get(et, 0) for et in event_types]
    policy_vals = [policy_stats.get('event_type', {}).get(et, 0) for et in event_types]

    bars1 = ax.bar(x - width/2, hist_vals, width, label='Historical', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, policy_vals, width, label='Policy', color='coral', alpha=0.8)

    ax.set_xlabel('Event Type')
    ax.set_ylabel('Ratio')
    ax.set_title('Event Type Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(event_types)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)


def plot_price_distribution(
    ax: plt.Axes,
    hist_data: Dict[str, np.ndarray],
    policy_data: Dict[str, np.ndarray],
    bins: int = 50,
    price_range: tuple = None  # None = auto range
):
    """Plot price distribution histograms."""
    hist_prices = hist_data['price']
    policy_prices = policy_data['price']

    # Filter only invalid prices (< -9000 means missing/invalid)
    hist_prices = hist_prices[hist_prices > -9000]
    policy_prices = policy_prices[policy_prices > -9000]

    # Auto-detect range if not specified
    if price_range is None:
        all_prices = np.concatenate([hist_prices, policy_prices]) if len(policy_prices) > 0 else hist_prices
        if len(all_prices) > 0:
            p5, p95 = np.percentile(all_prices, [5, 95])
            # Expand range slightly and round to nice numbers
            range_width = max(abs(p5), abs(p95), 20)
            price_range = (-range_width * 1.2, range_width * 1.2)
        else:
            price_range = (-100, 100)

    # Apply range filter for display
    hist_display = hist_prices[(hist_prices >= price_range[0]) & (hist_prices <= price_range[1])]
    policy_display = policy_prices[(policy_prices >= price_range[0]) & (policy_prices <= price_range[1])]

    ax.hist(hist_display, bins=bins, alpha=0.6, label=f'Historical (n={len(hist_prices)}, shown={len(hist_display)})',
            color='steelblue', density=True)
    if len(policy_display) > 0:
        ax.hist(policy_display, bins=bins, alpha=0.6, label=f'Policy (n={len(policy_prices)}, shown={len(policy_display)})',
                color='coral', density=True)
    else:
        ax.plot([], [], label=f'Policy (n={len(policy_prices)}, shown=0)', color='coral')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Mid Price')
    ax.axvline(x=-1, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(x=1, color='gray', linestyle=':', linewidth=0.8)

    ax.set_xlabel('Price (ticks relative to mid)')
    ax.set_ylabel('Density')
    ax.set_title('Price Distribution')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(price_range)


def plot_size_distribution(
    ax: plt.Axes,
    hist_data: Dict[str, np.ndarray],
    policy_data: Dict[str, np.ndarray],
    bins: int = 30,
    max_size: int = 500
):
    """Plot size distribution histograms."""
    hist_sizes = hist_data['size']
    policy_sizes = policy_data['size']

    # Filter valid sizes
    hist_sizes = hist_sizes[(hist_sizes > 0) & (hist_sizes <= max_size)]
    policy_sizes = policy_sizes[(policy_sizes > 0) & (policy_sizes <= max_size)]

    ax.hist(hist_sizes, bins=bins, alpha=0.6, label=f'Historical (n={len(hist_sizes)})',
            color='steelblue', density=True)
    ax.hist(policy_sizes, bins=bins, alpha=0.6, label=f'Policy (n={len(policy_sizes)})',
            color='coral', density=True)

    ax.set_xlabel('Order Size')
    ax.set_ylabel('Density')
    ax.set_title('Size Distribution')
    ax.legend(loc='upper right')


def plot_direction_distribution(
    ax: plt.Axes,
    hist_stats: Dict,
    policy_stats: Dict
):
    """Plot direction (buy/sell) distribution."""
    directions = ['sell', 'buy']
    x = np.arange(len(directions))
    width = 0.35

    hist_vals = [hist_stats.get('direction', {}).get(d, 0) for d in directions]
    policy_vals = [policy_stats.get('direction', {}).get(d, 0) for d in directions]

    bars1 = ax.bar(x - width/2, hist_vals, width, label='Historical', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, policy_vals, width, label='Policy', color='coral', alpha=0.8)

    ax.set_xlabel('Direction')
    ax.set_ylabel('Ratio')
    ax.set_title('Direction Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(directions)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)


def plot_time_distribution(
    ax: plt.Axes,
    hist_data: Dict[str, np.ndarray],
    policy_data: Dict[str, np.ndarray],
    bins: int = 30,
    max_interval: float = 1.0
):
    """Plot time interval distribution."""
    hist_dt = hist_data['delta_t_s'] + hist_data['delta_t_ns'] / 1e9
    policy_dt = policy_data['delta_t_s'] + policy_data['delta_t_ns'] / 1e9

    # Filter valid intervals
    hist_dt = hist_dt[(hist_dt >= 0) & (hist_dt <= max_interval)]
    policy_dt = policy_dt[(policy_dt >= 0) & (policy_dt <= max_interval)]

    ax.hist(hist_dt, bins=bins, alpha=0.6, label=f'Historical (n={len(hist_dt)})',
            color='steelblue', density=True)
    ax.hist(policy_dt, bins=bins, alpha=0.6, label=f'Policy (n={len(policy_dt)})',
            color='coral', density=True)

    ax.set_xlabel('Time Interval (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('Inter-Order Time Distribution')
    ax.legend(loc='upper right')


def plot_detailed_analysis(
    hist_data: Dict[str, np.ndarray],
    policy_data: Dict[str, np.ndarray],
    hist_stats: Dict,
    policy_stats: Dict,
    save_path: str = 'order_analysis_detailed.png'
):
    """
    Generate detailed 3x2 analysis chart.

    Includes:
    - Event type distribution
    - Direction distribution
    - Price distribution
    - Size distribution
    - Price buckets bar chart
    - Size buckets bar chart
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle('Detailed Order Analysis', fontsize=14, fontweight='bold')

    # 1. Event Type Distribution
    plot_event_type_distribution(axes[0, 0], hist_stats, policy_stats)

    # 2. Direction Distribution
    plot_direction_distribution(axes[0, 1], hist_stats, policy_stats)

    # 3. Price Distribution Histogram
    plot_price_distribution(axes[1, 0], hist_data, policy_data)

    # 4. Size Distribution Histogram
    plot_size_distribution(axes[1, 1], hist_data, policy_data)

    # 5. Price Buckets Bar Chart
    ax = axes[2, 0]
    price_buckets = ['neg10_neg5', 'neg5_neg1', 'neg1_0', '0', '0_1', '1_5', '5_10']
    bucket_labels = ['-10~-5', '-5~-1', '-1~0', '0', '0~1', '1~5', '5~10']
    x = np.arange(len(price_buckets))
    width = 0.35

    hist_vals = [hist_stats.get('price', {}).get(f'bucket_{b}', 0) for b in price_buckets]
    policy_vals = [policy_stats.get('price', {}).get(f'bucket_{b}', 0) for b in price_buckets]

    ax.bar(x - width/2, hist_vals, width, label='Historical', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, policy_vals, width, label='Policy', color='coral', alpha=0.8)
    ax.set_xlabel('Price Bucket (ticks)')
    ax.set_ylabel('Ratio')
    ax.set_title('Price Distribution by Bucket')
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=45)
    ax.legend()

    # 6. Size Buckets Bar Chart
    ax = axes[2, 1]
    size_buckets = ['1_10', '11_50', '51_100', '101_500', '500_plus']
    size_labels = ['1-10', '11-50', '51-100', '101-500', '500+']
    x = np.arange(len(size_buckets))

    hist_vals = [hist_stats.get('size', {}).get(f'bucket_{b}', 0) for b in size_buckets]
    policy_vals = [policy_stats.get('size', {}).get(f'bucket_{b}', 0) for b in size_buckets]

    ax.bar(x - width/2, hist_vals, width, label='Historical', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, policy_vals, width, label='Policy', color='coral', alpha=0.8)
    ax.set_xlabel('Size Bucket')
    ax.set_ylabel('Ratio')
    ax.set_title('Size Distribution by Bucket')
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[*] Saved detailed analysis to: {save_path}")
    plt.close()
