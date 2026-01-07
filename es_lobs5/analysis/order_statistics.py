"""
Order Statistics Analysis

Computes statistical metrics for decoded orders:
- Event type distribution (new/cancel/delete/execute)
- Direction distribution (buy/sell)
- Price distribution (relative to mid-price)
- Size distribution
- Time interval analysis
"""

import numpy as np
from typing import Dict, Optional, Tuple
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')


def decode_and_analyze_orders(
    tokens: np.ndarray,
    token_mode: int = 24,
) -> Tuple[np.ndarray, Dict]:
    """
    Decode order tokens and compute statistics.

    Args:
        tokens: shape (N, msg_len) encoded order tokens
        token_mode: 22 or 24 token mode

    Returns:
        (decoded_orders, stats_dict)
    """
    from lob.encoding import decode_msgs, Vocab

    v = Vocab(token_mode=token_mode)
    decoded = np.array(decode_msgs(tokens, v.ENCODING, token_mode=token_mode))
    stats = compute_order_stats(decoded)
    return decoded, stats


def compute_order_stats(decoded_orders: np.ndarray) -> Dict:
    """
    Compute order statistics from decoded orders.

    Args:
        decoded_orders: shape (N, 14) array with fields:
            [OID, event_type, direction, price_abs, price, size,
             delta_t_s, delta_t_ns, time_s, time_ns,
             p_ref, size_ref, time_s_ref, time_ns_ref]

    Returns:
        Dictionary of statistics
    """
    if len(decoded_orders) == 0:
        return {'error': 'No orders to analyze'}

    stats = {}
    n_orders = len(decoded_orders)
    stats['n_orders'] = n_orders

    # ========================================
    # 1. Event Type Distribution
    # ========================================
    event_types = decoded_orders[:, 1].astype(int)
    valid_events = (event_types >= 1) & (event_types <= 4)
    stats['event_type'] = {
        'new': float(np.mean(event_types == 1)),
        'cancel': float(np.mean(event_types == 2)),
        'delete': float(np.mean(event_types == 3)),
        'execute': float(np.mean(event_types == 4)),
        'invalid': float(np.mean(~valid_events)),
        'raw_counts': {
            'new': int(np.sum(event_types == 1)),
            'cancel': int(np.sum(event_types == 2)),
            'delete': int(np.sum(event_types == 3)),
            'execute': int(np.sum(event_types == 4)),
        }
    }

    # ========================================
    # 2. Direction Distribution
    # ========================================
    directions = decoded_orders[:, 2].astype(int)
    valid_dir = (directions >= 0) & (directions <= 1)
    stats['direction'] = {
        'sell': float(np.mean(directions == 0)),
        'buy': float(np.mean(directions == 1)),
        'invalid': float(np.mean(~valid_dir)),
    }

    # ========================================
    # 3. Price Distribution (relative to mid-price)
    # ========================================
    prices = decoded_orders[:, 4].astype(float)
    # Filter out NA values (-9999)
    valid_prices = prices[prices > -9000]
    if len(valid_prices) > 0:
        stats['price'] = {
            'mean': float(np.mean(valid_prices)),
            'std': float(np.std(valid_prices)),
            'min': float(np.min(valid_prices)),
            'max': float(np.max(valid_prices)),
            'median': float(np.median(valid_prices)),
            # Aggressive = within 1 tick of mid (crosses or improves spread)
            'aggressive_ratio': float(np.mean(np.abs(valid_prices) <= 1)),
            # Passive = far from mid (unlikely to execute immediately)
            'passive_ratio': float(np.mean(np.abs(valid_prices) > 5)),
            'n_valid': len(valid_prices),
            # Price distribution buckets
            'bucket_neg10_neg5': float(np.mean((valid_prices >= -10) & (valid_prices < -5))),
            'bucket_neg5_neg1': float(np.mean((valid_prices >= -5) & (valid_prices < -1))),
            'bucket_neg1_0': float(np.mean((valid_prices >= -1) & (valid_prices < 0))),
            'bucket_0': float(np.mean(valid_prices == 0)),
            'bucket_0_1': float(np.mean((valid_prices > 0) & (valid_prices <= 1))),
            'bucket_1_5': float(np.mean((valid_prices > 1) & (valid_prices <= 5))),
            'bucket_5_10': float(np.mean((valid_prices > 5) & (valid_prices <= 10))),
        }
    else:
        stats['price'] = {'error': 'No valid prices'}

    # ========================================
    # 4. Size Distribution
    # ========================================
    sizes = decoded_orders[:, 5].astype(float)
    valid_sizes = sizes[(sizes > 0) & (sizes < 10000)]
    if len(valid_sizes) > 0:
        stats['size'] = {
            'mean': float(np.mean(valid_sizes)),
            'std': float(np.std(valid_sizes)),
            'min': float(np.min(valid_sizes)),
            'max': float(np.max(valid_sizes)),
            'median': float(np.median(valid_sizes)),
            'n_valid': len(valid_sizes),
            # Size buckets
            'bucket_1_10': float(np.mean((valid_sizes >= 1) & (valid_sizes <= 10))),
            'bucket_11_50': float(np.mean((valid_sizes > 10) & (valid_sizes <= 50))),
            'bucket_51_100': float(np.mean((valid_sizes > 50) & (valid_sizes <= 100))),
            'bucket_101_500': float(np.mean((valid_sizes > 100) & (valid_sizes <= 500))),
            'bucket_500_plus': float(np.mean(valid_sizes > 500)),
        }
    else:
        stats['size'] = {'error': 'No valid sizes'}

    # ========================================
    # 5. Time Interval Analysis
    # ========================================
    delta_t_s = decoded_orders[:, 6].astype(float)
    delta_t_ns = decoded_orders[:, 7].astype(float)
    # Combine seconds and nanoseconds
    delta_t_total = delta_t_s + delta_t_ns / 1e9
    # Filter out invalid values
    valid_dt = delta_t_total[(delta_t_total >= 0) & (delta_t_total < 1000)]
    if len(valid_dt) > 0:
        stats['time'] = {
            'mean_interval_s': float(np.mean(valid_dt)),
            'std_interval_s': float(np.std(valid_dt)),
            'min_interval_s': float(np.min(valid_dt)),
            'max_interval_s': float(np.max(valid_dt)),
            'median_interval_s': float(np.median(valid_dt)),
            'n_valid': len(valid_dt),
        }
    else:
        stats['time'] = {'error': 'No valid time intervals'}

    # ========================================
    # 6. Validity Analysis
    # ========================================
    validity = np.array([is_valid_order(o) for o in decoded_orders])
    stats['validity'] = {
        'valid_ratio': float(np.mean(validity)),
        'n_valid': int(np.sum(validity)),
        'n_invalid': int(np.sum(~validity)),
    }

    return stats


def is_valid_order(decoded_order: np.ndarray) -> bool:
    """
    Check if a decoded order is valid.

    Args:
        decoded_order: shape (14,) decoded order

    Returns:
        True if order is valid
    """
    event_type = int(decoded_order[1])
    direction = int(decoded_order[2])
    price = float(decoded_order[4])
    size = float(decoded_order[5])

    is_valid = (
        (1 <= event_type <= 4) and
        (direction in [0, 1]) and
        (-999 <= price <= 999) and
        (0 < size < 10000)
    )
    return is_valid


def print_stats_comparison(
    hist_stats: Dict,
    policy_stats: Dict,
    title: str = "订单统计对比"
) -> str:
    """
    Print comparison report between historical and policy orders.

    Args:
        hist_stats: Statistics from historical orders
        policy_stats: Statistics from policy-generated orders
        title: Report title

    Returns:
        Formatted report string
    """
    sep = "=" * 60
    report = []
    report.append(sep)
    report.append(f" {title} ".center(60))
    report.append(sep)
    report.append(f"{'':20} {'历史订单':>15} {'Policy订单':>15}")
    report.append("-" * 60)

    # Number of orders
    report.append(f"{'订单数量':20} {hist_stats.get('n_orders', 'N/A'):>15} {policy_stats.get('n_orders', 'N/A'):>15}")
    report.append("")

    # Event Type
    report.append("Event Type:")
    for et in ['new', 'cancel', 'delete', 'execute', 'invalid']:
        h_val = hist_stats.get('event_type', {}).get(et, 0)
        p_val = policy_stats.get('event_type', {}).get(et, 0)
        report.append(f"  {et:18} {h_val:>14.2%} {p_val:>14.2%}")
    report.append("")

    # Direction
    report.append("Direction:")
    for d in ['sell', 'buy']:
        h_val = hist_stats.get('direction', {}).get(d, 0)
        p_val = policy_stats.get('direction', {}).get(d, 0)
        report.append(f"  {d:18} {h_val:>14.2%} {p_val:>14.2%}")
    report.append("")

    # Price
    report.append("Price (相对mid):")
    for metric in ['mean', 'std', 'median', 'aggressive_ratio', 'passive_ratio']:
        h_val = hist_stats.get('price', {}).get(metric, 0)
        p_val = policy_stats.get('price', {}).get(metric, 0)
        if 'ratio' in metric:
            report.append(f"  {metric:18} {h_val:>14.2%} {p_val:>14.2%}")
        else:
            report.append(f"  {metric:18} {h_val:>14.2f} {p_val:>14.2f}")
    report.append("")

    # Size
    report.append("Size:")
    for metric in ['mean', 'std', 'median', 'min', 'max']:
        h_val = hist_stats.get('size', {}).get(metric, 0)
        p_val = policy_stats.get('size', {}).get(metric, 0)
        report.append(f"  {metric:18} {h_val:>14.1f} {p_val:>14.1f}")
    report.append("")

    # Time
    report.append("时间间隔 (秒):")
    for metric in ['mean_interval_s', 'std_interval_s', 'median_interval_s']:
        h_val = hist_stats.get('time', {}).get(metric, 0)
        p_val = policy_stats.get('time', {}).get(metric, 0)
        report.append(f"  {metric:18} {h_val:>14.4f} {p_val:>14.4f}")
    report.append("")

    # Validity
    report.append("订单有效性:")
    h_val = hist_stats.get('validity', {}).get('valid_ratio', 0)
    p_val = policy_stats.get('validity', {}).get('valid_ratio', 0)
    report.append(f"  {'valid_ratio':18} {h_val:>14.2%} {p_val:>14.2%}")

    report.append(sep)

    report_str = "\n".join(report)
    print(report_str)
    return report_str


def get_raw_order_data(decoded_orders: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract raw order data for visualization.

    Args:
        decoded_orders: shape (N, 14) decoded orders

    Returns:
        Dictionary with raw arrays for each field
    """
    return {
        'event_type': decoded_orders[:, 1].astype(int),
        'direction': decoded_orders[:, 2].astype(int),
        'price': decoded_orders[:, 4].astype(float),
        'size': decoded_orders[:, 5].astype(float),
        'delta_t_s': decoded_orders[:, 6].astype(float),
        'delta_t_ns': decoded_orders[:, 7].astype(float),
    }
