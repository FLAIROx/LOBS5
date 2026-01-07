"""
Order Analysis Module for ES-LOBS5

Provides statistical analysis and visualization of:
1. Historical orders (from replay data)
2. Policy-generated orders (from ES training)
"""

from .order_statistics import (
    compute_order_stats,
    print_stats_comparison,
    is_valid_order,
    decode_and_analyze_orders,
    get_raw_order_data,
)
from .visualize import (
    plot_comparison,
    plot_price_distribution,
    plot_event_type_distribution,
    plot_detailed_analysis,
)

__all__ = [
    'compute_order_stats',
    'print_stats_comparison',
    'is_valid_order',
    'decode_and_analyze_orders',
    'get_raw_order_data',
    'plot_comparison',
    'plot_price_distribution',
    'plot_event_type_distribution',
    'plot_detailed_analysis',
]
