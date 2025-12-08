"""
PnL-based Fitness Functions for ES-JaxLOB Training.

In Evolution Strategies with JaxLOB, fitness = execution quality.
Higher PnL = better policy.


  Step-by-Step Interleaved Execution

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │               ES Eval (per thread, step-by-step interleaved)                │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │  Use jax.lax.scan to loop through each step:                                │
  │                                                                             │
  │  ┌──────────────────────────────────────────────────────────────────────┐   │
  │  │  Step t:                                                             │   │
  │  │                                                                      │   │
  │  │  1. World Model (frozen) generates background market messages        │   │
  │  │     └─→ world_msgs_t (K messages)                                    │   │
  │  │                                                                      │   │
  │  │  2. JaxLOB processes world_msgs_t                                    │   │
  │  │     └─→ book_state_t (updated order book)                            │   │
  │  │                                                                      │   │
  │  │  3. Policy (ES perturbed) **observes** book_state_t                  │   │
  │  │     └─→ policy_msg_t (1 trading action)                              │   │
  │  │                                                                      │   │
  │  │  4. JaxLOB processes policy_msg_t                                    │   │
  │  │     └─→ book_state_t' (state after execution)                        │   │
  │  │                                                                      │   │
  │  └──────────────────────────────────────────────────────────────────────┘   │
  │                                                                             │
  │  Repeat T steps → final_state.total_revenue = Fitness                       │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘

"""

import jax.numpy as jnp

__all__ = [
    'compute_pnl_fitness',
    'compute_execution_fitness',
    'compute_advantage_fitness',
    'compute_normalized_pnl_fitness',
]


def compute_pnl_fitness(total_revenue: float) -> float:
    """
    Compute fitness from JaxLOB execution total revenue.

    This is the primary fitness function for ES training.
    Higher revenue = better policy.

    Args:
        total_revenue: EnvState.total_revenue from JaxLOB

    Returns:
        Fitness score (same as total_revenue, higher is better)
    """
    return total_revenue


def compute_execution_fitness(
    total_revenue: float,
    slippage_rm: float = 0.0,
    vwap_rm: float = 0.0,
    init_price: float = 0.0,
    quant_executed: int = 0,
    slippage_weight: float = 0.0,
    vwap_weight: float = 0.0,
) -> float:
    """
    Compute combined execution quality fitness.

    Combines multiple execution metrics into a single fitness score.

    Args:
        total_revenue: Total revenue from execution
        slippage_rm: Rolling mean slippage
        vwap_rm: Rolling mean VWAP
        init_price: Initial price at episode start
        quant_executed: Total quantity executed
        slippage_weight: Weight for slippage penalty (0 = ignore)
        vwap_weight: Weight for VWAP deviation penalty (0 = ignore)

    Returns:
        Combined fitness score (higher is better)
    """
    fitness = total_revenue

    # Optionally penalize slippage
    if slippage_weight > 0:
        fitness = fitness - slippage_weight * jnp.abs(slippage_rm)

    # Optionally penalize VWAP deviation
    if vwap_weight > 0 and quant_executed > 0:
        avg_price = total_revenue / quant_executed
        vwap_deviation = jnp.abs(avg_price - vwap_rm)
        fitness = fitness - vwap_weight * vwap_deviation * quant_executed

    return fitness


def compute_advantage_fitness(
    total_revenue: float,
    vwap_rm: float,
    quant_executed: int,
) -> float:
    """
    Compute advantage over VWAP as fitness.

    Measures how much better/worse the policy performed compared
    to the market VWAP.

    Args:
        total_revenue: Total revenue from execution
        vwap_rm: Rolling mean VWAP
        quant_executed: Total quantity executed

    Returns:
        Advantage fitness (positive = beat VWAP, negative = underperformed)
    """
    if quant_executed == 0:
        return 0.0
    return total_revenue - vwap_rm * quant_executed


def compute_normalized_pnl_fitness(
    total_revenue: float,
    task_size: int,
    init_price: float,
    scale: float = 10000.0,
) -> float:
    """
    Compute normalized PnL fitness.

    Normalizes revenue by task size and initial price for comparable
    fitness across different market conditions.

    Args:
        total_revenue: Total revenue from execution
        task_size: Target execution quantity
        init_price: Initial price (tick-adjusted)
        scale: Normalization scale factor

    Returns:
        Normalized fitness in reasonable range
    """
    expected_revenue = task_size * init_price
    if expected_revenue == 0:
        return 0.0
    return (total_revenue - expected_revenue) / scale
