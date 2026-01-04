"""
PnL-based Fitness Functions for ES-JaxLOB Training.

In Evolution Strategies with JaxLOB, fitness = execution quality.
Higher PnL = better policy.

Step-by-Step Interleaved Execution:
- World Model generates K background market messages
- JaxLOB processes world_msgs → updates order book
- Policy generates 1 trading action
- JaxLOB processes policy_msg → updates state
- Repeat T steps → final_state.total_revenue = Fitness
"""

import jax.numpy as jnp

__all__ = [
    'compute_pnl_fitness',
    'compute_execution_fitness',
    'compute_advantage_fitness',
    'compute_normalized_pnl_fitness',
    'compute_cross_entropy_fitness',
]


def compute_pnl_fitness(total_revenue: float) -> float:
    """
    Compute fitness from JaxLOB execution total revenue.
    Higher revenue = better policy.
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
    Combines revenue, slippage penalty, and VWAP deviation.
    """
    fitness = total_revenue

    if slippage_weight > 0:
        fitness = fitness - slippage_weight * jnp.abs(slippage_rm)

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
    Positive = beat VWAP, Negative = underperformed.
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
    Normalizes by expected revenue for comparable fitness.
    """
    expected_revenue = task_size * init_price
    if expected_revenue == 0:
        return 0.0
    return (total_revenue - expected_revenue) / scale


def compute_cross_entropy_fitness(
    log_probs: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray = None,
) -> float:
    """
    Compute negative cross-entropy as fitness (for world model training).

    Higher = better predictions.
    This is used when training the world model to predict next tokens.

    Args:
        log_probs: (L, vocab_size) log probabilities
        targets: (L,) target token indices
        mask: (L,) optional mask for valid positions

    Returns:
        Negative mean cross-entropy (higher is better)
    """
    # Gather log probs at target indices
    L = targets.shape[0]
    ce = -log_probs[jnp.arange(L), targets]

    if mask is not None:
        ce = ce * mask
        return -jnp.sum(ce) / jnp.sum(mask)

    return -jnp.mean(ce)
