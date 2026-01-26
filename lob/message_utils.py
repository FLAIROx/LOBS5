"""
Lightweight message utilities for LOB generation.

This module contains simple message conversion functions that can be imported
without triggering heavy dependencies (gymnax_exchange, JaxLOB, etc.).

Used by both:
- lob/inference_no_errcorr.py
- es_lobs5/training/es_trainer.py
"""

import jax
import jax.numpy as jnp
from typing import Tuple

# Message field indices (14-column decoded format)
ORDER_ID_i = 0
EVENT_TYPE_i = 1
DIRECTION_i = 2
PRICE_ABS_i = 3
PRICE_i = 4
SIZE_i = 5
DTs_i = 6
DTns_i = 7
TIMEs_i = 8
TIMEns_i = 9
PRICE_REF_i = 10
SIZE_REF_i = 11
TIMEs_REF_i = 12
TIMEns_REF_i = 13


@jax.jit
def msg_to_jnp(m_raw: jax.Array) -> jax.Array:
    """Convert 14-column decoded message to 8-column JaxLOB format.

    Args:
        m_raw: (14,) decoded message array

    Returns:
        (8,) JaxLOB simulator message:
        [event_type, side*2-1, size, price_abs, order_id, 0(trader_id), time_s, time_ns]
    """
    m = m_raw.copy()

    return jnp.array([
        m[EVENT_TYPE_i],
        (m[DIRECTION_i] * 2) - 1,  # 0/1 -> -1/1
        m[SIZE_i],
        m[PRICE_ABS_i],
        m[ORDER_ID_i],
        0,  # TraderID
        m[TIMEs_i],
        m[TIMEns_i],
    ])


# Vectorized version for batch conversion
msgs_to_jnp = jax.vmap(msg_to_jnp)


@jax.jit
def construct_sim_msg(
        event_type: int,
        side: int,
        quantity: int,
        price: int,
        order_id: int,
        time_s: int,
        time_ns: int,
        trader_id: int = -88,
    ):
    """Construct JaxLOB simulator message.

    Args:
        event_type: 1=new, 2=cancel, 3=delete, 4=execute
        side: 0=sell, 1=buy (will be converted to -1/1)
        quantity: Order size
        price: Absolute price
        order_id: Order ID
        time_s: Timestamp seconds
        time_ns: Timestamp nanoseconds
        trader_id: Trader ID for multi-agent tracking (default -88)

    Returns:
        (8,) JaxLOB message array
    """
    return jnp.array([
        event_type,
        (side * 2) - 1,  # Convert 0/1 to -1/1
        quantity,
        price,
        order_id,
        trader_id,
        time_s,
        time_ns,
    ], dtype=jnp.int32)


@jax.jit
def construct_dummy_sim_msg(*args) -> jax.Array:
    """Create a dummy NOOP message (all -1)."""
    return jnp.ones((8,), dtype=jnp.int32) * (-1)


# NOTE: add_times() has different signature in inference_no_errcorr.py:
# - inference: add_times(a_s, a_ns, b_s, b_ns) - 4 separate args
# - If needed, import directly from inference_no_errcorr.py
