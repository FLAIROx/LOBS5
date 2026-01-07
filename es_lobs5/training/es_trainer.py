"""
ES Training with JaxLOB (Pure ES, Step-by-Step Interleaved).

This module implements Evolution Strategies training for trading policies
using JaxLOB as the execution environment.

Architecture (Step-by-Step):
    For each step t:
        1. World Model (frozen) generates K background market messages
           OR Historical Replay loads K messages from data
        2. JaxLOB processes background msgs -> updates order book
        3. Policy (ES perturbed) **observes** updated book state
        4. Policy generates 1 trading action
        5. JaxLOB processes policy_msg -> updates state

    Repeat T steps -> final_state -> Fitness (PnL)

Key Features:
- Both World Model and Policy initialized from same LOBS5 checkpoint
- World Model stays frozen (iterinfo=None), Policy trained with EGGROLL
- Policy can observe market changes before making decisions
- Fitness = PnL (profit/loss based on execution quality)
- Supports two background modes: world_model (autoregressive) and historical_replay (data)

gymanx_exchange env path: https://github.com/KangOxford/JaxMARL-HFT
"""

import os
import jax

# ============================================================================
# G4: Configure JAX persistent compilation cache (HyperscaleES pattern)
# This caches XLA compilation results to disk, allowing subsequent runs
# to skip the expensive compilation step (~150s -> <10s for warm start)
# Reference: HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py:9-11
# ============================================================================
_jax_cache_dir = os.path.expanduser("~/.cache/es_lobs5_jax_compilation")
os.makedirs(_jax_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)  # Cache all sizes
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # Cache all compile times
# ============================================================================

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial
import argparse
from tqdm import tqdm
import time
from typing import Tuple, Optional, NamedTuple, Dict, Any

# Lazy imports - these are loaded on first use to avoid import errors
OrderBook = None
LobState = None
Message_Tokenizer = None
encoding = None
get_best_bid_and_ask = None
_all_noisers = None
_ES_PaddedLobPredModel = None
_CommonParams = None
_simple_es_tree_key = None
_load_checkpoint_for_es = None


def _get_all_noisers():
    """Lazy load noisers."""
    global _all_noisers
    if _all_noisers is None:
        from ..utils.import_utils import get_all_noisers
        _all_noisers = get_all_noisers()
    return _all_noisers


def _get_es_model():
    """Lazy load ES model."""
    global _ES_PaddedLobPredModel
    if _ES_PaddedLobPredModel is None:
        from ..models import ES_PaddedLobPredModel as _model
        _ES_PaddedLobPredModel = _model
    return _ES_PaddedLobPredModel


def _get_common_params():
    """Lazy load CommonParams."""
    global _CommonParams, _simple_es_tree_key
    if _CommonParams is None:
        from ..models.common import CommonParams as _cp, simple_es_tree_key as _key
        _CommonParams = _cp
        _simple_es_tree_key = _key
    return _CommonParams


def _get_checkpoint_loader():
    """Lazy load checkpoint adapter."""
    global _load_checkpoint_for_es
    if _load_checkpoint_for_es is None:
        from ..adapters.checkpoint_adapter import load_checkpoint_for_es as _loader
        _load_checkpoint_for_es = _loader
    return _load_checkpoint_for_es

__all__ = ['ESTrainer', 'create_es_config', 'es_train']


def _lazy_import_jaxlob():
    """Lazy import JaxLOB to avoid import errors when not using this mode."""
    global OrderBook, LobState, Message_Tokenizer, encoding, get_best_bid_and_ask
    if OrderBook is None:
        from gymnax_exchange.jaxob.jorderbook import OrderBook as _OrderBook, LobState as _LobState
        from gymnax_exchange.jaxob.JaxOrderBookArrays import get_best_bid_and_ask as _get_best_bid_and_ask
        from lob.encoding import Message_Tokenizer as _Message_Tokenizer
        import lob.encoding as _encoding
        OrderBook = _OrderBook
        LobState = _LobState
        Message_Tokenizer = _Message_Tokenizer
        encoding = _encoding
        get_best_bid_and_ask = _get_best_bid_and_ask


# ============================================================================
# Field-Aware Token Masking for Constrained Decoding (24-token mode)
# ============================================================================
# Direct implementation for 24-token vocabulary structure:
#   Special: 0-3 (MASK, HIDDEN, NA, START)
#   time: 4-1003 (1000 values)
#   event_type: 1004-1007 (4 values: 1=new, 2=cancel, 3=delete, 4=execute)
#   size_digit: 1008-1107 (100 values: 0-99 for base-100)
#   price: 1108-2107 (1000 values: 0-999)
#   sign: 2108-2109 (2 values: -1, 1)
#   direction: 2110-2111 (2 values: 0=sell, 1=buy)
# ============================================================================

# Position -> (field_name, token_min, token_max) for 24-token messages
POSITION_TOKEN_RANGES_24 = {
    0: ("event_type", 1004, 1007),
    1: ("direction", 2110, 2111),
    2: ("price_sign", 2108, 2109),
    3: ("price", 1108, 2107),
    4: ("size_high", 1008, 1107),
    5: ("size_low", 1008, 1107),
    6: ("delta_t_s", 4, 1003),
    7: ("delta_t_ns_0", 4, 1003),
    8: ("delta_t_ns_1", 4, 1003),
    9: ("delta_t_ns_2", 4, 1003),
    10: ("time_s_0", 4, 1003),
    11: ("time_s_1", 4, 1003),
    12: ("time_ns_0", 4, 1003),
    13: ("time_ns_1", 4, 1003),
    14: ("time_ns_2", 4, 1003),
    15: ("price_ref_sign", 2108, 2109),
    16: ("price_ref", 1108, 2107),
    17: ("size_ref_high", 1008, 1107),
    18: ("size_ref_low", 1008, 1107),
    19: ("time_s_ref_0", 4, 1003),
    20: ("time_s_ref_1", 4, 1003),
    21: ("time_ns_ref_0", 4, 1003),
    22: ("time_ns_ref_1", 4, 1003),
    23: ("time_ns_ref_2", 4, 1003),
}

_FIELD_MASKS_24 = None

def get_field_masks_24(vocab_size: int = 2112):
    """Get field masks for constrained decoding (additive mask format).

    Creates masks directly from POSITION_TOKEN_RANGES_24, avoiding
    syntax_validation_matrix which has compatibility issues with 24-token mode.

    Returns:
        jnp.array of shape (24, vocab_size) where:
        - 0.0 for valid tokens
        - -1e9 for invalid tokens
    """
    global _FIELD_MASKS_24
    if _FIELD_MASKS_24 is None:
        masks = []
        for pos in range(24):
            _, tok_min, tok_max = POSITION_TOKEN_RANGES_24[pos]
            # Start with -1e9 (invalid) for all tokens
            mask = jnp.full(vocab_size, -1e9)
            # Set valid range to 0.0
            mask = mask.at[tok_min:tok_max+1].set(0.0)
            masks.append(mask)
        _FIELD_MASKS_24 = jnp.stack(masks)
    return _FIELD_MASKS_24


def create_es_config():
    """Create argument parser for ES training configuration."""
    parser = argparse.ArgumentParser(description='ES JaxLOB Training for LOBS5')

    # LOBS5 checkpoint (for both World Model and Policy initialization)
    parser.add_argument('--lobs5_checkpoint', type=str, required=True,
                        help='Path to LOBS5 checkpoint for model initialization')

    # ES configuration
    parser.add_argument('--noiser', type=str, default='eggroll',
                        choices=['open_es', 'eggroll', 'eggrollbs', 'sparse'])
    parser.add_argument('--sigma', type=float, default=0.01, help='Noise std')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=4, help='LORA rank')

    # Training configuration
    parser.add_argument('--n_perturbations', type=int, default=128,
                        help='Population size (number of ES perturbations, must be divisible by n_devices)')
    # Legacy alias alias
    parser.add_argument('--n_threads', type=int, default=None,
                        help='[DEPRECATED] Use --n_perturbations instead')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--n_steps', type=int, default=100, help='Steps per episode')
    parser.add_argument('--n_warmup_msgs', type=int, default=500,
                        help='Number of warmup messages to replay before episode starts (0 = no warmup)')
    parser.add_argument('--background_msgs_per_step', type=int, default=10,
                        help='Background messages per step (applies to both world_model and historical_replay)')
    # Legacy alias alias
    parser.add_argument('--world_msgs_per_step', type=int, default=None,
                        help='[DEPRECATED] Use --background_msgs_per_step instead')

    # Execution task
    parser.add_argument('--task', type=str, default='sell',
                        choices=['sell', 'buy'])
    parser.add_argument('--task_size', type=int, default=500,
                        help='Shares to execute')
    parser.add_argument('--tick_size', type=int, default=100,
                        help='Tick size in cents')

    # Token mode
    parser.add_argument('--token_mode', type=int, default=22, choices=[22, 24],
                        help='Token mode: 22 (single token size) or 24 (base-100 size)')

    # Background model configuration
    parser.add_argument('--background_mode', type=str, default='world_model',
                        choices=['world_model', 'historical_replay'],
                        help='Background message generation mode')
    parser.add_argument('--replay_data_path', type=str, default=None,
                        help='Path to historical data directory for replay mode')

    # Data directory for initial state
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to LOBSTER data directory for initial state')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./es_checkpoints')

    # W&B logging
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/username')

    return parser


# =============================================================================
# Data Format Detection and Logging Utilities
# =============================================================================
def detect_data_format(file_path: str, data: 'np.ndarray', token_mode: int = 24) -> dict:
    """
    Detect and log the data format of a loaded file.

    Args:
        file_path: Path to the loaded file
        data: Loaded numpy array
        token_mode: Expected token mode (22 or 24)

    Returns:
        dict with format info:
            - file_type: 'npy' or 'csv'
            - format_type: 'preproc', 'encoded', or 'unknown'
            - n_cols: number of columns
            - n_rows: number of rows
            - dtype: data type
    """
    import os

    # Detect file type
    file_ext = os.path.splitext(file_path)[1].lower()
    file_type = 'npy' if file_ext == '.npy' else ('csv' if file_ext == '.csv' else file_ext)

    # Detect data format
    n_cols = data.shape[1] if len(data.shape) > 1 else 1
    n_rows = data.shape[0]
    dtype = str(data.dtype)

    if n_cols == 14:
        format_type = 'preproc'
        format_desc = 'PREPROC (14 cols, raw decoded)'
    elif n_cols == token_mode:
        format_type = 'encoded'
        format_desc = f'ENCODED ({n_cols} cols, tokenized)'
    elif n_cols in [22, 24]:
        format_type = 'encoded'
        format_desc = f'ENCODED ({n_cols} cols, tokenized, token_mode mismatch)'
    elif n_cols == 43:
        # LOBSTER orderbook format: 10 levels × 2 sides × 2 fields (price, size) + 3 (bid_time, ask_time, seq_num)
        # Columns: [ask_price1, ask_size1, ..., ask_price10, ask_size10, bid_price1, bid_size1, ..., bid_price10, bid_size10, bid_time, ask_time, seq_num]
        format_type = 'orderbook'
        format_desc = f'ORDERBOOK (43 cols: 10-level LOB × 2 sides × 2 fields + 3 meta)'
    elif n_cols == 41:
        # LOBSTER orderbook format without seq_num: 10 levels × 2 sides × 2 fields + 1 (timestamp)
        format_type = 'orderbook'
        format_desc = f'ORDERBOOK (41 cols: 10-level LOB × 2 sides × 2 fields + 1 timestamp)'
    elif n_cols == 21:
        # LOBSTER orderbook format: 5 levels × 2 sides × 2 fields + 1 (timestamp)
        format_type = 'orderbook'
        format_desc = f'ORDERBOOK (21 cols: 5-level LOB × 2 sides × 2 fields + 1 timestamp)'
    else:
        format_type = 'unknown'
        format_desc = f'UNKNOWN ({n_cols} cols)'

    return {
        'file_type': file_type,
        'format_type': format_type,
        'format_desc': format_desc,
        'n_cols': n_cols,
        'n_rows': n_rows,
        'dtype': dtype,
        'file_path': file_path,
    }


def log_data_format(info: dict, prefix: str = "[DATA]") -> None:
    """
    Log data format information.

    Args:
        info: dict from detect_data_format()
        prefix: Log prefix string
    """
    import os
    print(f"{prefix} File: {os.path.basename(info['file_path'])}")
    print(f"{prefix}   Type: {info['file_type'].upper()}")
    print(f"{prefix}   Format: {info['format_desc']}")
    print(f"{prefix}   Shape: ({info['n_rows']}, {info['n_cols']}), dtype: {info['dtype']}")


# Helper function to convert decoded messages to JaxLOB format
@jax.jit
def decoded_msg_to_jaxlob_format(msg_decoded: jax.Array) -> jax.Array:
    """
    Convert 14-column decoded message to 8-column JaxLOB format.

    Args:
        msg_decoded: (14,) decoded message

    Returns:
        (8,) JaxLOB message [type, side, qty, price, trade_id, order_id, time_s, time_ns]
    """
    ORDER_ID_i = 0
    EVENT_TYPE_i = 1
    DIRECTION_i = 2
    PRICE_ABS_i = 3
    SIZE_i = 5
    TIMEs_i = 8
    TIMEns_i = 9

    return jnp.array([
        msg_decoded[EVENT_TYPE_i],
        (msg_decoded[DIRECTION_i] * 2) - 1,  # 0/1 -> -1/1
        msg_decoded[SIZE_i],
        msg_decoded[PRICE_ABS_i],
        0,  # trade_id
        msg_decoded[ORDER_ID_i],
        msg_decoded[TIMEs_i],
        msg_decoded[TIMEns_i],
    ], dtype=jnp.int32)


# Vectorized version for batch conversion
msgs_to_jnp = jax.jit(jax.vmap(decoded_msg_to_jaxlob_format))


def get_sim_msg_es(
    pred_msg_tokens: jnp.ndarray,
    sim: 'OrderBook',
    sim_state: 'LobState',
    mid_price: int,
    order_id: int,
    tick_size: int,
    encoder: Dict,
    trader_id: int = -88,
    token_mode: int = 22,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert predicted message tokens to JaxLOB format.

    Args:
        pred_msg_tokens: (22/24,) int32 - predicted message tokens
        sim: OrderBook instance
        sim_state: Current LobState
        mid_price: Current mid price
        order_id: Order ID to assign
        tick_size: Tick size
        encoder: Token encoder
        trader_id: Trader ID for tracking
        token_mode: Token mode (22 or 24)

    Returns:
        (sim_msg, msg_decoded)
    """
    _lazy_import_jaxlob()

    # Decode tokens to message fields
    msg_decoded = encoding.decode_msg(pred_msg_tokens, encoder, token_mode=token_mode)

    # Extract fields
    event_type = msg_decoded[1]
    quantity = msg_decoded[5]
    side = msg_decoded[2]
    rel_price = msg_decoded[4]
    time_s = msg_decoded[8]
    time_ns = msg_decoded[9]

    # Fault tolerance: validate and clamp decoded values
    is_valid_event = (event_type >= 1) & (event_type <= 4)
    is_valid_side = (side >= 0) & (side <= 1)
    is_valid_qty = quantity > 0
    is_valid_price = (rel_price >= -1000) & (rel_price <= 1000)
    is_valid_msg = is_valid_event & is_valid_side & is_valid_qty & is_valid_price

    # If invalid, create NOOP message (qty=0)
    safe_event_type = jnp.where(is_valid_msg, event_type, 0)
    safe_quantity = jnp.where(is_valid_msg, quantity, 0)
    safe_side = jnp.where(is_valid_msg, side, 0)
    safe_rel_price = jnp.where(is_valid_msg, rel_price, 0)

    # Calculate absolute price
    p_abs = mid_price + safe_rel_price * tick_size
    p_abs = jnp.maximum(p_abs, tick_size)

    # Construct JaxLOB message
    sim_msg = jnp.array([
        safe_event_type,
        (safe_side * 2) - 1,
        safe_quantity,
        p_abs,
        order_id,
        trader_id,
        time_s,
        time_ns,
    ], dtype=jnp.int32)

    return sim_msg, msg_decoded


def transform_L2_state_wrapper(
    cfg: 'Configuration',
    sim_state: 'LobState',
    price_levels: int = 500,
    tick_size: int = 100,
    in_shard_map: bool = False,
) -> jnp.ndarray:
    """
    Convert JaxLOB sim_state to model book input.

    Args:
        cfg: JaxLOB Configuration (not used, kept for API compatibility)
        sim_state: JaxLOB LobState
        price_levels: Volume image size (default 500)
        tick_size: Tick size in cents
        in_shard_map: If True, use pure versions without internal JIT to avoid
                      device placement conflicts in shard_map context.

    Returns:
        book_feat: (503,) = [mid_diff, time_s_norm, time_ns_norm, volume_image(500)]
    """
    _lazy_import_jaxlob()

    # Select function versions based on context
    if in_shard_map:
        # Pure versions for shard_map compatibility (no internal JIT)
        from gymnax_exchange.jaxob.JaxOrderBookArrays import get_L2_state_pure
        from preproc import transform_L2_state_pure
        _get_L2 = get_L2_state_pure
        _transform = transform_L2_state_pure
    else:
        # Original JIT versions for single-GPU performance
        from gymnax_exchange.jaxob.JaxOrderBookArrays import get_L2_state
        from preproc import transform_L2_state_gpu
        _get_L2 = get_L2_state
        _transform = transform_L2_state_gpu

    # Extract L2 from JaxLOB
    # Note: get_L2_state signature is (asks, bids, n_levels, cfg)
    l2_state = _get_L2(sim_state.asks, sim_state.bids, 10, cfg)
    l2_state = jnp.asarray(l2_state, dtype=jnp.int32)

    # Construct (43,) input
    metadata = jnp.array([0, 34200, 0], dtype=jnp.int32)
    book_input = jnp.concatenate([metadata, l2_state])

    # Apply training transform
    book_input_batched = book_input[None, :]
    book_feat_batched = _transform(book_input_batched, price_levels, tick_size)
    return book_feat_batched[0]


def get_mid_price(cfg: 'Configuration', sim_state: 'LobState', tick_size: int = 100) -> int:
    """Get current mid price from order book state.

    Args:
        cfg: JaxLOB Configuration (required for get_best_bid_and_ask)
        sim_state: Current LobState
        tick_size: Tick size in cents
    """
    _lazy_import_jaxlob()
    DEFAULT_MID = 10000

    best_ask, best_bid = get_best_bid_and_ask(cfg, sim_state.asks, sim_state.bids)

    bid_valid = (best_bid > 0) & (best_bid < 900000000)
    ask_valid = (best_ask > 0) & (best_ask < 900000000)

    best_bid = jnp.where(bid_valid, best_bid, DEFAULT_MID - tick_size)
    best_ask = jnp.where(ask_valid, best_ask, DEFAULT_MID + tick_size)

    mid = (best_bid + best_ask) // 2
    mid = jnp.where((mid > 0) & jnp.isfinite(mid), mid, DEFAULT_MID)

    return (mid // tick_size) * tick_size


class ESTrainer:
    """
    ES Trainer with JaxLOB environment.

    Implements step-by-step interleaved simulation where:
    - World Model generates background market order flow
    - Policy observes and generates trading actions
    - Both interact through JaxLOB order book simulation
    """

    def __init__(self, config):
        """Initialize ES trainer."""
        print("[INIT] Starting ESTrainer initialization")

        self.config = config

        # Legacy alias: n_threads -> n_perturbations
        if hasattr(config, 'n_threads') and getattr(config, 'n_threads', None) is not None:
            if not hasattr(config, 'n_perturbations') or getattr(config, 'n_perturbations', 128) == 128:
                print("[WARN] --n_threads is deprecated, use --n_perturbations instead")
                config.n_perturbations = config.n_threads
        # Ensure n_perturbations exists
        if not hasattr(config, 'n_perturbations'):
            config.n_perturbations = getattr(config, 'n_threads', 128)

        # Legacy alias: world_msgs_per_step -> background_msgs_per_step
        if hasattr(config, 'world_msgs_per_step') and getattr(config, 'world_msgs_per_step', None) is not None:
            if not hasattr(config, 'background_msgs_per_step') or getattr(config, 'background_msgs_per_step', 10) == 10:
                print("[WARN] --world_msgs_per_step is deprecated, use --background_msgs_per_step instead")
                config.background_msgs_per_step = config.world_msgs_per_step
        # Ensure background_msgs_per_step exists
        if not hasattr(config, 'background_msgs_per_step'):
            config.background_msgs_per_step = getattr(config, 'world_msgs_per_step', 10)

        _lazy_import_jaxlob()

        # Load LOBS5 checkpoint
        print(f"[INIT] Loading checkpoint from {config.lobs5_checkpoint}")
        load_checkpoint_for_es = _get_checkpoint_loader()
        self.lobs5_init, self.es_tree_key = load_checkpoint_for_es(config.lobs5_checkpoint)

        # Initialize noiser for Policy
        self._init_noiser()

        # Initialize JaxLOB simulator
        self._init_jaxlob()

        # Initialize historical replay data if needed
        self._init_historical_replay_data()

        print("[INIT] ESTrainer initialization complete")
        print(f"[INIT]   n_perturbations: {config.n_perturbations}")
        print(f"[INIT]   n_steps: {config.n_steps}")
        print(f"[INIT]   background_mode: {config.background_mode}")

        # ========================================================================
        # H3: Multi-GPU Mesh Configuration (MUST be before _compile_eval_batch)
        # Reference: HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py
        # Creates a 1D mesh along 'data' axis for data-parallel ES evaluation
        # ========================================================================
        self._n_devices = len(jax.devices())
        self._mesh = Mesh(jax.devices(), ('data',))
        print(f"[H3] Created mesh with {self._n_devices} devices")
        print(f"[H3] Mesh axis: {self._mesh.axis_names}")

        # ========================================================================
        # G1 + H1: Pre-compile eval_batch for faster subsequent epochs
        # The first epoch will trigger actual XLA compilation, but subsequent
        # epochs will reuse the cached compilation.
        # H1: With mesh available, this will use shard_map for multi-GPU.
        # ========================================================================
        print("[INIT] Building eval_batch function (compilation on first call)...")
        self._compiled_eval_batch = self._compile_eval_batch()

    def _init_noiser(self):
        """Initialize EGGROLL noiser for Policy.

        Note: solver=None uses default optax.sgd. The init_noiser API expects
        a callable (like optax.sgd), not a pre-built optimizer chain.
        See learned_lessons.md Lesson 4 for details.

        For EggRollBS (baseline subtraction), group_size must be > 0.
        Threads 0,1 in each group are baselines (no noise).
        """
        config = self.config
        all_noisers = _get_all_noisers()
        NOISER = all_noisers[config.noiser]

        self.noiser_cls = NOISER

        # Get group_size from config (required for EggRollBS, default 0 for EggRoll)
        group_size = getattr(config, 'group_size', 0)

        self.frozen_noiser_params, self.noiser_params = NOISER.init_noiser(
            self.lobs5_init.params,
            sigma=config.sigma,
            lr=config.lr,
            rank=config.lora_rank,
            freeze_nonlora=False,
            noise_reuse=0,
            group_size=group_size,
            solver=None,  # Uses default optax.sgd
        )

    def _init_jaxlob(self):
        """Initialize JaxLOB order book simulator.

        Uses Configuration-based API (nOrders/nTrades are in the config).
        """
        _lazy_import_jaxlob()

        # JaxLOB OrderBook now uses Configuration-based API
        from gymnax_exchange.jaxob.jaxob_config import Configuration
        from dataclasses import replace

        # Calculate required capacity
        n_warmup = getattr(self.config, 'n_warmup_msgs', 500)
        expected_orders = n_warmup + self.config.n_steps * (self.config.background_msgs_per_step + 1)
        n_orders = max(1000, int(expected_orders * 1.5))
        n_trades = max(500, self.config.n_steps * 2)

        # Create configuration with custom capacity
        jaxlob_cfg = replace(Configuration(), nOrders=n_orders, nTrades=n_trades)
        self.jaxlob_cfg = jaxlob_cfg  # Store for use in get_mid_price
        self.sim = OrderBook(cfg=jaxlob_cfg)

        print(f"[INIT] JaxLOB OrderBook initialized:")
        print(f"  nOrders: {n_orders} (capacity for order book)")
        print(f"  nTrades: {n_trades} (capacity for trade history)")
        print(f"  expected_orders: {expected_orders} ({n_warmup} warmup + {self.config.n_steps} steps × {self.config.background_msgs_per_step + 1} msgs)")

        # Create encoder from Vocab
        from lob.encoding import Vocab
        vocab = Vocab(token_mode=self.config.token_mode)
        self.encoder = vocab.ENCODING
        print(f"[INIT] token_mode: {self.config.token_mode}")

    def _init_historical_replay_data(self):
        """Pre-load historical data for replay mode."""
        if self.config.background_mode != 'historical_replay':
            self.replay_data_raw = None
            self.replay_tokens = None
            return

        if self.config.replay_data_path is None:
            raise ValueError("--replay_data_path required when background_mode=historical_replay")

        import os
        import glob
        import numpy as np
        from lob.encoding import encode_msgs

        data_path = self.config.replay_data_path
        message_files = sorted(glob.glob(os.path.join(data_path, '*message*proc.npy')))

        if len(message_files) == 0:
            raise FileNotFoundError(f"No message files found in {data_path}")

        # Use fixed file_idx if specified, otherwise random
        if hasattr(self.config, 'file_idx') and self.config.file_idx is not None:
            file_idx = self.config.file_idx % len(message_files)
        else:
            file_idx = np.random.randint(0, len(message_files))
        selected_file = message_files[file_idx]

        self.replay_data_date = os.path.basename(selected_file).split('_')[1]
        self.replay_data_dir = data_path

        msg_data = np.load(selected_file)

        # Log data format using standard utilities
        format_info = detect_data_format(selected_file, msg_data, self.config.token_mode)
        log_data_format(format_info, prefix="[INIT-REPLAY]")

        # Auto-detect data format: preproc (N, 14) vs encoded (N, 24)
        if format_info['format_type'] == 'preproc':
            # Preproc format: need to encode to tokens
            print(f"[INIT-REPLAY] Action: encoding to {self.config.token_mode}-token format...")
            self.replay_tokens = encode_msgs(msg_data, self.encoder, token_mode=self.config.token_mode)
            self.replay_data_raw = jnp.array(msg_data)
        elif format_info['format_type'] == 'encoded':
            # Encoded format: already tokenized, use directly
            print(f"[INIT-REPLAY] Action: using tokens directly (already encoded)")
            self.replay_tokens = jnp.array(msg_data)
            # For raw data, we need to decode back (or leave as None)
            self.replay_data_raw = None  # Not available in encoded format
        else:
            raise ValueError(f"Unknown data format: expected 14 (preproc) or {self.config.token_mode} (encoded), got {format_info['n_cols']} columns")

    def _shard_to_mesh(self, x):
        """Shard array across devices along 'data' axis.

        H3: Helper method for multi-GPU data distribution.
        Reference: HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py

        Args:
            x: Array to shard (typically population data with leading dimension = n_perturbations)

        Returns:
            Sharded array distributed across devices
        """
        return jax.device_put(x, NamedSharding(self._mesh, P('data')))

    def _create_initial_sim_state(self) -> Tuple['LobState', jnp.ndarray]:
        """
        Load initial JaxLOB state and message history from LOBSTER data.

        Returns:
            (sim_state, msg_history)
        """
        import numpy as np
        import glob
        from lob.encoding import encode_msgs

        config = self.config

        # Determine data directory
        if hasattr(config, 'data_dir') and config.data_dir:
            data_dir = config.data_dir
        elif config.background_mode == 'historical_replay' and config.replay_data_path:
            data_dir = config.replay_data_path
        else:
            data_dir = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022"

        # Find data files
        orderbook_files = sorted(glob.glob(f"{data_dir}/*orderbook_10_proc.npy"))
        message_files = sorted(glob.glob(f"{data_dir}/*message_10_proc.npy"))

        if len(orderbook_files) == 0:
            raise FileNotFoundError(f"No orderbook files found in {data_dir}")

        # Select data file
        if config.background_mode == 'historical_replay' and hasattr(self, 'replay_data_date'):
            matching_files = [f for f in orderbook_files if self.replay_data_date in f]
            if len(matching_files) == 0:
                raise FileNotFoundError(f"No orderbook file for date {self.replay_data_date}")
            file_idx = orderbook_files.index(matching_files[0])
        elif hasattr(config, 'file_idx'):
            file_idx = config.file_idx % len(orderbook_files)
        else:
            file_idx = np.random.randint(0, len(orderbook_files))

        # Load data
        ob = np.load(orderbook_files[file_idx])
        msg = np.load(message_files[file_idx])

        # Log data format using standard utilities
        ob_format_info = detect_data_format(orderbook_files[file_idx], ob, self.config.token_mode)
        msg_format_info = detect_data_format(message_files[file_idx], msg, self.config.token_mode)
        log_data_format(ob_format_info, prefix="[INIT-STATE] Orderbook")
        log_data_format(msg_format_info, prefix="[INIT-STATE] Message")

        # Auto-detect data format: preproc (N, 14) vs encoded (N, 24)
        if msg_format_info['format_type'] == 'unknown':
            raise ValueError(f"Unknown data format: expected 14 (preproc) or {self.config.token_mode} (encoded), got {msg_format_info['n_cols']} columns")

        if msg_format_info['format_type'] == 'encoded':
            # Need to decode for JaxLOB warmup
            from lob.encoding import decode_msgs, Vocab
            v = Vocab(token_mode=self.config.token_mode)
            print(f"[INIT-STATE] Action: decoding {msg_format_info['n_cols']}-token format for JaxLOB warmup...")
            msg_decoded = np.array(decode_msgs(msg, v.ENCODING, token_mode=self.config.token_mode))
            msg_raw = msg_decoded
            msg_tokens = msg  # Already tokenized
        else:
            # Preproc format
            print(f"[INIT-STATE] Action: using raw data, will encode to {self.config.token_mode}-token format")
            msg_raw = msg
            msg_tokens = None  # Will encode below

        # Initialize L2 book
        init_l2_book = jnp.array(ob[0, 3:43], dtype=jnp.int32)
        sim_state = self.sim.reset(init_l2_book)

        # Replay warmup messages to initialize order book state
        n_init_background_msgs = getattr(self.config, 'n_warmup_msgs', 500)
        n_replay = min(n_init_background_msgs, len(msg_raw))
        replay_msgs_raw = msg_raw[:n_replay]
        replay_jaxlob = msgs_to_jnp(replay_msgs_raw)
        sim_state = self.sim.process_orders_array(sim_state, replay_jaxlob)

        # Encode messages as context
        # msg_seq_len from frozen_params determines expected context size
        msg_seq_len = self.lobs5_init.frozen_params.get('msg_seq_len', 500)
        expected_context_len = msg_seq_len * self.config.token_mode

        if n_replay > 0:
            if msg_tokens is not None:
                # Already have tokens from encoded format
                tokens = msg_tokens[:n_replay]
            else:
                # Need to encode from preproc format
                tokens = encode_msgs(replay_msgs_raw, self.encoder, token_mode=self.config.token_mode)
            msg_history = tokens.flatten()
            # Pad or truncate to expected size
            if len(msg_history) < expected_context_len:
                # Pad with zeros at the beginning
                msg_history = jnp.concatenate([
                    jnp.zeros(expected_context_len - len(msg_history), dtype=msg_history.dtype),
                    msg_history
                ])
            elif len(msg_history) > expected_context_len:
                # Keep most recent tokens
                msg_history = msg_history[-expected_context_len:]
        else:
            # No warmup - initialize with zeros
            msg_history = jnp.zeros(expected_context_len, dtype=jnp.int32)

        print(f"  n_init_background_msgs (warmup): {n_replay}")
        print(f"  context_size: {msg_history.shape} (expected: {expected_context_len})")

        return sim_state, msg_history

    def create_world_common_params(self):
        """Create CommonParams for World Model (frozen, no noise)."""
        CommonParams = _get_common_params()
        return CommonParams(
            noiser=self.noiser_cls,
            frozen_noiser_params=self.frozen_noiser_params,
            noiser_params=self.noiser_params,
            params=self.lobs5_init.params,
            es_tree_key=self.es_tree_key,
            frozen_params=self.lobs5_init.frozen_params,
            iterinfo=None,  # No noise for World Model
        )

    def create_policy_common_params(self, epoch: int, thread_id: int):
        """Create CommonParams for Policy (with ES perturbation)."""
        CommonParams = _get_common_params()
        iterinfo = (jnp.int32(epoch), jnp.int32(thread_id))
        return CommonParams(
            noiser=self.noiser_cls,
            frozen_noiser_params=self.frozen_noiser_params,
            noiser_params=self.noiser_params,
            params=self.lobs5_init.params,
            es_tree_key=self.es_tree_key,
            frozen_params=self.lobs5_init.frozen_params,
            iterinfo=iterinfo,
        )

    # ========================================================================
    # G1: AOT Compilation for eval_batch
    # Reference: HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py
    # Pattern: build_generate_thread returns pure function -> jit(vmap(...)).lower().compile()
    # ========================================================================

    def _build_eval_thread(self, in_shard_map: bool = False):
        """Build a pure eval function with no self references for AOT compilation.

        Args:
            in_shard_map: If True, the returned function will be called inside
                         shard_map and will apply pvary to scan carry values.

        Returns a function that accepts all dynamic parameters explicitly:
        - noiser_params: Updated each epoch
        - params: Model weights, updated each epoch
        - key: Random key for this thread
        - thread_id: Thread index for ES perturbation
        - epoch: Current epoch number
        - initial_sim_state: Starting LOB state
        - initial_msg_history: Starting message context

        Static parameters (captured in closure):
        - noiser_cls, frozen_noiser_params, es_tree_key
        - frozen_params (model config)
        - All simulation parameters (jaxlob_cfg, encoder, replay data)
        """
        # Capture static references (avoid self in JIT)
        noiser_cls = self.noiser_cls
        frozen_noiser_params = self.frozen_noiser_params
        es_tree_key = self.es_tree_key
        frozen_params = self.lobs5_init.frozen_params
        CommonParams = _get_common_params()

        # Capture simulation references
        sim = self.sim
        jaxlob_cfg = self.jaxlob_cfg
        encoder = self.encoder
        config = self.config
        replay_tokens = self.replay_tokens
        replay_data_raw = self.replay_data_raw

        # Import simulate_episode dependencies
        ES_PaddedLobPredModel = _get_es_model()

        # H2: Capture in_shard_map flag for closure
        _in_shard_map = in_shard_map

        def eval_thread(noiser_params, params, key, thread_id, epoch,
                        initial_sim_state, initial_msg_history):
            """Pure eval function for single thread."""
            # Create CommonParams with dynamic values
            world_common_params = CommonParams(
                noiser=noiser_cls,
                frozen_noiser_params=frozen_noiser_params,
                noiser_params=noiser_params,
                params=params,
                es_tree_key=es_tree_key,
                frozen_params=frozen_params,
                iterinfo=None,  # World Model has no ES noise
            )

            iterinfo = (jnp.int32(epoch), jnp.int32(thread_id))
            policy_common_params = CommonParams(
                noiser=noiser_cls,
                frozen_noiser_params=frozen_noiser_params,
                noiser_params=noiser_params,
                params=params,
                es_tree_key=es_tree_key,
                frozen_params=frozen_params,
                iterinfo=iterinfo,
            )

            # Call simulate_episode through self (still need this for the complex logic)
            # Note: This is a hybrid approach - we've extracted the CommonParams creation
            # but the simulate_episode call still uses self. Full AOT would require
            # inlining simulate_episode here, but that's a larger refactor.
            # H2: Pass in_shard_map flag to enable pvary for scan carry values
            return self.simulate_episode(
                key, world_common_params, policy_common_params,
                initial_sim_state, initial_msg_history,
                thread_id=thread_id,
                in_shard_map=_in_shard_map,
            )

        return eval_thread

    def _compile_eval_batch(self):
        """Pre-compile the vmapped eval function for reuse across epochs.

        This follows the HyperscaleES pattern:
        1. Build pure eval function (_build_eval_thread)
        2. Wrap with vmap for parallel threads
        3. JIT compile with proper in_axes

        H1: When multi-GPU is available, uses shard_map to distribute threads
        across devices. Each device runs n_perturbations/n_devices threads in parallel.

        H2: When using shard_map, passes in_shard_map=True to _build_eval_thread
        so that pvary is applied to scan carry values.

        Returns a JIT-compiled function that can be called directly.

        Note: Full AOT compilation (.lower().compile()) requires ShapeDtypeStruct
        examples for all inputs including the complex LobState pytree. This
        implementation uses standard JIT which will compile on first call and
        cache for subsequent calls.
        """
        # Check if multi-GPU is available
        n_devices = getattr(self, '_n_devices', 1)

        # H2: Build eval_thread with in_shard_map flag based on whether we're using multi-GPU
        use_shard_map = n_devices > 1 and hasattr(self, '_mesh')
        _eval_thread = self._build_eval_thread(in_shard_map=use_shard_map)

        if use_shard_map:
            # ====================================================================
            # H1: shard_map + vmap for multi-GPU distribution
            # Reference: HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py
            #
            # Strategy:
            # - shard_map distributes data across devices (outer layer)
            # - vmap handles threads per device (inner layer)
            # - Each device gets n_perturbations/n_devices threads
            #
            # H2: in_shard_map=True enables pvary for scan carry values
            # ====================================================================
            print(f"[H1] Using shard_map with {n_devices} devices")
            print(f"[H2] pvary enabled for scan carry values")

            # Inner vmap: vectorize over keys and thread_ids within each device
            # After sharding, each device sees (n_perturbations/n_devices,) shaped arrays
            vmapped_eval = jax.vmap(
                _eval_thread,
                in_axes=(None, None, 0, 0, None, None, None)
            )

            # Outer shard_map: distribute across devices
            # in_specs:
            #   - noiser_params: P() - replicated (shared across all devices)
            #   - params: P() - replicated
            #   - keys: P('data') - sharded along data axis
            #   - thread_ids: P('data') - sharded
            #   - epoch: P() - replicated
            #   - initial_sim_state: P() - replicated (broadcast)
            #   - initial_msg_history: P() - replicated (broadcast)
            # out_specs:
            #   - fitnesses: P('data') - sharded (gather results)
            #   - infos: P('data') - sharded (pytree, handled automatically)
            sharded_eval = shard_map(
                vmapped_eval,
                mesh=self._mesh,
                in_specs=(
                    P(),        # noiser_params: replicated
                    P(),        # params: replicated
                    P('data'),  # keys: sharded
                    P('data'),  # thread_ids: sharded
                    P(),        # epoch: replicated
                    P(),        # initial_sim_state: replicated
                    P(),        # initial_msg_history: replicated
                ),
                out_specs=(
                    P('data'),  # fitnesses: sharded
                    P('data'),  # infos: sharded (pytree)
                ),
                check_rep=False,  # H2: Disable VMA check due to complex scan carry types
            )

            # JIT compile with donated args for memory optimization
            compiled_eval = jax.jit(sharded_eval)

            print("[H1] Pre-compiled eval_batch function with shard_map")
            print(f"[H1]   Mesh: {self._mesh.axis_names}")
            print(f"[H1]   Devices: {n_devices}")
        else:
            # ====================================================================
            # Fallback to single-GPU vmap (original G1 implementation)
            # ====================================================================
            print("[H1] Using single-GPU vmap (no mesh or single device)")

            # Define in_axes for vmap:
            # - noiser_params: None (shared across threads)
            # - params: None (shared)
            # - key: 0 (different per thread)
            # - thread_id: 0 (different per thread)
            # - epoch: None (shared)
            # - initial_sim_state: None (shared, broadcast)
            # - initial_msg_history: None (shared, broadcast)

            vmapped_eval = jax.vmap(
                _eval_thread,
                in_axes=(None, None, 0, 0, None, None, None)
            )

            # JIT compile with donated args for memory optimization
            # Note: We don't donate noiser_params/params as they're needed for gradient update
            compiled_eval = jax.jit(vmapped_eval)

            print("[G1] Pre-compiled eval_batch function (single-GPU)")

        return compiled_eval

    # ========================================================================

    def simulate_episode(
        self,
        key: jnp.ndarray,
        world_common_params,
        policy_common_params,
        sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
        thread_id: int = -1,
        in_shard_map: bool = False,  # H2: Flag to indicate if called from within shard_map
    ) -> Tuple[float, Dict]:
        """
        Run a complete episode with step-by-step interleaved simulation.

        Args:
            in_shard_map: If True, applies jax.lax.pvary to scan carry values
                         to mark them as varying along the 'data' axis.
                         Required when this function is called inside shard_map.

        Returns:
            (fitness, info_dict)
        """
        config = self.config
        fp = self.lobs5_init.frozen_params
        jaxlob_cfg = self.jaxlob_cfg  # Capture for use in nested functions

        # ========================================================================
        # G5 + G2: Extract self references to avoid capturing entire self in JIT
        # This prevents JAX from potentially recompiling when self attributes change
        # Reference: HyperscaleES/llm_experiments/utils.py - build_generate_thread pattern
        # ========================================================================
        process_order_array = self.sim.process_order_array
        sim_obj = self.sim
        encoder = self.encoder
        replay_tokens = self.replay_tokens
        replay_data_raw = self.replay_data_raw
        n_replay_msgs = replay_tokens.shape[0] if replay_tokens is not None else 0
        # ========================================================================

        # ========================================================================
        # H2: Helper function to apply pvary when inside shard_map
        # Reference: https://docs.jax.dev/en/latest/notebooks/shard_map.html#scan-vmap
        # When using shard_map + scan, initial carry values must be marked as
        # "varying" along the sharded axis using jax.lax.pvary.
        # ========================================================================
        def maybe_pvary(x):
            """Apply pvary if inside shard_map context."""
            if in_shard_map:
                return jax.lax.pvary(x, ('data',))
            return x

        def maybe_pvary_tree(tree):
            """Apply pvary to all leaves of a pytree if inside shard_map."""
            if in_shard_map:
                return jax.tree.map(lambda x: jax.lax.pvary(x, ('data',)), tree)
            return tree
        # ========================================================================

        # Get ES model class
        ES_PaddedLobPredModel = _get_es_model()

        # Initialize hidden states
        hiddens_world = ES_PaddedLobPredModel.initialize_carry(
            batch_size=1,
            ssm_size=fp.get('ssm_size', 256),
            n_message_layers=fp.get('n_message_layers', 2),
            n_book_pre_layers=fp.get('n_book_pre_layers', 1),
            n_book_post_layers=fp.get('n_book_post_layers', 1),
            n_fused_layers=fp.get('n_fused_layers', 4),
            d_model=fp.get('d_model', 256),
            conj_sym=fp.get('conj_sym', True),
        )
        hiddens_policy = ES_PaddedLobPredModel.initialize_carry(
            batch_size=1,
            ssm_size=fp.get('ssm_size', 256),
            n_message_layers=fp.get('n_message_layers', 2),
            n_book_pre_layers=fp.get('n_book_pre_layers', 1),
            n_book_post_layers=fp.get('n_book_post_layers', 1),
            n_fused_layers=fp.get('n_fused_layers', 4),
            d_model=fp.get('d_model', 256),
            conj_sym=fp.get('conj_sym', True),
        )

        # Message length based on token mode
        msg_len = 22 if config.token_mode == 22 else 24
        msg_seq_len = fp.get('msg_seq_len', 500)
        context_len = msg_len * msg_seq_len

        # Order ID ranges
        POLICY_ORDER_ID_START = 1000000
        WORLD_ORDER_ID_START = 2000000

        # Clear trades from historical replay
        # Note: JaxLOB trades have 8 columns, not 6!
        sim_state = sim_state._replace(
            trades=(jnp.ones((sim_state.trades.shape[0], 8)) * -1).astype(jnp.int32)
        )

        init_mid_price = get_mid_price(jaxlob_cfg, sim_state, config.tick_size)

        # Initialize message history
        if initial_msg_history is not None:
            msg_history = initial_msg_history
        else:
            msg_history = jnp.zeros((context_len,), dtype=jnp.int32)

        book_depth = fp.get('book_depth', 500)
        book_feat = transform_L2_state_wrapper(jaxlob_cfg, sim_state, price_levels=book_depth, tick_size=config.tick_size, in_shard_map=in_shard_map)

        # ========================================================================
        # Hidden State Warmup Phase
        # S5/RNN models need to process the initial context to build meaningful
        # hidden states. Without this, hiddens are zeros and outputs are random.
        # ========================================================================
        if initial_msg_history is not None and len(initial_msg_history) >= msg_len:
            n_warmup_msgs = len(initial_msg_history) // msg_len

            def fix_ema_shape(hiddens):
                """Keep only last token's EMA state to maintain shape for scan."""
                msg_h, book_h, fused_h, ema_state = hiddens
                ema_val, ema_count = ema_state
                # Take last position to maintain shape: (batch, seq, d) -> (batch, 1, d)
                ema_val = ema_val[:, -1:, :]
                return (msg_h, book_h, fused_h, (ema_val, ema_count))

            def warmup_step(carry, msg_idx):
                """Process one message through the model to update hidden state."""
                hiddens_w, hiddens_p = carry
                # Use jax.lax.dynamic_slice for JAX-traceable dynamic indexing
                start_idx = msg_idx * msg_len
                msg_tokens = jax.lax.dynamic_slice(
                    initial_msg_history, (start_idx,), (msg_len,)
                )

                # Update world model hiddens
                hiddens_w, _ = ES_PaddedLobPredModel._forward_step(
                    world_common_params, hiddens_w, msg_tokens, book_feat[None, :]
                )
                hiddens_w = fix_ema_shape(hiddens_w)

                # Update policy model hiddens
                hiddens_p, _ = ES_PaddedLobPredModel._forward_step(
                    policy_common_params, hiddens_p, msg_tokens, book_feat[None, :]
                )
                hiddens_p = fix_ema_shape(hiddens_p)

                return (hiddens_w, hiddens_p), None

            # H2: Apply pvary to initial carry values when inside shard_map
            warmup_init = (
                maybe_pvary_tree(hiddens_world),
                maybe_pvary_tree(hiddens_policy),
            )
            (hiddens_world, hiddens_policy), _ = jax.lax.scan(
                warmup_step,
                warmup_init,
                jnp.arange(n_warmup_msgs),
                length=n_warmup_msgs,
            )
        # ========================================================================

        task_size = jnp.int32(config.task_size)

        # Pre-compute field masks OUTSIDE step_fn to avoid JAX tracer leak
        # These masks constrain each token position to valid vocabulary ranges
        vocab_size = fp.get('d_output', 2112)  # Default 2112 for 24-token mode
        field_masks = get_field_masks_24(vocab_size=vocab_size)

        def step_fn(carry, step_idx):
            """Single step: Background messages -> Policy action."""
            (key, msg_history, hiddens_world, hiddens_policy,
             sim_state, book_feat, world_oid_offset, quant_executed) = carry

            key, key_world, key_policy = jax.random.split(key, 3)

            # Background message generation
            def historical_replay_step(wcarry, bg_msg_idx):
                """Load pre-encoded messages from historical data."""
                key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr = wcarry

                replayed_msg_tokens = replay_tokens[replay_ptr]
                replayed_msg_raw = replay_data_raw[replay_ptr]

                sim_msg = decoded_msg_to_jaxlob_format(replayed_msg_raw)
                bg_order_id = WORLD_ORDER_ID_START + oid_offset
                sim_msg = sim_msg.at[4].set(bg_order_id)
                sim_msg = sim_msg.at[5].set(-2000)

                sim_st = process_order_array(sim_st, sim_msg)
                book_f = transform_L2_state_wrapper(jaxlob_cfg, sim_st, price_levels=book_depth, tick_size=config.tick_size, in_shard_map=in_shard_map)
                msg_hist = jnp.concatenate([msg_hist[msg_len:], replayed_msg_tokens])

                new_replay_ptr = replay_ptr + 1
                # Use pre-extracted n_replay_msgs instead of self.replay_tokens.shape[0]
                new_replay_ptr = jnp.where(new_replay_ptr >= n_replay_msgs, jnp.int32(500), new_replay_ptr)
                oid_offset = oid_offset + 1

                return (key, msg_hist, hidden, sim_st, book_f, oid_offset, new_replay_ptr), replayed_msg_tokens

            def world_model_step(wcarry, world_msg_idx):
                """Generate message autoregressively using world model."""
                key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr = wcarry

                # Autoregressive token sampling
                def sample_one_token(token_carry, _):
                    key_t, msg_hist_t, hidden_t = token_carry
                    key_t, sample_key_t = jax.random.split(key_t)

                    hidden_t, log_probs_t = ES_PaddedLobPredModel._forward_step(
                        world_common_params, hidden_t, msg_hist_t[-msg_len:], book_f[None, :]
                    )
                    hidden_t = jax.tree.map(lambda h: h[:, -1:, :], hidden_t)

                    log_probs_t = jnp.nan_to_num(log_probs_t, nan=-1e9, posinf=1e9, neginf=-1e9)
                    next_token = jax.random.categorical(sample_key_t, log_probs_t[-1])

                    msg_hist_t = jnp.concatenate([msg_hist_t[1:], jnp.array([next_token])])

                    return (key_t, msg_hist_t, hidden_t), next_token

                key, sample_key = jax.random.split(key)
                # H2: Apply pvary to initial carry values when inside shard_map
                world_token_init = (
                    maybe_pvary(sample_key),
                    maybe_pvary(msg_hist),
                    maybe_pvary_tree(hidden),
                )
                (key, msg_hist, hidden), world_msg = jax.lax.scan(
                    sample_one_token,
                    world_token_init,
                    None,
                    length=msg_len,
                )

                # Convert to JaxLOB format
                mid_price = get_mid_price(jaxlob_cfg, sim_st, config.tick_size)
                world_order_id = WORLD_ORDER_ID_START + oid_offset
                sim_msg, _ = get_sim_msg_es(
                    world_msg, sim_obj, sim_st, mid_price, world_order_id, config.tick_size, encoder,
                    trader_id=-2000, token_mode=config.token_mode
                )

                sim_st = process_order_array(sim_st, sim_msg)
                book_f = transform_L2_state_wrapper(jaxlob_cfg, sim_st, price_levels=book_depth, tick_size=config.tick_size, in_shard_map=in_shard_map)
                msg_hist = jnp.concatenate([msg_hist[msg_len:], world_msg])
                oid_offset = oid_offset + 1

                return (key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr), world_msg

            # Select background generation function
            n_warmup_cfg = getattr(config, 'n_warmup_msgs', 500)
            if config.background_mode == 'historical_replay':
                step_fn_background = historical_replay_step
                replay_ptr_init = jnp.int32(n_warmup_cfg + step_idx * config.background_msgs_per_step)
            else:
                step_fn_background = world_model_step
                replay_ptr_init = jnp.int32(0)

            # Generate background messages
            # H2: Apply pvary to initial carry values when inside shard_map
            background_scan_init = (
                maybe_pvary(key_world),
                maybe_pvary(msg_history),
                maybe_pvary_tree(hiddens_world),
                maybe_pvary_tree(sim_state),
                maybe_pvary(book_feat),
                maybe_pvary(world_oid_offset),
                maybe_pvary(replay_ptr_init),
            )
            (key_world, msg_history, hiddens_world, sim_state, book_feat,
             world_oid_offset, _), _ = jax.lax.scan(
                step_fn_background,
                background_scan_init,
                jnp.arange(config.background_msgs_per_step),
                length=config.background_msgs_per_step,
            )

            # Policy generates action with field-aware constrained decoding
            # field_masks is pre-computed OUTSIDE step_fn to avoid tracer leak
            # Temperature controls sampling sharpness: T<1 sharpens, T>1 flattens
            # NOTE: T=0.1 tested but made distribution worse (amplified wrong peak preferences)
            # Using T=1.0 (standard sampling) as default
            temperature = getattr(config, 'temperature', 1.0)

            def sample_policy_token(token_carry, token_pos):
                """Sample next token with field-aware masking and temperature.

                Args:
                    token_carry: (key, msg_history, hiddens)
                    token_pos: Current position in 24-token message (0-23)

                The field mask ensures tokens are only sampled from valid ranges:
                - pos 0 (event_type): tokens 1004-1007
                - pos 1 (direction): tokens 2110-2111
                - pos 2 (price_sign): tokens 2108-2109
                - etc.

                Temperature < 1.0 sharpens the distribution (more deterministic)
                Temperature = 1.0 is standard sampling
                Temperature > 1.0 flattens the distribution (more random)
                """
                key_p, msg_hist_p, hidden_p = token_carry
                key_p, sample_key_p = jax.random.split(key_p)

                hidden_p, log_probs_p = ES_PaddedLobPredModel._forward_step(
                    policy_common_params, hidden_p, msg_hist_p[-msg_len:], book_feat[None, :]
                )
                hidden_p = jax.tree.map(lambda h: h[:, -1:, :], hidden_p)

                log_probs_p = jnp.nan_to_num(log_probs_p, nan=-1e9, posinf=1e9, neginf=-1e9)

                # Apply field-aware mask: add -inf to invalid token positions
                field_mask = field_masks[token_pos]
                masked_log_probs = log_probs_p[-1] + field_mask

                # Apply temperature scaling to sharpen/flatten the distribution
                # Temperature < 1 makes high-prob tokens more likely (sharper)
                # Temperature > 1 makes distribution more uniform (flatter)
                scaled_log_probs = masked_log_probs / temperature

                next_token_p = jax.random.categorical(sample_key_p, scaled_log_probs)

                msg_hist_p = jnp.concatenate([msg_hist_p[1:], jnp.array([next_token_p])])

                return (key_p, msg_hist_p, hidden_p), next_token_p

            key_policy, sample_key = jax.random.split(key_policy)
            # H2: Apply pvary to initial carry values when inside shard_map
            policy_token_init = (
                maybe_pvary(sample_key),
                maybe_pvary(msg_history),
                maybe_pvary_tree(hiddens_policy),
            )
            # Pass token positions (0-23) as xs to enable field-aware masking
            (key_policy, msg_history, hiddens_policy), policy_msg = jax.lax.scan(
                sample_policy_token,
                policy_token_init,
                jnp.arange(msg_len, dtype=jnp.int32),
                length=msg_len,
            )

            # Convert to JaxLOB format
            mid_price = get_mid_price(jaxlob_cfg, sim_state, config.tick_size)
            policy_order_id = POLICY_ORDER_ID_START + step_idx
            sim_msg, msg_decoded = get_sim_msg_es(
                policy_msg, sim_obj, sim_state, mid_price, policy_order_id, config.tick_size, encoder,
                trader_id=-1000, token_mode=config.token_mode
            )

            # Cancel previous unfilled policy order
            prev_policy_oid = POLICY_ORDER_ID_START + step_idx - 1
            is_prev_order_in_asks = sim_state.asks[:, 2] == prev_policy_oid
            sim_state = sim_state._replace(
                asks=jnp.where(is_prev_order_in_asks[:, None], jnp.array([0, 0, -1, -1, 0, 0]), sim_state.asks)
            )
            is_prev_order_in_bids = sim_state.bids[:, 2] == prev_policy_oid
            sim_state = sim_state._replace(
                bids=jnp.where(is_prev_order_in_bids[:, None], jnp.array([0, 0, -1, -1, 0, 0]), sim_state.bids)
            )

            # Truncate quantity to remaining task
            quant_remaining = task_size - quant_executed
            original_qty = sim_msg[2]
            truncated_qty = jnp.minimum(original_qty, jnp.maximum(quant_remaining, 0))
            sim_msg = sim_msg.at[2].set(truncated_qty)

            # Process order
            sim_state = process_order_array(sim_state, sim_msg)

            # Track execution
            trades = sim_state.trades
            is_new_trade = (trades[:, 0] != -1)
            is_policy_in_trade = ((trades[:, 2] == policy_order_id) | (trades[:, 3] == policy_order_id)) & is_new_trade
            step_executed = jnp.sum(jnp.where(is_policy_in_trade, jnp.abs(trades[:, 1]), 0))
            quant_executed = quant_executed + step_executed

            # Update state
            book_feat = transform_L2_state_wrapper(jaxlob_cfg, sim_state, price_levels=book_depth, tick_size=config.tick_size, in_shard_map=in_shard_map)
            msg_history = jnp.concatenate([msg_history[msg_len:], policy_msg])

            # Return policy_msg for order analysis (shape: (msg_len,))
            return (key, msg_history, hiddens_world, hiddens_policy, sim_state,
                    book_feat, world_oid_offset, quant_executed), policy_msg

        # Run episode
        # H2: Apply pvary to initial carry values when inside shard_map
        # This marks arrays as "varying" along the sharded axis to satisfy scan's type requirements
        main_scan_init = (
            maybe_pvary(key),
            maybe_pvary(msg_history),
            maybe_pvary_tree(hiddens_world),
            maybe_pvary_tree(hiddens_policy),
            maybe_pvary_tree(sim_state),
            maybe_pvary(book_feat),
            maybe_pvary(jnp.int32(0)),
            maybe_pvary(jnp.int32(0)),
        )
        # Capture policy_msgs_all for order analysis (shape: (n_steps, msg_len))
        (_, _, _, _, final_state, _, _, final_quant_executed), policy_msgs_all = jax.lax.scan(
            step_fn,
            main_scan_init,
            jnp.arange(config.n_steps),
            length=config.n_steps,
        )

        # Compute fitness (PnL)
        trades = final_state.trades
        valid_trades_mask = trades[:, 0] != -1

        passive_ids = trades[:, 2]
        aggr_ids = trades[:, 3]
        is_policy_passive = (passive_ids >= POLICY_ORDER_ID_START) & (passive_ids < WORLD_ORDER_ID_START) & valid_trades_mask
        is_policy_aggr = (aggr_ids >= POLICY_ORDER_ID_START) & (aggr_ids < WORLD_ORDER_ID_START) & valid_trades_mask
        is_policy_trade = is_policy_passive | is_policy_aggr

        is_sell_task = (config.task == 'sell')
        if is_sell_task:
            sell_revenue = jnp.sum(jnp.where(is_policy_trade, trades[:, 0] * jnp.abs(trades[:, 1]), 0))
            sell_quantity = jnp.sum(jnp.where(is_policy_trade, jnp.abs(trades[:, 1]), 0))
            pnl_raw = sell_revenue - init_mid_price * sell_quantity
            agent_quantity = sell_quantity
        else:
            buy_cost = jnp.sum(jnp.where(is_policy_trade, trades[:, 0] * jnp.abs(trades[:, 1]), 0))
            buy_quantity = jnp.sum(jnp.where(is_policy_trade, jnp.abs(trades[:, 1]), 0))
            pnl_raw = init_mid_price * buy_quantity - buy_cost
            agent_quantity = buy_quantity

        # Normalize PnL to -1 to 1 range using tanh
        # pnl_normalized = "number of ticks improvement for full task execution"
        # e.g., if you execute all task_size shares 1 tick better than mid, pnl_normalized = 1.0
        normalization_scale = config.task_size * config.tick_size
        pnl_normalized = pnl_raw / jnp.maximum(normalization_scale, 1.0)
        pnl = jnp.tanh(pnl_normalized)  # squash to -1 to 1, 0 = executed at mid price

        # Completion penalty (disabled - penalty was too large relative to PnL signal)
        # shortfall = jnp.maximum(config.task_size - agent_quantity, 0)
        # completion_penalty = -shortfall * init_mid_price / 1e6 * 0.1
        completion_penalty = jnp.float32(0.0)

        total_trades = jnp.sum(valid_trades_mask)
        agent_trades = jnp.sum(is_policy_trade)

        # Final fitness
        base_fitness = jnp.where(
            agent_quantity > 0,
            pnl + completion_penalty,
            jnp.where(total_trades > 0, -0.05, -0.1)
        )
        fitness = jnp.where(jnp.isfinite(base_fitness), base_fitness, 0.0)

        info = {
            'fitness': fitness,
            'pnl': pnl,                        # normalized to -1 to 1 (tanh)
            'pnl_raw': pnl_raw,                # raw value in cents
            'pnl_normalized': pnl_normalized,  # before tanh (in "ticks")
            'agent_quantity': agent_quantity,
            'agent_trades': agent_trades,
            'total_trades': total_trades,
            'completion_penalty': completion_penalty,
            'init_mid_price': init_mid_price,
            'policy_msgs': policy_msgs_all,    # shape: (n_steps, msg_len) for order analysis
        }

        return fitness, info

    def eval_single_thread(
        self,
        key: jnp.ndarray,
        thread_id: int,
        epoch: int,
        initial_sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """Evaluate one perturbed policy on a single episode."""
        world_common_params = self.create_world_common_params()
        policy_common_params = self.create_policy_common_params(epoch, thread_id)

        return self.simulate_episode(
            key, world_common_params, policy_common_params, initial_sim_state, initial_msg_history,
            thread_id=thread_id
        )

    def train_epoch(
        self,
        key: jnp.ndarray,
        epoch: int,
        initial_sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
    ) -> Tuple[float, jnp.ndarray, Dict]:
        """Run one training epoch."""
        n_perturbations = self.config.n_perturbations
        n_devices = getattr(self, '_n_devices', 1)

        # ========================================================================
        # H1: Validate n_perturbations divisibility for shard_map
        # When using multi-GPU, n_perturbations must be evenly divisible by n_devices
        # so each device gets the same number of perturbations to evaluate.
        # ========================================================================
        if n_devices > 1:
            assert n_perturbations % n_devices == 0, \
                f"[H1 ERROR] n_perturbations ({n_perturbations}) must be divisible by n_devices ({n_devices}). " \
                f"Consider using n_perturbations={n_devices * (n_perturbations // n_devices)} or n_perturbations={n_devices * ((n_perturbations // n_devices) + 1)}"

        # Generate keys for all perturbations
        keys = jax.random.split(key, n_perturbations)
        thread_ids = jnp.arange(n_perturbations)

        # ========================================================================
        # G1 + H1: Use pre-compiled eval_batch function
        # The function was compiled in __init__ and is reused here.
        # Arguments: (noiser_params, params, keys, thread_ids, epoch, sim_state, msg_history)
        #
        # H2: For multi-GPU with shard_map, shard inputs correctly across devices
        # - params/noiser_params: P() replicated to all devices
        # - keys/thread_ids: P('data') sharded across devices for parallel eval
        # ========================================================================
        n_devices = getattr(self, '_n_devices', 1)

        if n_devices > 1 and hasattr(self, '_mesh'):
            # Replicate params and noiser_params to all devices
            noiser_params_rep = jax.device_put(
                self.noiser_params,
                NamedSharding(self._mesh, P())
            )
            params_rep = jax.device_put(
                self.lobs5_init.params,
                NamedSharding(self._mesh, P())
            )
            # Shard keys and thread_ids across devices for parallel evaluation
            # Each device gets (n_perturbations/n_devices) threads to evaluate
            keys = jax.device_put(
                keys,
                NamedSharding(self._mesh, P('data'))
            )
            thread_ids = jax.device_put(
                thread_ids,
                NamedSharding(self._mesh, P('data'))
            )
        else:
            # Single GPU: use params as-is
            noiser_params_rep = self.noiser_params
            params_rep = self.lobs5_init.params

        fitnesses, infos = self._compiled_eval_batch(
            noiser_params_rep,
            params_rep,
            keys,
            thread_ids,
            jnp.int32(epoch),
            initial_sim_state,
            initial_msg_history,
        )

        # ES gradient update
        iterinfos = (
            jnp.full(n_perturbations, epoch, dtype=jnp.int32),
            thread_ids
        )

        # ========================================================================
        # H2: Use replicated params for gradient updates in multi-GPU mode
        # The fitnesses returned from shard_map are sharded, so params must
        # also be replicated to avoid device mismatch in do_updates
        # ========================================================================
        normalized_fitnesses = self.noiser_cls.convert_fitnesses(
            self.frozen_noiser_params, noiser_params_rep, fitnesses
        )

        noiser_params_updated, updated_params = self.noiser_cls.do_updates(
            self.frozen_noiser_params,
            noiser_params_rep,
            params_rep,
            self.es_tree_key,
            normalized_fitnesses,
            iterinfos,
            self.lobs5_init.es_map,
        )

        # Extract updated params back to single device for storage
        if n_devices > 1 and hasattr(self, '_mesh'):
            # Get first shard from replicated params
            self.noiser_params = jax.tree.map(
                lambda x: jax.device_put(x, jax.devices()[0]),
                noiser_params_updated
            )
            self.lobs5_init.params = jax.tree.map(
                lambda x: jax.device_put(x, jax.devices()[0]),
                updated_params
            )
        else:
            self.noiser_params = noiser_params_updated
            self.lobs5_init.params = updated_params

        aggregated_info = {k: jnp.mean(v) for k, v in infos.items()}

        return jnp.mean(fitnesses), fitnesses, aggregated_info

    def train(self, n_epochs: Optional[int] = None, resume_from: Optional[str] = None):
        """Run full training loop with automatic checkpointing.

        Args:
            n_epochs: Number of epochs to train (default: config.n_epochs)
            resume_from: Path to checkpoint directory to resume from
        """
        import os
        print("[TRAIN] Starting training loop")

        n_epochs = n_epochs or self.config.n_epochs
        key = jax.random.PRNGKey(self.config.seed)

        # Checkpointing configuration
        checkpoint_dir = getattr(self.config, 'checkpoint_dir', './es_checkpoints')
        checkpoint_every = getattr(self.config, 'checkpoint_every', 50)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume from checkpoint if specified
        start_epoch = 0
        best_fitness = -float('inf')
        if resume_from:
            try:
                self.load_checkpoint(resume_from)
                # Load training state
                import pickle
                state_path = os.path.join(resume_from, 'training_state.pkl')
                if os.path.exists(state_path):
                    with open(state_path, 'rb') as f:
                        state = pickle.load(f)
                    start_epoch = state.get('epoch', 0) + 1
                    best_fitness = state.get('best_fitness', -float('inf'))
                    key = jax.random.PRNGKey(self.config.seed)
                    # Fast-forward the key
                    for _ in range(start_epoch):
                        key, _ = jax.random.split(key)
                print(f"[TRAIN] Resumed from epoch {start_epoch}, best_fitness={best_fitness:.4f}")
            except Exception as e:
                print(f"[TRAIN] Warning: Could not resume from {resume_from}: {e}")
                print("[TRAIN] Starting fresh training")

        # Initialize W&B
        wandb_run = None
        if hasattr(self.config, 'wandb_project') and self.config.wandb_project:
            import wandb
            wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"es_jaxlob_n{self.config.n_perturbations}_s{self.config.seed}",
                config={
                    'n_perturbations': self.config.n_perturbations,
                    'n_steps': self.config.n_steps,
                    'noiser': self.config.noiser,
                    'sigma': self.config.sigma,
                    'lr': self.config.lr,
                    'lora_rank': self.config.lora_rank,
                    'checkpoint': self.config.lobs5_checkpoint,
                    'background_mode': self.config.background_mode,
                },
                resume='allow' if resume_from else None,
            )
            print(f"[TRAIN] W&B initialized: {wandb_run.url}")

        # Get initial state
        initial_sim_state, initial_msg_history = self._create_initial_sim_state()

        # Training loop
        for epoch in tqdm(range(start_epoch, n_epochs), desc='ES Training', initial=start_epoch, total=n_epochs):
            key, epoch_key = jax.random.split(key)

            mean_fitness, fitnesses, epoch_info = self.train_epoch(
                epoch_key, epoch, initial_sim_state, initial_msg_history
            )

            # Track best model
            is_best = mean_fitness > best_fitness
            if is_best:
                best_fitness = mean_fitness
                # Save best model
                best_path = os.path.join(checkpoint_dir, 'best')
                self.save_checkpoint(best_path)
                self._save_training_state(best_path, epoch, best_fitness)
                print(f"[TRAIN] New best model saved: fitness={best_fitness:.4f}")

            # Periodic checkpointing
            if (epoch + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}')
                self.save_checkpoint(ckpt_path)
                self._save_training_state(ckpt_path, epoch, best_fitness)
                # Also save as 'latest' for easy resumption
                latest_path = os.path.join(checkpoint_dir, 'latest')
                self.save_checkpoint(latest_path)
                self._save_training_state(latest_path, epoch, best_fitness)

            # Log to W&B
            if wandb_run:
                fitness_std = float(jnp.std(fitnesses))
                fitness_max = float(jnp.max(fitnesses))
                fitness_min = float(jnp.min(fitnesses))

                wandb_run.log({
                    'epoch': epoch,
                    'fitness/mean': float(mean_fitness),
                    'fitness/best_ever': float(best_fitness),
                    'fitness/std': fitness_std,
                    'fitness/max': fitness_max,
                    'fitness/min': fitness_min,
                    'pnl/mean': float(epoch_info['pnl']),
                    'execution/agent_quantity': float(epoch_info['agent_quantity']),
                    'execution/agent_trades': float(epoch_info['agent_trades']),
                    'execution/total_trades': float(epoch_info['total_trades']),
                })

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: mean={mean_fitness:.4f}, best={best_fitness:.4f}, std={jnp.std(fitnesses):.4f}")

        # Save final checkpoint
        final_path = os.path.join(checkpoint_dir, 'final')
        self.save_checkpoint(final_path)
        self._save_training_state(final_path, n_epochs - 1, best_fitness)
        print(f"[TRAIN] Final checkpoint saved to {final_path}")

        if wandb_run:
            wandb_run.finish()

        return self.lobs5_init.params

    def _save_training_state(self, path: str, epoch: int, best_fitness: float):
        """Save training state for resumption."""
        import os
        import pickle
        os.makedirs(path, exist_ok=True)
        state = {
            'epoch': epoch,
            'best_fitness': best_fitness,
        }
        with open(os.path.join(path, 'training_state.pkl'), 'wb') as f:
            pickle.dump(state, f)

    def save_checkpoint(self, path: str):
        """Save current policy params to checkpoint."""
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        checkpoint = {
            'params': self.lobs5_init.params,
            'frozen_params': self.lobs5_init.frozen_params,
            'noiser_params': self.noiser_params,
            'config': vars(self.config),
        }

        with open(os.path.join(path, 'es_checkpoint.pkl'), 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load policy params from checkpoint."""
        import os
        import pickle

        with open(os.path.join(path, 'es_checkpoint.pkl'), 'rb') as f:
            checkpoint = pickle.load(f)

        self.lobs5_init.params = checkpoint['params']
        self.noiser_params = checkpoint['noiser_params']

        print(f"Checkpoint loaded from {path}")


def es_train(config):
    """Main entry point for ES training."""
    trainer = ESTrainer(config)
    return trainer.train()


if __name__ == '__main__':
    parser = create_es_config()
    args = parser.parse_args()
    es_train(args)
