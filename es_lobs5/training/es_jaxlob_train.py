"""
ES Training with JaxLOB (Pure ES, Step-by-Step Interleaved).

This module implements Evolution Strategies training for trading policies
using JaxLOB as the execution environment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    DATA INITIALIZATION FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reference: lob/inference_no_errcorr.py (originally designed for sliding window)

IMPORTANT: The reference implementation was designed for sliding window mode
(window_size=500 → predict 1 message). However, LOBS5 is NOW trained as
FULL AUTOREGRESSIVE (no sliding window), so the data loading needs adaptation.

[1] Load Historical Data (LOBSTER preproc format)
    ┌─────────────────────────────────────────────┐
    │ Files:                                      │
    │   - orderbook_10_proc.npy  (N+1, 43)        │
    │     [0]: mid_diff (ticks)                   │
    │     [1]: time_s                             │
    │     [2]: time_ns                            │
    │     [3:43]: L2 book (40 values)             │
    │              [ask_p0, ask_q0, bid_p0, bid_q0,│
    │               ask_p1, ask_q1, bid_p1, bid_q1,│
    │               ...]                          │
    │                                             │
    │   - message_10_proc.npy    (N, 14)          │
    │     Decoded message fields (NOT 24 tokens!) │
    └─────────────────────────────────────────────┘

[2] Initialize Order Book
    ┌─────────────────────────────────────────────┐
    │ init_l2_book = book[0, 3:43]  (40 values)   │
    │ sim_state = sim.reset(init_l2_book)         │
    │                                             │
    │ Result: JaxLOB with 10 price levels         │
    └─────────────────────────────────────────────┘

[3] Replay Historical Messages (e.g., 500 messages)
    ┌─────────────────────────────────────────────┐
    │ messages[0:500] (14-column decoded)         │
    │   → encode to 24 tokens                     │
    │   → convert to JaxLOB format (8 values)     │
    │   → replay in simulator                     │
    │                                             │
    │ for msg in messages[0:500]:                 │
    │     tokens = encode_msg(msg)  # 14 → 24     │
    │     sim_msg = to_jaxlob(tokens)  # 24 → 8   │
    │     sim_state = sim.process_order_array(    │
    │                     sim_state, sim_msg)     │
    └─────────────────────────────────────────────┘

[4] Prepare Context for Generation (FULL AUTOREGRESSIVE)
    ┌─────────────────────────────────────────────┐
    │ msg_history = ALL 500 messages (11000 tok)  │
    │             = full context from replay      │
    │                                             │
    │ hidden_state = RNN state (carries history)  │
    │                                             │
    │ book_feat = extract_book_features(sim_state)│
    │           = current L2 state (503 values)   │
    └─────────────────────────────────────────────┘
    NOTE: Each forward pass uses msg_history[-msg_len:] (last 1 message)
          but hidden_state carries accumulated history from all 500 messages

[5] Generation Phase (Step 501+)
    ┌─────────────────────────────────────────────┐
    │ World Model + Policy start generating       │
    │   - Use FULL msg_history as RNN context     │
    │   - Forward pass: last 1 msg + hidden state │
    │   - Generate new messages autoregressively  │
    └─────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture (Step-by-Step):
    For each step t:
        1. World Model (frozen) generates K background market messages
        2. JaxLOB processes world_msgs → updates order book
        3. Policy (ES perturbed) **observes** updated book state
        4. Policy generates 1 trading action
        5. JaxLOB processes policy_msg → updates state

    Repeat T steps → final_state.total_revenue = Fitness

Key Features:
- Both World Model and Policy initialized from same LOBS5 checkpoint
- World Model stays frozen (iterinfo=None), Policy trained with EGGROLL
- Policy can observe market changes before making decisions
- Fitness = PnL (profit/loss based on execution quality)
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from functools import partial
import argparse
from tqdm import tqdm
import time
from typing import Tuple, Optional, NamedTuple, Dict, Any

# Import centralized utilities
from ..utils.import_utils import get_all_noisers
from ..utils import load_checkpoint_for_es
from ..models import ES_PaddedLobPredModel
from ..models.common import CommonParams, simple_es_tree_key
from .fitness import compute_pnl_fitness

# Lazy imports for JaxLOB
OrderBook = None
LobState = None
Message_Tokenizer = None
encoding = None
get_best_bid_and_ask = None  # From JaxOrderBookArrays

all_noisers = get_all_noisers()

__all__ = ['ESJaxLOBTrainer', 'create_es_jaxlob_config', 'es_jaxlob_train']


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


class EpisodeState(NamedTuple):
    """State for a single episode simulation."""
    key: jnp.ndarray
    msg_history: jnp.ndarray  # (context_len,) int32 - recent message tokens
    hidden_world: Tuple       # World Model hidden state
    hidden_policy: Tuple      # Policy hidden state
    sim_state: Any            # JaxLOB LobState
    book_feat: jnp.ndarray    # (d_book,) current book features
    step: int


def create_es_jaxlob_config():
    """Create argument parser for ES JaxLOB training configuration."""
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
    parser.add_argument('--n_threads', type=int, default=128, help='Population size')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--n_steps', type=int, default=100, help='Steps per episode')
    parser.add_argument('--world_msgs_per_step', type=int, default=10,
                        help='Background messages per step')

    # Encoder (optional - created from Vocab if not provided)
    parser.add_argument('--encoder_path', type=str, default=None,
                        help='Path to token encoder pickle file (created from Vocab if not provided)')

    # Initial book state
    parser.add_argument('--init_book_path', type=str, default=None,
                        help='Path to initial book state (random if None)')

    # Execution task
    parser.add_argument('--task', type=str, default='sell',
                        choices=['sell', 'buy'])
    parser.add_argument('--task_size', type=int, default=500,
                        help='Shares to execute')
    parser.add_argument('--tick_size', type=int, default=100,
                        help='Tick size in cents')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./es_jaxlob_checkpoints')
    parser.add_argument('--token_mode', type=int, default=22, choices=[22, 24],
                        help='Token mode: 22 (single token size) or 24 (base-100 size)')

    # Background model configuration
    parser.add_argument('--background_mode', type=str, default='world_model',
                        choices=['world_model', 'historical_replay'],
                        help='Background message generation mode: world_model (autoregressive) or historical_replay (from data)')
    parser.add_argument('--replay_data_path', type=str, default=None,
                        help='Path to historical data directory for replay mode (e.g., /path/to/GOOG/2016/)')

    # W&B logging
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/username')

    return parser


# Helper function to convert decoded messages to JaxLOB format
# Reference: lob/inference_no_errcorr.py:112-131 (msg_to_jnp, msgs_to_jnp)
# Copied here to avoid import issues with inference_no_errcorr.py
@jax.jit
def decoded_msg_to_jaxlob_format(msg_decoded: jax.Array) -> jax.Array:
    """
    Convert 14-column decoded message to 8-column JaxLOB format.

    Reference: lob/inference_no_errcorr.py:msg_to_jnp() (lines 112-129)

    Args:
        msg_decoded: (14,) decoded message
                     [order_id, event_type, direction, price_abs, price_rel, size,
                      delta_t_s, delta_t_ns, time_s, time_ns, ...]

    Returns:
        (8,) JaxLOB message [type, side, qty, price, trade_id, order_id, time_s, time_ns]
    """
    # Column indices
    ORDER_ID_i = 0
    EVENT_TYPE_i = 1
    DIRECTION_i = 2
    PRICE_ABS_i = 3
    SIZE_i = 5
    TIMEs_i = 8
    TIMEns_i = 9

    return jnp.array([
        msg_decoded[EVENT_TYPE_i],
        (msg_decoded[DIRECTION_i] * 2) - 1,  # 0/1 → -1/1
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
    trader_id: int = -88,  # FIX: Allow specifying trader_id (policy=-1000, world=-2000)
    token_mode: int = 22,  # Token mode for decoding (22 or 24)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert predicted message tokens to JaxLOB format.

    Simplified version of inference_no_errcorr.get_sim_msg for ES training.

    ============================================================
    FAULT TOLERANCE DESIGN (ES training without action/state space)
    ============================================================

    In ES training, there's no action space validation layer:
    - Traditional RL: action ∈ [-1, 1] → env.step() validates
    - ES: policy outputs token logits → direct execution

    Potential failure points:
    1. Token sampling: categorical() can sample any token 0..vocab_size
    2. Decoding: tokens → message fields may produce invalid values
    3. JaxLOB execution: invalid messages can corrupt orderbook state

    Fault tolerance strategy (JAX-compatible, no try/except):
    - Clamp decoded values to valid ranges
    - Use jnp.where() for conditional fallback
    - Invalid message → NOOP (event_type=0, qty=0)

    Why we DON'T use try/except:
    - JAX traces functions for JIT compilation
    - Python exceptions break tracing
    - Must use jnp.where() for conditional logic
    ============================================================

    Args:
        pred_msg_tokens: (24,) int32 - predicted message tokens
        sim: OrderBook instance
        sim_state: Current LobState
        mid_price: Current mid price
        order_id: Order ID to assign
        tick_size: Tick size
        encoder: Token encoder

    Returns:
        (sim_msg, msg_decoded)
        - sim_msg: (8,) int32 JaxLOB message format
        - msg_decoded: (14,) decoded message fields
    """
    _lazy_import_jaxlob()

    # Decode tokens to message fields
    msg_decoded = encoding.decode_msg(pred_msg_tokens, encoder, token_mode=token_mode)

    # Extract fields
    event_type = msg_decoded[1]  # EVENT_TYPE_i
    quantity = msg_decoded[5]    # SIZE_i
    side = msg_decoded[2]        # DIRECTION_i
    rel_price = msg_decoded[4]   # PRICE_i
    time_s = msg_decoded[8]      # TIMEs_i
    time_ns = msg_decoded[9]     # TIMEns_i

    # ============================================================
    # FAULT TOLERANCE: Validate and clamp decoded values
    # ============================================================

    # Valid event types: 1=new_order, 2=modify, 3=delete, 4=execute
    # Invalid → treat as NOOP (will set qty=0 later)
    is_valid_event = (event_type >= 1) & (event_type <= 4)

    # Valid side: 0=sell, 1=buy → mapped to -1, 1 for JaxLOB
    is_valid_side = (side >= 0) & (side <= 1)

    # Quantity must be positive for valid trade
    is_valid_qty = quantity > 0

    # Price must be within reasonable range (±1000 ticks from mid)
    is_valid_price = (rel_price >= -1000) & (rel_price <= 1000)

    # Combined validity check
    is_valid_msg = is_valid_event & is_valid_side & is_valid_qty & is_valid_price

    # If invalid, create NOOP message (qty=0 means no trade happens)
    # This is safe: JaxLOB will process but nothing changes
    safe_event_type = jnp.where(is_valid_msg, event_type, 0)
    safe_quantity = jnp.where(is_valid_msg, quantity, 0)
    safe_side = jnp.where(is_valid_msg, side, 0)
    safe_rel_price = jnp.where(is_valid_msg, rel_price, 0)

    # Calculate absolute price
    p_abs = mid_price + safe_rel_price * tick_size

    # Clamp price to positive (JaxLOB requirement)
    p_abs = jnp.maximum(p_abs, tick_size)

    # Construct JaxLOB message
    # Format: [type, side*2-1, qty, price, order_id, trader_id, time_s, time_ns]
    sim_msg = jnp.array([
        safe_event_type,
        (safe_side * 2) - 1,
        safe_quantity,
        p_abs,
        order_id,
        trader_id,  # FIX: Use specified trader_id (policy=-1000, world=-2000)
        time_s,
        time_ns,
    ], dtype=jnp.int32)

    return sim_msg, msg_decoded


def transform_L2_state_wrapper(
    sim_state: 'LobState',
    price_levels: int = 500,
    tick_size: int = 100,
) -> jnp.ndarray:
    """
    Wrapper to convert JaxLOB sim_state to model book input.

    ════════════════════════════════════════════════════════════════════
    OVERALL ARCHITECTURE - Where This Function Fits
    ════════════════════════════════════════════════════════════════════

    ES Training Episode (100 steps):

    ┌──────────────────────────────────────────────────────────┐
    │  [Init] _create_initial_sim_state()                      │
    │    ↓                                                     │
    │  initial_sim_state (JaxLOB with real L2 book)            │
    │  initial_msg_history (500 msgs, 12000 tokens)            │
    └──────────────────────────────────────────────────────────┘
                            ↓
    ┌──────────────────────────────────────────────────────────┐
    │  [Loop] For each step (1-100):                           │
    │                                                          │
    │    1. World Model generates K background messages        │
    │       ├─ Forward: (msg_history, book_feat) → log_probs   │
    │       ├─ Sample: tokens                                  │
    │       ├─ Execute: JaxLOB updates sim_state               │
    │       └─ Update: book_feat = THIS FUNCTION ◄─────────┐   │
    │                                                          │
    │    2. Policy generates 1 trading message                 │
    │       ├─ Forward: (msg_history, book_feat) → log_probs   │
    │       ├─ Sample: tokens                                  │
    │       ├─ Execute: JaxLOB updates sim_state               │
    │       └─ Update: book_feat = THIS FUNCTION ◄─────────┘   │
    └──────────────────────────────────────────────────────────┘
                            ↓
    ┌──────────────────────────────────────────────────────────┐
    │  [Fitness] Compute PnL from final_state.trades           │
    └──────────────────────────────────────────────────────────┘

    ════════════════════════════════════════════════════════════════════
    LOCAL VIEW - Detailed Function Call Chain (One Step)
    ════════════════════════════════════════════════════════════════════

    After JaxLOB executes a message:

         sim_state (updated)
              ↓
    ┌─────────────────────────────────────────┐
    │  transform_L2_state_wrapper()           │  ◄── THIS FUNCTION
    │                                         │
    │  Step 1: Extract L2 arrays              │
    │    ├─ get_L2_state(asks, bids, 10)      │
    │    └─ Output: (40,) raw L2              │
    │         [ask_p0, ask_q0, bid_p0, ...]   │
    │                                         │
    │  Step 2: Add metadata                   │
    │    ├─ mid_diff = 0                      │
    │    ├─ time_s = 34200                    │
    │    ├─ time_ns = 0                       │
    │    └─ Concat → (43,) input              │
    │                                         │
    │  Step 3: Apply training transform       │
    │    └─ transform_L2_state()              │
    │        (from preproc.py)                │
    │        ├─ Price → indices               │
    │        ├─ Build volume image (500)      │
    │        ├─ Norm time                     │
    │        └─ Output: (503,)                │
    └─────────────────────────────────────────┘
              ↓
         book_feat (503,)
              ↓
    ┌─────────────────────────────────────────┐
    │  ES_PaddedLobPredModel._forward_step()  │
    │                                         │
    │  Inputs:                                │
    │    - msg_history[-24:]                  │
    │    - book_feat[None, :] ◄── NEEDS (503,)│
    └─────────────────────────────────────────┘

    ════════════════════════════════════════════════════════════════════
    WHY WRAPPER IS NEEDED
    ════════════════════════════════════════════════════════════════════

    Can't use transform_L2_state() directly because:

    Training data format (from LOBSTER files):
      book_row = [mid_diff, time_s, time_ns, ask_p0, ask_q0, ...]  (43,)
                  ↓
      transform_L2_state(book_row) → (503,)

    ES inference has different source (JaxLOB simulator):
      sim_state = LobState(asks, bids, trades)  ← Different structure!
                  ↓ Need to convert first
      book_row = wrapper extracts and formats → (43,)
                  ↓ Then apply same transform
      transform_L2_state(book_row) → (503,)

    ════════════════════════════════════════════════════════════════════
    Reference: preproc.py:transform_L2_state() (lines 18-63)
              lobster_dataloader.py:523 calls transform_L2_state_numpy()
              Training config: --book_transform=True --book_depth=500

    Args:
        sim_state: JaxLOB LobState (asks, bids, trades arrays)
        price_levels: Volume image size (default 500, matches training)
        tick_size: Tick size in cents (default 100)

    Returns:
        book_feat: (503,) = [mid_diff, time_s_norm, time_ns_norm, volume_image(500)]
    """
    _lazy_import_jaxlob()
    from gymnax_exchange.jaxob.JaxOrderBookArrays import get_L2_state
    from preproc import transform_L2_state  # Reuse training transform

    # Extract L2 from JaxLOB: (40,) = [ask_p0, ask_q0, bid_p0, bid_q0, ...]
    l2_state = get_L2_state(sim_state.asks, sim_state.bids, 10)

    # Ensure l2_state is the right type for concatenation
    l2_state = jnp.asarray(l2_state, dtype=jnp.int32)

    # Construct (43,) input: [mid_diff, time_s, time_ns, L2(40)]
    # Use fixed values for simplicity (mid_diff and time are less critical for ES)
    metadata = jnp.array([0, 34200, 0], dtype=jnp.int32)  # [mid_diff, time_s, time_ns]
    book_input = jnp.concatenate([metadata, l2_state])

    # Apply training transform
    # Note: transform_L2_state is vmapped, expects (batch, 43) input
    # We have (43,), so add batch dimension then squeeze
    book_input_batched = book_input[None, :]  # (1, 43)
    book_feat_batched = transform_L2_state(book_input_batched, price_levels, tick_size)  # (1, 503)
    return book_feat_batched[0]  # (503,)


def get_mid_price(sim_state: 'LobState', tick_size: int = 100) -> int:
    """
    Get current mid price from order book state.

    ============================================================
    FAULT TOLERANCE: Handle edge cases
    ============================================================
    - Empty order book: Return default mid price (10000)
    - Invalid state: Return last known good price or default
    - NaN/Inf: Replace with default

    This ensures simulate_episode() never gets NaN mid_price.
    ============================================================
    """
    _lazy_import_jaxlob()  # Ensure get_best_bid_and_ask is imported
    DEFAULT_MID = 10000  # Safe fallback

    # Extract best bid/ask from order book arrays using JaxLOB utility
    # asks[:, 0] = prices (sorted ascending, -1 for empty slots)
    # bids[:, 0] = prices (sorted descending, -1 for empty slots)
    best_ask, best_bid = get_best_bid_and_ask(sim_state.asks, sim_state.bids)

    # Fault tolerance: check for invalid prices
    # get_best_bid_and_ask returns 999999999 for empty asks, -1 for empty bids
    # Note: In JAX, we use jnp.where for branching within JIT
    # FIX: LOBSTER prices can be ~7720700 ($77.20), need higher threshold
    # 999999999 is the empty marker, so use 900000000 as upper bound
    bid_valid = (best_bid > 0) & (best_bid < 900000000)
    ask_valid = (best_ask > 0) & (best_ask < 900000000)

    best_bid = jnp.where(bid_valid, best_bid, DEFAULT_MID - tick_size)
    best_ask = jnp.where(ask_valid, best_ask, DEFAULT_MID + tick_size)

    mid = (best_bid + best_ask) // 2

    # Final safety: ensure mid is finite and positive
    mid = jnp.where((mid > 0) & jnp.isfinite(mid), mid, DEFAULT_MID)

    return (mid // tick_size) * tick_size


class ESJaxLOBTrainer:
    """
    ES Trainer with JaxLOB environment.

    Implements step-by-step interleaved simulation where:
    - World Model generates background market order flow
    - Policy observes and generates trading actions
    - Both interact through JaxLOB order book simulation
    """

    def __init__(self, config):
        """
        Initialize ES JaxLOB trainer.

        Args:
            config: Namespace with training configuration
        """
        print("[INIT] ========================================")
        print("[INIT] Starting ESJaxLOBTrainer initialization")
        print("[INIT] ========================================")

        self.config = config

        print("[INIT] Step 1/4: Lazy importing JaxLOB...")
        _lazy_import_jaxlob()
        print("[INIT] Step 1/4: JaxLOB imported OK")

        # Load LOBS5 checkpoint (same for both models)
        print(f"[INIT] Step 2/4: Loading LOBS5 checkpoint from {config.lobs5_checkpoint}")
        self.lobs5_init, self.es_tree_key = load_checkpoint_for_es(
            config.lobs5_checkpoint,
        )
        print("[INIT] Step 2/4: Checkpoint loaded OK")

        # Initialize noiser for Policy
        print("[INIT] Step 3/4: Initializing noiser...")
        self._init_noiser()
        print("[INIT] Step 3/4: Noiser initialized OK")

        # Initialize JaxLOB simulator
        print("[INIT] Step 4/5: Initializing JaxLOB simulator...")
        self._init_jaxlob()
        print("[INIT] Step 4/5: JaxLOB simulator initialized OK")

        # Initialize historical replay data (if mode is historical_replay)
        print("[INIT] Step 5/5: Initializing historical replay data...")
        self._init_historical_replay_data()
        print("[INIT] Step 5/5: Historical replay data initialized OK")

        print("[INIT] ========================================")
        print(f"[INIT] ESJaxLOBTrainer initialization COMPLETE")
        print(f"[INIT]   - n_threads: {config.n_threads}")
        print(f"[INIT]   - n_steps per episode: {config.n_steps}")
        print(f"[INIT]   - world_msgs_per_step: {config.world_msgs_per_step}")
        print(f"[INIT]   - task_size: {config.task_size}")
        print("[INIT] ========================================")

    def _init_noiser(self):
        """Initialize EGGROLL noiser for Policy."""
        config = self.config
        NOISER = all_noisers[config.noiser]

        self.noiser_cls = NOISER
        # Use HyperscaleES API: init_noiser returns (frozen_noiser_params, noiser_params)
        self.frozen_noiser_params, self.noiser_params = NOISER.init_noiser(
            self.lobs5_init.params,
            sigma=config.sigma,
            lr=config.lr,
            rank=config.lora_rank,
            freeze_nonlora=False,
            noise_reuse=0,
        )

    def _init_jaxlob(self):
        """Initialize JaxLOB order book simulator."""
        _lazy_import_jaxlob()
        # Create OrderBook instance with larger capacity for ES training
        # Each episode: n_steps * (world_msgs_per_step + 1) orders
        # Plus 500 from replayed messages
        # Conservative: allow buffer for safety
        expected_orders = 500 + self.config.n_steps * (self.config.world_msgs_per_step + 1)
        n_orders = max(1000, int(expected_orders * 1.5))  # 1.5x buffer
        n_trades = max(500, self.config.n_steps * 2)  # Trades usually << orders
        self.sim = OrderBook(nOrders=n_orders, nTrades=n_trades)
        # Create encoder from Vocab class
        from lob.encoding import Vocab
        # Use token_mode from config (default=22 for backward compatibility)
        # Try both modes to find which one the checkpoint was trained with
        vocab = Vocab(token_mode=self.config.token_mode)
        self.encoder = vocab.ENCODING  # Dict[str, Tuple[jax.Array, jax.Array]]
        print(f"[INIT] Using token_mode={self.config.token_mode}, vocab_size={len(vocab)}")

    def _init_historical_replay_data(self):
        """Pre-load historical data for replay mode."""
        if self.config.background_mode != 'historical_replay':
            self.replay_data_raw = None
            self.replay_tokens = None
            print(f"[INIT]   Background mode: {self.config.background_mode} (no replay data needed)")
            return

        # Validate path
        if self.config.replay_data_path is None:
            raise ValueError("--replay_data_path required when background_mode=historical_replay")

        data_path = self.config.replay_data_path
        print(f"[INIT]   Background mode: historical_replay")
        print(f"[INIT]   Loading replay data from: {data_path}")

        # Find message files
        import os
        import glob
        message_files = sorted(glob.glob(os.path.join(data_path, '*message*proc.npy')))

        if len(message_files) == 0:
            raise FileNotFoundError(f"No message files found in {data_path}")

        # Randomly select a file (for variety across training runs)
        import numpy as np
        file_idx = np.random.randint(0, len(message_files))
        selected_file = message_files[file_idx]

        # Store filename for initial state to use the SAME file
        # Extract date from filename (e.g., "GOOG_2022-01-03_..." -> "2022-01-03")
        self.replay_data_date = os.path.basename(selected_file).split('_')[1]
        self.replay_data_dir = data_path

        msg_raw = np.load(selected_file)  # (N, 14)
        print(f"[INIT]   Loaded {msg_raw.shape[0]} raw messages from {os.path.basename(selected_file)}")
        print(f"[INIT]   Replay data date: {self.replay_data_date} (will use same date for initial state)")

        # Pre-encode all messages to tokens upfront (avoids encoding in JIT loop)
        from lob.encoding import encode_msgs
        self.replay_tokens = encode_msgs(msg_raw, self.encoder, token_mode=self.config.token_mode)  # (N, 22/24)
        self.replay_data_raw = jnp.array(msg_raw)  # Keep raw for JaxLOB conversion

        print(f"[INIT]   Pre-encoded {self.replay_tokens.shape[0]} messages to tokens")
        print(f"[INIT]   Token shape per message: {self.replay_tokens.shape[1]}")
        print(f"[INIT]   Replay data ready for sequential playback")

    def _create_initial_sim_state(self) -> Tuple['LobState', jnp.ndarray]:
        """
        Load initial JaxLOB state and message history from LOBSTER data.

        ════════════════════════════════════════════════════════════════════
        WHY THIS FUNCTION EXISTS - Role in Overall Training Flow
        ════════════════════════════════════════════════════════════════════

        ES training needs a REALISTIC starting point for World Model and Policy:

        [Problem 1] Empty order book → No trades possible
          - If we start with sim.reset() (empty book), there are no orders
          - World Model generates messages but nothing matches
          - Policy can't execute trades → fitness always 0

        [Problem 2] Synthetic book → Unrealistic market
          - Manually created L2 (e.g., mid=10000, spread=100) is artificial
          - Model was trained on REAL LOBSTER data distributions
          - Mismatch between training and inference distributions

        [Problem 3] No message history → Cold start
          - Autoregressive model needs context to generate coherent messages
          - Without history, model doesn't know current market state/trend
          - Like asking GPT to continue a story without showing the beginning

        [Solution] Warm start with real historical data:
          1. Load real L2 book from LOBSTER data → realistic initial spread/depth
          2. Replay 500 real messages → build up realistic order flow
          3. Use those 500 messages as context → model sees real market history

        This is the SAME approach used in inference_no_errcorr.py for evaluation.

        ════════════════════════════════════════════════════════════════════
        Reference Implementation
        ════════════════════════════════════════════════════════════════════

        lob/inference_no_errcorr.py:get_sim() does the same thing but with
        parameterized window size (n_inp_msgs). We use 500 to match
        training configuration (msg_seq_len=500).
        
        CAUTION: [< 500 >] is context length, it can be changed.

        Difference:
          - inference_no_errcorr.py: uses moving_window for evaluation
          - This function: fixed 500 messages for generation warm-start

        ════════════════════════════════════════════════════════════════════

        Process:
          1. Load orderbook_10_proc.npy: book[0, 3:43] → init_l2_book (40 values)
          2. Load message_10_proc.npy: messages[0:500] → replay to JaxLOB
          3. Encode all 500 messages → msg_history (12000 tokens)

        Returns:
            (sim_state, msg_history)
            - sim_state: JaxLOB state after replaying 500 historical messages
            - msg_history: (12000,) all 500 messages encoded as tokens
        """
        import numpy as np
        import glob
        from lob.encoding import encode_msgs

        config = self.config

        # Data directory (configurable or default to GOOG 2022)
        if hasattr(config, 'data_dir') and config.data_dir:
            data_dir = config.data_dir
        elif config.background_mode == 'historical_replay' and config.replay_data_path:
            # CRITICAL: When using historical_replay, use same data source for initial state
            # This ensures price levels are consistent between initial book and replayed messages
            data_dir = config.replay_data_path
            print(f"[INIT] Using replay_data_path for initial state (consistent price levels)")
        else:
            # Default: GOOG 2022 data
            data_dir = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022"

        # Find all data files
        orderbook_files = sorted(glob.glob(f"{data_dir}/*orderbook_10_proc.npy"))
        message_files = sorted(glob.glob(f"{data_dir}/*message_10_proc.npy"))

        if len(orderbook_files) == 0:
            raise FileNotFoundError(f"No orderbook files found in {data_dir}")

        # Select data file
        if config.background_mode == 'historical_replay' and hasattr(self, 'replay_data_date'):
            # CRITICAL: Use SAME date as replay data for consistent initial state
            # Find file matching the replay_data_date
            matching_files = [f for f in orderbook_files if self.replay_data_date in f]
            if len(matching_files) == 0:
                raise FileNotFoundError(f"No orderbook file found for date {self.replay_data_date} in {data_dir}")
            file_idx = orderbook_files.index(matching_files[0])
            print(f"[INIT] Using SAME date as replay data: {self.replay_data_date}")
        elif hasattr(config, 'file_idx'):
            file_idx = config.file_idx % len(orderbook_files)
        else:
            file_idx = np.random.randint(0, len(orderbook_files))

        print(f"Loading data from file {file_idx}: {orderbook_files[file_idx]}")

        # Load data
        ob = np.load(orderbook_files[file_idx])   # (N+1, 43)
        msg = np.load(message_files[file_idx])    # (N, 14)

        # Extract initial L2 book (row 0, columns 3-43)
        init_l2_book = jnp.array(ob[0, 3:43], dtype=jnp.int32)  # (40,)

        # Initialize JaxLOB with initial L2 book
        # Note: JaxLOB reset() only takes l2_book, not start_time
        sim_state = self.sim.reset(init_l2_book)

        # Replay first 500 messages (or fewer if data is shorter)
        n_replay = min(500, len(msg))
        replay_msgs_raw = msg[:n_replay]  # (n_replay, 14)

        # Convert to JaxLOB format and replay
        replay_jaxlob = msgs_to_jnp(replay_msgs_raw)  # (n_replay, 8)
        sim_state = self.sim.process_orders_array(sim_state, replay_jaxlob)

        # Encode all 500 messages as context (match training msg_seq_len=500)
        # Why use all 500:
        #   - Training uses msg_seq_len=500
        #   - Full autoregressive model needs full context
        #   - inference_no_errcorr.py uses moving_window but we use fixed 500
        tokens = encode_msgs(replay_msgs_raw, self.encoder)  # (500, 24)
        msg_history = tokens.flatten()  # (12000,) = 500 × 24

        print(f"  Initialized with {n_replay} messages, context size: {msg_history.shape}")

        return sim_state, msg_history

    def create_world_common_params(self) -> CommonParams:
        """Create CommonParams for World Model (frozen, no noise)."""
        # Use noop noiser or set iterinfo=None for no perturbation
        return CommonParams(
            noiser=self.noiser_cls,
            frozen_noiser_params=self.frozen_noiser_params,
            noiser_params=self.noiser_params,
            params=self.lobs5_init.params,
            es_tree_key=self.es_tree_key,
            frozen_params=self.lobs5_init.frozen_params,
            iterinfo=None,  # None = no noise for World Model
        )

    def create_policy_common_params(self, epoch: int, thread_id: int) -> CommonParams:
        """Create CommonParams for Policy (with ES perturbation)."""
        iterinfo = (jnp.int32(epoch), jnp.int32(thread_id))
        return CommonParams(
            noiser=self.noiser_cls,
            frozen_noiser_params=self.frozen_noiser_params,
            noiser_params=self.noiser_params,
            params=self.lobs5_init.params,  # Will be perturbed via iterinfo
            es_tree_key=self.es_tree_key,
            frozen_params=self.lobs5_init.frozen_params,
            iterinfo=iterinfo,  # (epoch, thread) for noise generation
        )

    def simulate_episode(
        self,
        key: jnp.ndarray,
        world_common_params: CommonParams,
        policy_common_params: CommonParams,
        sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
        thread_id: int = -1,  # For debug printing (only print thread 0)
    ) -> float:
        """
        Run a complete episode with step-by-step interleaved simulation.

        Args:
            key: JAX random key
            world_common_params: World Model params (frozen)
            policy_common_params: Policy params (ES perturbed)
            sim_state: Initial JaxLOB state
            initial_msg_history: (480,) optional initial message history from real data
            thread_id: Thread ID for debug printing (only thread 0 prints)

        Returns:
            fitness: total_revenue (PnL)
        """
        config = self.config
        fp = self.lobs5_init.frozen_params

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

        # Initialize episode state
        # msg_len must match token_mode: 22 for tok22, 24 for tok24
        msg_len = 22 if config.token_mode == 22 else 24

        # Context length: extract from checkpoint metadata (msg_seq_len)
        # Why msg_seq_len from checkpoint:
        #   1. Matches training configuration exactly
        #   2. Ensures consistency between training and inference
        #   3. Allows flexibility for different model sizes
        msg_seq_len = fp.get('msg_seq_len', 500)  # Extract from checkpoint metadata
        context_len = msg_len * msg_seq_len
        print(f"[EPISODE] msg_len={msg_len}, msg_seq_len={msg_seq_len}, context_len={context_len}")

        # ============================================================
        # ORDER ID DESIGN (FIX for trader identification)
        # ============================================================
        # Use distinct order_id ranges for policy vs world:
        # - Policy orders: 1000000, 1000001, 1000002, ...
        # - World orders:  2000000, 2000001, 2000002, ...
        # This allows easy identification in trades array
        # ============================================================
        POLICY_ORDER_ID_START = 1000000
        WORLD_ORDER_ID_START = 2000000

        # ============================================================
        # CRITICAL FIX: Clear trades array from historical replay
        # ============================================================
        # _create_initial_sim_state() replays 500 historical messages,
        # which produces trades with old order_ids (9M+).
        # These must be cleared before starting ES training,
        # otherwise fitness calculation includes historical trades.
        # NOTE: Only clear trades, NOT asks/bids (order book needs liquidity!)
        # ============================================================
        sim_state = sim_state._replace(
            trades=(jnp.ones((sim_state.trades.shape[0], 6)) * -1).astype(jnp.int32)
        )

        # FIX: Calculate init_mid_price BEFORE debug print
        init_mid_price = get_mid_price(sim_state, config.tick_size)

        # === DEBUG: Check order book after clearing trades ===
        n_asks = jnp.sum(sim_state.asks[:, 0] != -1)
        n_bids = jnp.sum(sim_state.bids[:, 0] != -1)
        jax.debug.print(
            "[DEBUG INIT] After clearing trades: n_asks={}, n_bids={}, init_mid_price={}",
            n_asks, n_bids, init_mid_price
        )

        # Use provided msg_history if available, otherwise zeros
        if initial_msg_history is not None:
            msg_history = initial_msg_history
        else:
            msg_history = jnp.zeros((context_len,), dtype=jnp.int32)

        # Extract book_depth from checkpoint metadata
        book_depth = fp.get('book_depth', 500)
        book_feat = transform_L2_state_wrapper(sim_state, price_levels=book_depth, tick_size=config.tick_size)

        # ============================================================
        # POLICY ORDER ID TRACKING & EXECUTION TRACKING
        # ============================================================
        # (init_mid_price already calculated above for debug print)

        # Policy order IDs follow a predictable pattern:
        # Each step: world_msgs_per_step world orders, then 1 policy order
        # So policy order IDs are: K, 2K+1, 3K+2, ... where K = world_msgs_per_step
        # We can compute this at the end instead of tracking explicitly
        #
        # Execution tracking (like JaxMARL-HFT exec_env.py):
        # - quant_executed: cumulative quantity executed by policy
        # - task_size: target quantity to execute (from config)
        # - done: True when task_done OR max_steps reached
        # - When quant_executed >= task_size, set qty=0 (no more orders)
        # - Use jax.lax.scan with fixed steps (like JaxMARL-HFT IPPO training)
        # ============================================================

        # Initialize execution tracking
        task_size = jnp.int32(config.task_size)

        # ============================================================
        # JAX.LAX.SCAN IMPLEMENTATION (like JaxMARL-HFT IPPO training)
        # ============================================================
        # Note: JaxMARL-HFT uses scan for training, while_loop only for timing tests.
        # Scan is more efficient for fixed-step training.
        # When task is complete, we set qty=0 to stop new orders.

        def step_fn(carry, step_idx):
            """Single step: World Model messages → Policy action."""
            (key, msg_history, hiddens_world, hiddens_policy,
             sim_state, book_feat, world_oid_offset, quant_executed) = carry

            key, key_world, key_policy = jax.random.split(key, 3)

            # ====== 1. BACKGROUND MODEL: Historical Replay or World Model ======
            # ============================================================
            # HISTORICAL REPLAY MODE: Load pre-encoded messages from data
            # ============================================================
            def historical_replay_step(wcarry, bg_msg_idx):
                """
                Sequential replay of historical market messages (no forward pass).

                NOTE: Variable names use 'world_*' prefix for interface compatibility,
                but this function replays HISTORICAL DATA, not model-generated messages.

                CRITICAL DESIGN: This replaces world model's 2200 forward passes/step
                with direct data lookup, achieving ~100x speedup for background generation.

                Args:
                    bg_msg_idx: Background message index (0 to world_msgs_per_step-1)
                """
                key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr = wcarry

                # ============================================================
                # HISTORICAL REPLAY: Load pre-encoded message from data
                # ============================================================
                # NO model inference! Just array lookup from pre-loaded data
                # ============================================================
                replayed_msg_tokens = self.replay_tokens[replay_ptr]  # (msg_len,) - from HISTORICAL DATA
                replayed_msg_raw = self.replay_data_raw[replay_ptr]  # (14,) - from HISTORICAL DATA

                # Convert to JaxLOB format using existing decoder
                sim_msg = decoded_msg_to_jaxlob_format(replayed_msg_raw)

                # Override order_id and trader_id for tracking
                # (Keep same ID scheme as world model for consistency)
                bg_order_id = WORLD_ORDER_ID_START + oid_offset
                sim_msg = sim_msg.at[4].set(bg_order_id)  # order_id
                sim_msg = sim_msg.at[5].set(-2000)        # trader_id (background/historical)

                # Process in simulator
                sim_st = self.sim.process_order_array(sim_st, sim_msg)

                # Update book features
                book_f = transform_L2_state_wrapper(sim_st, price_levels=book_depth, tick_size=config.tick_size)

                # Update msg_history with replayed tokens (keep context consistent with training)
                msg_hist = jnp.concatenate([msg_hist[msg_len:], replayed_msg_tokens])

                # Advance replay pointer
                new_replay_ptr = replay_ptr + 1

                # Handle data exhaustion: loop back to start (after warmup messages)
                n_replay_msgs = self.replay_tokens.shape[0]
                new_replay_ptr = jnp.where(
                    new_replay_ptr >= n_replay_msgs,
                    jnp.int32(500),  # Loop back to message 500 (after warmup)
                    new_replay_ptr
                )

                # === DEBUG: Print replayed messages from HISTORICAL DATA ===
                jax.lax.cond(
                    (thread_id == 0) & (step_idx < 5) & (bg_msg_idx < 3),
                    lambda: jax.debug.print(
                        "[REPLAY] step={:3d}, bg_msg={:3d}, oid={:7d}, event={:1d}, side={:1d}, qty={:5d}, price={:8d}, size_tok={:5d}",
                        step_idx, bg_msg_idx, bg_order_id,
                        replayed_msg_raw[1], replayed_msg_raw[2], replayed_msg_raw[5], sim_msg[3],
                        replayed_msg_tokens[4],  # size token
                        ordered=True
                    ),
                    lambda: None,
                )

                oid_offset = oid_offset + 1

                return (key, msg_hist, hidden, sim_st, book_f, oid_offset, new_replay_ptr), replayed_msg_tokens

            # ====== 1. World Model generates K background messages ======
            def world_msg_step(wcarry, world_msg_idx):
                key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr = wcarry

                # ============================================================
                # AUTOREGRESSIVE TOKEN-BY-TOKEN SAMPLING (FIX)
                # ============================================================
                # Model is trained autoregressively: given tokens[0:k], predict token[k+1]
                # We must sample one token at a time, appending to context
                # ============================================================
                def sample_one_token(token_carry, _):
                    """Sample one token autoregressively."""
                    key_t, msg_hist_t, hidden_t = token_carry
                    key_t, sample_key_t = jax.random.split(key_t)

                    # Forward with current context
                    hidden_t, log_probs_t = ES_PaddedLobPredModel._forward_step(
                        world_common_params, hidden_t, msg_hist_t[-msg_len:], book_f[None, :]
                    )
                    # Keep only last position's hidden
                    hidden_t = jax.tree.map(lambda h: h[:, -1:, :], hidden_t)

                    # Sample ONLY the last position (next token)
                    log_probs_t = jnp.nan_to_num(log_probs_t, nan=-1e9, posinf=1e9, neginf=-1e9)
                    next_token = jax.random.categorical(sample_key_t, log_probs_t[-1])  # Shape: ()

                    # Append to context (sliding window)
                    msg_hist_t = jnp.concatenate([msg_hist_t[1:], jnp.array([next_token])])

                    return (key_t, msg_hist_t, hidden_t), next_token

                # Sample 24 tokens autoregressively
                key, sample_key = jax.random.split(key)
                (key, msg_hist, hidden), world_msg = jax.lax.scan(
                    sample_one_token,
                    (sample_key, msg_hist, hidden),
                    None,
                    length=msg_len,  # Sample 24 tokens
                )
                # world_msg shape: (24,)
                # ============================================================

                # Convert to JaxLOB format and process
                mid_price = get_mid_price(sim_st, config.tick_size)
                # FIX: World order_id from WORLD range (2000000 + offset)
                world_order_id = WORLD_ORDER_ID_START + oid_offset
                sim_msg, msg_decoded = get_sim_msg_es(
                    world_msg, self.sim, sim_st, mid_price, world_order_id, config.tick_size, self.encoder,
                    trader_id=-2000,  # FIX: World Model trader ID
                    token_mode=config.token_mode
                )

                # === DEBUG: Print world model orders (only thread 0, first few steps) ===
                # tok22: size is token[4], tok24: size is tokens[4:6]
                jax.lax.cond(
                    (thread_id == 0) & (step_idx < 5) & (world_msg_idx < 3),
                    lambda: jax.debug.print(
                        "[WORLD] step={}, world_msg={}, oid={}, event={}, side={}, qty={}, price={}, size_tok={}",
                        step_idx, world_msg_idx, world_order_id,
                        msg_decoded[1], msg_decoded[2], msg_decoded[5], sim_msg[3],
                        world_msg[4],  # size token (tok22: single, tok24: high digit)
                        ordered=True
                    ),
                    lambda: None,
                )

                sim_st = self.sim.process_order_array(sim_st, sim_msg)

                # Update for next iteration
                book_f = transform_L2_state_wrapper(sim_st, price_levels=book_depth, tick_size=config.tick_size)
                msg_hist = jnp.concatenate([msg_hist[msg_len:], world_msg])
                oid_offset = oid_offset + 1

                return (key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr), world_msg

            # ============================================================
            # Generate world_msgs_per_step background messages SEQUENTIALLY
            # ============================================================
            # IMPORTANT: Sequential generation (not parallel) because:
            #   1. Each message modifies orderbook state
            #   2. Message N+1 depends on book state after message N
            #   3. Each message is also autoregressive (22 tokens sequentially)
            # Total sequential operations: 100 world msgs × 22 tokens = 2200 forward passes
            # ============================================================
            # MODE SWITCH: Select background message generation function
            # ============================================================
            background_mode = config.background_mode
            if background_mode == 'historical_replay':
                step_fn_background = historical_replay_step
                # Initialize replay pointer (from step counter + warmup offset)
                replay_ptr_init = jnp.int32(500 + step_idx * config.world_msgs_per_step)
            else:  # world_model (default)
                step_fn_background = world_msg_step
                replay_ptr_init = jnp.int32(0)  # Not used but kept for interface consistency

            # Generate background messages (world_model or historical_replay)
            (key_world, msg_history, hiddens_world, sim_state, book_feat,
             world_oid_offset, replay_ptr_final), _ = jax.lax.scan(
                step_fn_background,
                (key_world, msg_history, hiddens_world, sim_state, book_feat, world_oid_offset, replay_ptr_init),
                jnp.arange(config.world_msgs_per_step),
                length=config.world_msgs_per_step,
            )

            # ====== 2. Policy observes and generates action ======
            # ============================================================
            # AUTOREGRESSIVE TOKEN-BY-TOKEN SAMPLING (SEQUENTIAL)
            # ============================================================
            # Policy generates 1 action message with 22 tokens sequentially
            # Sequential operations: 22 forward passes for policy message
            # ============================================================
            def sample_policy_token(token_carry, _):
                """Sample one policy token autoregressively."""
                key_p, msg_hist_p, hidden_p = token_carry
                key_p, sample_key_p = jax.random.split(key_p)

                # Forward with current context (with ES noise)
                hidden_p, log_probs_p = ES_PaddedLobPredModel._forward_step(
                    policy_common_params, hidden_p, msg_hist_p[-msg_len:], book_feat[None, :]
                )
                # Keep only last position's hidden
                hidden_p = jax.tree.map(lambda h: h[:, -1:, :], hidden_p)

                # Sample ONLY the last position
                log_probs_p = jnp.nan_to_num(log_probs_p, nan=-1e9, posinf=1e9, neginf=-1e9)
                next_token_p = jax.random.categorical(sample_key_p, log_probs_p[-1])  # Shape: ()

                # Append to context
                msg_hist_p = jnp.concatenate([msg_hist_p[1:], jnp.array([next_token_p])])

                return (key_p, msg_hist_p, hidden_p), next_token_p

            # Sample 24 policy tokens autoregressively
            key_policy, sample_key = jax.random.split(key_policy)
            (key_policy, msg_history, hiddens_policy), policy_msg = jax.lax.scan(
                sample_policy_token,
                (sample_key, msg_history, hiddens_policy),
                None,
                length=msg_len,  # Sample 24 tokens
            )
            # policy_msg shape: (24,)
            # ============================================================

            # Convert to JaxLOB format
            mid_price = get_mid_price(sim_state, config.tick_size)
            # FIX: Policy order_id from POLICY range (1000000 + step_idx)
            policy_order_id = POLICY_ORDER_ID_START + step_idx
            sim_msg, msg_decoded = get_sim_msg_es(
                policy_msg, self.sim, sim_state, mid_price, policy_order_id, config.tick_size, self.encoder,
                trader_id=-1000,  # FIX: Policy trader ID (all policy orders have same trader)
                token_mode=config.token_mode
            )

            # ============================================================
            # CRITICAL: CANCEL PREVIOUS UNFILLED POLICY ORDER BEFORE SUBMITTING NEW ONE
            # ============================================================
            # IMPORTANT: Each step, cancel the previous step's policy order if it hasn't filled
            # This prevents orderbook from accumulating unfilled limit orders
            # Cancel method: Directly remove order from sim_state.asks/bids by order_id
            # (NOT via cancel message, which would add latency and complexity)
            # ============================================================
            prev_policy_oid = POLICY_ORDER_ID_START + step_idx - 1

            # Cancel from asks (if policy was selling)
            is_prev_order_in_asks = sim_state.asks[:, 2] == prev_policy_oid
            sim_state = sim_state._replace(
                asks=jnp.where(
                    is_prev_order_in_asks[:, None],
                    jnp.array([0, 0, -1, -1, 0, 0]),  # Mark as removed
                    sim_state.asks
                )
            )

            # Cancel from bids (if policy was buying)
            is_prev_order_in_bids = sim_state.bids[:, 2] == prev_policy_oid
            sim_state = sim_state._replace(
                bids=jnp.where(
                    is_prev_order_in_bids[:, None],
                    jnp.array([0, 0, -1, -1, 0, 0]),  # Mark as removed
                    sim_state.bids
                )
            )
            # ============================================================

            # ============================================================
            # EXECUTION LIMIT: Truncate quantity to remaining task
            # ============================================================
            # sim_msg format: [type, side, qty, price, order_id, trader_id, time_s, time_ns]
            quant_remaining = task_size - quant_executed
            original_qty = sim_msg[2]
            truncated_qty = jnp.minimum(original_qty, jnp.maximum(quant_remaining, 0))
            sim_msg = sim_msg.at[2].set(truncated_qty)

            # === DEBUG: Print policy orders (only thread 0, first 10 steps) ===
            # tok22: size is token[4], tok24: size is tokens[4:6]
            jax.lax.cond(
                (thread_id == 0) & (step_idx < 10),
                lambda: jax.debug.print(
                    "[POLICY] step={}, oid={}, event={}, side={}, qty={} (orig={}), price={}, size_tok={}",
                    step_idx, policy_order_id,
                    msg_decoded[1], msg_decoded[2],  # event_type, direction
                    truncated_qty, original_qty,     # qty after/before truncation
                    sim_msg[3],                      # price
                    policy_msg[4],                   # size token (tok22: single, tok24: high digit)
                    ordered=True
                ),
                lambda: None,
            )
            # ============================================================

            # ============================================================
            # EXECUTION LOGIC: JaxLOB Matching + Pseudo Execution (Last Step Only)
            # ============================================================
            # NORMAL STEPS (0-98): Use JaxLOB matching engine
            # LAST STEP (99): If not completed, add pseudo execution with 10% slippage
            # ============================================================

            # Process order through JaxLOB matching engine
            sim_state = self.sim.process_order_array(sim_state, sim_msg)

            # Track executed quantity from real trades
            trades = sim_state.trades
            is_new_trade = (trades[:, 0] != -1)  # valid trade
            is_policy_in_trade = ((trades[:, 2] == policy_order_id) | (trades[:, 3] == policy_order_id)) & is_new_trade
            step_executed_from_matching = jnp.sum(jnp.where(is_policy_in_trade, jnp.abs(trades[:, 1]), 0))

            # ============================================================
            # PSEUDO EXECUTION: Last step only, if task not completed
            # ============================================================
            # CRITICAL: Only execute on LAST STEP (step_idx == n_steps - 1)
            # If task is incomplete, force execute remaining quantity at slippage price
            # Rationale:
            #   - Ensures task completion for fitness calculation
            #   - Simulates aggressive market order execution
            #   - Sell: Execute at (Best Bid × 0.9) - 10% slippage below best
            #   - Buy:  Execute at (Best Ask × 1.1) - 10% slippage above best
            # ============================================================
            is_last_step = step_idx == (config.n_steps - 1)
            remaining_qty = config.task_size - (quant_executed + step_executed_from_matching)

            step_executed_pseudo = jnp.where(
                is_last_step & (remaining_qty > 0) & (truncated_qty > 0),
                remaining_qty,  # Force execute all remaining
                jnp.int32(0)    # No pseudo execution
            )

            # Calculate pseudo execution price (used for metrics only)
            best_ask = sim_state.asks[0, 0]
            best_bid = sim_state.bids[0, 0]
            is_sell = msg_decoded[2] == 0
            pseudo_exec_price = jnp.where(
                is_sell,
                jnp.int32(best_bid * 0.9),  # Sell: 10% below best bid
                jnp.int32(best_ask * 1.1),  # Buy:  10% above best ask
            )

            # Total execution this step
            step_executed = step_executed_from_matching + step_executed_pseudo
            quant_executed = quant_executed + step_executed
            # ============================================================

            # === DEBUG: Step separator and execution summary (aligned table format) ===
            jax.lax.cond(
                thread_id == 0,
                lambda: jax.debug.print(
                    "\n" + "-"*110 + "\n"
                    "[STEP {:3d}] oid={:7d} | qty={:4d} | price={:8d} | exec_step={:4d} | total={:4d}/{:4d} | remain={:4d}",
                    step_idx,
                    policy_order_id, truncated_qty, sim_msg[3],
                    step_executed,
                    quant_executed, config.task_size,
                    config.task_size - quant_executed,
                    ordered=True
                ),
                lambda: None,
            )

            # === DEBUG: Trade details (DISABLED to reduce log size and improve speed) ===
            # Uncomment if needed for debugging
            # def print_trade_table():
            #     jax.debug.print(
            #         "  ✓ EXECUTED {:4d} shares | Matching: {:4d} | Pseudo: {:4d}",
            #         step_executed, step_executed_from_matching, step_executed_pseudo,
            #         ordered=True
            #     )
            # jax.lax.cond((thread_id == 0) & (step_executed > 0), print_trade_table, lambda: None)

            # === DEBUG: Orderbook snapshot (detailed format, every step for thread 0) ===
            # Format matching logs_55M/print_orderbook
            jax.lax.cond(
                thread_id == 0,
                lambda: jax.debug.print(
                    "\n[BOOK-SNAP] Step {}\n"
                    "  ========== ASK (Sell) ==========\n"
                    "  L10: price={:8.2f}  qty={:5d}\n"
                    "  L9:  price={:8.2f}  qty={:5d}\n"
                    "  L8:  price={:8.2f}  qty={:5d}\n"
                    "  L7:  price={:8.2f}  qty={:5d}\n"
                    "  L6:  price={:8.2f}  qty={:5d}\n"
                    "  L5:  price={:8.2f}  qty={:5d}\n"
                    "  L4:  price={:8.2f}  qty={:5d}\n"
                    "  L3:  price={:8.2f}  qty={:5d}\n"
                    "  L2:  price={:8.2f}  qty={:5d}\n"
                    "  L1:  price={:8.2f}  qty={:5d} <- Best Ask\n"
                    "  ---------- SPREAD ----------\n"
                    "  L1:  price={:8.2f}  qty={:5d} <- Best Bid\n"
                    "  L2:  price={:8.2f}  qty={:5d}\n"
                    "  L3:  price={:8.2f}  qty={:5d}\n"
                    "  L4:  price={:8.2f}  qty={:5d}\n"
                    "  L5:  price={:8.2f}  qty={:5d}\n"
                    "  L6:  price={:8.2f}  qty={:5d}\n"
                    "  L7:  price={:8.2f}  qty={:5d}\n"
                    "  L8:  price={:8.2f}  qty={:5d}\n"
                    "  L9:  price={:8.2f}  qty={:5d}\n"
                    "  L10: price={:8.2f}  qty={:5d}\n"
                    "  ========== BID (Buy) ==========",
                    step_idx,
                    # ASK side (L10 to L1, reversed order)
                    sim_state.asks[9, 0] / config.tick_size, sim_state.asks[9, 1],
                    sim_state.asks[8, 0] / config.tick_size, sim_state.asks[8, 1],
                    sim_state.asks[7, 0] / config.tick_size, sim_state.asks[7, 1],
                    sim_state.asks[6, 0] / config.tick_size, sim_state.asks[6, 1],
                    sim_state.asks[5, 0] / config.tick_size, sim_state.asks[5, 1],
                    sim_state.asks[4, 0] / config.tick_size, sim_state.asks[4, 1],
                    sim_state.asks[3, 0] / config.tick_size, sim_state.asks[3, 1],
                    sim_state.asks[2, 0] / config.tick_size, sim_state.asks[2, 1],
                    sim_state.asks[1, 0] / config.tick_size, sim_state.asks[1, 1],
                    sim_state.asks[0, 0] / config.tick_size, sim_state.asks[0, 1],
                    # BID side (L1 to L10)
                    sim_state.bids[0, 0] / config.tick_size, sim_state.bids[0, 1],
                    sim_state.bids[1, 0] / config.tick_size, sim_state.bids[1, 1],
                    sim_state.bids[2, 0] / config.tick_size, sim_state.bids[2, 1],
                    sim_state.bids[3, 0] / config.tick_size, sim_state.bids[3, 1],
                    sim_state.bids[4, 0] / config.tick_size, sim_state.bids[4, 1],
                    sim_state.bids[5, 0] / config.tick_size, sim_state.bids[5, 1],
                    sim_state.bids[6, 0] / config.tick_size, sim_state.bids[6, 1],
                    sim_state.bids[7, 0] / config.tick_size, sim_state.bids[7, 1],
                    sim_state.bids[8, 0] / config.tick_size, sim_state.bids[8, 1],
                    sim_state.bids[9, 0] / config.tick_size, sim_state.bids[9, 1],
                    ordered=True
                ),
                lambda: None,
            )

            # Update state for next step
            book_feat = transform_L2_state_wrapper(sim_state, price_levels=book_depth, tick_size=config.tick_size)
            msg_history = jnp.concatenate([msg_history[msg_len:], policy_msg])

            return (key, msg_history, hiddens_world, hiddens_policy, sim_state,
                    book_feat, world_oid_offset, quant_executed), None

        # ============================================================
        # Run episode with jax.lax.scan (SEQUENTIAL, fixed steps)
        # ============================================================
        # 3-level sequential nesting:
        #   1. Steps loop: 100 steps (sequential via jax.lax.scan)
        #   2. World messages: 100 background msgs per step (sequential via jax.lax.scan)
        #   3. Token generation: 22 tokens per message (autoregressive via jax.lax.scan)
        # Total forward passes per episode: 100 steps × (100 world + 1 policy) × 22 tokens = 222,200
        # ============================================================
        (_, _, _, _, final_state, _, final_world_oid_offset, final_quant_executed), _ = jax.lax.scan(
            step_fn,
            (key, msg_history, hiddens_world, hiddens_policy, sim_state,
             book_feat, jnp.int32(0), jnp.int32(0)),  # world_oid_offset=0, quant_executed=0
            jnp.arange(config.n_steps),
            length=config.n_steps,
        )
        final_step_counter = config.n_steps  # Fixed step count

        # FIX: Calculate final_order_id for capacity check
        # Max order_id used is the last policy order
        final_order_id = POLICY_ORDER_ID_START + config.n_steps - 1

        # ============================================================
        # FORCE MARKET ORDER AT EPISODE END (like JaxMARL-HFT)
        # ============================================================
        # If episode ends (max_steps reached) but task not complete,
        # force a "doom trade" at current best price to close position.
        #
        # This simulates the market impact of being forced to execute
        # remaining quantity at unfavorable prices.
        # ============================================================
        quant_left = task_size - final_quant_executed

        # Get current best bid/ask for doom price
        best_ask, best_bid = get_best_bid_and_ask(final_state.asks, final_state.bids)

        # Doom price: for sell task use best_bid (aggressive sell), for buy task use best_ask
        # Currently assuming sell task (config.task == 'sell')
        is_sell_task = (config.task == 'sell')
        doom_price = jnp.where(is_sell_task, best_bid, best_ask)

        # Fallback if order book is empty
        doom_price = jnp.where(
            (doom_price > 0) & (doom_price < 999999999),
            doom_price,
            init_mid_price  # Use init_mid_price as fallback
        )

        # Create doom trade if there's remaining quantity
        # Add to trades array as a synthetic trade
        def add_doom_trade(state, quant, price):
            """Add a synthetic doom trade to close remaining position."""
            trades = state.trades
            # Find first empty slot
            empty_mask = trades[:, 0] == -1
            empty_idx = jnp.argmax(empty_mask)

            # Create doom trade record
            # Format: [price, quantity, buyer_id, seller_id, time_s, time_ns]
            # Use special IDs (-666666) to mark as doom trade (like JaxMARL-HFT)
            doom_trade = jnp.array([
                price,
                jnp.abs(quant),
                -666666,  # buyer_id (doom marker)
                -666666,  # seller_id (doom marker)
                0,        # time_s
                0,        # time_ns
            ], dtype=jnp.int32)

            # Only add if there's space and quant > 0
            should_add = (quant > 0) & (empty_idx < trades.shape[0])
            new_trades = jax.lax.cond(
                should_add,
                lambda t: t.at[empty_idx].set(doom_trade),
                lambda t: t,
                trades
            )

            return state._replace(trades=new_trades)

        # Apply doom trade if quant_left > 0
        final_state = jax.lax.cond(
            quant_left > 0,
            lambda s: add_doom_trade(s, quant_left, doom_price),
            lambda s: s,
            final_state
        )

        # Update final_quant_executed to include doom quantity
        final_quant_executed = final_quant_executed + jnp.maximum(quant_left, 0)
        # ============================================================

        # ============================================================
        # CAPACITY CHECK: Warn if JaxLOB arrays near full
        # ============================================================
        # JaxLOB uses fixed-size arrays. When full, it silently overwrites
        # the last row (see JaxOrderBookArrays.py:35-36).
        # We dynamically size arrays in _init_jaxlob(), but check usage here.
        #
        # JaxLOB array overflow
        # ============================================================
        n_trades_used = jnp.sum(final_state.trades[:, 0] != -1)

        # Warning thresholds (non-blocking, just for monitoring)
        jax.debug.print(
            "JaxLOB capacity: orders={}/{}, trades={}/{}",
            final_order_id, self.sim.nOrders,
            n_trades_used, self.sim.nTrades,
            ordered=True
        )

        # ============================================================
        # FITNESS FUNCTION: Real PnL Computation
        # ============================================================
        #
        # Policy Order ID Pattern:
        #   Each step generates: world_msgs_per_step world orders + 1 policy order
        #   Policy order IDs: K, 2K+1, 3K+2, ... where K = world_msgs_per_step
        #   Formula: policy_order_id[i] = i * (K + 1) + K
        #
        # Trades array structure (nTrades, 6):
        #   trades[:, 0]: execution_price
        #   trades[:, 1]: quantity
        #   trades[:, 2]: buyer_order_id
        #   trades[:, 3]: seller_order_id
        #   trades[:, 4]: timestamp_seconds
        #   trades[:, 5]: timestamp_nanoseconds
        # ============================================================

        trades = final_state.trades
        K = config.world_msgs_per_step
        tick_size = config.tick_size  # For price normalization

        # Policy order IDs follow pattern: K, 2K+1, 3K+2, ...
        # Formula: order_id is a policy order if (order_id - K) % (K + 1) == 0
        # And order_id >= K and order_id < total_orders
        # This is more JAX-friendly than jnp.isin

        # Valid trades mask (price != -1)
        valid_trades_mask = trades[:, 0] != -1
        n_valid_trades = jnp.sum(valid_trades_mask)

        # === DEBUG: Trades array structure ===
        jax.debug.print(
            "[DEBUG TRADES] n_valid={}, first 5 trades:\n"
            "  prices: {}\n"
            "  qtys: {}\n"
            "  col2 (passive_oid): {}\n"
            "  col3 (aggr_oid): {}",
            n_valid_trades,
            trades[:5, 0], trades[:5, 1], trades[:5, 2], trades[:5, 3]
        )

        # ============================================================
        # DOOM TRADE DETECTION (like JaxMARL-HFT)
        # ============================================================
        # Doom trades have special marker ID: -666666
        # These are forced liquidation trades at episode end
        # ============================================================
        is_doom_trade = (trades[:, 2] == -666666) | (trades[:, 3] == -666666)
        is_doom_trade = is_doom_trade & valid_trades_mask

        # ============================================================
        # IDENTIFY POLICY TRADES (FIX: Use order_id ranges)
        # ============================================================
        # trades[:, 2] = passive_oid, trades[:, 3] = aggr_oid
        # Policy orders have IDs in range [1000000, 2000000)
        # World orders have IDs in range [2000000, ...)
        # ============================================================
        passive_ids = trades[:, 2]
        aggr_ids = trades[:, 3]

        # Check if passive or aggressor is a policy order
        is_policy_passive = (passive_ids >= POLICY_ORDER_ID_START) & (passive_ids < WORLD_ORDER_ID_START) & valid_trades_mask
        is_policy_aggr = (aggr_ids >= POLICY_ORDER_ID_START) & (aggr_ids < WORLD_ORDER_ID_START) & valid_trades_mask
        is_policy_trade = is_policy_passive | is_policy_aggr

        # === DEBUG: Policy trade detection ===
        jax.debug.print(
            "[DEBUG POLICY] K={}, n_policy_trade={}, n_doom={}",
            K, jnp.sum(is_policy_trade), jnp.sum(is_doom_trade)
        )

        # ============================================================
        # COMPUTE REVENUE/COST (Simplified: assume all policy trades match task direction)
        # ============================================================
        # Since trades array doesn't store buy/sell direction explicitly,
        # we assume all policy trades are in the task direction (sell or buy)
        # ============================================================
        is_sell_task = (config.task == 'sell')

        if is_sell_task:
            # Sell task: all policy trades are sells
            sell_revenue = jnp.sum(jnp.where(is_policy_trade, trades[:, 0] * jnp.abs(trades[:, 1]), 0))
            sell_quantity = jnp.sum(jnp.where(is_policy_trade, jnp.abs(trades[:, 1]), 0))
            buy_cost = 0
            buy_quantity = 0
        else:
            # Buy task: all policy trades are buys
            sell_revenue = 0
            sell_quantity = 0
            buy_cost = jnp.sum(jnp.where(is_policy_trade, trades[:, 0] * jnp.abs(trades[:, 1]), 0))
            buy_quantity = jnp.sum(jnp.where(is_policy_trade, jnp.abs(trades[:, 1]), 0))

        # Total agent quantity (either as seller or buyer)
        agent_quantity = sell_quantity + buy_quantity

        # Track doom quantity for monitoring
        doom_quantity = jnp.sum(jnp.where(is_doom_trade, trades[:, 1], 0))

        # Compute PnL based on task type
        # For now, assume sell task (policy wants to sell shares at high prices)
        # PnL = revenue - expected_revenue = revenue - (init_mid_price * quantity)
        # Normalized by 1e6 to keep fitness in reasonable range

        # Expected revenue/cost at mid price
        expected_value = init_mid_price * agent_quantity

        # For sell task: higher revenue = better
        # PnL = (actual_revenue - expected_revenue) / 1e6
        sell_pnl = (sell_revenue - init_mid_price * sell_quantity) / 1e6

        # For buy task: lower cost = better
        # PnL = (expected_cost - actual_cost) / 1e6
        buy_pnl = (init_mid_price * buy_quantity - buy_cost) / 1e6

        # Combined PnL (both sell and buy activities contribute)
        pnl = sell_pnl + buy_pnl

        # Also compute total trade count for monitoring
        total_trades = jnp.sum(valid_trades_mask)
        agent_trades = jnp.sum(is_policy_trade)

        # ============================================================
        # VWAP AND ADVANTAGE CALCULATION (like JaxMARL-HFT)
        # ============================================================
        # VWAP = Volume Weighted Average Price of all OTHER trades (market benchmark)
        # Advantage vs VWAP = how much better we did than market average
        # Advantage vs init_mid = how much better we did than initial price
        # ============================================================

        # Identify other trades (not policy, not doom)
        is_other_trade = valid_trades_mask & ~is_policy_trade & ~is_doom_trade
        n_other_trades = jnp.sum(is_other_trade)

        # Compute VWAP of other trades
        # NOTE: Use absolute value for quantity (like JaxMARL-HFT)
        other_volume = jnp.sum(jnp.where(is_other_trade, jnp.abs(trades[:, 1]), 0))
        other_value = jnp.sum(jnp.where(is_other_trade, trades[:, 0] * jnp.abs(trades[:, 1]), 0))
        vwap = jnp.where(other_volume > 0, other_value / other_volume, init_mid_price)

        # === DEBUG: VWAP calculation ===
        jax.debug.print(
            "[DEBUG VWAP] n_other_trades={}, other_volume={}, other_value={}, vwap={}, init_mid_price={}",
            n_other_trades, other_volume, other_value, vwap, init_mid_price
        )

        # Agent's average execution price
        agent_revenue = sell_revenue + buy_cost  # Total value traded
        agent_avg_price = jnp.where(agent_quantity > 0, agent_revenue / agent_quantity, 0)

        # Direction switch: +1 for sell (want high price), -1 for buy (want low price)
        direction_switch = jnp.where(config.task == 'sell', 1.0, -1.0)

        # Advantage vs VWAP (like JaxMARL-HFT)
        # For sell: advantage = revenue - vwap * quantity (sold higher than market avg)
        # For buy: advantage = vwap * quantity - cost (bought lower than market avg)
        advantage_vwap = direction_switch * (sell_revenue - vwap * sell_quantity + vwap * buy_quantity - buy_cost) / 1e6

        # === DEBUG: Advantage calculation ===
        jax.debug.print(
            "[DEBUG ADVANTAGE] sell_rev={}, sell_qty={}, buy_cost={}, buy_qty={}, vwap*sell_qty={}, advantage_vwap={}",
            sell_revenue, sell_quantity, buy_cost, buy_quantity, vwap * sell_quantity, advantage_vwap
        )

        # Advantage vs init_mid_price (current PnL calculation)
        advantage_init = pnl  # Already computed above

        # ============================================================
        # CONVERT TO BASIS POINTS (bp) for financial interpretation
        # ============================================================
        # 1 bp = 0.01% = 0.0001
        # Advantage (bp) = (price_diff / base_price) × 10000
        #
        # For VWAP advantage:
        #   agent_avg_price vs vwap → (agent_avg_price - vwap) / vwap × 10000 bp
        # For init_mid advantage:
        #   agent_avg_price vs init_mid → (agent_avg_price - init_mid) / init_mid × 10000 bp
        # ============================================================

        # Advantage in bp (vs VWAP)
        # Price difference per unit: (agent_avg_price - vwap)
        # As percentage of vwap: (agent_avg_price - vwap) / vwap
        # In basis points: × 10000
        advantage_vwap_bp = jnp.where(
            (agent_quantity > 0) & (vwap > 0),
            ((agent_avg_price - vwap) / vwap) * 10000,  # bp
            0.0
        )

        # Advantage in bp (vs init_mid_price)
        advantage_init_bp = jnp.where(
            (agent_quantity > 0) & (init_mid_price > 0),
            ((agent_avg_price - init_mid_price) / init_mid_price) * 10000,  # bp
            0.0
        )

        # ============================================================

        # ============================================================
        # TASK COMPLETION PENALTY
        # ============================================================
        # Penalize incomplete execution of task_size
        # completion_ratio = agent_quantity / task_size
        # If completion_ratio < 1, apply penalty proportional to shortfall
        #
        # Penalty formula (like JaxMARL-HFT):
        #   shortfall = task_size - agent_quantity
        #   penalty = -shortfall * init_mid_price / 1e6  (same scale as PnL)
        #
        # This encourages agent to complete the full task
        # ============================================================
        shortfall = jnp.maximum(config.task_size - agent_quantity, 0)
        completion_penalty = -shortfall * init_mid_price / 1e6 * 0.1  # 10% of shortfall value

        # Use PnL as fitness if agent has trades, otherwise penalize
        # Penalty logic:
        #   - If market has trades but agent didn't participate → light penalty (-0.05)
        #   - If market has no trades at all → heavier penalty (-0.1)
        # This encourages agent to actively participate in trading
        base_fitness = jnp.where(
            agent_quantity > 0,
            pnl + completion_penalty,  # PnL + completion penalty
            jnp.where(
                total_trades > 0,
                -0.05,              # Market active but agent didn't trade → light penalty
                -0.1                # Market dead → heavier penalty
            )
        )

        # ============================================================
        # FAULT TOLERANCE: Handle NaN/Inf in fitness
        # ============================================================
        # If fitness is NaN or Inf (shouldn't happen with trade count,
        # but will be critical when using PnL), return 0 as safe fallback.
        # This prevents NaN from propagating through ES gradient updates.
        #
        # Why 0 instead of -inf:
        # - -inf would dominate the ES gradient update
        # - 0 = neutral fitness, episode has no effect on gradients
        # - Better for training stability
        # ============================================================
        fitness = jnp.where(jnp.isfinite(base_fitness), base_fitness, 0.0)

        # ============================================================
        # BUILD INFO DICT FOR LOGGING
        # ============================================================
        info = {
            'fitness': fitness,
            'pnl': pnl,
            # Advantage in dollar units (USD × 1e6 scale)
            'advantage_vwap': advantage_vwap,
            'advantage_init': advantage_init,
            # Advantage in basis points (bp)
            'advantage_vwap_bp': advantage_vwap_bp,
            'advantage_init_bp': advantage_init_bp,
            # Price benchmarks
            'vwap': vwap,
            'init_mid_price': init_mid_price,
            # Execution metrics
            'agent_quantity': agent_quantity,
            'agent_avg_price': agent_avg_price,
            'doom_quantity': doom_quantity,
            'total_trades': total_trades,
            'agent_trades': agent_trades,
            'completion_penalty': completion_penalty,
            'step_counter': final_step_counter,
        }

        # === DEBUG: Episode Summary (split into multiple prints to avoid truncation) ===
        def print_episode_summary():
            jax.debug.print(
                "\n======================================================================",
                ordered=True
            )
            jax.debug.print(
                "EPISODE SUMMARY (Thread 0)",
                ordered=True
            )
            jax.debug.print(
                "======================================================================",
                ordered=True
            )
            jax.debug.print(
                "Execution Status:",
                ordered=True
            )
            jax.debug.print(
                "  Target:           {} shares",
                config.task_size,
                ordered=True
            )
            jax.debug.print(
                "  Executed:         {} shares ({:.1f}%)",
                agent_quantity, (agent_quantity / config.task_size) * 100,
                ordered=True
            )
            jax.debug.print(
                "  Doom (forced):    {} shares",
                doom_quantity,
                ordered=True
            )
            jax.debug.print(
                "  Agent trades:     {}",
                agent_trades,
                ordered=True
            )
            jax.debug.print(
                "  Total trades:     {}",
                total_trades,
                ordered=True
            )
            jax.debug.print(
                "\nPrice Performance:",
                ordered=True
            )
            jax.debug.print(
                "  Init mid price:   {:.2f}",
                init_mid_price / config.tick_size,
                ordered=True
            )
            jax.debug.print(
                "  Agent avg price:  {:.2f}",
                agent_avg_price / config.tick_size,
                ordered=True
            )
            jax.debug.print(
                "  Market VWAP:      {:.2f}",
                vwap / config.tick_size,
                ordered=True
            )
            jax.debug.print(
                "\nAdvantage:",
                ordered=True
            )
            jax.debug.print(
                "  vs VWAP:          {:.2f} bp  ({:.4f} USD @1e6)",
                advantage_vwap_bp, advantage_vwap,
                ordered=True
            )
            jax.debug.print(
                "  vs Init Mid:      {:.2f} bp  ({:.4f} USD @1e6)",
                advantage_init_bp, advantage_init,
                ordered=True
            )
            jax.debug.print(
                "\nMetrics:",
                ordered=True
            )
            jax.debug.print(
                "  PnL:              {:.4f}",
                pnl,
                ordered=True
            )
            jax.debug.print(
                "  Fitness:          {:.6f}",
                fitness,
                ordered=True
            )
            jax.debug.print(
                "  Completion penalty: {:.4f}",
                completion_penalty,
                ordered=True
            )
            jax.debug.print(
                "======================================================================",
                ordered=True
            )

        jax.lax.cond(
            thread_id == 0,
            print_episode_summary,
            lambda: None,
        )

        return fitness, info

    def eval_single_thread(
        self,
        key: jnp.ndarray,
        thread_id: int,
        epoch: int,
        initial_sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """
        Evaluate one perturbed policy on a single episode.

        Args:
            key: JAX random key
            thread_id: Thread ID for noise generation
            epoch: Current epoch
            initial_sim_state: Initial JaxLOB state
            initial_msg_history: (480,) optional initial message history

        Returns:
            (fitness, info_dict) - fitness score and detailed metrics
        """
        world_common_params = self.create_world_common_params()
        policy_common_params = self.create_policy_common_params(epoch, thread_id)

        return self.simulate_episode(
            key, world_common_params, policy_common_params, initial_sim_state, initial_msg_history,
            thread_id=thread_id  # Pass thread_id for debug printing
        )

    def train_epoch(
        self,
        key: jnp.ndarray,
        epoch: int,
        initial_sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
    ) -> Tuple[float, jnp.ndarray, Dict]:
        """
        Run one training epoch.

        Args:
            key: JAX random key
            epoch: Current epoch number
            initial_sim_state: Initial JaxLOB state
            initial_msg_history: (480,) optional initial message history

        Returns:
            (mean_fitness, all_fitnesses, aggregated_info)
        """
        if epoch == 0:
            print(f"[EPOCH] Epoch {epoch}: Starting first epoch (JIT compilation happens here)...")

        n_threads = self.config.n_threads

        # Generate keys for all threads
        keys = jax.random.split(key, n_threads)
        thread_ids = jnp.arange(n_threads)

        # Evaluate all threads in parallel with vmap
        if epoch == 0:
            print(f"[EPOCH] Epoch {epoch}: Creating eval_fn partial...")
        eval_fn = partial(
            self.eval_single_thread,
            epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history,
        )

        # vmap returns (fitnesses, infos) where infos is a dict of arrays
        if epoch == 0:
            print(f"[EPOCH] Epoch {epoch}: Running vmap over {n_threads} threads (JIT compiling)...")
        fitnesses, infos = jax.vmap(eval_fn)(keys, thread_ids)
        if epoch == 0:
            print(f"[EPOCH] Epoch {epoch}: vmap complete, fitnesses shape: {fitnesses.shape}")

        # ES gradient update
        iterinfos = (
            jnp.full(n_threads, epoch, dtype=jnp.int32),
            thread_ids
        )

        # Normalize and update
        if epoch == 0:
            print(f"[EPOCH] Epoch {epoch}: Converting fitnesses...")
        normalized_fitnesses = self.noiser_cls.convert_fitnesses(
            self.frozen_noiser_params, self.noiser_params, fitnesses
        )

        if epoch == 0:
            print(f"[EPOCH] Epoch {epoch}: Running ES gradient update (do_updates)...")
        self.noiser_params, updated_params = self.noiser_cls.do_updates(
            self.frozen_noiser_params,
            self.noiser_params,
            self.lobs5_init.params,
            self.es_tree_key,
            normalized_fitnesses,
            iterinfos,
            self.lobs5_init.es_map,
        )
        # Update only the params, keep the ESInitResult structure
        self.lobs5_init.params = updated_params
        if epoch == 0:
            print(f"[EPOCH] Epoch {epoch}: Params updated")

        # Aggregate info across all threads (mean values)
        aggregated_info = {k: jnp.mean(v) for k, v in infos.items()}

        return jnp.mean(fitnesses), fitnesses, aggregated_info

    def train(self, n_epochs: Optional[int] = None):
        """
        Run full training loop.

        Args:
            n_epochs: Number of epochs (uses config if None)

        Returns:
            Final policy params
        """
        print("[TRAIN] ========================================")
        print("[TRAIN] Starting training loop")
        print("[TRAIN] ========================================")

        n_epochs = n_epochs or self.config.n_epochs
        key = jax.random.PRNGKey(self.config.seed)
        print(f"[TRAIN] n_epochs: {n_epochs}")

        # Initialize W&B if configured
        print("[TRAIN] Step 1: Initializing W&B...")
        wandb_run = None
        if hasattr(self.config, 'wandb_project') and self.config.wandb_project:
            import wandb
            wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"es_jaxlob_n{self.config.n_threads}_s{self.config.seed}",
                config={
                    'n_threads': self.config.n_threads,
                    'n_steps': self.config.n_steps,
                    'noiser': self.config.noiser,
                    'sigma': self.config.sigma,
                    'lr': self.config.lr,
                    'lora_rank': self.config.lora_rank,
                    'checkpoint': self.config.lobs5_checkpoint,
                }
            )
            print(f"[TRAIN] Step 1: W&B initialized: {wandb_run.url}")
        else:
            print("[TRAIN] Step 1: W&B disabled (no project configured)")

        # Get initial JaxLOB state and message history from real data
        print("[TRAIN] Step 2: Creating initial sim state from LOBSTER data...")
        initial_sim_state, initial_msg_history = self._create_initial_sim_state()
        print(f"[TRAIN] Step 2: Initial order book created with {self.sim.nOrders} order slots, {self.sim.nTrades} trade slots")

        print("[TRAIN] ========================================")
        print("[TRAIN] Step 3: Starting epoch loop...")
        print("[TRAIN] ========================================")

        # Training loop
        best_fitness = -float('inf')
        for epoch in tqdm(range(n_epochs), desc='ES JaxLOB Training'):
            key, epoch_key = jax.random.split(key)

            mean_fitness, fitnesses, epoch_info = self.train_epoch(
                epoch_key, epoch, initial_sim_state, initial_msg_history
            )

            if mean_fitness > best_fitness:
                best_fitness = mean_fitness

            # Log to W&B with extended metrics
            if wandb_run:
                # Basic fitness metrics
                fitness_std = float(jnp.std(fitnesses))
                fitness_max = float(jnp.max(fitnesses))
                fitness_min = float(jnp.min(fitnesses))

                # Fitness distribution percentiles
                fitness_sorted = jnp.sort(fitnesses)
                n = len(fitness_sorted)
                p25 = float(fitness_sorted[n // 4])
                p50 = float(fitness_sorted[n // 2])  # median
                p75 = float(fitness_sorted[3 * n // 4])

                # Count of positive/negative fitness (for PnL interpretation)
                n_positive = int(jnp.sum(fitnesses > 0))
                n_negative = int(jnp.sum(fitnesses < 0))
                n_zero = int(jnp.sum(fitnesses == 0))

                wandb_run.log({
                    # Epoch info
                    'epoch': epoch,

                    # Fitness summary
                    'fitness/mean': float(mean_fitness),
                    'fitness/best_ever': float(best_fitness),
                    'fitness/std': fitness_std,
                    'fitness/max': fitness_max,
                    'fitness/min': fitness_min,

                    # Fitness distribution
                    'fitness/p25': p25,
                    'fitness/median': p50,
                    'fitness/p75': p75,

                    # PnL breakdown
                    'pnl/n_positive': n_positive,
                    'pnl/n_negative': n_negative,
                    'pnl/n_zero': n_zero,
                    'pnl/positive_ratio': n_positive / n if n > 0 else 0,

                    # Advantage metrics in USD (× 1e6 scale)
                    'advantage/vs_vwap_usd': float(epoch_info['advantage_vwap']),
                    'advantage/vs_init_mid_usd': float(epoch_info['advantage_init']),
                    # Advantage metrics in basis points (bp)
                    'advantage/vs_vwap_bp': float(epoch_info['advantage_vwap_bp']),
                    'advantage/vs_init_mid_bp': float(epoch_info['advantage_init_bp']),
                    # Price benchmarks (raw units: price × 10000)
                    'advantage/vwap': float(epoch_info['vwap']),
                    'advantage/init_mid_price': float(epoch_info['init_mid_price']),

                    # Execution metrics
                    'execution/agent_quantity': float(epoch_info['agent_quantity']),
                    'execution/agent_avg_price': float(epoch_info['agent_avg_price']),
                    'execution/doom_quantity': float(epoch_info['doom_quantity']),
                    'execution/total_trades': float(epoch_info['total_trades']),
                    'execution/agent_trades': float(epoch_info['agent_trades']),
                    'execution/completion_penalty': float(epoch_info['completion_penalty']),
                    'execution/avg_steps': float(epoch_info['step_counter']),
                })

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: mean_fitness={mean_fitness:.4f}, "
                      f"best={best_fitness:.4f}, std={jnp.std(fitnesses):.4f}")

        if wandb_run:
            wandb_run.finish()

        return self.lobs5_init.params


def es_jaxlob_train(config):
    """
    Main entry point for ES JaxLOB training.

    Args:
        config: Namespace with training configuration
    """
    trainer = ESJaxLOBTrainer(config)
    return trainer.train()


if __name__ == '__main__':
    parser = create_es_jaxlob_config()
    args = parser.parse_args()
    es_jaxlob_train(args)
