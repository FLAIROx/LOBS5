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

[4] Prepare Context for Generation
    ┌─────────────────────────────────────────────┐
    │ msg_history = last 20 messages (480 tokens) │
    │             = messages[480:500] encoded     │
    │                                             │
    │ book_feat = extract_book_features(sim_state)│
    │           = current L2 state (40 values)    │
    └─────────────────────────────────────────────┘

[5] Generation Phase (Step 501+)
    ┌─────────────────────────────────────────────┐
    │ World Model + Policy start generating       │
    │   - Use msg_history as context              │
    │   - Use book_feat as current state          │
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
    msg_decoded = encoding.decode_msg(pred_msg_tokens, encoder)

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
        -88,  # trader_id placeholder
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
    bid_valid = (best_bid > 0) & (best_bid < 1000000)
    ask_valid = (best_ask > 0) & (best_ask < 1000000)

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
        self.config = config
        _lazy_import_jaxlob()

        # Load LOBS5 checkpoint (same for both models)
        print(f"Loading LOBS5 checkpoint from {config.lobs5_checkpoint}")
        self.lobs5_init, self.es_tree_key = load_checkpoint_for_es(
            config.lobs5_checkpoint,
        )

        # Initialize noiser for Policy
        self._init_noiser()

        # Initialize JaxLOB simulator
        self._init_jaxlob()

        print(f"ESJaxLOBTrainer initialized:")
        print(f"  - n_threads: {config.n_threads}")
        print(f"  - n_steps per episode: {config.n_steps}")
        print(f"  - world_msgs_per_step: {config.world_msgs_per_step}")

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
        vocab = Vocab()
        self.encoder = vocab.ENCODING  # Dict[str, Tuple[jax.Array, jax.Array]]

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

        # Data directory (configurable or default to GOOG 2016)
        if hasattr(config, 'data_dir') and config.data_dir:
            data_dir = config.data_dir
        else:
            data_dir = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2016"

        # Find all data files
        orderbook_files = sorted(glob.glob(f"{data_dir}/*orderbook_10_proc.npy"))
        message_files = sorted(glob.glob(f"{data_dir}/*message_10_proc.npy"))

        if len(orderbook_files) == 0:
            raise FileNotFoundError(f"No orderbook files found in {data_dir}")

        # Randomly select a data file (or use config.file_idx if provided)
        if hasattr(config, 'file_idx'):
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
    ) -> float:
        """
        Run a complete episode with step-by-step interleaved simulation.

        Args:
            key: JAX random key
            world_common_params: World Model params (frozen)
            policy_common_params: Policy params (ES perturbed)
            sim_state: Initial JaxLOB state
            initial_msg_history: (480,) optional initial message history from real data

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
        msg_len = 24  # tokens per message

        # Context length: 500 messages to match training (msg_seq_len=500)
        # Why 500:
        #   1. Matches training configuration (run_lobster_padded_large.sh:123)
        #   2. Provides full market history (~several minutes of trading)
        #   3. inference_no_errcorr.py uses parameterized n_inp_msgs (no hardcoded value)
        #      We choose 500 to match the training distribution
        context_len = msg_len * 500  # 500 messages = 12000 tokens

        # Use provided msg_history if available, otherwise zeros
        if initial_msg_history is not None:
            msg_history = initial_msg_history
        else:
            msg_history = jnp.zeros((context_len,), dtype=jnp.int32)

        book_feat = transform_L2_state_wrapper(sim_state, price_levels=500, tick_size=config.tick_size)
        order_id_counter = 500  # Start from 500 to avoid collision with replayed messages

        # ============================================================
        # POLICY ORDER ID TRACKING
        # ============================================================
        # Track initial mid price for PnL calculation
        init_mid_price = get_mid_price(sim_state, config.tick_size)

        # Policy order IDs follow a predictable pattern:
        # Each step: world_msgs_per_step world orders, then 1 policy order
        # So policy order IDs are: K, 2K+1, 3K+2, ... where K = world_msgs_per_step
        # We can compute this at the end instead of tracking explicitly
        # ============================================================

        def step_fn(carry, step_idx):
            """Single step: World Model messages → Policy action."""
            (key, msg_history, hiddens_world, hiddens_policy,
             sim_state, book_feat, order_id) = carry

            key, key_world, key_policy = jax.random.split(key, 3)

            # ====== 1. World Model generates K background messages ======
            def world_msg_step(wcarry, _):
                key, msg_hist, hidden, sim_st, book_f, oid = wcarry
                key, sample_key = jax.random.split(key)

                # World Model forward (no noise)
                hidden, log_probs = ES_PaddedLobPredModel._forward_step(
                    world_common_params, hidden, msg_hist[-msg_len:], book_f[None, :]
                )
                # Keep only the last position's hidden state for next iteration
                hidden = jax.tree.map(lambda h: h[:, -1:, :], hidden)

                # FAULT TOLERANCE: Handle NaN/Inf in log_probs before sampling
                # NaN → -1e9 (very low probability), Inf → clamp to ±1e9
                log_probs = jnp.nan_to_num(log_probs, nan=-1e9, posinf=1e9, neginf=-1e9)

                # Sample next message tokens
                world_msg = jax.random.categorical(sample_key, log_probs[-msg_len:])

                # Convert to JaxLOB format and process
                mid_price = get_mid_price(sim_st, config.tick_size)
                sim_msg, _ = get_sim_msg_es(
                    world_msg, self.sim, sim_st, mid_price, oid, config.tick_size, self.encoder
                )
                sim_st = self.sim.process_order_array(sim_st, sim_msg)

                # Update for next iteration
                book_f = transform_L2_state_wrapper(sim_st)
                msg_hist = jnp.concatenate([msg_hist[msg_len:], world_msg])
                oid = oid + 1

                return (key, msg_hist, hidden, sim_st, book_f, oid), world_msg

            # Generate world_msgs_per_step background messages
            (key_world, msg_history, hiddens_world, sim_state, book_feat, order_id), _ = jax.lax.scan(
                world_msg_step,
                (key_world, msg_history, hiddens_world, sim_state, book_feat, order_id),
                None,
                length=config.world_msgs_per_step,
            )

            # ====== 2. Policy observes and generates action ======
            key_policy, sample_key = jax.random.split(key_policy)

            # Policy forward (with ES noise via iterinfo)
            hiddens_policy, log_probs = ES_PaddedLobPredModel._forward_step(
                policy_common_params, hiddens_policy, msg_history[-msg_len:], book_feat[None, :]
            )
            # Keep only the last position's hidden state for next iteration
            hiddens_policy = jax.tree.map(lambda h: h[:, -1:, :], hiddens_policy)

            # FAULT TOLERANCE: Handle NaN/Inf in log_probs before sampling
            log_probs = jnp.nan_to_num(log_probs, nan=-1e9, posinf=1e9, neginf=-1e9)

            # Sample action tokens
            policy_msg = jax.random.categorical(sample_key, log_probs[-msg_len:])

            # Convert to JaxLOB format and process
            mid_price = get_mid_price(sim_state, config.tick_size)
            sim_msg, _ = get_sim_msg_es(
                policy_msg, self.sim, sim_state, mid_price, order_id, config.tick_size, self.encoder
            )
            sim_state = self.sim.process_order_array(sim_state, sim_msg)

            # Update state for next step
            book_feat = transform_L2_state_wrapper(sim_state)
            msg_history = jnp.concatenate([msg_history[msg_len:], policy_msg])
            order_id = order_id + 1

            return (key, msg_history, hiddens_world, hiddens_policy, sim_state, book_feat, order_id), None

        # Run episode
        (_, _, _, _, final_state, _, final_order_id), _ = jax.lax.scan(
            step_fn,
            (key, msg_history, hiddens_world, hiddens_policy, sim_state, book_feat, order_id_counter),
            jnp.arange(config.n_steps),  # Pass step indices
            length=config.n_steps,
        )

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

        # Policy order IDs follow pattern: K, 2K+1, 3K+2, ...
        # Formula: order_id is a policy order if (order_id - K) % (K + 1) == 0
        # And order_id >= K and order_id < total_orders
        # This is more JAX-friendly than jnp.isin

        # Valid trades mask (price != -1)
        valid_trades_mask = trades[:, 0] != -1

        # Check if trade involves policy as seller (for sell task)
        # trades[:, 3] = seller_order_id
        seller_ids = trades[:, 3]
        # Policy order check: (id - K) % (K+1) == 0 and id >= K
        is_policy_seller_id = ((seller_ids - K) % (K + 1) == 0) & (seller_ids >= K)
        is_policy_seller = is_policy_seller_id & valid_trades_mask

        # Check if trade involves policy as buyer (for buy task)
        # trades[:, 2] = buyer_order_id
        buyer_ids = trades[:, 2]
        is_policy_buyer_id = ((buyer_ids - K) % (K + 1) == 0) & (buyer_ids >= K)
        is_policy_buyer = is_policy_buyer_id & valid_trades_mask

        # Compute metrics for both sell and buy scenarios
        # Sell task: policy is seller, revenue = price * qty
        sell_revenue = jnp.sum(
            jnp.where(is_policy_seller, trades[:, 0] * trades[:, 1], 0)
        )
        sell_quantity = jnp.sum(
            jnp.where(is_policy_seller, trades[:, 1], 0)
        )

        # Buy task: policy is buyer, cost = price * qty
        buy_cost = jnp.sum(
            jnp.where(is_policy_buyer, trades[:, 0] * trades[:, 1], 0)
        )
        buy_quantity = jnp.sum(
            jnp.where(is_policy_buyer, trades[:, 1], 0)
        )

        # Total agent quantity (either as seller or buyer)
        agent_quantity = sell_quantity + buy_quantity

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
        agent_trades = jnp.sum(is_policy_seller | is_policy_buyer)

        # Use PnL as fitness if agent has trades, otherwise penalize
        # Penalty logic:
        #   - If market has trades but agent didn't participate → light penalty (-0.05)
        #   - If market has no trades at all → heavier penalty (-0.1)
        # This encourages agent to actively participate in trading
        fitness = jnp.where(
            agent_quantity > 0,
            pnl,                    # Agent has trades → use PnL
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
        fitness = jnp.where(jnp.isfinite(fitness), fitness, 0.0)

        return fitness

    def eval_single_thread(
        self,
        key: jnp.ndarray,
        thread_id: int,
        epoch: int,
        initial_sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
    ) -> float:
        """
        Evaluate one perturbed policy on a single episode.

        Args:
            key: JAX random key
            thread_id: Thread ID for noise generation
            epoch: Current epoch
            initial_sim_state: Initial JaxLOB state
            initial_msg_history: (480,) optional initial message history

        Returns:
            Fitness score (PnL)
        """
        world_common_params = self.create_world_common_params()
        policy_common_params = self.create_policy_common_params(epoch, thread_id)

        return self.simulate_episode(
            key, world_common_params, policy_common_params, initial_sim_state, initial_msg_history
        )

    def train_epoch(
        self,
        key: jnp.ndarray,
        epoch: int,
        initial_sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
    ) -> Tuple[float, jnp.ndarray]:
        """
        Run one training epoch.

        Args:
            key: JAX random key
            epoch: Current epoch number
            initial_sim_state: Initial JaxLOB state
            initial_msg_history: (480,) optional initial message history

        Returns:
            (mean_fitness, all_fitnesses)
        """
        n_threads = self.config.n_threads

        # Generate keys for all threads
        keys = jax.random.split(key, n_threads)
        thread_ids = jnp.arange(n_threads)

        # Evaluate all threads in parallel with vmap
        eval_fn = partial(
            self.eval_single_thread,
            epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history,
        )

        fitnesses = jax.vmap(eval_fn)(keys, thread_ids)

        # ES gradient update
        iterinfos = (
            jnp.full(n_threads, epoch, dtype=jnp.int32),
            thread_ids
        )

        # Normalize and update
        normalized_fitnesses = self.noiser_cls.convert_fitnesses(
            self.frozen_noiser_params, self.noiser_params, fitnesses
        )

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

        return jnp.mean(fitnesses), fitnesses

    def train(self, n_epochs: Optional[int] = None):
        """
        Run full training loop.

        Args:
            n_epochs: Number of epochs (uses config if None)

        Returns:
            Final policy params
        """
        n_epochs = n_epochs or self.config.n_epochs
        key = jax.random.PRNGKey(self.config.seed)

        # Initialize W&B if configured
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
            print(f"W&B initialized: {wandb_run.url}")

        # Get initial JaxLOB state and message history from real data
        initial_sim_state, initial_msg_history = self._create_initial_sim_state()
        print(f"Loaded initial order book with {self.sim.nOrders} order slots, {self.sim.nTrades} trade slots")
        print(f"Replayed 500 historical messages, msg_history shape: {initial_msg_history.shape}")

        # Training loop
        best_fitness = -float('inf')
        for epoch in tqdm(range(n_epochs), desc='ES JaxLOB Training'):
            key, epoch_key = jax.random.split(key)

            mean_fitness, fitnesses = self.train_epoch(
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
