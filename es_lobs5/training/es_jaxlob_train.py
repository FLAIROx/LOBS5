"""
ES Training with JaxLOB (Pure ES, Step-by-Step Interleaved).

This module implements Evolution Strategies training for trading policies
using JaxLOB as the execution environment.

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
- Fitness = total_revenue (PnL)
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


def extract_book_features(sim_state: 'LobState', l2_depth: int = 10) -> jnp.ndarray:
    """
    Extract book features from JaxLOB state for model input.

    Args:
        sim_state: JaxLOB LobState
        l2_depth: Number of price levels to include

    Returns:
        book_feat: (d_book,) book feature vector
    """
    # Extract L2 book state (simplified)
    # This should match the book feature format expected by the model
    # The actual implementation depends on your data preprocessing

    # Placeholder - extract bid/ask prices and quantities
    # In practice, this should match your LOBS5 book encoding
    return jnp.zeros((503,), dtype=jnp.float32)  # d_book default


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
        # Each episode: n_steps * (world_msgs + 1) messages
        # Conservative: allow 2x trades for safety
        n_orders = max(200, self.config.n_steps * 2)
        n_trades = max(200, self.config.n_steps * 2)
        self.sim = OrderBook(nOrders=n_orders, nTrades=n_trades)
        # Create encoder from Vocab class
        from lob.encoding import Vocab
        vocab = Vocab()
        self.encoder = vocab.ENCODING  # Dict[str, Tuple[jax.Array, jax.Array]]

    def _create_initial_sim_state(self) -> 'LobState':
        """
        Create initial JaxLOB state with a realistic order book.

        If config.init_book_path is provided, load from file.
        Otherwise, create a synthetic L2 book with:
        - Mid price: 10000 (in cents, = $100.00)
        - Spread: 100 (= 1 tick = $1.00)
        - 10 levels on each side
        - Decreasing quantity at each level

        Returns:
            LobState with initialized order book
        """
        config = self.config

        if hasattr(config, 'init_book_path') and config.init_book_path:
            # Load from file
            import numpy as np
            l2_data = np.load(config.init_book_path)
            return self.sim.reset(jnp.array(l2_data))

        # Create synthetic L2 book
        n_levels = 10
        mid_price = 10000  # $100.00 in cents
        tick_size = config.tick_size  # Usually 100 cents = $1.00
        base_qty = 100  # Base quantity at best price

        # Build L2 data: format expected by init_msgs_from_l2
        # The format is: [price, qty] pairs, interleaved asks and bids
        # asks: price ascending from mid + 1 tick
        # bids: price descending from mid - 1 tick
        l2_data = []
        for i in range(n_levels):
            ask_price = mid_price + (i + 1) * tick_size
            bid_price = mid_price - (i + 1) * tick_size
            # Quantity decreases with distance from mid
            qty = max(10, base_qty - i * 10)
            l2_data.extend([ask_price, qty, bid_price, qty])

        l2_book = jnp.array(l2_data, dtype=jnp.int32)
        return self.sim.reset(l2_book)

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
    ) -> float:
        """
        Run a complete episode with step-by-step interleaved simulation.

        Args:
            key: JAX random key
            world_common_params: World Model params (frozen)
            policy_common_params: Policy params (ES perturbed)
            sim_state: Initial JaxLOB state

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
        context_len = msg_len * 20  # 20 messages context
        msg_history = jnp.zeros((context_len,), dtype=jnp.int32)
        book_feat = extract_book_features(sim_state)
        order_id_counter = 0

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
                book_f = extract_book_features(sim_st)
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
            book_feat = extract_book_features(sim_state)
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

        # Use PnL as fitness if agent has trades, otherwise use total trade count as fallback
        # If total_trades > 0 but agent_trades == 0, use total_trades as proxy
        # This handles the case where the order book is matching but agent orders aren't hitting
        fitness = jnp.where(
            agent_quantity > 0,
            pnl,
            jnp.where(
                total_trades > 0,
                jnp.float32(total_trades) / 100.0,  # Normalized trade count as fallback
                -0.1  # Small penalty if no trades at all
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
    ) -> float:
        """
        Evaluate one perturbed policy on a single episode.

        Args:
            key: JAX random key
            thread_id: Thread ID for noise generation
            epoch: Current epoch
            initial_sim_state: Initial JaxLOB state

        Returns:
            Fitness score (PnL)
        """
        world_common_params = self.create_world_common_params()
        policy_common_params = self.create_policy_common_params(epoch, thread_id)

        return self.simulate_episode(
            key, world_common_params, policy_common_params, initial_sim_state
        )

    def train_epoch(
        self,
        key: jnp.ndarray,
        epoch: int,
        initial_sim_state: 'LobState',
    ) -> Tuple[float, jnp.ndarray]:
        """
        Run one training epoch.

        Args:
            key: JAX random key
            epoch: Current epoch number
            initial_sim_state: Initial JaxLOB state

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

        # Get initial JaxLOB state with realistic order book
        initial_sim_state = self._create_initial_sim_state()
        print(f"Initial order book created with {self.sim.nOrders} order slots, {self.sim.nTrades} trade slots")

        # Training loop
        best_fitness = -float('inf')
        for epoch in tqdm(range(n_epochs), desc='ES JaxLOB Training'):
            key, epoch_key = jax.random.split(key)

            mean_fitness, fitnesses = self.train_epoch(
                epoch_key, epoch, initial_sim_state
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
