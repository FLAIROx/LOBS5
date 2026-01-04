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
"""

import jax
import jax.numpy as jnp
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
    parser.add_argument('--n_threads', type=int, default=128, help='Population size')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--n_steps', type=int, default=100, help='Steps per episode')
    parser.add_argument('--world_msgs_per_step', type=int, default=10,
                        help='Background messages per step')

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
    sim_state: 'LobState',
    price_levels: int = 500,
    tick_size: int = 100,
) -> jnp.ndarray:
    """
    Convert JaxLOB sim_state to model book input.

    Args:
        sim_state: JaxLOB LobState
        price_levels: Volume image size (default 500)
        tick_size: Tick size in cents

    Returns:
        book_feat: (503,) = [mid_diff, time_s_norm, time_ns_norm, volume_image(500)]
    """
    _lazy_import_jaxlob()
    from gymnax_exchange.jaxob.JaxOrderBookArrays import get_L2_state
    from preproc import transform_L2_state

    # Extract L2 from JaxLOB
    l2_state = get_L2_state(sim_state.asks, sim_state.bids, 10)
    l2_state = jnp.asarray(l2_state, dtype=jnp.int32)

    # Construct (43,) input
    metadata = jnp.array([0, 34200, 0], dtype=jnp.int32)
    book_input = jnp.concatenate([metadata, l2_state])

    # Apply training transform
    book_input_batched = book_input[None, :]
    book_feat_batched = transform_L2_state(book_input_batched, price_levels, tick_size)
    return book_feat_batched[0]


def get_mid_price(sim_state: 'LobState', tick_size: int = 100) -> int:
    """Get current mid price from order book state."""
    _lazy_import_jaxlob()
    DEFAULT_MID = 10000

    best_ask, best_bid = get_best_bid_and_ask(sim_state.asks, sim_state.bids)

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
        print(f"[INIT]   n_threads: {config.n_threads}")
        print(f"[INIT]   n_steps: {config.n_steps}")
        print(f"[INIT]   background_mode: {config.background_mode}")

    def _init_noiser(self):
        """Initialize EGGROLL noiser for Policy."""
        config = self.config
        all_noisers = _get_all_noisers()
        NOISER = all_noisers[config.noiser]

        self.noiser_cls = NOISER
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

        expected_orders = 500 + self.config.n_steps * (self.config.world_msgs_per_step + 1)
        n_orders = max(1000, int(expected_orders * 1.5))
        n_trades = max(500, self.config.n_steps * 2)

        self.sim = OrderBook(nOrders=n_orders, nTrades=n_trades)

        # Create encoder from Vocab
        from lob.encoding import Vocab
        vocab = Vocab(token_mode=self.config.token_mode)
        self.encoder = vocab.ENCODING
        print(f"[INIT] Using token_mode={self.config.token_mode}")

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

        file_idx = np.random.randint(0, len(message_files))
        selected_file = message_files[file_idx]

        self.replay_data_date = os.path.basename(selected_file).split('_')[1]
        self.replay_data_dir = data_path

        msg_raw = np.load(selected_file)
        print(f"[INIT] Loaded {msg_raw.shape[0]} messages from {os.path.basename(selected_file)}")

        # Pre-encode all messages
        self.replay_tokens = encode_msgs(msg_raw, self.encoder, token_mode=self.config.token_mode)
        self.replay_data_raw = jnp.array(msg_raw)

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

        print(f"Loading data from {orderbook_files[file_idx]}")

        # Load data
        ob = np.load(orderbook_files[file_idx])
        msg = np.load(message_files[file_idx])

        # Initialize L2 book
        init_l2_book = jnp.array(ob[0, 3:43], dtype=jnp.int32)
        sim_state = self.sim.reset(init_l2_book)

        # Replay first 500 messages
        n_replay = min(500, len(msg))
        replay_msgs_raw = msg[:n_replay]
        replay_jaxlob = msgs_to_jnp(replay_msgs_raw)
        sim_state = self.sim.process_orders_array(sim_state, replay_jaxlob)

        # Encode messages as context
        tokens = encode_msgs(replay_msgs_raw, self.encoder, token_mode=self.config.token_mode)
        msg_history = tokens.flatten()

        print(f"  Initialized with {n_replay} messages, context size: {msg_history.shape}")

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

    def simulate_episode(
        self,
        key: jnp.ndarray,
        world_common_params,
        policy_common_params,
        sim_state: 'LobState',
        initial_msg_history: Optional[jnp.ndarray] = None,
        thread_id: int = -1,
    ) -> Tuple[float, Dict]:
        """
        Run a complete episode with step-by-step interleaved simulation.

        Returns:
            (fitness, info_dict)
        """
        config = self.config
        fp = self.lobs5_init.frozen_params

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
        sim_state = sim_state._replace(
            trades=(jnp.ones((sim_state.trades.shape[0], 6)) * -1).astype(jnp.int32)
        )

        init_mid_price = get_mid_price(sim_state, config.tick_size)

        # Initialize message history
        if initial_msg_history is not None:
            msg_history = initial_msg_history
        else:
            msg_history = jnp.zeros((context_len,), dtype=jnp.int32)

        book_depth = fp.get('book_depth', 500)
        book_feat = transform_L2_state_wrapper(sim_state, price_levels=book_depth, tick_size=config.tick_size)

        task_size = jnp.int32(config.task_size)

        def step_fn(carry, step_idx):
            """Single step: Background messages -> Policy action."""
            (key, msg_history, hiddens_world, hiddens_policy,
             sim_state, book_feat, world_oid_offset, quant_executed) = carry

            key, key_world, key_policy = jax.random.split(key, 3)

            # Background message generation
            def historical_replay_step(wcarry, bg_msg_idx):
                """Load pre-encoded messages from historical data."""
                key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr = wcarry

                replayed_msg_tokens = self.replay_tokens[replay_ptr]
                replayed_msg_raw = self.replay_data_raw[replay_ptr]

                sim_msg = decoded_msg_to_jaxlob_format(replayed_msg_raw)
                bg_order_id = WORLD_ORDER_ID_START + oid_offset
                sim_msg = sim_msg.at[4].set(bg_order_id)
                sim_msg = sim_msg.at[5].set(-2000)

                sim_st = self.sim.process_order_array(sim_st, sim_msg)
                book_f = transform_L2_state_wrapper(sim_st, price_levels=book_depth, tick_size=config.tick_size)
                msg_hist = jnp.concatenate([msg_hist[msg_len:], replayed_msg_tokens])

                new_replay_ptr = replay_ptr + 1
                n_replay_msgs = self.replay_tokens.shape[0]
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
                (key, msg_hist, hidden), world_msg = jax.lax.scan(
                    sample_one_token,
                    (sample_key, msg_hist, hidden),
                    None,
                    length=msg_len,
                )

                # Convert to JaxLOB format
                mid_price = get_mid_price(sim_st, config.tick_size)
                world_order_id = WORLD_ORDER_ID_START + oid_offset
                sim_msg, _ = get_sim_msg_es(
                    world_msg, self.sim, sim_st, mid_price, world_order_id, config.tick_size, self.encoder,
                    trader_id=-2000, token_mode=config.token_mode
                )

                sim_st = self.sim.process_order_array(sim_st, sim_msg)
                book_f = transform_L2_state_wrapper(sim_st, price_levels=book_depth, tick_size=config.tick_size)
                msg_hist = jnp.concatenate([msg_hist[msg_len:], world_msg])
                oid_offset = oid_offset + 1

                return (key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr), world_msg

            # Select background generation function
            if config.background_mode == 'historical_replay':
                step_fn_background = historical_replay_step
                replay_ptr_init = jnp.int32(500 + step_idx * config.world_msgs_per_step)
            else:
                step_fn_background = world_model_step
                replay_ptr_init = jnp.int32(0)

            # Generate background messages
            (key_world, msg_history, hiddens_world, sim_state, book_feat,
             world_oid_offset, _), _ = jax.lax.scan(
                step_fn_background,
                (key_world, msg_history, hiddens_world, sim_state, book_feat, world_oid_offset, replay_ptr_init),
                jnp.arange(config.world_msgs_per_step),
                length=config.world_msgs_per_step,
            )

            # Policy generates action
            def sample_policy_token(token_carry, _):
                key_p, msg_hist_p, hidden_p = token_carry
                key_p, sample_key_p = jax.random.split(key_p)

                hidden_p, log_probs_p = ES_PaddedLobPredModel._forward_step(
                    policy_common_params, hidden_p, msg_hist_p[-msg_len:], book_feat[None, :]
                )
                hidden_p = jax.tree.map(lambda h: h[:, -1:, :], hidden_p)

                log_probs_p = jnp.nan_to_num(log_probs_p, nan=-1e9, posinf=1e9, neginf=-1e9)
                next_token_p = jax.random.categorical(sample_key_p, log_probs_p[-1])

                msg_hist_p = jnp.concatenate([msg_hist_p[1:], jnp.array([next_token_p])])

                return (key_p, msg_hist_p, hidden_p), next_token_p

            key_policy, sample_key = jax.random.split(key_policy)
            (key_policy, msg_history, hiddens_policy), policy_msg = jax.lax.scan(
                sample_policy_token,
                (sample_key, msg_history, hiddens_policy),
                None,
                length=msg_len,
            )

            # Convert to JaxLOB format
            mid_price = get_mid_price(sim_state, config.tick_size)
            policy_order_id = POLICY_ORDER_ID_START + step_idx
            sim_msg, msg_decoded = get_sim_msg_es(
                policy_msg, self.sim, sim_state, mid_price, policy_order_id, config.tick_size, self.encoder,
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
            sim_state = self.sim.process_order_array(sim_state, sim_msg)

            # Track execution
            trades = sim_state.trades
            is_new_trade = (trades[:, 0] != -1)
            is_policy_in_trade = ((trades[:, 2] == policy_order_id) | (trades[:, 3] == policy_order_id)) & is_new_trade
            step_executed = jnp.sum(jnp.where(is_policy_in_trade, jnp.abs(trades[:, 1]), 0))
            quant_executed = quant_executed + step_executed

            # Update state
            book_feat = transform_L2_state_wrapper(sim_state, price_levels=book_depth, tick_size=config.tick_size)
            msg_history = jnp.concatenate([msg_history[msg_len:], policy_msg])

            return (key, msg_history, hiddens_world, hiddens_policy, sim_state,
                    book_feat, world_oid_offset, quant_executed), None

        # Run episode
        (_, _, _, _, final_state, _, _, final_quant_executed), _ = jax.lax.scan(
            step_fn,
            (key, msg_history, hiddens_world, hiddens_policy, sim_state,
             book_feat, jnp.int32(0), jnp.int32(0)),
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
            pnl = (sell_revenue - init_mid_price * sell_quantity) / 1e6
            agent_quantity = sell_quantity
        else:
            buy_cost = jnp.sum(jnp.where(is_policy_trade, trades[:, 0] * jnp.abs(trades[:, 1]), 0))
            buy_quantity = jnp.sum(jnp.where(is_policy_trade, jnp.abs(trades[:, 1]), 0))
            pnl = (init_mid_price * buy_quantity - buy_cost) / 1e6
            agent_quantity = buy_quantity

        # Completion penalty
        shortfall = jnp.maximum(config.task_size - agent_quantity, 0)
        completion_penalty = -shortfall * init_mid_price / 1e6 * 0.1

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
            'pnl': pnl,
            'agent_quantity': agent_quantity,
            'agent_trades': agent_trades,
            'total_trades': total_trades,
            'completion_penalty': completion_penalty,
            'init_mid_price': init_mid_price,
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
        n_threads = self.config.n_threads

        # Generate keys for all threads
        keys = jax.random.split(key, n_threads)
        thread_ids = jnp.arange(n_threads)

        # Evaluate all threads in parallel
        eval_fn = partial(
            self.eval_single_thread,
            epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history,
        )

        fitnesses, infos = jax.vmap(eval_fn)(keys, thread_ids)

        # ES gradient update
        iterinfos = (
            jnp.full(n_threads, epoch, dtype=jnp.int32),
            thread_ids
        )

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
        self.lobs5_init.params = updated_params

        aggregated_info = {k: jnp.mean(v) for k, v in infos.items()}

        return jnp.mean(fitnesses), fitnesses, aggregated_info

    def train(self, n_epochs: Optional[int] = None):
        """Run full training loop."""
        print("[TRAIN] Starting training loop")

        n_epochs = n_epochs or self.config.n_epochs
        key = jax.random.PRNGKey(self.config.seed)

        # Initialize W&B
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
                    'background_mode': self.config.background_mode,
                }
            )
            print(f"[TRAIN] W&B initialized: {wandb_run.url}")

        # Get initial state
        initial_sim_state, initial_msg_history = self._create_initial_sim_state()

        # Training loop
        best_fitness = -float('inf')
        for epoch in tqdm(range(n_epochs), desc='ES Training'):
            key, epoch_key = jax.random.split(key)

            mean_fitness, fitnesses, epoch_info = self.train_epoch(
                epoch_key, epoch, initial_sim_state, initial_msg_history
            )

            if mean_fitness > best_fitness:
                best_fitness = mean_fitness

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

        if wandb_run:
            wandb_run.finish()

        return self.lobs5_init.params

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
