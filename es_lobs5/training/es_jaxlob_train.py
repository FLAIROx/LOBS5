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

all_noisers = get_all_noisers()

__all__ = ['ESJaxLOBTrainer', 'create_es_jaxlob_config', 'es_jaxlob_train']


def _lazy_import_jaxlob():
    """Lazy import JaxLOB to avoid import errors when not using this mode."""
    global OrderBook, LobState, Message_Tokenizer, encoding
    if OrderBook is None:
        from gymnax_exchange.jaxob.jorderbook import OrderBook as _OrderBook, LobState as _LobState
        from lob.encoding import Message_Tokenizer as _Message_Tokenizer
        import lob.encoding as _encoding
        OrderBook = _OrderBook
        LobState = _LobState
        Message_Tokenizer = _Message_Tokenizer
        encoding = _encoding


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

    # Calculate absolute price
    p_abs = mid_price + rel_price * tick_size

    # Construct JaxLOB message
    # Format: [type, side*2-1, qty, price, order_id, trader_id, time_s, time_ns]
    sim_msg = jnp.array([
        event_type,
        (side * 2) - 1,
        quantity,
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
    """Get current mid price from order book state."""
    # Simplified - should extract from sim_state
    best_bid = sim_state.best_bid if hasattr(sim_state, 'best_bid') else 100000
    best_ask = sim_state.best_ask if hasattr(sim_state, 'best_ask') else 100100
    mid = (best_bid + best_ask) // 2
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
        self.frozen_noiser_params = NOISER.get_frozen_noiser_params(
            self.lobs5_init.params,
            self.lobs5_init.es_map,
            config.sigma,
            lora_rank=config.lora_rank,
        )
        self.noiser_params = NOISER.get_noiser_params(
            self.lobs5_init.params,
            self.lobs5_init.es_map,
            self.frozen_noiser_params,
        )

    def _init_jaxlob(self):
        """Initialize JaxLOB order book simulator."""
        _lazy_import_jaxlob()
        # Create OrderBook instance
        self.sim = OrderBook()
        # Create encoder from Vocab class
        from lob.encoding import Vocab
        vocab = Vocab()
        self.encoder = vocab.ENCODING  # Dict[str, Tuple[jax.Array, jax.Array]]

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

        def step_fn(carry, _):
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
        (_, _, _, _, final_state, _, _), _ = jax.lax.scan(
            step_fn,
            (key, msg_history, hiddens_world, hiddens_policy, sim_state, book_feat, order_id_counter),
            None,
            length=config.n_steps,
        )

        # Return PnL as fitness
        return compute_pnl_fitness(final_state.total_revenue)

    def eval_single_thread(
        self,
        key: jnp.ndarray,
        epoch: int,
        thread_id: int,
        initial_sim_state: 'LobState',
    ) -> float:
        """
        Evaluate one perturbed policy on a single episode.

        Args:
            key: JAX random key
            epoch: Current epoch
            thread_id: Thread ID for noise generation
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

        self.noiser_params, self.lobs5_init = self.noiser_cls.do_updates(
            self.frozen_noiser_params,
            self.noiser_params,
            self.lobs5_init.params,
            self.es_tree_key,
            normalized_fitnesses,
            iterinfos,
            self.lobs5_init.es_map,
        )

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

        # Get initial JaxLOB state
        # In practice, load from your data
        initial_sim_state = self.sim.reset()  # Placeholder

        # Training loop
        best_fitness = -float('inf')
        for epoch in tqdm(range(n_epochs), desc='ES JaxLOB Training'):
            key, epoch_key = jax.random.split(key)

            mean_fitness, fitnesses = self.train_epoch(
                epoch_key, epoch, initial_sim_state
            )

            if mean_fitness > best_fitness:
                best_fitness = mean_fitness

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: mean_fitness={mean_fitness:.4f}, "
                      f"best={best_fitness:.4f}, std={jnp.std(fitnesses):.4f}")

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
