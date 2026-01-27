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

# =============================================================================
# S5 SSM Parameter Training Strategy
# =============================================================================
#
# Default Behavior (freeze_nonlora=False):
# ┌──────────────┬───────────┬───────────────┬───────────────────────────────┐
# │  Parameter   │  Status   │ Training Mode │             Notes             │
# ├──────────────┼───────────┼───────────────┼───────────────────────────────┤
# │ Lambda_re/im │ Frozen    │ -             │ Always frozen (for stability) │
# │ embedding    │ Frozen    │ -             │ Always frozen                 │
# │ decoder      │ Frozen    │ -             │ Always frozen                 │
# ├──────────────┼───────────┼───────────────┼───────────────────────────────┤
# │ log_step     │ Trainable │ FULL          │ Time scale adaptation         │
# │ B, C         │ Trainable │ FULL          │ Input/output projection       │
# │ D            │ Trainable │ FULL          │ Skip/direct channel           │
# │ norm, bias   │ Trainable │ FULL          │ Normalization layers          │
# ├──────────────┼───────────┼───────────────┼───────────────────────────────┤
# │ out2/weight  │ Trainable │ LoRA          │ GLU output layer              │
# └──────────────┴───────────┴───────────────┴───────────────────────────────┘
#
# Summary:
# - Frozen: Core stability components (Lambda eigenvalues, embedding, decoder)
# - FULL training: SSM dynamics (B, C, D, log_step) and normalization layers
# - LoRA training: Only the GLU output layer uses Low-Rank Adaptation
#
# To use conservative mode (LoRA-only): --freeze_nonlora True
# =============================================================================

# =============================================================================
# PERGPU_PERTURBATIONS Scaling Test Results (2026-01-17):
# ------------------------------------------------------
# Max Stable:  14,336 (Total 57,344) - Job 1921017 - RUNNING
# First Fail:  16,384 (Total 65,536) - Job 1920937 - FAILED (OOM/Aborted)
#
# Jobs 18,432+ all fail with OOM (RESOURCE_EXHAUSTED ~48-80GB allocation)
# The "Aborted" status indicates XLA runtime forced abort to prevent deadlock
# after one replica hit OOM during distributed computation.
#
# Related files:
#   - es_lobs5/scripts/es_training.sh     (batch script)
#   - es_lobs5/scripts/es_training.py     (entry point)
#
# Reference logs:
#   - logs/es_train_1921017.out/.err  (max stable run)
#   - logs/es_train_1920937.out/.err  (first OOM failure)
#   - logs/es_train_1921018.out/.err  (detailed OOM traceback)
# =============================================================================

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
from jax.experimental.multihost_utils import process_allgather
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
create_trade = None
add_trade = None
_all_noisers = None
_ES_PaddedLobPredModel = None

# Flax inference globals (lazy loaded)
_flax_init_train_state = None
_flax_load_checkpoint = None
_flax_load_metadata = None
_CommonParams = None
_simple_es_tree_key = None
_load_checkpoint_for_es = None

# Training mode configuration (lazy loaded)
_training_modes = None


def _get_training_modes():
    """Lazy load training modes configuration."""
    global _training_modes
    if _training_modes is None:
        from .training_modes import (
            get_mode_config, get_param_classifier, apply_es_map_classification,
            print_mode_config, is_lora_mode, get_all_modes
        )
        _training_modes = {
            'get_mode_config': get_mode_config,
            'get_param_classifier': get_param_classifier,
            'apply_es_map_classification': apply_es_map_classification,
            'print_mode_config': print_mode_config,
            'is_lora_mode': is_lora_mode,
            'get_all_modes': get_all_modes,
        }
    return _training_modes


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
    global OrderBook, LobState, Message_Tokenizer, encoding, get_best_bid_and_ask, create_trade, add_trade, getCancelMsgs
    if OrderBook is None:
        from gymnax_exchange.jaxob.jorderbook import OrderBook as _OrderBook, LobState as _LobState
        from gymnax_exchange.jaxob.JaxOrderBookArrays import (
            get_best_bid_and_ask as _get_best_bid_and_ask,
            create_trade as _create_trade,
            add_trade as _add_trade,
            getCancelMsgs as _getCancelMsgs,
        )
        from lob.encoding import Message_Tokenizer as _Message_Tokenizer
        import lob.encoding as _encoding
        OrderBook = _OrderBook
        LobState = _LobState
        Message_Tokenizer = _Message_Tokenizer
        encoding = _encoding
        get_best_bid_and_ask = _get_best_bid_and_ask
        create_trade = _create_trade
        add_trade = _add_trade
        getCancelMsgs = _getCancelMsgs


# =============================================================================
# Helper Functions for Smart Cancellation (Ported from Gymnax Exchange)
# =============================================================================
from functools import partial

@jax.jit
def p_in_cnl_vmap(p, prices_cnl):
    return jnp.where((prices_cnl == p) & (p != 0), True, False)

def matching_masks(prices_a, prices_cnl):
    # Vectorized match check
    p_in_cnl_fn = jax.vmap(p_in_cnl_vmap, in_axes=(0, None))
    res = p_in_cnl_fn(prices_a, prices_cnl)
    return jnp.any(res, axis=1), jnp.any(res, axis=0)

def argsort_rev(arr):
    """ 'arr' sorted in descending order (LTR priority tie-breaker) """
    return (arr.shape[0] - 1 - jnp.argsort(arr[::-1]))[::-1]

def rank_rev(arr):
    """ Rank array in descending order, with ties having left-to-right priority. """
    return jnp.argsort(argsort_rev(arr))

def _filter_messages(action_msgs: jax.Array, cnl_msgs: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """ Filter out cancelation messages, when same actions should be placed again.
        NOTE: only simplifies cancellations if new action size <= old action size.
        To prevent multiple split orders, new larger orders still cancel the entire old order.
    """
    a_mask, c_mask = matching_masks(action_msgs[:, 3], cnl_msgs[:, 3])
    
    a_i = jnp.where(a_mask, size=a_mask.shape[0], fill_value=-1)[0]
    a = jnp.where(a_i == -1, 0, action_msgs[a_i][:, 2])
    c_i = jnp.where(c_mask, size=c_mask.shape[0], fill_value=-1)[0]
    c = jnp.where(c_i == -1, 0, cnl_msgs[c_i][:, 2])

    rel_cnl_quants = (c >= a) * a
    
    # Update action messages
    # If action size matches cancel size (or is smaller), we reduce action.
    # If action becomes 0, it means we don't need to send it (and don't need to cancel).
    action_msgs = action_msgs.at[:, 2].set(
        action_msgs[:, 2] - rel_cnl_quants[rank_rev(a_mask)])
        
    # Set actions with 0 quant to dummy messages (so process_orders skips/NOOPs them)
    # Note: NOOP in JaxLOB is often type 0 or quantity 0.
    # We ensure they are effectively NOOP.
    action_msgs = jnp.where(
        (action_msgs[:, 2] == 0)[:, None], # Broadcast
        0,
        action_msgs,
    )
    
    # Update cancel messages
    cnl_msgs = cnl_msgs.at[:, 2].set(cnl_msgs[:, 2] - rel_cnl_quants[rank_rev(c_mask)])
    cnl_msgs = jnp.where(
        (cnl_msgs[:, 2] == 0)[:, None],
        0,
        cnl_msgs,
    )
    return action_msgs, cnl_msgs


def _get_flax_loaders():
    """Lazy load Flax model initialization functions (same as run_inference.py)."""
    global _flax_init_train_state, _flax_load_checkpoint, _flax_load_metadata
    if _flax_init_train_state is None:
        from lob.init_train import init_train_state, load_checkpoint, load_metadata
        _flax_init_train_state = init_train_state
        _flax_load_checkpoint = load_checkpoint
        _flax_load_metadata = load_metadata
    return _flax_init_train_state, _flax_load_checkpoint, _flax_load_metadata


# ============================================================================
# Import shared utilities from lob/ for code reuse
# ============================================================================
from lob.validation_helpers import syntax_validation_matrix
from lob.encoding import Vocab
# Import lightweight message utils (avoids heavy gymnax_exchange imports)
from lob.message_utils import (
    msg_to_jnp,              # Replaces decoded_msg_to_jaxlob_format
    msgs_to_jnp,             # Replaces vmap version
    construct_sim_msg,       # Replaces inline construction in get_sim_msg_es
    construct_dummy_sim_msg, # NOOP message fallback
    ORDER_ID_i, EVENT_TYPE_i, DIRECTION_i, SIZE_i, TIMEs_i, TIMEns_i,  # Field indices
)

# ============================================================================
# Import inference module for code reuse (run_inference.py code path)
# This ensures ESTrainer uses the SAME data loading, encoding, and generation
# code as the validated run_inference.py
# ============================================================================
from lob import inference_no_errcorr as inference
from lob.lobster_dataloader import LOBSTER_Dataset

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

# DEPRECATED: This is replaced by syntax_validation_matrix from lob/validation_helpers.py
# which now correctly supports token_mode=24 with the get_encoder_key() fix.
# COMMENTED OUT to prevent accidental usage - use get_field_masks_from_validation_matrix() instead.
# Position -> (field_name, token_min, token_max) for 24-token messages
# POSITION_TOKEN_RANGES_24 = {
#     0: ("event_type", 1004, 1007),
#     1: ("direction", 2110, 2111),
#     2: ("price_sign", 2108, 2109),
#     3: ("price", 1108, 2107),
#     4: ("size_high", 1008, 1107),
#     5: ("size_low", 1008, 1107),
#     6: ("delta_t_s", 4, 1003),
#     7: ("delta_t_ns_0", 4, 1003),
#     8: ("delta_t_ns_1", 4, 1003),
#     9: ("delta_t_ns_2", 4, 1003),
#     10: ("time_s_0", 4, 1003),
#     11: ("time_s_1", 4, 1003),
#     12: ("time_ns_0", 4, 1003),
#     13: ("time_ns_1", 4, 1003),
#     14: ("time_ns_2", 4, 1003),
#     15: ("price_ref_sign", 2108, 2109),
#     16: ("price_ref", 1108, 2107),
#     17: ("size_ref_high", 1008, 1107),
#     18: ("size_ref_low", 1008, 1107),
#     19: ("time_s_ref_0", 4, 1003),
#     20: ("time_s_ref_1", 4, 1003),
#     21: ("time_ns_ref_0", 4, 1003),
#     22: ("time_ns_ref_1", 4, 1003),
#     23: ("time_ns_ref_2", 4, 1003),
# }

# _FIELD_MASKS_24 = None  # COMMENTED OUT - no longer needed

# DEPRECATED: Use get_field_masks_from_validation_matrix() instead.
# COMMENTED OUT to prevent accidental usage.
# def get_field_masks_24(vocab_size: int = 2112):
#     """Get field masks for constrained decoding (additive mask format).
#
#     Creates masks directly from POSITION_TOKEN_RANGES_24, avoiding
#     syntax_validation_matrix which has compatibility issues with 24-token mode.
#
#     Returns:
#         jnp.array of shape (24, vocab_size) where:
#         - 0.0 for valid tokens
#         - -1e9 for invalid tokens
#     """
#     global _FIELD_MASKS_24
#     if _FIELD_MASKS_24 is None:
#         masks = []
#         for pos in range(24):
#             _, tok_min, tok_max = POSITION_TOKEN_RANGES_24[pos]
#             # Start with -1e9 (invalid) for all tokens
#             mask = jnp.full(vocab_size, -1e9)
#             # Set valid range to 0.0
#             mask = mask.at[tok_min:tok_max+1].set(0.0)
#             masks.append(mask)
#         _FIELD_MASKS_24 = jnp.stack(masks)
#     return _FIELD_MASKS_24


def get_field_masks_from_validation_matrix(token_mode: int, vocab_size: int):
    """Create field masks by converting syntax_validation_matrix to additive format.

    This function uses the unified validation logic from lob/validation_helpers.py,
    ensuring that constraint fixes automatically propagate from LOB inference to ES training.

    Args:
        token_mode: 22 or 24
        vocab_size: Vocabulary size (12012 for token_mode=22, 2112 for token_mode=24)

    Returns:
        Additive masks of shape (MSG_LEN, vocab_size): 0.0=valid, -1e9=invalid
    """
    from lob.validation_helpers import syntax_validation_matrix
    from lob.encoding import Vocab

    v = Vocab(token_mode=token_mode)
    bool_mask = syntax_validation_matrix(v)  # (MSG_LEN, vocab_size), True=valid

    # Convert: True → 0.0 (valid), False → -1e9 (invalid)
    additive_mask = jnp.where(bool_mask, 0.0, -1e9)

    return additive_mask


def rank_transform(x: jax.Array) -> jax.Array:
    """Convert fitness values to ranks, normalized to [-0.5, 0.5].

    This prevents outliers from dominating and gives minority good
    strategies a relatively larger weight in gradient estimation.
    Particularly useful when most perturbations converge to a local
    optimum (e.g., "no trading") while a few find better strategies.

    Args:
        x: Fitness array of shape (n_perturbations,)

    Returns:
        Rank-transformed array in [-0.5, 0.5] range

    Reference:
        HyperscaleES/rl_experiments/eggroll.py:102-104
    """
    ranks = jax.scipy.stats.rankdata(x, axis=-1) - 1.0
    return ranks / (x.shape[-1] - 1) - 0.5


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_es_config():
    """Create argument parser for ES training configuration."""
    parser = argparse.ArgumentParser(description='ES JaxLOB Training for LOBS5')

    # LOBS5 checkpoint (for both World Model and Policy initialization)
    parser.add_argument('--lobs5_checkpoint', type=str, required=True,
                        help='Path to LOBS5 checkpoint for model initialization')

    # ES configuration
    parser.add_argument('--noiser', type=str, default='eggroll',
                        choices=['open_es', 'eggroll', 'eggrollbs', 'sparse'])
    parser.add_argument('--sigma', type=float, default=0.2, help='Initial noise std (default: 0.2 for exploration)')
    parser.add_argument('--sigma_decay', type=float, default=0.9997,
                        help='Sigma decay rate per epoch (0.9997: 0.2->0.01 over 10k epochs). Default: 0.9997')
    parser.add_argument('--sigma_min', type=float, default=0.01,
                        help='Minimum sigma (floor). Default: 0.01')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=4, help='LORA rank')

    # Training mode (replaces use_lora, freeze_nonlora, lora_v2, freeze_ssm)
    parser.add_argument('--mode', type=str, default='LORA_V1.6',
                        choices=['LORA', 'LORA_V1.5', 'LORA_V1.6', 'LORA_V2', 'FULL', 'LORA+SSM'],
                        help='Training mode: LORA (out2 only), LORA_V1.5 (all proj, train norms), '
                             'LORA_V1.6 (all proj, freeze norms), LORA_V2 (all proj + SSM), '
                             'FULL (no LoRA). Default: LORA_V1.6')

    # [DEPRECATED] Legacy boolean flags - kept for backwards compatibility
    # These are now ignored; use --mode instead
    parser.add_argument('--use_lora', type=str2bool, default=None,
                        help='[DEPRECATED] Use --mode instead. This flag is ignored.')
    parser.add_argument('--freeze_nonlora', type=str2bool, default=None,
                        help='[DEPRECATED] Use --mode instead. This flag is ignored.')
    parser.add_argument('--lora_v2', type=str2bool, default=None,
                        help='[DEPRECATED] Use --mode instead. This flag is ignored.')
    parser.add_argument('--freeze_ssm', type=str2bool, default=None,
                        help='[DEPRECATED] Use --mode instead. This flag is ignored.')

    # Training configuration
    parser.add_argument('--pergpu_perturbations', type=int, default=32,
                        help='Perturbations per GPU (Total = pergpu * n_devices)')
    parser.add_argument('--n_perturbations', type=int, default=None,
                        help='[DEPRECATED] Total population size. If set, overrides pergpu_perturbations.')
    # Legacy alias alias
    parser.add_argument('--n_threads', type=int, default=None,
                        help='[DEPRECATED] Use --n_perturbations instead')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--n_steps', type=int, default=10, help='Steps per episode')
    parser.add_argument('--n_warmup_msgs', type=int, default=10,
                        help='Number of warmup messages to replay before episode starts (0 = no warmup)')
    parser.add_argument('--background_msgs_per_step', type=int, default=50,
                        help='Background messages per step (applies to both world_model and historical_replay)')
    # Legacy alias alias
    parser.add_argument('--world_msgs_per_step', type=int, default=None,
                        help='[DEPRECATED] Use --background_msgs_per_step instead')

    # Execution task
    parser.add_argument('--task', type=str, default='sell',
                        choices=['sell', 'buy'])
    parser.add_argument('--task_size', type=int, default=50,
                        help='Shares to execute')
    parser.add_argument('--tick_size', type=int, default=100,
                        help='Tick size in cents')

    # Token mode (auto-detected from checkpoint if not specified)
    parser.add_argument('--token_mode', type=int, default=24, choices=[22, 24],
                        help='Token mode: 22 (single token size) or 24 (base-100 size). Auto-detected from checkpoint.')

    # Background model configuration
    parser.add_argument('--background_mode', type=str, default='world_model',
                        choices=['world_model', 'historical_replay'],
                        help='Background message generation mode')
    parser.add_argument('--replay_data_path', type=str, default=None,
                        help='Path to historical data directory for replay mode')
    parser.add_argument('--file_idx', type=int, default=None,
                        help='Fixed file index for replay data (default: random, wraps with modulo)')

    # Data directory for initial state
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to LOBSTER data directory for initial state')

    # Other
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output_dir', type=str, default='./es_checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./es_checkpoints',
                        help='Directory to save ES checkpoints')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint directory to resume training from')

    # W&B logging
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/username')

    # Multi-node distributed training
    parser.add_argument('--coord_addr', type=str, default=None,
                        help='Coordinator address (IP:port) for multi-node distributed training')
    parser.add_argument('--num_procs', type=int, default=None,
                        help='Total number of processes (one per node)')
    parser.add_argument('--proc_id', type=int, default=None,
                        help='Process ID for this node (0-indexed)')

    # Fitness shaping
    parser.add_argument('--rank_transform', type=str2bool, default=True,
                        help='Use rank-based fitness shaping (helps escape local optima like "no trading"). Default: True')

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


# REMOVED: decoded_msg_to_jaxlob_format and msgs_to_jnp are now imported
# from lob.inference_no_errcorr (see imports at top of file).
# This eliminates ~30 lines of duplicated code.


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

    # Construct JaxLOB message using shared function from lob.inference_no_errcorr
    sim_msg = construct_sim_msg(
        safe_event_type,
        safe_side,
        safe_quantity,
        p_abs,
        order_id,
        time_s,
        time_ns,
        trader_id=trader_id,
    )

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

        # ========================================================================
        # Multi-node distributed detection
        # NOTE: jax.distributed.initialize() is called in es_training.py BEFORE
        # importing this module, because it must happen before any JAX operations.
        # Here we just detect if we're in distributed mode.
        # ========================================================================
        if jax.process_count() > 1:
            self._is_distributed = True
            self._process_index = jax.process_index()
            self._process_count = jax.process_count()
            print(f"[DIST] Running in distributed mode: process {self._process_index} of {self._process_count}")
        else:
            self._is_distributed = False
            self._process_index = 0
            self._process_count = 1

        # Handle perturbations configuration
        # Priority: n_perturbations (explicit) > pergpu_perturbations (auto)
        
        # 1. Check deprecated n_threads
        if hasattr(config, 'n_threads') and getattr(config, 'n_threads', None) is not None:
            print("[WARN] --n_threads is deprecated, use --pergpu_perturbations instead")
            if getattr(config, 'n_perturbations', None) is None:
                 config.n_perturbations = config.n_threads

        # 2. Calculate effective n_perturbations
        total_devices = jax.device_count()
        
        if getattr(config, 'n_perturbations', None) is not None:
            # User explicitly set total count
            pass
        else:
            # Auto-calculate from per-gpu
            per_gpu = getattr(config, 'pergpu_perturbations', 32)
            config.n_perturbations = per_gpu * total_devices
            print(f"[INIT] Auto-calculated n_perturbations: {config.n_perturbations} ({per_gpu} per GPU * {total_devices} devices)")

        # Validate divisibility
        if config.n_perturbations % total_devices != 0:
            print(f"[WARN] n_perturbations ({config.n_perturbations}) is not divisible by n_devices ({total_devices})")

        # Legacy alias: world_msgs_per_step -> background_msgs_per_step
        if hasattr(config, 'world_msgs_per_step') and getattr(config, 'world_msgs_per_step', None) is not None:
            if not hasattr(config, 'background_msgs_per_step') or getattr(config, 'background_msgs_per_step', 50) == 10:
                print("[WARN] --world_msgs_per_step is deprecated, use --background_msgs_per_step instead")
                config.background_msgs_per_step = config.world_msgs_per_step
        # Ensure background_msgs_per_step exists
        if not hasattr(config, 'background_msgs_per_step'):
            config.background_msgs_per_step = getattr(config, 'world_msgs_per_step', 50)

        _lazy_import_jaxlob()

        # Load LOBS5 checkpoint
        print(f"[INIT] Loading checkpoint from {config.lobs5_checkpoint}")
        load_checkpoint_for_es = _get_checkpoint_loader()
        self.lobs5_init, self.es_tree_key = load_checkpoint_for_es(config.lobs5_checkpoint, seed=config.seed)

        # Auto-detect token_mode from checkpoint (like run_inference.py)
        # This overrides the command-line default to ensure correct encoding/decoding
        ckpt_token_mode = self.lobs5_init.frozen_params.get('token_mode', None)
        if ckpt_token_mode is not None:
            if config.token_mode != ckpt_token_mode:
                print(f"[INIT] WARNING: Command-line token_mode={config.token_mode} differs from checkpoint={ckpt_token_mode}")
                print(f"[INIT] Using checkpoint token_mode={ckpt_token_mode} for consistency")
            config.token_mode = ckpt_token_mode
        print(f"[INIT] token_mode: {config.token_mode}")

        # Initialize Flax model for inference (same code path as run_inference.py)
        # This provides correct token generation - separate from ES params
        self._init_flax_inference()

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
        #
        # Multi-node mode (HyperscaleES pattern):
        # - Each process has 1 local GPU (via SLURM --ntasks-per-node=4)
        # - Use jax.local_devices() for local mesh
        # - Cross-node gradient sync via process_allgather in train_epoch
        # ========================================================================
        if self._is_distributed:
            # Multi-node: Each process has 1 local device
            local_devices = jax.local_devices()
            self._n_devices = len(local_devices)
            self._mesh = Mesh(local_devices, ('data',))
            print(f"[H3] Multi-node: Using {self._n_devices} local device(s) (process {self._process_index})")
        else:
            # Single-node: Use all devices
            self._n_devices = len(jax.devices())
            self._mesh = Mesh(jax.devices(), ('data',))
            print(f"[H3] Single-node: Using {self._n_devices} device(s)")
        print(f"[H3] Mesh axis: {self._mesh.axis_names}")

        # ========================================================================
        # G1 + H1: Pre-compile eval_batch for faster subsequent epochs
        # The first epoch will trigger actual XLA compilation, but subsequent
        # epochs will reuse the cached compilation.
        # H1: With mesh available, this will use shard_map for multi-GPU.
        # ========================================================================
        print("[INIT] Building eval_batch function (compilation on first call)...")
        self._compiled_eval_batch = self._compile_eval_batch()

    def _init_flax_inference(self):
        """Initialize Flax model for inference (same code path as run_inference.py).

        This loads the original Flax model and train_state, which are required for
        using inference_no_errcorr._generate_msg() for token generation.

        The ES-converted params in self.lobs5_init are ONLY used for ES gradient updates.
        For inference/generation, we use the original Flax model.

        NOTE: We use load_flax_checkpoint() from checkpoint_adapter instead of
        load_checkpoint() from lob.init_train, because the checkpoint is in OCDBT
        format which Orbax's PyTreeCheckpointHandler doesn't recognize properly.
        """
        config = self.config
        init_train_state, _, load_metadata = _get_flax_loaders()

        print(f"[INIT-FLAX] Loading Flax model from {config.lobs5_checkpoint}")

        # Step 1: Load metadata (config) from checkpoint
        args = load_metadata(config.lobs5_checkpoint)
        token_mode = getattr(args, 'token_mode', 24)

        # Step 2: Initialize vocabulary
        from lob.encoding import Vocab
        self.vocab = Vocab(token_mode=token_mode)
        n_classes = len(self.vocab)

        # Step 3: Get frozen params for model dimensions
        fp = self.lobs5_init.frozen_params
        msg_seq_len = fp.get('msg_seq_len', 500)
        book_depth = fp.get('book_depth', 500)
        book_dim = fp.get('d_book', 503)

        # Step 4: Initialize Flax train_state and model class (with random params)
        self.flax_train_state, self.flax_model_cls, total_params = init_train_state(
            args,
            n_classes=n_classes,
            seq_len=msg_seq_len,
            book_dim=book_dim,
            book_seq_len=book_depth,
            train_size=1,  # dummy value for inference
        )
        print(f"[INIT-FLAX] Model parameters: {total_params:,}")

        # Step 5: Load checkpoint params using OCDBT-compatible loader
        # (same loader already used successfully in checkpoint_adapter.py)
        from es_lobs5.adapters.checkpoint_adapter import load_flax_checkpoint
        loaded_params, _ = load_flax_checkpoint(config.lobs5_checkpoint)
        print(f"[INIT-FLAX] Loaded {len(jax.tree_util.tree_leaves(loaded_params))} param arrays from checkpoint")

        # Step 6: Replace random params with loaded checkpoint params
        self.flax_train_state = self.flax_train_state.replace(params=loaded_params)

        # Step 7: Instantiate model for inference
        self.flax_model = self.flax_model_cls(training=False, step_rescale=1.0)
        self.flax_batchnorm = getattr(args, 'batchnorm', False)

        # Step 8: Pre-compute syntax validation matrix for token generation
        from lob.validation_helpers import syntax_validation_matrix
        self.syntax_valid_mask = syntax_validation_matrix(self.vocab)

        # Check BF16 mixed precision status (controlled by USE_BF16 env var, default='1')
        use_bf16 = os.environ.get('USE_BF16', '1') == '1'

        print(f"[INIT-FLAX] Flax model loaded successfully")
        print(f"[INIT-FLAX]   token_mode={token_mode}, batchnorm={self.flax_batchnorm}")
        print(f"[INIT-FLAX]   mixed_precision={'BF16' if use_bf16 else 'FP32'} (USE_BF16={os.environ.get('USE_BF16', '1')})")

    def _log_detailed_param_stats(self, params, es_map, config, freeze_nonlora):
        """Log detailed parameter structure with status for each tensor."""
        print(f"\n[NOISER] {'='*120}")
        print(f"[NOISER] {'Parameter Name':<60} | {'Shape':<15} | {'Type':<6} | {'Status':<12} | {'Physical':>12} | {'Effective':>12}")
        print(f"[NOISER] {'-'*120}")

        total_physical = 0
        total_trainable = 0
        total_effective = 0

        # Flatten with paths to iterate
        flat_params, tree_def = jax.tree_util.tree_flatten_with_path(params)
        flat_map = jax.tree_util.tree_flatten(es_map)[0]

        # Helper to format path
        def path_to_str(path):
            # path is a tuple of (DictKey, SequenceKey, etc)
            return "/".join([str(p.key) if hasattr(p, 'key') else str(p) for p in path])

        for (path, param), map_val in zip(flat_params, flat_map):
            name = path_to_str(path)
            shape = str(param.shape)
            physical_size = param.size
            total_physical += physical_size

            # map: 0=FULL, 1=LORA, 2=FIXED, 3=FIXED
            if map_val == 1: # LORA
                p_type = "LORA"
                status = "[Trainable]"
                total_trainable += physical_size
                # LoRA DoF = (in + out) * rank (assuming 2D)
                if param.ndim == 2:
                    eff_size = (param.shape[0] + param.shape[1]) * config.lora_rank
                else:
                    eff_size = 0 # Fallback or unlikely for LoRA
            elif map_val == 0: # FULL
                p_type = "FULL"
                if freeze_nonlora:
                    status = "[Frozen]"
                    eff_size = 0
                else:
                    status = "[Trainable]"
                    total_trainable += physical_size
                    eff_size = physical_size
            else: # FIXED
                p_type = "FIXED"
                status = "[Fixed]"
                eff_size = 0

            total_effective += eff_size

            # Print row
            print(f"[NOISER] {name:<60} | {shape:<15} | {p_type:<6} | {status:<12} | {physical_size:>12,} | {eff_size:>12,}")

        print(f"[NOISER] {'='*120}")
        print(f"[NOISER] SUMMARY:")
        print(f"[NOISER]   Total Physical Params:  {total_physical:>15,}")
        print(f"[NOISER]   Total Trainable Params: {total_trainable:>15,} ({(total_trainable/total_physical)*100:.2f}%)")
        print(f"[NOISER]   Total Effective Params: {total_effective:>15,} ({(total_effective/total_physical)*100:.2f}%)")
        print(f"[NOISER] {'='*120}\n")


    def _init_noiser(self):
        """Initialize EGGROLL noiser for Policy using declarative mode configuration.

        Training modes are defined in training_modes.py. The mode parameter
        replaces the old boolean flags (use_lora, freeze_nonlora, lora_v2, freeze_ssm).
        """
        config = self.config
        all_noisers = _get_all_noisers()
        NOISER = all_noisers[config.noiser]
        self.noiser_cls = NOISER

        # Get group_size from config (required for EggRollBS, default 0 for EggRoll)
        group_size = getattr(config, 'group_size', 0)

        # ========================================================================
        # MODE-BASED CONFIGURATION (replaces scattered boolean flags)
        # ========================================================================
        training_modes = _get_training_modes()
        mode = getattr(config, 'mode', 'LORA_V1.5')

        # Warn if deprecated flags are used
        deprecated_flags = ['use_lora', 'freeze_nonlora', 'lora_v2', 'freeze_ssm']
        used_deprecated = [f for f in deprecated_flags if getattr(config, f, None) is not None]
        if used_deprecated:
            print(f"[NOISER] WARNING: Deprecated flags ignored: {used_deprecated}")
            print(f"[NOISER]          Use --mode instead. Current mode: {mode}")

        # Get mode configuration
        mode_config = training_modes['get_mode_config'](mode)

        # Print mode configuration table
        training_modes['print_mode_config'](mode)

        # Apply es_map classification based on mode
        classifier = training_modes['get_param_classifier'](mode)
        self.lobs5_init.es_map = training_modes['apply_es_map_classification'](
            self.lobs5_init.es_map,
            self.lobs5_init.params,
            classifier
        )

        # Get freeze_nonlora from mode config
        freeze_nonlora = mode_config.freeze_nonlora

        # Initialize noiser
        self.frozen_noiser_params, self.noiser_params = NOISER.init_noiser(
            self.lobs5_init.params,
            sigma=config.sigma,
            lr=config.lr,
            rank=config.lora_rank,
            freeze_nonlora=freeze_nonlora,
            noise_reuse=0,
            group_size=group_size,
            solver=None,  # Uses default optax.sgd
        )

        # Log training mode summary
        is_lora = training_modes['is_lora_mode'](mode)
        if not is_lora:
            print(f"[NOISER] Full fine-tuning mode: {mode}")
            print(f"[NOISER] All valid parameters will be updated directly")
        else:
            print(f"[NOISER] LoRA training mode: {mode}, rank={config.lora_rank}")
            print(f"[NOISER] freeze_nonlora={freeze_nonlora}")

        # Detailed Parameter Logging
        self._log_detailed_param_stats(self.lobs5_init.params, self.lobs5_init.es_map, config, freeze_nonlora)

    def _init_jaxlob(self):
        """Initialize JaxLOB order book simulator.

        Uses Configuration-based API (nOrders/nTrades are in the config).
        """
        _lazy_import_jaxlob()

        # JaxLOB OrderBook now uses Configuration-based API
        from gymnax_exchange.jaxob.jaxob_config import Configuration
        from dataclasses import replace

        # Calculate required capacity
        n_warmup = getattr(self.config, 'n_warmup_msgs', 10)
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
        """Pre-load historical data for replay mode using LOBSTER_Dataset.

        REFACTORED: Uses inference.get_dataset() for consistent data loading.
        This ensures the same code path as run_inference.py, guaranteeing:
        - Correct token_mode (24-token) encoding
        - Proper raw message format for simulation
        - Consistent book initialization
        """
        if self.config.background_mode != 'historical_replay':
            self.replay_data_raw = None
            self.replay_tokens = None
            self.replay_dataset = None
            return

        if self.config.replay_data_path is None:
            raise ValueError("--replay_data_path required when background_mode=historical_replay")

        import os

        data_path = self.config.replay_data_path

        # Use inference.get_dataset() - SAME code path as run_inference.py
        # This ensures consistent token_mode handling
        n_warmup = getattr(self.config, 'n_warmup_msgs', 10)
        n_sim = getattr(self.config, 'n_sim_steps', 1000)

        self.replay_dataset = inference.get_dataset(
            data_dir=data_path,
            n_messages=n_warmup,  # warmup messages
            n_eval_messages=n_sim + 500,  # simulation messages + buffer
            token_mode=self.config.token_mode,
            test_split=0.0,  # Use all data for ES training
        )

        print(f"[INIT-REPLAY] Loaded dataset with {len(self.replay_dataset)} files")
        print(f"[INIT-REPLAY] token_mode={self.config.token_mode}, n_messages={n_warmup + n_sim + 500}")

        # Select file: fixed or random
        import numpy as np
        if hasattr(self.config, 'file_idx') and self.config.file_idx is not None:
            file_idx = self.config.file_idx % len(self.replay_dataset)
        else:
            file_idx = np.random.randint(0, len(self.replay_dataset))

        self.replay_file_idx = file_idx

        # Get data from dataset (consistent with run_inference.py)
        # Returns: (X_tokens_masked, y, book_data, X_raw, book_l2_init)
        data_tuple = self.replay_dataset[file_idx]

        # Unpack based on return format
        # With return_raw_msgs=True and use_book_data=True:
        # (tokens, y, book, raw_msgs, book_l2_init)
        msg_tokens, _, book_data, msg_raw, book_l2_init = data_tuple

        # Store raw messages and encode them ourselves
        # LOBSTER_Dataset's msg_tokens has masking applied (for training), not suitable for replay
        # Instead, encode raw messages directly like run_inference.py does
        from lob.encoding import encode_msgs

        self.replay_data_raw = jnp.array(msg_raw)
        self.init_book_l2 = jnp.array(book_l2_init)

        # Encode raw messages to tokens (same as run_inference.py)
        # Shape: (n_msgs, token_mode) for message-level indexing
        encoded = encode_msgs(msg_raw, self.encoder, token_mode=self.config.token_mode)
        self.replay_tokens = jnp.array(encoded)  # (n_msgs, token_mode)

        # H4: Store Ground Truth Book Data for plotting
        # book_data shape: (n_msgs, 43? or 21?)
        # Use simple jnp array for now
        self.replay_book_data = jnp.array(book_data)

        # H4: Verify data structure assumptions (User Request)
        # Check that Ask Price (Idx 3) > Bid Price (Idx 5) for a sample
        # to ensure we are using the correct columns.
        sample_indices = np.random.randint(0, len(book_data), size=100)
        sample_ask = np.array(book_data[sample_indices, 3])
        sample_bid = np.array(book_data[sample_indices, 5])
        
        # Check average spread is positive
        avg_spread = np.mean(sample_ask - sample_bid)
        if avg_spread <= 0:
            print(f"[WARN] Replay Data Validation Warning: Average spread is {avg_spread} (<=0).")
            print(f"       Ask P1 (Idx 3): {np.mean(sample_ask)}, Bid P1 (Idx 5): {np.mean(sample_bid)}")
            print("       Double check column indices in replay_book_data!")
        else:
            print(f"[INIT-REPLAY] Data structure verified: Avg Spread = {avg_spread:.2f} (Ask > Bid confirmed)")


        # Extract date from dataset files for logging
        from glob import glob
        msg_files = sorted(glob(os.path.join(data_path, '*message*.npy')))
        if msg_files and len(msg_files) > 0:
            # Use modulo to handle file_idx > len(msg_files)
            safe_idx = file_idx % len(msg_files)
            self.replay_data_date = os.path.basename(msg_files[safe_idx]).split('_')[1]
        else:
            self.replay_data_date = 'unknown'
        self.replay_data_dir = data_path

        print(f"[INIT-REPLAY] File {file_idx}: date={self.replay_data_date}")
        print(f"[INIT-REPLAY] Raw msgs shape: {msg_raw.shape}, Tokens shape: {msg_tokens.shape}")
        print(f"[INIT-REPLAY] Replay Book Data shape: {self.replay_book_data.shape}")
        print(f"[INIT-REPLAY] Init book L2 shape: {book_l2_init.shape}")

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
        """Load initial JaxLOB state and message history.

        REFACTORED: Uses data already loaded by _init_historical_replay_data().
        This eliminates duplicate file loading and ensures consistent data handling.

        Returns:
            (sim_state, msg_history): Initial simulation state and token context
        """
        config = self.config

        # Ensure replay data is initialized
        if not hasattr(self, 'init_book_l2') or self.init_book_l2 is None:
            # Fallback: initialize replay data now
            self._init_historical_replay_data()

        # 1. Initialize JaxLOB with L2 book from dataset
        # init_book_l2 was set by _init_historical_replay_data() using LOBSTER_Dataset
        sim_state = self.sim.reset(self.init_book_l2)
        print(f"[INIT-STATE] Initialized JaxLOB with L2 book shape: {self.init_book_l2.shape}")

        # 2. Warmup: replay messages to initialize order book state
        n_warmup = getattr(config, 'n_warmup_msgs', 10)
        n_replay = min(n_warmup, len(self.replay_data_raw))

        if n_replay > 0:
            replay_msgs_raw = self.replay_data_raw[:n_replay]
            replay_jaxlob = msgs_to_jnp(replay_msgs_raw)
            sim_state = self.sim.process_orders_array(sim_state, replay_jaxlob)
            print(f"[INIT-STATE] Replayed {n_replay} warmup messages")

        # 3. Build context tokens from replay_tokens (encoded by LOBSTER_Dataset)
        # CRITICAL: Do NOT pad with zeros - zeros are MASK tokens which corrupt RNN hidden states!
        # Instead, just use the actual warmup tokens and let simulate_episode() handle the warmup.
        warmup_tokens = self.replay_tokens[:n_replay]  # (n_replay, token_mode)
        msg_history = warmup_tokens.flatten()  # (n_replay * token_mode,)

        # Store n_warmup_msgs for simulate_episode() to use
        self._n_warmup_msgs = n_replay

        print(f"[INIT-STATE] Context: {msg_history.shape} (warmup={n_replay} real messages, no zero padding)")

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
            es_map=self.lobs5_init.es_map,
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
            es_map=self.lobs5_init.es_map,
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
        es_map = self.lobs5_init.es_map
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
                es_map=es_map,
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
                es_map=es_map,
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
        process_orders_array = self.sim.process_orders_array
        sim_obj = self.sim
        encoder = self.encoder
        replay_tokens = self.replay_tokens
        replay_data_raw = self.replay_data_raw
        if self.config.background_mode == 'historical_replay':
            replay_book_data = self.replay_book_data
        else:
            replay_book_data = None
        n_replay_msgs = replay_tokens.shape[0] if replay_tokens is not None else 0

        # ========================================================================
        # Syntax validation helpers (still used for constrained decoding)
        # ========================================================================
        syntax_valid_mask = self.syntax_valid_mask
        import lob.validation_helpers as valh_module
        # ========================================================================

        # ========================================================================
        # H2: Helper function to apply pvary when inside shard_map
        # Reference: https://docs.jax.dev/en/latest/notebooks/shard_map.html#scan-vmap
        # When using shard_map + scan, initial carry values must be marked as
        # "varying" along the sharded axis using jax.lax.pvary.
        # ========================================================================
        def maybe_pvary(x):
            """Apply pvary if inside shard_map context."""
            # DISABLED: pvary causes OOM by forcing global state materialization
            # when used with vmap. Standard vmap(scan) handles local batching correctly.
            # if in_shard_map:
            #     return jax.lax.pvary(x, ('data',))
            return x

        def maybe_pvary_tree(tree):
            """Apply pvary to all leaves of a pytree if inside shard_map."""
            # DISABLED: see maybe_pvary
            # if in_shard_map:
            #     return jax.tree.map(lambda x: jax.lax.pvary(x, ('data',)), tree)
            return tree
        # ========================================================================

        # Get ES model class (still used for world model)
        ES_PaddedLobPredModel = _get_es_model()

        # Initialize hidden states
        # CRITICAL: Match parameter names from checkpoint metadata exactly!
        # Checkpoint uses: n_layers (for fused), ssm_size_base (for SSM size)
        ssm_size = fp.get('ssm_size_base', fp.get('ssm_size', 256))  # Try both keys
        n_fused = fp.get('n_layers', fp.get('n_fused_layers', 4))    # Try both keys
        conj_sym = fp.get('conj_sym', True)
        d_model = fp.get('d_model', 256)
        print(f"[HIDDEN-INIT] ssm_size={ssm_size}, conj_sym={conj_sym}, n_fused={n_fused}, d_model={d_model}")

        # World model still uses ES hidden states (for background generation)
        hiddens_world = ES_PaddedLobPredModel.initialize_carry(
            batch_size=1,
            ssm_size=ssm_size,  # Pass full ssm_size - initialize_carry handles conj_sym
            n_message_layers=fp.get('n_message_layers', 2),
            n_book_pre_layers=fp.get('n_book_pre_layers', 1),
            n_book_post_layers=fp.get('n_book_post_layers', 1),
            n_fused_layers=n_fused,
            d_model=d_model,
            conj_sym=conj_sym,
        )

        # ========================================================================
        # ES MODEL: Policy uses ES hidden states (same as world model + warmup)
        # ========================================================================
        hiddens_policy = ES_PaddedLobPredModel.initialize_carry(
            batch_size=1,
            ssm_size=ssm_size,
            n_message_layers=fp.get('n_message_layers', 2),
            n_book_pre_layers=fp.get('n_book_pre_layers', 1),
            n_book_post_layers=fp.get('n_book_post_layers', 1),
            n_fused_layers=n_fused,
            d_model=d_model,
            conj_sym=conj_sym,
        )

        # Message length based on token mode
        msg_len = 22 if config.token_mode == 22 else 24
        msg_seq_len = fp.get('msg_seq_len', 500)
        context_len = msg_len * msg_seq_len

        # =====================================================================
        #                        TRADER ID CONSTANTS
        # =====================================================================
        # THREE DISTINCT TRADER IDs to identify order sources in JaxLOB:
        #
        #   (1) HISTORICAL_TRADER_ID = -2000
        #       → Orders replayed from historical market data files
        #       → Background liquidity from real market recordings
        #
        #   (2) POLICY_TRADER_ID = -1000
        #       → Orders generated by the ES-trained policy model
        #       → The agent being trained (our trading decisions)
        #
        #   (3) WORLD_TRADER_ID = -3000
        #       → Orders generated by the world model (background generator)
        #       → AI-generated background market activity
        #
        # These IDs enable tracking and analysis of order flow by source.
        # =====================================================================
        HISTORICAL_TRADER_ID = -2000
        POLICY_TRADER_ID = -1000
        WORLD_TRADER_ID = -3000

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
        field_masks = get_field_masks_from_validation_matrix(
            token_mode=config.token_mode,
            vocab_size=vocab_size
        )

        def step_fn(carry, step_idx):
            """Single step: Background messages -> Policy action."""
            (key, msg_history, hiddens_world, hiddens_policy,
             sim_state, book_feat, world_oid_offset, quant_executed, accum_revenue, accum_trades, accum_all_trades, accum_submitted) = carry

            key, key_world, key_policy = jax.random.split(key, 3)

            # Background message generation
            def historical_replay_step(wcarry, bg_msg_idx):
                """Load pre-encoded messages from historical data."""
                key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr = wcarry

                replayed_msg_tokens = replay_tokens[replay_ptr]
                replayed_msg_raw = replay_data_raw[replay_ptr]

                sim_msg = msg_to_jnp(replayed_msg_raw)  # Using imported function from lob.inference_no_errcorr
                bg_order_id = WORLD_ORDER_ID_START + oid_offset
                sim_msg = sim_msg.at[4].set(bg_order_id)
                sim_msg = sim_msg.at[5].set(HISTORICAL_TRADER_ID)

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
                    trader_id=WORLD_TRADER_ID, token_mode=config.token_mode
                )

                sim_st = process_order_array(sim_st, sim_msg)
                book_f = transform_L2_state_wrapper(jaxlob_cfg, sim_st, price_levels=book_depth, tick_size=config.tick_size, in_shard_map=in_shard_map)
                msg_hist = jnp.concatenate([msg_hist[msg_len:], world_msg])
                oid_offset = oid_offset + 1

                return (key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr), world_msg

            # Select background generation function
            n_warmup_cfg = getattr(config, 'n_warmup_msgs', 10)
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

            def sample_policy_token_es(token_carry, token_pos):
                """Sample next token using ES model (perturbed params).

                Uses ES_PaddedLobPredModel._forward_step with policy_common_params,
                which contains iterinfo=(epoch, thread_id) to activate LoRA noise.
                This matches the HyperscaleES reference implementation.

                Args:
                    token_carry: (key, msg_history, hiddens)
                    token_pos: Current position in 24-token message (0-23)
                """
                key_p, msg_hist_p, hidden_p = token_carry
                key_p, sample_key_p = jax.random.split(key_p)

                # Get syntax validation mask for current token position
                valid_mask = valh_module.get_valid_mask(syntax_valid_mask, token_pos)

                # ES model forward (single token)
                # msg_hist_p[-1:] shape: (1,) — only last token
                # book_feat[None, :] shape: (1, d_book)
                hidden_p, log_probs = ES_PaddedLobPredModel._forward_step(
                    policy_common_params, hidden_p, msg_hist_p[-1:], book_feat[None, :]
                )
                # log_probs shape: (1, d_output) — single token, already log_softmax
                # EMA state: single token → shape (batch, 1, d) → no fix_ema_shape needed

                # Apply syntax validation mask + renormalize
                log_probs = valh_module.filter_valid_pred(log_probs, valid_mask)

                # Sample next token (top_n=-1 = sample from full distribution)
                next_token_p = valh_module.fill_predicted_tok(
                    log_probs, -1, jnp.array([sample_key_p])
                )

                # Update message history
                msg_hist_p = jnp.concatenate([msg_hist_p[1:], next_token_p])

                # Return scalar token for scan output (squeeze the (1,) array)
                return (key_p, msg_hist_p, hidden_p), next_token_p[0]

            key_policy, sample_key = jax.random.split(key_policy)
            # H2: Apply pvary to initial carry values when inside shard_map
            policy_token_init = (
                maybe_pvary(sample_key),
                maybe_pvary(msg_history),
                maybe_pvary_tree(hiddens_policy),
            )
            # Pass token positions (0-23) as xs to enable field-aware masking
            # ES model path: matches HyperscaleES reference (warmup + token gen use same model)
            (key_policy, msg_history, hiddens_policy), policy_msg = jax.lax.scan(
                sample_policy_token_es,  # ES model path
                policy_token_init,
                jnp.arange(msg_len, dtype=jnp.int32),
                length=msg_len,
            )

            # Convert to JaxLOB format
            mid_price = get_mid_price(jaxlob_cfg, sim_state, config.tick_size)
            policy_order_id = POLICY_ORDER_ID_START + step_idx
            sim_msg, msg_decoded = get_sim_msg_es(
                policy_msg, sim_obj, sim_state, mid_price, policy_order_id, config.tick_size, encoder,
                trader_id=POLICY_TRADER_ID, token_mode=config.token_mode
            )

            # Truncate quantity to remaining task
            quant_remaining = task_size - quant_executed
            original_qty = sim_msg[2]
            truncated_qty = jnp.minimum(original_qty, jnp.maximum(quant_remaining, 0))
            sim_msg = sim_msg.at[2].set(truncated_qty)

            # =================================================================
            # SMART CANCELLATION LOGIC (Match Gymnax Exchange Standard)
            # =================================================================
            # 1. Generate Cancel Messages for this agent
            is_sell_task = (config.task == 'sell')
            bookside = sim_state.asks if is_sell_task else sim_state.bids
            side_int = -1 if is_sell_task else 1
            
            # Use getCancelMsgs (size=5: max concurrent active orders to cancel)
            # Note: gym_env uses world_state.time for cancel time. We use 0.
            cancel_msgs = getCancelMsgs(
                bookside, 
                POLICY_TRADER_ID, 
                5,          # size
                side_int,   # side
                0, 0        # time
            )

            # 2. Filter Messages (Remove redundant cancel+add pairs)
            # Expand sim_msg to (1,8) for batch processing
            action_msgs_in = sim_msg[None, :]
            action_msgs, cancel_msgs = _filter_messages(action_msgs_in, cancel_msgs)
            
            # 3. Combine Action and Cancel Messages
            # (N_cancel + 1) messages
            combined_msgs = jnp.concatenate([cancel_msgs, action_msgs])

            # =================================================================
            # FIX: Clear trades buffer before processing to prevent
            # quadratic accumulation of trades across steps.
            # =================================================================
            sim_state = sim_state._replace(trades=jnp.ones_like(sim_state.trades) * -1)

            # Process orders (Batch)
            sim_state = process_orders_array(sim_state, combined_msgs)

            # Track execution
            trades = sim_state.trades
            is_new_trade = (trades[:, 0] != -1)
            # FIX: Use POLICY_TRADER_ID check instead of Order ID.
            # This is CRITICAL for Smart Cancellation: if an old order (with old Order ID)
            # is preserved and fills, we must count it!
            # Col 6 = Passive Trader ID, Col 7 = Aggressive Trader ID
            is_policy_in_trade = ((trades[:, 6] == POLICY_TRADER_ID) | (trades[:, 7] == POLICY_TRADER_ID)) & is_new_trade
            step_executed = jnp.sum(jnp.where(is_policy_in_trade, jnp.abs(trades[:, 1]), 0))
            quant_executed = quant_executed + step_executed

            # Track revenue (accumulate Price * Qty)
            step_revenue = jnp.sum(jnp.where(is_policy_in_trade,
                                            trades[:, 0] * jnp.abs(trades[:, 1]),
                                            0)).astype(jnp.float32)
            accum_revenue = accum_revenue + step_revenue

            # Compute VWAP for this step (for trade visualization)
            # step_executed already computed above = sum(qty) where policy is involved
            # step_revenue = sum(price * qty) for policy trades
            step_vwap = jnp.where(
                step_executed > 0,
                step_revenue / step_executed.astype(jnp.float32),
                0.0
            )

            # Update state
            book_feat = transform_L2_state_wrapper(jaxlob_cfg, sim_state, price_levels=book_depth, tick_size=config.tick_size, in_shard_map=in_shard_map)
            msg_history = jnp.concatenate([msg_history[msg_len:], policy_msg])

            # Get best bid/ask for plotting
            best_ask, best_bid = get_best_bid_and_ask(jaxlob_cfg, sim_state.asks, sim_state.bids)

            # Track trades (count)
            # Track trades (count)
            step_trades = jnp.sum(is_policy_in_trade)
            accum_trades = accum_trades + step_trades

            # Track ALL trades (for total_trades metric - includes world model trades)
            step_all_trades = jnp.sum(is_new_trade)
            accum_all_trades = accum_all_trades + step_all_trades

            # Track submitted quantity (for execution probability calc)
            # sim_msg[2] is the truncated quantity (can actually be processed by book)
            step_submitted = sim_msg[2]
            accum_submitted = accum_submitted + step_submitted

            # Return policy_msg for order analysis (shape: (msg_len,))
            # Also return trade info for visualization: VWAP, qty, and bid/ask at trade time
            return (key, msg_history, hiddens_world, hiddens_policy, sim_state,
                    book_feat, world_oid_offset, quant_executed, accum_revenue, accum_trades, accum_all_trades, accum_submitted), \
                   (policy_msg, best_bid, best_ask, step_vwap, step_executed.astype(jnp.float32))

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
            maybe_pvary(jnp.int32(0)), # world_oid_offset
            maybe_pvary(jnp.int32(0)), # quant_executed
            maybe_pvary(jnp.float32(0)), # accum_revenue
            maybe_pvary(jnp.int32(0)), # accum_trades (policy/agent trades only)
            maybe_pvary(jnp.int32(0)), # accum_all_trades (all trades including world model)
            maybe_pvary(jnp.int32(0)), # accum_submitted
        )
        # Capture policy_msgs_all for order analysis (shape: (n_steps, msg_len))
        # Also capture trade traces: vwap, qty, bid, ask at each step for visualization
        (_, _, _, _, final_state, _, _, final_quant_executed, final_revenue, final_trades, final_all_trades, final_submitted), \
            (policy_msgs_all, bid_trace, ask_trace, trade_vwap_trace, trade_qty_trace) = jax.lax.scan(
            step_fn,
            main_scan_init,
            jnp.arange(config.n_steps),
            length=config.n_steps,
        )

        # =====================================================================
        # FORCED LIQUIDATION: Real market order to sweep the order book
        # This is the "hidden" final step that attempts to close remaining
        # position. Unlike doom_trade, this uses a REAL order that may NOT
        # fill completely if there's insufficient book depth.
        #
        # SELL: price = 0 → matches all bids from high to low
        # BUY: price = maxint → matches all asks from low to high
        # =====================================================================
        model_quantity = final_quant_executed  # Save quantity from model orders
        quant_remaining = task_size - final_quant_executed
        liquidation_order_id = POLICY_ORDER_ID_START + config.n_steps

        def _create_market_order(args):
            """Create real market order that sweeps the order book.

            Unlike doom_trade (virtual trade), this submits a REAL order
            through the JaxLOB matching engine. May have unfilled quantity
            if book depth is insufficient.

            Args:
                args: (sim_state, quant_to_liquidate) tuple

            Returns:
                Updated LobState after order processing
            """
            sim_state, quant_to_liquidate = args

            is_sell = (config.task == 'sell')
            # Market price: sell at 0 (matches all bids), buy at maxint (matches all asks)
            market_price = jnp.where(is_sell, 0, jaxlob_cfg.maxint)
            # JaxLOB side: -1 = sell (hit bids), 1 = buy (lift asks)
            side_jaxlob = jnp.where(is_sell, -1, 1)

            # Message format: [event_type, side, quantity, price, order_id, trader_id, time_s, time_ns]
            # Event type 1 = Limit order (at extreme price = effective market order)
            sim_msg = jnp.array([
                1,                      # event_type = Limit
                side_jaxlob,            # side (-1=sell, 1=buy)
                quant_to_liquidate,     # quantity
                market_price,           # price (0 or maxint)
                liquidation_order_id,   # order_id
                POLICY_TRADER_ID,       # trader_id
                0,                      # time_s
                0,                      # time_ns
            ], dtype=jnp.int32)

            return process_order_array(sim_state, sim_msg)

        # Only execute liquidation if there's remaining quantity
        final_state = jax.lax.cond(
            quant_remaining > 0,
            _create_market_order,
            lambda args: args[0],  # Return unchanged state
            (final_state, quant_remaining)
        )

        # FIX: Combine Loop Metrics (final_quant_executed, final_revenue)
        # with Liquidation Metrics (from final_state.trades).
        # =================================================================
        # Filter Liquidation Trades from final_state (which contains last step + liquidation)
        liquidation_oid = POLICY_ORDER_ID_START + config.n_steps
        trades = final_state.trades
        valid_trades = trades[:, 0] != -1
        is_liquidation = ((trades[:, 2] == liquidation_oid) | (trades[:, 3] == liquidation_oid)) & valid_trades
        
        liquidation_quantity_filled = jnp.sum(jnp.where(is_liquidation, jnp.abs(trades[:, 1]), 0))
        liquidation_revenue = jnp.sum(jnp.where(is_liquidation, trades[:, 0] * jnp.abs(trades[:, 1]), 0))
        liquidation_vwap = jnp.where(
            liquidation_quantity_filled > 0,
            liquidation_revenue / liquidation_quantity_filled.astype(jnp.float32),
            0.0
        )

        # Total metrics
        agent_quantity = final_quant_executed + liquidation_quantity_filled
        
        is_sell_task = (config.task == 'sell')
        if is_sell_task:
            total_revenue = final_revenue + liquidation_revenue
            pnl_raw = total_revenue - init_mid_price * agent_quantity
        else:
            total_cost = final_revenue + liquidation_revenue # Revenu accum is always P*Q
            pnl_raw = init_mid_price * agent_quantity - total_cost

        # PnL = raw profit/loss in cents (has real economic meaning)
        pnl = pnl_raw

        # NOTE: Normalization step removed - fitness is now computed as rank_transform(pnl) in train_epoch
        # Old approach: fitness_raw = pnl / (task_size * tick_size) then rank_transform(fitness_raw)
        # New approach: fitness = rank_transform(pnl) directly (simpler, cleaner semantics)
        # normalization_scale = config.task_size * config.tick_size
        # fitness_raw = pnl / jnp.maximum(normalization_scale, 1.0)

        # buffer_trades: diagnostic metric - trades in final step buffer only (renamed from old total_trades)
        buffer_trades = jnp.sum(valid_trades)

        # agent_trades = trades from model steps + trades from liquidation step
        liquidation_trades = jnp.sum(is_liquidation)
        agent_trades = final_trades + liquidation_trades

        # total_trades: ALL trades throughout the episode (world model + agent + liquidation)
        # FIX: Use accumulated all trades + liquidation trades from final buffer
        liquidation_all_trades = jnp.sum(valid_trades)  # All trades in liquidation step buffer
        total_trades = final_all_trades + liquidation_all_trades

        # Calculate execution breakdown
        # model_quantity: executed by model orders during regular steps (Normal Orders)
        # liquidation_quantity: executed by force_market_order at end (Market Orders)
        # doom_quantity: not executed due to insufficient book depth (Doom Orders)
        # NOTE: agent_quantity should never exceed task_size due to truncation
        liquidation_quantity = agent_quantity - model_quantity
        # Clamp doom_quantity to >= 0 (should not be negative if tracking is correct)
        doom_quantity = jnp.maximum(0, task_size - agent_quantity)

        # Sanitize pnl: replace non-finite values with 0.0
        # (fitness = rank_transform(pnl) is computed in train_epoch)
        pnl = jnp.where(jnp.isfinite(pnl), pnl, 0.0)

        info = {
            'pnl': pnl,                        # raw PnL in cents (fitness computed via rank_transform in train_epoch)
            # Execution breakdown
            'agent_quantity': agent_quantity,          # total executed = model + liquidation (may be < task_size!)
            'model_quantity': model_quantity,          # executed by model orders (Normal)
            'liquidation_quantity': liquidation_quantity,  # executed by force_market_order (Market)
            'doom_quantity': doom_quantity,    # NOT executed (Doom)
            'submitted_quantity': final_submitted, # Total quantity submitted by model (post-truncation)
            'agent_trades': agent_trades,
            'total_trades': total_trades,      # All trades (world + agent + liquidation)
            'buffer_trades': buffer_trades,    # Diagnostic: trades in final step buffer only
            'init_mid_price': init_mid_price,
            'init_mid_price': init_mid_price,
            'policy_msgs': policy_msgs_all,    # shape: (n_steps, msg_len) for order analysis
            'bid_trace': bid_trace,
            'ask_trace': ask_trace,
            # Trade visualization traces (shape: (n_steps,))
            'trade_vwap_trace': trade_vwap_trace,   # VWAP of agent trades per step
            'trade_qty_trace': trade_qty_trace,     # Quantity executed per step
            'liquidation_vwap': liquidation_vwap,   # VWAP of forced market order (0 if none)
        }

        # H4: Add Ground Truth Trace for "Whole Data Window" Plot
        # We assume standard LOBSTER format: Ask P1 (0), Ask S1 (1), Bid P1 (2), Bid S1 (3)
        # We need to slice replay_book_data for the duration of this episode
        # Duration = n_warmup + n_steps * bg_msgs
        if replay_book_data is not None:
             n_warmup = getattr(config, 'n_warmup_msgs', 10)
             n_bg = getattr(config, 'background_msgs_per_step', 10)
             total_msgs = n_warmup + config.n_steps * n_bg

             # Extract GT trace (Best Ask, Best Bid)
             # Extract GT trace (Best Ask, Best Bid)
             # replay_book_data has header cols 0,1,2.
             # LOBSTER data starts at col 3.
             # Ask Price 1 is at index 3, Bid Price 1 is at index 5
             gt_trace_full = replay_book_data[:total_msgs, [3, 5]]
             info['gt_bid_trace'] = gt_trace_full[:, 1] # Bid P1 (was ind 5)
             info['gt_ask_trace'] = gt_trace_full[:, 0] # Ask P1 (was ind 3)

             # ================================================================
             # H5: Extract Historical Trades from replay_data_raw
             # LOBSTER message format: [event_type, side, quantity, price, ...]
             # event_type = 4 means EXECUTION (historical trade)
             # ================================================================
             if replay_data_raw is not None:
                 # Get messages for this episode window
                 msg_window = replay_data_raw[:total_msgs]

                 # Extract event_type (col 0), side (col 1), qty (col 2), price (col 3)
                 event_types = msg_window[:, 0]
                 sides = msg_window[:, 1]      # 1 = buy, -1 = sell
                 quantities = msg_window[:, 2]
                 prices = msg_window[:, 3]

                 # Create mask for executions (event_type = 4)
                 is_execution = (event_types == 4)

                 # Store historical trade data (sparse representation)
                 # Shape: (total_msgs,) - non-zero where trade occurred
                 info['gt_trade_price'] = jnp.where(is_execution, prices, 0)
                 info['gt_trade_qty'] = jnp.where(is_execution, quantities, 0)
                 info['gt_trade_side'] = jnp.where(is_execution, sides, 0)
             else:
                 info['gt_trade_price'] = jnp.zeros((1,), dtype=jnp.int32)
                 info['gt_trade_qty'] = jnp.zeros((1,), dtype=jnp.int32)
                 info['gt_trade_side'] = jnp.zeros((1,), dtype=jnp.int32)
        else:
             # Placeholder for World Model mode
             info['gt_bid_trace'] = jnp.zeros((1,), dtype=jnp.int32)
             info['gt_ask_trace'] = jnp.zeros((1,), dtype=jnp.int32)
             info['gt_trade_price'] = jnp.zeros((1,), dtype=jnp.int32)
             info['gt_trade_qty'] = jnp.zeros((1,), dtype=jnp.int32)
             info['gt_trade_side'] = jnp.zeros((1,), dtype=jnp.int32)

        return pnl, info  # Return raw pnl; fitness = rank_transform(pnl) computed in train_epoch

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
        # Multi-node partitioning: Each process evaluates a DIFFERENT subset of
        # perturbations to avoid redundant computation across nodes.
        #
        # Example with 256 perturbations and 8 processes:
        #   Process 0: thread_ids [0-31]    (32 perturbations)
        #   Process 1: thread_ids [32-63]   (32 perturbations)
        #   ...
        #   Process 7: thread_ids [224-255] (32 perturbations)
        #
        # After evaluation, use process_allgather to combine results.
        # ========================================================================
        n_processes = getattr(self, '_process_count', 1)
        process_idx = getattr(self, '_process_index', 0)

        if self._is_distributed and n_processes > 1:
            # Validate divisibility
            assert n_perturbations % n_processes == 0, \
                f"[DIST ERROR] n_perturbations ({n_perturbations}) must be divisible by n_processes ({n_processes}). " \
                f"Consider using n_perturbations={n_processes * (n_perturbations // n_processes)}"

            # Each process evaluates a subset of perturbations
            local_n_perturbations = n_perturbations // n_processes
            start_idx = process_idx * local_n_perturbations
            end_idx = start_idx + local_n_perturbations

            # Generate keys only for this process's perturbations
            all_keys = jax.random.split(key, n_perturbations)
            keys = all_keys[start_idx:end_idx]
            thread_ids = jnp.arange(start_idx, end_idx)

            if epoch == 0:
                print(f"[DIST] Process {process_idx}/{n_processes}: evaluating thread_ids [{start_idx}-{end_idx-1}] ({local_n_perturbations} perturbations)")
        else:
            # Single-node or non-distributed: evaluate all perturbations
            keys = jax.random.split(key, n_perturbations)
            thread_ids = jnp.arange(n_perturbations)

        # ========================================================================
        # H1: Validate n_perturbations divisibility for shard_map
        # When using multi-GPU, n_perturbations must be evenly divisible by n_devices
        # so each device gets the same number of perturbations to evaluate.
        # ========================================================================
        local_n_perturbations = len(thread_ids)
        if n_devices > 1:
            assert local_n_perturbations % n_devices == 0, \
                f"[H1 ERROR] local_n_perturbations ({local_n_perturbations}) must be divisible by n_devices ({n_devices}). " \
                f"Consider adjusting n_perturbations to be divisible by (n_processes × n_devices)."

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

        pnls, infos = self._compiled_eval_batch(
            noiser_params_rep,
            params_rep,
            keys,
            thread_ids,
            jnp.int32(epoch),
            initial_sim_state,
            initial_msg_history,
        )

        # ========================================================================
        # Multi-node: Gather pnls and infos from all processes
        # In distributed mode, each process only has its local shard of results.
        # process_allgather collects all shards to form the complete arrays.
        #
        # After gathering, we have ALL n_perturbations pnl values.
        # The iterinfos must use GLOBAL thread_ids [0..n_perturbations-1]
        # for correct gradient computation across all perturbations.
        # ========================================================================
        if self._is_distributed and n_processes > 1:
            pnls = process_allgather(pnls, tiled=True)
            infos = jax.tree.map(
                lambda x: process_allgather(x, tiled=True),
                infos
            )
            # Use GLOBAL thread_ids for gradient update (all n_perturbations)
            global_thread_ids = jnp.arange(n_perturbations)
        else:
            global_thread_ids = thread_ids

        # ES gradient update (use global thread_ids after gathering)
        iterinfos = (
            jnp.full(n_perturbations, epoch, dtype=jnp.int32),
            global_thread_ids
        )

        # ========================================================================
        # H2: Use replicated params for gradient updates in multi-GPU mode
        # The pnls returned from shard_map are sharded, so params must
        # also be replicated to avoid device mismatch in do_updates
        # ========================================================================

        # fitness = rank_transform(pnl) - always apply rank transform
        # This maps raw PnL values to ranks normalized to [-0.5, 0.5]
        # Helps escape local optima like "no trading" by focusing on relative ordering
        fitnesses = rank_transform(pnls)

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

        # Extract updated params back to single LOCAL device for storage
        # CRITICAL: Must use jax.local_devices()[0] in multi-node to avoid cross-host issues
        if n_devices > 1 and hasattr(self, '_mesh'):
            # In multi-node mode: place on LOCAL device 0, not global device 0
            local_device = jax.local_devices()[0]
            self.noiser_params = jax.tree.map(
                lambda x: jax.device_put(x, local_device),
                noiser_params_updated
            )
            self.lobs5_init.params = jax.tree.map(
                lambda x: jax.device_put(x, local_device),
                updated_params
            )
        else:
            self.noiser_params = noiser_params_updated
            self.lobs5_init.params = updated_params

        # =========================================================================
        # Select example perturbation for visualization
        # Use best pnl perturbation instead of first (more likely to have trades)
        # =========================================================================
        best_idx = jnp.argmax(pnls)

        # Extract one example trace for plotting (from best perturbation)
        # We must pull this out BEFORE averaging, as averaging traces is meaningless/expensive
        # Note: infos['bid_trace'] shape is (n_perturbations, n_steps)
        example_bid_trace = infos['bid_trace'][best_idx]
        example_ask_trace = infos['ask_trace'][best_idx]

        # Ground Truth Traces (High Res) - same across perturbations, use [0]
        example_gt_bid_trace = infos['gt_bid_trace'][0]
        example_gt_ask_trace = infos['gt_ask_trace'][0]

        # Trade traces for visualization (from best perturbation)
        example_trade_vwap_trace = infos['trade_vwap_trace'][best_idx]
        example_trade_qty_trace = infos['trade_qty_trace'][best_idx]

        # Forced market order visualization (from best perturbation)
        example_liquidation_vwap = infos['liquidation_vwap'][best_idx]
        example_liquidation_qty = infos['liquidation_quantity'][best_idx]

        # Historical trade traces (same across perturbations, use [0])
        example_gt_trade_price = infos['gt_trade_price'][0]
        example_gt_trade_qty = infos['gt_trade_qty'][0]
        example_gt_trade_side = infos['gt_trade_side'][0]

        # Filter out heavy/non-scalar items before averaging
        infos_for_mean = {
            k: v for k, v in infos.items()
            if k not in ['bid_trace', 'ask_trace', 'policy_msgs', 'gt_bid_trace', 'gt_ask_trace',
                         'trade_vwap_trace', 'trade_qty_trace',
                         'gt_trade_price', 'gt_trade_qty', 'gt_trade_side']
        }
        aggregated_info = {k: jnp.mean(v) for k, v in infos_for_mean.items()}

        # Add example traces back to aggregated info
        aggregated_info['example_bid_trace'] = example_bid_trace
        aggregated_info['example_ask_trace'] = example_ask_trace
        aggregated_info['example_gt_bid_trace'] = example_gt_bid_trace
        aggregated_info['example_gt_ask_trace'] = example_gt_ask_trace
        aggregated_info['example_trade_vwap_trace'] = example_trade_vwap_trace
        aggregated_info['example_trade_qty_trace'] = example_trade_qty_trace
        # Forced market order (from best perturbation)
        aggregated_info['example_liquidation_vwap'] = example_liquidation_vwap
        aggregated_info['example_liquidation_qty'] = example_liquidation_qty
        # Historical trade traces
        aggregated_info['example_gt_trade_price'] = example_gt_trade_price
        aggregated_info['example_gt_trade_qty'] = example_gt_trade_qty
        aggregated_info['example_gt_trade_side'] = example_gt_trade_side

        return jnp.mean(pnls), pnls, aggregated_info

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
        
        # Log Doom Logic Parameters
        print("="*60)
        print("[CONFIG] Hybrid Liquidation Logic Enabled")
        print("[CONFIG] 1. Attempt Market Order (Limit 0/Inf) to sweep book")
        print("[CONFIG] 2. Unfilled Quantity Penalty (Doom Price):")
        if self.config.task == 'sell':
            print("[CONFIG]    CELL: 0.75 * Final Best Bid (25% Haircut)")
        else:
            print("[CONFIG]    BUY:  1.25 * Final Best Ask (25% Premium)")
        print("="*60)

        # Log Metric Definitions
        print("="*60)
        print("[CONFIG] Metric Definitions:")
        print(" [Normal Orders] : model_quantity       (Executed by model orders during regular steps)")
        print("                 : submitted_quantity   (Total Quantity Submitted by Model)")
        print("                 : execution_prob       (Executed / Submitted)")
        print(" [Market Orders] : liquidation_quantity (Executed by force_market_order at end)")
        print(" [Doom Orders]   : doom_quantity        (Unfilled quantity, subject to penalty)")
        print(" [Fill Rates]    : quantity / task_size")
        print(" [Agent Trades]  : agent_trades         (Count of agent trade executions/fills)")
        print(" [Total Trades]  : total_trades         (All trades in episode: world + agent + liquidation)")
        print(" [Buffer Trades] : buffer_trades        (Diagnostic: trades in final step buffer only)")
        print(" [Agent Qty]     : agent_quantity       (Total filled quantity = Normal + Market. Range: [0, task_size])")
        print("="*60)

        # Resume from checkpoint if specified
        start_epoch = 0
        best_pnl = -float('inf')
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
                    # Support both old 'best_fitness' and new 'best_pnl' keys for backward compatibility
                    best_pnl = state.get('best_pnl', state.get('best_fitness', -float('inf')))
                    key = jax.random.PRNGKey(self.config.seed)
                    # Fast-forward the key
                    for _ in range(start_epoch):
                        key, _ = jax.random.split(key)
                print(f"[TRAIN] Resumed from epoch {start_epoch}, best_pnl={best_pnl:.4f}")
            except Exception as e:
                print(f"[TRAIN] Warning: Could not resume from {resume_from}: {e}")
                print("[TRAIN] Starting fresh training")

        # Initialize W&B (only on rank 0 in distributed mode)
        wandb_run = None
        if hasattr(self.config, 'wandb_project') and self.config.wandb_project:
            if self._process_index == 0:
                import wandb
                # Get SLURM job ID if available
                job_id = os.environ.get("SLURM_JOB_ID", "local")
                n_procs = jax.process_count() if self._is_distributed else 1
                wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=f"es_n{self.config.n_perturbations}_s{self.config.seed}_j{job_id}",
                    config={
                        'n_perturbations': self.config.n_perturbations,
                        'n_steps': self.config.n_steps,
                        'noiser': self.config.noiser,
                        'sigma': self.config.sigma,
                        'sigma_decay': getattr(self.config, 'sigma_decay', 1.0),
                        'sigma_min': getattr(self.config, 'sigma_min', 0.01),
                        'lr': self.config.lr,
                        'lora_rank': self.config.lora_rank,
                        'checkpoint': self.config.lobs5_checkpoint,
                        'background_mode': self.config.background_mode,
                        'n_processes': n_procs,
                        'n_devices_total': len(jax.devices()),
                        'rank_transform': getattr(self.config, 'rank_transform', False),
                    },
                    resume='allow' if resume_from else None,
                )
                print(f"[TRAIN] W&B initialized: {wandb_run.url}")

        # Get initial state
        initial_sim_state, initial_msg_history = self._create_initial_sim_state()

        # Training loop
        # Sigma decay setup
        sigma_init = self.config.sigma
        sigma_decay = getattr(self.config, 'sigma_decay', 1.0)
        sigma_min = getattr(self.config, 'sigma_min', 0.01)

        for epoch in tqdm(range(start_epoch, n_epochs), desc='ES Training', initial=start_epoch, total=n_epochs):
            key, epoch_key = jax.random.split(key)

            # Apply sigma decay: σ_n = max(σ_0 × decay^n, σ_min)
            if sigma_decay < 1.0:
                current_sigma = max(sigma_init * (sigma_decay ** epoch), sigma_min)
                self.noiser_params["sigma"] = current_sigma
            else:
                current_sigma = sigma_init

            mean_pnl, pnls, epoch_info = self.train_epoch(
                epoch_key, epoch, initial_sim_state, initial_msg_history
            )

            # Track best model (by raw PnL, not rank-transformed fitness)
            is_best = mean_pnl > best_pnl
            if is_best:
                best_pnl = mean_pnl
                # Save best model (only on rank 0 in distributed mode)
                if self._process_index == 0:
                    best_path = os.path.join(checkpoint_dir, 'best')
                    self.save_checkpoint(best_path)
                    self._save_training_state(best_path, epoch, best_pnl)
                    print(f"[TRAIN] New best model saved: pnl={best_pnl:.4f}")

            # Periodic checkpointing (only on rank 0 in distributed mode)
            if (epoch + 1) % checkpoint_every == 0 and self._process_index == 0:
                ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}')
                self.save_checkpoint(ckpt_path)
                self._save_training_state(ckpt_path, epoch, best_pnl)
                # Also save as 'latest' for easy resumption
                latest_path = os.path.join(checkpoint_dir, 'latest')
                self.save_checkpoint(latest_path)
                self._save_training_state(latest_path, epoch, best_pnl)

            # Log to W&B
            if wandb_run:
                # PnL statistics (raw profit/loss in cents)
                pnl_std = float(jnp.std(pnls))
                pnl_max = float(jnp.max(pnls))
                pnl_min = float(jnp.min(pnls))

                # Fitness = rank_transform(pnl) - always compute for ES
                # Range: [-0.5, 0.5] based on relative ranking
                fitnesses = rank_transform(pnls)
                fitness_mean = float(jnp.mean(fitnesses))
                fitness_std = float(jnp.std(fitnesses))

                # Calculate execution rates
                task_size = self.config.task_size if self.config.task_size > 0 else 1.0
                agent_qty = float(epoch_info['agent_quantity'])
                model_qty = float(epoch_info.get('model_quantity', 0))
                liquidation_qty = float(epoch_info.get('liquidation_quantity', 0))
                doom_qty = float(epoch_info.get('doom_quantity', 0))
                submitted_qty = float(epoch_info.get('submitted_quantity', 0))

                fill_rate = agent_qty / task_size                    # Total fill rate (may be < 1.0!)
                model_fill_rate = model_qty / task_size              # Model orders fill rate
                liquidation_fill_rate = liquidation_qty / task_size  # Force market order fill rate
                unfill_rate = doom_qty / task_size               # Unfilled rate (book depth insufficient)
                
                # Execution Probability (Filled / Submitted)
                # Avoid division by zero
                exec_prob = model_qty / (submitted_qty + 1e-9)

                # Generate Best Bid/Ask Price Plot (Market Data Only)
                if 'example_gt_bid_trace' in epoch_info and 'example_gt_ask_trace' in epoch_info:
                    try:
                        import matplotlib.pyplot as plt
                        
                        # Ground Truth (High Res)
                        gt_bid = epoch_info['example_gt_bid_trace']
                        gt_ask = epoch_info['example_gt_ask_trace']
                        
                        # Compute Mid Price
                        # Check for valid prices (non-zero) to avoid weird mid prices
                        valid_mask = (gt_bid > 0) & (gt_ask > 0)
                        gt_mid = (gt_bid + gt_ask) / 2
                        # Handle invalid (zero) values if necessary, but plotting usually handles nans/zeros ok visually 
                        # or we can mask them. For now, plot direct values.
                        
                        # Get configuration for X-axis labeling
                        n_warmup = getattr(self.config, 'n_warmup_msgs', 10)
                        n_bg = getattr(self.config, 'background_msgs_per_step', 50)
                        n_steps = self.config.n_steps

                        # Build merged trace: interleave GT data with agent bid/ask at step boundaries
                        # Each step adds 1 agent data point (post-trade book state), extending total by n_steps
                        import numpy as np
                        agent_bid_data = epoch_info.get('example_bid_trace')
                        agent_ask_data = epoch_info.get('example_ask_trace')
                        if agent_bid_data is not None and agent_ask_data is not None:
                            agent_bid_arr = np.array(agent_bid_data)
                            agent_ask_arr = np.array(agent_ask_data)
                            gt_bid_np = np.array(gt_bid)
                            gt_ask_np = np.array(gt_ask)
                            merged_bid_list = []
                            merged_ask_list = []
                            # Warmup portion (pure GT)
                            merged_bid_list.extend(gt_bid_np[:n_warmup].tolist())
                            merged_ask_list.extend(gt_ask_np[:n_warmup].tolist())
                            # Trading portion: n_bg bg points + 1 agent point per step
                            for s in range(n_steps):
                                gt_start = n_warmup + s * n_bg
                                gt_end = gt_start + n_bg
                                merged_bid_list.extend(gt_bid_np[gt_start:gt_end].tolist())
                                merged_ask_list.extend(gt_ask_np[gt_start:gt_end].tolist())
                                merged_bid_list.append(float(agent_bid_arr[s]))
                                merged_ask_list.append(float(agent_ask_arr[s]))
                            plot_bid = np.array(merged_bid_list)
                            plot_ask = np.array(merged_ask_list)
                            plot_mid = (plot_bid + plot_ask) / 2
                            step_width = n_bg + 1  # 51 per step (50 bg + 1 policy)
                        else:
                            plot_bid = np.array(gt_bid)
                            plot_ask = np.array(gt_ask)
                            plot_mid = np.array(gt_mid)
                            step_width = n_bg

                        # X-axis from -n_warmup (warmup phase is negative)
                        plot_x = list(range(-n_warmup, len(plot_bid) - n_warmup))

                        # Create static plot
                        fig, ax = plt.subplots(figsize=(12, 6))

                        # Plot merged bid/ask/mid (includes agent impact at step boundaries)
                        ax.plot(plot_x, plot_ask, label='Market Ask', color='red', alpha=0.6, linewidth=1.0)
                        ax.plot(plot_x, plot_mid, label='Mid Price', color='black', alpha=0.8, linewidth=1.0, linestyle=':')
                        ax.plot(plot_x, plot_bid, label='Market Bid', color='green', alpha=0.6, linewidth=1.0)

                        # Add vertical separator lines
                        ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Warmup End')
                        for i in range(1, n_steps + 1):
                            ax.axvline(x=i * step_width, color='gray', linestyle=':', alpha=0.3)

                        # Set custom X-axis ticks at warmup start, 0, and each step boundary
                        ticks = [-n_warmup, 0]
                        for i in range(1, n_steps + 1):
                            ticks.append(i * step_width)
                        ax.set_xticks(ticks)

                        ax.set_title(f"Market Trace (Data Window) - Epoch {epoch}")
                        ax.set_xlabel("Message Index (Warmup < 0 | Trading >= 0)")
                        ax.set_ylabel("Price")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Log to WandB
                        wandb_run.log({"market_data_trace": wandb.Image(fig)}, commit=False)
                        plt.close(fig)

                        # Second chart: Market Data with Trade Markers
                        if 'example_trade_vwap_trace' in epoch_info:
                            fig2, ax2 = plt.subplots(figsize=(12, 6))

                            # Plot merged bid/ask/mid (same as first chart, includes agent impact)
                            ax2.plot(plot_x, plot_ask, label='Market Ask', color='red', alpha=0.6, linewidth=1.0)
                            ax2.plot(plot_x, plot_mid, label='Mid Price', color='black', alpha=0.8, linewidth=1.0, linestyle=':')
                            ax2.plot(plot_x, plot_bid, label='Market Bid', color='green', alpha=0.6, linewidth=1.0)

                            # Add vertical separator lines
                            ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Warmup End')
                            for i in range(1, n_steps + 1):
                                ax2.axvline(x=i * step_width, color='gray', linestyle=':', alpha=0.3)

                            # Get trade traces
                            trade_vwap = epoch_info['example_trade_vwap_trace']
                            trade_qty = epoch_info['example_trade_qty_trace']

                            # Get bid/ask traces for execution quality comparison
                            bid_trace_data = epoch_info['example_bid_trace']
                            ask_trace_data = epoch_info['example_ask_trace']

                            # Add trade markers
                            # Determine task type for color logic
                            is_sell_task = getattr(self.config, 'task', 'sell') == 'sell'

                            for step_idx in range(n_steps):
                                qty = float(trade_qty[step_idx])
                                if qty > 0:  # Trade occurred
                                    # X position: at agent point in merged trace
                                    x_pos = step_idx * step_width + n_bg
                                    price = float(trade_vwap[step_idx])
                                    # Use pre-trade bid/ask from merged trace for color comparison
                                    # This is the last bg msg before the agent's order (visually matches the chart curves)
                                    pre_trade_idx = n_warmup + step_idx * step_width + n_bg - 1
                                    if pre_trade_idx < len(plot_bid):
                                        bid = float(plot_bid[pre_trade_idx])
                                        ask = float(plot_ask[pre_trade_idx])
                                    else:
                                        bid = float(bid_trace_data[step_idx])
                                        ask = float(ask_trace_data[step_idx])

                                    # Determine execution quality color
                                    if is_sell_task:
                                        # Sell: higher is better (green=at ask, red=at bid)
                                        if price >= ask:
                                            color = 'limegreen'  # At Ask (best for sell)
                                            marker = '^'         # Up triangle
                                        elif price > bid:
                                            color = 'gold'       # In spread
                                            marker = 'o'         # Circle
                                        else:
                                            color = 'orangered'  # At/Below Bid (worst for sell)
                                            marker = 'v'         # Down triangle
                                    else:
                                        # Buy: lower is better (green=at bid, red=at ask)
                                        if price <= bid:
                                            color = 'limegreen'  # At Bid (best for buy)
                                            marker = 'v'         # Down triangle
                                        elif price < ask:
                                            color = 'gold'       # In spread
                                            marker = 'o'         # Circle
                                        else:
                                            color = 'orangered'  # At/Above Ask (worst for buy)
                                            marker = '^'         # Up triangle

                                    # Marker size proportional to quantity
                                    size = 50 + qty * 3
                                    ax2.scatter(x_pos, price, c=color, s=size, marker=marker,
                                               edgecolors='black', linewidths=0.5, zorder=5)
                                    # Label each trade marker with its volume
                                    ax2.annotate(f'{int(qty)}', (x_pos, price),
                                                 textcoords="offset points", xytext=(0, 12),
                                                 ha='center', fontsize=8, fontweight='bold',
                                                 color='black',
                                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                                           alpha=0.7, edgecolor='none'))

                            # Add forced market order marker (if liquidation occurred)
                            liq_qty = float(epoch_info.get('example_liquidation_qty', 0))
                            if liq_qty > 0:
                                liq_vwap = float(epoch_info['example_liquidation_vwap'])
                                liq_x = n_steps * step_width
                                ax2.scatter(liq_x, liq_vwap, c='magenta', s=120, marker='D',
                                           edgecolors='black', linewidths=1.0, zorder=6)
                                ax2.annotate(f'MO:{int(liq_qty)}', (liq_x, liq_vwap),
                                             textcoords="offset points", xytext=(0, 12),
                                             ha='center', fontsize=8, fontweight='bold',
                                             color='darkmagenta',
                                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                                       alpha=0.7, edgecolor='none'))

                            # Add trade legend
                            from matplotlib.lines import Line2D
                            if is_sell_task:
                                legend_elements = [
                                    Line2D([0], [0], marker='^', color='w', markerfacecolor='limegreen',
                                           markersize=10, label='At Ask (Best Sell)'),
                                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                                           markersize=10, label='In Spread'),
                                    Line2D([0], [0], marker='v', color='w', markerfacecolor='orangered',
                                           markersize=10, label='At/Below Bid'),
                                ]
                            else:
                                legend_elements = [
                                    Line2D([0], [0], marker='v', color='w', markerfacecolor='limegreen',
                                           markersize=10, label='At Bid (Best Buy)'),
                                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                                           markersize=10, label='In Spread'),
                                    Line2D([0], [0], marker='^', color='w', markerfacecolor='orangered',
                                           markersize=10, label='At/Above Ask'),
                                ]
                            legend_elements.append(
                                Line2D([0], [0], marker='D', color='w', markerfacecolor='magenta',
                                       markersize=10, label='Forced Market Order'))
                            # Combine with line legends
                            handles, labels = ax2.get_legend_handles_labels()
                            ax2.legend(handles=legend_elements + handles, loc='upper left')

                            ax2.set_title(f"Market Trace with Trades - Epoch {epoch}")
                            ax2.set_xlabel("Message Index (Warmup < 0 | Trading >= 0)")
                            ax2.set_ylabel("Price")
                            ax2.set_xticks(ticks)
                            ax2.grid(True, alpha=0.3)

                            wandb_run.log({"market_data_trace_with_trades": wandb.Image(fig2)}, commit=False)
                            plt.close(fig2)

                        # ==========================================================
                        # Third chart: Market Data with Historical Trades Only
                        # Shows trades from the original LOBSTER data (event_type=4)
                        # ==========================================================
                        if 'example_gt_trade_price' in epoch_info:
                            gt_trade_price = epoch_info['example_gt_trade_price']
                            gt_trade_qty = epoch_info['example_gt_trade_qty']
                            gt_trade_side = epoch_info['example_gt_trade_side']

                            # Check if we have any historical trades
                            has_trades = jnp.sum(gt_trade_qty > 0) > 0

                            if has_trades:
                                fig3, ax3 = plt.subplots(figsize=(12, 6))

                                # Plot GT (same as other charts)
                                ax3.plot(gt_steps, gt_ask, label='Market Ask', color='red', alpha=0.6, linewidth=1.0)
                                ax3.plot(gt_steps, gt_mid, label='Mid Price', color='black', alpha=0.8, linewidth=1.0, linestyle=':')
                                ax3.plot(gt_steps, gt_bid, label='Market Bid', color='green', alpha=0.6, linewidth=1.0)

                                # Add vertical separator lines
                                ax3.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Warmup End')
                                for i in range(1, n_steps + 1):
                                    ax3.axvline(x=i * n_bg, color='gray', linestyle=':', alpha=0.3)

                                # Plot historical trades
                                # gt_trade_price, gt_trade_qty, gt_trade_side have shape (total_msgs,)
                                # Plot at message index positions (X-axis = gt_steps)
                                for msg_idx in range(len(gt_trade_price)):
                                    qty = float(gt_trade_qty[msg_idx])
                                    if qty > 0:  # Trade occurred
                                        x_pos = gt_steps[msg_idx]  # Message index position
                                        price = float(gt_trade_price[msg_idx])
                                        side = int(gt_trade_side[msg_idx])

                                        # Color by side: Buy = blue, Sell = purple
                                        if side == 1:  # Buy
                                            color = 'dodgerblue'
                                            marker = '^'  # Up triangle
                                        else:  # Sell (side == -1)
                                            color = 'mediumorchid'
                                            marker = 'v'  # Down triangle

                                        # Marker size proportional to quantity
                                        size = 30 + qty * 2
                                        ax3.scatter(x_pos, price, c=color, s=size, marker=marker,
                                                   edgecolors='black', linewidths=0.3, alpha=0.7, zorder=5)

                                # Add historical trade legend
                                from matplotlib.lines import Line2D
                                legend_elements = [
                                    Line2D([0], [0], marker='^', color='w', markerfacecolor='dodgerblue',
                                           markersize=10, label='Historical Buy'),
                                    Line2D([0], [0], marker='v', color='w', markerfacecolor='mediumorchid',
                                           markersize=10, label='Historical Sell'),
                                ]
                                handles, labels = ax3.get_legend_handles_labels()
                                ax3.legend(handles=legend_elements + handles, loc='upper left')

                                ax3.set_title(f"Market Trace with Historical Trades - Epoch {epoch}")
                                ax3.set_xlabel("Message Index (Warmup < 0 | Trading >= 0)")
                                ax3.set_ylabel("Price")
                                ax3.set_xticks(ticks)
                                ax3.grid(True, alpha=0.3)

                                wandb_run.log({"market_data_trace_historical_trades": wandb.Image(fig3)}, commit=False)
                                plt.close(fig3)

                    except Exception as e:
                        print(f"[WARN] Failed to generate price plot: {e}")

                # Build base metrics dict
                metrics = {
                    'epoch': epoch,
                    # PnL metrics (raw profit/loss in cents)
                    'pnl/mean': float(mean_pnl),
                    'pnl/best_ever': float(best_pnl),
                    'pnl/std': pnl_std,
                    'pnl/max': pnl_max,
                    'pnl/min': pnl_min,
                    # Fitness metrics (rank_transform(pnl), range [-0.5, 0.5])
                    'fitness/mean': fitness_mean,
                    'fitness/std': fitness_std,
                    # Per-perturbation distribution histograms
                    'pnl/distribution': wandb.Histogram(pnls.tolist()),
                    'fitness/distribution': wandb.Histogram(fitnesses.tolist()),

                    # Section 1: Normal Orders (Model Steps)
                    'normal_order/quantity': model_qty,
                    'normal_order/fill_rate': model_fill_rate,
                    'normal_order/submitted_quantity': submitted_qty,
                    'normal_order/submission_ratio': submitted_qty / task_size,  # 提交倍数
                    'normal_order/execution_prob': exec_prob,
                    
                    # Section 2: Market Orders (Liquidation Step)
                    'market_order/quantity': liquidation_qty,
                    'market_order/fill_rate': liquidation_fill_rate,
                    
                    # Section 3: Doom Orders (Unfilled Penalty)
                    'doom_order/quantity': doom_qty,
                    'doom_order/fill_rate': unfill_rate,

                    # Section 4: Execution Summary
                    'execution/agent_quantity': agent_qty,
                    'execution/fill_rate': fill_rate,
                    'execution/agent_trades': float(epoch_info['agent_trades']),
                    'execution/total_trades': float(epoch_info['total_trades']),
                    'execution/buffer_trades': float(epoch_info['buffer_trades']),  # Diagnostic: trades in final buffer
                }

                # Add current sigma (useful for tracking sigma decay)
                metrics['es/sigma'] = current_sigma

                # Add full distributions as matplotlib plots (wandb.Image gives per-epoch step slider)
                import numpy as np
                import matplotlib.pyplot as plt

                pnls_np = np.array(pnls)
                fitnesses_np = np.array(fitnesses)

                # PnL distribution
                fig_pnl, ax_pnl = plt.subplots(figsize=(8, 4))
                ax_pnl.hist(pnls_np, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
                ax_pnl.axvline(x=float(np.mean(pnls_np)), color='red', linestyle='--',
                               label=f'Mean={float(np.mean(pnls_np)):.0f}')
                ax_pnl.axvline(x=0, color='black', linestyle=':', alpha=0.5, label='Breakeven')
                ax_pnl.set_title(f'PnL Distribution - Epoch {epoch} (N={len(pnls_np)})')
                ax_pnl.set_xlabel('PnL (cents)')
                ax_pnl.set_ylabel('Count')
                ax_pnl.legend()
                ax_pnl.grid(True, alpha=0.3)
                metrics['pnl/distribution'] = wandb.Image(fig_pnl)
                plt.close(fig_pnl)

                # Fitness distribution
                fig_fit, ax_fit = plt.subplots(figsize=(8, 4))
                ax_fit.hist(fitnesses_np, bins=50, color='coral', edgecolor='black', alpha=0.7)
                ax_fit.axvline(x=0, color='black', linestyle='--', label='Mean=0')
                ax_fit.set_xlim(-0.55, 0.55)
                ax_fit.set_title(f'Fitness Distribution - Epoch {epoch} (N={len(fitnesses_np)})')
                ax_fit.set_xlabel('Fitness (rank_transform)')
                ax_fit.set_ylabel('Count')
                ax_fit.legend()
                ax_fit.grid(True, alpha=0.3)
                metrics['fitness/distribution'] = wandb.Image(fig_fit)
                plt.close(fig_fit)

                wandb_run.log(metrics)

                # Print warning if doom_qty > 0
                if doom_qty > 0:
                    print(f"[WARNING] Epoch {epoch}: {doom_qty:.0f} shares unfilled (unfill_rate={unfill_rate:.1%})")

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: pnl_mean={mean_pnl:.4f}, pnl_best={best_pnl:.4f}, pnl_std={jnp.std(pnls):.4f}")

        # Save final checkpoint
        final_path = os.path.join(checkpoint_dir, 'final')
        self.save_checkpoint(final_path)
        self._save_training_state(final_path, n_epochs - 1, best_pnl)
        print(f"[TRAIN] Final checkpoint saved to {final_path}")

        if wandb_run:
            wandb_run.finish()

        return self.lobs5_init.params

    def _save_training_state(self, path: str, epoch: int, best_pnl: float):
        """Save training state for resumption."""
        import os
        import pickle
        os.makedirs(path, exist_ok=True)
        state = {
            'epoch': epoch,
            'best_pnl': best_pnl,  # Raw PnL in cents (fitness = rank_transform(pnl))
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
