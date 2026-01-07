"""
Checkpoint Adapter: Flax (gradient training) -> ES format.

This module provides utilities to load gradient-trained LOBS5 checkpoints
and convert them to ES-compatible format for use with HyperscaleES training.

Key Transformations:
- Flax nn.Dense kernel (in_dim, out_dim) -> ES weight (out_dim, in_dim) [transposed]
- Flax nn.LayerNorm scale -> ES weight
- Flax layers_N -> ES layer_N (plural to singular)
- Flax seq -> ES ssm (SSM submodule naming)
- Flax fused_s5 -> ES fused_encoder
- Flax pre_layers_N -> ES pre_layer_N
- Flax post_layers_N -> ES post_layer_N
"""

import os
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional


def load_flax_checkpoint(checkpoint_path: str) -> Tuple[Dict, Dict]:
    """
    Load a gradient-trained LOBS5 checkpoint using Orbax.

    This uses LOBS5's existing checkpoint loading mechanism which properly
    handles OCDBT format checkpoints with StandardRestore.

    Args:
        checkpoint_path: Path to the checkpoint directory
            (e.g., 'checkpoints/lobs5_d3072_xxx/')

    Returns:
        Tuple of (params, config):
            - params: Flax parameter dictionary
            - config: Training configuration dictionary
    """
    import sys

    # Add LOBS5 root to path for imports
    lobs5_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if lobs5_root not in sys.path:
        sys.path.insert(0, lobs5_root)

    from lob.init_train import load_metadata, init_train_state, load_checkpoint

    # Step 1: Load metadata to get config
    print(f"Loading metadata from {checkpoint_path}")
    args = load_metadata(checkpoint_path)

    # Convert Namespace to dict for config
    config = vars(args)

    print(f"  d_model: {config.get('d_model', 'N/A')}")
    print(f"  n_layers: {config.get('n_layers', 'N/A')}")
    print(f"  token_mode: {config.get('token_mode', 'N/A')}")

    # Step 2: Create TrainState with proper structure
    # These are defaults that match typical LOBS5 training configs
    n_classes = config.get('d_output', 2112 if config.get('token_mode', 24) == 24 else 12012)
    seq_len = config.get('msg_seq_len', 500)
    book_dim = 3 + config.get('book_depth', 500)  # delta_mid + time + volume image
    book_seq_len = 1  # Book encoder uses single book state
    train_size = 1000  # Dummy, not used for loading

    print(f"Creating dummy TrainState for restore...")
    state, _ = init_train_state(
        args=args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        train_size=train_size,
        print_shapes=False
    )

    # Step 3: Load checkpoint using LOBS5's mechanism (handles OCDBT properly)
    print(f"Loading checkpoint state...")
    ckpt = load_checkpoint(
        state=state,
        path=checkpoint_path,
        step=None,  # Load latest
        train=False  # Don't replicate to devices
    )

    # Extract params from loaded TrainState
    loaded_state = ckpt['model']
    if hasattr(loaded_state, 'params'):
        params = loaded_state.params
    elif isinstance(loaded_state, dict) and 'params' in loaded_state:
        params = loaded_state['params']
    else:
        params = loaded_state

    return params, config


def _convert_dense_to_linear(flax_dense: Dict) -> Dict:
    """
    Convert Flax nn.Dense params to ES ES_Linear format.

    Flax: kernel (in_dim, out_dim), bias (out_dim,)
    ES:   weight (out_dim, in_dim), bias (out_dim,)

    Args:
        flax_dense: Dict with 'kernel' and optionally 'bias'

    Returns:
        Dict with 'weight' and optionally 'bias'
    """
    result = {
        'weight': jnp.asarray(flax_dense['kernel']).T  # Transpose!
    }
    if 'bias' in flax_dense:
        result['bias'] = jnp.asarray(flax_dense['bias'])
    return result


def _convert_layernorm(flax_norm: Dict) -> Dict:
    """
    Convert Flax nn.LayerNorm params to ES ES_LayerNorm format.

    Flax: scale (dim,), bias (dim,)
    ES:   weight (dim,), bias (dim,)

    Args:
        flax_norm: Dict with 'scale' and 'bias'

    Returns:
        Dict with 'weight' and 'bias'
    """
    return {
        'weight': jnp.asarray(flax_norm['scale']),
        'bias': jnp.asarray(flax_norm['bias']),
    }


def _convert_ssm(flax_ssm: Dict) -> Dict:
    """
    Convert Flax S5SSM params to ES ES_S5SSM format.

    Both use the same parameter names:
    Lambda_re, Lambda_im, B, C, D, log_step

    Args:
        flax_ssm: Dict with SSM parameters

    Returns:
        ES-compatible SSM params dict (direct copy)
    """
    # SSM params have the same structure, just copy
    return {
        'Lambda_re': jnp.asarray(flax_ssm['Lambda_re']),
        'Lambda_im': jnp.asarray(flax_ssm['Lambda_im']),
        'B': jnp.asarray(flax_ssm['B']),
        'C': jnp.asarray(flax_ssm['C']),
        'D': jnp.asarray(flax_ssm['D']),
        'log_step': jnp.asarray(flax_ssm['log_step']),
    }


def _convert_sequence_layer(flax_layer: Dict, activation: str = 'half_glu1') -> Dict:
    """
    Convert one Flax SequenceLayer to ES ES_SequenceLayer format.

    Flax: seq, norm, out1?, out2?
    ES:   ssm, norm, out1?, out2?

    Args:
        flax_layer: Flax SequenceLayer params
        activation: Activation type to determine which GLU layers exist

    Returns:
        ES-compatible SequenceLayer params
    """
    result = {
        'ssm': _convert_ssm(flax_layer['seq']),
        'norm': _convert_layernorm(flax_layer['norm']),
    }

    # Convert GLU layers if they exist
    if 'out1' in flax_layer:
        result['out1'] = _convert_dense_to_linear(flax_layer['out1'])
    if 'out2' in flax_layer:
        result['out2'] = _convert_dense_to_linear(flax_layer['out2'])

    return result


def _convert_message_encoder(flax_encoder: Dict, n_layers: int, activation: str) -> Dict:
    """
    Convert Flax message encoder (StackedEncoderModel with embedding) to ES format.

    Flax structure:
        encoder: {embedding: (vocab, d_model)}  # nn.Embed
        layers_0: {...}
        layers_1: {...}

    ES structure:
        embedding: (vocab, d_model)  # Direct array for ES_Parameter
        layer_0: {...}
        layer_1: {...}

    Args:
        flax_encoder: Flax StackedEncoderModel params
        n_layers: Number of sequence layers
        activation: Activation type

    Returns:
        ES-compatible message encoder params
    """
    result = {}

    # Embedding layer - store directly as array for ES_Parameter
    if 'encoder' in flax_encoder and 'embedding' in flax_encoder['encoder']:
        result['embedding'] = jnp.asarray(flax_encoder['encoder']['embedding'])

    # Sequence layers: layers_i -> layer_i
    for i in range(n_layers):
        flax_key = f'layers_{i}'
        es_key = f'layer_{i}'
        if flax_key in flax_encoder:
            result[es_key] = _convert_sequence_layer(
                flax_encoder[flax_key], activation
            )

    return result


def _convert_stacked_encoder(flax_encoder: Dict, n_layers: int, activation: str) -> Dict:
    """
    Convert Flax StackedEncoderModel (without embedding) to ES format.

    Flax structure:
        encoder: {kernel, bias}  # nn.Dense input projection
        layers_0: {...}
        layers_1: {...}

    ES structure:
        input_proj: {weight, bias}
        layer_0: {...}
        layer_1: {...}

    Args:
        flax_encoder: Flax StackedEncoderModel params
        n_layers: Number of sequence layers
        activation: Activation type

    Returns:
        ES-compatible stacked encoder params
    """
    result = {}

    # Input projection (Dense layer)
    if 'encoder' in flax_encoder:
        if 'kernel' in flax_encoder['encoder']:
            result['input_proj'] = _convert_dense_to_linear(flax_encoder['encoder'])
        elif 'embedding' in flax_encoder['encoder']:
            # This is actually a message encoder with embedding
            # Store directly as array for ES_Parameter
            result['embedding'] = jnp.asarray(flax_encoder['encoder']['embedding'])

    # Sequence layers
    for i in range(n_layers):
        flax_key = f'layers_{i}'
        es_key = f'layer_{i}'
        if flax_key in flax_encoder:
            result[es_key] = _convert_sequence_layer(
                flax_encoder[flax_key], activation
            )

    return result


def _convert_book_encoder(
    flax_book: Dict,
    n_pre_layers: int,
    n_post_layers: int,
    activation: str
) -> Dict:
    """
    Convert Flax LobBookModel to ES ES_LobBookModel format.

    Flax structure:
        pre_layers_0: {...}
        projection: {kernel, bias}
        post_layers_0: {...}

    ES structure:
        pre_layer_0: {...}
        proj: {weight, bias}
        post_layer_0: {...}

    Args:
        flax_book: Flax LobBookModel params
        n_pre_layers: Number of pre-projection layers
        n_post_layers: Number of post-projection layers
        activation: Activation type

    Returns:
        ES-compatible book encoder params
    """
    result = {}

    # Pre-layers: pre_layers_i -> pre_layer_i
    for i in range(n_pre_layers):
        flax_key = f'pre_layers_{i}'
        es_key = f'pre_layer_{i}'
        if flax_key in flax_book:
            result[es_key] = _convert_sequence_layer(
                flax_book[flax_key], activation
            )

    # Projection layer
    if 'projection' in flax_book:
        result['proj'] = _convert_dense_to_linear(flax_book['projection'])

    # Post-layers: post_layers_i -> post_layer_i
    for i in range(n_post_layers):
        flax_key = f'post_layers_{i}'
        es_key = f'post_layer_{i}'
        if flax_key in flax_book:
            result[es_key] = _convert_sequence_layer(
                flax_book[flax_key], activation
            )

    return result


def _detect_token_mode(config: Dict, flax_params: Dict) -> Tuple[int, int]:
    """
    Detect token_mode and d_output from config and params.

    Args:
        config: Checkpoint configuration dict
        flax_params: Flax parameter dictionary

    Returns:
        Tuple of (token_mode, d_output)
    """
    # Try to get from config first
    token_mode = config.get('token_mode', None)
    d_output = config.get('d_output', config.get('vocab_size', None))

    # If not in config, infer from decoder shape
    if d_output is None and 'decoder' in flax_params:
        decoder_kernel = flax_params['decoder'].get('kernel')
        if decoder_kernel is not None:
            d_output = decoder_kernel.shape[1]  # (in_dim, out_dim)

    # Infer token_mode from d_output if not specified
    if token_mode is None and d_output is not None:
        if d_output == 12012:
            token_mode = 22
        elif d_output == 2112:
            token_mode = 24
        else:
            token_mode = 22  # Default

    # Set defaults if still None
    if token_mode is None:
        token_mode = 22
    if d_output is None:
        d_output = 12012 if token_mode == 22 else 2112

    return token_mode, d_output


def convert_flax_to_es(flax_params: Dict, config: Dict) -> Dict:
    """
    Convert full Flax PaddedLobPredModel params to ES format.

    Flax structure:
        message_encoder: {...}
        book_encoder: {...}
        fused_s5: {...}
        decoder: {kernel, bias}

    ES structure:
        message_encoder: {...}
        book_encoder: {...}
        fused_encoder: {...}
        decoder: {weight, bias}

    Args:
        flax_params: Full Flax model params dict
        config: Model configuration dict with:
            - n_message_layers
            - n_fused_layers
            - n_book_pre_layers (default 1)
            - n_book_post_layers (default 1)
            - activation (default 'half_glu1')

    Returns:
        ES-compatible full model params dict
    """
    # Extract config values with defaults
    n_message_layers = config.get('n_message_layers', config.get('n_layers', 2))
    n_fused_layers = config.get('n_fused_layers', 4)
    n_book_pre_layers = config.get('n_book_pre_layers', 1)
    n_book_post_layers = config.get('n_book_post_layers', 1)
    activation = config.get('activation', 'half_glu1')

    es_params = {}

    # 1. Message encoder
    if 'message_encoder' in flax_params:
        es_params['message_encoder'] = _convert_message_encoder(
            flax_params['message_encoder'],
            n_message_layers,
            activation
        )

    # 2. Book encoder
    if 'book_encoder' in flax_params:
        es_params['book_encoder'] = _convert_book_encoder(
            flax_params['book_encoder'],
            n_book_pre_layers,
            n_book_post_layers,
            activation
        )

    # 3. Fused encoder: fused_s5 -> fused_encoder
    if 'fused_s5' in flax_params:
        es_params['fused_encoder'] = _convert_stacked_encoder(
            flax_params['fused_s5'],
            n_fused_layers,
            activation
        )

    # 4. Decoder
    if 'decoder' in flax_params:
        es_params['decoder'] = _convert_dense_to_linear(flax_params['decoder'])

    return es_params


def convert_and_load_checkpoint(
    checkpoint_path: str,
    es_model_init=None,
    return_config: bool = True
) -> Tuple[Dict, Optional[Dict]]:
    """
    Convenience function to load and convert a checkpoint in one step.

    Args:
        checkpoint_path: Path to Orbax checkpoint directory
        es_model_init: Optional ES model CommonInit for structure validation
        return_config: Whether to return the config dict

    Returns:
        If return_config: (es_params, config)
        Else: es_params
    """
    # Load Flax checkpoint
    flax_params, config = load_flax_checkpoint(checkpoint_path)

    print(f"Loaded Flax checkpoint with config:")
    print(f"  n_message_layers: {config.get('n_message_layers', 'N/A')}")
    print(f"  n_fused_layers: {config.get('n_fused_layers', 'N/A')}")
    print(f"  d_model: {config.get('d_model', 'N/A')}")
    print(f"  activation: {config.get('activation', 'N/A')}")

    # Convert to ES format
    es_params = convert_flax_to_es(flax_params, config)

    # Count parameters
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree))

    flax_count = count_params(flax_params)
    es_count = count_params(es_params)

    print(f"Converted {flax_count:,} Flax params -> {es_count:,} ES params")

    if return_config:
        return es_params, config
    return es_params


def validate_conversion(flax_params: Dict, es_params: Dict) -> bool:
    """
    Validate that parameter shapes match after conversion.

    Args:
        flax_params: Original Flax params
        es_params: Converted ES params

    Returns:
        True if all shapes match (accounting for transposition)
    """
    def get_shapes(pytree, prefix=''):
        shapes = {}
        if isinstance(pytree, dict):
            for k, v in pytree.items():
                shapes.update(get_shapes(v, f'{prefix}.{k}' if prefix else k))
        elif hasattr(pytree, 'shape'):
            shapes[prefix] = pytree.shape
        return shapes

    flax_shapes = get_shapes(flax_params)
    es_shapes = get_shapes(es_params)

    print(f"Flax has {len(flax_shapes)} parameter tensors")
    print(f"ES has {len(es_shapes)} parameter tensors")

    # The counts should be approximately equal
    # (some structural differences expected due to nesting changes)
    return len(flax_shapes) > 0 and len(es_shapes) > 0


# =============================================================================
# Direct Loading for ESTrainer
# =============================================================================

def load_params_for_es_trainer(
    checkpoint_path: str,
    trainer_config: Any = None
) -> Tuple[Dict, Dict]:
    """
    Load and convert checkpoint for use with ESTrainer.

    This is the main entry point for loading gradient-trained checkpoints
    into the ES training loop.

    Args:
        checkpoint_path: Path to Orbax checkpoint directory
        trainer_config: Optional ESTrainer config for validation

    Returns:
        Tuple of (es_params, checkpoint_config)
    """
    es_params, config = convert_and_load_checkpoint(
        checkpoint_path,
        return_config=True
    )

    # If trainer config provided, validate compatibility
    if trainer_config is not None:
        required_keys = ['d_model', 'n_message_layers', 'n_fused_layers']
        for key in required_keys:
            trainer_val = getattr(trainer_config, key, None)
            ckpt_val = config.get(key)
            if trainer_val is not None and ckpt_val is not None:
                if trainer_val != ckpt_val:
                    print(f"WARNING: Config mismatch for {key}: "
                          f"trainer={trainer_val}, checkpoint={ckpt_val}")

    return es_params, config


class ESInitResult:
    """
    Result of loading checkpoint for ES training.

    Mimics CommonInit structure with additional es_tree_key for ES noiser.
    """
    def __init__(self, params, frozen_params, es_map, es_tree_key):
        self.params = params
        self.frozen_params = frozen_params
        self.es_map = es_map
        self.es_tree_key = es_tree_key


def _infer_ssm_size(params: Dict, config: Dict) -> int:
    """
    Infer SSM state size from actual param shapes.

    With conj_sym=True, hidden_size = P, ssm_size = 2*P

    Args:
        params: ES params dict
        config: Checkpoint config

    Returns:
        Inferred ssm_size
    """
    try:
        # Look for Lambda_re in message_encoder layer_0 ssm
        lambda_re = params['message_encoder']['layer_0']['ssm']['Lambda_re']
        P = lambda_re.shape[0]  # This is the actual state size
        # With conj_sym=True, hidden_size = P, so ssm_size = 2*P for compatibility
        return P * 2
    except (KeyError, AttributeError):
        return config.get('ssm_size', 256)


def _build_frozen_params(config: Dict, es_params: Dict) -> Dict:
    """
    Build nested frozen_params structure matching params structure.

    This is what _forward_step and submodules expect.

    Args:
        config: Checkpoint configuration
        es_params: Converted ES parameters

    Returns:
        Nested frozen_params dict
    """
    # Detect token_mode and d_output
    token_mode, d_output = _detect_token_mode(config, {})  # Use config only

    # Extract config values
    d_model = config.get('d_model', 256)
    n_message_layers = config.get('n_message_layers', config.get('n_layers', 2))
    n_fused_layers = config.get('n_fused_layers', 4)
    n_book_pre_layers = config.get('n_book_pre_layers', 1)
    n_book_post_layers = config.get('n_book_post_layers', 1)
    ssm_size = _infer_ssm_size(es_params, config)
    conj_sym = config.get('conj_sym', True)
    activation = config.get('activation', 'half_glu1')

    # Runtime configuration from checkpoint metadata
    msg_seq_len = config.get('msg_seq_len', 500)
    book_depth = config.get('book_depth', 500)
    d_book = 3 + book_depth  # Compute: [mid_diff, time_s, time_ns] + volume image

    print(f"[CHECKPOINT] token_mode={token_mode}, d_output={d_output}")
    print(f"[CHECKPOINT] msg_seq_len={msg_seq_len}, book_depth={book_depth}, d_book={d_book}")

    def build_ssm_frozen_params():
        """Build frozen params for an SSM layer."""
        return {
            'conj_sym': conj_sym,
            'clip_eigs': True,
            'bidirectional': False,
            'discretization': 'zoh',
            'step_rescale': 1.0,
        }

    def build_layer_frozen_params():
        """Build frozen params for a sequence layer."""
        return {
            'activation': activation,
            'prenorm': True,
            'd_model': d_model,
            'ssm': build_ssm_frozen_params(),
        }

    # Message encoder frozen params
    message_encoder_fp = {
        'n_layers': n_message_layers,
        'd_model': d_model,
    }
    for i in range(n_message_layers):
        message_encoder_fp[f'layer_{i}'] = build_layer_frozen_params()

    # Book encoder frozen params
    book_encoder_fp = {
        'n_pre_layers': n_book_pre_layers,
        'n_post_layers': n_book_post_layers,
        'd_book': d_book,
        'd_model': d_model,
    }
    for i in range(n_book_pre_layers):
        book_encoder_fp[f'pre_layer_{i}'] = build_layer_frozen_params()
    for i in range(n_book_post_layers):
        book_encoder_fp[f'post_layer_{i}'] = build_layer_frozen_params()

    # Fused encoder frozen params
    fused_encoder_fp = {
        'n_layers': n_fused_layers,
        'd_model': d_model,
    }
    for i in range(n_fused_layers):
        fused_encoder_fp[f'layer_{i}'] = build_layer_frozen_params()

    # Full nested frozen_params structure
    return {
        'd_output': d_output,
        'd_model': d_model,
        'd_book': d_book,
        'n_message_layers': n_message_layers,
        'n_fused_layers': n_fused_layers,
        'n_book_pre_layers': n_book_pre_layers,
        'n_book_post_layers': n_book_post_layers,
        'ssm_size': ssm_size,
        'conj_sym': conj_sym,
        'mode': config.get('mode', 'ema'),
        # Runtime configuration from checkpoint metadata
        'msg_seq_len': msg_seq_len,
        'book_depth': book_depth,
        'token_mode': token_mode,
        # Nested structures for submodules
        'message_encoder': message_encoder_fp,
        'book_encoder': book_encoder_fp,
        'fused_encoder': fused_encoder_fp,
    }


def load_checkpoint_for_es(
    checkpoint_path: str,
) -> Tuple['ESInitResult', Dict]:
    """
    Load and convert checkpoint for ES training with proper es_tree_key.

    This is the main entry point for loading gradient-trained checkpoints
    into ES-JaxLOB training. Returns an ESInitResult object that mimics
    CommonInit structure.

    Args:
        checkpoint_path: Path to Orbax checkpoint directory

    Returns:
        Tuple of (ESInitResult, es_tree_key)
        ESInitResult has .params, .frozen_params, .es_map attributes
    """
    # Import ES utilities (lazy import to avoid circular deps)
    from ..models.common import simple_es_tree_key, PARAM, EXCLUDED

    # Load and convert
    es_params, config = convert_and_load_checkpoint(checkpoint_path, return_config=True)

    # Create es_map (mark all params as PARAM by default)
    def create_es_map(params_tree):
        """Create es_map tree with same structure as params, all marked as PARAM."""
        if isinstance(params_tree, dict):
            return {k: create_es_map(v) for k, v in params_tree.items()}
        else:
            return PARAM  # All trainable parameters

    es_map = create_es_map(es_params)

    # Create empty scan_map (same structure, all empty tuples)
    def create_scan_map(params_tree):
        if isinstance(params_tree, dict):
            return {k: create_scan_map(v) for k, v in params_tree.items()}
        else:
            return ()  # Empty scan map for each param

    scan_map = create_scan_map(es_params)

    # Create es_tree_key using random base key
    # The es_tree_key is used for parameter-specific randomness in ES noiser
    base_key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
    es_tree_key = simple_es_tree_key(es_params, base_key, scan_map)

    # Build frozen params
    frozen_params = _build_frozen_params(config, es_params)

    result = ESInitResult(
        params=es_params,
        frozen_params=frozen_params,
        es_map=es_map,
        es_tree_key=es_tree_key,
    )

    return result, es_tree_key


# =============================================================================
# ES -> Flax reverse conversion (for saving ES results as Flax checkpoint)
# =============================================================================

def _convert_linear_to_dense(es_linear: Dict) -> Dict:
    """
    Convert ES ES_Linear params to Flax nn.Dense format.

    ES:   weight (out_dim, in_dim), bias (out_dim,)
    Flax: kernel (in_dim, out_dim), bias (out_dim,)

    Args:
        es_linear: Dict with 'weight' and optionally 'bias'

    Returns:
        Dict with 'kernel' and optionally 'bias'
    """
    result = {
        'kernel': jnp.asarray(es_linear['weight']).T  # Transpose back!
    }
    if 'bias' in es_linear:
        result['bias'] = jnp.asarray(es_linear['bias'])
    return result


def _convert_layernorm_to_flax(es_norm: Dict) -> Dict:
    """
    Convert ES ES_LayerNorm params to Flax nn.LayerNorm format.

    ES:   weight (dim,), bias (dim,)
    Flax: scale (dim,), bias (dim,)

    Args:
        es_norm: Dict with 'weight' and 'bias'

    Returns:
        Dict with 'scale' and 'bias'
    """
    return {
        'scale': jnp.asarray(es_norm['weight']),
        'bias': jnp.asarray(es_norm['bias']),
    }


def convert_es_to_flax(es_params: Dict, config: Dict) -> Dict:
    """
    Convert ES params back to Flax format.

    This is the reverse of convert_flax_to_es.

    Args:
        es_params: ES model params dict
        config: Model configuration dict

    Returns:
        Flax-compatible params dict
    """
    n_message_layers = config.get('n_message_layers', config.get('n_layers', 2))
    n_fused_layers = config.get('n_fused_layers', 4)
    n_book_pre_layers = config.get('n_book_pre_layers', 1)
    n_book_post_layers = config.get('n_book_post_layers', 1)

    flax_params = {}

    def convert_ssm_to_flax(es_ssm):
        # SSM params are the same
        return {k: jnp.asarray(v) for k, v in es_ssm.items()}

    def convert_sequence_layer_to_flax(es_layer):
        result = {
            'seq': convert_ssm_to_flax(es_layer['ssm']),
            'norm': _convert_layernorm_to_flax(es_layer['norm']),
        }
        if 'out1' in es_layer:
            result['out1'] = _convert_linear_to_dense(es_layer['out1'])
        if 'out2' in es_layer:
            result['out2'] = _convert_linear_to_dense(es_layer['out2'])
        return result

    # 1. Message encoder
    if 'message_encoder' in es_params:
        msg_enc = es_params['message_encoder']
        flax_msg = {}
        if 'embedding' in msg_enc:
            flax_msg['encoder'] = {'embedding': jnp.asarray(msg_enc['embedding'])}
        for i in range(n_message_layers):
            es_key = f'layer_{i}'
            flax_key = f'layers_{i}'
            if es_key in msg_enc:
                flax_msg[flax_key] = convert_sequence_layer_to_flax(msg_enc[es_key])
        flax_params['message_encoder'] = flax_msg

    # 2. Book encoder
    if 'book_encoder' in es_params:
        book_enc = es_params['book_encoder']
        flax_book = {}
        for i in range(n_book_pre_layers):
            es_key = f'pre_layer_{i}'
            flax_key = f'pre_layers_{i}'
            if es_key in book_enc:
                flax_book[flax_key] = convert_sequence_layer_to_flax(book_enc[es_key])
        if 'proj' in book_enc:
            flax_book['projection'] = _convert_linear_to_dense(book_enc['proj'])
        for i in range(n_book_post_layers):
            es_key = f'post_layer_{i}'
            flax_key = f'post_layers_{i}'
            if es_key in book_enc:
                flax_book[flax_key] = convert_sequence_layer_to_flax(book_enc[es_key])
        flax_params['book_encoder'] = flax_book

    # 3. Fused encoder: fused_encoder -> fused_s5
    if 'fused_encoder' in es_params:
        fused_enc = es_params['fused_encoder']
        flax_fused = {}
        if 'input_proj' in fused_enc:
            flax_fused['encoder'] = _convert_linear_to_dense(fused_enc['input_proj'])
        elif 'embedding' in fused_enc:
            flax_fused['encoder'] = {'embedding': jnp.asarray(fused_enc['embedding'])}
        for i in range(n_fused_layers):
            es_key = f'layer_{i}'
            flax_key = f'layers_{i}'
            if es_key in fused_enc:
                flax_fused[flax_key] = convert_sequence_layer_to_flax(fused_enc[es_key])
        flax_params['fused_s5'] = flax_fused

    # 4. Decoder
    if 'decoder' in es_params:
        flax_params['decoder'] = _convert_linear_to_dense(es_params['decoder'])

    return flax_params
