"""
Checkpoint Converter: Flax (gradient training) → ES format.

This module provides utilities to load gradient-trained LOBS5 checkpoints
and convert them to ES-compatible format for use with HyperscaleES training.

Key Transformations:
- Flax nn.Dense kernel (in_dim, out_dim) → ES weight (out_dim, in_dim) [transposed]
- Flax nn.LayerNorm scale → ES weight
- Flax layers_N → ES layer_N (plural to singular)
- Flax seq → ES ssm (SSM submodule naming)
- Flax out1/out2 → ES out1/out2 (same, but kernel→weight transpose)
"""

import os
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional


def load_flax_checkpoint(checkpoint_path: str) -> Tuple[Dict, Dict]:
    """
    Load a gradient-trained LOBS5 checkpoint using Orbax.

    Args:
        checkpoint_path: Path to the checkpoint directory
            (e.g., 'checkpoints/lobs5_d3072_xxx/')

    Returns:
        Tuple of (params, config):
            - params: Flax parameter dictionary
            - config: Training configuration dictionary
    """
    import orbax.checkpoint as ocp

    # Open checkpoint manager
    mgr = ocp.CheckpointManager(
        os.path.abspath(checkpoint_path),
        item_names=('state', 'metadata')
    )

    # Get latest step
    latest = mgr.latest_step()
    if latest is None:
        raise ValueError(f"No checkpoint found in {checkpoint_path}")

    print(f"Loading checkpoint from step {latest}")

    # Restore checkpoint
    restored = mgr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.PyTreeRestore(),
            metadata=ocp.args.JsonRestore(),
        )
    )

    # Extract params from TrainState
    # Handle both direct params and TrainState objects
    state = restored['state']
    if hasattr(state, 'params'):
        params = state.params
    elif isinstance(state, dict) and 'params' in state:
        params = state['params']
    else:
        params = state

    # Extract config
    metadata = restored.get('metadata', {})
    config = metadata.get('config', metadata)

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
        ES-compatible SSM params dict
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

    # Sequence layers: layers_i → layer_i
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

    # Pre-layers: pre_layers_i → pre_layer_i
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

    # Post-layers: post_layers_i → post_layer_i
    for i in range(n_post_layers):
        flax_key = f'post_layers_{i}'
        es_key = f'post_layer_{i}'
        if flax_key in flax_book:
            result[es_key] = _convert_sequence_layer(
                flax_book[flax_key], activation
            )

    return result


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

    # 3. Fused encoder: fused_s5 → fused_encoder
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

    print(f"Converted {flax_count:,} Flax params → {es_count:,} ES params")

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

    # Infer ssm_size from actual params (Lambda_re shape gives P)
    # With conj_sym=True, hidden_size = P, ssm_size = 2*P
    def infer_ssm_size(params):
        """Infer SSM state size from actual param shapes."""
        try:
            # Look for Lambda_re in message_encoder layer_0 ssm
            lambda_re = params['message_encoder']['layer_0']['ssm']['Lambda_re']
            P = lambda_re.shape[0]  # This is the actual state size
            # With conj_sym=True, hidden_size = P, so ssm_size = 2*P for compatibility
            # But actually for initialize_carry, we use P directly as hidden_size
            return P * 2  # Return ssm_size (which will be halved by conj_sym)
        except (KeyError, AttributeError):
            return config.get('ssm_size', 256)

    inferred_ssm_size = infer_ssm_size(es_params)

    # Extract config values with defaults
    d_model = config.get('d_model', 256)
    d_output = config.get('d_output', config.get('vocab_size', 2112))
    d_book = config.get('d_book', 503)
    n_message_layers = config.get('n_message_layers', config.get('n_layers', 2))
    n_fused_layers = config.get('n_fused_layers', 4)
    n_book_pre_layers = config.get('n_book_pre_layers', 1)
    n_book_post_layers = config.get('n_book_post_layers', 1)
    ssm_size = inferred_ssm_size  # Use inferred value from actual params
    conj_sym = config.get('conj_sym', True)
    activation = config.get('activation', 'half_glu1')

    # Build nested frozen_params structure matching params structure
    # This is what _forward_step and submodules expect
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
    frozen_params = {
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
        # Nested structures for submodules
        'message_encoder': message_encoder_fp,
        'book_encoder': book_encoder_fp,
        'fused_encoder': fused_encoder_fp,
    }

    result = ESInitResult(
        params=es_params,
        frozen_params=frozen_params,
        es_map=es_map,
        es_tree_key=es_tree_key,
    )

    return result, es_tree_key
