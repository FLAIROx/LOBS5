"""
ES-compatible Stacked Encoder Model.

Stacks multiple ES_SequenceLayer with input projection.

Based on: s5/seq_model.py
"""

import jax
import jax.numpy as jnp
from functools import partial

from .common import (
    Model, CommonInit, CommonParams,
    PARAM, MM_PARAM, EXCLUDED,
    merge_inits, merge_frozen, call_submodule,
    ES_Parameter, ES_Linear, ES_LayerNorm,
)
from .s5_ssm import init_hippo_matrices
from .sequence_layer import ES_SequenceLayer

__all__ = ['ES_StackedEncoder']


class ES_StackedEncoder(Model):
    """
    ES-compatible stacked encoder model.

    Architecture:
        Input → Dense (input projection) → [SequenceLayer] × n_layers → Output

    Supports:
        - Multiple stacked S5 layers
        - Input projection
        - RNN mode for step-by-step inference
    """

    @classmethod
    def rand_init(
        cls,
        key,
        d_input: int,
        d_model: int,
        n_layers: int,
        # SSM params
        ssm_size: int,
        blocks: int,
        C_init: str = 'trunc_standard_normal',
        discretization: str = 'zoh',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conj_sym: bool = True,
        clip_eigs: bool = True,
        bidirectional: bool = False,
        step_rescale: float = 1.0,
        # Layer params
        activation: str = 'gelu',
        prenorm: bool = False,
        dtype=jnp.float32,
    ) -> CommonInit:
        """
        Initialize stacked encoder.

        Args:
            key: JAX random key
            d_input: Input dimension
            d_model: Hidden dimension (H)
            n_layers: Number of sequence layers
            ssm_size: State space size (P * blocks)
            blocks: Number of SSM blocks (J)
            C_init: C matrix initialization
            discretization: 'zoh' or 'bilinear'
            dt_min, dt_max: Timescale range
            conj_sym: Conjugate symmetry
            clip_eigs: Clip eigenvalues
            bidirectional: Bidirectional processing
            step_rescale: Timescale scaling
            activation: Activation function
            prenorm: Use pre-normalization
            dtype: Data type

        Returns:
            CommonInit
        """
        keys = jax.random.split(key, n_layers + 2)

        # Initialize HiPPO matrices (shared across layers)
        Lambda_re_init, Lambda_im_init, V, Vinv = init_hippo_matrices(
            ssm_size, blocks, conj_sym
        )

        # Input projection (GPT-style init: stddev = 0.02 / sqrt(n_layers))
        gpt_scale = 0.02 / jnp.sqrt(n_layers)
        input_proj_init = ES_Linear.rand_init(
            keys[0], d_input, d_model, use_bias=True, dtype=dtype, scale=gpt_scale
        )

        # Initialize sequence layers
        layer_inits = {}
        for i in range(n_layers):
            layer_init = ES_SequenceLayer.rand_init(
                keys[i + 1],
                d_model=d_model,
                ssm_size=ssm_size,
                Lambda_re_init=Lambda_re_init,
                Lambda_im_init=Lambda_im_init,
                V=V,
                Vinv=Vinv,
                blocks=blocks,
                C_init=C_init,
                discretization=discretization,
                dt_min=dt_min,
                dt_max=dt_max,
                conj_sym=conj_sym,
                clip_eigs=clip_eigs,
                bidirectional=bidirectional,
                step_rescale=step_rescale,
                activation=activation,
                prenorm=prenorm,
                dtype=dtype,
            )
            layer_inits[f'layer_{i}'] = layer_init

        # Merge all initializations
        merged = merge_inits(
            input_proj=input_proj_init,
            **layer_inits,
        )

        # Add frozen params
        return merge_frozen(
            merged,
            n_layers=n_layers,
            d_model=d_model,
            ssm_size=ssm_size,
            conj_sym=conj_sym,
        )

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        """
        Forward pass through stacked encoder.

        Args:
            common_params: CommonParams with noiser and params
            x: Input sequence (L, d_input)

        Returns:
            Output sequence (L, d_model)
        """
        fp = common_params.frozen_params
        n_layers = fp['n_layers']

        # Input projection
        x = call_submodule(ES_Linear, 'input_proj', common_params, x)

        # Pass through sequence layers
        for i in range(n_layers):
            x = call_submodule(ES_SequenceLayer, f'layer_{i}', common_params, x)

        return x

    @classmethod
    def _forward_rnn(cls, common_params: CommonParams, hiddens, x, resets=None):
        """
        RNN mode forward pass.

        Args:
            common_params: CommonParams with noiser and params
            hiddens: List of hidden states, one per layer
            x: Input sequence (L, d_input)
            resets: Optional reset signals (L,)

        Returns:
            (new_hiddens, output_sequence)
        """
        fp = common_params.frozen_params
        n_layers = fp['n_layers']

        # Input projection
        x = call_submodule(ES_Linear, 'input_proj', common_params, x)

        # Pass through sequence layers
        new_hiddens = []
        for i in range(n_layers):
            layer_params = common_params._replace(
                frozen_params=common_params.frozen_params.get(f'layer_{i}', {}),
                params=common_params.params[f'layer_{i}'],
                es_tree_key=common_params.es_tree_key[f'layer_{i}'],
            )
            hidden_i, x = ES_SequenceLayer._forward_rnn(
                layer_params, hiddens[i], x, resets
            )
            new_hiddens.append(hidden_i)

        return new_hiddens, x

    @staticmethod
    def initialize_carry(batch_size, ssm_size, n_layers, conj_sym=True):
        """
        Initialize hidden states for RNN mode.

        Args:
            batch_size: Batch size
            ssm_size: State space size
            n_layers: Number of layers
            conj_sym: Whether conjugate symmetry is used

        Returns:
            List of hidden states
        """
        if conj_sym:
            hidden_size = ssm_size // 2
        else:
            hidden_size = ssm_size

        return [
            jnp.zeros((batch_size, 1, hidden_size), dtype=jnp.complex64)
            for _ in range(n_layers)
        ]


class ES_StackedEncoderWithScan(Model):
    """
    ES-compatible stacked encoder using jax.lax.scan for efficiency.

    Uses scan over layers with shared scan_map for vmap compatibility.
    This is more memory-efficient for deep models.
    """

    @classmethod
    def rand_init(
        cls,
        key,
        d_input: int,
        d_model: int,
        n_layers: int,
        ssm_size: int,
        blocks: int,
        C_init: str = 'trunc_standard_normal',
        discretization: str = 'zoh',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conj_sym: bool = True,
        clip_eigs: bool = True,
        bidirectional: bool = False,
        step_rescale: float = 1.0,
        activation: str = 'gelu',
        prenorm: bool = False,
        dtype=jnp.float32,
    ) -> CommonInit:
        """
        Initialize stacked encoder with scan-compatible parameter layout.

        Parameters are stacked along axis 0 for efficient scanning.
        """
        keys = jax.random.split(key, n_layers + 2)

        # Initialize HiPPO matrices
        Lambda_re_init, Lambda_im_init, V, Vinv = init_hippo_matrices(
            ssm_size, blocks, conj_sym
        )

        # Input projection
        gpt_scale = 0.02 / jnp.sqrt(n_layers)
        input_proj_init = ES_Linear.rand_init(
            keys[0], d_input, d_model, use_bias=True, dtype=dtype, scale=gpt_scale
        )

        # Initialize all layers and stack parameters
        layer_inits = []
        for i in range(n_layers):
            layer_init = ES_SequenceLayer.rand_init(
                keys[i + 1],
                d_model=d_model,
                ssm_size=ssm_size,
                Lambda_re_init=Lambda_re_init,
                Lambda_im_init=Lambda_im_init,
                V=V,
                Vinv=Vinv,
                blocks=blocks,
                C_init=C_init,
                discretization=discretization,
                dt_min=dt_min,
                dt_max=dt_max,
                conj_sym=conj_sym,
                clip_eigs=clip_eigs,
                bidirectional=bidirectional,
                step_rescale=step_rescale,
                activation=activation,
                prenorm=prenorm,
                dtype=dtype,
            )
            layer_inits.append(layer_init)

        # Stack layer params along axis 0
        def stack_params(inits):
            """Stack parameters from list of CommonInit."""
            stacked = {}
            for key in inits[0].params:
                if isinstance(inits[0].params[key], dict):
                    stacked[key] = stack_params([
                        CommonInit(None, init.params[key], None, None)
                        for init in inits
                    ])
                else:
                    stacked[key] = jnp.stack([init.params[key] for init in inits])
            return stacked

        def stack_es_map(inits):
            """Stack es_map with (0,) prepended for layer dimension."""
            stacked = {}
            for key in inits[0].es_map:
                if isinstance(inits[0].es_map[key], dict):
                    stacked[key] = stack_es_map([
                        CommonInit(None, None, None, init.es_map[key])
                        for init in inits
                    ])
                else:
                    stacked[key] = inits[0].es_map[key]  # Same type for all layers
            return stacked

        def stack_scan_map(inits):
            """Stack scan_map with (0,) for layer dimension."""
            stacked = {}
            for key in inits[0].scan_map:
                if isinstance(inits[0].scan_map[key], dict):
                    stacked[key] = stack_scan_map([
                        CommonInit(None, None, init.scan_map[key], None)
                        for init in inits
                    ])
                else:
                    # Add layer dimension (0,) to existing scan_map
                    stacked[key] = (0,) + inits[0].scan_map[key]
            return stacked

        stacked_layer_params = stack_params(layer_inits)
        stacked_layer_es_map = stack_es_map(layer_inits)
        stacked_layer_scan_map = stack_scan_map(layer_inits)

        # Combine with input projection
        params = {
            'input_proj': input_proj_init.params,
            'layers': stacked_layer_params,
        }
        es_map = {
            'input_proj': input_proj_init.es_map,
            'layers': stacked_layer_es_map,
        }
        scan_map = {
            'input_proj': input_proj_init.scan_map,
            'layers': stacked_layer_scan_map,
        }

        frozen_params = {
            'n_layers': n_layers,
            'd_model': d_model,
            'ssm_size': ssm_size,
            'conj_sym': conj_sym,
            # Include layer frozen params (same for all layers)
            'layer_config': layer_inits[0].frozen_params,
        }

        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        """
        Forward pass using jax.lax.scan over layers.
        """
        fp = common_params.frozen_params
        n_layers = fp['n_layers']

        # Input projection
        input_proj_params = common_params._replace(
            frozen_params=None,
            params=common_params.params['input_proj'],
            es_tree_key=common_params.es_tree_key['input_proj'],
        )
        x = ES_Linear._forward(input_proj_params, x)

        # Scan over layers
        def scan_fn(x, layer_inputs):
            layer_params, layer_keys = layer_inputs
            layer_common = common_params._replace(
                frozen_params=fp.get('layer_config', {}),
                params=layer_params,
                es_tree_key=layer_keys,
            )
            x = ES_SequenceLayer._forward(layer_common, x)
            return x, None

        x, _ = jax.lax.scan(
            scan_fn, x,
            (common_params.params['layers'], common_params.es_tree_key['layers'])
        )

        return x
