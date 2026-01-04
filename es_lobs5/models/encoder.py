"""
ES-compatible Stacked Encoder Model.

Stacks multiple ES_SequenceLayer with input projection.
"""

import jax
import jax.numpy as jnp

from .common import (
    Model, CommonInit, CommonParams,
    PARAM, MM_PARAM, EXCLUDED,
    merge_inits, merge_frozen, call_submodule,
    ES_Parameter, ES_Linear, ES_LayerNorm,
)
from .s5_layer import ES_SequenceLayer
from ..adapters.hippo_adapter import get_hippo_params

__all__ = ['ES_StackedEncoder']


class ES_StackedEncoder(Model):
    """
    ES-compatible stacked encoder model.

    Architecture:
        Input -> Dense (input projection) -> [SequenceLayer] x n_layers -> Output

    Supports:
        - Multiple stacked S5 layers
        - Input projection
        - RNN mode for step-by-step inference
        - Shared HiPPO initialization across layers
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
        Initialize stacked encoder.

        Args:
            key: JAX random key
            d_input: Input dimension
            d_model: Hidden dimension (H)
            n_layers: Number of sequence layers
            ssm_size: State space size (P * blocks)
            blocks: Number of SSM blocks (J)
            C_init: C matrix initialization method
            discretization: 'zoh' or 'bilinear'
            dt_min, dt_max: Timescale range for discretization
            conj_sym: Use conjugate symmetry (halves state size)
            clip_eigs: Clip eigenvalues for stability
            bidirectional: Bidirectional processing
            step_rescale: Timescale scaling factor
            activation: Activation function ('gelu', 'relu', etc.)
            prenorm: Use pre-normalization
            dtype: Data type

        Returns:
            CommonInit with initialized parameters
        """
        keys = jax.random.split(key, n_layers + 2)

        # Initialize HiPPO matrices (shared across layers)
        hippo = get_hippo_params(ssm_size, blocks, conj_sym)
        Lambda_re_init = hippo['Lambda_re_init']
        Lambda_im_init = hippo['Lambda_im_init']
        V = hippo['V']
        Vinv = hippo['Vinv']

        # Input projection (GPT-style init: stddev = 0.02 / sqrt(n_layers))
        gpt_scale = 0.02 / jnp.sqrt(n_layers)
        input_proj_init = ES_Linear.rand_init(
            keys[0], d_input, d_model, use_bias=True, dtype=dtype
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

        # Add frozen params for runtime access
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
        RNN mode forward pass for step-by-step inference.

        Args:
            common_params: CommonParams with noiser and params
            hiddens: List of hidden states, one per layer
            x: Input sequence (L, d_input)
            resets: Optional reset signals (L,) for resetting hidden states

        Returns:
            (new_hiddens, output_sequence)
        """
        fp = common_params.frozen_params
        n_layers = fp['n_layers']

        # Input projection
        x = call_submodule(ES_Linear, 'input_proj', common_params, x)

        # Pass through sequence layers with hidden state management
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
            List of hidden states, one per layer
        """
        if conj_sym:
            hidden_size = ssm_size // 2
        else:
            hidden_size = ssm_size

        return [
            jnp.zeros((batch_size, 1, hidden_size), dtype=jnp.complex64)
            for _ in range(n_layers)
        ]
