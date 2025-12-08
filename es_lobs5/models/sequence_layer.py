"""
ES-compatible S5 Sequence Layer.

Wraps ES_S5SSM with normalization, activation, dropout, and residual connection.

Based on: s5/layers.py
"""

import jax
import jax.numpy as jnp
from functools import partial

from .common import (
    Model, CommonInit, CommonParams,
    PARAM, MM_PARAM, EXCLUDED,
    merge_inits, merge_frozen, call_submodule,
    ES_Parameter, ES_Linear, ES_LayerNorm,
    ACTIVATIONS,
)
from .s5_ssm import ES_S5SSM

__all__ = ['ES_SequenceLayer']


class ES_SequenceLayer(Model):
    """
    ES-compatible S5 sequence layer.

    Architecture:
        [prenorm] → SSM → activation → [GLU gates] → residual → [postnorm]

    Supports:
        - Pre/post normalization
        - Multiple activation types (gelu, full_glu, half_glu1, half_glu2)
        - Dropout (disabled during ES training, ES doesn't use dropout)
        - RNN mode for step-by-step inference
    """

    @classmethod
    def rand_init(
        cls,
        key,
        d_model: int,
        # SSM params
        ssm_size: int,
        Lambda_re_init: jnp.ndarray,
        Lambda_im_init: jnp.ndarray,
        V: jnp.ndarray,
        Vinv: jnp.ndarray,
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
        Initialize sequence layer.

        Args:
            key: JAX random key
            d_model: Feature dimension (H)
            ssm_size: State space size (P * blocks)
            Lambda_re_init, Lambda_im_init: Initial eigenvalues
            V, Vinv: Eigenvector matrices
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
        keys = jax.random.split(key, 4)

        # P is the Lambda size, which is already correctly sized from init_hippo_matrices
        # For conj_sym=True: P = (ssm_size // blocks // 2) * blocks = ssm_size // 2
        # Lambda_re_init already has this shape
        P = Lambda_re_init.shape[0]

        # Initialize SSM
        ssm_init = ES_S5SSM.rand_init(
            keys[0],
            H=d_model,
            P=P,
            Lambda_re_init=Lambda_re_init,
            Lambda_im_init=Lambda_im_init,
            V=V,
            Vinv=Vinv,
            C_init=C_init,
            discretization=discretization,
            dt_min=dt_min,
            dt_max=dt_max,
            conj_sym=conj_sym,
            clip_eigs=clip_eigs,
            bidirectional=bidirectional,
            step_rescale=step_rescale,
            dtype=dtype,
        )

        # Initialize LayerNorm
        norm_init = ES_LayerNorm.rand_init(keys[1], d_model, dtype)

        # Initialize GLU gates if needed
        # GPT-style initialization: stddev=0.02
        gpt_scale = 0.02

        if activation == 'full_glu':
            out1_init = ES_Linear.rand_init(
                keys[2], d_model, d_model, use_bias=True, dtype=dtype, scale=gpt_scale
            )
            out2_init = ES_Linear.rand_init(
                keys[3], d_model, d_model, use_bias=True, dtype=dtype, scale=gpt_scale
            )
            merged = merge_inits(
                ssm=ssm_init,
                norm=norm_init,
                out1=out1_init,
                out2=out2_init,
            )
        elif activation in ['half_glu1', 'half_glu2']:
            out2_init = ES_Linear.rand_init(
                keys[2], d_model, d_model, use_bias=True, dtype=dtype, scale=gpt_scale
            )
            merged = merge_inits(
                ssm=ssm_init,
                norm=norm_init,
                out2=out2_init,
            )
        else:
            # Simple activation (gelu, relu, etc.)
            merged = merge_inits(
                ssm=ssm_init,
                norm=norm_init,
            )

        # Add frozen params for activation and prenorm config
        return merge_frozen(
            merged,
            activation=activation,
            prenorm=prenorm,
            d_model=d_model,
        )

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        """
        Forward pass through sequence layer.

        Args:
            common_params: CommonParams with noiser and params
            x: Input sequence (L, d_model)

        Returns:
            Output sequence (L, d_model)
        """
        fp = common_params.frozen_params
        activation = fp['activation']
        prenorm = fp['prenorm']

        # Residual connection
        skip = x

        # Pre-normalization
        if prenorm:
            x = call_submodule(ES_LayerNorm, 'norm', common_params, x)

        # SSM
        x = call_submodule(ES_S5SSM, 'ssm', common_params, x)

        # Activation and GLU gates
        if activation == 'full_glu':
            x = jax.nn.gelu(x)
            gate = call_submodule(ES_Linear, 'out1', common_params, x)
            gate_sigmoid = jax.nn.sigmoid(
                call_submodule(ES_Linear, 'out2', common_params, x)
            )
            x = gate * gate_sigmoid
        elif activation == 'half_glu1':
            x = jax.nn.gelu(x)
            gate_sigmoid = jax.nn.sigmoid(
                call_submodule(ES_Linear, 'out2', common_params, x)
            )
            x = x * gate_sigmoid
        elif activation == 'half_glu2':
            x1 = jax.nn.gelu(x)
            gate_sigmoid = jax.nn.sigmoid(
                call_submodule(ES_Linear, 'out2', common_params, x1)
            )
            x = x * gate_sigmoid
        elif activation == 'gelu':
            x = jax.nn.gelu(x)
        elif activation in ACTIVATIONS:
            x = ACTIVATIONS[activation](x)
        else:
            raise NotImplementedError(f"Activation: {activation} not implemented")

        # Residual connection
        x = skip + x

        # Post-normalization
        if not prenorm:
            x = call_submodule(ES_LayerNorm, 'norm', common_params, x)

        return x

    @classmethod
    def _forward_rnn(cls, common_params: CommonParams, hidden, x, resets=None):
        """
        RNN mode forward pass.

        Args:
            common_params: CommonParams with noiser and params
            hidden: Hidden state (1, P) complex
            x: Input sequence (L, d_model)
            resets: Optional reset signals (L,)

        Returns:
            (new_hidden, output_sequence)
        """
        fp = common_params.frozen_params
        activation = fp['activation']
        prenorm = fp['prenorm']

        # Residual connection
        skip = x

        # Pre-normalization
        if prenorm:
            x = call_submodule(ES_LayerNorm, 'norm', common_params, x)

        # SSM in RNN mode
        hidden, x = ES_S5SSM._forward_rnn(
            common_params._replace(
                frozen_params=common_params.frozen_params.get('ssm', common_params.frozen_params),
                params=common_params.params['ssm'],
                es_tree_key=common_params.es_tree_key['ssm'],
            ),
            hidden, x, resets
        )

        # Activation and GLU gates (same as _forward)
        if activation == 'full_glu':
            x = jax.nn.gelu(x)
            gate = call_submodule(ES_Linear, 'out1', common_params, x)
            gate_sigmoid = jax.nn.sigmoid(
                call_submodule(ES_Linear, 'out2', common_params, x)
            )
            x = gate * gate_sigmoid
        elif activation == 'half_glu1':
            x = jax.nn.gelu(x)
            gate_sigmoid = jax.nn.sigmoid(
                call_submodule(ES_Linear, 'out2', common_params, x)
            )
            x = x * gate_sigmoid
        elif activation == 'half_glu2':
            x1 = jax.nn.gelu(x)
            gate_sigmoid = jax.nn.sigmoid(
                call_submodule(ES_Linear, 'out2', common_params, x1)
            )
            x = x * gate_sigmoid
        elif activation == 'gelu':
            x = jax.nn.gelu(x)
        elif activation in ACTIVATIONS:
            x = ACTIVATIONS[activation](x)
        else:
            raise NotImplementedError(f"Activation: {activation} not implemented")

        # Residual connection
        x = skip + x

        # Post-normalization
        if not prenorm:
            x = call_submodule(ES_LayerNorm, 'norm', common_params, x)

        return hidden, x

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Initialize hidden state for RNN mode."""
        return jnp.zeros((batch_size, 1, hidden_size), dtype=jnp.complex64)
