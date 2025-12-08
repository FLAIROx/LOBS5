"""
ES-compatible LOB Prediction Models.

This module provides ES-compatible versions of the LOB prediction models
that work with the HyperscaleES noiser framework.

Based on: lob/lob_seq_model.py
"""

import jax
import jax.numpy as jnp
import math
from functools import partial

from .common import (
    Model, CommonInit, CommonParams,
    PARAM, MM_PARAM, EXCLUDED,
    merge_inits, merge_frozen, call_submodule,
    ES_Parameter, ES_Linear, ES_LayerNorm, ES_MM,
)
from .s5_ssm import init_hippo_matrices
from .sequence_layer import ES_SequenceLayer
from .stacked_encoder import ES_StackedEncoder

__all__ = ['ES_LobBookModel', 'ES_PaddedLobPredModel']


class ES_LobBookModel(Model):
    """
    ES-compatible LOB Book Encoder.

    Architecture:
        Book Input (L_b, d_book) → Pre-layers (S5) → Dense projection → Post-layers (S5) → Output

    Used to encode order book state features.
    """

    @classmethod
    def rand_init(
        cls,
        key,
        d_book: int,
        d_model: int,
        n_pre_layers: int,
        n_post_layers: int,
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
        Initialize LOB Book Encoder.

        Args:
            key: JAX random key
            d_book: Input book feature dimension
            d_model: Output/hidden dimension
            n_pre_layers: Number of pre-projection layers
            n_post_layers: Number of post-projection layers
            (other args same as ES_StackedEncoder)

        Returns:
            CommonInit
        """
        keys = jax.random.split(key, 4)

        # Initialize HiPPO matrices
        Lambda_re_init, Lambda_im_init, V, Vinv = init_hippo_matrices(
            ssm_size, blocks, conj_sym
        )

        n_total_layers = n_pre_layers + n_post_layers
        gpt_scale = 0.02 / jnp.sqrt(max(n_total_layers, 1))

        # Pre-layers: process book features at d_book dimension
        pre_layer_inits = {}
        for i in range(n_pre_layers):
            layer_init = ES_SequenceLayer.rand_init(
                keys[0],
                d_model=d_book,  # Pre-layers work at d_book dimension
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
            keys = jax.random.split(keys[0], 2)
            pre_layer_inits[f'pre_layer_{i}'] = layer_init

        # Projection from d_book to d_model
        proj_init = ES_Linear.rand_init(
            keys[1], d_book, d_model, use_bias=True, dtype=dtype, scale=gpt_scale
        )

        # Post-layers: process at d_model dimension
        post_layer_inits = {}
        for i in range(n_post_layers):
            layer_init = ES_SequenceLayer.rand_init(
                keys[2],
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
            keys = jax.random.split(keys[2], 2)
            post_layer_inits[f'post_layer_{i}'] = layer_init

        # Merge all initializations
        merged = merge_inits(
            proj=proj_init,
            **pre_layer_inits,
            **post_layer_inits,
        )

        return merge_frozen(
            merged,
            n_pre_layers=n_pre_layers,
            n_post_layers=n_post_layers,
            d_book=d_book,
            d_model=d_model,
        )

    @classmethod
    def _forward(cls, common_params: CommonParams, x, integration_timesteps=None):
        """
        Forward pass through book encoder.

        Args:
            common_params: CommonParams with noiser and params
            x: Book input (L_b, d_book)
            integration_timesteps: Optional timestep info (unused in ES version)

        Returns:
            Encoded book features (L_b, d_model)
        """
        fp = common_params.frozen_params
        n_pre_layers = fp['n_pre_layers']
        n_post_layers = fp['n_post_layers']

        # Pre-layers
        for i in range(n_pre_layers):
            x = call_submodule(ES_SequenceLayer, f'pre_layer_{i}', common_params, x)

        # Project to d_model
        x = call_submodule(ES_Linear, 'proj', common_params, x)

        # Post-layers
        for i in range(n_post_layers):
            x = call_submodule(ES_SequenceLayer, f'post_layer_{i}', common_params, x)

        return x

    @classmethod
    def _forward_rnn(cls, common_params: CommonParams, hiddens, x, resets=None, integration_timesteps=None):
        """
        RNN mode forward pass.

        Args:
            common_params: CommonParams
            hiddens: Tuple of (pre_hiddens, post_hiddens)
            x: Book input
            resets: Reset signals
            integration_timesteps: Unused

        Returns:
            (new_hiddens, output)
        """
        fp = common_params.frozen_params
        n_pre_layers = fp['n_pre_layers']
        n_post_layers = fp['n_post_layers']

        pre_hiddens, post_hiddens = hiddens
        new_pre_hiddens = []
        new_post_hiddens = []

        # Pre-layers
        for i in range(n_pre_layers):
            layer_params = common_params._replace(
                frozen_params=common_params.frozen_params.get(f'pre_layer_{i}', {}),
                params=common_params.params[f'pre_layer_{i}'],
                es_tree_key=common_params.es_tree_key[f'pre_layer_{i}'],
            )
            hidden_i, x = ES_SequenceLayer._forward_rnn(
                layer_params, pre_hiddens[i], x, resets
            )
            new_pre_hiddens.append(hidden_i)

        # Project
        x = call_submodule(ES_Linear, 'proj', common_params, x)

        # Post-layers
        for i in range(n_post_layers):
            layer_params = common_params._replace(
                frozen_params=common_params.frozen_params.get(f'post_layer_{i}', {}),
                params=common_params.params[f'post_layer_{i}'],
                es_tree_key=common_params.es_tree_key[f'post_layer_{i}'],
            )
            hidden_i, x = ES_SequenceLayer._forward_rnn(
                layer_params, post_hiddens[i], x, resets
            )
            new_post_hiddens.append(hidden_i)

        return (new_pre_hiddens, new_post_hiddens), x

    @staticmethod
    def initialize_carry(batch_size, ssm_size, n_pre_layers, n_post_layers, conj_sym=True):
        """Initialize hidden states for RNN mode."""
        if conj_sym:
            hidden_size = ssm_size // 2
        else:
            hidden_size = ssm_size

        pre_hiddens = [
            jnp.zeros((batch_size, 1, hidden_size), dtype=jnp.complex64)
            for _ in range(n_pre_layers)
        ]
        post_hiddens = [
            jnp.zeros((batch_size, 1, hidden_size), dtype=jnp.complex64)
            for _ in range(n_post_layers)
        ]
        return (pre_hiddens, post_hiddens)


class ES_PaddedLobPredModel(Model):
    """
    ES-compatible Padded LOB Prediction Model.

    Architecture:
        Message Input (L_m, vocab) → Message Encoder → \
                                                        → Concatenate → Fused S5 → Pool/Last → Decoder → LogSoftmax
        Book Input (L_b, d_book)   → Book Encoder   → /

    Main model for LOB next-token prediction.
    """

    @classmethod
    def rand_init(
        cls,
        key,
        d_output: int,
        d_model: int,
        d_book: int,
        n_message_layers: int,
        n_fused_layers: int,
        n_book_pre_layers: int = 1,
        n_book_post_layers: int = 1,
        # SSM params
        ssm_size: int = 256,
        blocks: int = 8,
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
        mode: str = 'pool',
        dtype=jnp.float32,
    ) -> CommonInit:
        """
        Initialize PaddedLobPredModel.

        Args:
            key: JAX random key
            d_output: Output dimension (vocab size)
            d_model: Hidden dimension
            d_book: Book feature dimension
            n_message_layers: Number of message encoder layers
            n_fused_layers: Number of fused encoder layers
            n_book_pre_layers: Number of book pre-projection layers
            n_book_post_layers: Number of book post-projection layers
            (other args same as ES_StackedEncoder)

        Returns:
            CommonInit
        """
        keys = jax.random.split(key, 5)

        # Initialize HiPPO matrices (shared)
        Lambda_re_init, Lambda_im_init, V, Vinv = init_hippo_matrices(
            ssm_size, blocks, conj_sym
        )

        n_total_layers = n_message_layers + n_book_pre_layers + n_book_post_layers + n_fused_layers
        gpt_scale = 0.02 / math.sqrt(n_total_layers)

        # Message encoder (with embedding)
        # Input is token indices, so we need embedding
        message_encoder_init = _init_message_encoder(
            keys[0],
            vocab_size=d_output,
            d_model=d_model,
            n_layers=n_message_layers,
            ssm_size=ssm_size,
            blocks=blocks,
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
            activation=activation,
            prenorm=prenorm,
            dtype=dtype,
            gpt_scale=gpt_scale,
        )

        # Book encoder
        book_encoder_init = ES_LobBookModel.rand_init(
            keys[1],
            d_book=d_book,
            d_model=d_model,
            n_pre_layers=n_book_pre_layers,
            n_post_layers=n_book_post_layers,
            ssm_size=ssm_size,
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

        # Fused S5 encoder (input is 2*d_model after concatenation)
        fused_encoder_init = ES_StackedEncoder.rand_init(
            keys[2],
            d_input=2 * d_model,
            d_model=d_model,
            n_layers=n_fused_layers,
            ssm_size=ssm_size,
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

        # Decoder (output layer)
        decoder_init = ES_Linear.rand_init(
            keys[3], d_model, d_output, use_bias=True, dtype=jnp.float32, scale=gpt_scale
        )

        # Merge all
        merged = merge_inits(
            message_encoder=message_encoder_init,
            book_encoder=book_encoder_init,
            fused_encoder=fused_encoder_init,
            decoder=decoder_init,
        )

        return merge_frozen(
            merged,
            d_output=d_output,
            d_model=d_model,
            d_book=d_book,
            n_message_layers=n_message_layers,
            n_fused_layers=n_fused_layers,
            n_book_pre_layers=n_book_pre_layers,
            n_book_post_layers=n_book_post_layers,
            ssm_size=ssm_size,
            conj_sym=conj_sym,
            mode=mode,
        )

    @classmethod
    def _forward(cls, common_params: CommonParams, x_m, x_b,
                 message_integration_timesteps=None, book_integration_timesteps=None):
        """
        Forward pass for single output (pool/last mode).

        Args:
            common_params: CommonParams with noiser and params
            x_m: Message tokens (L_m,) int32
            x_b: Book features (L_b, d_book) float
            message_integration_timesteps: Unused in ES
            book_integration_timesteps: Unused in ES

        Returns:
            Log probabilities (d_output,) or (L, d_output) depending on mode
        """
        fp = common_params.frozen_params
        mode = fp['mode']

        # Message encoding (with embedding)
        x_m = _forward_message_encoder(common_params, x_m, message_integration_timesteps)

        # Book encoding
        x_b = call_submodule(ES_LobBookModel, 'book_encoder', common_params, x_b, book_integration_timesteps)

        # Concatenate message and book features
        x = jnp.concatenate([x_m, x_b], axis=1)

        # Fused encoding
        x = call_submodule(ES_StackedEncoder, 'fused_encoder', common_params, x)

        # Pooling
        if mode == 'pool':
            x = jnp.mean(x, axis=0)
        elif mode == 'last':
            x = x[-1]
        elif mode == 'none':
            pass  # Keep full sequence
        elif mode == 'ema':
            # EMA pooling
            alpha = 2.0 / (22 + 1.0)
            x = _ewma_simple(x, alpha)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        # Decode
        x = call_submodule(ES_Linear, 'decoder', common_params, x)

        # Log softmax (in float32 for numerical stability)
        x = x.astype(jnp.float32)
        return jax.nn.log_softmax(x, axis=-1)

    @classmethod
    def _forward_ar(cls, common_params: CommonParams, x_m, x_b,
                    message_integration_timesteps=None, book_integration_timesteps=None):
        """
        Autoregressive forward pass (returns per-token predictions).

        Same as _forward but keeps mode='none' to return full sequence.
        """
        fp = common_params.frozen_params

        # Message encoding
        x_m = _forward_message_encoder(common_params, x_m, message_integration_timesteps)

        # Book encoding (cast to bfloat16 if using mixed precision)
        x_b = call_submodule(ES_LobBookModel, 'book_encoder', common_params, x_b, book_integration_timesteps)

        # Concatenate
        x = jnp.concatenate([x_m, x_b], axis=1)

        # Fused encoding
        x = call_submodule(ES_StackedEncoder, 'fused_encoder', common_params, x)

        # No pooling - return per-token predictions
        x = call_submodule(ES_Linear, 'decoder', common_params, x)

        # Log softmax in float32
        x = x.astype(jnp.float32)
        return jax.nn.log_softmax(x, axis=-1)

    @staticmethod
    def initialize_carry(batch_size, ssm_size, n_message_layers, n_book_pre_layers,
                         n_book_post_layers, n_fused_layers, d_model, conj_sym=True):
        """Initialize hidden states for RNN mode."""
        if conj_sym:
            hidden_size = ssm_size // 2
        else:
            hidden_size = ssm_size

        message_hiddens = ES_StackedEncoder.initialize_carry(
            batch_size, ssm_size, n_message_layers, conj_sym
        )
        book_hiddens = ES_LobBookModel.initialize_carry(
            batch_size, ssm_size, n_book_pre_layers, n_book_post_layers, conj_sym
        )
        fused_hiddens = ES_StackedEncoder.initialize_carry(
            batch_size, ssm_size, n_fused_layers, conj_sym
        )
        # EMA state
        ema_state = (jnp.zeros((batch_size, 1, d_model)), jnp.ones((batch_size, 1, 1)))

        return (message_hiddens, book_hiddens, fused_hiddens, ema_state)

    @classmethod
    def _forward_step(cls, common_params: CommonParams, hiddens, x_m, x_b, resets=None):
        """
        Single-step forward pass for step-by-step episode simulation.

        This is used for ES-JaxLOB training where we need to interleave
        model inference with JaxLOB simulation step by step.

        Args:
            common_params: CommonParams with noiser and params
            hiddens: Tuple of (message_hiddens, book_hiddens, fused_hiddens, ema_state)
            x_m: Message tokens (L_step,) int32 - typically 24 tokens for one message
            x_b: Book features (L_b_step, d_book) float - current book state
            resets: Optional reset signals

        Returns:
            (new_hiddens, log_probs)
            - new_hiddens: Updated hidden states
            - log_probs: Log probabilities (L_step, d_output) for next token predictions
        """
        fp = common_params.frozen_params
        message_hiddens, book_hiddens, fused_hiddens, ema_state = hiddens

        # Message encoder RNN forward
        msg_params = common_params._replace(
            frozen_params=common_params.frozen_params.get('message_encoder', {}),
            params=common_params.params['message_encoder'],
            es_tree_key=common_params.es_tree_key['message_encoder'],
        )
        n_msg_layers = msg_params.frozen_params.get('n_layers', 0)

        # Embedding lookup
        embedding = call_submodule(ES_Parameter, 'embedding', msg_params)
        x_m_emb = embedding[x_m.ravel()]  # (L_step, d_model)

        # Message sequence layers (RNN mode)
        new_message_hiddens = []
        for i in range(n_msg_layers):
            layer_params = msg_params._replace(
                frozen_params=msg_params.frozen_params.get(f'layer_{i}', {}),
                params=msg_params.params[f'layer_{i}'],
                es_tree_key=msg_params.es_tree_key[f'layer_{i}'],
            )
            hidden_i, x_m_emb = ES_SequenceLayer._forward_rnn(
                layer_params, message_hiddens[i], x_m_emb, resets
            )
            new_message_hiddens.append(hidden_i)

        # Book encoder RNN forward
        book_params = common_params._replace(
            frozen_params=common_params.frozen_params.get('book_encoder', {}),
            params=common_params.params['book_encoder'],
            es_tree_key=common_params.es_tree_key['book_encoder'],
        )
        new_book_hiddens, x_b_enc = ES_LobBookModel._forward_rnn(
            book_params, book_hiddens, x_b, resets
        )

        # Concatenate message and book features
        # x_m_emb: (L_step, d_model), x_b_enc: (L_b_step, d_model)
        # Need to broadcast/align - typically book features are expanded to match message length
        if x_m_emb.shape[0] != x_b_enc.shape[0]:
            # Repeat book features to match message length
            x_b_enc = jnp.broadcast_to(x_b_enc[-1:], (x_m_emb.shape[0], x_b_enc.shape[-1]))

        x_concat = jnp.concatenate([x_m_emb, x_b_enc], axis=-1)  # (L_step, 2*d_model)

        # Fused encoder RNN forward
        fused_params = common_params._replace(
            frozen_params=common_params.frozen_params.get('fused_encoder', {}),
            params=common_params.params['fused_encoder'],
            es_tree_key=common_params.es_tree_key['fused_encoder'],
        )
        new_fused_hiddens, x_fused = ES_StackedEncoder._forward_rnn(
            fused_params, fused_hiddens, x_concat, resets
        )

        # EMA pooling (optional, for single token prediction)
        ema_val, ema_count = ema_state
        alpha = 2.0 / (22 + 1.0)
        new_ema_val = alpha * x_fused + (1 - alpha) * ema_val
        new_ema_count = ema_count + 1
        new_ema_state = (new_ema_val, new_ema_count)

        # Decode
        x_out = call_submodule(ES_Linear, 'decoder', common_params, x_fused)

        # Log softmax in float32
        x_out = x_out.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(x_out, axis=-1)

        new_hiddens = (new_message_hiddens, new_book_hiddens, new_fused_hiddens, new_ema_state)
        return new_hiddens, log_probs


# =============================================================================
# Helper Functions
# =============================================================================

def _init_message_encoder(
    key,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    ssm_size: int,
    blocks: int,
    Lambda_re_init,
    Lambda_im_init,
    V,
    Vinv,
    C_init: str,
    discretization: str,
    dt_min: float,
    dt_max: float,
    conj_sym: bool,
    clip_eigs: bool,
    bidirectional: bool,
    step_rescale: float,
    activation: str,
    prenorm: bool,
    dtype,
    gpt_scale: float,
) -> CommonInit:
    """
    Initialize message encoder with embedding layer.

    The message encoder uses an embedding layer (vocab_size → d_model)
    followed by stacked S5 layers.
    """
    keys = jax.random.split(key, n_layers + 2)

    # Embedding layer
    embedding_init = ES_Parameter.rand_init(
        keys[0],
        shape=(vocab_size, d_model),
        scale=gpt_scale,
        dtype=dtype,
        es_type=EXCLUDED,  # Don't perturb embeddings
    )

    # Sequence layers
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

    merged = merge_inits(
        embedding=embedding_init,
        **layer_inits,
    )

    return merge_frozen(merged, n_layers=n_layers)


def _forward_message_encoder(common_params: CommonParams, x, integration_timesteps=None):
    """Forward pass through message encoder."""
    msg_params = common_params._replace(
        frozen_params=common_params.frozen_params.get('message_encoder', {}),
        params=common_params.params['message_encoder'],
        es_tree_key=common_params.es_tree_key['message_encoder'],
    )

    n_layers = msg_params.frozen_params.get('n_layers', 0)

    # Embedding lookup (x is int32 tokens)
    embedding = call_submodule(ES_Parameter, 'embedding', msg_params)
    x = embedding[x.ravel()]

    # Sequence layers
    for i in range(n_layers):
        x = call_submodule(ES_SequenceLayer, f'layer_{i}', msg_params, x)

    return x


def _ewma_simple(x, alpha):
    """Simple exponential weighted moving average."""
    def scan_fn(carry, xi):
        ewma = alpha * xi + (1 - alpha) * carry
        return ewma, ewma

    _, ewma_seq = jax.lax.scan(scan_fn, jnp.zeros_like(x[0]), x)
    return ewma_seq[-1]
