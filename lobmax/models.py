"""
LOBMAX Models

Main model classes for LOB prediction using MaxText Transformer.
Replaces S5-based PaddedLobPredModel with Transformer-based implementation.
"""

import os
import sys
from functools import partial
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import linen as nn

# Add MaxText layers to path (avoid importing MaxText __init__.py which has orbax version conflicts)
MAXTEXT_LAYERS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'maxtext', 'src', 'MaxText', 'layers')
MAXTEXT_SRC_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'maxtext', 'src')
if MAXTEXT_SRC_PATH not in sys.path:
    sys.path.insert(0, MAXTEXT_SRC_PATH)

# Import directly from layer modules to avoid MaxText __init__.py
import importlib.util
def _import_maxtext_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import normalizations
_normalizations = _import_maxtext_module(
    'normalizations', 
    os.path.join(MAXTEXT_LAYERS_PATH, 'normalizations.py')
)
rms_norm = _normalizations.rms_norm

from lobmax.config import LOBMAXConfig
from lobmax.layers import (
    StackedTransformerEncoder,
    TransformerBookEncoder,
    TransformerLayer,
)


class LOBMAXModel(nn.Module):
    """
    LOB Prediction Model using MaxText Transformer.
    
    Replaces PaddedLobPredModel with Transformer-based architecture:
    
    Architecture:
        Message: Embed -> N Transformer layers -> 
                                                   ├-> Concat -> Proj -> Fused Transformer -> Output
        Book:    Dense -> M Transformer layers -> 
    
    Args:
        config: LOBMAXConfig with model parameters
        mesh: JAX device mesh for sharding
        training: Training mode (affects dropout)
    """
    config: LOBMAXConfig
    mesh: Mesh
    training: bool = True
    
    def setup(self):
        cfg = self.config
        dtype = cfg.get_dtype()
        
        # Message Encoder: Embed + Transformer layers
        self.message_encoder = StackedTransformerEncoder(
            config=cfg,
            mesh=self.mesh,
            n_layers=cfg.n_message_layers,
            d_model=cfg.d_model,
            use_embed_layer=True,
            vocab_size=cfg.vocab_size,
            training=self.training,
            name="message_encoder",
        )
        
        # Book Encoder: Pre-layers + Projection + Post-layers
        self.book_encoder = TransformerBookEncoder(
            config=cfg,
            mesh=self.mesh,
            d_book=cfg.d_book,
            training=self.training,
            name="book_encoder",
        )
        
        # Fusion projection: 2*d_model -> d_model
        self.fusion_projection = nn.Dense(
            features=cfg.d_model,
            dtype=dtype,
            name="fusion_projection",
        )
        
        # Fused Transformer
        self.fused_layers = [
            TransformerLayer(
                config=cfg,
                mesh=self.mesh,
                layer_idx=i,
                d_model=cfg.d_model,
                training=self.training,
                name=f"fused_layer_{i}",
            )
            for i in range(cfg.n_layers)
        ]
        
        # Final normalization
        # Final normalization
        self.final_norm = rms_norm(
            num_features=cfg.d_model,
            dtype=dtype,
            weight_dtype=cfg.get_weight_dtype(),
            epsilon=cfg.normalization_layer_epsilon,
            kernel_axes=("norm",),
            name="final_norm",
        )
        
        # Output decoder
        self.decoder = nn.Dense(
            features=cfg.n_classes,
            dtype=jnp.float32,  # Keep output in FP32 for stability
            name="output_decoder",
        )
    
    def __call__(
        self,
        x_m: jnp.ndarray,
        x_b: jnp.ndarray,
        message_integration_timesteps: Optional[jnp.ndarray] = None,
        book_integration_timesteps: Optional[jnp.ndarray] = None,
        positions: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Forward pass for LOB prediction.
        
        Args:
            x_m: Message tokens (L,) int32 or (B, L) int32
            x_b: Book state (L, d_book) or (B, L, d_book) float
            message_integration_timesteps: Unused (S5 compatibility)
            book_integration_timesteps: Unused (S5 compatibility)
            positions: Optional position indices for RoPE
            
        Returns:
            Log probabilities (n_classes,) or (L, n_classes) or (B, L, n_classes)
            depending on config.mode
        """
        cfg = self.config
        
        # Handle shape normalization
        # x_m: tokens, x_b: continuous features
        original_m_ndim = x_m.ndim
        original_b_ndim = x_b.ndim
        
        if x_m.ndim == 1:
            x_m = x_m[None, :]  # (L,) -> (1, L)
        if x_b.ndim == 2:
            x_b = x_b[None, :, :]  # (L, d_book) -> (1, L, d_book)
        
        # Validate consistency
        assert x_m.shape[0] == x_b.shape[0], "Batch dim mismatch"
        assert x_m.shape[1] == x_b.shape[1], "Seq len mismatch"
        
        B, L = x_m.shape
        original_L = L
        
        # Pad for Flash Attention (block size 128)
        # Note: We rely on the model to handle padding tokens gracefully (e.g. valid mask or learning to ignore)
        ALIGN = 128
        if L % ALIGN != 0:
            pad_len = ALIGN - (L % ALIGN)
            x_m = jnp.pad(x_m, ((0, 0), (0, pad_len)), constant_values=0)
            x_b = jnp.pad(x_b, ((0, 0), (0, pad_len), (0, 0)), constant_values=0.0)
            
            if positions is not None:
                if positions.ndim == 1:
                    positions = jnp.pad(positions, (0, pad_len), constant_values=0)
                else:
                    positions = jnp.pad(positions, ((0, 0), (0, pad_len)), constant_values=0)
                    
            # Update L to padded length
            L = L + pad_len
        
        # Create positions if not provided (using padded L)
        if positions is None:
            positions = jnp.arange(L)[None, :].repeat(B, axis=0)
        
        # === Encode Message and Book ===
        x_m_enc = self.message_encoder(x_m, positions=positions)  # (B, L, d_model)
        x_b_enc = self.book_encoder(x_b, positions=positions)      # (B, L, d_model)
        
        # === Concatenate and Project ===
        x = jnp.concatenate([x_m_enc, x_b_enc], axis=-1)  # (B, L, 2*d_model)
        x = self.fusion_projection(x)                      # (B, L, d_model)
        
        # === Fused Transformer ===
        for layer in self.fused_layers:
            x = layer(x, positions=positions)
        
        # === Final Norm ===
        x = self.final_norm(x)
        
        # Undo Padding if it was applied
        if x.shape[1] > original_L:
            x = x[:, :original_L, :]
        
        # === Pooling / Mode ===
        if cfg.mode == "pool":
            x = jnp.mean(x, axis=1)  # (B, d_model)
        elif cfg.mode == "last":
            x = x[:, -1, :]  # (B, d_model)
        elif cfg.mode == "ema":
            # Exponential moving average
            alpha = 2.0 / (22 + 1.0)
            # Simple EMA implementation
            def ema_step(carry, x_t):
                return carry * (1 - alpha) + x_t * alpha, None
            x, _ = jax.lax.scan(ema_step, jnp.zeros_like(x[:, 0, :]), jnp.moveaxis(x, 1, 0))
            # x is now (B, d_model)
        elif cfg.mode == "none":
            pass  # Keep (B, L, d_model)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")
        
        # === Output Decoder ===
        logits = self.decoder(x)
        
        # === Log Softmax ===
        output = nn.log_softmax(logits, axis=-1)
        
        # Restore original batch dimension if needed
        if original_m_ndim == 1 and cfg.mode != "none":
            output = output[0]  # (1, n_classes) -> (n_classes,)
        elif original_m_ndim == 1 and cfg.mode == "none":
            output = output[0]  # (1, L, n_classes) -> (L, n_classes)
        
        return output
    
    def __call_ar__(
        self,
        x_m: jnp.ndarray,
        x_b: jnp.ndarray,
        message_integration_timesteps: Optional[jnp.ndarray] = None,
        book_integration_timesteps: Optional[jnp.ndarray] = None,
        positions: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Autoregressive call (same as __call__ for Transformer).
        S5 has different RNN mode, but Transformer doesn't need this distinction.
        """
        return self.__call__(
            x_m, x_b,
            message_integration_timesteps,
            book_integration_timesteps,
            positions,
        )


# Batched version using vmap
def _create_batch_lobmax():
    """Create batched version of LOBMAXModel."""
    return nn.vmap(
        LOBMAXModel,
        in_axes=(0, 0, 0, 0),  # x_m, x_b, msg_timesteps, book_timesteps all batched
        out_axes=0,
        variable_axes={"params": None, "dropout": None, "batch_stats": None},
        split_rngs={"params": False, "dropout": True},
        axis_name='batch',
        methods={
            '__call__': {
                'in_axes': (0, 0, 0, 0),
                'out_axes': 0,
                'variable_axes': {"params": None, "dropout": None, "batch_stats": None},
                'split_rngs': {"params": False, "dropout": True},
                'axis_name': 'batch',
            },
            '__call_ar__': {
                'in_axes': (0, 0, 0, 0),
                'out_axes': 0,
                'variable_axes': {"params": None, "dropout": None, "batch_stats": None},
                'split_rngs': {"params": False, "dropout": True},
                'axis_name': 'batch',
            },
        },
    )


# Note: The batched model handles batching internally in __call__,
# so we don't actually need nn.vmap. But we keep this for compatibility
# with the S5 interface which uses BatchPaddedLobPredModel.
BatchLOBMAXModel = LOBMAXModel  # Transformer handles batching internally


def create_lobmax_model(
    config: LOBMAXConfig,
    mesh: Mesh,
    training: bool = True,
) -> LOBMAXModel:
    """
    Factory function to create LOBMAX model.
    
    Args:
        config: Model configuration
        mesh: JAX device mesh
        training: Training mode
        
    Returns:
        LOBMAXModel instance
    """
    return LOBMAXModel(
        config=config,
        mesh=mesh,
        training=training,
    )


def count_parameters(params) -> int:
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def get_model_summary(config: LOBMAXConfig) -> str:
    """Get human-readable model summary."""
    # Approximate parameter count
    d = config.d_model
    h = config.num_heads
    mlp = config.mlp_dim
    n_msg = config.n_message_layers
    n_book = config.n_book_pre_layers + config.n_book_post_layers
    n_fused = config.n_layers
    
    # Per transformer layer: QKV + O + FFN
    params_per_layer = (
        3 * d * d +  # QKV
        d * d +      # O
        3 * d * mlp  # SwiGLU FFN (gate, up, down)
    )
    
    # Embedding
    embed_params = config.vocab_size * d
    
    # Book projection
    book_proj_params = config.d_book * d
    
    # Fusion projection
    fusion_params = 2 * d * d
    
    # Output
    output_params = d * config.n_classes
    
    total_layers = n_msg + n_book + n_fused
    total_params = (
        embed_params +
        book_proj_params +
        fusion_params +
        output_params +
        total_layers * params_per_layer
    )
    
    summary = f"""
LOBMAX Model Summary
====================
Architecture: LLaMA-2 Style Transformer

Dimensions:
  - d_model: {d}
  - num_heads: {h}
  - head_dim: {config.head_dim}
  - mlp_dim: {mlp}
  - vocab_size: {config.vocab_size}
  - d_book: {config.d_book}
  - n_classes: {config.n_classes}

Layers:
  - Message Encoder: {n_msg} layers
  - Book Encoder: {config.n_book_pre_layers} pre + {config.n_book_post_layers} post layers
  - Fused Transformer: {n_fused} layers
  - Total: {total_layers} transformer layers

Attention:
  - Kernel: {config.attention}
  - RoPE: {config.rope_type}
  - Max Length: {config.max_target_length}

Estimated Parameters: ~{total_params / 1e9:.2f}B
"""
    return summary
