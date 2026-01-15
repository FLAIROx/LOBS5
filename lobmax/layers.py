"""
LOBMAX Layers

Transformer layers based on MaxText's LLaMA-style implementation.
These layers replace S5's SequenceLayer with Transformer attention + FFN.
"""

import os
import sys
from functools import partial
from typing import Any, Optional, Tuple

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
    sys.modules[module_name] = module  # Register in sys.modules
    spec.loader.exec_module(module)
    return module

# Import common_types first (needed by other modules)
_common_types = _import_maxtext_module(
    'MaxText.common_types',
    os.path.join(MAXTEXT_SRC_PATH, 'MaxText', 'common_types.py')
)
MODEL_MODE_TRAIN = _common_types.MODEL_MODE_TRAIN

# Import normalizations
_normalizations = _import_maxtext_module(
    'MaxText.layers.normalizations',
    os.path.join(MAXTEXT_LAYERS_PATH, 'normalizations.py')
)
rms_norm = _normalizations.rms_norm

# Import linears
_linears = _import_maxtext_module(
    'MaxText.layers.linears',
    os.path.join(MAXTEXT_LAYERS_PATH, 'linears.py')
)
linears = _linears

# Import attentions
_attentions = _import_maxtext_module(
    'MaxText.layers.attentions',
    os.path.join(MAXTEXT_LAYERS_PATH, 'attentions.py')
)
attention_as_linen = _attentions.attention_as_linen

from lobmax.config import LOBMAXConfig


class TransformerLayer(nn.Module):
    """
    Single Transformer layer (LLaMA-style) to replace S5 SequenceLayer.
    
    Architecture:
        x -> RMSNorm -> Attention -> residual -> RMSNorm -> FFN -> residual
    
    This maintains the same interface as S5's SequenceLayer:
        __call__(x) -> x with shape (L, d_model)
    
    Args:
        config: LOBMAXConfig with model parameters
        mesh: JAX device mesh for sharding
        layer_idx: Layer index (for naming)
        d_model: Model dimension (overrides config if provided)
    """
    config: LOBMAXConfig
    mesh: Mesh
    layer_idx: int = 0
    d_model: int = None  # Allow override for book encoder
    training: bool = True
    
    def setup(self):
        cfg = self.config
        d_model = self.d_model or cfg.d_model
        dtype = cfg.get_dtype()
        weight_dtype = cfg.get_weight_dtype()
        
        # Input shape for attention initialization
        # Use dummy shape to allow initialization in setup()
        # MaxText needs shape primarily for the last dimension (d_model)
        dummy_shape = (1, 1, d_model)
        
        # === Define Submodules in setup() for Remat compatibility ===
        self.pre_attn_norm = rms_norm(
            num_features=d_model,
            dtype=dtype,
            weight_dtype=weight_dtype,
            name=f"pre_attn_norm", # Layer index is implicit in scope
            epsilon=cfg.normalization_layer_epsilon,
            kernel_axes=("norm",),
        )
        
        # Attention
        self.attention = attention_as_linen(
            config=cfg,
            num_query_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_target_length=cfg.max_target_length,
            max_prefill_predict_length=cfg.max_prefill_predict_length,
            attention_kernel=cfg.attention,
            inputs_q_shape=dummy_shape,
            inputs_kv_shape=dummy_shape,
            mesh=self.mesh,
            dtype=dtype,
            weight_dtype=weight_dtype,
            dropout_rate=cfg.dropout_rate if self.training else 0.0,
            name=f"attention",
            float32_qk_product=cfg.float32_qk_product,
            float32_logits=cfg.float32_logits,
            model_mode=MODEL_MODE_TRAIN if self.training else "prefill",
        )
        
        self.pre_ffn_norm = rms_norm(
            num_features=d_model,
            dtype=dtype,
            weight_dtype=weight_dtype,
            name=f"pre_ffn_norm",
            epsilon=cfg.normalization_layer_epsilon,
            kernel_axes=("norm",),
        )
        
        self.mlp = linears.mlp_block(
            in_features=d_model,
            intermediate_dim=cfg.mlp_dim,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate if self.training else 0.0,
            dtype=dtype,
            weight_dtype=weight_dtype,
            name=f"mlp",
            model_mode=MODEL_MODE_TRAIN if self.training else "prefill",
            config=cfg,
            quant=None,
            mesh=self.mesh,
        )
        
        if self.training and cfg.dropout_rate > 0:
            self.dropout = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))
    
    @nn.remat
    @nn.compact
    def __call__(self, x, positions=None):
        """
        Apply transformer layer.
        
        Args:
            x: Input tensor (L, d_model) or (B, L, d_model)
            positions: Position indices (L,) or (B, L) for RoPE
            
        Returns:
            Output tensor with same shape as input
        """
        # Handle both (L, d_model) and (B, L, d_model) inputs
        original_shape = x.shape
        if x.ndim == 2:
            # (L, d_model) -> (1, L, d_model)
            x = x[None, :, :]
            if positions is not None and positions.ndim == 1:
                positions = positions[None, :]
        
        B, L, D = x.shape
        
        # Default positions if not provided
        if positions is None:
            positions = jnp.arange(L)[None, :].repeat(B, axis=0)
        
        # Segment IDs (not used for simple sequences)
        segment_ids = None
        
        # === Pre-Norm + Attention ===
        residual = x
        x = self.pre_attn_norm(x)
        
        attn_out, _ = self.attention(
            x,  # query
            x,  # key/value
            positions,
            decoder_segment_ids=segment_ids,
            deterministic=not self.training,
            model_mode=MODEL_MODE_TRAIN if self.training else "prefill",
        )
        
        x = residual + attn_out
        
        # === Pre-Norm + FFN ===
        residual = x
        x = self.pre_ffn_norm(x)
        
        ffn_out = self.mlp(x, deterministic=not self.training)
        
        x = residual + ffn_out
        
        # Dropout on output
        if self.training and self.config.dropout_rate > 0:
            x = self.dropout(x, deterministic=not self.training)
        
        # Restore original shape if needed
        if len(original_shape) == 2:
            x = x[0]  # (1, L, d_model) -> (L, d_model)
        
        return x


class StackedTransformerEncoder(nn.Module):
    """
    Stack of Transformer layers to replace S5's StackedEncoderModel.
    
    Maintains same interface:
        __call__(x, integration_timesteps) -> x
    
    Args:
        config: LOBMAXConfig
        mesh: JAX device mesh
        n_layers: Number of transformer layers
        d_model: Model dimension
        use_embed_layer: Whether to use embedding layer (for tokens)
        vocab_size: Vocabulary size (if use_embed_layer=True)
        training: Training mode
    """
    config: LOBMAXConfig
    mesh: Mesh
    n_layers: int
    d_model: int = None
    use_embed_layer: bool = False
    vocab_size: int = -1
    training: bool = True
    
    def setup(self):
        cfg = self.config
        d_model = self.d_model or cfg.d_model
        dtype = cfg.get_dtype()
        
        # Encoder (embedding or projection)
        if self.use_embed_layer:
            self.encoder = nn.Embed(
                num_embeddings=self.vocab_size,
                features=d_model,
                dtype=dtype,
                name="token_embedding",
            )
        else:
            self.encoder = nn.Dense(
                features=d_model,
                dtype=dtype,
                name="input_projection",
            )
        
        # Stack of transformer layers
        self.layers = [
            TransformerLayer(
                config=cfg,
                mesh=self.mesh,
                layer_idx=i,
                d_model=d_model,
                training=self.training,
                name=f"layer_{i}",
            )
            for i in range(self.n_layers)
        ]
    
    def __call__(self, x, integration_timesteps=None, positions=None):
        """
        Forward pass through stacked transformer.
        
        Args:
            x: Input (L, d_input) or (B, L, d_input)
            integration_timesteps: Unused (compatibility with S5 interface)
            positions: Optional position indices for RoPE
            
        Returns:
            Output (L, d_model) or (B, L, d_model)
        """
        # Handle 2D input
        # Handle inputs
        original_ndim = x.ndim
        
        if self.use_embed_layer:
            # x is indices: (L,) -> (1, L), (B, L) -> (B, L)
            if x.ndim == 1:
                x = x[None, :]
            # B, L = x.shape  <- Don't unpack yet, we need 3D after embedding
        else:
            # x is features: (L, D) -> (1, L, D), (B, L, D) -> (B, L, D)
            if x.ndim == 2:
                x = x[None, :, :]
        
        # We need B and L for positions
        if x.ndim == 2:
            B, L = x.shape
        elif x.ndim == 3:
            B, L, _ = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Default positions
        if positions is None:
            positions = jnp.arange(L)[None, :].repeat(B, axis=0)
        
        # Encode input
        x = self.encoder(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, positions=positions)
        
        # Restore original shape
        if not self.use_embed_layer and original_ndim == 2:
            x = x[0]
        
        return x


class TransformerBookEncoder(nn.Module):
    """
    Book Encoder with Transformer layers.
    Replaces S5's LobBookModel.
    
    Architecture:
        x -> Pre-layers -> Dense projection -> Post-layers
    
    Args:
        config: LOBMAXConfig
        mesh: JAX device mesh
        d_book: Input book dimension
        training: Training mode
    """
    config: LOBMAXConfig
    mesh: Mesh
    d_book: int
    training: bool = True
    
    def setup(self):
        cfg = self.config
        dtype = cfg.get_dtype()
        
        # Pre-processing layers (operate on d_model after projection)
        self.pre_layers = [
            TransformerLayer(
                config=cfg,
                mesh=self.mesh,
                layer_idx=i,
                d_model=cfg.d_model,  # Use model dimension (aligned for Attention)
                training=self.training,
                name=f"pre_layer_{i}",
            )
            for i in range(cfg.n_book_pre_layers)
        ]
        
        # Projection to d_model
        self.projection = nn.Dense(
            features=cfg.d_model,
            dtype=dtype,
            name="book_projection",
        )
        
        # Post-processing layers (operate on d_model)
        self.post_layers = [
            TransformerLayer(
                config=cfg,
                mesh=self.mesh,
                layer_idx=i,
                d_model=cfg.d_model,
                training=self.training,
                name=f"post_layer_{i}",
            )
            for i in range(cfg.n_book_post_layers)
        ]
    
    def __call__(self, x, integration_timesteps=None, positions=None):
        """
        Forward pass through book encoder.
        
        Args:
            x: Book state (L, d_book) or (B, L, d_book)
            integration_timesteps: Unused (S5 compatibility)
            positions: Position indices for RoPE
            
        Returns:
            Encoded book (L, d_model) or (B, L, d_model)
        """
        # Handle 2D input
        original_ndim = x.ndim
        if x.ndim == 2:
            x = x[None, :, :]
        
        B, L, _ = x.shape
        
        # Default positions
        if positions is None:
            positions = jnp.arange(L)[None, :].repeat(B, axis=0)
        
        # Project to d_model first to match Attention config (needed for Flash Attention)
        x = self.projection(x)
        
        # Pre-layers (now operate on d_model)
        for layer in self.pre_layers:
            x = layer(x, positions=positions)
        
        # Post-layers (operate on d_model)
        for layer in self.post_layers:
            x = layer(x, positions=positions)
        
        # Restore original shape
        if original_ndim == 2:
            x = x[0]
        
        return x
