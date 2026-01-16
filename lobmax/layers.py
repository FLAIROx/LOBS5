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
from flax.linen import partitioning as nn_partitioning

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

# Import MoE
_moe = _import_maxtext_module(
    'MaxText.layers.moe',
    os.path.join(MAXTEXT_LAYERS_PATH, 'moe.py')
)
get_routed_moe = _moe.get_routed_moe

# Import initializers
_initializers = _import_maxtext_module(
    'MaxText.layers.initializers',
    os.path.join(MAXTEXT_LAYERS_PATH, 'initializers.py')
)
nd_dense_init = _initializers.nd_dense_init


from lobmax.config import LOBMAXConfig


def get_remat_policy(policy_name: str):
    """
    Get JAX checkpoint policy by name.

    Uses MaxText's built-in checkpoint_name markers in attention/linears layers.

    Policies:
        - "none": No rematerialization (save all activations)
        - "full": Recompute everything (nothing_saveable)
        - "save_dot_except_mlp": Save attention context + out_proj only (~50% memory savings)
        - "minimal": Save all MaxText-marked tensors (~30% memory savings)
        - default: Fallback to checkpoint_dots_with_no_batch_dims
    """
    if policy_name == "none":
        return None
    elif policy_name == "full":
        return jax.checkpoint_policies.nothing_saveable
    elif policy_name == "save_dot_except_mlp":
        # Only save attention context and output projection
        return jax.checkpoint_policies.save_only_these_names('context', 'out_proj')
    elif policy_name == "minimal":
        # Save all MaxText-marked tensors
        return jax.checkpoint_policies.save_only_these_names(
            'context', 'out_proj', 'qkv_proj',
            'query_proj', 'key_proj', 'value_proj',
            'mlpwi', 'mlpwi_0', 'mlpwi_1', 'mlpwo'
        )
    else:
        # Backward compatibility fallback
        return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims


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
        
        if cfg.num_experts > 1:
            # Use MoE Layer
            # Use nd_dense_init(1.0, 'fan_in', 'truncated_normal') as default like in MoE module
            kernel_init = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
            
            self.mlp = get_routed_moe(
                config=cfg,
                num_experts=cfg.num_experts,
                num_experts_per_tok=cfg.num_experts_per_tok,
                mesh=self.mesh,
                kernel_init=kernel_init,
                kernel_axes=('embed', None), # Standard axes
                intermediate_dim=cfg.mlp_dim,
                weight_dtype=weight_dtype,
                dtype=dtype,
                quant=None,
                name=f"moe_mlp",
            )
        else:
            # Use Dense MLP Layer
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
        
        if self.config.num_experts > 1:
            # MoE returns (output, load_balance_loss)
            ffn_out, lb_loss = self.mlp(x)
            # We currently ignore load balance loss in the forward pass return here
            # Ideally, this should be added to the total loss.
            # However, TransformerLayer return signature is (x, ())
            # For now, we rely on the auxiliary loss being handled if we can propagate it,
            # or just use the gradient from the routing decisions if it's differentiable.
            # MaxText adds it to the loss.
            # But the 'aux' output of scan is empty tuple.
            # TODO: Propagate load balance loss.
            pass 
        else:
            ffn_out = self.mlp(x, deterministic=not self.training)
        
        x = residual + ffn_out
        
        # Dropout on output
        if self.training and self.config.dropout_rate > 0:
            x = self.dropout(x, deterministic=not self.training)
        
        # Restore original shape if needed
        if len(original_shape) == 2:
            x = x[0]  # (1, L, d_model) -> (L, d_model)

        # Return (carry, output) for nn.scan compatibility
        # Output is an empty tuple (Flax axes_scan requires tuple structure)
        return x, ()


class StackedTransformerEncoder(nn.Module):
    """
    Stack of Transformer layers using nn.scan for efficient XLA compilation.

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

    def _create_scanned_layers(self):
        """Create nn.scan wrapped transformer layers."""
        cfg = self.config
        d_model = self.d_model or cfg.d_model

        # ScanIn for proper parameter handling during init vs apply
        initializing = self.is_mutable_collection("params")
        params_spec = cfg.param_scan_axis if initializing else nn_partitioning.ScanIn(cfg.param_scan_axis)

        # Apply remat based on policy (uses MaxText's built-in checkpoint_name markers)
        layer_cls = TransformerLayer
        policy = get_remat_policy(cfg.remat_policy)
        if policy is not None:
            layer_cls = nn.remat(TransformerLayer, policy=policy)

        return nn.scan(
            layer_cls,
            variable_axes={"params": params_spec, "intermediates": 0},
            split_rngs={"params": True, "dropout": cfg.enable_dropout},
            in_axes=(nn.broadcast,),  # positions broadcast to all layers
            out_axes=(),  # Match empty tuple output structure
            length=self.n_layers,
            metadata_params={nn.PARTITION_NAME: 'layers'},  # Required for LogicallyPartitioned variables
        )(
            config=cfg,
            mesh=self.mesh,
            d_model=d_model,
            training=self.training,
            name="layers",
        )

    @nn.compact
    def __call__(self, x, integration_timesteps=None, positions=None):
        """
        Forward pass through stacked transformer using nn.scan.

        Args:
            x: Input (L, d_input) or (B, L, d_input)
            integration_timesteps: Unused (compatibility with S5 interface)
            positions: Optional position indices for RoPE

        Returns:
            Output (L, d_model) or (B, L, d_model)
        """
        original_ndim = x.ndim

        if self.use_embed_layer:
            if x.ndim == 1:
                x = x[None, :]
        else:
            if x.ndim == 2:
                x = x[None, :, :]

        if x.ndim == 2:
            B, L = x.shape
        elif x.ndim == 3:
            B, L, _ = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        if positions is None:
            positions = jnp.arange(L)[None, :].repeat(B, axis=0)

        # Encode input
        x = self.encoder(x)

        # Apply transformer layers via nn.scan
        scanned_layers = self._create_scanned_layers()
        x, _ = scanned_layers(x, positions)

        # Restore original shape
        if not self.use_embed_layer and original_ndim == 2:
            x = x[0]

        return x


class TransformerBookEncoder(nn.Module):
    """
    Book Encoder with Transformer layers using nn.scan.
    Replaces S5's LobBookModel.

    Architecture:
        x -> Dense projection -> Pre-layers -> Post-layers

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

        # Projection to d_model
        self.projection = nn.Dense(
            features=cfg.d_model,
            dtype=dtype,
            name="book_projection",
        )

    def _create_scanned_layers(self, n_layers: int, name: str):
        """Create nn.scan wrapped transformer layers."""
        cfg = self.config

        initializing = self.is_mutable_collection("params")
        params_spec = cfg.param_scan_axis if initializing else nn_partitioning.ScanIn(cfg.param_scan_axis)

        # Apply remat based on policy (uses MaxText's built-in checkpoint_name markers)
        layer_cls = TransformerLayer
        policy = get_remat_policy(cfg.remat_policy)
        if policy is not None:
            layer_cls = nn.remat(TransformerLayer, policy=policy)

        return nn.scan(
            layer_cls,
            variable_axes={"params": params_spec, "intermediates": 0},
            split_rngs={"params": True, "dropout": cfg.enable_dropout},
            in_axes=(nn.broadcast,),
            out_axes=(),  # Match empty tuple output structure
            length=n_layers,
            metadata_params={nn.PARTITION_NAME: 'layers'},  # Required for LogicallyPartitioned variables
        )(
            config=cfg,
            mesh=self.mesh,
            d_model=cfg.d_model,
            training=self.training,
            name=name,
        )

    @nn.compact
    def __call__(self, x, integration_timesteps=None, positions=None):
        """
        Forward pass through book encoder using nn.scan.

        Args:
            x: Book state (L, d_book) or (B, L, d_book)
            integration_timesteps: Unused (S5 compatibility)
            positions: Position indices for RoPE

        Returns:
            Encoded book (L, d_model) or (B, L, d_model)
        """
        cfg = self.config
        original_ndim = x.ndim
        if x.ndim == 2:
            x = x[None, :, :]

        B, L, _ = x.shape

        if positions is None:
            positions = jnp.arange(L)[None, :].repeat(B, axis=0)

        # Project to d_model first
        x = self.projection(x)

        # Pre-layers via nn.scan
        if cfg.n_book_pre_layers > 0:
            pre_layers = self._create_scanned_layers(cfg.n_book_pre_layers, "pre_layers")
            x, _ = pre_layers(x, positions)

        # Post-layers via nn.scan
        if cfg.n_book_post_layers > 0:
            post_layers = self._create_scanned_layers(cfg.n_book_post_layers, "post_layers")
            x, _ = post_layers(x, positions)

        # Restore original shape
        if original_ndim == 2:
            x = x[0]

        return x
