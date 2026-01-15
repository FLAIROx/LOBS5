"""
LOBMAX Configuration

Minimal configuration compatible with MaxText Attention module.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple
import jax.numpy as jnp


@dataclass
class LOBMAXConfig:
    """
    Configuration for LOBMAX model.
    
    This config provides the minimal set of parameters needed to use
    MaxText's Attention module within LOBMAX.
    """
    
    # === Core Model Parameters ===
    d_model: int = 2048           # Model dimension (emb_dim in MaxText)
    num_heads: int = 16           # Number of attention heads
    num_kv_heads: int = 16        # Number of KV heads (for GQA, set < num_heads)
    head_dim: int = 128           # Dimension per head (d_model / num_heads)
    mlp_dim: int = 8192           # FFN intermediate dimension (4 * d_model)
    
    # === Layer Configuration ===
    n_message_layers: int = 2     # Layers for Message Encoder
    n_book_pre_layers: int = 1    # Layers for Book Encoder (before projection)
    n_book_post_layers: int = 1   # Layers for Book Encoder (after projection)
    n_layers: int = 24            # Layers for Fused Transformer
    
    # === Vocabulary ===
    vocab_size: int = 128         # Message token vocabulary size
    d_book: int = 40              # Book state dimension (input)
    n_classes: int = 128          # Output classes (prediction targets)
    
    # === Sequence Length ===
    max_target_length: int = 12288  # Maximum sequence length (24 * 512)
    max_prefill_predict_length: int = -1
    
    # === Attention Configuration ===
    attention: str = "flash"      # Attention kernel: "dot_product", "flash", "cudnn_flash_te"
    attention_type: str = "global"  # AttentionType enum value
    float32_qk_product: bool = True   # Compute QK product in float32
    float32_logits: bool = True       # Cast logits to float32
    
    # === RoPE Configuration ===
    rope_type: str = "llama3.1"   # RoPE type: "default", "llama3.1", "yarn"
    rope_min_timescale: int = 1
    rope_max_timescale: int = 10000
    rope_linear_scaling_factor: float = 1.0
    rope_use_scale: bool = True   # LLaMA 3.1 scaling
    
    # === Normalization ===
    normalization_layer_epsilon: float = 1e-6
    prenorm: bool = True
    
    # === Dropout ===
    dropout_rate: float = 0.0
    enable_dropout: bool = False
    
    # === Data Types ===
    dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"
    
    # === Decoder Block Type ===
    decoder_block: str = "llama2"
    
    # === Sharding (simplified for data parallel) ===
    shard_mode: str = "auto"      # ShardMode.AUTO
    
    # === FFN Configuration === 
    mlp_activations: Tuple[str, str] = ("silu", "linear")  # SwiGLU
    fused_qkv: bool = False       # Whether to fuse QKV projections
    use_qk_norm: bool = False     # QK normalization
    use_bias: bool = False        # Bias in projections
    
    # === Pooling Mode ===
    mode: str = "none"            # "pool", "last", "none", "ema"
    
    # === MaxText Compatibility ===
    scan_layers: bool = False
    record_internal_nn_metrics: bool = False
    matmul_precision: str = "default"
    use_iota_embed: bool = False
    logits_via_embedding: bool = False
    logits_dot_in_fp32: bool = True
    cast_logits_to_fp32: bool = True
    final_logits_soft_cap: float = 0.0
    normalize_embedding_logits: bool = False
    
    # === Cache Configuration (for inference) ===
    prefill_cache_axis_order: str = "1,2,0,3"
    ar_cache_axis_order: str = "1,2,0,3"
    compute_axis_order: str = "0,1,2,3"
    reshape_q: bool = False
    
    # === Additional MaxText Required Fields ===
    use_untrainable_positional_embedding: bool = False
    trainable_position_size: int = 0
    parameter_memory_host_offload: bool = False
    ici_context_autoregressive_parallelism: int = 1
    local_rope_max_timescale: int = -1
    attention_sink: bool = False
    chunk_attn_window_size: int = 0
    expert_shard_attention_option: str = "off"
    
    # === kv cache quantization ===
    quantize_kvcache: bool = False
    
    # === RoPE Advanced ===
    max_position_embeddings: int = 12288
    original_max_position_embeddings: int = 12288
    rope_factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    rope_attention_scaling: float = 1.0
    rope_interleave: bool = False
    rope_truncate: bool = False
    partial_rotary_factor: float = 1.0
    
    # === Paged Attention (Unused) ===
    pagedattn_num_pages: int = 0
    pagedattn_pages_per_compute_block: int = 0
    pagedattn_tokens_per_page: int = 0
    
    # === ViT (Unused) ===
    hidden_size_for_vit: int = 0
    image_size_for_vit: int = 0
    num_attention_heads_for_vit: int = 0
    patch_size_for_vit: int = 0
    rope_theta_for_vit: int = 0
    spatial_merge_size_for_vit: int = 0
    
    # === Misc MaxText ===
    use_chunked_prefill: bool = False
    
    # === MoBA (Mixture of Block Attention) ===
    moba_chunk_size: int = 1024
    moba_topk: int = 2
    
    # === Model Name (for MaxText checking) ===
    model_name: str = "lobmax"
    
    # === Logical Axis Rules ===
    logical_axis_rules: Tuple[Tuple[str, str], ...] = (
        ('batch', 'data'),
        ('activation_batch', 'data'),
        ('activation_length', 'model'),
        ('heads', 'model'),
        ('kv', 'model'),
        ('embed', 'model'),
        ('mlp', 'model'),
        ('vocab', 'model'),
        ('norm', 'data'),
    )
    
    @property
    def emb_dim(self) -> int:
        """Alias for d_model (MaxText naming)."""
        return self.d_model
    
    @property
    def base_emb_dim(self) -> int:
        """Base embedding dimension."""
        return self.d_model
    
    @property
    def num_query_heads(self) -> int:
        """Alias for num_heads."""
        return self.num_heads
    
    def get_dtype(self):
        """Get JAX dtype from string."""
        if self.dtype == "bfloat16":
            return jnp.bfloat16
        elif self.dtype == "float32":
            return jnp.float32
        elif self.dtype == "float16":
            return jnp.float16
        else:
            return jnp.float32
    
    def get_weight_dtype(self):
        """Get JAX weight dtype from string."""
        if self.weight_dtype == "bfloat16":
            return jnp.bfloat16
        elif self.weight_dtype == "float32":
            return jnp.float32
        elif self.weight_dtype == "float16":
            return jnp.float16
        else:
            return jnp.float32
    
    @classmethod
    def from_args(cls, args):
        """Create config from argparse namespace."""
        return cls(
            d_model=getattr(args, 'd_model', 2048),
            num_heads=getattr(args, 'num_heads', 16),
            num_kv_heads=getattr(args, 'num_kv_heads', 16),
            head_dim=getattr(args, 'head_dim', 128),
            mlp_dim=getattr(args, 'mlp_dim', 8192),
            n_message_layers=getattr(args, 'n_message_layers', 2),
            n_book_pre_layers=getattr(args, 'n_book_pre_layers', 1),
            n_book_post_layers=getattr(args, 'n_book_post_layers', 1),
            n_layers=getattr(args, 'n_layers', 24),
            vocab_size=getattr(args, 'vocab_size', 128),
            n_classes=getattr(args, 'n_classes', 128),
            max_target_length=getattr(args, 'max_target_length', 12288),
            max_position_embeddings=getattr(args, 'max_target_length', 12288),
            original_max_position_embeddings=getattr(args, 'max_target_length', 12288),
            attention=getattr(args, 'attention_kernel', 'flash'),
            rope_type=getattr(args, 'rope_type', 'llama3.1'),
            dropout_rate=getattr(args, 'p_dropout', 0.0),
            dtype="bfloat16" if getattr(args, 'use_bf16', True) else "float32",
            weight_dtype="bfloat16" if getattr(args, 'use_bf16', True) else "float32",
            prenorm=getattr(args, 'prenorm', True),
            mode=getattr(args, 'mode', 'none'),
        )
