"""
JAX Native Sliding Window Attention
Uses jax.nn.dot_product_attention with local_window_size for O(n*W) complexity.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Optional


class JAXNativeAttention(nn.Module):
    """
    JAX Native Attention using jax.nn.dot_product_attention.

    This is a lightweight wrapper to test JAX's native sliding window support
    without requiring Transformer Engine installation.
    """
    num_heads: int
    head_dim: int
    sliding_window_size: int = 0  # 0 = global, >0 = local sliding window
    dropout_rate: float = 0.0
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key_value: jnp.ndarray,
        positions: jnp.ndarray,
        decoder_segment_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        model_mode: str = "train",
    ):
        """
        Forward pass using JAX native attention.

        Args:
            query: [batch, seq_len, d_model]
            key_value: [batch, seq_len, d_model] (same as query for self-attention)
            positions: [seq_len] - position indices (not used in this simple version)
            decoder_segment_ids: Optional segment IDs
            deterministic: Whether to use dropout
            model_mode: "train" or "prefill"

        Returns:
            (output, None): Output tensor and dummy cache (for compatibility)
        """
        batch, seq_len, d_model = query.shape

        # Linear projections: Q, K, V
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, head_dim]
        q = nn.Dense(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="query"
        )(query)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)

        k = nn.Dense(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="key"
        )(key_value)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)

        v = nn.Dense(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="value"
        )(key_value)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim] for JAX attention
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # ===== JAX Native Attention with Sliding Window =====
        if self.sliding_window_size > 0:
            # Use local sliding window
            print(f"[JAX Native Attention] Using local_window_size=({self.sliding_window_size}, 0) - causal")
            local_window = (self.sliding_window_size, 0)  # (left, right) - causal
        else:
            # Global attention (no window)
            print(f"[JAX Native Attention] Using global attention (no window)")
            local_window = None

        # Call JAX native dot_product_attention
        attn_output = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            local_window_size=local_window,
            implementation="cudnn",  # Use cuDNN for GPU acceleration
            is_causal=True,  # Causal masking for autoregressive
        )

        # Transpose back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

        # Merge heads: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, d_model]
        attn_output = attn_output.reshape(batch, seq_len, self.num_heads * self.head_dim)

        # Output projection
        output = nn.Dense(
            features=d_model,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="out"
        )(attn_output)

        # Return (output, None) for compatibility with MaxText interface
        # MaxText attention returns (output, cache), we don't use cache here
        return output, None


def create_jax_native_attention(
    num_query_heads: int,
    head_dim: int,
    sliding_window_size: int = 0,
    dropout_rate: float = 0.0,
    dtype: Any = jnp.bfloat16,
    **kwargs  # Accept and ignore other MaxText-specific kwargs
):
    """
    Factory function to create JAX Native Attention layer.

    This function signature matches MaxText's attention_as_linen for easy drop-in replacement.
    """
    return JAXNativeAttention(
        num_heads=num_query_heads,
        head_dim=head_dim,
        sliding_window_size=sliding_window_size,
        dropout_rate=dropout_rate,
        dtype=dtype,
    )
