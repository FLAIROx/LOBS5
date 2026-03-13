"""Transformer block as drop-in replacement for S5SSM.

Same interface: __call__(input_sequence: (L, H)) -> (L, H).
TransformerBlock is a complete Pre-LN Transformer block with both
internal residual connections and layer norms. When used inside
SequenceLayer, the S5-specific norm/activation/residual wrapping
is bypassed (detected via is_transformer flag).
"""
from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn


def _flash_attention_fn(query, key, value):
    """cuDNN flash attention for Flax MultiHeadDotProductAttention.

    is_causal=True handles causal masking inside the fused kernel —
    no materialized (L, L) attention matrix, O(L) memory instead of O(L²).
    Accepts 3D (T, N, H) inputs — JAX's _ensure_4d handles batch dim internally.

    Falls back to XLA when head_dim is incompatible with cuDNN
    (must be <= 256 and multiple of 8). This handles the book encoder
    pre-layers where H=d_book=503, n_heads=1 → head_dim=503.
    Branch is resolved at JIT trace time (zero runtime cost).
    """
    head_dim = query.shape[-1]
    if head_dim <= 256 and head_dim % 8 == 0:
        impl = 'cudnn'
    else:
        impl = 'xla'
    return jax.nn.dot_product_attention(
        query, key, value,
        is_causal=True,
        implementation=impl,
    )


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> jnp.ndarray:
    """Generate sinusoidal positional encodings.

    Args:
        seq_len: sequence length L
        d_model: model dimension H

    Returns:
        (L, H) positional encoding matrix
    """
    positions = jnp.arange(seq_len)[:, None]  # (L, 1)
    dims = jnp.arange(d_model)[None, :]       # (1, H)
    angles = positions / jnp.power(10000.0, (2 * (dims // 2)) / d_model)
    # sin on even indices, cos on odd indices
    pe = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return pe


def positional_encoding_at_positions(positions: jnp.ndarray,
                                     d_model: int) -> jnp.ndarray:
    """Sinusoidal PE at arbitrary absolute positions (for KV cache inference)."""
    pos = positions[:, None].astype(jnp.float32)
    dims = jnp.arange(d_model)[None, :]
    angles = pos / jnp.power(10000.0, (2 * (dims // 2)) / d_model)
    return jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))


class TransformerBlock(nn.Module):
    """Drop-in replacement for S5SSM. Same interface: (L, H) -> (L, H).

    Complete Pre-LN Transformer block with:
    - Sinusoidal positional encoding
    - Causal self-attention with Pre-LN
    - FFN sublayer with Pre-LN
    - Two internal residual connections

    When used in SequenceLayer, set is_transformer=True to bypass the
    external S5-specific norm/activation/residual wrapping.
    """
    H: int           # d_model
    n_heads: int     # number of attention heads
    d_ff: int        # FFN intermediate dim (default 4*H)
    max_cache_len: int = 25000  # KV cache size for AR inference
    dropout: float = 0.0
    training: bool = True
    dtype: jnp.dtype = jnp.float32  # compute dtype (bf16 for mixed precision)
    use_flash: bool = False  # cuDNN flash attention (Hopper+ GPU required)

    # Accept and ignore S5-specific kwargs for compatibility with SequenceLayer
    step_rescale: float = 1.0

    def setup(self):
        # Adjust n_heads if H is not divisible (e.g. book pre-layers with H=d_book=503)
        effective_n_heads = self.n_heads
        while effective_n_heads > 1 and self.H % effective_n_heads != 0:
            effective_n_heads -= 1
        attn_kwargs = dict(
            num_heads=effective_n_heads,
            qkv_features=self.H,
            dropout_rate=self.dropout,
            deterministic=not self.training,
            dtype=self.dtype,
        )
        if self.use_flash:
            attn_kwargs['attention_fn'] = _flash_attention_fn
        self.attn = nn.MultiHeadDotProductAttention(**attn_kwargs)
        self.norm1 = nn.LayerNorm(dtype=self.dtype)
        self.norm2 = nn.LayerNorm(dtype=self.dtype)
        self.ff = nn.Sequential([
            nn.Dense(self.d_ff, dtype=self.dtype),
            nn.gelu,
            nn.Dense(self.H, dtype=self.dtype),
        ])
        self.drop = nn.Dropout(self.dropout, deterministic=not self.training)

    def __call__(self, input_sequence):
        """Forward pass with causal masking and positional encoding.

        Args:
            input_sequence: (L, H) input tensor

        Returns:
            (L, H) output tensor
        """
        L = input_sequence.shape[0]

        # Add sinusoidal positional encoding (FP32 for sin/cos precision)
        pe = sinusoidal_positional_encoding(L, self.H)
        x = input_sequence + pe
        x = x.astype(self.dtype)  # cast to compute dtype (e.g. bf16)

        # Flash attention: is_causal=True in cuDNN kernel — no mask materialization
        # Standard: explicit (L, L) bool mask for XLA attention
        if not self.use_flash:
            mask = nn.make_causal_mask(jnp.ones((1, L)), dtype=bool)  # (1, 1, L, L)
            mask = mask[0, 0]  # (L, L) — compatible with unbatched attention
        else:
            mask = None

        # Pre-norm attention + residual
        h = self.norm1(x)
        h = self.attn(h, mask=mask)
        h = self.drop(h)
        x = x + h

        # Pre-norm FFN + residual
        h = self.norm2(x)
        h = self.ff(h)
        h = self.drop(h)
        x = x + h

        return x

    def __call_rnn__(self, hidden, input_sequence, resets):
        """Autoregressive inference with static KV cache.

        Accesses nn.MultiHeadDotProductAttention's projection weights
        directly via self.attn.variables (no param name changes needed).

        hidden arrives as (k_cache, v_cache, pos) where shapes may carry
        a leading-1 placeholder dim from initialize_cache (for inner vmap
        compatibility).  We squeeze it so the attention logic works on 3D.
        """
        k_cache, v_cache, pos = hidden
        # Squeeze leading-1 from initialize_cache: (1, nh, L, hd) → (nh, L, hd)
        k_cache = jnp.squeeze(k_cache, axis=0) if k_cache.ndim == 4 else k_cache
        v_cache = jnp.squeeze(v_cache, axis=0) if v_cache.ndim == 4 else v_cache
        pos = jnp.squeeze(pos) if pos.ndim >= 1 else pos
        # k_cache: (nh, max_len, hd), v_cache: same, pos: () scalar

        L = input_sequence.shape[0]
        max_len = k_cache.shape[1]

        # Effective n_heads (same logic as setup)
        nh = self.n_heads
        while nh > 1 and self.H % nh != 0:
            nh -= 1
        hd = self.H // nh

        # PE at absolute positions [pos, pos+L)
        positions = pos + jnp.arange(L)
        pe = positional_encoding_at_positions(positions, self.H)
        x = (input_sequence + pe).astype(self.dtype)

        # Pre-norm
        h = self.norm1(x)

        # Access MHDA's projection weights from variable tree
        attn_p = self.attn.variables.get('params', {})
        q = jnp.einsum('...d,dnk->...nk', h, attn_p['query']['kernel'])
        k_new = jnp.einsum('...d,dnk->...nk', h, attn_p['key']['kernel'])
        v_new = jnp.einsum('...d,dnk->...nk', h, attn_p['value']['kernel'])
        if 'bias' in attn_p.get('query', {}):
            q = q + attn_p['query']['bias']
            k_new = k_new + attn_p['key']['bias']
            v_new = v_new + attn_p['value']['bias']

        # Write new K/V into cache
        k_new_t = jnp.transpose(k_new, (1, 0, 2))  # (nh, L, hd)
        v_new_t = jnp.transpose(v_new, (1, 0, 2))
        k_cache = jax.lax.dynamic_update_slice(k_cache, k_new_t, (0, pos, 0))
        v_cache = jax.lax.dynamic_update_slice(v_cache, v_new_t, (0, pos, 0))

        # Attend: Q over new tokens, K/V over cache
        valid_len = pos + L
        qt = jnp.transpose(q, (1, 0, 2))  # (nh, L, hd)
        scale = jnp.sqrt(jnp.float32(hd))
        w = jnp.einsum('hqd,hkd->hqk', qt, k_cache) / scale

        # Causal + valid mask
        cache_positions = jnp.arange(max_len)
        query_positions = pos + jnp.arange(L)
        mask = (cache_positions[None, :] <= query_positions[:, None]) & \
               (cache_positions[None, :] < valid_len)
        w = jnp.where(mask[None, :, :], w, jnp.finfo(self.dtype).min)
        w = jax.nn.softmax(w, axis=-1)
        attn_out = jnp.einsum('hqk,hkd->hqd', w, v_cache)
        attn_out = jnp.transpose(attn_out, (1, 0, 2))  # (L, nh, hd)

        # Output projection
        out = jnp.einsum('...nk,nkd->...d', attn_out, attn_p['out']['kernel'])
        if 'bias' in attn_p.get('out', {}):
            out = out + attn_p['out']['bias']
        x = x + self.drop(out)

        # Pre-norm FFN + residual
        h = self.norm2(x)
        h = self.ff(h)
        h = self.drop(h)
        x = x + h

        new_hidden = (k_cache, v_cache, pos + L)
        return new_hidden, x

    @staticmethod
    def initialize_cache(batch_size, n_heads, head_dim, max_cache_len,
                         dtype=jnp.float32):
        """Create empty KV cache for one Transformer layer.

        Returns 4D tensors with leading dim=1 — placeholder batch dim
        for the inner nn.vmap (BatchPaddedLobPredModel).  The outer vmap
        (generate_batched) handles the real batch; the inner vmap maps
        over this axis-0.  ``__call_rnn__`` squeezes the leading dim so
        the attention logic works on 3D (n_heads, max_cache_len, head_dim).
        """
        k = jnp.zeros((1, n_heads, max_cache_len, head_dim), dtype=dtype)
        v = jnp.zeros((1, n_heads, max_cache_len, head_dim), dtype=dtype)
        idx = jnp.zeros((1,), dtype=jnp.int32)
        return (k, v, idx)


# Flag for SequenceLayer to detect transformer blocks
TransformerBlock.is_transformer = True


def init_TransformerBlock(H, n_heads, d_ff=0, dropout=0.0, dtype=jnp.float32,
                          use_flash=False, remat=False, max_cache_len=25000,
                          **kwargs):
    """Convenience factory matching init_S5SSM interface.

    Accepts and ignores S5-specific kwargs (Lambda_re_init, V, Vinv, etc.)
    for compatibility with code that passes SSM-specific parameters.

    Args:
        H: model dimension (d_model)
        n_heads: number of attention heads
        d_ff: FFN intermediate dim. 0 means 4*H.
        dropout: dropout rate
        dtype: compute dtype (e.g. jnp.bfloat16 for mixed precision)
        use_flash: use cuDNN flash attention (requires Hopper+ GPU)
        remat: gradient checkpointing via nn.remat (trades compute for memory)
        **kwargs: ignored (S5 compatibility)

    Returns:
        partial(TransformerBlock, ...) — same pattern as init_S5SSM
    """
    if d_ff <= 0:
        d_ff = 4 * H

    block_cls = TransformerBlock
    if remat:
        block_cls = nn.remat(TransformerBlock)
        # nn.remat subclass inherits is_transformer via Python MRO

    return partial(
        block_cls,
        H=H,
        n_heads=n_heads,
        d_ff=d_ff,
        max_cache_len=max_cache_len,
        dropout=dropout,
        dtype=dtype,
        use_flash=use_flash,
    )
