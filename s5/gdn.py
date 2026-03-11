"""Gated Delta Networks (GDN) / Kimi Delta Attention (KDA) SSM layer.

Drop-in replacement for S5SSM with the same interface:
  __call__(input_sequence: (L, H)) -> (L, H)
  __call_rnn__(hidden, input_sequence, resets) -> hidden, (L, H)

References:
  - GDN: Yang et al., "Gated Delta Networks", ICLR 2025
  - KDA: Kimi Team, "Kimi-VL", 2025 (per-key-dim alpha variant)
  - FLA: Triton-based Flash Linear Attention (naive chunkwise algorithm)
"""
from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

import os

# Triton WY kernel: disabled by default until validated.
# Enable with: USE_TRITON_WY=1
_USE_TRITON_WY = os.environ.get('USE_TRITON_WY', '0') == '1'
if _USE_TRITON_WY:
    try:
        from s5.gdn_triton_kernels import wy_correction_fused, HAS_TRITON
    except ImportError:
        HAS_TRITON = False
else:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Causal depthwise Conv1d (safe path — no reliance on nn.Conv padding mode)
# ---------------------------------------------------------------------------
class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise 1D convolution: (L, C) -> (L, C) with left-padding."""
    channels: int
    kernel_size: int = 4

    @nn.compact
    def __call__(self, x):
        # x: (L, C)
        k = self.kernel_size
        C = self.channels
        kernel = self.param('kernel', lecun_normal(), (k, C))  # (k, C)
        bias = self.param('bias', nn.initializers.zeros, (C,))
        # Left-pad: (L, C) -> (L + k - 1, C)
        x_padded = jnp.pad(x, ((k - 1, 0), (0, 0)))
        # Depthwise conv: feature_group_count=C means each channel is convolved independently
        # RHS shape for ('NTC', 'TIO', 'NTC'): (k, I_per_group=1, O=C)
        kernel_reshaped = kernel[:, :, None]  # (k, C, 1)
        # Transpose to (k, 1, C) for TIO format where I=1 per group, O=C
        kernel_reshaped = kernel[:, None, :]  # (k, 1, C)
        windows = jax.lax.conv_general_dilated(
            x_padded[None, :, :],           # (1, L+k-1, C)
            kernel_reshaped,                # (k, 1, C)
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NTC', 'TIO', 'NTC'),
            feature_group_count=C,
        )  # (1, L, C)
        return windows[0] + bias


# ---------------------------------------------------------------------------
# GDNSSM Module
# ---------------------------------------------------------------------------
class GDNSSM(nn.Module):
    """Gated Delta Network SSM layer.

    Same interface as S5SSM:
      __call__(input_sequence) -> output_sequence      (L, H) -> (L, H)
      __call_rnn__(hidden, input_sequence, resets) -> (hidden, output_sequence)

    Args:
        H:             feature dim (d_model). Overridable via partial(ssm, H=d_book).
        num_heads:     number of attention heads.
        head_dim:      key/query dimension per head.
        expand_v:      value expansion factor (head_v_dim = head_dim * expand_v).
        chunk_size:    chunkwise parallel chunk size.
        use_conv:      whether to apply causal depthwise Conv1d(k=4) on q,k,v.
        use_kda:       per-key-dim alpha (KDA) vs per-head scalar alpha (GDN).
        step_rescale:  ACCEPTED but IGNORED — SequenceLayer passes this to all SSMs.
    """
    H: int
    num_heads: int
    head_dim: int = 128
    expand_v: int = 2
    chunk_size: int = 64
    use_conv: bool = True
    use_kda: bool = False
    step_rescale: float = 1.0  # ignored, compatibility with SequenceLayer

    def setup(self):
        # Auto-adjust for small H (book pre-layers where H=d_book)
        self.eff_heads = min(self.num_heads, max(1, self.H // self.head_dim))
        self.eff_head_dim = min(self.head_dim, self.H)
        self.head_v_dim = self.eff_head_dim * self.expand_v

        nh = self.eff_heads
        hd = self.eff_head_dim
        hvd = self.head_v_dim

        # Projections
        self.q_proj = nn.Dense(nh * hd, use_bias=False)
        self.k_proj = nn.Dense(nh * hd, use_bias=False)
        self.v_proj = nn.Dense(nh * hvd, use_bias=False)

        # Beta gate (write strength): per-head scalar
        self.b_proj = nn.Dense(nh, use_bias=True)

        # Alpha gate (decay/erase): per-key-dim (KDA) or per-head (GDN)
        alpha_dim = nh * hd if self.use_kda else nh
        self.gk_proj = nn.Dense(alpha_dim, use_bias=True)

        # Output gate
        self.g_proj = nn.Dense(nh * hvd, use_bias=False)

        # Output projection: merge heads back to H
        self.o_proj = nn.Dense(self.H, use_bias=False)

        # Optional causal Conv1d(k=4) on q, k, v
        if self.use_conv:
            self.q_conv = CausalDepthwiseConv1d(channels=nh * hd, kernel_size=4)
            self.k_conv = CausalDepthwiseConv1d(channels=nh * hd, kernel_size=4)
            self.v_conv = CausalDepthwiseConv1d(channels=nh * hvd, kernel_size=4)

        # RMSNorm per head (applied to output before gating)
        self.out_norm = nn.RMSNorm(hvd)

        # Feedthrough parameter (matches S5 convention)
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

    def __call__(self, input_sequence):
        """Chunkwise parallel forward (training mode).

        Args:
            input_sequence: (L, H)
        Returns:
            output: (L, H)
        """
        L_orig = input_sequence.shape[0]
        x = input_sequence
        nh = self.eff_heads
        hd = self.eff_head_dim
        hvd = self.head_v_dim
        C = self.chunk_size

        # --- Projections ---
        q = self.q_proj(x)  # (L, nh*hd)
        k = self.k_proj(x)  # (L, nh*hd)
        v = self.v_proj(x)  # (L, nh*hvd)

        # Optional Conv1d + SiLU
        if self.use_conv:
            q = nn.silu(self.q_conv(q))
            k = nn.silu(self.k_conv(k))
            v = nn.silu(self.v_conv(v))

        # L2 normalize q, k; scale q by 1/sqrt(head_dim) (per FLA/Qwen3 reference)
        q = q.reshape(L_orig, nh, hd)
        k = k.reshape(L_orig, nh, hd)
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        q = q * (hd ** -0.5)

        v = v.reshape(L_orig, nh, hvd)

        # Beta (write gate): sigmoid, per-head
        beta = jax.nn.sigmoid(self.b_proj(x))  # (L, nh)

        # Alpha (decay gate): log-sigmoid
        gk_raw = self.gk_proj(x)  # (L, nh*hd) or (L, nh)
        alpha_log = jax.nn.log_sigmoid(gk_raw)  # negative values (decay)

        if self.use_kda:
            alpha_log = alpha_log.reshape(L_orig, nh, hd)  # per-key-dim
        else:
            alpha_log = alpha_log.reshape(L_orig, nh, 1)   # per-head, broadcast over hd

        # Output gate
        g = nn.silu(self.g_proj(x)).reshape(L_orig, nh, hvd)  # (L, nh, hvd)

        # --- Pad to multiple of chunk_size ---
        pad_len = (C - L_orig % C) % C
        L_padded = L_orig + pad_len
        num_chunks = L_padded // C

        if pad_len > 0:
            q = jnp.pad(q, ((0, pad_len), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, pad_len), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, pad_len), (0, 0), (0, 0)))
            beta = jnp.pad(beta, ((0, pad_len), (0, 0)))
            alpha_log = jnp.pad(alpha_log, ((0, pad_len), (0, 0), (0, 0)))
            g = jnp.pad(g, ((0, pad_len), (0, 0), (0, 0)))

        # Reshape to chunks: (num_chunks, C, nh, dim)
        q = q.reshape(num_chunks, C, nh, hd)
        k = k.reshape(num_chunks, C, nh, hd)
        v = v.reshape(num_chunks, C, nh, hvd)
        beta = beta.reshape(num_chunks, C, nh)
        alpha_log = alpha_log.reshape(num_chunks, C, nh, -1)  # (nc, C, nh, hd or 1)
        g = g.reshape(num_chunks, C, nh, hvd)

        # --- Chunkwise parallel computation ---
        o = _chunkwise_gdn(q, k, v, beta, alpha_log, nh, hd, hvd, C, num_chunks)

        # --- Post-processing ---
        # o: (num_chunks, C, nh, hvd)
        o = o.reshape(L_padded, nh, hvd)
        o = o[:L_orig]  # unpad
        g = g.reshape(L_padded, nh, hvd)[:L_orig]

        # RMSNorm per head, then multiply by output gate
        o = self.out_norm(o.reshape(-1, hvd)).reshape(L_orig, nh, hvd)
        o = o * g

        # Merge heads and project
        o = o.reshape(L_orig, nh * hvd)
        o = self.o_proj(o)  # (L, H)

        # Feedthrough
        Du = input_sequence * self.D[None, :]
        return o + Du

    def __call_rnn__(self, hidden, input_sequence, resets):
        """Fused recurrent forward (inference mode).

        Args:
            hidden: (1, nh, hvd, hd) float32 — the state matrix S
            input_sequence: (L, H)
            resets: (L,) or None — reset signals (unused for now, kept for interface)
        Returns:
            new_hidden: (1, nh, hvd, hd) float32
            output: (L, H)
        """
        L = input_sequence.shape[0]
        x = input_sequence
        nh = self.eff_heads
        hd = self.eff_head_dim
        hvd = self.head_v_dim

        # --- Projections ---
        q = self.q_proj(x)  # (L, nh*hd)
        k = self.k_proj(x)
        v = self.v_proj(x)  # (L, nh*hvd)

        if self.use_conv:
            q = nn.silu(self.q_conv(q))
            k = nn.silu(self.k_conv(k))
            v = nn.silu(self.v_conv(v))

        # L2 normalize; scale q by 1/sqrt(head_dim) (per FLA/Qwen3 reference)
        q = q.reshape(L, nh, hd)
        k = k.reshape(L, nh, hd)
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        q = q * (hd ** -0.5)

        v = v.reshape(L, nh, hvd)

        beta = jax.nn.sigmoid(self.b_proj(x))  # (L, nh)

        gk_raw = self.gk_proj(x)
        alpha_log = jax.nn.log_sigmoid(gk_raw)
        if self.use_kda:
            alpha_log = alpha_log.reshape(L, nh, hd)
        else:
            alpha_log = alpha_log.reshape(L, nh, 1)

        g = nn.silu(self.g_proj(x)).reshape(L, nh, hvd)

        # --- Sequential scan ---
        S_init = hidden[0]  # (nh, hvd, hd)

        def rnn_step(S, inp):
            q_t, k_t, v_t, beta_t, alpha_log_t, g_t = inp
            # S: (nh, hvd, hd)
            # alpha_log_t: (nh, hd) or (nh, 1)
            alpha = jnp.exp(alpha_log_t)  # (nh, hd) or (nh, 1)
            # Decay state: broadcast alpha over hvd dim
            S = S * alpha[:, None, :]  # (nh, hvd, hd) * (nh, 1, hd_or_1)

            # Delta rule: v - S @ k
            Sk = jnp.einsum('nvk,nk->nv', S, k_t)  # (nh, hvd)
            delta = v_t - Sk  # (nh, hvd)

            # Update: S += beta * outer(delta, k)
            S = S + jnp.einsum('nv,nk->nvk', beta_t[:, None] * delta, k_t)

            # Output: o = S @ q
            o_t = jnp.einsum('nvk,nk->nv', S, q_t)  # (nh, hvd)
            return S, o_t

        # beta: (L, nh) -> need (L, nh, 1) for elementwise with delta (nh, hvd)
        # Pack inputs for scan
        scan_inputs = (
            q,                 # (L, nh, hd)
            k,                 # (L, nh, hd)
            v,                 # (L, nh, hvd)
            beta,              # (L, nh)
            alpha_log,         # (L, nh, hd or 1)
            g,                 # (L, nh, hvd) — not used in step, but need for output
        )

        S_final, o_seq = jax.lax.scan(rnn_step, S_init, scan_inputs)
        # o_seq: (L, nh, hvd)

        # RMSNorm per head, gating
        o_seq = self.out_norm(o_seq.reshape(-1, hvd)).reshape(L, nh, hvd)
        o_seq = o_seq * g  # (L, nh, hvd)

        # Merge heads
        o_seq = o_seq.reshape(L, nh * hvd)
        o_seq = self.o_proj(o_seq)  # (L, H)

        # Feedthrough
        Du = input_sequence * self.D[None, :]
        output = o_seq + Du

        return S_final[None], output  # (1, nh, hvd, hd), (L, H)


# ---------------------------------------------------------------------------
# Chunkwise GDN computation with WY correction (true delta rule)
# ---------------------------------------------------------------------------
def _chunkwise_gdn(q, k, v, beta, alpha_log, nh, hd, hvd, C, num_chunks):
    """Chunkwise gated delta rule with WY correction.

    Implements the algorithm from Yang et al. "Gated Delta Networks" (ICLR 2025),
    matching torch_chunk_gated_delta_rule from FLA/Qwen3-next reference.

    The WY correction ensures __call__ (chunkwise) == __call_rnn__ (recurrent)
    by accounting for within-chunk sequential dependencies in the delta rule.

    Args:
        q:         (num_chunks, C, nh, hd)
        k:         (num_chunks, C, nh, hd)
        v:         (num_chunks, C, nh, hvd)
        beta:      (num_chunks, C, nh)
        alpha_log: (num_chunks, C, nh, hd_or_1)

    Returns:
        o: (num_chunks, C, nh, hvd)
    """
    nc = num_chunks

    # Cumulative decay within each chunk
    decay_cum = jnp.cumsum(alpha_log, axis=1)  # (nc, C, nh, hd_or_1)
    alpha_dim = alpha_log.shape[-1]

    # Masks
    causal_mask = jnp.tril(jnp.ones((C, C)))        # (C, C) incl diagonal
    strict_lower = jnp.tril(jnp.ones((C, C)), k=-1)  # (C, C) excl diagonal

    # Beta-scaled keys and values
    k_beta = k * beta[:, :, :, None]   # (nc, C, nh, hd)
    v_beta = v * beta[:, :, :, None]   # (nc, C, nh, hvd)

    # =====================================================================
    # Step 1: Compute causal decay mask + intra-chunk attention + WY L matrix
    # =====================================================================
    if alpha_dim == 1:
        # GDN: scalar decay per head
        decay_cum_s = decay_cum[:, :, :, 0]  # (nc, C, nh)
        decay_diff = decay_cum_s[:, :, None, :] - decay_cum_s[:, None, :, :]
        causal_4d = causal_mask[None, :, :, None]
        decay_diff_safe = jnp.where(causal_4d, decay_diff, 0.0)
        decay_mask_4d = jnp.exp(decay_diff_safe) * causal_4d  # (nc, C, C, nh)

        # q@k^T via batched matmul (for intra-chunk output attention)
        q_mat = q.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
        k_mat = k.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
        qk = jnp.matmul(q_mat, k_mat.swapaxes(-1, -2))
        qk = qk.reshape(nc, nh, C, C).transpose(0, 2, 3, 1)
        intra_attn = qk * decay_mask_4d  # (nc, C, C, nh) — NO beta

        # WY L matrix: -(k_beta @ k^T) * decay_mask, strictly lower triangular
        kb_mat = k_beta.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
        kk = jnp.matmul(kb_mat, k_mat.swapaxes(-1, -2))
        kk = kk.reshape(nc, nh, C, C).transpose(0, 2, 3, 1)
        L = -kk * decay_mask_4d * strict_lower[None, :, :, None]
    else:
        # KDA: per-dim decay
        decay_cum_i = decay_cum[:, :, None, :, :]  # (nc, C, 1, nh, hd)
        decay_cum_j = decay_cum[:, None, :, :, :]  # (nc, 1, C, nh, hd)
        decay_diff_kda = decay_cum_i - decay_cum_j
        causal_5d = causal_mask[None, :, :, None, None]
        decay_diff_safe = jnp.where(causal_5d, decay_diff_kda, 0.0)
        decay_weights = jnp.exp(decay_diff_safe)  # (nc, C, C, nh, hd)

        # Intra-chunk attention (q@k with per-dim decay, NO beta)
        intra_attn = jnp.einsum('bihd,bjhd,bijhd->bijh',
                                q, k, decay_weights)
        intra_attn = intra_attn * causal_mask[None, :, :, None]

        # WY L matrix: -(k_beta . k * per-dim decay), strictly lower tri
        L = -jnp.einsum('bihd,bjhd,bijhd->bijh',
                         k_beta, k, decay_weights)
        L = L * strict_lower[None, :, :, None]

    # =====================================================================
    # Step 2: WY correction — solve (I - L) x = b
    # =====================================================================
    # (I - L) is lower triangular since L is strictly lower triangular.
    k_with_decay = k_beta * jnp.exp(decay_cum)  # (nc, C, nh, hd)

    if HAS_TRITON:
        # Fused Triton kernel: L construction + block-inverse + application in SRAM
        v_corrected, k_cumdecay = wy_correction_fused(
            k_beta, k, v_beta, decay_mask_4d, k_with_decay, L)
    else:
        # Fallback: JAX solve_triangular
        L_mat = L.transpose(0, 3, 1, 2).reshape(nc * nh, C, C)
        I_minus_L = jnp.eye(C)[None, :, :] - L_mat

        vb_mat = v_beta.transpose(0, 2, 1, 3).reshape(nc * nh, C, hvd)
        v_corrected = jax.scipy.linalg.solve_triangular(
            I_minus_L, vb_mat, lower=True)
        v_corrected = v_corrected.reshape(nc, nh, C, hvd).transpose(0, 2, 1, 3)

        kwd_mat = k_with_decay.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
        k_cumdecay = jax.scipy.linalg.solve_triangular(
            I_minus_L, kwd_mat, lower=True)
        k_cumdecay = k_cumdecay.reshape(nc, nh, C, hd).transpose(0, 2, 1, 3)

    # =====================================================================
    # Step 3: Precompute per-chunk quantities for cross-chunk scan
    # =====================================================================
    chunk_total_decay = jnp.exp(decay_cum[:, -1, :, :])  # (nc, nh, hd_or_1)

    # Raw keys decayed to end of chunk (NOT k_beta — matches recurrent path)
    decay_to_end = jnp.exp(decay_cum[:, -1:, :, :] - decay_cum)  # (nc, C, nh, hd_or_1)
    k_decayed = k * decay_to_end  # (nc, C, nh, hd)

    # --- Precompute state-independent parts (batched across all chunks) ---
    # Split: o_intra = intra_attn @ v_new = intra_attn @ v_corrected - (intra_attn @ k_cumdecay) @ S
    # Split: delta_S = k_decayed^T @ v_new = k_decayed^T @ v_corrected - (k_decayed^T @ k_cumdecay) @ S
    # This moves the large matmuls out of the sequential scan.

    # o_intra_base[c] = intra_attn[c] @ v_corrected[c]  (nc, C, nh, hvd)
    ia_mat = intra_attn.transpose(0, 3, 1, 2).reshape(nc * nh, C, C)       # (nc*nh, C, C)
    vc_mat = v_corrected.transpose(0, 2, 1, 3).reshape(nc * nh, C, hvd)    # (nc*nh, C, hvd)
    o_intra_base = jnp.matmul(ia_mat, vc_mat)                              # (nc*nh, C, hvd)
    o_intra_base = o_intra_base.reshape(nc, nh, C, hvd).transpose(0, 2, 1, 3)  # (nc, C, nh, hvd)

    # A_k[c] = intra_attn[c] @ k_cumdecay[c]  (nc, C, nh, hd)
    kc_mat = k_cumdecay.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)      # (nc*nh, C, hd)
    A_k = jnp.matmul(ia_mat, kc_mat)                                       # (nc*nh, C, hd)
    A_k = A_k.reshape(nc, nh, C, hd).transpose(0, 2, 1, 3)                 # (nc, C, nh, hd)

    # delta_S_base[c] = k_decayed[c]^T @ v_corrected[c]  (nc, nh, hd, hvd)
    kd_mat = k_decayed.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)       # (nc*nh, C, hd)
    delta_S_base = jnp.matmul(kd_mat.swapaxes(-1, -2), vc_mat)             # (nc*nh, hd, hvd)
    delta_S_base = delta_S_base.reshape(nc, nh, hd, hvd).transpose(0, 1, 3, 2)  # (nc, nh, hvd, hd)

    # B_k[c] = k_decayed[c]^T @ k_cumdecay[c]  (nc, nh, hd, hd)
    B_k = jnp.matmul(kd_mat.swapaxes(-1, -2), kc_mat)                      # (nc*nh, hd, hd)
    B_k = B_k.reshape(nc, nh, hd, hd)                                       # (nc, nh, hd, hd)

    # =====================================================================
    # Step 4: Cross-chunk scan with delta correction (precomputed split)
    # =====================================================================
    def scan_fn(S, chunk_data):
        (o_intra_base_c, A_k_c, delta_S_base_c, B_k_c,
         q_c, decay_cum_c, chunk_decay_c) = chunk_data

        # Inter-chunk output: (q * exp(decay_cum)) @ S
        decay_exp = jnp.exp(decay_cum_c)  # (C, nh, hd_or_1)
        dq = decay_exp * q_c  # (C, nh, hd)
        o_inter = jnp.einsum('nvk,cnk->cnv', S, dq)  # (C, nh, hvd)

        # Intra-chunk: precomputed base - correction from state
        # o_intra = intra_attn @ v_corrected - (intra_attn @ k_cumdecay) @ S
        intra_corr = jnp.einsum('cnk,nvk->cnv', A_k_c, S)  # (C, nh, hvd)
        o_intra = o_intra_base_c - intra_corr

        o_c = o_inter + o_intra

        # State update: precomputed base - correction from state
        # delta_S = k_decayed^T @ v_corrected - (k_decayed^T @ k_cumdecay) @ S
        state_corr = jnp.einsum('nkm,nvm->nvk', B_k_c, S)  # (nh, hvd, hd)
        delta_S = delta_S_base_c - state_corr
        S_new = S * chunk_decay_c[:, None, :] + delta_S

        return S_new, o_c

    S_init = jnp.zeros((nh, hvd, hd), dtype=jnp.float32)
    _, o_chunks = jax.lax.scan(
        scan_fn, S_init,
        (o_intra_base, A_k, delta_S_base, B_k,
         q, decay_cum, chunk_total_decay))

    return o_chunks


# ---------------------------------------------------------------------------
# Factory function (matches init_S5SSM interface)
# ---------------------------------------------------------------------------
def init_GDN_SSM(H, num_heads, head_dim=128, expand_v=2, chunk_size=64,
                 use_conv=True, use_kda=False):
    """Create a GDNSSM partial — same pattern as init_S5SSM.

    Returns:
        functools.partial[GDNSSM] with all config bound except step_rescale.
    """
    return partial(GDNSSM, H=H, num_heads=num_heads, head_dim=head_dim,
                   expand_v=expand_v, chunk_size=chunk_size,
                   use_conv=use_conv, use_kda=use_kda)
