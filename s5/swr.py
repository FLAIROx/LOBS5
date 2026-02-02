"""Sliding Window Recurrences (SWR) for S5 SSM.

This module implements the Block Two-Pass (B2P) algorithm from the paper:
"Sliding Window Recurrences for Sequence Models" (arXiv:2512.13921)

The algorithm replaces the standard associative_scan with a two-pass approach:
- Pass I: Local recurrence using matrix multiplication (utilizes Tensor Cores)
- Pass II: Neighbor update using parallel broadcast (rank-1 updates)

This achieves higher Arithmetic Intensity (FLOPs/byte) compared to standard scan.
"""

import jax
import jax.numpy as np
from functools import partial


def build_transfer_matrix(Lambda_bar, window_size):
    """Build the local transfer matrix M for a given window size.

    The transfer matrix M is a lower triangular matrix where:
        M[i, j, p] = Lambda_bar[p]^{i-j}  if i >= j
                   = 0                     if i < j

    This matrix represents the recurrence relation within a window:
        x[i] = sum_{j=0}^{i} Lambda^{i-j} * Bu[j]
             = M @ Bu  (matrix multiplication)

    Args:
        Lambda_bar: (P,) complex64, discretized diagonal state matrix
        window_size: int, the window size W

    Returns:
        M: (W, W, P) complex64, transfer matrix
    """
    P = Lambda_bar.shape[0]
    W = window_size

    # Compute powers of Lambda_bar
    # powers[k, p] = Lambda_bar[p]^k for k in [0, W-1]
    # We need powers from 0 to W-1
    powers = np.zeros((W, P), dtype=Lambda_bar.dtype)
    powers = powers.at[0].set(np.ones(P, dtype=Lambda_bar.dtype))  # Lambda^0 = 1

    # Iteratively compute powers: powers[k] = powers[k-1] * Lambda_bar
    def compute_power(carry, k):
        prev_power = carry
        new_power = prev_power * Lambda_bar
        return new_power, new_power

    _, powers_rest = jax.lax.scan(compute_power, powers[0], np.arange(1, W))
    powers = np.concatenate([powers[0:1], powers_rest], axis=0)  # (W, P)

    # Build the lower triangular matrix
    # M[i, j, p] = powers[i-j, p] if i >= j, else 0
    # Using broadcasting: diff = i - j for all (i, j) pairs
    i_idx = np.arange(W)[:, None]  # (W, 1)
    j_idx = np.arange(W)[None, :]  # (1, W)
    diff = i_idx - j_idx           # (W, W)

    # Clamp negative indices to 0 (will be masked out anyway)
    diff_clamped = np.maximum(diff, 0)  # (W, W)

    # Gather powers: M[i, j, :] = powers[diff[i, j], :]
    M = powers[diff_clamped]  # (W, W, P)

    # Apply lower triangular mask
    mask = (i_idx >= j_idx).astype(np.float32)  # (W, W)
    M = M * mask[:, :, None]  # (W, W, P)

    return M


def apply_ssm_swr_pass1(M, Bu_windows):
    """Pass I: Parallel local recurrence using matrix multiplication.

    For each window k, compute:
        x_k = M @ Bu_k  (matrix-vector product along window dimension)

    This is equivalent to running the recurrence within each window,
    but expressed as a matrix multiplication for better GPU utilization.

    Args:
        M: (W, W, P) complex64, transfer matrix
        Bu_windows: (N, W, P) complex64, input split into N windows of size W

    Returns:
        xs_local: (N, W, P) complex64, local recurrence results
        carriers: (N, P) complex64, final state of each window (for Pass II)
    """
    N, W, P = Bu_windows.shape

    # For complex matrices, we need to handle real and imaginary parts
    # M @ Bu for each window and each P (state dimension)
    #
    # Approach: Use einsum for batched matrix-vector product
    # M has shape (W, W, P) - for each p, M[:,:,p] is W x W
    # Bu_windows has shape (N, W, P) - for each n and p, Bu_windows[n,:,p] is W
    #
    # We want: xs_local[n, i, p] = sum_j M[i, j, p] * Bu_windows[n, j, p]
    # This is: einsum('ijp,njp->nip', M, Bu_windows)

    xs_local = np.einsum('ijp,njp->nip', M, Bu_windows)  # (N, W, P)

    # Extract carriers (final state of each window)
    carriers = xs_local[:, -1, :]  # (N, P)

    return xs_local, carriers


def apply_ssm_swr_pass2(xs_local, carriers, Lambda_bar, window_size):
    """Pass II: Parallel neighbor update using broadcast.

    For each window k >= 1, update using the carrier from window k-1:
        x_k[i] += carrier_{k-1} * Lambda_bar^{W-i}

    This propagates information from previous windows.

    Args:
        xs_local: (N, W, P) complex64, local recurrence results from Pass I
        carriers: (N, P) complex64, final states from Pass I
        Lambda_bar: (P,) complex64, discretized diagonal state matrix
        window_size: int, the window size W

    Returns:
        xs: (N, W, P) complex64, complete recurrence results
    """
    N, W, P = xs_local.shape

    # Build decay vector: decay[i] = Lambda_bar^{W-i} for i in [0, W-1]
    # decay[0] = Lambda_bar^W, decay[W-1] = Lambda_bar^1
    decay_powers = np.arange(W, 0, -1)  # [W, W-1, ..., 1]

    # Compute Lambda_bar^k for each power
    # Using the formula: Lambda_bar^k = exp(k * log(Lambda_bar))
    # But for complex numbers, we should compute iteratively for stability

    # Alternative: precompute powers similar to transfer matrix
    # decay_vec[i, p] = Lambda_bar[p]^{decay_powers[i]}
    #                 = Lambda_bar[p]^{W-i}

    # For efficiency, we can compute from the transfer matrix's last column
    # M[:, 0, :] = [Lambda^0, Lambda^1, ..., Lambda^{W-1}]
    # We need [Lambda^W, Lambda^{W-1}, ..., Lambda^1] = reversed * Lambda

    # Simple approach: compute powers directly
    def compute_decay_vec(Lambda_bar, W):
        # Compute Lambda_bar^k for k = 1, 2, ..., W
        powers = [Lambda_bar]  # Lambda^1
        for _ in range(W - 1):
            powers.append(powers[-1] * Lambda_bar)
        # powers = [Lambda^1, Lambda^2, ..., Lambda^W]
        powers = np.stack(powers, axis=0)  # (W, P)
        # Reverse to get [Lambda^W, Lambda^{W-1}, ..., Lambda^1]
        return powers[::-1]  # (W, P)

    # JIT-friendly version using scan
    def compute_power_scan(carry, _):
        return carry * Lambda_bar, carry * Lambda_bar

    init_power = Lambda_bar  # Lambda^1
    _, powers = jax.lax.scan(compute_power_scan, init_power, None, length=W-1)
    powers = np.concatenate([init_power[None, :], powers], axis=0)  # (W, P), [Lambda^1, ..., Lambda^W]
    decay_vec = powers[::-1]  # (W, P), [Lambda^W, ..., Lambda^1]

    # Shift carriers: carriers_shifted[k] = carriers[k-1] for k >= 1
    #                                     = 0             for k = 0
    carriers_shifted = np.concatenate([
        np.zeros((1, P), dtype=carriers.dtype),
        carriers[:-1]
    ], axis=0)  # (N, P)

    # Compute update: update[n, i, p] = carriers_shifted[n, p] * decay_vec[i, p]
    update = carriers_shifted[:, None, :] * decay_vec[None, :, :]  # (N, W, P)

    # Apply update
    xs = xs_local + update

    return xs


def apply_ssm_swr(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional,
                  window_size=16, hidden_in=None, return_hidden=False):
    """Compute SSM output using Sliding Window Recurrences (SWR).

    This is a drop-in replacement for apply_ssm() that uses the Block Two-Pass
    algorithm for higher Arithmetic Intensity.

    BF16 strategy (matching apply_ssm):
    - B @ u: BF16 matmul (Tensor Core)
    - Pass I/II: FP32 (complex stability)
    - C @ x: BF16 matmul (Tensor Core)

    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        input_sequence (float32/bfloat16): input sequence            (L, H)
        conj_sym (bool):         whether conjugate symmetry is enforced
        bidirectional (bool):    whether bidirectional setup is used
        window_size (int):       SWR window size W (default 16)
        hidden_in (complex64):   optional initial hidden state       (1, P)
        return_hidden (bool):    whether to return final hidden state

    Returns:
        If return_hidden=False:
            ys (float32): the SSM outputs (S5 layer preactivations)  (L, H)
        If return_hidden=True:
            (hidden_out, ys): hidden_out is (1, P) complex64
    """
    L, H = input_sequence.shape
    P = Lambda_bar.shape[0]

    # Ensure input is FP32 for stability
    input_fp32 = input_sequence.astype(np.float32)

    # ========================================================================
    # Step 1: Compute B @ u (BF16 matmul for Tensor Core)
    # ========================================================================
    input_T = input_fp32.T.astype(np.bfloat16)  # (H, L) BF16
    B_re = B_bar.real.astype(np.bfloat16)  # (P, H) BF16
    B_im = B_bar.imag.astype(np.bfloat16)  # (P, H) BF16

    Bu_re = np.matmul(B_re, input_T)  # (P, L) BF16
    Bu_im = np.matmul(B_im, input_T)  # (P, L) BF16

    # Cast to FP32 complex for scan stability and transpose to (L, P)
    Bu_elements = (Bu_re.astype(np.float32) + 1j * Bu_im.astype(np.float32)).T  # (L, P)

    # ========================================================================
    # Step 2: Handle sequence length padding for window alignment
    # ========================================================================
    W = window_size

    # Compute number of windows and required padding
    N = (L + W - 1) // W  # Number of windows (ceiling division)
    L_padded = N * W
    pad_length = L_padded - L

    if pad_length > 0:
        # Pad Bu_elements with zeros
        Bu_elements = np.concatenate([
            Bu_elements,
            np.zeros((pad_length, P), dtype=Bu_elements.dtype)
        ], axis=0)  # (L_padded, P)

    # Reshape into windows
    Bu_windows = Bu_elements.reshape(N, W, P)  # (N, W, P)

    # ========================================================================
    # Step 3: Build transfer matrix (can be cached for repeated calls)
    # ========================================================================
    M = build_transfer_matrix(Lambda_bar, W)  # (W, W, P)

    # ========================================================================
    # Step 4: Pass I - Local recurrence (parallel matmul)
    # ========================================================================
    xs_local, carriers = apply_ssm_swr_pass1(M, Bu_windows)  # (N, W, P), (N, P)

    # ========================================================================
    # Step 5: Handle hidden_in (TBPTT support)
    # ========================================================================
    if hidden_in is not None:
        # hidden_in acts as the carrier from "window -1"
        # Prepend it to carriers_shifted in Pass II
        #
        # Modification: carriers_shifted[0] = hidden_in instead of zeros
        # We handle this by modifying xs_local for window 0

        # Compute decay vector for hidden_in contribution
        decay_powers = np.arange(W, 0, -1)  # [W, W-1, ..., 1]

        def compute_power_scan(carry, _):
            return carry * Lambda_bar, carry * Lambda_bar

        init_power = Lambda_bar
        _, powers = jax.lax.scan(compute_power_scan, init_power, None, length=W-1)
        powers = np.concatenate([init_power[None, :], powers], axis=0)
        decay_vec = powers[::-1]  # (W, P)

        # Update window 0 with hidden_in contribution
        hidden_in_squeezed = hidden_in.squeeze(0)  # (P,)
        hidden_contribution = hidden_in_squeezed[None, :] * decay_vec  # (W, P)
        xs_local = xs_local.at[0].add(hidden_contribution)

        # Update carrier for window 0
        carriers = carriers.at[0].add(hidden_in_squeezed * Lambda_bar ** W)

    # ========================================================================
    # Step 6: Pass II - Neighbor update (parallel broadcast)
    # ========================================================================
    xs = apply_ssm_swr_pass2(xs_local, carriers, Lambda_bar, W)  # (N, W, P)

    # ========================================================================
    # Step 7: Reshape back and remove padding
    # ========================================================================
    xs = xs.reshape(L_padded, P)  # (L_padded, P)

    if pad_length > 0:
        xs = xs[:L]  # (L, P) - remove padding

    # Extract hidden_out if needed
    hidden_out = None
    if return_hidden:
        hidden_out = xs[np.newaxis, -1]  # (1, P)

    # ========================================================================
    # Step 8: Handle bidirectional (if enabled)
    # ========================================================================
    if bidirectional:
        # For bidirectional, we need a backward pass
        # This is a simplified implementation - reverse the sequence and apply SWR again
        Bu_elements_orig = (Bu_re.astype(np.float32) + 1j * Bu_im.astype(np.float32)).T[:L]
        Bu_reversed = Bu_elements_orig[::-1]

        if pad_length > 0:
            Bu_reversed = np.concatenate([
                Bu_reversed,
                np.zeros((pad_length, P), dtype=Bu_reversed.dtype)
            ], axis=0)

        Bu_windows_rev = Bu_reversed.reshape(N, W, P)
        xs_local_rev, carriers_rev = apply_ssm_swr_pass1(M, Bu_windows_rev)
        xs_rev = apply_ssm_swr_pass2(xs_local_rev, carriers_rev, Lambda_bar, W)
        xs_rev = xs_rev.reshape(L_padded, P)

        if pad_length > 0:
            xs_rev = xs_rev[:L]

        xs_rev = xs_rev[::-1]  # Reverse back
        xs = np.concatenate((xs, xs_rev), axis=-1)  # (L, 2P)

    # ========================================================================
    # Step 9: Compute C @ x (BF16 matmul for Tensor Core)
    # ========================================================================
    xs_T = xs.T  # (P, L) or (2P, L) for bidirectional
    C_re = C_tilde.real.astype(np.bfloat16)  # (H, P) or (H, 2P)
    C_im = C_tilde.imag.astype(np.bfloat16)
    xs_re = xs_T.real.astype(np.bfloat16)
    xs_im = xs_T.imag.astype(np.bfloat16)

    # Complex matmul: (C_re + i*C_im) @ (xs_re + i*xs_im) -> take real part
    ys_re = np.matmul(C_re, xs_re) - np.matmul(C_im, xs_im)  # (H, L)

    # Transpose and apply conjugate symmetry
    ys = (2 * ys_re if conj_sym else ys_re).T.astype(np.float32)  # (L, H)

    if return_hidden:
        return hidden_out, ys
    return ys


# ============================================================================
# Optimized version with cached transfer matrix
# ============================================================================

@partial(jax.jit, static_argnums=(4, 5, 6, 8))
def apply_ssm_swr_jit(Lambda_bar, B_bar, C_tilde, input_sequence,
                      conj_sym, bidirectional, window_size,
                      hidden_in, return_hidden):
    """JIT-compiled version of apply_ssm_swr.

    Static arguments: conj_sym, bidirectional, window_size, return_hidden
    """
    return apply_ssm_swr(Lambda_bar, B_bar, C_tilde, input_sequence,
                         conj_sym, bidirectional, window_size,
                         hidden_in, return_hidden)
