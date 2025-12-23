from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# ============================================================================
# Full BF16 Training: Complex Number Representation
# ============================================================================
# We represent complex numbers as tuples of (real_bf16, imag_bf16) to enable
# full BF16 computation including associative_scan.
#
# JAX's complex64 = 2×float32, which doesn't use Tensor Cores.
# By using explicit (real, imag) BF16 pairs, all operations use BF16.
# ============================================================================

@jax.vmap
def binary_operator_bf16(q_i, q_j):
    """Binary operator for parallel scan using BF16 complex representation.

    Complex multiplication: (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
    Complex addition: (a + jb) + (e + jf) = (a + e) + j(b + f)

    Args:
        q_i: tuple ((A_re, A_im), (b_re, b_im)) - BF16 complex pairs
        q_j: tuple ((A_re, A_im), (b_re, b_im)) - BF16 complex pairs
    Returns:
        ((A_out_re, A_out_im), (b_out_re, b_out_im))
    """
    (A_i_re, A_i_im), (b_i_re, b_i_im) = q_i
    (A_j_re, A_j_im), (b_j_re, b_j_im) = q_j

    # A_out = A_j * A_i (complex multiplication)
    A_out_re = A_j_re * A_i_re - A_j_im * A_i_im
    A_out_im = A_j_re * A_i_im + A_j_im * A_i_re

    # b_out = A_j * b_i + b_j (complex multiply-add)
    # A_j * b_i
    Ab_re = A_j_re * b_i_re - A_j_im * b_i_im
    Ab_im = A_j_re * b_i_im + A_j_im * b_i_re
    # + b_j
    b_out_re = Ab_re + b_j_re
    b_out_im = Ab_im + b_j_im

    return (A_out_re, A_out_im), (b_out_re, b_out_im)


@jax.vmap
def binary_operator_bf16_reset(q_i, q_j):
    """Binary operator for parallel scan with reset using BF16 complex representation.

    Args:
        q_i: tuple ((A_re, A_im), (b_re, b_im), reset_flag) - BF16 complex pairs + reset
        q_j: tuple ((A_re, A_im), (b_re, b_im), reset_flag) - BF16 complex pairs + reset
    Returns:
        ((A_out_re, A_out_im), (b_out_re, b_out_im), reset_out)
    """
    (A_i_re, A_i_im), (b_i_re, b_i_im), c_i = q_i
    (A_j_re, A_j_im), (b_j_re, b_j_im), c_j = q_j

    # A_out = A_j * A_i (complex multiplication)
    A_mul_re = A_j_re * A_i_re - A_j_im * A_i_im
    A_mul_im = A_j_re * A_i_im + A_j_im * A_i_re

    # b_out = A_j * b_i + b_j (complex multiply-add)
    Ab_re = A_j_re * b_i_re - A_j_im * b_i_im
    Ab_im = A_j_re * b_i_im + A_j_im * b_i_re
    b_add_re = Ab_re + b_j_re
    b_add_im = Ab_im + b_j_im

    # Apply reset logic: if c_j, use A_j and b_j directly
    one_minus_cj = 1 - c_j
    A_out_re = A_mul_re * one_minus_cj + A_j_re * c_j
    A_out_im = A_mul_im * one_minus_cj + A_j_im * c_j
    b_out_re = b_add_re * one_minus_cj + b_j_re * c_j
    b_out_im = b_add_im * one_minus_cj + b_j_im * c_j
    c_out = c_i * one_minus_cj + c_j

    return (A_out_re, A_out_im), (b_out_re, b_out_im), c_out


# Legacy operators for backward compatibility (used by es_lobs5)
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


@jax.vmap
def binary_operator_reset(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i)*(1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j)*(1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )


# ============================================================================
# DEPRECATED: Full BF16 Helpers (not used with FP32 scan)
# ============================================================================
# These helpers were for Full BF16 scan (all BF16 throughout pipeline).
# DEPRECATED Reason: Numerical explosion in associative_scan over 12000 timesteps
#   - BF16 precision: 7 bits mantissa → cumulative error (1+ε)^14 ≈ 1.15
#   - Observed: xs_re ∈ [-3e8, 3e8] (vs expected O(1-100))
#   - Result: Precision loss → NaN in downstream layers
#   - Logs: bf16_debug_1673380.out, bf16_debug_1673381.out
#   - Mathematical proof: bf16_numerical_analysis.md
# ============================================================================

# def complex_to_bf16_pair(z_complex):
#     """Convert complex64 array to BF16 (real, imag) pair.
#     DEPRECATED: Used for Full BF16 scan, causes numerical explosion.
#     """
#     return z_complex.real.astype(np.bfloat16), z_complex.imag.astype(np.bfloat16)  # DEPRECATED


# def bf16_pair_to_complex(real_bf, imag_bf):
#     """Convert BF16 (real, imag) pair back to complex64.
#     DEPRECATED: Used for Full BF16 scan, causes numerical explosion.
#     """
#     return real_bf.astype(np.float32) + 1j * imag_bf.astype(np.float32)  # DEPRECATED


# def complex_matvec_bf16_real_x_bf16_out(A_complex, x_bf16):
#     """Compute y = A_complex @ x_real, returning BF16 pair.
#     DEPRECATED: Returns BF16 pair for Full BF16 scan.
#     Reason: BF16 scan numerically unstable, use complex_matvec_bf16_real_x (returns FP32) instead.
#     """
#     # [DEBUG BF16] Check inputs - DEPRECATED, kept for reference
#     # jax.debug.print("[complex_matvec_bf16_real_x_bf16_out] A_complex has NaN: {}, shape: {}", np.any(np.isnan(A_complex)), A_complex.shape)  # DEBUG BF16
#     # jax.debug.print("[complex_matvec_bf16_real_x_bf16_out] x_bf16 has NaN: {}, shape: {}, dtype: {}", np.any(np.isnan(x_bf16)), x_bf16.shape, x_bf16.dtype)  # DEBUG BF16
#
#     A_re_bf, A_im_bf = complex_to_bf16_pair(A_complex)  # DEPRECATED
#
#     # jax.debug.print("[complex_matvec_bf16_real_x_bf16_out] A_re_bf has NaN: {}, range: [{}, {}]", np.any(np.isnan(A_re_bf)), np.min(A_re_bf), np.max(A_re_bf))  # DEBUG BF16
#     # jax.debug.print("[complex_matvec_bf16_real_x_bf16_out] A_im_bf has NaN: {}, range: [{}, {}]", np.any(np.isnan(A_im_bf)), np.min(A_im_bf), np.max(A_im_bf))  # DEBUG BF16
#
#     # 2x BF16 matmuls (Tensor Core accelerated)
#     real_bf = np.matmul(A_re_bf, x_bf16)  # DEPRECATED
#     imag_bf = np.matmul(A_im_bf, x_bf16)  # DEPRECATED
#
#     # jax.debug.print("[complex_matvec_bf16_real_x_bf16_out] real_bf has NaN: {}, range: [{}, {}]", np.any(np.isnan(real_bf)), np.min(real_bf), np.max(real_bf))  # DEBUG BF16
#     # jax.debug.print("[complex_matvec_bf16_real_x_bf16_out] imag_bf has NaN: {}, range: [{}, {}]", np.any(np.isnan(imag_bf)), np.min(imag_bf), np.max(imag_bf))  # DEBUG BF16
#
#     return real_bf, imag_bf  # DEPRECATED


# def complex_matvec_bf16_bf16_out(A_complex, x_re_bf, x_im_bf):
#     """Compute y = A_complex @ x_complex, all in BF16.
#     DEPRECATED: All BF16 computation for Full BF16 scan.
#     Reason: BF16 scan numerically unstable, use complex_matvec_bf16 (returns FP32) instead.
#     """
#     A_re_bf, A_im_bf = complex_to_bf16_pair(A_complex)  # DEPRECATED
#
#     # 4x BF16 matmuls (Tensor Core accelerated)
#     rr = np.matmul(A_re_bf, x_re_bf)  # DEPRECATED
#     ii = np.matmul(A_im_bf, x_im_bf)  # DEPRECATED
#     ri = np.matmul(A_re_bf, x_im_bf)  # DEPRECATED
#     ir = np.matmul(A_im_bf, x_re_bf)  # DEPRECATED
#
#     # Combine results (still BF16)
#     real_bf = rr - ii  # DEPRECATED
#     imag_bf = ri + ir  # DEPRECATED
#
#     return real_bf, imag_bf  # DEPRECATED


# Legacy functions for backward compatibility (used by es_lobs5)
def _to_bf16_real_imag(A_complex):
    """Convert complex array to BF16 real and imaginary parts."""
    return A_complex.real.astype(np.bfloat16), A_complex.imag.astype(np.bfloat16)


def complex_matvec_bf16_real_x(A_complex, x_real):
    """Compute y = A_complex @ x_real using BF16 matvec kernels.
    Legacy function - returns complex64 for backward compatibility.
    """
    A_re_bf, A_im_bf = _to_bf16_real_imag(A_complex)
    x_bf = x_real.astype(np.bfloat16)
    real_bf = np.matmul(A_re_bf, x_bf)
    imag_bf = np.matmul(A_im_bf, x_bf)
    return real_bf.astype(np.float32) + 1j * imag_bf.astype(np.float32)


def complex_matvec_bf16(A_complex, x_complex):
    """Compute y = A_complex @ x_complex using BF16 matvec kernels.
    Legacy function - returns complex64 for backward compatibility.
    """
    A_re_bf, A_im_bf = _to_bf16_real_imag(A_complex)
    x_re_bf = x_complex.real.astype(np.bfloat16)
    x_im_bf = x_complex.imag.astype(np.bfloat16)
    rr = np.matmul(A_re_bf, x_re_bf)
    ii = np.matmul(A_im_bf, x_im_bf)
    ri = np.matmul(A_re_bf, x_im_bf)
    ir = np.matmul(A_im_bf, x_re_bf)
    real_bf = rr - ii
    imag_bf = ri + ir
    return real_bf.astype(np.float32) + 1j * imag_bf.astype(np.float32)


# ============================================================================
# DEPRECATED: Full BF16 SSM (including scan) - CAUSES NUMERICAL EXPLOSION
# ============================================================================
# def apply_ssm_full_bf16(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
#     """
#     DEPRECATED: Full BF16 implementation causes numerical explosion.
#
#     Problem: BF16 associative_scan accumulates errors over 12000 timesteps:
#       - BF16 precision: 7 bits mantissa (vs FP32 23 bits)
#       - Recursive depth: log2(12000) = 14 layers
#       - Observed growth: α=3.3 per layer → 3.3^13 ≈ 10^7× explosion
#       - Result: xs_re ∈ [-3e8, 3e8] → precision loss → NaN
#
#     Mathematical proof:
#       von Neumann stability: ρ(Λ) + k·ε < 1
#       FP32: 0.99 + 14×10⁻⁷ ≈ 0.99 ✅ stable
#       BF16: 0.99 + 14×0.008 ≈ 1.11 ❌ unstable
#
#     Logs: bf16_debug_1673380.out, bf16_debug_1673381.out
#     See: bf16_numerical_analysis.md for full derivation
#     """
#     # [Full BF16 implementation commented out - see git history b180c1d]
#     pass

# ============================================================================
# Correct BF16 SSM: BF16 matmul + FP32 scan (Reference: 7DEC working implementation)
# ============================================================================

def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """Compute the LxH output of discretized SSM given an LxH input.

    BF16 strategy: Use BF16 for matmul (Tensor Core), FP32 for scan (stability)

    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        input_sequence (float32/bfloat16): input sequence            (L, H)
        conj_sym (bool):         whether conjugate symmetry is enforced
        bidirectional (bool):    whether bidirectional setup is used
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    # [DEBUG BF16] Check inputs for NaN
    # jax.debug.print("[apply_ssm] Lambda_bar has NaN: {}, shape: {}", np.any(np.isnan(Lambda_bar)), Lambda_bar.shape)  # DEBUG BF16
    # jax.debug.print("[apply_ssm] B_bar has NaN: {}, shape: {}", np.any(np.isnan(B_bar)), B_bar.shape)  # DEBUG BF16
    # jax.debug.print("[apply_ssm] C_tilde has NaN: {}, shape: {}", np.any(np.isnan(C_tilde)), C_tilde.shape)  # DEBUG BF16
    # jax.debug.print("[apply_ssm] input_sequence has NaN: {}, shape: {}, dtype: {}", np.any(np.isnan(input_sequence)), input_sequence.shape, input_sequence.dtype)  # DEBUG BF16

    # Ensure input is FP32 for scan stability (Reference: 7DEC:354)
    input_fp32 = input_sequence.astype(np.float32)

    # Broadcast Lambda_bar to (L, P) - required by @jax.vmap'd binary_operator
    # Note: This creates 48.8 MB of duplicated data, but removing it would require
    # rewriting binary_operator without vmap (complex refactor)
    Lambda_elements = Lambda_bar * np.ones((input_fp32.shape[0], Lambda_bar.shape[0]))

    # BF16 batched matmul: B_bar @ u (optimized: single matmul instead of vmap)
    # B_bar: (P, H) complex → input: (L, H) real → Bu: (L, P) complex
    input_T = input_fp32.T.astype(np.bfloat16)  # (H, L) BF16
    B_re = B_bar.real.astype(np.bfloat16)  # (P, H) BF16
    B_im = B_bar.imag.astype(np.bfloat16)  # (P, H) BF16

    # Single batched matmul (Tensor Core!) - replaces 12000 vmap calls
    Bu_re = np.matmul(B_re, input_T)  # (P, L) BF16
    Bu_im = np.matmul(B_im, input_T)  # (P, L) BF16

    # Cast to FP32 for scan stability and transpose to (L, P)
    Bu_elements = (Bu_re.astype(np.float32) + 1j * Bu_im.astype(np.float32)).T
    # jax.debug.print("[apply_ssm] Bu_elements has NaN: {}, dtype: {}", np.any(np.isnan(Bu_elements)), Bu_elements.dtype)  # DEBUG BF16

    # FP32 scan: binary_operator works on complex64 for numerical stability
    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    # jax.debug.print("[apply_ssm] After FP32 scan - xs has NaN: {}, dtype: {}, range: [{}, {}]", np.any(np.isnan(xs)), xs.dtype, np.min(np.abs(xs)), np.max(np.abs(xs)))  # DEBUG BF16

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)

    # BF16 batched matmul: C_tilde @ xs (optimized: single matmul instead of vmap)
    # C_tilde: (H, P) complex → xs: (L, P) complex → ys: (L, H) real
    xs_T = xs.T  # (P, L) complex64
    C_re = C_tilde.real.astype(np.bfloat16)  # (H, P) BF16
    C_im = C_tilde.imag.astype(np.bfloat16)  # (H, P) BF16
    xs_re = xs_T.real.astype(np.bfloat16)  # (P, L) BF16
    xs_im = xs_T.imag.astype(np.bfloat16)  # (P, L) BF16

    # Complex matmul: (C_re + i*C_im) @ (xs_re + i*xs_im) → take real part
    # = C_re @ xs_re - C_im @ xs_im (real part)
    ys_re = np.matmul(C_re, xs_re) - np.matmul(C_im, xs_im)  # (H, L) BF16

    # Transpose and apply conjugate symmetry
    ys = (2 * ys_re if conj_sym else ys_re).T.astype(np.float32)  # (L, H) FP32

    # jax.debug.print("[apply_ssm] Final ys has NaN: {}, dtype: {}, range: [{}, {}]", np.any(np.isnan(ys)), ys.dtype, np.min(ys), np.max(ys))  # DEBUG BF16

    return ys

def apply_ssm_rnn(Lambda_bar, B_bar, C_tilde, hidden, input_sequence, resets, conj_sym, bidirectional):
    """Compute the LxH output of discretized SSM in RNN mode.

    BF16 strategy: Use BF16 for matmul (Tensor Core), FP32 for scan (stability)

    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        hidden: hidden state - complex64 (1, P)
        input_sequence (float32/bfloat16): input sequence            (L, H)
        resets (bool): reset signals                                 (L,)
        conj_sym (bool): whether conjugate symmetry is enforced
        bidirectional (bool): whether bidirectional (not supported in RNN mode)
    Returns:
        hidden_out: complex64 (1, P) for next call
        ys (float32): the SSM outputs                                (L, H)
    """
    # Ensure input is FP32 for scan stability (Reference: 7DEC:167-170)
    input_fp32 = input_sequence.astype(np.float32)

    # Broadcast Lambda_bar, keep as complex64 (FP32) for scan
    Lambda_elements = Lambda_bar * np.ones((input_fp32.shape[0], Lambda_bar.shape[0]))

    # BF16 matmul: B_bar @ u, returns complex64 (FP32) for scan
    Bu_elements = jax.vmap(lambda u: complex_matvec_bf16_real_x(B_bar, u))(input_fp32)

    # Prepend hidden state (complex64)
    Lambda_elements = np.concatenate([
        np.ones((1, Lambda_bar.shape[0])),
        Lambda_elements,
    ])

    Bu_elements = np.concatenate([
        hidden,
        Bu_elements,
    ])

    # FP32 scan: binary_operator works on complex64 for numerical stability
    if resets is None:
        _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    else:
        resets = np.concatenate([
            np.zeros(1, dtype=resets.dtype),
            resets,
        ])
        _, xs, _ = jax.lax.associative_scan(binary_operator_reset, (Lambda_elements, Bu_elements, resets))

    # Extract hidden state (complex64) for next call
    hidden_out = xs[np.newaxis, -1]
    xs = xs[1:]

    if bidirectional:
        raise ValueError("Cannot expect a bidirectional view if doing rnn")

    # BF16 matmul: C_tilde @ x, returns complex64 (FP32) (Reference: 7DEC:200-201)
    if conj_sym:
        return hidden_out, jax.vmap(lambda x: 2 * complex_matvec_bf16(C_tilde, x).real)(xs)
    else:
        return hidden_out, jax.vmap(lambda x: complex_matvec_bf16(C_tilde, x).real)(xs)


class S5SSM(nn.Module):
    Lambda_re_init: jax.Array
    Lambda_im_init: jax.Array
    V: jax.Array
    Vinv: jax.Array
    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative.
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param("C1",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)
                self.C2 = self.param("C2",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param("C",
                                    lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                    C_shape)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # [DEBUG BF16] Check discretization inputs
        # jax.debug.print("[S5SSM.setup] log_step range: [{}, {}]", np.min(self.log_step), np.max(self.log_step))  # DEBUG BF16
        # jax.debug.print("[S5SSM.setup] step = exp(log_step) range: [{}, {}]", np.min(step), np.max(step))  # DEBUG BF16
        # jax.debug.print("[S5SSM.setup] Lambda range: |Lambda| ∈ [{}, {}]", np.min(np.abs(self.Lambda)), np.max(np.abs(self.Lambda)))  # DEBUG BF16

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

        # [DEBUG BF16] Check discretization outputs
        # jax.debug.print("[S5SSM.setup] Lambda_bar range: |Lambda_bar| ∈ [{}, {}]", np.min(np.abs(self.Lambda_bar)), np.max(np.abs(self.Lambda_bar)))  # DEBUG BF16
        # jax.debug.print("[S5SSM.setup] B_bar range: |B_bar| ∈ [{}, {}]", np.min(np.abs(self.B_bar)), np.max(np.abs(self.B_bar)))  # DEBUG BF16

        # [DEBUG BF16] Check for NaN after discretization in setup()
        # jax.debug.print("[S5SSM.setup] After discretization - Lambda_bar has NaN: {}", np.any(np.isnan(self.Lambda_bar)))  # DEBUG BF16
        # jax.debug.print("[S5SSM.setup] After discretization - B_bar has NaN: {}", np.any(np.isnan(self.B_bar)))  # DEBUG BF16
        # jax.debug.print("[S5SSM.setup] After discretization - C_tilde has NaN: {}", np.any(np.isnan(self.C_tilde)))  # DEBUG BF16
        # jax.debug.print("[S5SSM.setup] After discretization - D has NaN: {}", np.any(np.isnan(self.D)))  # DEBUG BF16

    def __call__(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.

        BF16 strategy: BF16 matmul + FP32 scan (Reference: 7DEC:343-367)

        Args:
             input_sequence (float32/bfloat16): input sequence (L, H)
        Returns:
            output sequence (float32/bfloat16): (L, H)
        """
        # [DEBUG BF16] Check discretized params in setup()
        # jax.debug.print("[S5SSM.__call__] self.Lambda_bar has NaN: {}", np.any(np.isnan(self.Lambda_bar)))  # DEBUG BF16
        # jax.debug.print("[S5SSM.__call__] self.B_bar has NaN: {}", np.any(np.isnan(self.B_bar)))  # DEBUG BF16
        # jax.debug.print("[S5SSM.__call__] self.C_tilde has NaN: {}", np.any(np.isnan(self.C_tilde)))  # DEBUG BF16
        # jax.debug.print("[S5SSM.__call__] self.D has NaN: {}", np.any(np.isnan(self.D)))  # DEBUG BF16
        # jax.debug.print("[S5SSM.__call__] input_sequence dtype: {}, has NaN: {}", input_sequence.dtype, np.any(np.isnan(input_sequence)))  # DEBUG BF16

        # Save input dtype and cast to FP32 for SSM (apply_ssm returns FP32)
        input_dtype = input_sequence.dtype
        input_fp32 = input_sequence.astype(np.float32)

        # apply_ssm: BF16 matmul + FP32 scan, returns FP32
        ys = apply_ssm(self.Lambda_bar,
                       self.B_bar,
                       self.C_tilde,
                       input_fp32,
                       self.conj_sym,
                       self.bidirectional)

        # jax.debug.print("[S5SSM.__call__] ys from apply_ssm has NaN: {}, dtype: {}", np.any(np.isnan(ys)), ys.dtype)  # DEBUG BF16

        # D feedthrough in FP32
        Du = jax.vmap(lambda u: self.D * u)(input_fp32)
        # jax.debug.print("[S5SSM.__call__] Du has NaN: {}", np.any(np.isnan(Du)))  # DEBUG BF16

        output = ys + Du
        # jax.debug.print("[S5SSM.__call__] Final output has NaN: {}, range: [{}, {}]", np.any(np.isnan(output)), np.min(output), np.max(output))  # DEBUG BF16

        # Cast output back to input dtype (BF16 if needed)
        return output.astype(input_dtype)

    def __call_rnn__(self, hidden, input_sequence, resets):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using RNN-style forward pass.

        BF16 strategy: BF16 matmul + FP32 scan (Reference: 7DEC:369-391)

        Args:
             hidden: complex64 (1, P) hidden state
             input_sequence (float32/bfloat16): input sequence (L, H)
             resets (bool): reset signals (L,)
        Returns:
            hidden_out: complex64 (1, P) for next call
            output (float32): (L, H)
        """
        # Cast to FP32 for SSM (apply_ssm_rnn returns FP32)
        input_fp32 = input_sequence.astype(np.float32)

        # apply_ssm_rnn: BF16 matmul + FP32 scan, returns complex64 hidden + FP32 output
        hidden_out, ys = apply_ssm_rnn(self.Lambda_bar,
                                        self.B_bar,
                                        self.C_tilde,
                                        hidden,
                                        input_fp32,
                                        None,
                                        self.conj_sym,
                                        self.bidirectional)

        # D feedthrough in FP32
        Du = jax.vmap(lambda u: self.D * u)(input_fp32)
        output = ys + Du

        return hidden_out, output



def init_S5SSM(H,
               P,
               Lambda_re_init,
               Lambda_im_init,
               V,
               Vinv,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM,
                   H=H,
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
                   bidirectional=bidirectional)
