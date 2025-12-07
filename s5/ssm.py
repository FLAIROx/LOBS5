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
# Full BF16 Helpers for Complex Matrix-Vector Operations
# ============================================================================
# All operations stay in BF16 throughout the computation pipeline.
# Only cast to FP32 at the very end for loss computation.
# ============================================================================

def complex_to_bf16_pair(z_complex):
    """Convert complex64 array to BF16 (real, imag) pair.

    Args:
        z_complex: complex64 array
    Returns:
        (real_bf16, imag_bf16): tuple of bfloat16 arrays
    """
    return z_complex.real.astype(np.bfloat16), z_complex.imag.astype(np.bfloat16)


def bf16_pair_to_complex(real_bf, imag_bf):
    """Convert BF16 (real, imag) pair back to complex64.

    Args:
        real_bf: bfloat16 real part
        imag_bf: bfloat16 imaginary part
    Returns:
        complex64 array
    """
    return real_bf.astype(np.float32) + 1j * imag_bf.astype(np.float32)


def complex_matvec_bf16_real_x_bf16_out(A_complex, x_bf16):
    """Compute y = A_complex @ x_real, returning BF16 pair.

    For complex matrix × real vector:
        y_real = A_real @ x
        y_imag = A_imag @ x

    Args:
        A_complex: complex64 matrix (P, H)
        x_bf16: bfloat16 vector (H,)
    Returns:
        (y_real_bf16, y_imag_bf16): tuple of bfloat16 arrays (P,)
    """
    A_re_bf, A_im_bf = complex_to_bf16_pair(A_complex)

    # 2x BF16 matmuls (Tensor Core accelerated)
    real_bf = np.matmul(A_re_bf, x_bf16)
    imag_bf = np.matmul(A_im_bf, x_bf16)

    return real_bf, imag_bf


def complex_matvec_bf16_bf16_out(A_complex, x_re_bf, x_im_bf):
    """Compute y = A_complex @ x_complex, all in BF16.

    Complex multiplication: (a + jb)(c + jd) = (ac - bd) + j(ad + bc)

    Args:
        A_complex: complex64 matrix (H, P)
        x_re_bf: bfloat16 real part (P,)
        x_im_bf: bfloat16 imaginary part (P,)
    Returns:
        (y_real_bf16, y_imag_bf16): tuple of bfloat16 arrays (H,)
    """
    A_re_bf, A_im_bf = complex_to_bf16_pair(A_complex)

    # 4x BF16 matmuls (Tensor Core accelerated)
    rr = np.matmul(A_re_bf, x_re_bf)
    ii = np.matmul(A_im_bf, x_im_bf)
    ri = np.matmul(A_re_bf, x_im_bf)
    ir = np.matmul(A_im_bf, x_re_bf)

    # Combine results (still BF16)
    real_bf = rr - ii
    imag_bf = ri + ir

    return real_bf, imag_bf


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
# Full BF16 SSM Apply Functions
# ============================================================================

def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """Compute the LxH output of discretized SSM given an LxH input.

    Full BF16 implementation: all intermediate computations in BF16.

    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        input_sequence (float32/bfloat16): input sequence            (L, H)
        conj_sym (bool):         whether conjugate symmetry is enforced
        bidirectional (bool):    whether bidirectional setup is used
    Returns:
        ys (bfloat16): the SSM outputs (S5 layer preactivations)     (L, H)
    """
    L = input_sequence.shape[0]

    # [DEBUG BF16] Check inputs for NaN
    jax.debug.print("[apply_ssm] Lambda_bar has NaN: {}, shape: {}", np.any(np.isnan(Lambda_bar)), Lambda_bar.shape)  # DEBUG BF16
    jax.debug.print("[apply_ssm] B_bar has NaN: {}, shape: {}", np.any(np.isnan(B_bar)), B_bar.shape)  # DEBUG BF16
    jax.debug.print("[apply_ssm] C_tilde has NaN: {}, shape: {}", np.any(np.isnan(C_tilde)), C_tilde.shape)  # DEBUG BF16
    jax.debug.print("[apply_ssm] input_sequence has NaN: {}, shape: {}, dtype: {}", np.any(np.isnan(input_sequence)), input_sequence.shape, input_sequence.dtype)  # DEBUG BF16

    # Convert input to BF16
    input_bf16 = input_sequence.astype(np.bfloat16)
    jax.debug.print("[apply_ssm] input_bf16 has NaN: {}", np.any(np.isnan(input_bf16)))  # DEBUG BF16

    # Convert Lambda_bar to BF16 pair and broadcast to (L, P)
    Lambda_re_bf, Lambda_im_bf = complex_to_bf16_pair(Lambda_bar)
    jax.debug.print("[apply_ssm] Lambda_re_bf has NaN: {}, range: [{}, {}]", np.any(np.isnan(Lambda_re_bf)), np.min(Lambda_re_bf), np.max(Lambda_re_bf))  # DEBUG BF16
    jax.debug.print("[apply_ssm] Lambda_im_bf has NaN: {}, range: [{}, {}]", np.any(np.isnan(Lambda_im_bf)), np.min(Lambda_im_bf), np.max(Lambda_im_bf))  # DEBUG BF16

    Lambda_re_elements = np.broadcast_to(Lambda_re_bf, (L, Lambda_re_bf.shape[0]))
    Lambda_im_elements = np.broadcast_to(Lambda_im_bf, (L, Lambda_im_bf.shape[0]))

    # Compute Bu = B_bar @ u for each timestep, output as BF16 pair
    def compute_Bu(u_bf16):
        return complex_matvec_bf16_real_x_bf16_out(B_bar, u_bf16)

    Bu_re_elements, Bu_im_elements = jax.vmap(compute_Bu)(input_bf16)
    jax.debug.print("[apply_ssm] Bu_re_elements has NaN: {}, shape: {}", np.any(np.isnan(Bu_re_elements)), Bu_re_elements.shape)  # DEBUG BF16
    jax.debug.print("[apply_ssm] Bu_im_elements has NaN: {}, shape: {}", np.any(np.isnan(Bu_im_elements)), Bu_im_elements.shape)  # DEBUG BF16

    # Run associative scan with BF16 binary operator
    (_, _), (xs_re, xs_im) = jax.lax.associative_scan(
        binary_operator_bf16,
        ((Lambda_re_elements, Lambda_im_elements), (Bu_re_elements, Bu_im_elements))
    )
    jax.debug.print("[apply_ssm] After scan - xs_re has NaN: {}, range: [{}, {}]", np.any(np.isnan(xs_re)), np.min(xs_re), np.max(xs_re))  # DEBUG BF16
    jax.debug.print("[apply_ssm] After scan - xs_im has NaN: {}, range: [{}, {}]", np.any(np.isnan(xs_im)), np.min(xs_im), np.max(xs_im))  # DEBUG BF16

    if bidirectional:
        (_, _), (xs2_re, xs2_im) = jax.lax.associative_scan(
            binary_operator_bf16,
            ((Lambda_re_elements, Lambda_im_elements), (Bu_re_elements, Bu_im_elements)),
            reverse=True
        )
        xs_re = np.concatenate((xs_re, xs2_re), axis=-1)
        xs_im = np.concatenate((xs_im, xs2_im), axis=-1)

    # Compute C @ x for each timestep, output as BF16
    def compute_Cx(x_re_bf, x_im_bf):
        y_re, y_im = complex_matvec_bf16_bf16_out(C_tilde, x_re_bf, x_im_bf)
        # For SSM output we only need real part: 2*Re(C @ x) or Re(C @ x)
        return y_re

    if conj_sym:
        ys = jax.vmap(lambda xr, xi: 2 * compute_Cx(xr, xi))(xs_re, xs_im)
    else:
        ys = jax.vmap(compute_Cx)(xs_re, xs_im)

    jax.debug.print("[apply_ssm] Final ys has NaN: {}, range: [{}, {}]", np.any(np.isnan(ys)), np.min(ys), np.max(ys))  # DEBUG BF16

    return ys  # Returns BF16
    
def apply_ssm_rnn(Lambda_bar, B_bar, C_tilde, hidden, input_sequence, resets, conj_sym, bidirectional):
    """Compute the LxH output of discretized SSM in RNN mode.

    Full BF16 implementation: all intermediate computations in BF16.
    Hidden state is stored as BF16 pair (hidden_re, hidden_im).

    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        hidden: hidden state - either complex64 (1, P) or BF16 pair ((1, P), (1, P))
        input_sequence (float32/bfloat16): input sequence            (L, H)
        resets (bool): reset signals                                 (L,)
        conj_sym (bool): whether conjugate symmetry is enforced
        bidirectional (bool): whether bidirectional (not supported in RNN mode)
    Returns:
        hidden_out: BF16 pair ((1, P), (1, P)) for next call
        ys (bfloat16): the SSM outputs                               (L, H)
    """
    L = input_sequence.shape[0]
    P = Lambda_bar.shape[0]

    # Convert input to BF16
    input_bf16 = input_sequence.astype(np.bfloat16)

    # Convert Lambda_bar to BF16 pair
    Lambda_re_bf, Lambda_im_bf = complex_to_bf16_pair(Lambda_bar)

    # Handle hidden state: convert from complex64 if needed
    if isinstance(hidden, tuple) and len(hidden) == 2:
        # Already BF16 pair
        hidden_re, hidden_im = hidden
    else:
        # Convert from complex64
        hidden_re, hidden_im = complex_to_bf16_pair(hidden)

    # Broadcast Lambda to (L+1, P) - prepend ones for initial state
    ones_re = np.ones((1, P), dtype=np.bfloat16)
    ones_im = np.zeros((1, P), dtype=np.bfloat16)
    Lambda_re_elements = np.concatenate([ones_re, np.broadcast_to(Lambda_re_bf, (L, P))])
    Lambda_im_elements = np.concatenate([ones_im, np.broadcast_to(Lambda_im_bf, (L, P))])

    # Compute Bu = B_bar @ u for each timestep
    def compute_Bu(u_bf16):
        return complex_matvec_bf16_real_x_bf16_out(B_bar, u_bf16)

    Bu_re_seq, Bu_im_seq = jax.vmap(compute_Bu)(input_bf16)

    # Prepend hidden state
    Bu_re_elements = np.concatenate([hidden_re, Bu_re_seq])
    Bu_im_elements = np.concatenate([hidden_im, Bu_im_seq])

    # Run associative scan
    if resets is None:
        (_, _), (xs_re, xs_im) = jax.lax.associative_scan(
            binary_operator_bf16,
            ((Lambda_re_elements, Lambda_im_elements), (Bu_re_elements, Bu_im_elements))
        )
    else:
        # Prepend zero reset for initial state
        resets_padded = np.concatenate([np.zeros(1, dtype=resets.dtype), resets])
        (_, _), (xs_re, xs_im), _ = jax.lax.associative_scan(
            binary_operator_bf16_reset,
            ((Lambda_re_elements, Lambda_im_elements), (Bu_re_elements, Bu_im_elements), resets_padded)
        )

    # Extract hidden state for next call (last state)
    hidden_out = (xs_re[np.newaxis, -1], xs_im[np.newaxis, -1])

    # Remove the prepended initial state
    xs_re = xs_re[1:]
    xs_im = xs_im[1:]

    if bidirectional:
        raise ValueError("Cannot expect a bidirectional view if doing rnn")

    # Compute C @ x for each timestep
    def compute_Cx(x_re_bf, x_im_bf):
        y_re, y_im = complex_matvec_bf16_bf16_out(C_tilde, x_re_bf, x_im_bf)
        return y_re

    if conj_sym:
        ys = jax.vmap(lambda xr, xi: 2 * compute_Cx(xr, xi))(xs_re, xs_im)
    else:
        ys = jax.vmap(compute_Cx)(xs_re, xs_im)

    return hidden_out, ys  # Returns BF16 pair and BF16 output


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

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

        # [DEBUG BF16] Check for NaN after discretization in setup()
        jax.debug.print("[S5SSM.setup] After discretization - Lambda_bar has NaN: {}", np.any(np.isnan(self.Lambda_bar)))  # DEBUG BF16
        jax.debug.print("[S5SSM.setup] After discretization - B_bar has NaN: {}", np.any(np.isnan(self.B_bar)))  # DEBUG BF16
        jax.debug.print("[S5SSM.setup] After discretization - C_tilde has NaN: {}", np.any(np.isnan(self.C_tilde)))  # DEBUG BF16
        jax.debug.print("[S5SSM.setup] After discretization - D has NaN: {}", np.any(np.isnan(self.D)))  # DEBUG BF16

    def __call__(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.

        Full BF16 implementation: all computations in BF16.

        Args:
             input_sequence (float32/bfloat16): input sequence (L, H)
        Returns:
            output sequence (bfloat16): (L, H)
        """
        # [DEBUG BF16] Check discretized params in setup()
        jax.debug.print("[S5SSM.__call__] self.Lambda_bar has NaN: {}", np.any(np.isnan(self.Lambda_bar)))  # DEBUG BF16
        jax.debug.print("[S5SSM.__call__] self.B_bar has NaN: {}", np.any(np.isnan(self.B_bar)))  # DEBUG BF16
        jax.debug.print("[S5SSM.__call__] self.C_tilde has NaN: {}", np.any(np.isnan(self.C_tilde)))  # DEBUG BF16
        jax.debug.print("[S5SSM.__call__] self.D has NaN: {}", np.any(np.isnan(self.D)))  # DEBUG BF16

        # Full BF16: convert input to BF16
        input_bf16 = input_sequence.astype(np.bfloat16)
        jax.debug.print("[S5SSM.__call__] input_sequence dtype: {}, has NaN: {}", input_sequence.dtype, np.any(np.isnan(input_sequence)))  # DEBUG BF16

        # apply_ssm now returns BF16
        ys = apply_ssm(self.Lambda_bar,
                       self.B_bar,
                       self.C_tilde,
                       input_bf16,
                       self.conj_sym,
                       self.bidirectional)

        jax.debug.print("[S5SSM.__call__] ys from apply_ssm has NaN: {}", np.any(np.isnan(ys)))  # DEBUG BF16

        # D feedthrough in BF16
        D_bf16 = self.D.astype(np.bfloat16)
        jax.debug.print("[S5SSM.__call__] D_bf16 has NaN: {}", np.any(np.isnan(D_bf16)))  # DEBUG BF16

        Du = jax.vmap(lambda u: D_bf16 * u)(input_bf16)
        jax.debug.print("[S5SSM.__call__] Du has NaN: {}", np.any(np.isnan(Du)))  # DEBUG BF16

        output = ys + Du
        jax.debug.print("[S5SSM.__call__] Final output has NaN: {}, range: [{}, {}]", np.any(np.isnan(output)), np.min(output), np.max(output))  # DEBUG BF16

        # Return BF16 output
        return output

    def __call_rnn__(self, hidden, input_sequence, resets):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using RNN-style forward pass.

        Full BF16 implementation: all computations in BF16.
        Hidden state is BF16 pair (hidden_re, hidden_im).

        Args:
             hidden: BF16 pair ((1, P), (1, P)) or complex64 (1, P)
             input_sequence (float32/bfloat16): input sequence (L, H)
             resets (bool): reset signals (L,)
        Returns:
            hidden_out: BF16 pair for next call
            output (bfloat16): (L, H)
        """
        # Full BF16: convert input to BF16
        input_bf16 = input_sequence.astype(np.bfloat16)

        # apply_ssm_rnn now returns BF16 pair and BF16 output
        hidden_out, ys = apply_ssm_rnn(self.Lambda_bar,
                                        self.B_bar,
                                        self.C_tilde,
                                        hidden,
                                        input_bf16,
                                        resets,
                                        self.conj_sym,
                                        self.bidirectional)

        # D feedthrough in BF16
        D_bf16 = self.D.astype(np.bfloat16)
        Du = jax.vmap(lambda u: D_bf16 * u)(input_bf16)
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
