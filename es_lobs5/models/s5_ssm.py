"""
ES-compatible S5 State Space Model.

This module provides an ES-compatible implementation of the S5 SSM layer
that works with the HyperscaleES noiser framework.

The key challenge is handling complex-valued parameters (Lambda, B, C)
which are stored as real tensors with shape (..., 2) for (real, imag).

REUSES: Core SSM logic from s5/ssm.py (discretization, parallel scan, apply_ssm)
ADDS: ES parameter wrapping via CommonParams and noiser injection
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eigh

from .common import (
    Model, CommonInit, CommonParams,
    PARAM, EXCLUDED,
)

# Import SSM core functions from s5/ssm.py (no duplication!)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from s5.ssm import (
    discretize_zoh,
    discretize_bilinear,
    binary_operator,
    binary_operator_reset,
    apply_ssm,
    apply_ssm_rnn,
)
# Import initialization helpers from s5/ssm_init.py
from s5.ssm_init import init_CV, init_VinvB, init_log_steps

__all__ = [
    'ES_S5SSM',
    'make_DPLR_HiPPO',
    'init_hippo_matrices',
    # Re-export for convenience (but they come from s5/ssm.py)
    'discretize_zoh',
    'discretize_bilinear',
    'apply_ssm',
]


# =============================================================================
# HiPPO Initialization (from s5/ssm_init.py)
# =============================================================================

def make_HiPPO(N):
    """Create a HiPPO-LegS matrix."""
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """Make components for NPLR representation of HiPPO-LegS."""
    hippo = make_HiPPO(N)
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Make components for DPLR representation of HiPPO-LegS.

    Returns:
        Lambda: eigenvalues (complex)
        P: low-rank term
        B: conjugated input matrix
        V: eigenvectors
        B_orig: original B
    """
    A, P, B = make_NPLR_HiPPO(N)
    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def init_hippo_matrices(ssm_size, blocks, conj_sym=True):
    """
    Initialize HiPPO matrices for S5.

    Args:
        ssm_size: Total state space size (H * blocks)
        blocks: Number of SSM blocks (J)
        conj_sym: Whether to enforce conjugate symmetry

    Returns:
        Lambda_re_init, Lambda_im_init, V, Vinv as jnp arrays
    """
    block_size = ssm_size // blocks
    Lambda_list, V_list = [], []

    for _ in range(blocks):
        Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
        if conj_sym:
            # Take only upper half (conjugate pairs)
            Lambda = Lambda[:block_size // 2]
            V = V[:, :block_size // 2]
        Lambda_list.append(Lambda)
        V_list.append(V)

    # Stack blocks diagonally
    Lambda = jnp.concatenate(Lambda_list)
    V = jax.scipy.linalg.block_diag(*V_list)
    Vinv = V.conj().T

    Lambda_re_init = Lambda.real
    Lambda_im_init = Lambda.imag

    return Lambda_re_init, Lambda_im_init, V, Vinv


# =============================================================================
# ES_S5SSM Model
# =============================================================================

class ES_S5SSM(Model):
    """
    ES-compatible S5 State Space Model.

    Parameters are classified for ES as follows:
    - Lambda_re, Lambda_im: EXCLUDED (stability critical)
    - log_step: EXCLUDED (stability critical)
    - B: PARAM (safe to perturb)
    - C: MM_PARAM (LORA for efficiency, used in matmul)
    - D: PARAM (simple feedthrough)

    Complex parameters (B, C) are stored as (..., 2) real tensors.
    """

    @classmethod
    def rand_init(
        cls,
        key,
        H: int,
        P: int,
        Lambda_re_init: jnp.ndarray,
        Lambda_im_init: jnp.ndarray,
        V: jnp.ndarray,
        Vinv: jnp.ndarray,
        C_init: str = 'trunc_standard_normal',
        discretization: str = 'zoh',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conj_sym: bool = True,
        clip_eigs: bool = True,
        bidirectional: bool = False,
        step_rescale: float = 1.0,
        dtype=jnp.float32,
    ) -> CommonInit:
        """
        Initialize S5 SSM parameters.

        Args:
            key: JAX random key
            H: Feature dimension (hidden size)
            P: State space dimension
            Lambda_re_init, Lambda_im_init: Initial eigenvalues (from HiPPO)
            V, Vinv: Eigenvector matrices (from HiPPO)
            C_init: C matrix initialization method
            discretization: 'zoh' or 'bilinear'
            dt_min, dt_max: Timescale range
            conj_sym: Conjugate symmetry (reduces params by half)
            clip_eigs: Clip eigenvalues to left-half plane
            bidirectional: Use bidirectional processing
            step_rescale: Timescale scaling factor
            dtype: Data type

        Returns:
            CommonInit with params, frozen_params, es_map, scan_map
        """
        keys = jax.random.split(key, 4)

        local_P = 2 * P if conj_sym else P

        # Initialize B (input matrix) - use init_VinvB from s5/ssm_init.py
        B = init_VinvB(lecun_normal(), keys[0], (local_P, H), Vinv).astype(dtype)

        # Initialize C (output matrix) - use init_CV/trunc_standard_normal from s5/ssm_init.py
        if bidirectional:
            C_shape = (H, 2 * local_P, 2)
        else:
            C_shape = (H, local_P, 2)

        from s5.ssm_init import trunc_standard_normal
        C = init_CV(trunc_standard_normal, keys[1], C_shape, V).astype(dtype)

        # Initialize D (feedthrough)
        D = (jax.random.normal(keys[2], (H,)) * 1.0).astype(dtype)

        # Initialize log_step - use init_log_steps from s5/ssm_init.py
        # Note: init_log_steps expects (key, (P, dt_min, dt_max))
        log_step = init_log_steps(keys[3], (P, dt_min, dt_max)).astype(dtype)

        # Build params dict
        params = {
            'Lambda_re': Lambda_re_init.astype(dtype),
            'Lambda_im': Lambda_im_init.astype(dtype),
            'B': B,
            'C': C,
            'D': D,
            'log_step': log_step,
        }

        # ES map: which params get which perturbation type
        es_map = {
            'Lambda_re': EXCLUDED,  # Stability critical
            'Lambda_im': EXCLUDED,  # Stability critical
            'B': PARAM,             # Safe to perturb
            'C': PARAM,             # Note: Could use MM_PARAM but C is used in complex matmul
            'D': PARAM,             # Simple feedthrough
            'log_step': EXCLUDED,   # Stability critical
        }

        # Scan map: no vmap dimensions for these params
        scan_map = {k: () for k in params}

        # Frozen params (not learned, but needed for forward)
        frozen_params = {
            'V': V,
            'Vinv': Vinv,
            'H': H,
            'P': P,
            'conj_sym': conj_sym,
            'clip_eigs': clip_eigs,
            'bidirectional': bidirectional,
            'discretization': discretization,
            'step_rescale': step_rescale,
        }

        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        """
        Compute S5 SSM forward pass.

        Args:
            common_params: CommonParams with noiser and params
            x: Input sequence (L, H) float

        Returns:
            Output sequence (L, H) float
        """
        fp = common_params.frozen_params
        noiser = common_params.noiser

        # Get (potentially noisy) parameters
        def get_param(name):
            param = common_params.params[name]
            es_key = common_params.es_tree_key[name]
            if common_params.iterinfo is None:
                return param
            return noiser.get_noisy_standard(
                common_params.frozen_noiser_params,
                common_params.noiser_params,
                param, es_key, common_params.iterinfo
            )

        Lambda_re = get_param('Lambda_re')
        Lambda_im = get_param('Lambda_im')
        B = get_param('B')
        C = get_param('C')
        D = get_param('D')
        log_step = get_param('log_step')

        # Reconstruct complex eigenvalues
        Lambda = Lambda_re + 1j * Lambda_im
        if fp['clip_eigs']:
            Lambda = jnp.clip(Lambda.real, None, -1e-4) + 1j * Lambda.imag

        # Reconstruct complex B and C from (real, imag) storage
        B_tilde = B[..., 0] + 1j * B[..., 1]
        C_tilde = C[..., 0] + 1j * C[..., 1]

        # Compute discretization step
        step = fp['step_rescale'] * jnp.exp(log_step[:, 0])

        # Discretize
        if fp['discretization'] == 'zoh':
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        else:
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

        # Cast input to float32 for SSM computation
        input_dtype = x.dtype
        x_fp32 = x.astype(jnp.float32)

        # Apply SSM
        ys = apply_ssm(
            Lambda_bar, B_bar, C_tilde, x_fp32,
            fp['conj_sym'], fp['bidirectional']
        )

        # Add feedthrough
        Du = jax.vmap(lambda u: D * u)(x_fp32)
        output = ys + Du

        return output.astype(input_dtype)

    @classmethod
    def _forward_rnn(cls, common_params: CommonParams, hidden, x, resets=None):
        """
        Compute S5 SSM forward pass in RNN mode (step-by-step).

        Uses apply_ssm_rnn from s5/ssm.py for the core computation.

        Args:
            common_params: CommonParams with noiser and params
            hidden: Hidden state (batch, 1, P) or (1, P) complex
            x: Input sequence (L, H) float
            resets: Optional reset signals (L,) bool

        Returns:
            new_hidden, output_sequence
        """
        fp = common_params.frozen_params
        noiser = common_params.noiser

        # Handle batch dimension: initialize_carry creates (batch, 1, P),
        # but apply_ssm_rnn expects (1, P)
        has_batch_dim = hidden.ndim == 3
        if has_batch_dim:
            hidden = hidden.squeeze(0)  # (batch=1, 1, P) → (1, P)

        # Get (potentially noisy) parameters (same as _forward)
        def get_param(name):
            param = common_params.params[name]
            es_key = common_params.es_tree_key[name]
            if common_params.iterinfo is None:
                return param
            return noiser.get_noisy_standard(
                common_params.frozen_noiser_params,
                common_params.noiser_params,
                param, es_key, common_params.iterinfo
            )

        Lambda_re = get_param('Lambda_re')
        Lambda_im = get_param('Lambda_im')
        B = get_param('B')
        C = get_param('C')
        D = get_param('D')
        log_step = get_param('log_step')

        # Reconstruct complex
        Lambda = Lambda_re + 1j * Lambda_im
        if fp['clip_eigs']:
            Lambda = jnp.clip(Lambda.real, None, -1e-4) + 1j * Lambda.imag

        B_tilde = B[..., 0] + 1j * B[..., 1]
        C_tilde = C[..., 0] + 1j * C[..., 1]

        # Discretize
        step = fp['step_rescale'] * jnp.exp(log_step[:, 0])
        if fp['discretization'] == 'zoh':
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        else:
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

        # Cast input to float32 for SSM computation
        input_dtype = x.dtype
        x_fp32 = x.astype(jnp.float32)

        # Apply SSM RNN mode - uses apply_ssm_rnn from s5/ssm.py
        hidden_out, ys = apply_ssm_rnn(
            Lambda_bar, B_bar, C_tilde, hidden, x_fp32, resets,
            fp['conj_sym'], fp['bidirectional']
        )

        # Restore batch dimension if it was present
        if has_batch_dim:
            hidden_out = hidden_out[None, ...]  # (1, P) → (1, 1, P)

        # Add feedthrough
        Du = jax.vmap(lambda u: D * u)(x_fp32)
        output = ys + Du

        return hidden_out, output.astype(input_dtype)
