"""
HiPPO Initialization Adapter.

Provides HiPPO-LegS initialization for S5 SSM by directly reusing
s5/ssm_init.py - no code duplication.

Usage:
    from es_lobs5.adapters.hippo_adapter import get_hippo_params

    params = get_hippo_params(ssm_size=256, blocks=8, conj_sym=True)
    # params['Lambda_re_init'], params['Lambda_im_init'], params['V'], params['Vinv']
"""

import jax.numpy as jnp
from jax.scipy.linalg import block_diag

# Reuse s5's HiPPO initialization directly
from s5.ssm_init import make_DPLR_HiPPO


def get_hippo_params(ssm_size: int, blocks: int, conj_sym: bool = True) -> dict:
    """
    Get HiPPO-LegS initialization parameters.

    Directly reuses s5/ssm_init.py - no code duplication.

    Args:
        ssm_size: Total state space size (P). Must be divisible by blocks.
        blocks: Number of SSM blocks (J). Each block gets ssm_size/blocks states.
        conj_sym: If True, enforce conjugate symmetry by taking only the upper
                  half of eigenvalues (reduces effective state size by 2x but
                  ensures real-valued outputs).

    Returns:
        dict: {
            'Lambda_re_init': jnp.ndarray - Eigenvalue real parts, shape depends on conj_sym:
                              - conj_sym=True: (ssm_size // 2,)
                              - conj_sym=False: (ssm_size,)
            'Lambda_im_init': jnp.ndarray - Eigenvalue imaginary parts, same shape as Lambda_re
            'V': jnp.ndarray - Eigenvector matrix (block diagonal)
            'Vinv': jnp.ndarray - Inverse eigenvector matrix (V^H for unitary V)
        }

    Example:
        >>> params = get_hippo_params(ssm_size=256, blocks=8, conj_sym=True)
        >>> params['Lambda_re_init'].shape  # (128,) since conj_sym halves it
        >>> params['V'].shape  # (256, 128) - full size to half

    Note:
        When conj_sym=True (default), eigenvalues come in conjugate pairs.
        We only store one of each pair, halving memory and ensuring real outputs
        when combined with their conjugates during computation.
    """
    if ssm_size % blocks != 0:
        raise ValueError(f"ssm_size ({ssm_size}) must be divisible by blocks ({blocks})")

    block_size = ssm_size // blocks

    if conj_sym and block_size % 2 != 0:
        raise ValueError(
            f"With conj_sym=True, block_size ({block_size}) must be even. "
            f"Either use conj_sym=False or adjust ssm_size/blocks."
        )

    Lambda_list = []
    V_list = []

    for _ in range(blocks):
        # Get HiPPO DPLR components from s5
        # Returns: Lambda (complex), P, B, V, B_orig
        Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)

        if conj_sym:
            # Eigenvalues come in conjugate pairs for real matrices.
            # Take only upper half - the other half is just the conjugate.
            # This halves memory and ensures real-valued outputs.
            half = block_size // 2
            Lambda = Lambda[:half]
            V = V[:, :half]

        Lambda_list.append(Lambda)
        V_list.append(V)

    # Concatenate eigenvalues from all blocks
    Lambda = jnp.concatenate(Lambda_list)

    # Build block diagonal eigenvector matrix
    V = block_diag(*V_list)

    # For unitary V, V^{-1} = V^H (conjugate transpose)
    Vinv = V.conj().T

    return {
        'Lambda_re_init': Lambda.real,
        'Lambda_im_init': Lambda.imag,
        'V': V,
        'Vinv': Vinv,
    }


def get_effective_state_size(ssm_size: int, conj_sym: bool = True) -> int:
    """
    Get the effective state size after applying conj_sym.

    Args:
        ssm_size: Total configured state space size
        conj_sym: Whether conjugate symmetry is enforced

    Returns:
        Effective state size (ssm_size // 2 if conj_sym else ssm_size)
    """
    return ssm_size // 2 if conj_sym else ssm_size
