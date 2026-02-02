"""Tests for Sliding Window Recurrences (SWR) implementation.

This module tests the SWR algorithm against the standard associative_scan
to verify numerical correctness and TBPTT compatibility.

Run tests:
    python -m pytest tests/test_swr.py -v

Or run directly:
    python tests/test_swr.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from s5.swr import (
    build_transfer_matrix,
    apply_ssm_swr_pass1,
    apply_ssm_swr_pass2,
    apply_ssm_swr,
)
from s5.ssm import apply_ssm


def create_test_ssm_params(P=64, H=128, seed=42):
    """Create test SSM parameters.

    Args:
        P: State size (latent dimension)
        H: Hidden/feature size
        seed: Random seed

    Returns:
        Lambda_bar, B_bar, C_tilde: SSM parameters
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Lambda_bar: discretized diagonal state matrix (P,) complex
    # Use |Lambda| < 1 for stability (as in actual S5 with clip_eigs)
    Lambda_re = jax.random.uniform(k1, (P,), minval=-0.99, maxval=-0.01)
    Lambda_im = jax.random.uniform(k2, (P,), minval=-0.5, maxval=0.5)
    Lambda_bar = Lambda_re + 1j * Lambda_im

    # B_bar: discretized input matrix (P, H) complex
    B_re = jax.random.normal(k3, (P, H)) * 0.1
    B_im = jax.random.normal(k4, (P, H)) * 0.1
    B_bar = B_re + 1j * B_im

    # C_tilde: output matrix (H, P) complex
    k5, k6 = jax.random.split(k4)
    C_re = jax.random.normal(k5, (H, P)) * 0.1
    C_im = jax.random.normal(k6, (H, P)) * 0.1
    C_tilde = C_re + 1j * C_im

    return Lambda_bar, B_bar, C_tilde


def test_build_transfer_matrix():
    """Test transfer matrix construction."""
    print("\n=== Test: build_transfer_matrix ===")

    P = 8
    W = 4
    Lambda_bar = jnp.array([0.9 + 0.1j] * P, dtype=jnp.complex64)

    M = build_transfer_matrix(Lambda_bar, W)

    # Check shape
    assert M.shape == (W, W, P), f"Expected shape {(W, W, P)}, got {M.shape}"

    # Check lower triangular structure
    for i in range(W):
        for j in range(W):
            if i < j:
                # Upper triangle should be zero
                assert jnp.allclose(M[i, j, :], 0), f"M[{i},{j},:] should be 0"
            else:
                # M[i, j, p] = Lambda_bar[p]^{i-j}
                expected = Lambda_bar ** (i - j)
                assert jnp.allclose(M[i, j, :], expected, rtol=1e-5), \
                    f"M[{i},{j},:] = {M[i,j,0]} != {expected[0]}"

    print("  Shape check: PASSED")
    print("  Lower triangular structure: PASSED")
    print("  Power values: PASSED")


def test_swr_vs_scan_numerical():
    """Test SWR output vs standard associative_scan."""
    print("\n=== Test: SWR vs Scan Numerical Correctness ===")

    # Test parameters
    P = 64
    H = 128
    L = 256  # Sequence length
    W = 16   # Window size

    Lambda_bar, B_bar, C_tilde = create_test_ssm_params(P, H)

    # Generate random input
    key = jax.random.PRNGKey(123)
    input_sequence = jax.random.normal(key, (L, H), dtype=jnp.float32)

    # Run standard associative_scan
    ys_scan = apply_ssm(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=False,
        hidden_in=None, return_hidden=False
    )

    # Run SWR
    ys_swr = apply_ssm_swr(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=False,
        window_size=W,
        hidden_in=None, return_hidden=False
    )

    # Compute errors
    abs_error = jnp.abs(ys_swr - ys_scan)
    rel_error = abs_error / (jnp.abs(ys_scan) + 1e-8)

    max_abs_error = jnp.max(abs_error)
    mean_abs_error = jnp.mean(abs_error)
    max_rel_error = jnp.max(rel_error)
    mean_rel_error = jnp.mean(rel_error)

    print(f"  Sequence length: {L}, Window size: {W}, State size: {P}")
    print(f"  Max absolute error:  {max_abs_error:.6e}")
    print(f"  Mean absolute error: {mean_abs_error:.6e}")
    print(f"  Max relative error:  {max_rel_error:.6e}")
    print(f"  Mean relative error: {mean_rel_error:.6e}")

    # SWR should be exact (not an approximation) for linear recurrence
    # Tolerance is for floating point precision
    assert max_rel_error < 1e-3, f"Max relative error {max_rel_error} too large"
    print("  PASSED: SWR matches associative_scan within tolerance")


def test_swr_hidden_state():
    """Test SWR hidden state carry (TBPTT compatibility)."""
    print("\n=== Test: SWR Hidden State Carry (TBPTT) ===")

    P = 32
    H = 64
    L = 128
    W = 16

    Lambda_bar, B_bar, C_tilde = create_test_ssm_params(P, H, seed=456)

    key = jax.random.PRNGKey(789)
    input_sequence = jax.random.normal(key, (L, H), dtype=jnp.float32)

    # Split sequence into two parts
    L1 = L // 2
    input_part1 = input_sequence[:L1]
    input_part2 = input_sequence[L1:]

    # Process with hidden state carry
    hidden_out1, ys_part1 = apply_ssm_swr(
        Lambda_bar, B_bar, C_tilde, input_part1,
        conj_sym=True, bidirectional=False,
        window_size=W,
        hidden_in=None, return_hidden=True
    )

    hidden_out2, ys_part2 = apply_ssm_swr(
        Lambda_bar, B_bar, C_tilde, input_part2,
        conj_sym=True, bidirectional=False,
        window_size=W,
        hidden_in=hidden_out1, return_hidden=True
    )

    # Concatenate results
    ys_segmented = jnp.concatenate([ys_part1, ys_part2], axis=0)

    # Process full sequence
    ys_full = apply_ssm_swr(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=False,
        window_size=W,
        hidden_in=None, return_hidden=False
    )

    # Compare
    abs_error = jnp.abs(ys_segmented - ys_full)
    max_abs_error = jnp.max(abs_error)
    mean_abs_error = jnp.mean(abs_error)

    print(f"  Full sequence: {L}, Split at: {L1}")
    print(f"  Hidden state shape: {hidden_out1.shape}")
    print(f"  Max absolute error:  {max_abs_error:.6e}")
    print(f"  Mean absolute error: {mean_abs_error:.6e}")

    # With proper hidden state carry, results should match
    assert max_abs_error < 1e-3, f"Hidden state carry error {max_abs_error} too large"
    print("  PASSED: Hidden state carry works correctly")


def test_swr_different_window_sizes():
    """Test SWR with different window sizes."""
    print("\n=== Test: SWR with Different Window Sizes ===")

    P = 32
    H = 64
    L = 256

    Lambda_bar, B_bar, C_tilde = create_test_ssm_params(P, H, seed=111)

    key = jax.random.PRNGKey(222)
    input_sequence = jax.random.normal(key, (L, H), dtype=jnp.float32)

    # Reference: standard scan
    ys_ref = apply_ssm(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=False
    )

    window_sizes = [8, 16, 32, 64]

    for W in window_sizes:
        ys_swr = apply_ssm_swr(
            Lambda_bar, B_bar, C_tilde, input_sequence,
            conj_sym=True, bidirectional=False,
            window_size=W
        )

        max_rel_error = jnp.max(jnp.abs(ys_swr - ys_ref) / (jnp.abs(ys_ref) + 1e-8))

        print(f"  Window size {W}: max relative error = {max_rel_error:.6e}")
        assert max_rel_error < 1e-3, f"Window size {W} error too large"

    print("  PASSED: All window sizes work correctly")


def test_swr_sequence_length_not_divisible():
    """Test SWR when sequence length is not divisible by window size."""
    print("\n=== Test: SWR with Non-Divisible Sequence Length ===")

    P = 32
    H = 64
    W = 16

    Lambda_bar, B_bar, C_tilde = create_test_ssm_params(P, H, seed=333)

    # Test various sequence lengths that don't divide evenly by W
    seq_lengths = [100, 123, 250, 333]

    for L in seq_lengths:
        key = jax.random.PRNGKey(L)
        input_sequence = jax.random.normal(key, (L, H), dtype=jnp.float32)

        # Reference
        ys_ref = apply_ssm(
            Lambda_bar, B_bar, C_tilde, input_sequence,
            conj_sym=True, bidirectional=False
        )

        # SWR
        ys_swr = apply_ssm_swr(
            Lambda_bar, B_bar, C_tilde, input_sequence,
            conj_sym=True, bidirectional=False,
            window_size=W
        )

        assert ys_swr.shape == ys_ref.shape, f"Shape mismatch: {ys_swr.shape} vs {ys_ref.shape}"

        max_rel_error = jnp.max(jnp.abs(ys_swr - ys_ref) / (jnp.abs(ys_ref) + 1e-8))
        print(f"  L={L} (L%W={L%W}): max relative error = {max_rel_error:.6e}")
        assert max_rel_error < 1e-3

    print("  PASSED: Non-divisible sequence lengths handled correctly")


def run_all_tests():
    """Run all SWR tests."""
    print("=" * 60)
    print("Running SWR (Sliding Window Recurrences) Tests")
    print("=" * 60)

    test_build_transfer_matrix()
    test_swr_vs_scan_numerical()
    test_swr_hidden_state()
    test_swr_different_window_sizes()
    test_swr_sequence_length_not_divisible()

    print("\n" + "=" * 60)
    print("All SWR tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
