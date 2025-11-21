#!/usr/bin/env python3
"""
Test script to verify numerical equivalence between original and chunked associative scan.

This script tests:
1. Numerical accuracy (max difference < 1e-5)
2. Different sequence lengths and chunk sizes
3. Bidirectional mode
4. Memory usage comparison
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Add project root to path
sys.path.append('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from s5.ssm import apply_ssm_original, apply_ssm_chunked, apply_ssm


def create_test_data(L, P, H, key):
    """Create stable random test data for SSM."""
    keys = jax.random.split(key, 7)

    # Generate stable Lambda_bar (eigenvalues with magnitude < 1)
    # Important: For SSM stability, all eigenvalues must have magnitude < 1
    Lambda_real = jax.random.normal(keys[0], (P,)) * 0.3
    Lambda_imag = jax.random.normal(keys[1], (P,)) * 0.3
    Lambda_bar = Lambda_real + 1j * Lambda_imag

    # Simple and gentle stabilization: scale if any eigenvalue has magnitude >= 0.95
    max_magnitude = jnp.max(jnp.abs(Lambda_bar))
    if max_magnitude > 0.95:
        Lambda_bar = Lambda_bar * 0.95 / max_magnitude

    # Optional: ensure negative real parts for continuous-time stability
    # (but more gently - only shift if positive)
    Lambda_bar = Lambda_bar - jnp.maximum(Lambda_bar.real, 0.0) - 0.01

    # Generate B_bar and C_tilde with proper complex values
    B_real = jax.random.normal(keys[2], (P, H)) * 0.01
    B_imag = jax.random.normal(keys[3], (P, H)) * 0.01
    B_bar = B_real + 1j * B_imag

    C_real = jax.random.normal(keys[4], (H, P)) * 0.01
    C_imag = jax.random.normal(keys[5], (H, P)) * 0.01
    C_tilde = C_real + 1j * C_imag

    # Real-valued input with small magnitude
    input_sequence = jax.random.normal(keys[6], (L, H), dtype=jnp.float32) * 0.1

    return Lambda_bar, B_bar, C_tilde, input_sequence


def test_numerical_equivalence(L, P, H, chunk_size=None, bidirectional=False):
    """Test that chunked and original produce same results."""
    print(f"\n{'='*60}")
    print(f"Testing L={L}, P={P}, H={H}, chunk_size={chunk_size}, bidirectional={bidirectional}")
    print(f"{'='*60}")

    # Create test data
    key = jax.random.PRNGKey(42)
    Lambda_bar, B_bar, C_tilde, input_sequence = create_test_data(L, P, H, key)

    if bidirectional:
        # For bidirectional, C_tilde needs 2P columns
        C_tilde = jax.random.normal(key, (H, 2*P), dtype=jnp.complex64) * 0.1

    # Run original implementation
    print("Running original implementation...")
    start_time = time.time()
    ys_orig, Bu_orig, Lambda_orig, xs_orig = apply_ssm_original(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=bidirectional
    )
    orig_time = time.time() - start_time
    print(f"  Original time: {orig_time:.3f}s")

    # Run chunked implementation
    print("Running chunked implementation...")
    start_time = time.time()
    ys_chunked, Bu_chunked, Lambda_chunked, xs_chunked = apply_ssm_chunked(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=bidirectional, chunk_size=chunk_size
    )
    chunked_time = time.time() - start_time
    print(f"  Chunked time: {chunked_time:.3f}s")

    # Compare results
    print("\nComparing results:")

    # Check outputs (ys)
    max_diff_ys = jnp.max(jnp.abs(ys_orig - ys_chunked))
    print(f"  Max diff in ys: {max_diff_ys:.2e}")

    # Check hidden states (xs)
    max_diff_xs = jnp.max(jnp.abs(xs_orig - xs_chunked))
    print(f"  Max diff in xs: {max_diff_xs:.2e}")

    # Check Bu elements
    max_diff_Bu = jnp.max(jnp.abs(Bu_orig - Bu_chunked))
    print(f"  Max diff in Bu: {max_diff_Bu:.2e}")

    # Check Lambda elements
    max_diff_Lambda = jnp.max(jnp.abs(Lambda_orig - Lambda_chunked))
    print(f"  Max diff in Lambda: {max_diff_Lambda:.2e}")

    # Performance comparison
    print(f"\nPerformance:")
    print(f"  Speedup: {orig_time/chunked_time:.2f}x")
    print(f"  Overhead: {(chunked_time - orig_time)/orig_time * 100:.1f}%")

    # Verify accuracy
    tolerance = 1e-5
    passed = (max_diff_ys < tolerance and max_diff_xs < tolerance and
              max_diff_Bu < tolerance and max_diff_Lambda < tolerance)

    if passed:
        print(f"✅ Test PASSED! All differences < {tolerance}")
    else:
        print(f"❌ Test FAILED! Some differences >= {tolerance}")

    return passed, max_diff_ys


def test_automatic_selection():
    """Test the automatic selection mechanism."""
    print("\n" + "="*60)
    print("Testing automatic selection (apply_ssm)")
    print("="*60)

    key = jax.random.PRNGKey(42)

    # Test short sequence (should use original)
    print("\nShort sequence (L=1000, should use original):")
    os.environ['JAX_USE_CHUNKED_SCAN'] = 'auto'
    os.environ['JAX_DEBUG_PRINT'] = 'false'  # Disable debug for cleaner output

    Lambda_bar, B_bar, C_tilde, input_sequence = create_test_data(1000, 512, 1024, key)
    ys, _, _, _ = apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, True, False)
    print(f"  Output shape: {ys.shape}")

    # Test long sequence (should use chunked)
    print("\nLong sequence (L=12000, should use chunked):")
    Lambda_bar, B_bar, C_tilde, input_sequence = create_test_data(12000, 512, 1024, key)
    ys, _, _, _ = apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, True, False)
    print(f"  Output shape: {ys.shape}")

    # Test forcing original
    print("\nForcing original (L=12000, JAX_USE_CHUNKED_SCAN=false):")
    os.environ['JAX_USE_CHUNKED_SCAN'] = 'false'
    ys, _, _, _ = apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, True, False)
    print(f"  Output shape: {ys.shape}")

    # Test forcing chunked
    print("\nForcing chunked (L=1000, JAX_USE_CHUNKED_SCAN=true):")
    os.environ['JAX_USE_CHUNKED_SCAN'] = 'true'
    os.environ['JAX_CHUNK_SIZE'] = '500'  # Force small chunk size for short sequence
    Lambda_bar, B_bar, C_tilde, input_sequence = create_test_data(1000, 512, 1024, key)
    ys, _, _, _ = apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, True, False)
    print(f"  Output shape: {ys.shape}")

    print("\n✅ Automatic selection test completed!")


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUITE FOR CHUNKED ASSOCIATIVE SCAN")
    print("="*60)

    all_passed = True

    # Test configurations
    test_configs = [
        # (L, P, H, chunk_size, bidirectional)
        (3000, 512, 1024, 1000, False),    # Basic test
        (3000, 512, 1024, 1500, False),    # Different chunk size
        (6000, 512, 1024, 2000, False),    # Larger sequence
        (12000, 512, 1024, 3000, False),   # Full sequence (500 orders)
        (12000, 1024, 1024, 3000, False),  # Larger state size
        (12000, 512, 1024, None, False),   # Auto chunk size
        (6000, 512, 1024, 2000, True),     # Bidirectional
    ]

    results = []
    for L, P, H, chunk_size, bidirectional in test_configs:
        passed, max_diff = test_numerical_equivalence(L, P, H, chunk_size, bidirectional)
        results.append((L, P, H, chunk_size, bidirectional, passed, max_diff))
        all_passed = all_passed and passed

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("\n{:<10} {:<10} {:<10} {:<12} {:<15} {:<10} {:<15}".format(
        "L", "P", "H", "chunk_size", "bidirectional", "passed", "max_diff"
    ))
    print("-"*92)

    for L, P, H, chunk_size, bidirectional, passed, max_diff in results:
        chunk_str = str(chunk_size) if chunk_size else "auto"
        passed_str = "✅ PASS" if passed else "❌ FAIL"
        print(f"{L:<10} {P:<10} {H:<10} {chunk_str:<12} {str(bidirectional):<15} {passed_str:<10} {max_diff:.2e}")

    # Test automatic selection
    test_automatic_selection()

    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Chunked implementation is numerically equivalent!")
    else:
        print("⚠️ SOME TESTS FAILED! Please review the results above.")
    print("="*60)

    return all_passed


def test_memory_usage():
    """Test memory usage comparison (requires monitoring tools)."""
    print("\n" + "="*60)
    print("MEMORY USAGE TEST")
    print("="*60)
    print("\nNote: This test provides timing comparisons.")
    print("For actual memory measurements, use external monitoring tools.")

    # Test with BSZ=1 (single sequence) first
    L, P, H = 12000, 1024, 1024
    key = jax.random.PRNGKey(42)

    Lambda_bar, B_bar, C_tilde, input_sequence = create_test_data(L, P, H, key)

    # Compile both versions
    print("\nCompiling functions...")

    # Force compilation
    apply_ssm_original_jit = jax.jit(apply_ssm_original, static_argnums=(4, 5))
    apply_ssm_chunked_jit = jax.jit(
        partial(apply_ssm_chunked, chunk_size=3000),
        static_argnums=(4, 5)
    )

    print("Warming up original...")
    _ = apply_ssm_original_jit(Lambda_bar, B_bar, C_tilde, input_sequence, True, False)

    print("Warming up chunked...")
    _ = apply_ssm_chunked_jit(Lambda_bar, B_bar, C_tilde, input_sequence, True, False)

    # Time multiple runs
    print("\nTiming 10 runs each...")

    # Original
    start = time.time()
    for _ in range(10):
        _ = apply_ssm_original_jit(Lambda_bar, B_bar, C_tilde, input_sequence, True, False)
    orig_time = (time.time() - start) / 10

    # Chunked
    start = time.time()
    for _ in range(10):
        _ = apply_ssm_chunked_jit(Lambda_bar, B_bar, C_tilde, input_sequence, True, False)
    chunked_time = (time.time() - start) / 10

    print(f"\nAverage runtime (L={L}, P={P}, H={H}):")
    print(f"  Original: {orig_time*1000:.1f} ms")
    print(f"  Chunked:  {chunked_time*1000:.1f} ms")
    print(f"  Overhead: {(chunked_time - orig_time)/orig_time * 100:.1f}%")

    print("\n✅ Memory usage test completed!")
    print("Note: Monitor actual memory with nvidia-smi or system tools during training.")


if __name__ == "__main__":
    print("JAX devices:", jax.devices())

    # Run tests
    all_passed = run_comprehensive_tests()

    # Memory usage test
    test_memory_usage()

    # Exit code
    sys.exit(0 if all_passed else 1)