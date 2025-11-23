#!/usr/bin/env python3
"""
Test INT8 quantization for book volume data
Compare memory usage and numerical accuracy
"""

import jax
import jax.numpy as jnp
import numpy as np

def quantize_to_int8(x, scale=None, zero_point=None):
    """
    Quantize FP32/BF16 to INT8

    Args:
        x: input array (FP32 or BF16)
        scale: quantization scale (computed if None)
        zero_point: zero point (computed if None)

    Returns:
        x_int8, scale, zero_point
    """
    x_fp32 = x.astype(jnp.float32)

    if scale is None or zero_point is None:
        # Symmetric quantization
        x_max = jnp.max(jnp.abs(x_fp32))
        scale = x_max / 127.0  # INT8 range: -128 to 127
        zero_point = 0

    # Quantize
    x_scaled = x_fp32 / scale
    x_int8 = jnp.clip(jnp.round(x_scaled), -128, 127).astype(jnp.int8)

    return x_int8, scale, zero_point


def dequantize_from_int8(x_int8, scale, zero_point):
    """
    Dequantize INT8 back to FP32

    Args:
        x_int8: quantized array (INT8)
        scale: quantization scale
        zero_point: zero point

    Returns:
        x_fp32: dequantized array
    """
    x_fp32 = x_int8.astype(jnp.float32) * scale + zero_point
    return x_fp32


def test_book_volume_quantization():
    """
    Test INT8 quantization on simulated book volume data
    """
    print("=" * 80)
    print("INT8 Quantization Test for Book Volume Data")
    print("=" * 80)

    # Simulate book volume data
    # Typical shape: (batch, seq_len, num_levels)
    batch_size = 128
    seq_len = 500
    num_levels = 21  # 10 bid + 10 ask + 1 (example)

    # Generate realistic volume data (positive values, typically 0-10000)
    np.random.seed(42)
    book_volumes_fp32 = np.random.exponential(scale=100, size=(batch_size, seq_len, num_levels)).astype(np.float32)
    book_volumes_bf16 = book_volumes_fp32.astype(jnp.bfloat16)

    print(f"\nData shape: {book_volumes_fp32.shape}")
    print(f"Data range: [{book_volumes_fp32.min():.2f}, {book_volumes_fp32.max():.2f}]")
    print(f"Data mean: {book_volumes_fp32.mean():.2f}")
    print(f"Data std: {book_volumes_fp32.std():.2f}")

    # Memory usage
    mem_fp32 = book_volumes_fp32.nbytes / (1024**2)
    mem_bf16 = book_volumes_bf16.nbytes / (1024**2)

    print("\n" + "-" * 80)
    print("Memory Usage:")
    print("-" * 80)
    print(f"FP32: {mem_fp32:.2f} MB")
    print(f"BF16: {mem_bf16:.2f} MB")

    # Test quantization
    book_volumes_jax = jnp.array(book_volumes_fp32)
    x_int8, scale, zero_point = quantize_to_int8(book_volumes_jax)

    mem_int8 = x_int8.nbytes / (1024**2)
    scale_mem = scale.nbytes / (1024**2) if isinstance(scale, jnp.ndarray) else 0.000001

    print(f"INT8: {mem_int8:.2f} MB (+ {scale_mem:.6f} MB for scale)")
    print(f"\nMemory savings:")
    print(f"  FP32 → INT8: {(1 - mem_int8/mem_fp32)*100:.1f}%")
    print(f"  BF16 → INT8: {(1 - mem_int8/mem_bf16)*100:.1f}%")

    # Dequantize and check accuracy
    book_volumes_dequant = dequantize_from_int8(x_int8, scale, zero_point)

    # Compute errors
    abs_error = jnp.abs(book_volumes_jax - book_volumes_dequant)
    rel_error = abs_error / (jnp.abs(book_volumes_jax) + 1e-8)

    print("\n" + "-" * 80)
    print("Numerical Accuracy:")
    print("-" * 80)
    print(f"Quantization scale: {scale:.6f}")
    print(f"Max absolute error: {jnp.max(abs_error):.4f}")
    print(f"Mean absolute error: {jnp.mean(abs_error):.4f}")
    print(f"Max relative error: {jnp.max(rel_error)*100:.2f}%")
    print(f"Mean relative error: {jnp.mean(rel_error)*100:.2f}%")

    # Check percentiles
    percentiles = [50, 90, 95, 99, 99.9]
    print(f"\nRelative error percentiles:")
    for p in percentiles:
        err_p = jnp.percentile(rel_error, p) * 100
        print(f"  {p:5.1f}%: {err_p:.2f}%")

    # Test impact on downstream computation
    print("\n" + "-" * 80)
    print("Impact on Downstream Computation:")
    print("-" * 80)

    # Example: sum across levels (common operation)
    sum_fp32 = jnp.sum(book_volumes_jax, axis=-1)
    sum_int8 = jnp.sum(book_volumes_dequant, axis=-1)
    sum_error = jnp.abs(sum_fp32 - sum_int8) / (jnp.abs(sum_fp32) + 1e-8)

    print(f"Sum operation:")
    print(f"  Max relative error: {jnp.max(sum_error)*100:.2f}%")
    print(f"  Mean relative error: {jnp.mean(sum_error)*100:.2f}%")

    # Example: mean across levels
    mean_fp32 = jnp.mean(book_volumes_jax, axis=-1)
    mean_int8 = jnp.mean(book_volumes_dequant, axis=-1)
    mean_error = jnp.abs(mean_fp32 - mean_int8) / (jnp.abs(mean_fp32) + 1e-8)

    print(f"\nMean operation:")
    print(f"  Max relative error: {jnp.max(mean_error)*100:.2f}%")
    print(f"  Mean relative error: {jnp.mean(mean_error)*100:.2f}%")

    # Test gradient flow (simulate backward pass)
    print("\n" + "-" * 80)
    print("Gradient Flow Test:")
    print("-" * 80)

    def simple_loss(x):
        return jnp.sum(x ** 2)

    grad_fn = jax.grad(simple_loss)

    # FP32 gradients
    grad_fp32 = grad_fn(book_volumes_jax)

    # INT8 gradients (through dequantization)
    def loss_with_dequant(x_int8_input):
        x_dequant = dequantize_from_int8(x_int8_input, scale, zero_point)
        return simple_loss(x_dequant)

    # Note: gradients w.r.t. INT8 don't make sense,
    # we'd need to store original FP32 for backward pass
    print("WARNING: INT8 quantization for activations requires:")
    print("  1. Store FP32/BF16 during forward for backward pass")
    print("  2. Or use quantization-aware training")
    print("  3. For inference only: INT8 is beneficial")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print(f"Memory savings: FP32→INT8 = {(1 - mem_int8/mem_fp32)*100:.1f}%")
    print(f"Numerical accuracy: Mean relative error = {jnp.mean(rel_error)*100:.2f}%")
    print()

    if jnp.mean(rel_error) < 0.01:  # < 1% error
        print("✅ GOOD: INT8 quantization maintains high accuracy")
        print("   Recommended for: Inference, or as activation compression")
    elif jnp.mean(rel_error) < 0.05:  # < 5% error
        print("⚠️  MODERATE: INT8 quantization has noticeable error")
        print("   Recommended for: Inference only, test on validation set")
    else:
        print("❌ BAD: INT8 quantization degrades accuracy significantly")
        print("   Not recommended for this data distribution")

    print("\nFor TRAINING:")
    print("  - Keep FP32/BF16 for backward pass")
    print("  - Can use INT8 for storage/transfer only")
    print("  - Requires custom autograd rules")

    print("\nFor INFERENCE:")
    print("  - INT8 quantization is beneficial")
    print("  - 75% memory savings")
    print("  - Potential speedup on INT8-optimized hardware")


if __name__ == "__main__":
    test_book_volume_quantization()
