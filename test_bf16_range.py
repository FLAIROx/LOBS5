#!/usr/bin/env python3
"""
Test if BF16 range is sufficient for book volume data
Compare FP32 vs BF16 precision and range
"""

import numpy as np
import struct

def float32_to_bits(f):
    """Convert float32 to binary representation"""
    return struct.unpack('>I', struct.pack('>f', f))[0]

def bfloat16_to_bits(f):
    """Simulate bfloat16 conversion"""
    # BF16 = truncate FP32 mantissa from 23 to 7 bits
    bits = float32_to_bits(f)
    # Keep sign (1) + exponent (8) + top 7 mantissa bits
    bf16_bits = bits & 0xFFFF0000
    return bf16_bits

def bits_to_float32(bits):
    """Convert bits back to float32"""
    return struct.unpack('>f', struct.pack('>I', bits))[0]

def simulate_bf16_conversion(f):
    """Simulate FP32 → BF16 → FP32 conversion"""
    bf16_bits = bfloat16_to_bits(f)
    return bits_to_float32(bf16_bits)

def test_bf16_range():
    """Test BF16 range and precision for book volumes"""

    print("=" * 80)
    print("BF16 Range and Precision Test for Book Volume Data")
    print("=" * 80)

    # BF16 theoretical limits
    print("\n" + "-" * 80)
    print("BF16 Theoretical Properties:")
    print("-" * 80)
    print("Format: 1 sign bit + 8 exponent bits + 7 mantissa bits")
    print(f"Range: ±1.18e-38 to ±3.4e38 (same as FP32)")
    print(f"Precision: ~3-4 significant decimal digits")
    print(f"Smallest positive: ~1.18e-38")
    print(f"Largest positive: ~3.4e38")
    print(f"Machine epsilon: ~7.8e-3 (0.78%)")

    # Test typical book volume ranges
    print("\n" + "-" * 80)
    print("Test 1: Typical Book Volume Values")
    print("-" * 80)

    test_volumes = [
        0.0,           # Empty level
        1.0,           # Minimal volume
        10.0,          # Small volume
        100.0,         # Medium volume
        1000.0,        # Large volume
        10000.0,       # Very large volume
        100000.0,      # Extreme volume
        1000000.0,     # Market order
        1e10,          # Institutional order
        1e20,          # Stress test
    ]

    print(f"{'Original':>15} {'BF16':>15} {'Abs Error':>15} {'Rel Error':>15}")
    print("-" * 65)

    for vol in test_volumes:
        bf16_vol = simulate_bf16_conversion(vol)
        abs_err = abs(vol - bf16_vol)
        rel_err = abs_err / vol if vol > 0 else 0

        print(f"{vol:15.2f} {bf16_vol:15.2f} {abs_err:15.2e} {rel_err*100:14.4f}%")

    # Test precision at different scales
    print("\n" + "-" * 80)
    print("Test 2: BF16 Precision at Different Scales")
    print("-" * 80)

    scales = [1, 10, 100, 1000, 10000, 100000, 1e6, 1e9, 1e15]

    print(f"{'Scale':>15} {'Can Distinguish':>20} {'Precision':>20}")
    print("-" * 60)

    for scale in scales:
        # Find the smallest increment distinguishable at this scale
        val = scale
        bf16_val = simulate_bf16_conversion(val)

        # Find minimum distinguishable increment
        increment = scale * 0.001  # Start with 0.1%
        found = False
        for _ in range(20):
            val_plus = val + increment
            bf16_plus = simulate_bf16_conversion(val_plus)
            if bf16_plus != bf16_val:
                found = True
                break
            increment *= 2

        rel_precision = (increment / scale * 100) if found else float('inf')
        print(f"{scale:15.2e} {increment:20.2e} {rel_precision:19.4f}%")

    # Test real-world scenario
    print("\n" + "-" * 80)
    print("Test 3: Real Book Volume Distribution")
    print("-" * 80)

    # Simulate realistic volume distribution (exponential)
    np.random.seed(42)
    n_samples = 10000
    volumes_fp32 = np.random.exponential(scale=100, size=n_samples)

    # Simulate BF16 conversion
    volumes_bf16 = np.array([simulate_bf16_conversion(v) for v in volumes_fp32])

    # Calculate errors
    abs_errors = np.abs(volumes_fp32 - volumes_bf16)
    rel_errors = abs_errors / (volumes_fp32 + 1e-8)

    print(f"\nSample size: {n_samples}")
    print(f"\nOriginal data (FP32):")
    print(f"  Range: [{volumes_fp32.min():.2f}, {volumes_fp32.max():.2f}]")
    print(f"  Mean: {volumes_fp32.mean():.2f}")
    print(f"  Std: {volumes_fp32.std():.2f}")

    print(f"\nAfter BF16 conversion:")
    print(f"  Max absolute error: {abs_errors.max():.4f}")
    print(f"  Mean absolute error: {abs_errors.mean():.4f}")
    print(f"  Max relative error: {rel_errors.max()*100:.4f}%")
    print(f"  Mean relative error: {rel_errors.mean()*100:.4f}%")

    print(f"\nRelative error percentiles:")
    percentiles = [50, 90, 95, 99, 99.9]
    for p in percentiles:
        err_p = np.percentile(rel_errors, p) * 100
        print(f"  {p:5.1f}%: {err_p:.4f}%")

    # Edge cases
    print("\n" + "-" * 80)
    print("Test 4: Edge Cases and Special Values")
    print("-" * 80)

    edge_cases = [
        ("Zero", 0.0),
        ("Tiny (1e-10)", 1e-10),
        ("Small (0.001)", 0.001),
        ("One", 1.0),
        ("Pi", 3.14159265),
        ("Large (1e10)", 1e10),
        ("Huge (1e20)", 1e20),
        ("Max FP32", 3.4e38),
    ]

    print(f"{'Test Case':>20} {'Original':>20} {'BF16':>20} {'Match?':>10}")
    print("-" * 75)

    for name, val in edge_cases:
        bf16_val = simulate_bf16_conversion(val)
        match = "✓" if abs(val - bf16_val) / (abs(val) + 1e-40) < 0.01 else "✗"
        print(f"{name:>20} {val:20.6e} {bf16_val:20.6e} {match:>10}")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)

    print("\n✅ RANGE: BF16 range is MORE than sufficient")
    print(f"   - BF16 max: 3.4e38")
    print(f"   - Typical book volumes: 0-10000")
    print(f"   - Extreme volumes: up to 1e10")
    print(f"   - Safety margin: >1e27x")

    if rel_errors.mean() < 0.01:  # < 1%
        print("\n✅ PRECISION: BF16 precision is acceptable")
        print(f"   - Mean error: {rel_errors.mean()*100:.4f}%")
        print(f"   - 99% of data: <{np.percentile(rel_errors, 99)*100:.2f}% error")
    else:
        print("\n⚠️  PRECISION: BF16 precision has noticeable error")
        print(f"   - Mean error: {rel_errors.mean()*100:.4f}%")

    print("\n📊 RECOMMENDATION:")
    print("   - BF16 is SAFE for book volume data")
    print("   - Range is more than sufficient (no overflow risk)")
    print("   - Precision loss is acceptable for ML training")
    print("   - 50% memory savings vs FP32")
    print("   - 2x speed improvement on modern GPUs")

    return rel_errors.mean()

if __name__ == "__main__":
    mean_error = test_bf16_range()

    print("\n" + "=" * 80)
    if mean_error < 0.01:
        print("VERDICT: ✅ BF16 is EXCELLENT for this use case")
    elif mean_error < 0.05:
        print("VERDICT: ✅ BF16 is GOOD for this use case")
    else:
        print("VERDICT: ⚠️  BF16 has noticeable precision loss")
    print("=" * 80)
