#!/usr/bin/env python3
"""
精确定位 NaN 产生的确切位置
"""

import jax
import jax.numpy as jnp
import sys

sys.path.append('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from s5.ssm import binary_operator
from functools import partial
from jax import remat


def trace_scan_processing():
    """追踪 scan 处理的每一步"""
    print("="*60)
    print("追踪 scan 处理流程")
    print("="*60)

    # 设置参数
    L, P, H = 300, 32, 64
    chunk_size = 100
    n_chunks = L // chunk_size

    # 创建测试数据
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 7)

    # 使用与 test_chunked_equivalence.py 相同的数据生成
    Lambda_real = jax.random.normal(keys[0], (P,)) * 0.3
    Lambda_imag = jax.random.normal(keys[1], (P,)) * 0.3
    Lambda_bar = Lambda_real + 1j * Lambda_imag

    # 稳定化 - 检查不同方法
    print("\n1. Lambda_bar 稳定化测试:")
    print(f"  原始 Lambda_bar[0:3]: {Lambda_bar[:3]}")
    print(f"  原始最大模: {jnp.max(jnp.abs(Lambda_bar)):.4f}")

    # 方法1：原始测试中的方法
    Lambda_magnitude = jnp.abs(Lambda_bar)
    Lambda_v1 = Lambda_bar / jnp.maximum(Lambda_magnitude, 0.01) * 0.95
    Lambda_v1 = Lambda_v1 - jnp.abs(Lambda_v1.real) - 0.01
    print(f"\n  方法1 (当前): {Lambda_v1[:3]}")
    print(f"    最大模: {jnp.max(jnp.abs(Lambda_v1)):.4f}")

    # 方法2：温和缩放
    Lambda_v2 = Lambda_bar * 0.9 / jnp.maximum(jnp.abs(Lambda_bar), 1.0)
    print(f"\n  方法2 (温和缩放): {Lambda_v2[:3]}")
    print(f"    最大模: {jnp.max(jnp.abs(Lambda_v2)):.4f}")

    # 使用方法2继续测试
    Lambda_bar = Lambda_v2
    B_bar = jnp.ones((P, H), dtype=jnp.complex64) * 0.01
    input_sequence = jnp.ones((L, H), dtype=jnp.float32) * 0.1

    # Reshape to chunks
    chunks = input_sequence.reshape(n_chunks, chunk_size, H)

    print("\n2. 手动处理 chunks:")

    # scan_chunk_forward 的实现
    def scan_chunk_forward(carry_state, chunk):
        """模拟 apply_ssm_chunked 中的 scan_chunk_forward"""
        Lambda_elements = Lambda_bar * jnp.ones((chunk_size, P))
        Bu_elements = jax.vmap(lambda u: B_bar @ u)(chunk)

        Lambda_with_carry = jnp.concatenate([
            jnp.ones((1, P), dtype=Lambda_elements.dtype),
            Lambda_elements
        ], axis=0)

        Bu_with_carry = jnp.concatenate([
            carry_state[jnp.newaxis, :],
            Bu_elements
        ], axis=0)

        # 使用 associative_scan
        scan_fn = partial(jax.lax.associative_scan, binary_operator)
        Lambda_cum, Bu_cum = scan_fn((Lambda_with_carry, Bu_with_carry))

        xs = Bu_cum[1:]
        new_carry = Bu_cum[-1]

        # 返回格式与 apply_ssm_chunked 相同
        return new_carry, (xs, Lambda_elements, Bu_elements)

    # 初始 carry
    init_carry = jnp.zeros(P, dtype=Lambda_bar.dtype)

    # 使用 jax.lax.scan
    print(f"\n  处理 {n_chunks} 个 chunks...")
    final_carry, accumulated = jax.lax.scan(scan_chunk_forward, init_carry, chunks)

    print(f"\n3. 分析 scan 返回值:")
    print(f"  final_carry shape: {final_carry.shape}")
    print(f"  accumulated type: {type(accumulated)}")

    if isinstance(accumulated, tuple):
        print(f"  accumulated 是元组，长度: {len(accumulated)}")
        xs_chunks, Lambda_chunks, Bu_chunks = accumulated
        print(f"    xs_chunks shape: {xs_chunks.shape}")
        print(f"    Lambda_chunks shape: {Lambda_chunks.shape}")
        print(f"    Bu_chunks shape: {Bu_chunks.shape}")

        # 检查 xs_chunks 的内容
        print(f"\n  xs_chunks 包含 NaN: {jnp.any(jnp.isnan(xs_chunks))}")
        if jnp.any(jnp.isnan(xs_chunks)):
            nan_mask = jnp.isnan(xs_chunks)
            nan_indices = jnp.where(nan_mask)
            if len(nan_indices[0]) > 0:
                first_nan = (nan_indices[0][0], nan_indices[1][0], nan_indices[2][0])
                print(f"    第一个 NaN 位置: chunk={first_nan[0]}, pos={first_nan[1]}, dim={first_nan[2]}")

        # Reshape
        xs = xs_chunks.reshape(L, P)
        print(f"\n  Reshape 后 xs shape: {xs.shape}")
        print(f"  xs 包含 NaN: {jnp.any(jnp.isnan(xs))}")


def test_stable_lambda_generation():
    """测试更稳定的 Lambda 生成方法"""
    print("\n" + "="*60)
    print("测试稳定的 Lambda 生成")
    print("="*60)

    P = 512
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 2)

    # 生成基础 Lambda
    Lambda_real = jax.random.normal(keys[0], (P,)) * 0.3
    Lambda_imag = jax.random.normal(keys[1], (P,)) * 0.3
    Lambda_base = Lambda_real + 1j * Lambda_imag

    print("\n不同稳定化方法对比:")

    # 方法1：激进方法（当前）
    Lambda_magnitude = jnp.abs(Lambda_base)
    Lambda_m1 = Lambda_base / jnp.maximum(Lambda_magnitude, 0.01) * 0.95
    Lambda_m1 = Lambda_m1 - jnp.abs(Lambda_m1.real) - 0.01

    print(f"\n方法1 (激进):")
    print(f"  示例值: {Lambda_m1[:3]}")
    print(f"  实部范围: [{jnp.min(Lambda_m1.real):.4f}, {jnp.max(Lambda_m1.real):.4f}]")
    print(f"  模范围: [{jnp.min(jnp.abs(Lambda_m1)):.4f}, {jnp.max(jnp.abs(Lambda_m1)):.4f}]")

    # 方法2：简单缩放
    max_magnitude = jnp.max(jnp.abs(Lambda_base))
    if max_magnitude > 0.95:
        Lambda_m2 = Lambda_base * 0.95 / max_magnitude
    else:
        Lambda_m2 = Lambda_base

    print(f"\n方法2 (简单缩放):")
    print(f"  示例值: {Lambda_m2[:3]}")
    print(f"  实部范围: [{jnp.min(Lambda_m2.real):.4f}, {jnp.max(Lambda_m2.real):.4f}]")
    print(f"  模范围: [{jnp.min(jnp.abs(Lambda_m2)):.4f}, {jnp.max(jnp.abs(Lambda_m2)):.4f}]")

    # 方法3：逐元素限制
    Lambda_m3 = Lambda_base / jnp.maximum(jnp.abs(Lambda_base), 1.05) * 0.95

    print(f"\n方法3 (逐元素限制):")
    print(f"  示例值: {Lambda_m3[:3]}")
    print(f"  实部范围: [{jnp.min(Lambda_m3.real):.4f}, {jnp.max(Lambda_m3.real):.4f}]")
    print(f"  模范围: [{jnp.min(jnp.abs(Lambda_m3)):.4f}, {jnp.max(jnp.abs(Lambda_m3)):.4f}]")


def check_xs_vs_original():
    """对比原始版本和 chunked 版本的 xs 值"""
    print("\n" + "="*60)
    print("对比原始和 chunked 版本")
    print("="*60)

    from s5.ssm import apply_ssm_original, apply_ssm_chunked

    L, P, H = 300, 32, 64
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    # 使用简单稳定的数据
    Lambda_bar = jnp.ones(P, dtype=jnp.complex64) * (0.8 + 0.1j)
    B_bar = jnp.ones((P, H), dtype=jnp.complex64) * 0.01
    C_tilde = jnp.ones((H, P), dtype=jnp.complex64) * 0.01
    input_sequence = jnp.ones((L, H), dtype=jnp.float32) * 0.1

    print("\n测试简单稳定数据:")
    print(f"  Lambda_bar[0]: {Lambda_bar[0]}, 模: {jnp.abs(Lambda_bar[0]):.4f}")

    # 原始版本
    ys_orig, Bu_orig, Lambda_orig, xs_orig = apply_ssm_original(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=False
    )
    print(f"\n原始版本:")
    print(f"  xs_orig shape: {xs_orig.shape}")
    print(f"  xs_orig[0,0]: {xs_orig[0,0]}")
    print(f"  xs_orig 包含 NaN: {jnp.any(jnp.isnan(xs_orig))}")
    print(f"  xs_orig type: {type(xs_orig)}")

    # Chunked 版本
    ys_chunked, Bu_chunked, Lambda_chunked, xs_chunked = apply_ssm_chunked(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=False, chunk_size=100
    )
    print(f"\nChunked 版本:")
    print(f"  xs_chunked shape: {xs_chunked.shape}")
    print(f"  xs_chunked[0,0]: {xs_chunked[0,0] if not jnp.isnan(xs_chunked[0,0]) else 'NaN'}")
    print(f"  xs_chunked 包含 NaN: {jnp.any(jnp.isnan(xs_chunked))}")
    print(f"  xs_chunked type: {type(xs_chunked)}")

    if not jnp.any(jnp.isnan(xs_chunked)):
        diff = jnp.max(jnp.abs(xs_orig - xs_chunked))
        print(f"\n最大差异: {diff:.2e}")
    else:
        print("\n无法比较，chunked 版本有 NaN")

        # 尝试找出问题
        print("\n深入诊断:")
        print(f"  Bu_chunked 包含 NaN: {jnp.any(jnp.isnan(Bu_chunked))}")
        print(f"  Lambda_chunked 包含 NaN: {jnp.any(jnp.isnan(Lambda_chunked))}")
        print(f"  ys_chunked 包含 NaN: {jnp.any(jnp.isnan(ys_chunked))}")


if __name__ == "__main__":
    trace_scan_processing()
    test_stable_lambda_generation()
    check_xs_vs_original()

    print("\n" + "="*60)
    print("精确诊断完成！")
    print("="*60)