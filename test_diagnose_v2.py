#!/usr/bin/env python3
"""
深入诊断 - 使用实际测试数据重现 NaN 问题
"""

import jax
import jax.numpy as jnp
import sys
import numpy as np

sys.path.append('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from s5.ssm import apply_ssm_original, apply_ssm_chunked, binary_operator


def test_random_complex_generation():
    """测试复数生成的问题"""
    print("="*60)
    print("测试复数生成方式")
    print("="*60)

    P = 32
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    # 方式1：直接指定 dtype=complex64（可能有问题）
    print("\n方式1: jax.random.normal(..., dtype=complex64)")
    try:
        Lambda_v1 = jax.random.normal(keys[0], (P,), dtype=jnp.complex64)
        print(f"  成功！前3个值: {Lambda_v1[:3]}")
        print(f"  实部非零: {jnp.any(Lambda_v1.real != 0)}")
        print(f"  虚部非零: {jnp.any(Lambda_v1.imag != 0)}")
    except Exception as e:
        print(f"  失败: {e}")

    # 方式2：分别生成实部和虚部（推荐）
    print("\n方式2: 分别生成实部和虚部")
    Lambda_real = jax.random.normal(keys[1], (P,))
    Lambda_imag = jax.random.normal(keys[2], (P,))
    Lambda_v2 = Lambda_real + 1j * Lambda_imag
    print(f"  前3个值: {Lambda_v2[:3]}")
    print(f"  dtype: {Lambda_v2.dtype}")

    # 应用稳定化变换
    Lambda_v2_stable = Lambda_v2 - 0.5 - 1j * jnp.abs(Lambda_v2.imag)
    print(f"  稳定化后: {Lambda_v2_stable[:3]}")


def diagnose_with_actual_test_data():
    """使用与实际测试相同的数据"""
    print("\n" + "="*60)
    print("使用实际测试数据进行诊断")
    print("="*60)

    L, P, H = 3000, 512, 1024  # 实际测试的尺寸
    chunk_size = 1000
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    # 按照实际测试的方式生成数据
    print("\n1. 生成测试数据（与 test_chunked_equivalence.py 相同）:")

    # 检查 jax.random.normal 对 complex64 的行为
    Lambda_bar_test = jax.random.normal(keys[0], (P,), dtype=jnp.complex64)
    print(f"  Lambda_bar (直接生成): dtype={Lambda_bar_test.dtype}")
    print(f"    前3个值: {Lambda_bar_test[:3]}")
    print(f"    虚部是否为0: {jnp.all(Lambda_bar_test.imag == 0)}")

    # 如果虚部为0，这会导致问题！
    Lambda_bar_test = Lambda_bar_test - 0.5 - 1j * jnp.abs(Lambda_bar_test.imag)
    print(f"  Lambda_bar (变换后): {Lambda_bar_test[:3]}")

    # 正确的生成方式
    Lambda_real = jax.random.normal(keys[1], (P,))
    Lambda_imag = jax.random.normal(keys[2], (P,))
    Lambda_bar = (Lambda_real + 1j * Lambda_imag) * 0.1  # 缩放以确保稳定
    Lambda_bar = Lambda_bar - 0.5 - 0.5j  # 确保稳定性

    B_real = jax.random.normal(keys[3], (P, H)) * 0.1
    B_imag = jax.random.normal(keys[4], (P, H)) * 0.1
    B_bar = B_real + 1j * B_imag

    C_tilde = jax.random.normal(keys[5], (H, P), dtype=jnp.complex64) * 0.1
    input_sequence = jax.random.normal(keys[0], (L, H), dtype=jnp.float32)

    print(f"\n  Lambda_bar: shape={Lambda_bar.shape}, dtype={Lambda_bar.dtype}")
    print(f"  B_bar: shape={B_bar.shape}, dtype={B_bar.dtype}")
    print(f"  input_sequence: shape={input_sequence.shape}, dtype={input_sequence.dtype}")

    # 测试 chunked 版本
    print("\n2. 测试 chunked 版本:")
    try:
        ys, Bu, Lambda, xs = apply_ssm_chunked(
            Lambda_bar, B_bar, C_tilde, input_sequence,
            conj_sym=True, bidirectional=False, chunk_size=chunk_size
        )
        print(f"  成功!")
        print(f"  xs 包含 NaN: {jnp.any(jnp.isnan(xs))}")
        print(f"  ys 包含 NaN: {jnp.any(jnp.isnan(ys))}")

        if jnp.any(jnp.isnan(xs)):
            # 找出第一个 NaN 的位置
            nan_mask = jnp.isnan(xs)
            nan_indices = jnp.where(nan_mask)
            if len(nan_indices[0]) > 0:
                first_nan_idx = (nan_indices[0][0], nan_indices[1][0])
                print(f"  第一个 NaN 位置: {first_nan_idx}")
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()


def analyze_scan_return_structure():
    """详细分析 associative_scan 的返回值结构"""
    print("\n" + "="*60)
    print("详细分析 associative_scan 返回值")
    print("="*60)

    P = 32
    L = 10  # 小尺寸便于手动验证

    # 创建简单数据
    Lambda_elements = jnp.ones((L, P), dtype=jnp.complex64) * 0.9
    Bu_elements = jnp.ones((L, P), dtype=jnp.complex64) * 0.1

    print(f"\n输入形状: Lambda_elements={Lambda_elements.shape}, Bu_elements={Bu_elements.shape}")

    # 调用 associative_scan
    result = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    print(f"\n返回值类型: {type(result)}")
    if isinstance(result, tuple):
        print(f"返回元组，长度: {len(result)}")
        for i, r in enumerate(result):
            print(f"  result[{i}]: shape={r.shape if hasattr(r, 'shape') else 'N/A'}, "
                  f"type={type(r)}")

        # 手动计算验证
        print("\n手动验证前3个元素:")
        print("期望值:")
        print("  xs[0] = Bu[0] = 0.1")
        print("  xs[1] = Lambda[1]*xs[0] + Bu[1] = 0.9*0.1 + 0.1 = 0.19")
        print("  xs[2] = Lambda[2]*xs[1] + Bu[2] = 0.9*0.19 + 0.1 = 0.271")

        if len(result) == 2:
            Lambda_cum, Bu_cum = result
            print("\n实际值（Bu_cum，即 hidden states）:")
            print(f"  xs[0] = {Bu_cum[0, 0]}")
            print(f"  xs[1] = {Bu_cum[1, 0]}")
            print(f"  xs[2] = {Bu_cum[2, 0]}")


def test_chunked_scan_internals():
    """测试 chunked scan 的内部逻辑"""
    print("\n" + "="*60)
    print("测试 chunked scan 内部逻辑")
    print("="*60)

    # 手动实现一个 chunk 的处理
    P = 32
    chunk_size = 100
    H = 64

    Lambda_bar = jnp.ones(P, dtype=jnp.complex64) * (0.9 + 0.1j)
    B_bar = jnp.ones((P, H), dtype=jnp.complex64) * 0.1
    chunk = jnp.ones((chunk_size, H), dtype=jnp.float32)
    carry_state = jnp.ones(P, dtype=jnp.complex64) * 0.5  # 非零 carry

    print(f"\n测试非零 carry state 的处理:")
    print(f"  carry_state[0]: {carry_state[0]}")

    # 计算
    Lambda_elements = Lambda_bar * jnp.ones((chunk_size, P))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(chunk)

    # Prepend carry
    Lambda_with_carry = jnp.concatenate([
        jnp.ones((1, P), dtype=Lambda_elements.dtype),
        Lambda_elements
    ], axis=0)

    Bu_with_carry = jnp.concatenate([
        carry_state[jnp.newaxis, :],
        Bu_elements
    ], axis=0)

    # 调用 scan
    Lambda_cum, Bu_cum = jax.lax.associative_scan(
        binary_operator, (Lambda_with_carry, Bu_with_carry)
    )

    print(f"\n返回值形状: Lambda_cum={Lambda_cum.shape}, Bu_cum={Bu_cum.shape}")

    # 验证第一个输出是否正确包含 carry
    print(f"\n第一个元素（应该包含 carry）:")
    print(f"  Bu_cum[0, 0]: {Bu_cum[0, 0]} (应该等于 carry_state[0] = {carry_state[0]})")
    print(f"  Bu_cum[1, 0]: {Bu_cum[1, 0]} (应该是 Lambda*carry + Bu[0])")

    expected_first = Lambda_bar[0] * carry_state[0] + Bu_elements[0, 0]
    print(f"  期望的 Bu_cum[1, 0]: {expected_first}")

    # 提取结果
    xs = Bu_cum[1:]  # 移除 prepended carry
    print(f"\n提取后的 xs 形状: {xs.shape}")
    print(f"  xs[0, 0]: {xs[0, 0]} (第一个真实的 hidden state)")


if __name__ == "__main__":
    print(f"JAX 版本: {jax.__version__}")

    # 运行所有诊断
    test_random_complex_generation()
    analyze_scan_return_structure()
    test_chunked_scan_internals()
    diagnose_with_actual_test_data()

    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)