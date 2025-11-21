#!/usr/bin/env python3
"""
诊断 bidirectional 模式的问题
"""

import jax
import jax.numpy as jnp
import sys

sys.path.append('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from s5.ssm import apply_ssm_original, apply_ssm_chunked, binary_operator


def test_reverse_scan_behavior():
    """测试 reverse scan 的基本行为"""
    print("="*60)
    print("测试 reverse scan 基本行为")
    print("="*60)

    P = 8
    L = 4

    # 创建简单数据便于手工验证
    Lambda_elements = jnp.array([[0.9]*P]*L, dtype=jnp.complex64)
    Bu_elements = jnp.array([[float(i)]*P for i in range(L)], dtype=jnp.complex64)

    print("\n输入数据:")
    print(f"  Lambda_elements: shape={Lambda_elements.shape}")
    print(f"  Bu_elements: \n{Bu_elements[:,0]}")  # 只显示第一维

    # Forward scan
    print("\n1. Forward scan:")
    Lambda_cum_f, Bu_cum_f = jax.lax.associative_scan(
        binary_operator, (Lambda_elements, Bu_elements)
    )
    print(f"  Bu_cum_f (first dim): {Bu_cum_f[:,0]}")
    print(f"  Expected: [0, 0.9*0+1, 0.9*(0.9*0+1)+2, ...]")
    print(f"           = [0, 1, 1.9, 2.71]")

    # Reverse scan
    print("\n2. Reverse scan:")
    Lambda_cum_r, Bu_cum_r = jax.lax.associative_scan(
        binary_operator, (Lambda_elements, Bu_elements), reverse=True
    )
    print(f"  Bu_cum_r (first dim): {Bu_cum_r[:,0]}")
    print(f"  Output order: indices 0,1,2,3 (same as input)")
    print(f"  But computed in reverse dependency order")

    # 手工计算 reverse scan 的预期值
    print("\n  手工计算 reverse scan:")
    print("  result[3] = Bu[3] = 3")
    print("  result[2] = Lambda[2]*Bu[3] + Bu[2] = 0.9*3 + 2 = 4.7")
    print("  result[1] = Lambda[1]*result[2] + Bu[1] = 0.9*4.7 + 1 = 5.23")
    print("  result[0] = Lambda[0]*result[1] + Bu[0] = 0.9*5.23 + 0 = 4.707")
    print(f"  Expected: [4.707, 5.23, 4.7, 3]")


def test_chunked_backward_simple():
    """测试简单的 chunked backward scan"""
    print("\n" + "="*60)
    print("测试 chunked backward scan")
    print("="*60)

    L, P, H = 100, 8, 16
    chunk_size = 50

    # 创建稳定的测试数据
    Lambda_bar = jnp.ones(P, dtype=jnp.complex64) * 0.9
    B_bar = jnp.ones((P, H), dtype=jnp.complex64) * 0.1
    C_tilde = jnp.ones((H, 2*P), dtype=jnp.complex64) * 0.1  # 2P for bidirectional
    input_sequence = jnp.arange(L)[:, None] * jnp.ones((L, H), dtype=jnp.float32) * 0.01

    print(f"\n测试参数: L={L}, P={P}, H={H}, chunk_size={chunk_size}")
    print(f"  Lambda_bar[0]: {Lambda_bar[0]}")

    # Original bidirectional
    print("\n1. Original bidirectional:")
    ys_orig, Bu_orig, Lambda_orig, xs_orig = apply_ssm_original(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=True
    )
    print(f"  xs_orig shape: {xs_orig.shape}")  # Should be (L, 2P)
    print(f"  xs_orig[0:3, 0] (forward): {xs_orig[0:3, 0]}")
    print(f"  xs_orig[0:3, P] (backward): {xs_orig[0:3, P]}")

    # Chunked bidirectional
    print("\n2. Chunked bidirectional:")
    ys_chunked, Bu_chunked, Lambda_chunked, xs_chunked = apply_ssm_chunked(
        Lambda_bar, B_bar, C_tilde, input_sequence,
        conj_sym=True, bidirectional=True, chunk_size=chunk_size
    )
    print(f"  xs_chunked shape: {xs_chunked.shape}")
    print(f"  xs_chunked[0:3, 0] (forward): {xs_chunked[0:3, 0]}")
    print(f"  xs_chunked[0:3, P] (backward): {xs_chunked[0:3, P]}")

    # Compare
    print("\n3. 对比:")
    diff_forward = jnp.max(jnp.abs(xs_orig[:, :P] - xs_chunked[:, :P]))
    diff_backward = jnp.max(jnp.abs(xs_orig[:, P:] - xs_chunked[:, P:]))
    print(f"  Forward 最大差异: {diff_forward:.2e}")
    print(f"  Backward 最大差异: {diff_backward:.2e}")

    if diff_backward > 1e-5:
        # 找出第一个不匹配的位置
        diff_mask = jnp.abs(xs_orig[:, P:] - xs_chunked[:, P:]) > 1e-5
        if jnp.any(diff_mask):
            indices = jnp.where(diff_mask)
            first_idx = (indices[0][0], indices[1][0])
            print(f"\n  第一个不匹配位置: {first_idx}")
            print(f"    Original: {xs_orig[first_idx[0], P + first_idx[1]]}")
            print(f"    Chunked:  {xs_chunked[first_idx[0], P + first_idx[1]]}")


def test_backward_carry_logic():
    """测试 backward carry 的逻辑"""
    print("\n" + "="*60)
    print("测试 backward carry 逻辑")
    print("="*60)

    P = 4
    chunk_size = 3

    Lambda_bar = jnp.ones(P, dtype=jnp.complex64) * 0.9

    # Chunk data: [0, 1, 2]
    Lambda_elements = Lambda_bar * jnp.ones((chunk_size, P))
    Bu_elements = jnp.array([[float(i)]*P for i in range(chunk_size)], dtype=jnp.complex64)

    print(f"\nChunk 数据 (indices 0,1,2):")
    print(f"  Bu: {Bu_elements[:,0]}")

    # 测试不同的 carry 处理方式
    carry_state = jnp.array([10.0]*P, dtype=jnp.complex64)  # 明显的 carry 值

    print(f"\n方法1: Prepend carry")
    Lambda_pre = jnp.concatenate([
        jnp.ones((1, P), dtype=Lambda_elements.dtype),
        Lambda_elements
    ], axis=0)
    Bu_pre = jnp.concatenate([
        carry_state[jnp.newaxis, :],
        Bu_elements
    ], axis=0)

    _, Bu_cum_pre = jax.lax.associative_scan(
        binary_operator, (Lambda_pre, Bu_pre), reverse=True
    )
    print(f"  Result: {Bu_cum_pre[:,0]}")
    print(f"  Bu_cum[0] (carry): {Bu_cum_pre[0,0]}")
    print(f"  Bu_cum[1:4] (states): {Bu_cum_pre[1:4,0]}")

    print(f"\n方法2: Append carry")
    Lambda_app = jnp.concatenate([
        Lambda_elements,
        jnp.ones((1, P), dtype=Lambda_elements.dtype)
    ], axis=0)
    Bu_app = jnp.concatenate([
        Bu_elements,
        carry_state[jnp.newaxis, :]
    ], axis=0)

    _, Bu_cum_app = jax.lax.associative_scan(
        binary_operator, (Lambda_app, Bu_app), reverse=True
    )
    print(f"  Result: {Bu_cum_app[:,0]}")
    print(f"  Bu_cum[0:3] (states): {Bu_cum_app[0:3,0]}")
    print(f"  Bu_cum[3] (carry): {Bu_cum_app[3,0]}")

    print(f"\n对比无 carry 的 reverse scan:")
    _, Bu_cum_plain = jax.lax.associative_scan(
        binary_operator, (Lambda_elements, Bu_elements), reverse=True
    )
    print(f"  Plain reverse: {Bu_cum_plain[:,0]}")

    print(f"\n哪个方法正确？")
    print("  如果 carry 来自右边（index 3），我们期望:")
    print("  result[2] = 0.9*carry + 2 = 0.9*10 + 2 = 11")
    print("  result[1] = 0.9*result[2] + 1 = 0.9*11 + 1 = 10.9")
    print("  result[0] = 0.9*result[1] + 0 = 0.9*10.9 + 0 = 9.81")


if __name__ == "__main__":
    test_reverse_scan_behavior()
    test_backward_carry_logic()
    test_chunked_backward_simple()

    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)