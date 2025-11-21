#!/usr/bin/env python3
"""
诊断 chunked associative scan 的 NaN 问题
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path
sys.path.append('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from s5.ssm import apply_ssm_original, apply_ssm_chunked, binary_operator


def diagnose_chunked_scan():
    """诊断 chunked scan 实现的问题"""

    print("="*60)
    print("诊断 Chunked Associative Scan NaN 问题")
    print("="*60)

    # 简单测试案例
    L, P, H = 300, 32, 64  # 小尺寸便于调试
    chunk_size = 100

    # 创建简单的测试数据（使用稳定的值避免数值问题）
    key = jax.random.PRNGKey(42)
    Lambda_bar = jnp.ones(P, dtype=jnp.complex64) * (0.9 + 0.1j)  # 稳定的复数
    B_bar = jnp.ones((P, H), dtype=jnp.complex64) * 0.1
    C_tilde = jnp.ones((H, P), dtype=jnp.complex64) * 0.1
    input_sequence = jnp.ones((L, H), dtype=jnp.float32)

    print("\n1. 输入数据检查:")
    print(f"  Lambda_bar: shape={Lambda_bar.shape}, dtype={Lambda_bar.dtype}")
    print(f"  Lambda_bar[0]: {Lambda_bar[0]}")
    print(f"  B_bar: shape={B_bar.shape}, dtype={B_bar.dtype}")
    print(f"  C_tilde: shape={C_tilde.shape}, dtype={C_tilde.dtype}")
    print(f"  input_sequence: shape={input_sequence.shape}, dtype={input_sequence.dtype}")

    # 测试 binary_operator
    print("\n2. 测试 binary_operator:")
    # 创建两个简单的输入
    q_i = (jnp.ones(P, dtype=jnp.complex64), jnp.ones(P, dtype=jnp.complex64))
    q_j = (jnp.ones(P, dtype=jnp.complex64) * 0.9, jnp.ones(P, dtype=jnp.complex64) * 0.1)

    print(f"  q_i: ({q_i[0].shape}, {q_i[1].shape})")
    print(f"  q_j: ({q_j[0].shape}, {q_j[1].shape})")

    result_bin = binary_operator(q_i, q_j)
    print(f"  binary_operator 结果: type={type(result_bin)}")
    if isinstance(result_bin, tuple):
        print(f"    元素个数: {len(result_bin)}")
        for i, r in enumerate(result_bin):
            print(f"    result[{i}]: shape={r.shape}, dtype={r.dtype}")
            print(f"      包含 NaN: {jnp.any(jnp.isnan(r))}")

    # 手动执行第一个 chunk 来调试
    print("\n3. 手动执行第一个 chunk:")

    n_chunks = L // chunk_size
    chunks = input_sequence.reshape(n_chunks, chunk_size, H)
    print(f"  Chunks shape: {chunks.shape} (n_chunks={n_chunks})")

    # 初始 carry
    init_carry = jnp.zeros(P, dtype=Lambda_bar.dtype)  # 使用 Lambda_bar.dtype
    print(f"  init_carry: shape={init_carry.shape}, dtype={init_carry.dtype}")

    # 处理第一个 chunk
    chunk = chunks[0]
    print(f"  chunk shape: {chunk.shape}")

    # 计算 Lambda 和 Bu elements
    Lambda_elements = Lambda_bar * jnp.ones((chunk_size, P))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(chunk)

    print(f"  Lambda_elements: shape={Lambda_elements.shape}, dtype={Lambda_elements.dtype}")
    print(f"  Bu_elements: shape={Bu_elements.shape}, dtype={Bu_elements.dtype}")
    print(f"  Lambda_elements[0,0]: {Lambda_elements[0,0]}")
    print(f"  Bu_elements[0,0]: {Bu_elements[0,0]}")

    # Prepend carry
    print("\n4. Prepend carry state:")
    Lambda_with_carry = jnp.concatenate([
        jnp.ones((1, P), dtype=Lambda_elements.dtype),
        Lambda_elements
    ], axis=0)

    Bu_with_carry = jnp.concatenate([
        init_carry[jnp.newaxis, :],
        Bu_elements
    ], axis=0)

    print(f"  Lambda_with_carry: shape={Lambda_with_carry.shape}, dtype={Lambda_with_carry.dtype}")
    print(f"  Bu_with_carry: shape={Bu_with_carry.shape}, dtype={Bu_with_carry.dtype}")
    print(f"  Lambda_with_carry[0,0]: {Lambda_with_carry[0,0]}")
    print(f"  Bu_with_carry[0,0]: {Bu_with_carry[0,0]}")

    # 调用 associative_scan
    print("\n5. 调用 associative_scan:")
    print(f"  输入是元组: (Lambda_with_carry, Bu_with_carry)")
    print(f"  形状: ({Lambda_with_carry.shape}, {Bu_with_carry.shape})")

    try:
        result = jax.lax.associative_scan(binary_operator, (Lambda_with_carry, Bu_with_carry))

        print(f"\n  返回值类型: {type(result)}")

        # 详细分析返回值结构
        if isinstance(result, tuple):
            print(f"  返回值是元组，包含 {len(result)} 个元素")

            for i, r in enumerate(result):
                print(f"\n  result[{i}]:")
                if isinstance(r, tuple):
                    print(f"    是元组，包含 {len(r)} 个元素")
                    for j, rr in enumerate(r):
                        if hasattr(rr, 'shape'):
                            print(f"    [{i}][{j}]: shape={rr.shape}, dtype={rr.dtype}")
                            print(f"            包含 NaN: {jnp.any(jnp.isnan(rr))}")
                            if not jnp.any(jnp.isnan(rr)):
                                print(f"            Min: {jnp.min(jnp.abs(rr)):.6f}, Max: {jnp.max(jnp.abs(rr)):.6f}")
                else:
                    if hasattr(r, 'shape'):
                        print(f"    shape={r.shape}, dtype={r.dtype}")
                        print(f"    包含 NaN: {jnp.any(jnp.isnan(r))}")
                        if not jnp.any(jnp.isnan(r)):
                            print(f"    Min: {jnp.min(jnp.abs(r)):.6f}, Max: {jnp.max(jnp.abs(r)):.6f}")
        else:
            print(f"  返回值不是元组！类型: {type(result)}")
            if hasattr(result, 'shape'):
                print(f"  shape: {result.shape}")

        # 尝试不同的解包方式
        print("\n6. 尝试不同的解包方式:")

        # 方式1：假设返回两个值
        try:
            val1, val2 = result
            print("  成功解包为两个值:")
            print(f"    val1: type={type(val1)}")
            print(f"    val2: type={type(val2)}")

            # 检查 val2 是否是元组（Lambda_cum, Bu_cum）
            if isinstance(val2, tuple):
                print(f"    val2 是元组，包含 {len(val2)} 个元素")
                Lambda_cum, Bu_cum = val2
                print(f"      Lambda_cum: shape={Lambda_cum.shape}")
                print(f"      Bu_cum: shape={Bu_cum.shape}")
                print(f"      Bu_cum 包含 NaN: {jnp.any(jnp.isnan(Bu_cum))}")
        except Exception as e:
            print(f"  解包失败: {e}")

    except Exception as e:
        print(f"  associative_scan 出错: {e}")
        import traceback
        traceback.print_exc()

    # 测试原始版本作为对比
    print("\n7. 测试原始版本作为对比:")
    try:
        ys_orig, Bu_orig, Lambda_orig, xs_orig = apply_ssm_original(
            Lambda_bar, B_bar, C_tilde, input_sequence,
            conj_sym=True, bidirectional=False
        )
        print(f"  原始版本成功!")
        print(f"  xs_orig shape: {xs_orig.shape}, dtype={xs_orig.dtype}")
        print(f"  xs_orig 包含 NaN: {jnp.any(jnp.isnan(xs_orig))}")

        # 检查 xs_orig 的类型
        print(f"  xs_orig 类型: {type(xs_orig)}")
        if isinstance(xs_orig, tuple):
            print("  注意：xs_orig 是元组！")
    except Exception as e:
        print(f"  原始版本出错: {e}")

    # 测试完整的 chunked 版本
    print("\n8. 测试完整 chunked 版本:")
    try:
        ys, Bu, Lambda, xs = apply_ssm_chunked(
            Lambda_bar, B_bar, C_tilde, input_sequence,
            conj_sym=True, bidirectional=False, chunk_size=chunk_size
        )
        print(f"  Chunked 版本成功!")
        print(f"  xs shape: {xs.shape}, dtype={xs.dtype}")
        print(f"  xs 包含 NaN: {jnp.any(jnp.isnan(xs))}")
        print(f"  ys 包含 NaN: {jnp.any(jnp.isnan(ys))}")
    except Exception as e:
        print(f"  Chunked 版本出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("诊断完成！请将上述输出发送给我进行分析。")
    print("="*60)


if __name__ == "__main__":
    print(f"JAX 版本: {jax.__version__}")
    print(f"设备: {jax.devices()}")
    diagnose_chunked_scan()