#!/usr/bin/env python3
"""
诊断 xs_backward_chunks 的 reshape 和 flip 过程
"""

import jax
import jax.numpy as jnp
import sys

sys.path.append('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from s5.ssm import binary_operator
from functools import partial
from jax import remat


def trace_backward_chunks_processing():
    """追踪 backward chunks 的处理和重组"""
    print("="*60)
    print("追踪 backward chunks 的处理")
    print("="*60)

    # 简单参数
    L, P, H = 6, 4, 8  # 超小尺寸便于追踪
    chunk_size = 3
    n_chunks = 2

    Lambda_bar = jnp.ones(P, dtype=jnp.complex64) * 0.9
    B_bar = jnp.ones((P, H), dtype=jnp.complex64) * 0.1

    # 输入：每个位置的值等于其索引
    input_sequence = jnp.arange(L)[:, None] * jnp.ones((L, H), dtype=jnp.float32)
    print(f"\n输入序列（每行的值）: {jnp.arange(L)}")
    print(f"  Chunk 0: indices [0, 1, 2]")
    print(f"  Chunk 1: indices [3, 4, 5]")

    # Reshape to chunks
    chunks = input_sequence.reshape(n_chunks, chunk_size, H)
    print(f"\nChunks shape: {chunks.shape}")

    # 定义 scan_chunk_backward（模拟当前实现）
    def scan_chunk_backward(carry_state, chunk):
        """模拟当前的 scan_chunk_backward"""
        Lambda_elements = Lambda_bar * jnp.ones((chunk_size, P))
        Bu_elements = jax.vmap(lambda u: B_bar @ u)(chunk)

        # Append carry
        Lambda_with_carry = jnp.concatenate([
            Lambda_elements,
            jnp.ones((1, P), dtype=Lambda_elements.dtype)
        ], axis=0)

        Bu_with_carry = jnp.concatenate([
            Bu_elements,
            carry_state[jnp.newaxis, :]
        ], axis=0)

        # Reverse scan
        _, Bu_cum = jax.lax.associative_scan(
            binary_operator,
            (Lambda_with_carry, Bu_with_carry),
            reverse=True
        )

        # Extract
        xs = Bu_cum[:-1]
        new_carry = Bu_cum[0]

        print(f"    Chunk Bu values (input): {Bu_elements[:,0]}")
        print(f"    Carry (appended): {carry_state[0]}")
        print(f"    Bu_cum after scan: {Bu_cum[:,0]}")
        print(f"    Extracted xs: {xs[:,0]}")
        print(f"    New carry: {new_carry[0]}")

        return new_carry, xs

    # Process chunks in reverse order
    print(f"\n处理 chunks (反向顺序):")

    init_carry = jnp.zeros(P, dtype=jnp.complex64)
    flipped_chunks = jnp.flip(chunks, axis=0)

    print(f"\nFlipped chunks order:")
    print(f"  flipped_chunks[0] 是 chunk_1 (indices 3,4,5)")
    print(f"  flipped_chunks[1] 是 chunk_0 (indices 0,1,2)")

    # 手动处理第一个（chunk_1）
    print(f"\n--- 处理 flipped_chunks[0] (chunk_1, indices 3,4,5) ---")
    carry1, xs1 = scan_chunk_backward(init_carry, flipped_chunks[0])

    # 手动处理第二个（chunk_0）
    print(f"\n--- 处理 flipped_chunks[1] (chunk_0, indices 0,1,2) ---")
    carry2, xs0 = scan_chunk_backward(carry1, flipped_chunks[1])

    # 使用 scan
    print(f"\n使用 jax.lax.scan:")
    _, xs_backward_chunks = jax.lax.scan(
        scan_chunk_backward,
        init_carry,
        flipped_chunks
    )

    print(f"\nxs_backward_chunks shape: {xs_backward_chunks.shape}")
    print(f"  xs_backward_chunks[0,:,0] (chunk_1 results): {xs_backward_chunks[0,:,0]}")
    print(f"  xs_backward_chunks[1,:,0] (chunk_0 results): {xs_backward_chunks[1,:,0]}")

    # Reshape
    xs_reshaped = xs_backward_chunks.reshape(L, P)
    print(f"\nReshape to ({L}, {P}):")
    print(f"  xs_reshaped[:,0] = {xs_reshaped[:,0]}")
    print(f"  Should be: [chunk_1_results(3,4,5), chunk_0_results(0,1,2)]")

    # Flip
    xs_final = jnp.flip(xs_reshaped, axis=0)
    print(f"\nFlip axis=0:")
    print(f"  xs_final[:,0] = {xs_final[:,0]}")
    print(f"  Should be: [chunk_0_results(0,1,2), chunk_1_results(3,4,5)]")

    # 对比原始 reverse scan
    print(f"\n对比原始 reverse scan (全序列):")
    Lambda_elements_full = Lambda_bar * jnp.ones((L, P))
    Bu_elements_full = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, Bu_cum_full = jax.lax.associative_scan(
        binary_operator,
        (Lambda_elements_full, Bu_elements_full),
        reverse=True
    )
    print(f"  Original reverse scan: {Bu_cum_full[:,0]}")

    # 比较
    print(f"\n结果对比:")
    print(f"  Chunked:  {xs_final[:,0]}")
    print(f"  Original: {Bu_cum_full[:,0]}")
    print(f"  Diff: {jnp.abs(xs_final[:,0] - Bu_cum_full[:,0])}")


if __name__ == "__main__":
    trace_backward_chunks_processing()

    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)