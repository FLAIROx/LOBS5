#!/usr/bin/env python
"""
验证 jax.lax.scan + grad 是否能降低 XLA compilation memory

测试方法：
1. 创建简化的"假模型"（模拟 S5 结构）
2. 测试两种方式：
   A. 直接 grad（当前方式）- 处理完整序列
   B. scan + grad（TBPTT 方式）- 分块处理
3. 观察 XLA 编译内存警告的差异
"""

import jax
import jax.numpy as jnp
import os

# 确保能看到 XLA 内存警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

print("=" * 70)
print("XLA Compilation Memory Test: Direct Grad vs Scan+Grad")
print("=" * 70)

# ========== 测试配置 ==========
# 使用真实规模配置以触发编译内存警告
BSZ = 32          # Batch size（接近真实的 64，但稍小以避免 OOM）
D_MODEL = 1024    # Model dimension（真实大小）
N_LAYERS = 12     # Number of layers（真实大小）
FULL_SEQ_LEN = 12000  # 完整序列长度（真实大小！）
N_CHUNKS = 4      # 分块数量
CHUNK_SIZE = FULL_SEQ_LEN // N_CHUNKS  # 3000 per chunk

print(f"\nTest Configuration:")
print(f"  Batch Size: {BSZ}")
print(f"  Model Dimension: {D_MODEL}")
print(f"  Number of Layers: {N_LAYERS}")
print(f"  Full Sequence Length: {FULL_SEQ_LEN}")
print(f"  Number of Chunks: {N_CHUNKS}")
print(f"  Chunk Size: {CHUNK_SIZE}")
print("=" * 70)

# ========== 创建假模型参数 ==========
def create_fake_params(d_model, n_layers):
    """创建模拟 S5 模型的参数"""
    key = jax.random.PRNGKey(42)
    params = {}

    for i in range(n_layers):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        params[f'layer_{i}'] = {
            'W': jax.random.normal(subkey1, (d_model, d_model)) * 0.02,
            'b': jnp.zeros(d_model)
        }

    # 输出层
    key, subkey = jax.random.split(key)
    params['output'] = {
        'W': jax.random.normal(subkey, (d_model, d_model)) * 0.02,
        'b': jnp.zeros(d_model)
    }

    return params

# ========== 假的前向传播 ==========
def fake_forward(params, inputs):
    """
    模拟多层模型的前向传播
    结构类似：Encoder layers + Decoder
    """
    x = inputs  # (BSZ, seq_len, d_model)

    # 模拟多层 S5 encoder
    for i in range(N_LAYERS):
        layer_params = params[f'layer_{i}']
        # 矩阵乘法 + 激活
        x = jnp.dot(x, layer_params['W']) + layer_params['b']
        x = jax.nn.gelu(x)

    # 输出投影（模拟 decoder）
    logits = jnp.dot(x, params['output']['W']) + params['output']['b']
    return logits  # (BSZ, seq_len, d_model)

# ========== 创建测试数据 ==========
key = jax.random.PRNGKey(0)
params = create_fake_params(D_MODEL, N_LAYERS)

inputs_full = jax.random.normal(key, (BSZ, FULL_SEQ_LEN, D_MODEL))
labels_full = jax.random.normal(key, (BSZ, FULL_SEQ_LEN, D_MODEL))

# ========== 测试 A：直接 grad ==========
print("\n" + "=" * 70)
print("[TEST A] Direct Grad - Full Sequence")
print("=" * 70)
print(f"Processing sequence length: {FULL_SEQ_LEN}")
print("Compiling...")

@jax.jit
def train_step_direct(params, inputs, labels):
    """
    模拟当前的 train_step：
    一次性处理完整序列 L=2400
    """
    def loss_fn(p):
        logits = fake_forward(p, inputs)  # (BSZ, 2400, D_MODEL)
        loss = jnp.mean((logits - labels)**2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

# 触发编译
try:
    loss_direct, grads_direct = train_step_direct(params, inputs_full, labels_full)
    print(f"✓ Compilation successful")
    print(f"  Loss: {loss_direct:.6f}")

    # 计算梯度 norm
    grad_norm_direct = jnp.sqrt(sum(
        jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads_direct)
    ))
    print(f"  Gradient norm: {grad_norm_direct:.6f}")

except Exception as e:
    print(f"✗ Compilation failed: {e}")

# ========== 测试 B：scan + grad ==========
print("\n" + "=" * 70)
print("[TEST B] Scan + Grad - Chunked Sequence (TBPTT)")
print("=" * 70)
print(f"Processing {N_CHUNKS} chunks of size {CHUNK_SIZE} each")
print("Compiling...")

# 准备分块数据
inputs_chunked = inputs_full.reshape(BSZ, N_CHUNKS, CHUNK_SIZE, D_MODEL)
inputs_chunked = inputs_chunked.transpose(1, 0, 2, 3)  # (4, BSZ, 600, D_MODEL)

labels_chunked = labels_full.reshape(BSZ, N_CHUNKS, CHUNK_SIZE, D_MODEL)
labels_chunked = labels_chunked.transpose(1, 0, 2, 3)

@jax.jit
def train_step_scan(params, inputs_chunked, labels_chunked):
    """
    模拟 TBPTT train_step：
    使用 scan 遍历 chunks，每次只处理 chunk_size=600
    """

    def compute_chunk_grad(carry, chunk_data):
        grads_accum, loss_accum = carry
        inputs_chunk, labels_chunk = chunk_data

        # ========== 关键：定义 chunk 的 loss ==========
        # 这个 loss_fn 只看到 chunk_size=600
        # XLA 只编译这个大小的图！
        def loss_fn_chunk(p):
            logits = fake_forward(p, inputs_chunk)  # (BSZ, 600, D_MODEL)
            loss = jnp.mean((logits - labels_chunk)**2)
            return loss

        # 计算这个 chunk 的梯度
        loss_chunk, grads_chunk = jax.value_and_grad(loss_fn_chunk)(params)

        # 累积梯度
        grads_accum = jax.tree_util.tree_map(lambda a, g: a + g, grads_accum, grads_chunk)
        loss_accum = loss_accum + loss_chunk

        return (grads_accum, loss_accum), None

    # 初始化累积器
    init_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    init_loss = jnp.array(0.0)

    # ========== 关键：使用 scan 遍历 chunks ==========
    # scan 生成 while loop，不展开
    # XLA 只编译 scan body（处理一个 chunk）
    (final_grads, final_loss), _ = jax.lax.scan(
        compute_chunk_grad,
        (init_grads, init_loss),
        (inputs_chunked, labels_chunked),
        length=N_CHUNKS
    )

    # 平均梯度和 loss
    loss = final_loss / N_CHUNKS
    grads = jax.tree_util.tree_map(lambda g: g / N_CHUNKS, final_grads)

    return loss, grads

# 触发编译
try:
    loss_scan, grads_scan = train_step_scan(params, inputs_chunked, labels_chunked)
    print(f"✓ Compilation successful")
    print(f"  Loss: {loss_scan:.6f}")

    # 计算梯度 norm
    grad_norm_scan = jnp.sqrt(sum(
        jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads_scan)
    ))
    print(f"  Gradient norm: {grad_norm_scan:.6f}")

except Exception as e:
    print(f"✗ Compilation failed: {e}")

# ========== 验证等价性 ==========
print("\n" + "=" * 70)
print("[VERIFICATION] Gradient Equivalence Check")
print("=" * 70)

# 对比梯度
grad_diff = jax.tree_util.tree_map(
    lambda a, b: jnp.max(jnp.abs(a - b)),
    grads_direct,
    grads_scan
)
max_diff = max(jax.tree_util.tree_leaves(grad_diff))

print(f"Loss difference: {abs(loss_direct - loss_scan):.2e}")
print(f"Max gradient difference: {max_diff:.2e}")

if max_diff < 1e-4:
    print("✅ Gradients are equivalent!")
    print("   → TBPTT (scan+grad) produces correct gradients")
else:
    print(f"⚠️  Gradients differ by {max_diff:.2e}")
    print("   → May indicate numerical issues or implementation error")

# ========== 最终结论 ==========
print("\n" + "=" * 70)
print("🎯 CONCLUSION")
print("=" * 70)
print("\n📊 Check the warnings above for 'Can't reduce memory':")
print("\n   If Test B has LOWER memory warning than Test A:")
print("   → ✅ TBPTT with scan CAN reduce XLA compilation memory")
print("   → ✅ Safe to implement in main code")
print("\n   If Test B has SAME memory warning as Test A:")
print("   → ❌ scan+grad does NOT reduce compilation memory")
print("   → ❌ Need to find alternative solution")
print("\n" + "=" * 70)
