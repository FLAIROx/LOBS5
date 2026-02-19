# ssm_stable 分支：跨节点梯度同步 Bug 修复指南

> 创建日期: 2026-02-18
> 来源: B1 分支调试发现（commit 9c64e1c 修复了 B1）
> 状态: **待修复** — ssm_stable 尚未修改

---

## Bug 描述

`ssm_stable` 分支的 `sharding_utils.py` (或等效代码) 使用 `jax.local_devices()` 创建 Mesh，
导致**每个 node 独立训练**，无跨节点梯度同步。

```
当前行为:  Node 0 独立训练 ←→ Node 1 独立训练 (模型参数立即分叉)
正确行为:  Node 0 + Node 1 → 全局 mesh → 自动 allreduce 梯度 (1 个统一模型)
```

## 代码证据

### sharding_utils.py (或 ssm_stable 等效位置)
```python
# BUG: local mesh, 无跨节点同步
local_devs = jax.local_devices()
devices = local_devs[:num_devices]
mesh = Mesh(devices, ('data',))

# 注释甚至明确写了:
# "Gradient sync across nodes would require psum across processes (not yet implemented)"
```

## B1 的修复 (commit 9c64e1c)

### sharding_utils.py
```python
if jax.process_count() > 1:
    devices = jax.devices()  # 全局: 所有节点的所有 GPU
else:
    devices = jax.local_devices()[:num_devices]  # 单机: 本地 GPU

devices_array = np.array(devices).reshape(-1)
mesh = Mesh(devices_array, axis_names=('data',))
```

### train.py (mesh 初始化)
```python
if jax.process_count() > 1:
    mesh = initialize_mesh(jax.device_count())  # 全局
else:
    mesh = initialize_mesh(args.num_devices)    # 单机
```

### train_helpers.py (device_put → make_array_from_process_local_data)
```python
# BUG: device_put 在全局 mesh 下把 local batch 当全局数组
inputs = tuple(jax.device_put(inp, sh) for inp, sh in zip(inputs, inputs_sh))
# → 28 样本 / 8 devices = 3.5 ❌ ValueError

# FIX: 每个 process 提供本地分片，JAX 拼装全局数组
inputs = tuple(jax.make_array_from_process_local_data(sh, inp) for inp, sh in zip(inputs, inputs_sh))
# → 每 process 28 样本映射到本地 4 devices，全局 56 样本 / 8 devices = 7 ✅
```

## ssm_stable 需要修改的文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `sharding_utils.py` (或等效) | `jax.local_devices()` → `jax.devices()` when multi-node | 核心修复 |
| `train.py` | mesh 初始化传 `jax.device_count()` | 配套 |
| `train_helpers.py` | `jax.device_put` → `jax.make_array_from_process_local_data` | **必须**，否则 ValueError |
| Orbax checkpoint | **可能需要** 所有 rank 创建 CheckpointManager | 全局 mesh 可能触发 Orbax barrier |

## 注意事项

1. **Orbax checkpoint**: 全局 mesh 下 Orbax 可能在 `save()` 内部使用分布式 barrier。
   如果只有 rank 0 创建 CheckpointManager，其他 rank 不参与 → **死锁**。
   B1 的 `deduplicate_trainstate` 先转为本地数组再存，可能规避此问题。
   **需实测确认**。如果 hang，修复：所有 rank 创建 ckpt_mgr + 所有 rank 调用 save。

2. **LR Scaling**: 修复梯度同步后，global batch size 真正生效。
   之前"多节点训练"的 LR 是为单节点有效 BSZ 调的，修复后需要重新 sweep LR。

3. **`prep_batch` 里的 num_devices**: 已确认不影响（jit+sharding 路径不使用该参数）。

4. **JAX_COMPILATION_CACHE_DIR**: 修复后 HLO 与之前完全不同（有全局 allreduce），
   **必须清理旧 cache** 或使用新的 cache 目录。

## 验证方法

1. 提交 2 节点 job，检查 log 中的 `[Sharding] Global mesh with N devices across M processes`
2. 对比修复前后 Epoch 1 Train Loss — 应该**不同**（修复前是独立训练，修复后有梯度平均）
3. 如果 checkpoint 阶段 hang → 说明 Orbax barrier 问题，需修复 ckpt_mgr 创建逻辑
