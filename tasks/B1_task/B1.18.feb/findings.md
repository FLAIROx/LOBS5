# Findings — B1 多节点修复

## 1. pmap 在 multi-host 下的根本限制

- XLA autotuner 在 multi-host pmap 编译时 crash
- 错误: `device_type:"DEVICE_TYPE_INVALID"` in `autotuner.cc:260`
- 即使移除所有 XLA flags（triton_gemm、command_buffer 等）仍然 crash
- 单节点完全正常 (job 2356422)
- 结论: **pmap 的 multi-host 路径有 XLA bug，必须用 jit+sharding**

## 2. ssm_stable 的 jit+sharding 方案

- ssm_stable 用 `jax.jit + NamedSharding` 替代 `jax.pmap + pmean`
- Mesh 目前用 `jax.local_devices()`（节点内同步），不是全局 `jax.devices()`
- 关键文件: `sharding_utils.py` (Mesh 创建), `train_helpers.py` (create_jit_train_step)
- ssm_stable 的 batch 脚本不设任何 --xla_gpu_* 优化 flags
- ssm_stable 的 run_train.py 处理 jax.distributed.initialize + shutdown

## 3. ssm_stable 包含的额外功能（B1 不需要）

| 功能 | 说明 | 是否需要 |
|------|------|----------|
| BF16 混合精度 | `use_bf16=True` 默认开启 | ❌ B1 不用 |
| TBPTT ar_hidden | 截断反向传播 | ❌ B1 不用 |
| SWR v3 | 滑动窗口循环 | ❌ B1 不用 |
| MODEL_PRESET 系统 | 55M-2.5B 预设 | ❌ B1 手动配置 |
| GoodputMonitor | MFU 跟踪 | ❌ 额外依赖 |
| profiling_utils | 性能分析 | ❌ 额外依赖 |
| optax LR schedule | 替代手动 per-step | ⚠️ 不同范式 |

## 4. B1 的 19 个功能完整列表

见 `features.md`。核心独有改动:
- in-place offset 更新 (persistent_workers 前提)
- drop_last=True with sampler (防 XLA recompile)
- checkpoint 失败 exit(1) (防僵尸 job)
- 移除 jax.clear_caches() (防 OOM)
- b_seq_eval 索引修正 (bug fix)

## 5. 模型配置一致性

ssm_stable `MODEL_PRESET=55M` = B1 "75M":
- d_model=1024, n_layers=12, blocks=16, ssm_size_base=1024
- 唯一不同: ssm_lr_base (ssm_stable: 5e-5, B1: 5e-4)

## 6. 关键 commit 节点

| Commit | 描述 | 状态 |
|--------|------|------|
| 41d336b | B1 干净 base (verified test data path) | ✅ 用户选择从这里分支 |
| a493683 | 恢复 jax.distributed.initialize | 可参考 |
| e12e10f | 添加 sync barrier | 可参考 |
| ec41df4 | 移除 XLA flags (不够) | 已被证明不够 |
| 3c78c2f | ssm_stable 整体替换 (错误做法) | ❌ 需要废弃 |
| fda0372 | 去掉 clear_caches + BSZ=4 | ✅ pipeline 通过，⚠️ caveat |

## 7. 跨 Epoch OOM 的完整诊断（2026-02-18）

### 根因
jit+sharding 模式下 `train_step` 需要在单个 GPU 的 BFC pool 中分配 **71.62 GiB 连续块**。
`eval_step`（不同形状）运行后造成 BFC pool 碎片化，第 3 个 epoch 的 `train_step` 在碎片化的 pool 里找不到足够的连续空间。

### 关键观察：Epoch 3 临界点
- BSZ=8 时：Epoch 1 ✅ → Epoch 2 ✅ → Epoch 3 ❌ OOM
- 这是**离散跳变**，不是线性积累——每个 epoch 末的 `deduplicate_trainstate + checkpoint + eval` 循环在 BFC pool 中制造特定模式的碎片
- 经过两轮完整的 train+eval+checkpoint 之后，BFC pool 的碎片化程度首次达到无法满足 71.62 GiB 连续请求的临界

### 所有尝试的对比

| 方案 | 失败 Epoch | 失败原因 |
|------|-----------|---------|
| 无修改 | Epoch 2 | BFC 碎片化，71.62 GiB 找不到 |
| jax.clear_caches() | Epoch 3 | 每 epoch 重编译（193s税）+ NCCL clique 积累 |
| TF_GPU_ALLOCATOR=cuda_malloc_async | Epoch 3 | TF 变量，对 JAX/XLA BFC 无效 |
| del jit + recreate | Epoch 3 | 同上，NCCL clique 仍积累 |
| XLA_PYTHON_CLIENT_PREALLOCATE=false | Epoch 2 | CUDA 地址空间更碎片，更差 |
| **BSZ=4，无 clear_caches** | **无 OOM** | workspace 71→35 GiB，碎片池仍可满足 |

### Epoch 3 OOM 的精确时间序列证据

从 job 2358257 (CURTAIL_EPOCHS=10, clear_caches) 的 log 中观察到：

```
Epoch 1 train step 0:   0%| | 1/1097 [00:43<..., 43.80s/it]  ← JIT 编译（193s 总计）
Epoch 1 train step 1:   0%| | 2/1097 [00:43<..., 18.11s/it]  ← 后续步骤 <0.5s
...
Epoch 1 val step 0:     0%| | 1/679  [00:08<...,  8.26s/it]  ← eval JIT 编译
Epoch 1 val step 1:     0%| | 2/679  [00:08<...,  3.48s/it]  ← 后续步骤快
Epoch 2 train step 0:   0%| | 1/1097 [00:32<..., 32.62s/it]  ← 再次 JIT 编译（clear_caches 生效）
...
Epoch 2 val step 0:     0%| | 1/679  [00:00<...,  2.36it/s]  ← 极快，复用 eval 编译
Epoch 3 train step 0:   E0218 ... RESOURCE_EXHAUSTED ...      ← 立刻 OOM，无编译 warmup
```

**关键推断**：Epoch 3 的 `train_step` 没有 193s warmup 就直接 OOM，说明：
- jax.clear_caches() 清除了 Python 层 trace 缓存
- 但 XLA C++ 层的 compiled executable 仍然存活（没有被真正释放）
- Epoch 3 复用了 XLA C++ 缓存的 executable（所以无需重新编译）
- 但 **执行时**找不到 71.62 GiB 的连续内存 → OOM

### CURTAIL_EPOCHS=10 的时间分布（BSZ=8 + clear_caches）

| 阶段 | 耗时 |
|------|------|
| train step 0（JIT 编译） | ~193s |
| train steps 1-9（实际计算） | ~5s |
| eval step 0（JIT 编译） | ~43s |
| eval steps 1-9（实际计算） | ~2s |
| test steps 1-10（复用 eval）| ~1s |
| checkpoint + gc.collect | ~20s |
| **合计（per epoch）** | **~265s ≈ 4.4 分钟** |

20 epochs × 4.4 分钟 = **88 分钟**，远超 30 分钟 job 时限。
这也是为什么 CURTAIL_EPOCHS=10 无法完成 20 epoch 的根本原因（与 OOM 问题叠加）。

去掉 jax.clear_caches() 后（BSZ=4）：
- JIT 只在 Epoch 1 编译一次（193s）
- 后续 epoch：train ~5s + eval ~2s + ckpt ~20s ≈ **27s/epoch**
- 20 epochs 总计：193 + 19×27 ≈ **706s ≈ 12 分钟** ✅（job 2358280 实测 13 分 23 秒）

### 当前状态（Caveat）
- BSZ=4 是**临时绕过**，不是真正的修复
- 减半 BSZ 影响训练效率和全局 batch size 语义
- **用户确认**：BSZ=8 之前是可以通过的，这是从 pmap→jit+sharding 迁移引入的新问题

## 8. 未来方向：让 BSZ=8 恢复工作

### 方向 A：remat（gradient checkpointing）
- 在反向传播中**重新计算激活**而不是保存，直接减少 workspace 峰值需求
- 预计可从 71.62 GiB → ~40 GiB，同时保持 BSZ=8
- JAX 实现：`jax.remat` / `flax.linen.remat`，wrapping SSM scan 层

### 方向 B：Epoch 边界 BFC 整理
- 在 checkpoint 保存之后、下一 epoch 开始之前：
  ```python
  jax.block_until_ready(state)  # 等所有异步 GPU 操作完成
  gc.collect()                   # 释放 Python 对象（含 checkpoint）
  # 然后进入下一 epoch
  ```
- 目前 `gc.collect()` 在 wandb 日志之前，可能 checkpoint buffers 还没完全释放

### 方向 C：分析 BFC 碎片结构
- 设置 `XLA_FLAGS='--xla_dump_to=/tmp/xla_dump'` 在 OOM 前获取 BFC allocator dump
- 了解 71.62 GiB 里具体哪些 buffer 不能被释放/合并
- 基于分析结果定向优化

### 方向 D：SSM parallel scan 的内存优化
- SSM 的 associative scan 是主要内存消耗来源（O(seq_len × state_size × batch) 中间状态）
- 可以探索 `scan_axis` 配置或分块扫描来降低峰值内存
