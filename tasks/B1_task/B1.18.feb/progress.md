# Progress Log — B1 多节点修复 (v2)

## Branch: `exp/B1-ignore-times-v2`

## Session: 2026-02-18

### 00:00 — 恢复上下文
- 从 forked session b1612c81 恢复
- 当前 branch: exp/B1-ignore-times, HEAD: 3c78c2f (ssm_stable 整体替换)

### 00:10 — 分析问题
- 确认 3c78c2f commit 引入了太多 ssm_stable 的功能 (BF16, TBPTT, MFU 等)
- 用户要求: 只加 jit+sharding，不要其他功能
- 用户要求: 从 41d336b 创建新 branch `exp/B1-ignore-times-v2`

### 00:20 — 创建规划文件 + 详细对比
- task_plan.md: 列出 5 个 phase
- findings.md: 记录所有发现
- pmap_to_jit_sharding_migration.md: 完整 12 章节对比分析
- four_issues_diagnosis.md: 4 个问题的诊断

### 00:30 — 创建 v2 分支
- `git branch exp/B1-ignore-times-v2 41d336b && git checkout exp/B1-ignore-times-v2`
- 从干净的 B1 base 开始

### 00:40 — Commit 1: cf1cc67 — 问题 2 (JAX cache + XLA flags)
- 删除 `JAX_COMPILATION_CACHE_DIR`, `JAX_DEFAULT_MATMUL_PRECISION`, `JAX_COMPILER_ENABLE_REMAT_PASS`
- 删除 `TF_GPU_ALLOCATOR`, 全部 `--xla_gpu_*` flags
- 降低 MEM_FRACTION 0.95→0.90
- **文件**: `train_full_autoreg.batch` (2+ / 17-)

### 00:45 — Commit 2: bf327c4 — 问题 3 (Barrier/sync/shutdown)
- 添加 `sync_global_devices("jax_distributed_init")` — init 后同步
- 添加 `sync_global_devices("end-of-train")` — 训练结束前同步
- 添加 `jax.distributed.shutdown()` — 清理 distributed runtime
- **文件**: `run_train.py` (17+ / 1-)

### 01:00 — Commit 3: 18f832d — 问题 4 (pmap → jit+sharding)
核心迁移，保留 B1 所有 19 个功能：

**新增**: `lob/sharding_utils.py` (136 行)
**修改**: `lob/train_helpers.py` + `lob/train.py`

关键改动:
- 删 `device_reshape`, `@pmap`, `pmean`, `jax_utils.replicate`
- 加 `create_jit_train_step()`, `create_jit_eval_step()`
- `train_epoch`/`validate` 加 mesh + device_put
- `loss[0]` → `loss` (scalar)

### 问题 1 (Orbax) — 暂不改
- local mesh 下不触发跨节点 barrier

## Commit 链

```
41d336b  fix(data): verified GOOG Jan 2023 test set path            ← v2 分支点
  └── cf1cc67  fix(batch): remove JAX cache + XLA flags
      └── bf327c4  fix(train): add sync barriers + shutdown
          └── 18f832d  refactor(train): pmap → jit+sharding          ← HEAD
```

### Phase 4: 验证

#### Job 2357062 — 2-node smoke test #1 ❌ FAILED
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=100, EPOCHS=20
- **结果**: Node1 完成 epoch 1 训练，但 checkpoint save crash
- **错误**: `deduplicate_trainstate` 的 `x[0]` 对 0-dim scalar 失败
- **根因**: pmap state 有 device dim，jit+sharding 没有
- **修复**: e1f4916 — 检测 sharding 属性来区分 pmap/jit 模式
- **log**: `logs_lobs5/training_2357062_node*.log`
- **好消息**: 训练循环本身跑通了（JIT编译、mesh、sync barrier 全部正常）

#### Job 2357532 — 2-node smoke test #2 (含 deduplicate fix)
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=100, EPOCHS=20
- **代码**: HEAD e1f4916 (含 deduplicate_trainstate 修复)
- **状态**: PENDING
- **log**: `logs_lobs5/lobs5_2357532.out` + `logs_lobs5/training_2357532_node*.log`

## Commit 链 (更新)

```
41d336b  fix(data): verified GOOG Jan 2023 test set path
  └── cf1cc67  fix(batch): remove JAX cache + XLA flags
      └── bf327c4  fix(train): add sync barriers + shutdown
          └── 18f832d  refactor(train): pmap → jit+sharding
              └── e1f4916  fix(checkpoint): deduplicate_trainstate  ← HEAD
```

#### Job 2357532 结果 ❌ FAILED
- **错误**: `deduplicate_trainstate` 中 `jax.device_put(state, device=jax.devices('gpu')[0])` crash
- **根因 A**: ckpt 构建在 `is_main_process` 之外 → rank 1 也执行了 deduplicate
- **根因 B**: `jax.devices('gpu')[0]` 可能指向全局 device 0（跨节点）
- **修复**: 491f092 — 两处改动：
  1. `train.py`: ckpt 构建移入 `if is_main_process` (对齐 ssm_stable)
  2. `init_train.py`: `device_get` → `device_put(local_devices[0])` (安全的拓扑无关实现)

#### Job 2358097 — 2-node smoke test #3
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=100, EPOCHS=20
- **代码**: HEAD 491f092
- **状态**: PENDING

## Commit 链 (更新)

```
41d336b  fix(data): verified GOOG Jan 2023 test set path
  └── cf1cc67  fix(batch): remove JAX cache + XLA flags
      └── bf327c4  fix(train): add sync barriers + shutdown
          └── 18f832d  refactor(train): pmap → jit+sharding
              └── 491f092  fix(checkpoint): deduplicate_trainstate v2  ← HEAD
```

#### Job 2358097 结果 — 部分成功 ⚠️
- Epoch 1 训练+验证+测试全部跑通 ✅
- Checkpoint save 通过了 ✅ (deduplicate fix 生效)
- 新错误: `train.py:390` — `learning_rate[0]` 对 scalar 失败
- 修复: 49c265d — 去掉 4 处 `learning_rate[0]` 的 `[0]`

#### Job 2358114 — 2-node smoke test #4
- **代码**: HEAD 49c265d
- **状态**: PENDING

## Commit 链 (更新)

```
41d336b  fix(data): verified GOOG Jan 2023 test set path
  └── cf1cc67  fix(batch): remove JAX cache + XLA flags
      └── bf327c4  fix(train): add sync barriers + shutdown
          └── 18f832d  refactor(train): pmap → jit+sharding
              └── 491f092  fix(checkpoint): deduplicate_trainstate v2
                  └── 49c265d  fix(train): lr[0] → lr scalar  ← HEAD
```

#### Job 2358114 结果 — 部分成功 ⚠️
- ✅ Epoch 1 训练+验证+测试全部跑通
- ✅ Checkpoint save 成功
- ✅ wandb logging 成功（lr 不再 crash）
- ✅ 两节点同步正常
- ❌ Epoch 2 训练开始时 OOM: `Out of memory while trying to allocate 71.62GiB`
- Node 1 先 OOM → abort → Node 0 NCCL connection lost
- 可能原因: epoch 1 的 eval/test 缓存未释放，epoch 2 train_step 分配不够
- wandb: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/sl4v6397

### OOM 调查结果 (ssm_stable 对比)

**核心差异**: ssm_stable 在 epoch 末尾调用 `jax.clear_caches()`，B1 注释掉了。
**但这不是根因**: B1 注释掉它是对的 — `clear_caches()` 会导致 recompile，recompile 本身更耗内存。

**真正原因**: validate 返回后，Python 变量引用的 JAX arrays（logits, preds, ce_means 等）仍占 GPU 内存。
加上 ckpt dict 持有一份 deduplicated state，epoch 间内存持续累积直到 GC 回收。

**修复方案**:
1. checkpoint save 后显式 `del ckpt`
2. validate 后显式 `del` 大变量
3. epoch 末尾 `gc.collect()` 确保 Python GC 回收

### OOM 修复迭代

#### Job 2358225 — 2-node smoke test #5 (070facd: clear_caches + MEM_FRACTION=0.80)
- ✅ Epoch 1-2 通过
- ❌ Epoch 3 同样 OOM (71.62 GiB)
- 根因深化：`jax.clear_caches()` 清 Python 级 trace 缓存，但 `jit_train_step` 对象仍持有 XLA executable GPU buffer 引用
- wandb: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/w7dourqk

#### Job 2358255 — 2-node smoke test #6 (caf58db: del jit + recreate) — 错误方向
- ssm_stable 根本没有 del/recreate jit 函数
- **废弃**

### 最终根因（2026-02-18 真正结论）

**ssm_stable 用 pmap，不用 jit+sharding。** pmap 每个设备独立分配内存，不需要单 GPU 上的 71.62 GiB 连续块。
`TF_GPU_ALLOCATOR` 是 TensorFlow 变量，对 JAX/XLA 无效。

**真正的多 epoch OOM 解法：**
1. **去掉 jax.clear_caches()** — 它每 epoch 触发重编译（193s税）和新 NCCL clique 积累
2. **降低 PER_GPU_BSZ 8→4** — workspace 从 71.62 GiB 降到 ~35.81 GiB，BFC 碎片化后仍能满足

#### Job 2358257 — ❌ Epoch 3 OOM (TF_GPU_ALLOCATOR 对 JAX 无效)

#### Job 2358266 — ❌ 脚本语法错误（注释里的 "can't" 单引号破坏 bash -c 块）

#### Job 2358268 — ❌ Epoch 2 OOM (PREALLOCATE=false 更差：CUDA 地址空间更碎片)

#### Job 2358280 — ✅ 2-node smoke test #最终 (fda0372: BSZ=4, no clear_caches)
- **代码**: HEAD fda0372
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=10, EPOCHS=20, **PER_GPU_BSZ=4**
- **关键**: 去掉 jax.clear_caches()，降低 BSZ
- **结果**: 全部 20 epoch 通过，无 OOM ✅
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/juv0tktc
- **Loss 趋势**: Train 7.64→2.24, Val 3.72→2.14, Acc 0.507→0.602

## 下一步
- [x] 修复跨 epoch OOM ✅ (PER_GPU_BSZ=4, no clear_caches)
- [ ] 用 BSZ=8 验证是否还 OOM（确认是否 BSZ 问题）
- [ ] 生产训练：大 epoch 数量
