# Progress Log — B1 多节点修复 (v2)

## Branch: `exp/B1-ignore-times-v2`

---

## 背景：2026-02-14 长训练（单节点）

**Job 2316200** — `exp/B1-ignore-times` 单节点完整训练
- **Branch**: `exp/B1-ignore-times`，commit `ed88579`
- **配置**: 1 node, 4 GPU, epochs=40, BSZ=32
- **耗时**: 18 小时 20 分钟，后被取消
- **wandb**: kang-oxford/lobs5-75M-B1/u3e28srp (State: Crashed)
- **问题**: Test Loss 持续上涨，Test Accuracy 不稳定
  - 怀疑原因: scale up 时 Global Batch Size 变大但 LR 未调整
  - 另一怀疑: test data path 处理逻辑有问题（后来验证是 test_dir_name 路径错误）
- **结论**: 触发对 B1 多节点训练的全面调查

---

## 2026-02-17：初次多节点尝试（`exp/B1-ignore-times`，pmap 路径）

这批 job 在 B1 原始分支（pmap 实现）上尝试 2-node 训练，逐步发现并修复了多个 bug。

### Job 2355636 — 2-node 测试 #0（未启动）
- **状态**: CANCELLED before starting（排队时取消）
- **无日志**

### Job 2355647 — 2-node 测试 #1 ❌ NCCL hang（17 分钟）
- **配置**: 2 nodes (nid010025-010026), 8 GPU
- **症状**: 两节点 NCCL 通信 hang，17 分钟后手动 scancel
- **根因**: `jax.distributed.initialize()` 后缺少全局 barrier，node 0 和 node 1 步调不一致
- **修复**: commit `e12e10f` — 添加 `sync_global_devices("jax_distributed_init")`
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/rc8jd7y9
- **log**: `logs_lobs5/training_2355647_node*.log`

### Job 2355695 — 2-node 测试 #2 ❌ 重复测试（2 分 47 秒）
- **配置**: 2 nodes (nid010064-010065), 8 GPU
- **状态**: CANCELLED after 2:47
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/mu6nh9hv
- **说明**: 调试中的短暂测试，确认问题重现

### Job 2355708 — 2-node 测试 #3 ❌ NCCL hang 重试（2 分 45 秒）
- **配置**: 2 nodes (nid010025-010026), 8 GPU
- **状态**: CANCELLED after 2:45，同 2355647 问题
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/k8b0h6ww

### Job 2356386 — 2-node 测试 #4 ❌ Orbax barrier mismatch（3 分 36 秒）
- **配置**: 2 nodes (nid010087-010088), 8 GPU
- **症状**: Orbax CheckpointManager 内部调用了 distributed barrier，但只有 rank 0 创建了 CheckpointManager，rank 1 没有参与，导致 barrier 不匹配 hang/crash
- **修复**: commit `37a7ae4` — 所有 rank 都创建 CheckpointManager，满足 Orbax 的 barrier 要求
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/j35agw6y
- **log**: `logs_lobs5/training_2356386_node*.log`

### Jobs 2356395, 2356410, 2356445, 2356470 — 2-node 测试 #5-8 ❌ XLA autotuner crash（各约 3 分钟）
- **配置**: 2 nodes, 8 GPU，每个 job 约 3 分钟
- **错误**: `Sharding autotuning failed: device_type:"DEVICE_TYPE_INVALID"` in `autotuner.cc:260`
- **即使移除了所有 XLA flags**（triton_gemm、command_buffer 等）仍然 crash
- **结论**: pmap 的 multi-host 路径有 XLA bug，**无法通过调参绕过，必须换 jit+sharding**
- **wandb**:
  - 2356395: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/s6688tzc
  - 2356410: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/4evd8g2n
  - 2356445: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/d2yf359q
  - 2356470: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/ak4ltd79
- **log**: `logs_lobs5/training_235639*/training_235644*/training_235647*_node*.log`

### Job 2356422 — 单节点对照测试 ✅（6 分 48 秒）
- **配置**: 1 node (nid010001), 4 GPU
- **结果**: 单节点完全正常，Epoch 1 训练通过，JIT 编译正常，无 autotuner 错误
- **结论**: XLA autotuner bug 只在 multi-host pmap 路径触发，单节点 pmap 不受影响
- **注意**: log 里有 `CUDA_ERROR_NO_DEVICE` 噪音，这是 force_cpu 数据加载 worker 尝试检测 GPU 时的预期错误，不影响训练
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/h465acsd

### Job 2356765 — ssm_stable 直接移植测试 ❌（31 秒，1 node）
- **配置**: 1 node (nid010961), commit `3c78c2f`（ssm_stable 整体替换版本）
- **目的**: 验证直接用 ssm_stable 的 run_train.py 能否在 B1 环境跑起来
- **错误 1**: `ModuleNotFoundError: No module named 'lob.profiling_utils'`
  - ssm_stable 有 GoodputMonitor 依赖，B1 没有这个模块
- **错误 2**: `local_device_ids=[0,1,2,3,4,5,6,7]`，但节点只有 4 个 GPU
  - ssm_stable 的 run_train.py 硬编码了 8 个 local_device_ids（为 2×4 GPU 节点设计）
  - `W: Allowed device set contains 8 devices, but platform only sees 4`
- **结论**: ssm_stable 直接移植不可行，需要手术式迁移
- **耗时**: 31 秒即 import 阶段失败
- **log**: `logs_lobs5/training_2356765_node0.log`

---

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
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/urht0vqs
- **log**: `logs_lobs5/training_2357062_node*.log`
- **好消息**: 训练循环本身跑通了（JIT编译、mesh、sync barrier 全部正常）

#### Job 2357532 — 2-node smoke test #2 (含 deduplicate fix) ❌
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=100, EPOCHS=20
- **代码**: HEAD e1f4916 (含 deduplicate_trainstate 修复)
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/6mpazbbl
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

#### Job 2358097 — 2-node smoke test #3 ⚠️ 部分成功
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=100, EPOCHS=20
- **代码**: HEAD 491f092
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/kl26zrqw

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
- **代码**: HEAD caf58db
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=10, EPOCHS=20, MEM_FRACTION=0.90
- **假设**: 每 epoch 末 `del jit_train_step, jit_eval_step` → `gc.collect()` → `jax.clear_caches()` → 重建 JIT，以释放旧 XLA executable 占用的 GPU 内存
- **实际**: Epoch 1 ✅，Epoch 2 ✅，Epoch 3 step 0 ❌ 同样 OOM 71.62 GiB
- **ssm_stable 从未这样做**，是错误方向
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/ymzh0jm3
- **log**: `logs_lobs5/training_2358255_node1.log`

### 最终根因（2026-02-18 真正结论）

**ssm_stable 用 pmap，不用 jit+sharding。** pmap 每个设备独立分配内存，不需要单 GPU 上的 71.62 GiB 连续块。
`TF_GPU_ALLOCATOR` 是 TensorFlow 变量，对 JAX/XLA 无效。

**真正的多 epoch OOM 解法：**
1. **去掉 jax.clear_caches()** — 它每 epoch 触发重编译（193s税）和新 NCCL clique 积累
2. **降低 PER_GPU_BSZ 8→4** — workspace 从 71.62 GiB 降到 ~35.81 GiB，BFC 碎片化后仍能满足

#### Job 2358257 — 2-node smoke test #7 (84c52ba: TF_GPU_ALLOCATOR=cuda_malloc_async) ❌
- **代码**: HEAD 84c52ba
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=10, EPOCHS=20, MEM_FRACTION=0.90
- **假设**: `TF_GPU_ALLOCATOR=cuda_malloc_async` 使用 CUDA 异步分配器代替 BFC，从根本上避免碎片化
- **实际**: Epoch 1 ✅，Epoch 2 ✅，Epoch 3 step 0 ❌ 同样 OOM 71.62 GiB
- **根因**: `TF_GPU_ALLOCATOR` 是 TensorFlow 的环境变量，JAX/XLA 的 BFC allocator 完全不读取它
- **Train Loss 一致性**: 所有失败 job 的 Epoch 1 Train Loss 均为 7.59739，Epoch 2 为 3.28439，说明训练结果可复现，只是内存问题
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/lmwpkrx4
- **log**: `logs_lobs5/training_2358257_node1.log`

#### Job 2358266 — ❌ 立刻失败（bash -c 单引号语法错误，8秒）
- **代码**: HEAD 3c5a55c (PREALLOCATE=false 版本)
- **配置**: 同 2358268，2 nodes, 30 min，CURTAIL_EPOCHS=10, EPOCHS=20
- **错误**: `run_train.py: error: argument --epochs: invalid int value: '"${EPOCHS:-40}"'`
  + `syntax error: unexpected end of file` at line 203
- **根因**: 注释 `# ...BFC can't guarantee after fragmentation...` 里的 `can't` 含单引号 `'`
  这个 `'` 提前关闭了 `srun bash -c '...'` 的外层单引号字符串，导致：
  1. `${EPOCHS:-40}` 未被展开，以字面字符串传给 Python
  2. srun 块的 closing `'` 在文件 EOF 找不到，语法报错
- **耗时**: 8 秒即失败（脚本解析阶段）
- **修复**: b49e469 — 将 `can't` 改为不含撇号的写法
- **log**: `logs_lobs5/lobs5_2358266.err`

#### Job 2358268 — 2-node smoke test #8 (3c5a55c 修复语法后: PREALLOCATE=false) ❌
- **代码**: HEAD b49e469 (修复注释单引号后)
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=10, EPOCHS=20, **XLA_PYTHON_CLIENT_PREALLOCATE=false**
- **假设**: 禁用 BFC 预分配大池，改用 CUDA 原生 malloc，让 CUDA driver 处理碎片化
- **实际**: Epoch 1 ✅，Epoch 2 step 0 ❌ OOM 71.62 GiB（比之前更差，退步到 Epoch 2）
- **根因**: 没有 BFC 大池时，CUDA 虚拟地址空间更加碎片化，大块连续请求更难满足
  - OOM 错误含 `[tf-allocator-allocation-error='']`，这是 XLA 对分配失败的通用提示格式
- **结论**: PREALLOCATE=false 是反直觉的更差选择；BFC 预分配大池虽然会碎片化，但总体上比无池时更好管理大块分配
- **耗时**: 12 分钟 19 秒（Epoch 1 完整跑完，Epoch 2 第 0 步失败）
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/5kyixdlw
- **log**: `logs_lobs5/training_2358268_node1.log`

#### Job 2358280 — ⚠️ Caveat: BSZ=4 workaround (fda0372: BSZ=4, no clear_caches)
- **代码**: HEAD fda0372
- **配置**: 2 nodes, 8 GPU, 30 min, CURTAIL_EPOCHS=10, EPOCHS=20, **PER_GPU_BSZ=4**
- **关键**: 去掉 jax.clear_caches()，降低 BSZ
- **结果**: 全部 20 epoch 通过，无 OOM ✅
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/juv0tktc
- **Loss 趋势**: Train 7.64→2.24, Val 3.72→2.14, Acc 0.507→0.602
- ⚠️ **Caveat**: BSZ 从 8 减到 4 是临时绕过方案，实际减半了 Batch Size，影响训练效率和超参配置

## 关键观察（用户澄清）
- BSZ=8 时，**Epoch 2 能通过**，只有 Epoch 3 开始出现 OOM（不是从 Epoch 2 就失败）
- 所以原来 clear_caches 方案 + BSZ=8 能到 Epoch 3，说明问题是在第 3 个 epoch 的内存累积到临界点
- **真实目标：让 BSZ=8 能正常多 epoch 训练**，BSZ=4 只是验证 pipeline 能通过的 workaround

## BSZ Sweep（2026-02-18，2节点，CURTAIL_EPOCHS=10，EPOCHS=20，无 clear_caches）

| BSZ | Job | Global BSZ | 20 Epoch 结果 | Train Loss | Val Loss | Val Acc | 耗时 | wandb |
|-----|-----|-----------|--------------|-----------|---------|---------|------|-------|
| 4 | 2358280 | 32 | ✅ 20/20 | 2.215 | 2.136 | 0.602 | 13:23 | [juv0tktc](https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/juv0tktc) |
| 5 | 2358296 | 40 | ✅ 20/20 | 2.200 | 2.141 | 0.601 | ~13min | [1j3sgajw](https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/1j3sgajw) |
| 6 | 2358297 | 48 | ✅ 20/20 | 2.197 | 2.147 | 0.601 | ~13min | [vz5zxub4](https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/vz5zxub4) |
| **7** | **2358298** | **56** | **✅ 20/20** | **2.186** | **2.146** | **0.601** | **13:19** | **[is6i6vzr](https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/is6i6vzr)** |
| 8 | 2358299 | 64 | ❌ 19/20 OOM | 2.190 | 2.145 | 0.602 | 13:57 | [8vaezg2d](https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/8vaezg2d) |

**结论：PER_GPU_BSZ=7 是最优选择**
- 最大的能跑完 20 epoch 的 BSZ
- Loss 值与 BSZ=4 几乎相同（2.186 vs 2.215），global batch 大 75%（56 vs 32）
- BSZ=8 在 Epoch 20 OOM（极限临界，比之前 no-clear-caches 的 Epoch 2 OOM 好得多）
- BSZ=5/6 也可用，相对更安全的余量

**16节点推算（BSZ=7）**：16×4×7 = **448 global BSZ**

## 全局 Mesh 修复 + NCCL 问题（2026-02-18 下半）

### 发现：跨节点梯度同步 Bug
- `sharding_utils.py:29` 注释明确写 "Gradient sync across nodes would require psum across processes (not yet implemented)"
- 每 node 用 `jax.local_devices()` 创建 LOCAL mesh → 多节点实际是 N 个独立模型
- **修复 commit 9c64e1c**: `jax.local_devices()` → `jax.devices()` 创建全局 mesh
- **修复 commit 0076b03**: batch 脚本 LR/WARMUP 环境变量覆盖

### Job 2358444 — 全局 mesh 验证 #1 ❌ device_put ValueError
- **配置**: 2 nodes, BSZ=7, CURTAIL_EPOCHS=10, EPOCHS=5
- **代码**: HEAD 0076b03
- **错误**: `ValueError: global size of dimension 0 should be divisible by 8, but it is equal to 28`
- **根因**: `jax.device_put(local_batch, global_sharding)` 把 28 样本当全局数组，但 8 device mesh 要求整除
- **修复 commit ed28bff**: `device_put` → `jax.make_array_from_process_local_data`（6 处）
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/djz3cyhk

### Job 2358460 — 全局 mesh 验证 #2 ❌ NCCL cross-node hang (TIMEOUT 30 min)
- **配置**: 2 nodes, BSZ=7, CURTAIL_EPOCHS=10, EPOCHS=5
- **代码**: HEAD ed28bff
- **成功部分**:
  - `[Sharding] Global mesh with 8 devices across 2 processes` ✅
  - `[*] State distributed via sharding (replicated across 8 devices)` ✅
  - `[JIT] Created JIT-compiled train/eval_step with sharding` ✅
  - Node 1 到达 "Starting Training Epoch 1" + "Batch 0" ✅
- **失败部分**: Batch 0 的 `jit_train_step` 调用 hang 了 25+ 分钟
  - train_step 的反向传播需要跨节点 allreduce（NCCL）
  - State 分发成功是因为 replicated sharding 不需要 allreduce
  - NCCL 跨节点通信在 Slingshot 互连上可能缺少 OFI plugin
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/w3je1ukq

### 当前阻塞
- **NCCL 跨节点 allreduce hang**: 全局 mesh 下 XLA 插入的 allreduce 需要 NCCL 跨节点通信
- 节点内 NVLink 正常（之前所有 local-mesh job 都通过）
- 跨节点 Slingshot 上的 NCCL 配置可能有问题

### 下一步
- [ ] 诊断 NCCL: `NCCL_DEBUG=INFO` 提交 diagnostic job
- [ ] 检查 aws-ofi-nccl / cray NCCL plugin 是否可用
- [ ] 备选方案: local mesh + 手动 gradient allreduce

### Job 2358472 — 全局 mesh 验证 #3 ❌ TIMEOUT (aws-ofi-nccl 加载成功但仍 hang)
- **配置**: 2 nodes, BSZ=7, 30 min
- **代码**: HEAD f6b4e16 (aws-ofi-nccl plugin)
- NCCL OFI: aws-ofi-nccl 1.8.1 加载成功，4 comm, 40 rings, GDRDMA ✅
- 仍然 hang 在 Batch 0 → 证明不是 NCCL 问题
- **wandb**: 同 2358460

### Job 2358487 — 1h 诊断 (JAX_LOG_COMPILES=1) → 发现 Orbax 死锁
- **配置**: 2 nodes, BSZ=7, 1h, JAX_LOG_COMPILES=1
- **关键发现**: train_step 编译从未开始！Rank 0 卡在 CheckpointManager()
- **根因**: Orbax 分布式 barrier 死锁（见 findings 9d）

### Job 2358489 — 最小 global mesh 测试 ✅ ALL TESTS PASSED (10s)
- **配置**: 2 nodes
- **结果**: 全局 mesh + 跨节点 allreduce 全部通过

### Job 2358495 — Orbax 修复后验证 ❌ XLA Triton GEMM crash
- **配置**: 2 nodes, BSZ=7, 30 min
- **代码**: HEAD 61eb6f3 (Orbax fix: all ranks create CheckpointManager)
- ✅ Orbax 死锁修复成功，两节点都到 Batch 0
- ❌ train_step 编译时 autotuner.cc crash: `__triton_gemm` + `DEVICE_TYPE_INVALID`
- **修复**: commit 1ae5519 — `--xla_gpu_enable_triton_gemm=false`
- **wandb**: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/vo1x9blk

### Job 2358497 — triton_gemm 禁用后 ❌ 编译卡死 20+ 分钟
- **配置**: 2 nodes, BSZ=7, 30 min
- **代码**: HEAD 1ae5519 (triton_gemm disabled)
- ✅ 不再 crash，两节点都到 Batch 0
- ❌ GPU 利用率仅 1.3 GB，编译未在 GPU 上执行
- **假说**: cuBLAS autotuner 多主机死锁
- **修复**: commit 45bbe0c — `--xla_gpu_autotune_level=0`

### Job 2358860 — autotuning 完全禁用 🔄 IN PROGRESS
- **配置**: 2 nodes, BSZ=7, 30 min
- **代码**: HEAD 45bbe0c (triton_gemm=false + autotune_level=0)
- **验证中**

## Session 3 — Epoch 2+ hang 修复 (2026-02-18 下半)

### 背景
全局 mesh 工作后，Epoch 1 通过但 Epoch 2 training hang。

### Job 2368024 — Epoch 2 hang（session 3 初始 job）
- Epoch 1 完成: Train Loss 7.69806, Val Loss 3.67015, Test Acc 0.5098
- Epoch 2 Batch 0: 数据加载完成但 `train_fn()` hang 14+ 分钟
- CUDA_ERROR_NO_DEVICE × 5（DataLoader workers，不影响训练）

### Job 2368536 — sync checkpoint only ❌
- commit 490c575: `enable_async_checkpointing=False`
- Epoch 2 step 0 完成 (2.29s)，step 1 hang
- **结论**: sync checkpoint alone 不够

### Job 2368882 — debug block_until_ready ✅ KEY VALIDATION
- commit 9508198: block_until_ready + verbose debug for first 3 steps of epoch >= 1
- **3 个 epoch 全部成功!**
- Epoch 2: Val Loss 2.677, Val Acc 0.5400
- Epoch 3: Val Loss 2.405, Val Acc 0.5632
- W&B: good-spaceship-122 / runs/3b06xt03

### Job 2369091 — block step 0 only (clean) ❌
- commit 2acb86c: block_until_ready only at batch_idx==0
- Epoch 2 step 0 本身 hang（state 在 epoch 边界已被 LR mutation 污染）

### Job 2369429 — re-shard only (no block) ❌
- commit a2c2f4e: epoch 开始时 re-shard，无 block_until_ready
- Epoch 2 step 0-1 通过，step 2 hang（每步 LR mutation 重新污染）

### Job 2369971 — optax schedule (HEAD=6335eb5) ✅ 40/40 EPOCH
- 实际运行代码: HEAD (optax schedules + wandb fix)
- W&B: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/giwfdaig
- Best Val Loss: 2.05655, Best Val Acc: 61.82% at Epoch 38
- 节点: nid[010996-010997], 运行时间: 14:31

### Job 2370026 — optax schedule (HEAD=6335eb5) ✅ 40/40 EPOCH
- W&B: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/y8roj5w1
- Best Val Loss: 2.05658, Best Val Acc: 61.82% at Epoch 38
- 节点: nid[011047-011048], 运行时间: 15:20

### Job 2370046 — optax schedule (HEAD=6335eb5) ✅ 40/40 EPOCH
- W&B: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/05nx52xk
- Best Val Loss: 2.05653, Best Val Acc: 61.82% at Epoch 40
- 节点: nid[011051-011052], 运行时间: 14:31

### Job 2369989 — timeout (15min 时限不够) ⏰
- TIMEOUT at 15:14

## 当前 Commit 链（HEAD = 4722ecf）
```
41d336b  ← v2 分支点（干净 B1）
  ...
  45bbe0c  fix: disable autotune_level=0
  4e4f8d7  fix: NCCL P2P env override
  9143892  fix: skip deduplicate in multi-host
  105eaee  fix: re-shard state before checkpoint save
  490c575  fix: disable async checkpoint
  9508198  debug: verbose logging for Epoch 2+ hang
  2acb86c  fix: block first step each epoch
  a2c2f4e  fix: re-shard state before each epoch
  ec1afd4  fix: create globally-replicated LR arrays
  7e16fb5  fix: block every step in multi-host
  4722ecf  refactor: optax schedules replace inject_hyperparams
  b952008  fix: correct steps_per_epoch for multi-host
  6335eb5  fix: wandb LR logging with schedule mode  ← HEAD
```

---

## 2026-02-19: DDP 实现历程回顾 + 参考实现分析

**会话**: dc29e4b2 (同 ssm-stable 分析会话)

### 完成项
- [x] 回顾 B1 DDP 完整历程 (三阶段, 20 commits, 12 个 bug)
- [x] 分析 HyperscaleES 多节点模式 (shard_map + process_allgather)
- [x] 分析 MaxText 多节点模式 (ICI/DCN hybrid mesh + SPMD 自动 allreduce)
- [x] 对比 ssm-stable 分支 (LOCAL mesh, 无跨节点梯度同步, BF16 混合精度)
- [x] 三方对比表 + MaxText 核心文件引用
- [x] 写入 findings.md (B1_task + ssm-stable-08-feb 两处)

### 关键结论
- B1 的 jit+sharding 做法与 MaxText 核心架构一致（SPMD 自动 allreduce）
- ssm-stable 多节点 = N 个独立模型（LOCAL mesh，无梯度同步）
- HyperscaleES 用 ES 不做 backprop，模式不可直接借鉴
- 未来 >1B 参数时升级路径: 1D mesh → 2D mesh (data+fsdp)

### 相关文档
- `tasks/ssm-stable-08-feb/findings.md` — ssm-stable 分支完整分析
- `tasks/B1_task/B1.18.feb/findings.md` — B1 DDP 历程 + 参考实现对比
