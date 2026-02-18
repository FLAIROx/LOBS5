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

## 下一步
- [x] 验证 pipeline 可以多 epoch 训练 ✅ (BSZ=4 workaround, job 2358280)
- [ ] **真正目标**: 让 BSZ=8 能稳定多 epoch（研究 Epoch 3 的临界点）
  - **A** remat/gradient checkpointing：workspace 71→40 GiB，保持 BSZ=8（优先推荐）
  - **B** Epoch 边界 BFC 整理：`jax.block_until_ready(state)` 在 checkpoint 后确保异步释放
  - **C** XLA dump 分析：`--xla_dump_to` 在 Epoch 3 OOM 前获取 BFC 碎片结构
  - **D** SSM scan 内存优化：探索分块 associative scan 降低峰值
- [ ] 生产训练：大 epoch 数量（BSZ=4 workaround 或等 BSZ=8 修复）

## 当前 Commit 链（HEAD = 83b2d05）
```
41d336b  ← v2 分支点（干净 B1）
  18f832d  refactor: pmap → jit+sharding
  491f092  fix: deduplicate_trainstate
  49c265d  fix: lr[0] scalar
  070facd  fix: restore jax.clear_caches（后来移除）
  caf58db  fix: del+recreate jit（错误方向）
  84c52ba  fix: TF_GPU_ALLOCATOR（无效）
  8147fd6  docs: OOM 分析
  3c5a55c  fix: PREALLOCATE=false（更差）
  b49e469  fix: 修复注释里的单引号 bash 语法 bug
  fda0372  fix: 去掉 clear_caches + BSZ=4 workaround
  5db3599  docs: 记录 OOM 最终修复
  f4e69cf  docs: job 2358280 成功记录
  83b2d05  docs: BSZ=4 标记为 caveat（当前 HEAD）
```
