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

## 7. B1 → ssm_stable 四大原始问题的诊断结论（迁移前）

在决定用手术式迁移方案之前，对 B1 和 ssm_stable 做了详细的四大问题对比（见 `four_issues_diagnosis.md`）：

### 问题 1：Orbax CheckpointManager — 暂不改（local mesh 下安全）
- B1 和 ssm_stable **都是**只有 rank 0 创建 CheckpointManager
- 之前 job 2356386 遇到的 Orbax barrier mismatch，是因为在 `exp/B1-ignore-times`（pmap 路径）上，某次改动让所有 rank 都创建了 CM，Orbax 内部 barrier 要求所有 rank 参与，导致 hang
- ssm_stable 用 `local_devices()` 创建 mesh（节点内 mesh，非全局 mesh），Orbax 把每个节点视为独立的单机，**不会触发跨节点 barrier**
- **结论**：只要用 local mesh，只有 rank 0 创建 CM 是正确的。如果未来改为 global mesh，需要让所有 rank 创建 CM

### 问题 2：JAX_COMPILATION_CACHE_DIR — 必须删除
- B1 batch 脚本设了 `JAX_COMPILATION_CACHE_DIR="/lus/.../jax_cache"` 持久化缓存
- ssm_stable **没有设**，每次都重新编译
- **从 pmap 改为 jit+sharding 后，编译出来的 HLO 完全不同**（多节点 allreduce/sharding annotation 不同）
- 旧缓存可能导致 stale/incompatible compiled HLO 被加载
- **修复**：删除 `JAX_COMPILATION_CACHE_DIR`，等多节点验证完后再考虑加回来

### 问题 3：缺两个 Barrier + shutdown
- B1 缺少 `sync_global_devices("jax_distributed_init")` → job 2355647 NCCL hang 的直接根因
- B1 缺少 `sync_global_devices("end-of-train")` → 一个 node 先退出，另一个 NCCL comm 断开
- B1 缺少 `jax.distributed.shutdown()` → 进程退出时可能 hang 或 segfault
- 这三个是 ssm_stable 在多节点验证后加入的，B1 直接移植过来

### 问题 4：原有 B1 batch 脚本里的 XLA flags（全部删除）
ssm_stable 没有任何 `--xla_gpu_*` flags，B1 原来有以下会导致 autotuner crash 的 flags：
```bash
JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32       # 删除
JAX_COMPILER_ENABLE_REMAT_PASS=true              # 删除
XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
           --xla_gpu_all_reduce_combine_threshold_bytes=67108864
           --xla_gpu_triton_gemm_any=true
           --xla_gpu_enable_command_buffer="      # 全部删除
```
这些在多节点 pmap 路径下触发 `autotuner.cc:260: device_type:"DEVICE_TYPE_INVALID"` crash。

---

## 8. 跨 Epoch OOM 的完整诊断（2026-02-18）

### 根因
jit+sharding 模式下 `train_step` 需要在单个 GPU 的 BFC pool 中分配 **71.62 GiB 连续块**。
`eval_step`（不同形状）运行后造成 BFC pool 碎片化，第 3 个 epoch 的 `train_step` 在碎片化的 pool 里找不到足够的连续空间。

### 关键观察：Epoch 3 临界点
- BSZ=8 时：Epoch 1 ✅ → Epoch 2 ✅ → Epoch 3 ❌ OOM
- 这是**离散跳变**，不是线性积累——每个 epoch 末的 `deduplicate_trainstate + checkpoint + eval` 循环在 BFC pool 中制造特定模式的碎片
- 经过两轮完整的 train+eval+checkpoint 之后，BFC pool 的碎片化程度首次达到无法满足 71.62 GiB 连续请求的临界

### NCCL rendezvous 超时日志（Epoch 3 OOM 的伴随证据）

在 job 2358257 的 Epoch 2→3 边界处，log 中出现了以下 NCCL 警告：
```
E0218 03:50:19 rendezvous.cc:100] [id=1] This thread has been waiting for
  `acquire clique for rank 3; clique=devices=[4,5,6,7]; is_p2p=0; run_id=-1558191007`
  for 10 seconds and may be stuck. Expected 4 threads to join the rendezvous,
  but not all of them arrived on time.
```
- **`run_id=-1558191007`**：每次 `jax.clear_caches()` 后重新编译，NCCL 会分配新的 `run_id`
- **`clique=devices=[4,5,6,7]`**：这是节点上 GPU 4-7 的 clique（对应第二组设备）
- 这些警告说明 rank 3 在等待其他 rank 加入 NCCL rendezvous，说明旧的 NCCL clique 资源没有被释放，新的 rendezvous 建立受阻
- **这是 clear_caches → 重编译 → 新 NCCL clique 积累的直接日志证据**

### 训练 Loss 在所有 OOM job 中的可复现性（诊断证据）

所有 BSZ=8 的 OOM job，无论使用哪种 memory management 方案，Epoch 1 和 Epoch 2 的训练 loss 完全一致：
- Epoch 1 Train Loss：**7.59739**（每个 job 都相同）
- Epoch 2 Train Loss：**3.28439**（每个 job 都相同）

这说明：
1. 训练本身的逻辑是**完全正确**的（梯度同步、数据加载、模型前向都没有问题）
2. OOM 是**纯粹的内存管理问题**，不影响训练数值结果
3. BSZ=4 job（2358280）的 loss 曲线从 7.64 开始（更高，因为更小的 global batch size），这是 BSZ 减半带来的统计效应，**不是 bug**

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

---

## 9. 全局 Mesh 修复详情（2026-02-18）

### 9a. device_put 在全局 mesh 下的语义错误

**Job 2358444 错误** (train_helpers.py:542):
```
ValueError: ... global size of dimension 0 should be divisible by 8,
but it is equal to 28 (full shape: (28, 11000))
```

| API | 语义 | Local mesh (4 dev) | Global mesh (8 dev) |
|-----|------|-------------------|-------------------|
| `jax.device_put(data, sh)` | data = 全局数组 | 28/4=7 ✅ | 28/8=3.5 ❌ |
| `jax.make_array_from_process_local_data(sh, data)` | data = 本进程分片 | 28→4×7 ✅ | 28→4×7 ✅ |

**修复** (commit ed28bff): 替换 `train_helpers.py` 中 6 处 device_put。

### 9b. prep_batch 不使用 num_devices（安全确认）

`prep_batch()` 的 `num_devices` 参数在 jit+sharding 路径下**完全不使用**
（`train_helpers.py:411` 注释: "jit+sharding: no device_reshape needed"）。
真正的设备分配由 `jax.make_array_from_process_local_data(sharding, data)` 完成。
全局 mesh 不影响数据准备流程。

### 9c. State 分发正常工作

`jax.jit(lambda s: s, out_shardings=state_shardings)(state)` 在全局 mesh 下正常：
- 每 process 独立初始化相同 state（seed=42）
- jit 配合 replicated out_shardings 自动复制到全局 mesh 所有设备
- Job 2358444 log 确认: `State distributed via sharding (replicated across 8 devices) ✅`

### 9d. Orbax CheckpointManager 分布式 barrier 死锁（真正根因！）

**之前的误诊**: 以为是 NCCL 跨节点 allreduce hang 或 XLA 编译慢。实际根因是 Orbax。

**Job 2358487 的 JAX_LOG_COMPILES=1 揭露真相**:
- 所有小函数的编译都在 0.01-0.15s 内完成
- state distribution 编译完成（0.15s）
- 但 `train_step` 编译从未开始！（日志里没有 "Compiling jit(train_step)"）

**时序死锁分析**:
```
Rank 0: create_jit_eval_step ✅ → CheckpointManager() → Orbax barrier → 等 Rank 1 ❌
Rank 1: create_jit_eval_step ✅ → (跳过 ckpt_mgr) → Training Epoch 1 → Batch 0 → jit_train_step() → 等 Rank 0 参与 JIT 编译 ❌
```

经典分布式死锁：Rank 0 等 Orbax barrier（Rank 1 没创建 ckpt_mgr），Rank 1 等 JIT 编译（需要 Rank 0）。

**修复** (commit 61eb6f3):
- 所有 rank 创建 CheckpointManager（same directory, Orbax primary_host=0 决定谁写）
- 所有 rank 调用 save_checkpoint（Orbax 协调 barrier）
- Rank 0 的 checkpoint 路径通过 `broadcast_one_to_all` 广播到所有 rank

**之前的误诊 (9e) 更正**: NCCL 跨节点 hang 不是根因。Job 2358472 的 NCCL_DEBUG=INFO 显示:
- aws-ofi-nccl 1.8.1 加载成功，Provider=cxi（Slingshot），GDRDMA ✅
- 4 个 communicator 全部 Init COMPLETE（nranks=2, 每 GPU 一个 pair）
- Connected all 40 rings via OFI/GDRDMA ✅
- NCCL 初始化完美，问题在初始化之后的代码逻辑（Orbax barrier）

### 9e. 最小 global mesh 测试（job 2358489）

**结果**: ALL 4 TESTS PASSED（10 秒完成），证明:
- 全局 mesh 创建 ✅
- 跨节点 replicated 计算 ✅
- 跨节点 sharded→replicated allreduce ✅
- 跨节点梯度 allreduce（DDP-like）✅
- 大矩阵 allreduce（compile 1.06s, exec 0.8ms）✅
- **但 device_count=2（每 node 只 1 GPU）**，需修复测试脚本用 local_device_ids

### 9f. NCCL OFI 配置（从 job 2358472 确认）

| 项目 | 值 |
|------|------|
| Plugin | aws-ofi-nccl 1.8.1-aws |
| Provider | cxi（Slingshot）|
| Transport | GDRDMA |
| NICs | 4 个 |
| Rings | 40 channels, PXN=0, GDR=1 |
| Communicators | 4 个 (nranks=2, nNodes=2) |

### 9g. Orbax 修复后 — XLA Triton GEMM Autotuner 崩溃（job 2358495）

Orbax CheckpointManager 死锁修复后（commit 61eb6f3），训练成功到达 Batch 0，但 train_step JIT 编译时 XLA autotuner crash：

```
F0218 07:34:20 autotuner.cc:260] Check failed: cached_config.has_value()
Sharding autotuning failed: no config found for HLO:
%gemm_fusion_dot.260 = f32[7,11000,2048]{2,1,0} fusion(...)
  backend_config={"fusion_backend_config":{"kind":"__triton_gemm"},
                  "device_type":"DEVICE_TYPE_INVALID"}
```

- `__triton_gemm` fusion 的 `device_type` 为 `DEVICE_TYPE_INVALID`
- 多主机模式下 Triton GEMM autotuner 无法确定设备类型
- **修复** (commit 1ae5519): `--xla_gpu_enable_triton_gemm=false`，强制 cuBLAS

### 9h. cuBLAS Autotuner 死锁（job 2358497）

禁用 Triton GEMM 后，不再 crash，但 train_step 编译卡死 20+ 分钟：
- 两个节点都到达 "Epoch 0, Batch 0"
- GPU 使用率仅 1.3 GB（编译未实际在 GPU 上运行）
- **假说**：cuBLAS autotuner 在多主机下也有死锁，两个主机的 autotuner 以不同顺序 tune 不同 op，当遇到涉及 allreduce 的 op 时互相等待
- **修复** (commit 45bbe0c): `--xla_gpu_autotune_level=0` 完全禁用 autotuning
- 代价：运行速度可能略慢（cuBLAS 选择非最优算法），但编译应能完成
- **正在验证**: job 2358860

## 10. NCCL P2P Override — 4-GPU/node hang 的真正根因 (session 3)

`run_train.py` 行 33-37 硬编码 `os.environ["NCCL_P2P_DISABLE"] = "0"` 覆盖了 batch 脚本的 `NCCL_P2P_DISABLE=1`。GH200 4-GPU 节点缺少直连 NVLink，P2P=enabled 导致 NCCL hang。
- **修复** (commit 4e4f8d7): 改用 `os.environ.setdefault(...)` 让 batch 脚本优先
- **验证**: job 2366796 — NCCL 初始化成功，8 GPU 跨 2 节点连接，训练 10 步完成

## 11. Checkpoint 序列化问题 — multi-host SingleDeviceSharding

### 11a. deduplicate_trainstate 问题
- `deduplicate_trainstate(state)` 将 globally-sharded state 转为 SingleDeviceSharding
- Orbax 在 multi-host 模式拒绝序列化 SingleDeviceSharding 数组
- 错误: `Cannot serialize host local jax.Array (name=step, sharding=SingleDeviceSharding)`
- **修复** (commit 9143892): multi-host 时跳过 deduplicate

### 11b. Learning Rate 标量 SingleDeviceSharding
- `update_learning_rate_per_step` (train_helpers.py:85) 创建 numpy scalar 赋值给 state
- 这些变成 SingleDeviceSharding，Orbax 无法序列化
- **修复** (commit 105eaee): checkpoint save 前 re-shard 整个 state:
  ```python
  ckpt_state = jax.jit(lambda s: s, out_shardings=state_shardings)(state)
  ```

## 12. Epoch 2 Hang — Orbax async fork 破坏 NCCL

### 根因分析
- Orbax 默认 `enable_async_checkpointing=True`
- 异步写 checkpoint 时 fork 子进程，子进程继承父进程的 CUDA context handle
- 子进程尝试 `cuInit(0)` → `CUDA_ERROR_NO_DEVICE`（日志中出现 5+ 次）
- fork+CUDA 是 NVIDIA 明确禁止的操作，子进程对 GPU state 的访问破坏父进程的 NCCL communicator
- 导致 Epoch 2 train_step 的 all-reduce collective deadlock

### 症状
- Epoch 1 完全正常: 10 步训练 + 验证 + 测试 + checkpoint 保存
- Epoch 2 Batch 0: 数据加载完成、内存打印正常，但 `train_fn()` hang 14+ 分钟直到超时

### 修复
- (commit 490c575): `enable_async_checkpointing=False` 强制同步写入
- job 2368536: Epoch 2 step 1 仍 hang（sync checkpoint 不够）
- **根因修正**：实际是 JAX async dispatch + Python LR mutation 竞态

## 13. Epoch 2 Hang 的真正根因 — async dispatch + state mutation 竞态

### 精确诊断过程 (job 2368882)
添加 `block_until_ready()` 和精细化 debug logging 后发现：
- Epoch 2 steps 0-2 全部在 0.5-1.0s 完成
- Steps 3-9 也正常 (~2 it/s)
- Validation + Test 正常 (~7 it/s)
- **Epoch 2、3 全部成功，Epoch 4 进行中被 30min 时限 kill**

### 竞态链
1. `train_fn` 返回 async futures (GPU 未完成)
2. `update_learning_rate_per_step` 在 Python 层用 `np.array()` 替换 LR 字段 → SingleDeviceSharding
3. 下一步 `train_fn` 收到混合的 "async GPU futures + host-local numpy" state
4. Multi-host 模式 NCCL all-reduce deadlock

### 仅在 epoch 边界触发的原因
- epoch 中间 pipeline 已建立，donation 提供自然同步点
- eval/checkpoint/gc 周期打破了 pipeline，首步回到"新鲜"起点

### 修复 (commit 2acb86c)
```python
# Multi-host: block at first step of each epoch
if mesh is not None and batch_idx == 0:
    loss.block_until_ready()
```
- 开销: ~0.5s/epoch (可忽略)
- 确保 state 物化后再被 Python 修改

### 验证结果 (job 2368882)
| Epoch | Val Loss | Val Acc | Test Acc |
|-------|----------|---------|----------|
| 1 | 3.670 | 0.5091 | 0.5098 |
| 2 | 2.677 | 0.5400 | 0.5416 |
| 3 | 2.405 | 0.5632 | 0.5653 |
- W&B: good-spaceship-122 / runs/3b06xt03

## 14. block_until_ready 仅 step 0 不够 — job 2369091 ❌

### 实验 (commit 2acb86c)
- 只在 `batch_idx == 0` 时 block
- **结果**: Epoch 2 step 0 本身就 hang 了

### 原因分析
- step 0 的 `block_until_ready()` 本身触发了 NCCL deadlock
- 因为 state 在 epoch 边界已经被 `update_learning_rate_per_step` 污染为 SingleDeviceSharding
- 两个 host 的 SingleDeviceSharding 设备 ID 不同（host 0: device 0, host 1: device 4）
- `block_until_ready()` 强制物化→发现 sharding 不一致→deadlock

## 15. Epoch re-shard 方案不够 — job 2369429 ❌

### 实验 (commit a2c2f4e)
- 每 epoch 开始前 re-shard：`state = jax.jit(lambda s: s, out_shardings=state_shardings)(state)`
- 移除了 block_until_ready
- **结果**: Epoch 2 step 0-1 通过，step 2 hang

### 原因分析
- re-shard 只修复了 epoch 边界时的 sharding 不一致
- 但 **每一步** 的 `update_learning_rate_per_step` 都会重新引入 SingleDeviceSharding
- step 0 通过是因为 re-shard 的 jit 提供了同步
- step 1 通过是偶然的（donation 提供了部分同步）
- step 2 时 async pipeline 导致 Python LR mutation 追上 GPU 执行，再次触发不一致

### 关键洞察
- **每步都会被污染**，不能只在 epoch 边界修复
- 需要每步都 block 或每步都 re-shard

## 16. Combined fix: 每步 block + epoch re-shard — job 2369971 🔄

### 方案 (commit 7e16fb5)
```python
# train_helpers.py — 每步 block
if mesh is not None:
    loss.block_until_ready()

# train.py — epoch 开始 re-shard
if is_distributed and epoch > 0:
    state = jax.jit(lambda s: s, out_shardings=state_shardings)(state)
```

### 设计理由
1. **每步 block**: GPU-bound（0.5s/step），Python 层 block 无额外开销
2. **Epoch re-shard**: 保险措施，修复 eval/checkpoint 引入的任何 sharding 漂移
3. 两者协同，覆盖所有可能的 async+mutation 竞态窗口

### 状态
- Job 2369971: PENDING（等 2369429 释放资源）— 测试旧方案
- 配置: 2 nodes, BSZ=7, CURTAIL_EPOCHS=10, EPOCHS=5

## 17. 根因修复: optax schedule 替代 inject_hyperparams — commit 4722ecf

### ssm-stable-08-feb 分支的关键差异

| 方面 | 当前分支 (bandaid) | ssm-stable (根因修复) |
|------|--------------------|-----------------------|
| 优化器 | `inject_hyperparams(adam)(lr=scalar)` | `adam(learning_rate=schedule_fn)` |
| LR 更新 | Python 层 `np.array()` → SingleDeviceSharding | 在 JIT 内部由 optax 自动计算 |
| 死锁防护 | `block_until_ready` 每步 | 不需要 |
| 核心问题 | LR 数组 sharding 不一致 | LR 从不离开 JIT |

### ssm-stable 没有全局 mesh!

```python
# ssm-stable sharding_utils.py:
local_devs = jax.local_devices()        # ← 只有本地 GPU
devices = local_devs[:num_devices]
# 注释: "Gradient sync across nodes would require psum across processes (not yet implemented)"
```

所以 ssm-stable 从未遇到 NCCL deadlock——因为每个节点独立训练！

### 组合方案

采用 ssm-stable 的 optax schedule 消除根因 + B1 的全局 mesh 实现真正 DDP：

```python
# train_helpers.py — create_lobs5_learning_rate_schedule
warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=base_lr, transition_steps=warmup_end_step)
cosine_schedule = make_cos_schedule(base_lr, lr_min, cosine_steps)
schedule = optax.join_schedules([warmup_schedule, cosine_schedule], [warmup_end_step])

# create_train_state — 直接传 schedule 给 optimizer
tx = optax.multi_transform({
    "none": optax.sgd(learning_rate=0.0),
    "ssm": optax.adam(learning_rate=ssm_lr_schedule),  # schedule, not scalar!
    "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
}, ssm_fn)

# train_epoch — lr_params=None 时跳过 update_learning_rate_per_step
# 不需要 block_until_ready — 没有 sharding 污染
```

### 验证
- Job 2370026: 2 nodes, BSZ=7, CURTAIL_EPOCHS=10 — 验证 optax schedule + 全局 mesh
- 预期: 多 epoch 完成，无 NCCL deadlock，无 block_until_ready 开销

## 18. WandB LR Logging Bug Fix — commit 6335eb5

### 问题
`inject_hyperparams` 模式下 LR 存在 `state.opt_state.inner_states[key].inner_state.hyperparams['learning_rate']`。
切换到 optax schedule 后，optimizer state 结构变了：
- `optax.adam(lr=schedule_fn)` → `(ScaleByAdamState, ScaleState)` tuple
- 没有 `.inner_state.hyperparams` → `AttributeError` → Epoch 1 结束时 crash

### 修复
在 `train.py` 中创建 schedule 函数用于 logging：
```python
lr_schedule_fn = create_lobs5_learning_rate_schedule(...)
# wandb.log 用 float(lr_schedule_fn(step)) 代替 state 读取
```

### 关键洞察
optax 优化器的 state 结构取决于用不用 `inject_hyperparams`：

| 组件 | inject_hyperparams | 直接 schedule |
|------|-------------------|---------------|
| State 类型 | `InjectStatefulHyperparamsState` | `(ScaleByAdamState, ScaleState)` |
| LR 存储 | `.hyperparams['learning_rate']` | 不存储，从 step 计算 |
| 可变性 | Python 层可变 | JIT 内部只读 |

## 19. 当前 Commit 链（HEAD = 6335eb5）
```
...
4722ecf  refactor: optax schedules replace inject_hyperparams
b952008  fix: correct steps_per_epoch for multi-host
6335eb5  fix: wandb LR logging with schedule mode  ← HEAD
```

### ✅ 验证通过 (2026-02-19 02:32-02:47 UTC)

3 个独立 jobs 全部 40/40 epoch 完成，零 NCCL deadlock：

| Job ID | 节点 | Epochs | Best Val Loss | Best Val Acc | W&B |
|--------|------|--------|---------------|-------------|-----|
| 2369971 | nid[010996-010997] | 40/40 ✅ | 2.05655 | 61.82% | [giwfdaig](https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/giwfdaig) |
| 2370026 | nid[011047-011048] | 40/40 ✅ | 2.05658 | 61.82% | [y8roj5w1](https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/y8roj5w1) |
| 2370046 | nid[011051-011052] | 40/40 ✅ | 2.05653 | 61.82% | [05nx52xk](https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/05nx52xk) |
| 2369989 | (timeout 15min) | N/A | N/A | N/A | — |

Loss 趋势: Train 8.09→2.08, Val 3.68→2.06, Test 3.68→2.03, Acc 52%→62.5%

注意：Python 输出在 `logs_lobs5/training_<JOBID>_node<N>.log`（srun --output 重定向）

---

## 20. DDP 实现完整历程总结 (2026-02-19 回顾)

### 三阶段时间线

```
Feb 14-16  问题发现: 单节点长训练 Test Loss 上涨
     │
Feb 17     pmap multi-host 尝试 (7 次 crash)
     │        ├── NCCL hang → 加 barrier
     │        ├── Orbax mismatch → 全 rank 创建 ckpt_mgr
     │        └── XLA autotuner crash → pmap multi-host 不可用
     │     决策: 手术式 pmap → jit+sharding
     │
Feb 18     ┌─ Phase 1: GPU 进程架构探索 (1h, 9 commits)
  18:35     │    ├── process-per-GPU 尝试 → NCCL OOM → 放弃
  -19:38    │    ├── 保留: CUDA_MODULE_LOADING=EAGER
           │    ├── 保留: NCCL_P2P_DISABLE=1
           │    └── 保留: setdefault() 替代硬编码
           │
           ├─ Phase 2: Checkpoint 序列化修复 (45min, 3 commits)
  19:51     │    ├── skip deduplicate_trainstate
  -20:37    │    ├── jax.jit(identity, out_shardings=...) re-shard
           │    └── 禁用 async checkpoint (fork 破坏 NCCL)
           │
           └─ Phase 3: Epoch 2 NCCL 死锁 (2h, 8 commits)
  21:06          ├── 诊断: inject_hyperparams + np.array() → SingleDeviceSharding
  -23:08         ├── bandaid: block_until_ready → 3 epoch 验证成功
                └── 根因修复: optax schedules 替代 inject_hyperparams
```

### 12 个 Bug → Fix 完整映射表

| # | Bug | 根因 | 修复 commit | 方法 |
|---|-----|------|------------|------|
| 1 | pmap multi-host XLA crash | autotuner DEVICE_TYPE_INVALID | — | 放弃 pmap，改 jit+sharding |
| 2 | process-per-GPU NCCL OOM | 8 进程 communicator 初始化 OOM | — | 回退 1-proc/node |
| 3 | NCCL P2P hang | GH200 无直连 NVLink | `4e4f8d7` | `NCCL_P2P_DISABLE=1` |
| 4 | env var 被 Python 覆盖 | `os.environ[...] = "0"` 硬编码 | `4e4f8d7` | `setdefault()` |
| 5 | GPU kernel loading hang | 多 GPU 延迟加载 | `4e4f8d7` | `CUDA_MODULE_LOADING=EAGER` |
| 6 | Triton GEMM crash | 多节点 autotuner 不兼容 | `1ae5519` | `triton_gemm=false` |
| 7 | cuBLAS autotune 死锁 | 多节点 autotune 协调 | `45bbe0c` | `autotune_level=0` |
| 8 | Orbax 分布式死锁 | 只 rank 0 创建 ckpt_mgr | `f6b4e16` | 全 rank 创建 |
| 9 | Checkpoint 序列化 | deduplicate → SingleDeviceSharding | `9143892`+`105eaee` | skip + re-shard |
| 10 | Async ckpt 破坏 NCCL | fork 子进程继承 CUDA context | `490c575` | `async=False` |
| 11 | Epoch 2 hang | LR mutation → 不一致 sharding | `4722ecf` | optax schedules |
| 12 | steps_per_epoch 错误 | bsz 是 per-process 非 global | `b952008` | `÷ process_count` |

### 最终架构

```
┌─────────────────────────────────────────────────┐
│  SLURM: 1 process/node, 4 GPUs/process          │
│  ntasks-per-node=1, gres=gpu:4, gpu-bind=none   │
├─────────────────────────────────────────────────┤
│  CUDA: CUDA_MODULE_LOADING=EAGER                 │
│  XLA:  triton_gemm=false, autotune_level=0       │
│  NCCL: P2P_DISABLE=1, TIMEOUT=3600              │
│  JAX:  PREALLOCATE=true, MEM_FRACTION=0.90       │
├─────────────────────────────────────────────────┤
│  Mesh: GLOBAL — jax.devices() 跨节点所有 GPU     │
│  Sharding: P('data') data, P(None) params        │
│  梯度同步: XLA 自动 allreduce (无显式 pmean)      │
├─────────────────────────────────────────────────┤
│  LR: optax schedules (warmup → cosine)           │
│      在 JIT 内部从 state.step 计算               │
│      不经过 Python，不可能创建 SingleDeviceSharding│
├─────────────────────────────────────────────────┤
│  Checkpoint: sync (禁用 async), re-shard 后保存   │
│  env vars: setdefault() 让 batch 脚本优先         │
└─────────────────────────────────────────────────┘
```

---

## 21. 参考实现对比 (HyperscaleES + MaxText + ssm-stable)

详见 `tasks/ssm-stable-08-feb/findings.md` 的完整分析。

### 定位对比

| 项目 | 训练范式 | Mesh | 梯度同步 | FSDP |
|------|---------|------|---------|------|
| **HyperscaleES** | ES (无 backprop) | 1D global | 不需要 (确定性噪声) | 无 |
| **MaxText** | LLM backprop | 2D ICI/DCN hybrid | SPMD 自动 | 有 (默认) |
| **ssm-stable** | S5 backprop | 1D **local** | ❌ 未实现 | 无 |
| **B1 (当前)** | S5 backprop | 1D **global** | ✅ SPMD 自动 | 无 |

### B1 与 MaxText 的架构一致性

B1 的 `jax.jit + in/out_shardings` + `make_array_from_process_local_data` 模式
与 MaxText 的 SPMD 路径本质相同。两者都：
- 用全局 mesh 声明设备拓扑
- 用 NamedSharding + PartitionSpec 声明数据/参数分布
- 依赖 XLA 编译器自动插入 collective 通信

### 未来扩展路径 (>1B 参数时)

当前 75M 模型用纯 DP (1D mesh) 足够。扩展到 1B+ 时：
- Mesh: 1D `('data',)` → 2D `('data','fsdp')`
- 参考 MaxText: `ici_fsdp=4` (节点内), `dcn_data=N` (跨节点)
- FSDP 减少每 GPU 显存占用 N 倍

---

## 22. 关键教训 (★ Insights)

### ★ Insight: inject_hyperparams 是多节点的定时炸弹
它把 LR 暴露为 Python mutable value，每次手动更新创建 host-local SingleDeviceSharding
（host 0 用 device:0，host 1 用 device:4）。NCCL all-reduce 需要一致 sharding，
不一致 = 死锁。optax schedules 让 LR 在 JIT 内部计算，永远不离开 XLA 域。

### ★ Insight: jax.jit(identity, out_shardings=target) 是多节点的瑞士军刀
任何时候 Python 操作污染了 sharding，用这个 identity JIT 就能重新投影回正确的
全局 NamedSharding。B1 在 checkpoint save 和 epoch 边界都用了它。

### ★ Insight: 多节点 JAX bug 的共同模式
所有 12 个 bug 都有一个共同模式：Python host-side 操作（np.array, os.environ,
fork, clear_caches）破坏了 XLA 编译器预期的全局一致性。
解决方案都是：尽量让一切留在 JIT 内部。

---

## 23. DDP 验证完成 (2026-02-19)

**三个 job 全部 40/40 epoch 完成，DDP 目标达成！**

| Job | Commit | 方案 | Epochs | Best Test Acc | W&B |
|-----|--------|------|--------|--------------|-----|
| 2369971 | 7e16fb5 | block every step (bandaid) | 40/40 | 62.51% | `giwfdaig` |
| 2370026 | 4722ecf | optax schedule (有 steps bug) | 40/40 | 62.51% | `y8roj5w1` |
| **2370046** | **b952008** | **optax schedule (修复版)** | **40/40** | **62.50%** | **`05nx52xk`** |

### 训练曲线 (job 2370046, 生产代码)

| Epoch | Train Loss | Val Loss | Test Loss | Val Acc | Test Acc |
|-------|-----------|----------|-----------|---------|----------|
| 1     | 8.09      | 3.68     | 3.68      | 52.2%   | 51.7%    |
| 5     | 2.38      | 2.27     | 2.25      | 57.7%   | 58.1%    |
| 10    | 2.22      | 2.17     | 2.14      | 59.5%   | 60.0%    |
| 20    | 2.13      | 2.09     | 2.06      | 61.3%   | 61.8%    |
| 30    | 2.09      | 2.06     | 2.04      | 61.7%   | 62.2%    |
| 40    | 2.08      | 2.06     | 2.03      | 61.8%   | 62.5%    |

### 关键确认
1. **optax schedule 根因修复有效** — 40 epoch 零死锁，无需 block_until_ready
2. **三个方案结果一致** — 说明 DDP 梯度同步正确（不同修复方式不影响训练）
3. **CURTAIL_EPOCHS=10 测试充分** — 40 epoch × 11 steps = 440 global steps
