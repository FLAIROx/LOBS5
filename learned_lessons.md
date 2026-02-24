# Learned Lessons

## B2: 1-128 GPU Scaling Benchmark with 2D Mesh + FSDP (2026-02-20)

### Scaling Results (75M model, GOOG 2022, PER_GPU_BSZ=7)

| GPU | Nodes | s/step | samp/s | eff% | 并行模式 |
|-----|-------|--------|--------|------|----------|
| 1   | 1     | 0.410  | 17.1   | 100% | 无并行 |
| 2   | 1     | 0.430  | 32.6   | 95%  | DDP (1D mesh) |
| 4   | 1     | 0.455  | 61.5   | 90%  | 纯 FSDP (2D mesh 1×4) |
| 8   | 2     | 0.565  | 99.1   | 73%  | FSDP+DDP (2D mesh 2×4) |
| 16  | 4     | 0.695  | 161.2  | 59%  | FSDP+DDP (2D mesh 4×4) |
| 32  | 8     | 0.875  | 256.0  | 47%  | FSDP+DDP (2D mesh 8×4) |
| 64  | 16    | 2.130  | 210.3  | 19%  | FSDP+DDP (2D mesh 16×4) ⚠ |
| 128 | 32    | 1.745  | 513.5  | 23%  | FSDP+DDP (2D mesh 32×4) |

### Key Findings
1. **Scaling wall at 32 GPU** for 75M model — peak throughput/cost efficiency
2. **2D Mesh vs 1D DDP**: 16 GPU 9.5x faster (0.695 vs 6.58 s/step)
3. **Intra-node scaling** near-linear (90-100% eff), **cross-node** adds ~24% per doubling
4. 64 GPU anomalously slower than 128 GPU (sharded autotuning kernel quality issue)

### 128 GPU Fix Chain
1. `numpy.linalg.eigh()` — avoid cuSolver handle contention at 128 CUDA contexts
2. `--xla_gpu_shard_autotuning=false` for 32+ nodes — avoid DEVICE_TYPE_INVALID
3. `NCCL_BUFFSIZE=2097152` — avoid NCCL OOM from 32-node channel buffers
4. **Do NOT use shard_autotuning=false on 16 nodes** — causes 30+ min JIT freeze

---

## A2: ssm_stable 22tok Scaleup (2026-02-08)

### BSZ Sweep Results (360M, 22tok, BF16, 96GB GH200)

| BSZ/GPU | Eff BSZ | Status | MFU    | Memory (alloc) |
|---------|---------|--------|--------|----------------|
| 2       | 8       | OK     | 41.0%  | fits           |
| 4       | 16      | OK     | 50.3%  | fits           |
| 5       | 20      | OOM    | -      | 76.01 GiB      |
| 6       | 24      | OOM    | -      | 77.14 GiB      |
| 7       | 28      | OOM    | -      | 91.97 GiB      |
| 8       | 32      | OOM    | -      | 110.48 GiB     |
| 16      | 64      | OOM    | -      | 221.05 GiB     |
| 28      | 112     | OOM    | -      | 383.66 GiB     |

**Max BSZ: 4/GPU (MFU 50.3%)**

### Key Insights

1. **BF16 vs FP32**: BF16 enables bsz=4 (vs FP32 bsz=2), MFU jumps from 21% to 50%
2. **22tok vocab overhead**: vocab=12012 (vs 24tok vocab=2112) makes embedding/decoder 6x larger, severely limiting BSZ compared to 24tok sweep (24tok bsz=28 OK)
3. **jit+shardings vs pmap**: MFU doubled from ~21% (pmap, kang) to ~50% (jit+shardings, ssm_stable) on same model size

### Bugs Fixed

1. **Python scoping bug**: `from X import Y` inside conditional block shadows module-level import, causing `UnboundLocalError` even when condition is False
2. **maxtext relative path**: lobmax uses `../../maxtext` which breaks in git worktrees under `experiments/`. Fix: symlink `experiments/maxtext -> AlphaTrade/maxtext`
3. **BSZ arg**: Removed `--global_bsz` from CLI (was wrong value in multi-node), rely on `PER_GPU_BSZ` env var override in `run_train.py:448`
4. **Data leakage**: GOOG_2018_2022_combined had 18 files from 2023-01 (test set overlap). Created clean symlink dir with 2490 files

## B1: ignore_times Baseline (2026-02-15)

### Disk Quota Zombie Job Incident (Job 2316200)

**问题**: 8 节点训练跑到 epoch 11 时，项目磁盘配额 (200T) 耗尽，导致：
- orbax async checkpoint 后台 silently fail（不 crash 主循环）
- wandb 本地文件写入失败 → 云端同步断开
- srun output 文件停止更新
- 训练循环在 epoch 11 中途崩溃，进程卡在 tqdm atexit deadlock
- **12 小时 × 8 节点空转**，GPU 利用率 0%

**根因**: `enable_async_checkpointing=True`（orbax 默认值）使 checkpoint 写入失败不会 raise 到主线程

**修复**:
1. `lob/train.py`: `save_checkpoint()` 包裹 `try/except OSError` → `sys.exit(1)`
2. `train_full_autoreg.batch`: 训练前检查配额，>195T 时警告

**关键教训**:
- 项目配额检查用 `lfs quota -p`，不是 `lfs quota -u`（user quota 显示 0=无限，实际限制在 project 上）
- orbax async checkpoint 默认 silently fail — **必须**加 error handling
- Lustre 配额缓存会延迟数小时，计算节点即使配额已恢复也可能继续报错
- **GDB + PyRun_SimpleString 可以从 deadlocked 进程中抢救 GPU 数据**（gc.get_objects → TrainState → serialization）

### CRITICAL BUG: 多节点训练无跨节点梯度同步 (2026-02-17 发现)

**Bug**: commit `6765e5e` (2026-02-14) 移除了 `jax.distributed.initialize()`，导致多节点训练中
`jax.lax.pmean(grads)` 只在本节点 4 GPU 间同步，不跨节点。
8 个节点各自独立训练 1/8 的数据，不是真正的 DDP（全局梯度同步）。

**segfault 根因**: Job 2313306/2313308 确实是 segfault（`lobs5_2313306.err`: "task 7: Segmentation fault
(core dumped)"）。Node 7 segfault → srun 终止其余 node。segfault 发生在
`jax.distributed.initialize()` 成功之后（日志显示 "32 total GPUs"），与
`CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"` 设了 8 个 ID（只有 4 GPU）有关：
warning `Allowed device set contains 8 devices, but platform only sees 4`。
设备数不匹配可能在首次 pmap 编译或 NCCL 初始化时触发 segfault。

**实际行为（非 DDP）**:
- 每个 node 独立创建 wandb run + 独立保存 checkpoint（73 个目录，8 nodes × ~9 epochs）
- 每个 node 只看 1/8 数据（DistributedSampler）、只在本地 4 GPU 做 pmean
- 不是"只有 Node 0 的结果被保存" — 所有 8 个 node 都有完整的训练输出
- 但对于单模型目标，等价于 8 份独立的子数据集训练，7/8 算力冗余

**影响**:
- Job 2316200 (8 nodes, ~18h) — 每个 node 只训练了 1/8 数据
- Job 2315995, 2314857 — 同样
- 无法通过增加节点实现训练加速

**修复**:
1. `CUDA_VISIBLE_DEVICES="0,1,2,3"` (匹配实际 GPU 数)
2. 恢复 `jax.distributed.initialize()` 使 pmean 跨节点
3. wandb/checkpoint 只在 rank 0 操作（避免 8 份重复）

**教训**:
- 多节点训练验证: 所有 node 同一 step 的 loss 必须 bit-identical（pmean 跨节点后值相同）
- segfault 诊断要看 `.err` 文件（srun error output），不只看 node log
- `CUDA_VISIBLE_DEVICES` 必须与实际 GPU 数量严格匹配
- 加入多节点功能时，应先用 2 nodes + curtail_epochs 做 5 分钟验证
- 不要硬编码设备数 — 应动态检测

## B1: pmap → jit+sharding 迁移 (2026-02-18)

### pmap→jit 迁移的 `[0]` 索引陷阱
- pmap 给所有张量加 device 维度 `(num_devices, ...)`，代码到处用 `[0]` 取值
- jit+sharding 保持原始形状，所有 `[0]` 必须清理
- 关键位置: `loss[0]`, `deduplicate_trainstate(x[0])`, `learning_rate[0]`
- **迁移时必须全文搜索 `[0]` 并逐一确认哪些是 pmap device dim 相关的**

### `jax.devices()` vs `jax.local_devices()` 多节点陷阱
- `jax.devices('gpu')[0]` 在多节点下可能指向全局 device 0（跨节点）
- `jax.local_devices()[0]` 保证是当前进程的本地设备
- **多节点代码必须用 `local_devices()`，永远不要假设 `devices()[0]` 是本地的**

### `jax.clear_caches()` 是双刃剑
- 清除 JIT 编译缓存 → 下一 epoch 重新编译
- 编译本身需要大量临时内存 → 可能导致 OOM
- ssm_stable 调用它，B1 注释掉了（因为导致 recompile OOM at bsz=3）
- **替代方案**: 显式 `del` 大变量 + `gc.collect()`，而非 clear_caches

### Checkpoint 保存只在 rank 0 执行
- ssm_stable: `if is_main_process:` 包裹整个 ckpt 构建+保存
- B1 之前: ckpt 构建在 if 外面 → 所有 rank 都调 deduplicate_trainstate → rank 1 crash
- **ckpt dict 构建和保存必须都在 `if is_main_process` 内**

### Epoch 间 OOM 的真实根因 + 正确解法（jit+sharding 专属问题）
- OOM 是 `PjRtLoadedExecutable::Execute()` 执行期申请 71.62 GiB 连续块失败
- **根因**: jit+sharding 模式下，train_step 需要单 GPU BFC pool 里的大连续块
  - pmap 没有这个问题（每个设备独立分配，不需要 71.62 GiB 连续块）
  - eval_step 碎片化 BFC pool → 下一次 train_step 找不到连续空间
- **jax.clear_caches() 不是解法**: 每 epoch 重编译（193s/epoch 税）+ 新 NCCL clique 积累
  - 导致 epoch 2 变快但 epoch 3 又 OOM（NCCL clique 内存积累）
  - `TF_GPU_ALLOCATOR=cuda_malloc_async` 是 TensorFlow 变量，对 JAX/XLA **无效**
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false` 反而更差（CUDA 地址空间更碎片化）
- **正确解法**: **不调 jax.clear_caches()** + **降低 PER_GPU_BSZ 8→4**
  - workspace 从 71.62 GiB → ~35.81 GiB，BFC 碎片化后仍能分配
  - JIT 只在第一 epoch 编译，后续所有 epoch 复用（快！）
- **bash -c 陷阱**: 注释里的英文缩略词（如 can't）含单引号，会提前关闭 bash -c '...' 块
- `del ckpt` + `gc.collect()` 保留（释放 checkpoint state 副本，好实践）
- ⚠️ **Caveat**: BSZ=4 是临时 workaround，真实目标是 BSZ=8 能稳定多 epoch（BSZ=8 时 Epoch 1+2 通过，Epoch 3 OOM）

### JAX 的两层缓存架构（OOM 调试中发现）
- **Python trace 缓存**：`jax.clear_caches()` 清的是这层，下次调用重新 trace → 触发重编译
- **XLA C++ 编译缓存**：compiled executable 存在 XLA 后端，即使 Python 层被清，C++ 层仍存活
- **实验证据**：Epoch 3 train_step 没有 193s warmup 就直接 OOM，说明 XLA executable 被复用（不需重编），但执行时内存不足
- **意义**：jax.clear_caches() 并不等于"释放 GPU 上的 compiled executable 内存"，它只重置 Python 的 dispatch 路径

### JAX_COMPILATION_CACHE_DIR 切换架构时必须清理
- 持久化 cache 会保存 pmap 路径的 HLO，改为 jit+sharding 后 HLO 完全不同
- 旧 cache 可能导致加载 stale/incompatible compiled HLO，行为不可预测
- **切换 pmap↔jit+sharding 时必须删除或用新的 cache 目录**

### NCCL clique 在 jax.clear_caches() 后的积累
- 每次 `clear_caches()` 强制重编译 → 产生新的 NCCL run_id → 申请新的 NCCL clique
- 旧 clique 的 GPU 内存不一定立刻释放（NCCL 内部维护通信缓冲区池）
- **log 证据**：`rendezvous.cc:100` 10 秒超时警告，`run_id` 在每个 epoch 后变化
- 跨节点 NCCL clique 每个约 8 GiB，积累 2 个就额外消耗 ~16 GiB，Epoch 3 时达到临界

### CURTAIL_EPOCHS 与 clear_caches 叠加的时间陷阱
- CURTAIL_EPOCHS=10 本意是快速测试（10 步即结束训练），但配合 clear_caches() 变成慢测试
- 10 步 train + 10 步 eval 实际计算只要 7 秒，JIT 重编译要 193+43=236 秒
- **99% 的 epoch 时间都在编译**，20 epochs 需要 88 分钟（远超 30 分钟 job 时限）
- 去掉 clear_caches 后，20 epochs 只需 12 分钟（JIT 只编译一次）

### ssm_stable 直接移植不可行的两个具体原因
- `lob.profiling_utils`（GoodputMonitor）：ssm_stable 独有模块，B1 没有
- `local_device_ids=[0,1,2,3,4,5,6,7]`：ssm_stable 的 run_train.py 为 2×4 GPU 节点硬编码了 8 个 device ID，在 4 GPU 节点上触发 `Allowed device set contains 8 devices, but platform only sees 4`
- **教训**：直接测试 31 秒即揭示问题，避免了在错误方向上浪费时间

### Orbax local mesh 下的安全边界
- **Orbax 在 distributed mode 下内部使用 barrier，要求所有 rank 同步**
- ssm_stable 用 `jax.local_devices()` 创建 Mesh（每节点独立），Orbax 视为单机，不触发跨节点 barrier
- 如果改为 `jax.devices()`（全局 Mesh），Orbax 会尝试跨节点 barrier，只有 rank 0 创建 CheckpointManager 就会 hang
- **结论**：local mesh + rank 0 创建 ckpt_mgr = 安全；global mesh + rank 0 创建 ckpt_mgr = 可能死锁
- **2026-02-18 更新**: B1 已修复为 global mesh（commit 9c64e1c）。`deduplicate_trainstate` 先转为本地数组再存，可能不触发 Orbax 分布式路径。需实测确认。如果 checkpoint 时 hang，修复方案：所有 rank 创建 ckpt_mgr + 所有 rank 调用 save。

### 跨节点梯度同步 bug（2026-02-18 修复）
- **Bug**: `sharding_utils.py:33` 用 `jax.local_devices()` 创建 mesh，多节点时每个 node 独立训练，无梯度同步
- **代码注释**: "Gradient sync across nodes would require psum across processes (not yet implemented)"
- **误导日志**: print 声称 "gradient sync via psum" 但 psum 根本没实现
- **影响**: 之前所有多节点训练（包括 ssm_stable）实际上都是 N 个独立模型
- **修复**: 改为 `jax.devices()` 创建全局 mesh，JAX 自动 allreduce（commit 9c64e1c）
- **受影响的下游代码**（确认安全）:
  - `prep_batch()`: 不使用 num_devices（line 411 注释已说明）
  - `create_train_state()`: 用 local num_devices 做 dummy init（不影响）
  - `deduplicate_trainstate()`: `jax.device_get()` 对 replicated 全局数组仍有效
  - Orbax checkpoint: 可能风险（见上条），需实测
- **ssm_stable 同样存在此 bug**: 需要同步修复（见 `tasks/B1_task/B1.18.feb/ssm_stable_global_mesh_fix.md`）

## B2 BF16 Mixed Precision (2026-02-19)

### BF16 Sandwich 策略
- **Scan 必须 FP32**: BF16 的 7-bit 尾数在 14 层递归后累积误差 (1+ε)^14 ≈ 1.15，实测 NaN
- **Matmul 可以 BF16**: B@u, C@xs 是单次矩阵乘，BF16 误差不累积，GH200 Tensor Core 2x 加速
- **vmap→batch matmul**: 独立优化，消除 12000 次 Python 循环开销

### XLA 多节点 Autotuner 修复
- `--xla_gpu_autotune_level=0` 绝对禁止（性能退化 10-24x）
- `--xla_gpu_shard_autotuning=false` 是正确方案: 禁用跨节点 autotune 分片，避免 DEVICE_TYPE_INVALID crash
- 所有主流框架（MaxText, HyperscaleES）都用默认 level=4

### BF16 速度提升
- 1N: 1.69x faster (440→260 ms/step), 41% GPU-hrs 节省
- 2N: 1.28x faster (460→360 ms/step), 22% GPU-hrs 节省
- 2N 加速低于 1N 因为 allreduce 通信不受 BF16 影响（Amdahl 定律）
- 精度完全无损: Val Loss 2.056 (BF16) vs 2.06 (FP32), Test Acc 62.50% (完全一致)

## Git Worktree 目录布局与分支安全 (2026-02-22)

### 目录结构

| 目录 | 用途 | 分支示例 |
|------|------|----------|
| `/projects/s5e/quant/AlphaTrade/LOBS5` | 主 repo | `ignore-times-shard-map` (主开发) |
| `/projects/s5e/quant/AlphaTrade/experiments/exp_*` | 实验 worktree | `exp/D2-*`, `exp/B1-*`, `exp/A1-*` 等 |
| `LOBS5/.claude/worktrees/C5*` | Claude worktree | `exp/C5-shard-map`, `exp/C5a1-32N-optimize` |

### 事故 (2026-02-22): eval watchdog commit 到错误分支

**经过**: 在主 repo 目录 (`LOBS5/`) 下编写 eval watchdog 代码并 commit。当时主 repo 的 `HEAD` 指向 `exp/D2-correct-inference-with-pipeline-fixes`（上个 session 切过去后没切回来），而非目标 `ignore-times-shard-map`。导致 commit `0ae7e6d` 提交到了 D2 分支。

**修复**: 切回 `ignore-times-shard-map` 后 `git cherry-pick 0ae7e6d` → `6c2fade`。D2 分支需 `git reset HEAD~1` 清理。

### 操作规则

1. **编码前必检查分支**: 任何 `git commit` 前必须 `git branch --show-current` 确认在正确分支
2. **不要在主 repo 切分支**: 如果改动属于实验分支（如 D2、B2），必须 `cd` 到对应 worktree 目录操作
3. **Session 开始时验证**: 新 session 的第一步应确认 `pwd` + `git branch` 状态
4. **跨分支共享改动**: 对多个分支都有意义的改动（如 watchdog），需分别到各 worktree 路径下 cherry-pick

## 2026-02-23: A1 vs A2 Scale-Up 分支对比分析

### 模型规模（两分支相同）

| 维度 | Baseline (main) | A1/A2 Scale-Up | 倍数 |
|------|-----------------|----------------|------|
| d_model | 1024 | 2048 | 2x |
| n_layers | 12 | 24 | 2x |
| ssm_size | 1024 | 2048 | 2x |
| blocks | 16 | 32 | 2x |
| 参数量 | ~55M | **~360M** | ~6.5x |

### 训练数据差异

| 分支 | 数据 | 交易日数 | 路径 |
|------|------|----------|------|
| A1-kang | GOOG 2018-2022 合并 | ~1254天 (5年) | `GOOG_2018_2022_combined/` ⚠️ 有 test 泄漏 |
| A2-ssm | GOOG 2022 单年 | ~249天 | `GOOG_2022/` (干净 split) |

### 核心基础设施差异

| 功能 | A1 (scaleup-kang) | A2 (scaleup-ssm) |
|------|-------------------|-------------------|
| 梯度同步 | `pmap` + `pmean` (1D flat) | `shard_map` + `psum` (2D mesh) |
| 多节点初始化 | 无 `jax.distributed.initialize()` | ✅ 完整 JAX 多进程初始化 |
| 数据采样 | 所有 rank 加载相同数据 | `DistributedSampler` 分片 |
| WandB | 所有 rank 写 (重复) | rank 0 only |
| 模型预设 | 固定 360M | 55M-2.5B 可配置 (11 种) |
| 训练模式 | Full AR only | AR / TBPTT / SWR 三选一 |
| Checkpoint | 完整恢复 | 支持部分恢复 (结构变化) |
| A1 独有 | Triton 订单簿匹配测试 | - |
| A2 独有 | - | SWR 算法 (`s5/swr.py`, 367行) |

### 关键 Insights

1. **A2 是 A1 的超集** — A2 `TRAIN_VERSION=v2` 等价 A1 的 Full AR，但额外提供 TBPTT 和 SWR
2. **pmap→shard_map 决定扩展天花板** — A2 的 2D mesh 是后续 C5 达到 16N 83% 效率的基础
3. **A1 的 `GOOG_2018_2022_combined` 有 test 泄漏风险** — CLAUDE.md 标注为错误数据集
4. **GOOG 完整数据路径**: `/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/` 下有 2016-2022 共 7 个年份目录

---

## 2026-02-23: Eval NCCL Deadlock at 32N — process_allgather is the killer

### ★ Insight: process_allgather 在大规模 eval 中会死锁
- `process_allgather` 每次调用触发 128-rank NCCL AllGather，**无内置 timeout**
- 在 eval 循环内每 batch 调用 2 次 → 28 batch = 56 次全局同步 → 死锁概率极高
- **MaxText（Google 官方）完全不用 process_allgather** — 改用 addressable_shards（零 NCCL）

### ★ Insight: 训练 collective 稳定但 eval collective 不稳定的原因
- 训练用 shard_map + pmean（XLA 编译图内，确定性执行）
- eval 用 process_allgather（Python 层面，依赖 rank 同步时序）
- XLA 内的 collective 由编译器调度，比 Python 层面的 collective 可靠得多

### ★ Insight: T5X 的 assert_equal(num_batches) 防御模式
- eval 循环前断言所有 host 的 batch 数一致
- 如果 DataLoader 分配不均 → 立即报错（而非死锁 20 分钟）
- 这是 Google 在生产中验证过的标准防御

---

## 2026-02-23: LR Sweep 32N — lr=5e-3 梯度爆炸

### 实验配置
- 32N (128 GPU), PER_GPU_BSZ=10, Global BSZ=1280, 40 epochs
- 分支: `ignore-times-shard-map`, ignore_times=True
- 对比 LR: 1e-3 / 3e-3 / 5e-3

### lr=5e-3 灾难性失败 (Job 2439703, W&B: qqz17itg)

| Epoch | Train Loss | Test Loss | Test Acc | Test PPL |
|-------|-----------|-----------|----------|----------|
| 1     | 2.16      | 1.72      | 67.3%    | 29.2     |
| 2     | **910.39**| 6.83      | 54.7%    | 116.0    |
| 3     | 24.67     | **136.46**| 55.2%    | 2319.8   |
| 4     | 18.31     | 3.71      | 53.7%    | 63.0     |
| 5     | 3.49      | 2.48      | 59.7%    | 42.2     |
| 6     | 2.35      | 2.15      | 60.7%    | 36.5     |

- **Epoch 2 梯度爆炸**: Train Loss 2.16 → 910 (420x spike)
- Warmup 结束后 (Epoch 1) 切入 cosine decay，LR 峰值 5e-3 太高
- 6 个 epoch 后仍远未恢复到 Epoch 1 水平 (Test Loss 2.15 vs 1.72)

### lr=1e-3 稳定收敛 (Job 2439704, W&B: r0ily8fd)

| Epoch | Train Loss | Test Loss | Test Acc | Test PPL |
|-------|-----------|-----------|----------|----------|
| 1     | 2.54      | 1.80      | 66.0%    | 30.5     |
| 2     | 1.68      | 1.68      | 67.7%    | 28.6     |
| 3     | 1.57      | 1.62      | 68.7%    | 27.6     |
| 4     | 1.56      | 1.62      | 68.7%    | 27.5     |
| 5     | 1.48      | 1.55      | **69.7%**| 26.4     |

- 持续改善，每 epoch 稳定下降，Epoch 5 时仍在改善

### 关键教训

1. **128 GPU (BSZ=1280) 的 LR 上界 < 5e-3**: Adam 的 v̂ 追踪梯度二阶矩有滞后，一次 gradient spike 就能引发连锁发散
2. **√κ scaling 法则的安全边界**: baseline LR=5e-4, κ=45.7 (BSZ 28→1280), √κ=6.76 → 理论 LR≈3.4e-3。5e-3 超过理论值 1.5x，直接爆炸
3. **Warmup 掩盖高 LR 风险**: Epoch 1 (warmup 阶段) LR 从 0 线性爬到 5e-3，loss 正常下降。Epoch 2 LR 在峰值开始 cosine decay 才暴露问题
4. **大 batch 训练的 LR sweet spot**: 对于 75M S5 SSM + BSZ=1280, 安全范围大约 1e-3 ~ 3e-3。需要等 3e-3 (Job 2439874) 结果确认上界

## 2026-02-23: G1 360M BSZ Sweep — XLA 内存非单调

### BSZ Sweep 结果 (401M params, GH200 85.5GB, MEM_FRACTION=0.90)

| BSZ | 结果 | 内存需求 | 稳态速度 (1N) | Job ID |
|-----|------|---------|--------------|--------|
| 2 | ✅ OK | < 71.6 GiB | 0.70 s/step | 2440057 |
| 3 | ❌ OOM | 76.28 GiB | — | 2440151 |
| 4 | ✅ OK | < 71.6 GiB | 1.36 s/step | 2440150 |
| 5 | ❌ OOM | 77.08 GiB | — | 2440158 |
| 6 | ❌ OOM | 89.09 GiB | — | 2440159 |
| 7 | ❌ OOM | 108.95 GiB | — | 2440160 |
| 8 | ❌ OOM | 210.15 GiB | — | 2439975 |

### ★ Key Insight: XLA 内存分配不是 BSZ 的单调函数
- BSZ=3 OOM (76.28 GiB) 但 BSZ=4 OK — XLA 编译器对不同 BSZ 选择不同的 memory layout/fusion 策略
- BSZ=3 和 BSZ=5 的 OOM 边界都在 ~76-77 GiB，刚好卡在 MEM_FRACTION=0.90 × 85.5GB ≈ 76.95 GB
- 代码库无 remat/gradient checkpointing，BSZ=4 是当前 401M 模型的实际上限

### G1 生产配置
- BSZ=4, Global BSZ=512 (32N), LR=7e-4, 分支 exp/G1-scale-up, commit f2b39ce3

## 2026-02-23: G6 Prodigy (Learning-Rate-Free Optimizer) 结论

### 实验配置
- Prodigy (optax.contrib.prodigy) vs AdamW, 11 个 job, 2N/32N, BSZ=8/10/12
- 分支: `exp/G6-auto-lr` (基于 `ignore-times-shard-map`), 代码改动仅 42 行
- 详细数据: `tasks/G6-auto-lr/findings.md`

### 核心发现
1. **Prodigy estim_lr 准确**: 2N gBSZ=64 时 estim_lr≈8.4e-4, 精确落入 AdamW 最佳区间 [5e-4, 1e-3]
2. **公平对比 Prodigy 略逊**: Tuned AdamW 74.41% Val Acc vs Prodigy 72.72% (差 1.7pp)
3. **速度零影响**: 4x optimizer state 不影响吞吐 (<1% 差异)
4. **稳定性风险**: Prodigy 有 loss spike (ep14: 1.39→5.48) 和结果方差高的问题
5. **Batch size scaling 偏保守**: gBSZ 20x 增大, estim_lr 仅 2x 增大 (vs sqrt 理论 4.47x)

### 最佳实践
- 用 Prodigy 做探索性实验, 读取 estim_lr 作为 AdamW LR 起点
- Production 训练用 AdamW + cosine decay, LR 参考 Prodigy 估计值
- 75M S5 SSM 最佳 LR: 2N→5e-4~1e-3, 32N→1e-3

---

## 2026-02-23: 32N NCCL 死锁完整修复 — Deep Research + 系统性验证

### 背景
32N (128 GPU) 训练间歇性 NCCL 死锁，影响多个 job:
- Job 2426449: Epoch 14, Step 356
- Job 2438407: Epoch 0, Batch 1
- Job 2439704: Epoch 6, Batch 58 (5 个健康 epoch 后)

### 方法论
3 个 parallel deep research agents (NCCL releases, JAX/XLA flags, Isambard/CSCS/ALCF docs) + HLO profiling + 5 个单变量验证 job。

---

### ★ Insight 1: NCCL 2.29.3 ARM CAS Weak Failure Bug (最重要发现)
- **NCCL 2.29.3** (2025-02-03) changelog: "Fix CAS usage in case of weak failure which was causing a hang on ARM"
- ARM 的 LDXR/STXR (Load-Exclusive/Store-Exclusive) 指令会 **spurious failure** — 这是 ARM 架构规范允许的行为
- NCCL < 2.29.3 的 CAS (Compare-And-Swap) 循环没有正确重试 spurious failure → **spin loop 永不退出** → 线程卡死
- GH200 = ARM Grace CPU + H200 GPU → **完美命中** 此 bug
- 概率性触发: P(fail) ≈ 1e-6/CAS op, 但 32N 训练每步 336 AllReduce × 数百万次原子操作 → 累积触发
- **修复**: `pip install nvidia-nccl-cu12==2.29.3` (base conda env)
- **验证**: Job 2447130 日志确认 `NCCL version 2.29.3+cuda12.9`

### ★ Insight 2: XLA nccl_stub.cc 动态加载机制 (误判纠正)
- 初始分析: `nm` 显示 xla_cuda_plugin.so 有 328 个 NCCL local symbols (`t ncclAllReduce`) → 误以为 NCCL 被静态编译
- **实际机制**: 这些 `t` 符号是 `nccl_stub.cc` 中的 **thin wrapper/stub**，不是完整 NCCL 实现
- 三阶段加载链:
  ```
  Stage 1: _load("nccl") → importlib.import_module("nvidia.nccl") → ctypes.cdll.LoadLibrary(绝对路径)
  Stage 2: xla_cuda_plugin.so 注册 → RPATH $ORIGIN/../../nvidia/nccl/lib
  Stage 3: nccl_stub.cc → dlopen("libnccl.so.2") → 复用 Stage 1 已加载的 handle
  ```
- **关键**: pip 升级 nvidia-nccl-cu12 即可改变运行时 NCCL 版本
- **LD_PRELOAD 也有效**: 在 Stage 1 之前加载，所有后续 dlopen 复用 PRELOAD 版本
- **教训**: `nm` 的 local symbol `t` 不等于静态编译 — 可能是 dlopen+dlsym wrapper

### ★ Insight 3: CXI Eager Message Race Condition (三大 HPC 中心共识)
- **CSCS Alps + Isambard + ALCF Polaris** 三方文档一致推荐:
  ```bash
  export FI_CXI_RDZV_GET_MIN=0       # 禁用 eager GET 最小尺寸
  export FI_CXI_RDZV_THRESHOLD=0     # 强制所有消息走 rendezvous
  export FI_CXI_RDZV_EAGER_SIZE=0    # rendezvous 内无 eager 数据
  ```
- **机制**: CXI NIC 的 eager buffer 在 128 GPU bursty traffic 下偶尔耗尽 → 后续 rendezvous 无法完成 → NCCL AllReduce 永久阻塞
- **我们之前完全没设置这三个变量** — 这是 CSCS/Isambard 文档标注为 "prevents NCCL timeouts" 的首要 workaround
- **验证**: Job 2447130 (32N, 4 epoch) 未挂起，性能无回归

### ★ Insight 4: 之前 37% 性能回归的误诊 (单变量测试方法论)
- **Job 2382150**: 同时开了 8 个 NCCL/CXI 变量 → 37% 回归 (9.78 → 13.36 s/step) → 全部注释掉
- **问题**: 无法确定哪个变量导致回归，把**无辜的 hang prevention 变量也一起禁用了**
- **后续单变量测试结果**:
  | 变量 | 影响 | 验证 Job |
  |------|------|---------|
  | NCCL_CROSS_NIC=1 | ❌ 性能杀手 | 2382150 |
  | NCCL_NET_GDR_LEVEL=PHB | ❌ 性能杀手 | 2382150 |
  | NCCL_PROTO=^LL128 | ❌ 23% 回归 | 2447647 |
  | FI_CXI_RDZV_*=0 | ✅ 无影响 | 2447130 |
  | FI_CXI_DISABLE_HOST_REGISTER=1 | ✅ 无影响 | 2447130 |
  | FI_MR_CACHE_MONITOR=userfaultfd | ✅ 无影响 | 2447130 |
  | MPICH_GPU_SUPPORT_ENABLED=0 | ✅ 无影响 | 2447130 |
- **教训**: 多变量同时测试 → 无法归因 → 必须单变量逐个验证

### ★ Insight 5: LL128 在 NCCL 2.29.3 + Slingshot 上是有益的 (反直觉)
- **CSCS/Isambard 文档**: "NCCL_PROTO=^LL128 — LL128 typically performs worse on Slingshot"
- **但这是基于旧版 NCCL (< 2.27)**: NCCL 2.27+ 对 LL128 做了多 NIC 拓扑优化
- **实测 (32N, Job 2447647 vs 2447130)**:
  | 配置 | 稳态速度 (step 250-300) |
  |------|------------------------|
  | LL128 默认开启 (baseline) | 0.91 s/step |
  | NCCL_PROTO=^LL128 (禁用) | 1.12 s/step |
  | **差异** | **+23% 更慢** |
- **为什么有帮助**: LL128 是低延迟协议，适合 medium-sized messages; shard_map 2D mesh 的 inter-node pmean 产生大量 medium messages
- **教训**: HPC 文档的建议可能基于旧版本; 新版本可能已修复; 必须在自己的环境实测

### ★ Insight 6: BlueConnect 与 shard_map 2D Mesh 不兼容 (3x 回归)
- `--xla_gpu_all_reduce_blueconnect_num_devices_per_host=4`: XLA 自动将 AllReduce 分解为 ReduceScatter + AllReduce + AllGather
- **问题**: shard_map 已经手动做了 2-level 分层 (pmean('gpus') intra-node + pmean('nodes') inter-node)
- **双重分层**: BlueConnect 再次分解已经是 "部分" 的 AllReduce → 不必要的通信 → 3x 回归 (3.55 vs 1.18 s/step)
- **pmean('nodes') 的特殊情况**: 32-rank inter-node group 中每 host 只有 1 GPU → BlueConnect 的 intra-host ReduceScatter 找不到同 host 伙伴 → 退化或死锁
- **教训**: BlueConnect 和 shard_map 2D mesh 是互斥方案; 用了 shard_map 就不能开 BlueConnect
- **验证**: Job 2440967 (3.55 s/step)，Commit 4f57f24b 移除

### ★ Insight 7: HLO Profiling 推翻 eval 假说
- **假说**: eval out_shardings=(None,None,None) 强制 XLA 插入 128-rank flat AllReduce → eval 期间 NCCL communicator 切换 → 死锁
- **HLO 验证 (Job 2440089)**: eval_step **零 NCCL collectives**; train_step 有 336 AllReduce/step
- **train 336 AllReduce 组成**: 169 intra-node (replica_groups: {0,1,2,3},{4,5,6,7},...) + 167 inter-node (replica_groups: {0,4,8,...},{1,5,9,...},...)
- **累积量化**: 32N × 336 × 549 steps/epoch × 6 epochs = ~1.1M collectives — 即使 P(single fail) = 1e-6, 累积 P(at least one) ≈ 67%
- **教训**: 假说必须用 HLO dump 验证; XLA 编译器的行为不能靠代码推断

### ★ Insight 8: NCCL_TIMEOUT 对 CXI 层面 hang 无效
- `NCCL_TIMEOUT=3600` 在 32N 上未能阻止 6h hang (Job 2426449)
- **原因**: CXI 层的 hang 发生在 NCCL 之下 — `libfabric fi_cq_read()` 永远等不到 completion event
- NCCL 认为 "传输仍在进行"（not timeout, just waiting），所以 NCCL_TIMEOUT 不触发
- **正确方案**: XLA 层面的 `--xla_gpu_nccl_terminate_on_error=true` + `--xla_gpu_nccl_termination_timeout_seconds=600`
- 这让 XLA 在 NCCL rendezvous 超时后直接终止进程（而非永久挂起）

### ★ Insight 9: Auto-Resume 设计哲学
- 32N 下 NCCL 死锁概率永远 > 0 (即使修复了所有已知根因)
- **哲学**: 与其消灭死锁，不如让训练能自动恢复
- **实现**: batch script 末尾检查 srun exit code:
  - 非零退出 + 非 CURTAIL_EPOCHS + 非 NO_AUTO_RESUME → 自动 `sbatch --dependency=afterany:$SLURM_JOB_ID`
  - 从最新 checkpoint 恢复: `ls -1d checkpoints/epoch_* | sort -V | tail -1`
  - 最大 3 次连续 auto-resume (NO_AUTO_RESUME_DEPTH 计数器)
- **srun --kill-on-bad-exit=1**: watchdog 杀一个 rank → srun 杀全部 127 个 → 触发 auto-resume

---

### 完整 CXI Resilience 配置 (>=8N 生产配置)

```bash
# --- 既有 resilience ---
FI_CXI_RDZV_RETRIES=100           # 默认5, 容忍瞬态 fabric 错误
FI_CXI_OFLOW_BUF_SIZE=8388608     # 8MB overflow buffer
FI_CXI_REQ_BUF_SIZE=8388608       # 8MB request buffer

# --- CXI hang prevention (CSCS + Isambard + ALCF 共识) ---
FI_CXI_RDZV_GET_MIN=0             # 禁用 eager GET
FI_CXI_RDZV_THRESHOLD=0           # 强制 rendezvous-only
FI_CXI_RDZV_EAGER_SIZE=0          # rendezvous 无 eager data
FI_CXI_RDZV_PROTO=alt_read        # 替代 read 协议 (ALCF 验证 540 nodes)

# --- Host register deadlock 防护 ---
FI_CXI_DISABLE_HOST_REGISTER=1    # 防止 host buffer 注册死锁
FI_MR_CACHE_MONITOR=userfaultfd   # MR 缓存监控

# --- GPU-aware MPI 冲突防护 ---
MPICH_GPU_SUPPORT_ENABLED=0       # "easily leads to deadlocks" - CSCS
```

### XLA NCCL Flags (生产配置)

```bash
--xla_gpu_nccl_terminate_on_error=true         # NCCL 错误时快速失败
--xla_gpu_nccl_termination_timeout_seconds=600  # 10min NCCL 超时
--xla_gpu_all_reduce_combine_threshold_bytes=134217728  # 128MB 合并
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_highest_priority_async_stream=true
```

### 有害的 NCCL/XLA 配置速查表

| 配置 | 影响 | 验证 Job | 绝不使用 |
|------|------|---------|---------|
| NCCL_P2P_DISABLE=1 | 25x 减速 | — | ★★★★★ |
| xla_gpu_autotune_level=0 | 10-24x 退化 | — | ★★★★★ |
| NCCL_CROSS_NIC=1 | 37% 退化 | 2382150 | ★★★★ |
| NCCL_NET_GDR_LEVEL=PHB | 37% 退化 | 2382150 | ★★★★ |
| BlueConnect (blueconnect_num_devices_per_host=4) | 3x 回归 | 2440967 | ★★★★ (shard_map) |
| NCCL_PROTO=^LL128 | 23% 回归 | 2447647 | ★★★ (NCCL 2.29.3) |
| NCCL_ALGO=TREE | crash / 2.2x 慢 | — | ★★★ |

### 根因覆盖率总结

| 根因 | 概率 | 修复 | 验证 Job |
|------|------|------|---------|
| NCCL ARM CAS weak failure | ~50% | pip nvidia-nccl-cu12==2.29.3 | 2447130 |
| CXI eager message race | ~25% | FI_CXI_RDZV_*=0 | 2447130 |
| Host register deadlock | ~10% | DISABLE_HOST_REGISTER=1 | 2447130 |
| Multi-communicator ordering | ~5% | NCCL_LAUNCH_ORDER_IMPLICIT=1 | (已有) |
| LL128 协议问题 | ~5% | **不修复** — LL128 有益 | 2447647 |
| CXI transient errors | ~5% | RDZV_RETRIES=100 + auto-resume | (已有) |
| **总覆盖率** | **~95%** | | |

### Commits

| Commit | 描述 |
|--------|------|
| 6f79e3aa | CXI hang prevention + NCCL fail-fast + auto-resume |
| 4f57f24b | 移除 BlueConnect (3x 回归) |
| 13ec875a | ^LL128 注释更新 (23% 回归, 不启用) |
| 050989cc | LD_PRELOAD NCCL 验证 (exp/nccl-version-fix worktree) |

### 详细报告索引

| 报告 | 路径 |
|------|------|
| NCCL 加载机制分析 | `subagent_nccl_version_fix_20260223.md` |
| ^LL128 Benchmark | `subagent_nccl_proto_ll128_20260223.md` |
| Eval 死锁研究 | `subagent_nccl_eval_deadlock_research_20260222.md` |
| NCCL task plan | `tasks/nccl/task_plan.md` |
| NCCL findings | `tasks/nccl/findings.md` |
| NCCL progress | `tasks/nccl/progress.md` |

## 2026-02-24: NCCL Thunk Init 死锁 — 从误诊到根因修复

### 调试时间线

```
假说 1: wandb 0.25.0 GPU 探测 → 排除 (Job 2451505 零 wandb 仍死锁)
假说 2: XLA timeout flag 引起 → 排除 (有/无 flag 都死锁)
假说 3: 节点分配 → 排除 (相同节点有通过也有失败)
根因: NCCL 2.28.9 ARM CAS bug → ✅ (NCCL 2.29.3 release note 确认)
```

### ★ Insight: LD_LIBRARY_PATH vs RPATH 优先级陷阱

Linux 动态链接器的搜索顺序: **RPATH > LD_LIBRARY_PATH > RUNPATH > ld.so.cache**

`xla_cuda_plugin.so` 编译时嵌入 RPATH 指向 conda env 的 `nvidia/nccl/lib`。
任何 LD_LIBRARY_PATH override 都无效 — linker 先找到 RPATH 就不再搜索。

**验证方法**: `readelf -d xla_cuda_plugin.so | grep RPATH`
**绕过方法**: 直接替换 RPATH 指向的文件 (pip install 到同一 env) 或 `patchelf --set-rpath`

### ★ Insight: Conda Env 退役 → Base Env 迁移

lob env 不可修的三重问题:
1. jax 0.6.1 (API 不兼容 0.9.0.1 代码)
2. pip shebang 指向已删路径 (`rm -rf miniforge3`)
3. Lustre project quota 1MB (pip install 无法写入)

Base env 已包含所有正确版本 (jax 0.9.0.1 + NCCL 2.29.3 + wandb 0.21.3)。
切换后 RPATH 自动指向 base env NCCL 2.29.3 — 不需要任何 override。

**Commit**: `e61596c2` | **验证**: Job 2451587

### ★ Insight: 误诊的代价

| 误诊 | 采取的错误行动 | 浪费 |
|------|--------------|------|
| "wandb 0.25.0 GPU 探测" | 延迟 import、WANDB__DISABLE_STATS、PYTHONPATH hack | 5+ jobs, ~3h GPU |
| "XLA timeout flag 导致" | 移除有用的 1500s timeout | 调试时间 |
| "节点分配问题" | 请求 --contiguous | 排队时间 |

**教训**: 排除法比猜测法有效 — Job 2451505 (零 wandb) 一步排除了最大嫌疑

## XLA Thunk Init Rendezvous Timeout (2026-02-24)

### 问题
360M 模型 (G1) 在 32N/128GPU 上 XLA thunk 初始化阶段崩溃。所有 32 节点完全相同的模式：GPU 0,1,2 到达 rendezvous，GPU 3 缺席。30 秒超时后 SIGABRT。

### 根因
- XLA thunk init rendezvous 默认 30 秒超时
- 360M 模型计算图比 75M 大 ~4.8x，thunk init 时间 ~28-32 秒（边界条件）
- GPU 3 因 NCCL channel 分配顺序总是最后完成，在慢节点上超时
- Kangli job 2441019 (不同节点分配 nid010064) 刚好通过 (~28s)
- Job 2451140/2451366 (nid010000) 超时 (~31s)

### 修复
```bash
# 在 XLA_FLAGS 中添加（jaxlib 0.9.0.1 已验证接受）
--xla_gpu_executable_terminate_timeout_seconds=1500
```

### 可用的 XLA Timeout Flags (jaxlib 0.9.0.1)
| Flag | 默认值 | 说明 |
|------|--------|------|
| `xla_gpu_executable_terminate_timeout_seconds` | 30 | thunk init rendezvous 终止超时 |
| `xla_gpu_executable_warn_stuck_timeout_seconds` | 10 | thunk init 警告超时 |
| `xla_gpu_first_collective_call_terminate_timeout_seconds` | 40 | 第一次 collective call 终止超时 |
| `xla_gpu_first_collective_call_warn_stuck_timeout_seconds` | 20 | 第一次 collective call 警告超时 |
| `xla_gpu_nccl_termination_timeout_seconds` | -1 | NCCL 操作终止超时 |

### 关键教训
1. **XLA rendezvous timeout 不是硬编码的** — 有 XLA flag 可以调整，只是鲜为人知
2. **大模型 + 多 GPU 的 thunk init 时间不可忽略** — 75M 在 30s 内轻松完成，360M 刚好在边界
3. **GPU 3 总是最慢** — 单进程 4 GPU 架构中，NCCL channel 分配有顺序性，最后一个 GPU 延迟到达
4. **节点分配影响 thunk init 速度** — 不同的物理节点可能有 ~3-5 秒的差异
5. **增大 timeout 只影响一次性初始化，不影响稳态训练速度和 NCCL 死锁检测**

## 2026-02-24: Git Worktree 权限双层结构

### ★ Insight: 每个 worktree 有两个位置需要写权限

1. **实际工作目录** `experiments/exp_*/` — 代码文件、checkpoints
2. **主 repo 的 `.git/worktrees/<name>/`** 元数据目录 — index, HEAD, refs/ 等

任何 git 操作（包括 `git status`）都需要在 `.git/worktrees/<name>/` 内创建 `index.lock`。
只修实际工作目录的权限而不修 `.git/worktrees/` 会导致"目录能写文件但 git 命令报错"的诡异现象。

### 修复命令
```bash
# 两个位置都需要递归 g+rw
chmod -R g+rw /lus/.../AlphaTrade/experiments/
chmod -R g+rw /lus/.../AlphaTrade/LOBS5/.git/worktrees/
# 确保 setgid 位继承组
find /lus/.../AlphaTrade/experiments/ -type d -exec chmod g+s {} +
```

### Future Worktree 权限自动化
- `git worktree add` 继承默认 umask (022) → 新目录为 2755 (无 g+w)
- 要永久修复: 在 shell profile 设 `umask 002` 或在 sbatch 脚本开头加 `umask 002`
- 否则每次创建新 worktree 后需要重跑 chmod

---

## 32N 实验记录规则 (2026-02-24)

所有同时满足 **32 nodes** 且 **非 KTL (Keep Time Large, 即 IGNORE_TIMES=True)** 的实验，必须记录到 `tasks/large-train/experiments.md`。

### 表格格式
```
| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
```

### 规则
- **Task 列**: 标注来源 task（如 `G0-baselines-sweeplr`, `G3-ignore-time` 等）
- **KTL 实验** (IGNORE_TIMES=False) 也记录在同一文件，用 IGNORE_TIMES 列区分
- Job 提交后立即添加行（PENDING 状态），跑完后补充 Best metrics 和 W&B URL
- 文件路径: `tasks/large-train/experiments.md`

---

### 更正：Git Worktree 权限是三层结构（非双层）
- 第三层：`.git/logs/refs/heads/` 和 `.git/refs/heads/` — 分支 reflog 和 ref 指针
- commit/checkout 等操作需要写 `.git/logs/refs/heads/exp/<branch>`
- **正确做法**: `chmod -R g+rw .git/` 一次修完整个 `.git/`，不要逐子目录修
