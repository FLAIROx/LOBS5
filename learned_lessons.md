# Learned Lessons

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

### Epoch 间 OOM 的真实根因 + 正确解法
- OOM 是 `PjRtLoadedExecutable::Execute()` 执行期申请 71.62 GiB 连续块失败（BFC allocator 碎片化）
- **正确解法（来自 ssm_stable）**: `TF_GPU_ALLOCATOR=cuda_malloc_async`
  - 使用 CUDA 异步分配器代替默认 BFC，从根本上避免碎片化
  - ssm_stable 在 2026-01-04 加入了这个 flag
  - 配合 `gc.collect() + jax.clear_caches()` 即可，**不需要** del/recreate jit 函数
- **错误做法** (del+recreate jit): ssm_stable 从未这样做，是错误方向
- **错误做法** (降低 MEM_FRACTION 0.80): 不必要，有了 cuda_malloc_async 用 0.90 即可
- `del ckpt` 仍然保留（释放 checkpoint state 副本，与 OOM 无关但是好实践）
- 参考: `subagent_ssm_stable_memory_management_20260218.md`
