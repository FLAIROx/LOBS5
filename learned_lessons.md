# Learned Lessons & Insights

This file captures educational insights and lessons learned during development.

---

## 2026-01-06

### Claude Code Memory 系统

- `CLAUDE.md` 文件是 Claude Code 的持久化记忆，每次对话开始时会自动加载
- 有两个层级：全局 (`~/.claude/CLAUDE.md`) 和项目级 (`项目根目录/CLAUDE.md`)
- 项目级的指令会覆盖全局指令，适合存放项目特定的配置

### 安全规则的重要性

- 删除操作是不可逆的，特别是在 HPC 环境中，数据恢复困难
- 安全规则可以阻止自动执行 `rm`、`git rm`、`rmdir` 等危险命令
- 即使是清理临时文件，也应该先确认

### HPC 集群架构

- **Login 节点** 是共享资源，仅用于编辑代码、提交作业、查看结果
- 在 login 节点运行计算任务会影响其他用户，可能导致账户被限制
- **sbatch** 将作业提交到调度队列，由 SLURM 分配计算节点资源

### 当前集群配置

- ARM 架构（非 x86），某些软件可能需要专门编译
- 4x GPU per node - 适合数据并行训练
- 856 GB 内存 - 可以加载大型数据集到内存
- 288 cores - 高度并行的数据预处理能力

### Git Commit 最佳实践

**Conventional Commits 格式：**
```
<type>(<scope>): <short description>

[optional body - 详细说明]
[optional footer - 关联 issue 等]
```

**常用类型：**
| Type | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `refactor` | 重构（不改变功能） |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `chore` | 构建/工具/依赖更新 |
| `style` | 代码格式（不影响逻辑） |

**示例：**
- `feat(model): add gradient checkpointing for S5 layer`
- `fix(dataloader): resolve memory leak in batch prefetching`
- `perf(training): optimize JAX compilation with donate_argnums`
- `refactor(config): split hyperparameters into separate files`

**原子性原则：**
- 每个 commit 应该是一个逻辑完整的变更
- 可以独立理解、独立 revert
- 避免在一个 commit 中混合不同类型的修改

---

### E2 Multi-GPU Validation Finding

**Issue:** ESTrainer's `jax.vmap` does NOT distribute across multiple GPUs automatically.

**Evidence (Job 1836224):**
- 4 GPUs detected and accessible
- `jax.pmap` distribution test PASS
- ES Training: Memory only on Device 0 (477 MB), Devices 1-3 at 0 B

**Root Cause:**
`jax.vmap` vectorizes operations but executes them on a single device. It's designed for batching, not multi-device distribution.

**Solutions for True Multi-GPU Training:**
1. **`jax.pmap`**: Parallel map that shards data across devices
2. **`jax.device_put_sharded`**: Explicit device placement
3. **`jax.sharding`** (JAX 0.4+): Flexible sharding API

### Subagent 输出管理

**为什么使用单独文件：**
- 每个分析任务的输出可能很长，混在一起难以查阅
- 单独文件便于版本控制和 diff
- 可以按主题/日期快速定位历史分析

**命名规范：**
```
subagent_<主题>_<日期YYYYMMDD>.md
```

**示例：**
- `subagent_multi_gpu_analysis_20260106.md`
- `subagent_memory_profiling_20260106.md`
- `subagent_xla_compilation_debug_20260106.md`

**优点：**
- 持久化记录，不会随对话丢失
- 便于团队共享和 review
- 可以作为项目文档的一部分

**Example Pattern:**
```python
# Single GPU (current):
fitnesses, infos = jax.vmap(eval_fn)(keys, thread_ids)

# Multi-GPU with pmap:
n_devices = jax.local_device_count()
batch_per_device = n_threads // n_devices
keys = keys.reshape(n_devices, batch_per_device, ...)
thread_ids = thread_ids.reshape(n_devices, batch_per_device)
fitnesses, infos = jax.pmap(jax.vmap(eval_fn))(keys, thread_ids)
```

---

### Python Output Buffering in HPC Jobs

**Issue:** Python stdout is buffered by default, causing logs to appear delayed or not at all in SLURM job output files.

**Symptoms:**
- Job appears "hung" with no output
- Output appears all at once when job ends
- Print statements don't show progress in real-time

**Solutions:**
1. `python -u script.py` (unbuffered output)
2. `print(..., flush=True)` (flush after each print)
3. `PYTHONUNBUFFERED=1` environment variable
4. `sys.stdout.flush()` after important prints

---

### XLA Compilation Time Scaling

**Issue:** XLA compilation time scales superlinearly with graph complexity.

**Evidence:**
- n_steps=100 with 256 threads: 55+ min compile time (hangs)
- n_steps=10 with 256 threads: ~150s first epoch (acceptable)

**Formula (rough estimate):**
```
Compile_time ≈ O(n_threads × n_steps × graph_complexity)
```

**Best Practices:**
1. Start with small n_steps for testing
2. Use `jax.jit` with `donate_argnums` to reduce memory
3. Avoid deeply nested `jax.lax.scan` operations
4. Profile with `JAX_LOG_COMPILES=1`

---

### Compilation Optimization Benchmark Results (Job 1839408)

**Date**: 2026-01-06

**Implemented Optimizations (G4/G5/G2):**
- G4: Persistent JAX compilation cache (`~/.cache/es_lobs5_jax_compilation`)
- G5: Extract self.xxx references in simulate_episode
- G2: Replace self references in nested functions (step_fn, historical_replay_step, world_model_step)

**Benchmark Results:**

| Test | Status | Key Metrics |
|------|--------|-------------|
| I1 (Compilation Time) | **PASS** | Epoch 0: 158s, Epochs 1-4: ~78s, CV=0.66% |
| I2 (Recompilation) | **PASS** | No recompilation in epochs 1-4 |
| I3 (Persistent Cache) | **PASS** | 1.07x speedup, 127 new cache files |
| I4 (Multi-GPU) | **PASS** | 4 GPUs detected, memory on GPU 0 only (expected with vmap) |

**Key Findings:**
1. **50% reduction** in subsequent epoch time (from ~150s to ~78s)
2. **No recompilation** detected during training
3. **Persistent cache working** - cache grew to 495 files (13MB)
4. **Multi-GPU not utilized** with current vmap implementation - needs H1-H3 shard_map

**Next Steps:**
- G1: AOT compilation for eval_batch
- H1-H3: shard_map multi-GPU implementation

---
