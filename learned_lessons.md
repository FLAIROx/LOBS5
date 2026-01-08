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
- ~~G1: AOT compilation for eval_batch~~ **COMPLETED**
- ~~H1-H3: shard_map multi-GPU implementation~~ **代码完成，运行时问题待解决**

---

### G1 AOT Compilation Results (Job 1839973)

**Date**: 2026-01-06

**Implementation:**
- Added `_build_eval_thread()` - returns pure function with captured static params
- Added `_compile_eval_batch()` - JIT pre-compiles vmap(eval_thread)
- Modified `train_epoch` to use pre-compiled function

**Benchmark Results:**

| Metric | Before G1 | After G1 | Improvement |
|--------|-----------|----------|-------------|
| Epoch 0 (compile) | ~158s | **76.64s** | 52% faster |
| Epoch 1+ (cached) | ~78s | **~9.4s** | **88% faster** |
| Compile:Execute ratio | 2.0x | **8.1x** | 4x better |
| Timing CV | 0.66% | 1.19% | Stable |

**Key Findings:**
1. **Pre-compiled JIT function** eliminates per-epoch `partial` creation overhead
2. Combined with G4 persistent cache, even Epoch 0 is faster
3. **Subsequent epochs now dominated by execution**, not compilation

**Commit**: `1bad9bd feat(es): implement G1 AOT compilation for eval_batch`

---

### E2 Large-Scale Validation Results (Job 1840612)

**Date**: 2026-01-06
**Configuration**: 256 threads × 50 steps × 5 epochs (旧版代码，无 shard_map)

**Results**:

| Epoch | Time | Throughput | Mean Fitness |
|-------|------|------------|--------------|
| 0 | 430.22s | 0.6 th/s | -10.25 |
| 1 | 361.28s | 0.7 th/s | -10.23 |
| 2 | 340.54s | 0.8 th/s | -10.21 |
| 3 | 358.96s | 0.7 th/s | -10.22 |
| 4 | 358.91s | 0.7 th/s | -10.25 |

**Key Findings:**
1. **大规模计算图可行**: 256×50 训练稳定完成
2. G1 优化前 E2 配置 `n_steps=10` (避免 55+ min 编译)
3. G1 优化后可扩展到 `n_steps=50`
4. Epoch 时间: Epoch 0 (430s) vs Epoch 1+ (~355s 平均)
5. 单 GPU 模式 (vmap)，未启用 multi-GPU

**Status**: PASS (training), FAIL (memory_distributed - 预期，未用 shard_map)

---

### H Series: Multi-GPU shard_map Implementation & Challenges

**Date**: 2026-01-06

#### H3: Mesh Configuration - ✅ COMPLETED

**Implementation** (Commit `73fb882`):
```python
# Lines 383-391 in __init__
self._n_devices = len(jax.devices())
self._mesh = Mesh(jax.devices(), ('data',))
print(f"[H3] Created mesh with {self._n_devices} devices")
```

**Status**: ✅ Working
- Mesh creation successful
- 4 devices detected
- `_shard_to_mesh()` helper method added

---

#### H1: shard_map Implementation - ⚠️ CODE COMPLETE, RUNTIME BLOCKED

**Implementation** (Commits `73fb882`, `384dc33`, `0a7874a`, `6f150bb`):

1. **Added imports**:
   ```python
   from jax.experimental.shard_map import shard_map
   from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
   ```

2. **Modified `_compile_eval_batch()`**:
   ```python
   if n_devices > 1 and hasattr(self, '_mesh'):
       # shard_map + vmap pattern
       sharded_eval = shard_map(
           jax.vmap(_eval_thread, in_axes=(None, None, 0, 0, None, None, None)),
           mesh=self._mesh,
           in_specs=(
               P(),        # noiser_params: replicated
               P(),        # params: replicated
               P('data'),  # keys: sharded
               P('data'),  # thread_ids: sharded
               P(),        # epoch: replicated
               P(),        # initial_sim_state: replicated
               P(),        # initial_msg_history: replicated
           ),
           out_specs=(P('data'), P('data')),
           check_rep=False,  # Due to complex scan carry types
       )
   ```

3. **Added divisibility check in `train_epoch()`**:
   ```python
   assert n_threads % n_devices == 0, f"[H1 ERROR] n_threads must be divisible by n_devices"
   ```

4. **Added params replication**:
   ```python
   noiser_params_rep = jax.device_put(self.noiser_params, NamedSharding(self._mesh, P()))
   params_rep = jax.device_put(self.lobs5_init.params, NamedSharding(self._mesh, P()))
   ```

**Status**: ⚠️ CODE COMPLETE, RUNTIME BLOCKED

---

#### H2: pvary for scan Compatibility - ⚠️ PARTIAL FIX, DEEPER ISSUES

**Problem Encountered**:
```
Error: Received incompatible devices for jitted computation
Got argument noiser_params['sigma'] with device ids [0, 1, 2, 3] (from shard_map)
but pjit inside jit expects device ids [0] (single device)
Location: transform_L2_state_wrapper at es_trainer.py:318
```

**Attempted Fixes**:

1. **pvary for scan carry** (Commit `384dc33`):
   - Added `in_shard_map` parameter to `simulate_episode`
   - Added `maybe_pvary()` and `maybe_pvary_tree()` helpers
   - Applied pvary to 4 scan initial carry values
   - **Result**: New error - "Collective pvary must be applied to non-varying type"

2. **check_rep=False** (Commit `0a7874a`):
   - Disabled VMA (Varying Manual Axis) checking in shard_map
   - **Result**: Progressed to device incompatibility error

3. **Params replication** (Commit `6f150bb`):
   - Replicated params/noiser_params to all devices before shard_map call
   - **Result**: New error - internal pjit still expects single device [0]

---

#### Root Cause Analysis

**The Core Problem**:
```
                    shard_map (outer)
                    ↓
      Params on [0,1,2,3] (replicated)
                    ↓
              eval_thread (vmap)
                    ↓
            simulate_episode
                    ↓
      transform_L2_state_wrapper (内部有隐式 pjit)
                    ↓
            ❌ Expects params on [0] only
```

**Technical Details**:

1. **Nested JIT Conflict**:
   - `shard_map` distributes computation across devices [0,1,2,3]
   - Each device receives replicated params (on all devices via `P()`)
   - But `transform_L2_state_wrapper` and other helper functions have **隐式 pjit**
   - These pjit assume **单设备执行**，与 shard_map 的多设备上下文冲突

2. **pvary Double-Application Issue**:
   - shard_map 自动标记 sharded inputs 为 varying
   - 手动 pvary 再次标记已 varying 的值 → 错误
   - 但不用 pvary → scan carry type mismatch

3. **scan + shard_map Complexity**:
   - `simulate_episode` 有嵌套 scan (4 个)
   - 每个 scan 的 carry 包含复杂 pytree (hiddens, sim_state)
   - JAX 的 VMA (Varying Manual Axis) 系统要求精确的类型声明

---

#### Potential Solutions

**Option 1: Use pmap Instead of shard_map**
```python
# pmap is older but simpler API
pmapped_eval = jax.pmap(
    jax.vmap(_eval_thread, ...),
    axis_name='device'
)
```
- **Pros**: More mature, better compatibility with nested JIT
- **Cons**: Deprecated, shard_map is the future

**Option 2: Inline All Functions into shard_map Body**
- Remove all internal `jax.jit` decorators
- Inline `transform_L2_state_wrapper`, `get_mid_price`, etc.
- **Pros**: Eliminates nested JIT conflicts
- **Cons**: Massive refactoring, loses code modularity

**Option 3: Use Manual Device Assignment**
```python
# Manually assign threads to devices without shard_map
devices = jax.devices()
results = []
for i, device in enumerate(devices):
    with jax.default_device(device):
        threads_per_device = n_threads // len(devices)
        result = eval_threads_on_device(...)
        results.append(result)
fitnesses = jnp.concatenate(results)
```
- **Pros**: Full control, no shard_map complexity
- **Cons**: Manual device management, less elegant

**Option 4: Relax JIT Constraints**
- Add `with jax.disable_jit()` in specific sections
- Use `static_argnums` more aggressively
- **Pros**: Minimal code changes
- **Cons**: May hurt performance

**Recommended Approach**:
Start with **Option 1 (pmap)** as it requires minimal changes and is proven to work with complex nested structures.

---

#### H2 Validation Test Results (Multiple Attempts)

| Job | Config | Result | Error |
|-----|--------|--------|-------|
| 1841316 | First attempt | FAIL | pvary type mismatch |
| 1841728 | With pvary helpers | FAIL | Double-varying error |
| 1841736 | check_rep=False | FAIL | Device incompatibility (params) |
| 1841749 | Params replication | FAIL | Device incompatibility (internal pjit) |

**Common Pattern**:
- Initialization succeeds
- Mesh and shard_map setup works
- Fails at **first epoch** when internal helper functions are called
- Error location: `transform_L2_state_wrapper` and similar JIT-compiled helpers

---

#### Lessons Learned

1. **shard_map + 嵌套 JIT 很复杂**:
   - shard_map 假设函数体中所有操作都知道多设备上下文
   - 隐式 JIT 装饰器（如 Flax layers 内部的 pjit）不兼容
   - 需要 "全有或全无" - 要么完全内联，要么完全单设备

2. **pvary 的双重性质**:
   - shard_map **自动**标记 sharded inputs 为 varying
   - scan carry 需要**手动** pvary 标记
   - 但不能对已 varying 的值再次 pvary

3. **设备放置传播**:
   - `jax.device_put(x, NamedSharding(...))` 复制到所有设备
   - 但这不会改变内部函数的设备期望
   - 内部 pjit 仍然假设单设备执行

4. **check_rep=False 的作用**:
   - 禁用 VMA（Varying Manual Axis）一致性检查
   - 允许 scan carry type mismatch
   - 但不解决设备放置冲突

---

#### Next Steps for Multi-GPU

**Short-term** (推荐):
- 尝试 `pmap` 替代 `shard_map`
- pmap API 更简单，与嵌套 JIT 兼容性更好

**Long-term** (如需完整 shard_map):
- 深度重构: 移除所有隐式 JIT，内联到 shard_map 函数体
- 或等待 JAX 改进 shard_map + nested JIT 支持

---

## 2026-01-07

### Constrained Decoding 与 Token Distribution 问题

**Date**: 2026-01-07

#### 问题背景

ES-LOBS5 中 policy 生成的订单出现严重分布偏差：
- 历史订单: size mean=94, median=41
- Policy 订单: size mean=5786, median=6239 (60x 偏差!)
- 时间间隔也异常: 历史 0.0009s vs policy 534s

#### 根本原因分析

1. **Constrained Decoding 正确工作**: Token 100% 在有效范围内 (e.g., size_digit 1008-1107)

2. **模型输出近似均匀分布**:
   - 在 100 个有效 size token 中，模型输出的 log_probs 几乎均匀
   - 随机采样给出期望值 = (0+99)/2 = 50
   - 解码后 size = 50×100 + 50 = 5050 (接近观察到的 5786)

3. **Temperature T=0.1 测试失败**:
   - 本意: 让 T<1 锐化分布，增加高概率 token 权重
   - 结果: 让结果更糟糕!
     - Size: 7868 (更大!)
     - Event Type: 77.6% cancel (历史只有 0.25%)
     - Direction: 95.6% buy (历史 48.5%)
   - 原因: Temperature 放大了模型的**错误峰值偏好**

#### 关键洞察

**语言模型不理解 LOB 订单语义**:
- 模型把 token 当作"语言"而非"结构化金融数据"
- 即使使用正确的 vocabulary 约束，模型内部不理解:
  - size=100 vs size=10000 的经济含义差异
  - event_type=cancel 的市场影响
  - 合理的订单时间间隔

**数据问题排除**:
- 训练数据 (2022): size mean=11.4, 95% < 100
- 测试数据 (JAN2023): size mean=85.6, 51% < 100
- 都远小于 policy 输出的 5786

#### 解决方向

1. **Fine-tune on Task-Specific Data**:
   - 在 ES 任务数据上微调模型，让它学习正确的分布

2. **Constrain Output Distribution**:
   - 不只约束 token 范围，还约束 token 分布
   - 可以用历史分布作为先验

3. **Hybrid Approach**:
   - 对 event_type, direction 等类别字段: 使用规则
   - 对 price, size 等连续字段: 使用模型

4. **调整训练目标**:
   - 当前模型可能训练于 next-token prediction
   - 需要加入订单语义相关的损失函数

#### 代码修改

```python
# es_trainer.py: 添加 temperature 参数支持
temperature = getattr(config, 'temperature', 1.0)  # 默认 1.0

# Apply temperature scaling
scaled_log_probs = masked_log_probs / temperature
next_token = jax.random.categorical(sample_key, scaled_log_probs)
```

**注意**: T=0.1 测试证明单纯调整 temperature 无法解决问题。需要从模型训练或分布约束层面解决。

---

---

## 2026-01-08

### 22-Token vs 24-Token Mode 系统性修复

**Date**: 2026-01-08

#### 问题背景

Checkpoint 使用 24-token mode，但 inference 代码按 22-token 写的。连续修复了 12+ 个 bug，说明这是**系统性架构问题**，需要项目化处理。

**Token Mode 差异**:

| 属性 | 22-token | 24-token |
|------|----------|----------|
| MSG_LEN | 22 | 24 |
| size 编码 | 1 token (base-10000) | 2 tokens (base-100) |
| Vocab Size | 12,012 | 2,112 |
| 编码器 key | `'size'` | `'size_digit'` |

#### 根本原因

**Class-Level State 问题**:
```python
class Message_Tokenizer:
    TOK_LENS = TOK_LENS_22  # 可变类变量 - 问题根源
    MSG_LEN = np.sum(TOK_LENS)  # 在类定义时计算

    @classmethod
    def set_token_mode(cls, token_mode):
        cls.TOK_LENS = ...  # 修改全局状态
```

**后果**:
1. Import 顺序影响结果
2. 不同模块可能看到不同的 `MSG_LEN`
3. Module-level 计算（如 `REF_LEN`）使用默认值

#### TDD 方法解决

**Step 1: 先写测试** (5 个测试文件, 115 个测试)
- `tests/conftest.py` - fixtures
- `tests/test_message_tokenizer.py` - 实例测试
- `tests/test_vocab.py` - 编码器测试
- `tests/test_encoding.py` - 编解码测试
- `tests/test_validation_helpers.py` - 验证测试

**Step 2: 代码修复**
1. Message_Tokenizer 添加 `__init__(token_mode=24)` (instance-level state)
2. Vocab 默认值 22→24
3. 移除 validation_helpers.py 全局 Vocab
4. run_inference.py 传递 `v=v`
5. REF_LEN 默认 24-token 计算

#### 关键代码修改

**Message_Tokenizer (encoding.py)**:
```python
# 改为默认 24-token
TOK_LENS = TOK_LENS_24  # Was: TOK_LENS_22

# 添加 instance-level state
def __init__(self, token_mode: int = 24) -> None:
    self.token_mode = token_mode
    self.tok_lens = self.TOK_LENS_24 if token_mode == 24 else self.TOK_LENS_22
    ...

# 标记旧 API 为 deprecated
@classmethod
def set_token_mode(cls, token_mode):
    warnings.warn("deprecated, use Message_Tokenizer(token_mode=N)", DeprecationWarning)
```

**Vocab (encoding.py)**:
```python
def __init__(self, token_mode=24) -> None:  # Was: token_mode=22
```

#### 关键洞察

1. **TDD 的价值**: 测试让我们发现了 Vocab 默认值遗漏的问题

2. **Deprecation Warning 策略**: 不直接删除旧 API，而是标记为 deprecated，允许平滑迁移

3. **Instance vs Class State**: 
   - Class state 适合常量 (如 `TOK_LENS_22`, `TOK_LENS_24`)
   - Instance state 适合配置 (如 `token_mode`, `msg_len`)

4. **默认值的重要性**: 默认值应与主要用例一致（24-token 用于 checkpoint）

#### 测试结果

```
============ 115 passed, 3 skipped, 21 warnings ============
```

3 个 skipped 测试是已知行为差异：
- NA token 解码返回 0（不是 -9999）
- START token 在部分位置允许

#### Commits

- `49c6722` refactor(encoding): add instance-level state to Message_Tokenizer
- `a1dd385` fix(encoding): change Vocab default token_mode from 22 to 24

---
