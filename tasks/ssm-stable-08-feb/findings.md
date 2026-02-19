# ssm-stable-08-feb 分支 DDP 分析 Findings

> 日期: 2026-02-18
> 分析目标: `ssm-stable-08-feb` 的 DDP (全局 Mesh) 实现方式，对比 `exp/B1-ignore-times-v2`

---

## 核心发现：ssm-stable 实际上 **没有** 实现跨节点梯度同步

尽管 `sharding_utils.py` 注释写着 *"Gradient sync across nodes is handled by jax.lax.psum in train_step"*，
但实际 `train_step` 里 **没有 psum/pmean 调用**。`run_train.py` 中有坦白的 TODO：

```python
# NOTE: Currently each node trains independently with local gradient sync only
# True distributed training requires cross-node gradient aggregation (psum across processes)
# TODO: Implement proper multi-node gradient sync for effective global batch = micro_bsz × total_gpus
```

ssm-stable 多节点运行时 = N 个独立训练，不是 DDP。

---

## 架构对比

```
ssm-stable-08-feb (LOCAL mesh)           exp/B1-ignore-times-v2 (GLOBAL mesh)
================================         ====================================

   Node 0          Node 1                      Node 0          Node 1
  [GPU0-3]        [GPU4-7]                   [GPU0-3]        [GPU4-7]
     │               │                          │               │
  LOCAL mesh      LOCAL mesh              <═══ GLOBAL mesh ═══>
  (4 devices)     (4 devices)               (8+ devices total)
     │               │                          │               │
  独立 train       独立 train               JAX 自动 allreduce
  ❌ 无跨节点通信   ❌ 无跨节点通信              ✅ 梯度全同步
     │               │                          │               │
  各自 ckpt        各自 ckpt               Orbax barrier 同步
  (rank 0 only)   (不参与)                 (ALL ranks 参与)
```

---

## 5 维度详细对比

| 维度 | ssm-stable-08-feb | exp/B1-ignore-times-v2 |
|------|-------------------|------------------------|
| **Mesh 创建** | `jax.local_devices()` → 仅本地 4 GPU | `jax.devices()` → 跨节点全部 GPU |
| **梯度聚合** | 仅 local mesh 内 4 GPU 聚合 | GLOBAL mesh 自动 allreduce (不需要显式 pmean) |
| **有效 BSZ** | `micro_bsz × 4` (每节点独立) | `micro_bsz × 4 × N_nodes` (真 DDP) |
| **数据分发** | 普通 numpy → local device_put | `make_array_from_process_local_data()` → global sharding |
| **Checkpoint** | rank 0 only 创建 + 保存 | ALL ranks 创建 Manager (Orbax barrier) + re-shard + 禁用 async |

---

## NCCL 配置差异

| 配置项 | ssm-stable | B1 |
|--------|------------|-----|
| `NCCL_P2P_DISABLE` | 硬编码 `=0` (覆盖 batch 脚本) | `setdefault` (batch 脚本 `=1` 生效，GH200 无 NVLink) |
| `NCCL_TIMEOUT` | 硬编码 600s | batch 脚本 3600s |
| `NCCL_DEBUG` | 未设 | `INFO` |
| XLA FLAGS | 无 | `triton_gemm=false`, `autotune_level=0` (防多节点死锁) |
| aws-ofi-nccl | 未加载 | 已加载 (Slingshot 网络) |

---

## B1 解决的 5 个多节点 Bug（ssm-stable 中都不存在/未触发）

1. **NCCL P2P override** — `os.environ[...] = "0"` 硬编码覆盖 batch 脚本 → B1 改 `setdefault`
2. **Checkpoint 序列化** — multi-host 需 re-shard state 为 NamedSharding → B1 增加 `jax.jit(identity, out_shardings=...)`
3. **Async checkpoint fork** — fork 子进程破坏 NCCL → B1 禁用 `enable_async_checkpointing`
4. **Epoch 2 hang** — JAX async dispatch + Python LR mutation 竞态 → B1 加 `block_until_ready()`
5. **jax.clear_caches()** — 导致 JIT 重编译 + NCCL clique 泄漏 OOM → B1 只用 `gc.collect()`

---

## 其他差异

| 方面 | ssm-stable | B1 |
|------|------------|-----|
| LR 管理 | optax schedules only | 双路径: optax schedules + inject_hyperparams + mesh-aware LR array |
| Epoch 间清理 | `jax.clear_caches()` + `gc.collect()` | 仅 `gc.collect()` (clear_caches 导致 NCCL clique 泄漏) |
| DataLoader 刷新 | del + 重建 trainloader | 原地 `reset_train_offsets()` |
| WandB multi-rank | 全部 disabled | rank 0 online, 其余 offline |
| TBPTT 支持 | 有 (gradient_chunking + ar_hidden) | 无 |
| Model presets | 有 (55M, 360M, 1B, etc.) | 无 (手动参数) |

---

## ★ Insights

### ★ Insight: 为什么 GLOBAL mesh 不需要显式 pmean？
JAX 的核心设计：当 state 标记为 `P(None)` (replicated) 而 data 标记为 `P('data')` (sharded) 时，
`jax.value_and_grad` 计算出的梯度在 replicated state 上自动触发 allreduce。
XLA 编译器在 HLO 层面插入 `all-reduce-sum` 并除以 shard 数。
这就是 SPMD 编程模型的威力——只需声明数据分布方式，编译器自动生成通信代码。

### ★ Insight: ssm-stable 的 LOCAL mesh 为什么"看起来能跑"？
因为 `jax.distributed.initialize()` 确实建立了跨节点连接，SLURM 也正确分配了 data shard。
每个节点独立训练自己那份数据，loss 会下降，wandb 看起来正常——
只是各节点的模型在第一步之后就开始 diverge，等于跑了 N 个独立实验。

### ★ Insight: ssm-stable 有价值的独有功能
虽然 DDP 未完成，但 ssm-stable 有几个 B1 没有的功能：
- **TBPTT** (Truncated Backpropagation Through Time) — gradient chunking + ar_hidden 两种模式
- **SWR** (Sliding Window Recurrences) — 提高 Arithmetic Intensity
- **Model presets** — 55M / 360M / 1B / 1.4B / 2B / 2.5B
- **Prodigy optimizer** — 自适应 LR 估计
这些功能未来可以移植到 B1 分支。

---

## BF16 混合精度训练对比 (2026-02-19)

**ssm-stable 实现了完整的 BF16 混合精度；B1 完全没有 BF16，全 FP32。**

### 各组件 dtype 策略

| 组件 | ssm-stable | B1 | 设计理由 |
|------|-----------|-----|---------|
| Embedding | **BF16** | FP32 | Tensor Core 加速 |
| Dense layers | **BF16** | FP32 | Tensor Core 加速 |
| LayerNorm | **BF16** | FP32 | 跟随 activation dtype |
| SSM `B@u` matmul | **BF16 batch matmul** | FP32 complex64 vmap | Tensor Core |
| SSM scan | **FP32** (complex64) | FP32 (complex64) | 递归数值稳定性 |
| SSM `C@xs` matmul | **BF16 batch matmul** | FP32 complex64 vmap | Tensor Core |
| D feedthrough | FP32 | FP32 | |
| Decoder (输出层) | **故意 FP32** | FP32 | 数值稳定性 |
| 参数存储 | FP32 | FP32 | 梯度更新精度 |
| 优化器 moments | FP32 | FP32 | Adam 累积精度 |

### BF16 sandwich 数据流 (ssm-stable)

```
Input ──→ Embed(BF16) ──→ Dense(BF16) ──→ LayerNorm(BF16)
                                              │
                              ┌────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ B @ u  (BF16)   │ ← Tensor Core 加速
                    └────────┬────────┘
                             │ cast → FP32
                    ┌────────▼────────┐
                    │ assoc_scan      │ ← complex64 (=FP32) 保数值稳定
                    └────────┬────────┘
                             │ cast → BF16
                    ┌────────▼────────┐
                    │ C @ xs (BF16)   │ ← Tensor Core 加速
                    └────────┬────────┘
                             │ cast → FP32
                    ┌────────▼────────┐
                    │ + Du (FP32)     │
                    └────────┬────────┘
                             ▼
                    Decoder(FP32) ──→ Loss(FP32)
```

### BF16 启用方式

```python
# run_train.py — 默认开启
parser.add_argument("--use_bf16", type=str2bool, default=True)
os.environ['USE_BF16'] = '1' if args.use_bf16 else '0'

# 模型层通过环境变量读取
use_bf16 = os.environ.get('USE_BF16', '1') == '1'
compute_dtype = jnp.bfloat16 if use_bf16 else jnp.float32
```

### SSM apply_ssm 关键改动 (vmap → batch matmul)

ssm-stable 把逐步 vmap 改写为批量矩阵乘：
```python
# B1: vmap 逐步 (complex64, 慢)
Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

# ssm-stable: batch matmul (BF16, Tensor Core)
input_T = input_fp32.T.astype(np.bfloat16)     # (H, L)
B_re = B_bar.real.astype(np.bfloat16)           # (P, H)
Bu_re = np.matmul(B_re, input_T)                # (P, L) ← single Tensor Core op!
Bu_im = np.matmul(B_im, input_T)                # (P, L)
Bu_elements = (Bu_re.astype(np.float32) + 1j * Bu_im.astype(np.float32)).T
```

### 全 BF16 scan 实验（已弃用）

ssm-stable 曾尝试在 BF16 域执行 associative_scan，但数值爆炸：
- BF16 尾数 7 bits vs FP32 23 bits
- 序列长 L=12000 → 递归深度 log₂(12000)≈14 层
- 实测 xs 值域飙到 `[-3e8, 3e8]`（预期 O(1-100)）→ NaN
- 结论：scan 必须留 FP32

### 移植到 B1 需改的文件

| 文件 | 改动内容 |
|------|---------|
| `run_train.py` | 加 `--use_bf16` 参数 + 设 `USE_BF16` 环境变量 |
| `s5/ssm.py` | **最大改动** — vmap → BF16 batch matmul |
| `s5/layers.py` | 加 `dtype` 字段，传给 Dense/LayerNorm |
| `s5/seq_model.py` | 加 `dtype` + `USE_BF16` 环境变量读取 |
| `lob/lob_seq_model.py` | 模型类加 `dtype` 传播 |

### ★ Insight: SSM 的 BF16 sandwich 策略
S5 的 associative scan 是递归操作 `x_{t+1} = A·x_t + B·u_t`，序列长 L=12000 时递归深度
`log₂(12000)≈14` 层。BF16 尾数只有 7 bits（vs FP32 23 bits），累积误差在 BF16 下会爆炸。
解决方案是 "BF16 sandwich"：scan 两侧的矩阵乘（`B@u` 和 `C@xs`）用 BF16 利用 Tensor Core，
scan 本身留 FP32。既获得大部分 BF16 加速，又保住递归数值稳定性。

### ★ Insight: vmap → batch matmul 是独立于 BF16 的性能优化
ssm-stable 把 `vmap(lambda u: B @ u)` 改写成 `np.matmul(B_re, input_T)`，
即使不考虑 BF16，这本身也消除了 vmap 开销，让 XLA 生成单个大矩阵乘而非逐步循环。
这个优化可以独立移植到 B1。

---

## 参考实现分析：HyperscaleES + MaxText (2026-02-19)

### HyperscaleES 多节点模式

**核心文件**: `/projects/s5e/quant/AlphaTrade/HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py`

HyperscaleES 是进化策略 (ES)，不做 backprop，因此梯度同步模式与传统 DDP 完全不同。

**Mesh 创建**:
```python
mesh = jax.make_mesh((len(jax.devices()),), ('data',))  # 1D 全局 mesh
```

**参数同步策略**: ES 的巧妙设计 — 不需要 gradient allreduce
```
1. 每 GPU 独立生成样本 (shard_map, P('data'))
2. process_allgather(fitness) — 唯一跨节点通信
3. 每 GPU 用相同 fitness + 确定性噪声 → 独立算出相同的参数更新
```

**编排 API**: `shard_map`（手动指定 in_specs/out_specs），而非 `jax.jit` + in/out_shardings

**SLURM 配置**: 每 GPU 一个进程（`srun --ntasks-per-node=4`），与 LOBS5 的"每节点一个进程看 4 GPU"不同

**对 LOBS5 的借鉴**: 有限。ES 的 process_allgather 模式适合无梯度场景，但我们是标准 backprop，需要 gradient allreduce。`mu.sync_global_devices()` 用于 wandb init/finish 的 barrier 模式可以参考。

---

### MaxText 多节点模式（核心参考）

**MaxText 是 Google 的 JAX 大规模训练参考实现，与 LOBS5 B1 的架构最为接近。**

#### ICI/DCN Hybrid Mesh（最重要的设计）

```python
# maxtext_utils.py:1027-1097
mesh = Mesh(
    create_hybrid_device_mesh(
        ici_parallelism,    # 节点内: [1, 4]  → 4-way FSDP
        dcn_parallelism,    # 跨节点: [N, 1]  → N-way DP
        devices=jax.devices()
    ),
    axis_names=('data', 'fsdp')
)
```

```
┌──────────────────────────────────────────────────────────┐
│  MaxText 2D Mesh: DCN(data) × ICI(fsdp)                 │
│                                                          │
│  DCN axis (跨节点, Slingshot, 延迟高)                     │
│  ↓                                                       │
│  Node 0: ─── ICI axis (节点内, NVLink, 带宽高) ───→     │
│           [GPU0] [GPU1] [GPU2] [GPU3]                    │
│              ↑ FSDP: 参数分片到 4 GPU                     │
│                                                          │
│  Node 1: [GPU4] [GPU5] [GPU6] [GPU7]                    │
│              ↑ FSDP: 相同分片方式                         │
│                                                          │
│  DP: 节点间只做梯度 allreduce (通信量小)                   │
│  FSDP: 节点内参数分片 (利用高带宽互联)                     │
└──────────────────────────────────────────────────────────┘
```

#### SPMD 自动梯度同步

MaxText 的 train_step 与 B1 一样，**不需要显式 psum/pmean**：

```python
# train_utils.py:83-111
p_train_step = jax.jit(
    train_step,
    in_shardings=(state_mesh_shardings, data_sharding, None),
    out_shardings=(state_mesh_shardings, None),
    donate_argnums=0,
)
```

XLA 编译器自动插入：
- forward: all-gather params over fsdp axis
- backward: reduce-scatter grads over fsdp axis
- all-reduce grads over data axis

#### 逻辑轴 → 物理轴映射

MaxText 的抽象层：模型代码只声明逻辑轴名，映射规则在配置文件中定义。

```
逻辑轴 (模型层面)           物理轴 (mesh 层面)
─────────────────          ──────────────────
activation_batch    →      ['data', 'fsdp']
embed               →      ['fsdp']
mlp                 →      ['tensor']
heads               →      ['tensor']
```

#### 多主机数据加载

```python
# multihost_dataloading.py
def _form_global_array(path, array, global_mesh):
    global_shape = (process_count * local_batch, ...)
    sharding = NamedSharding(mesh, P(mesh.axis_names))
    local_buffers = jax.device_put(np.split(array, n_local_devices), local_devices)
    return jax.make_array_from_single_device_arrays(global_shape, sharding, local_buffers)
```

#### Checkpoint 策略

- Orbax async checkpoint（默认启用，与 B1 的禁用不同）
- Single replica restoring：只从一个 replica 读 ckpt，广播到其他
- 数据迭代器状态也做 checkpoint（支持 scale-up/down）

---

### 三方对比总表

| 特性 | HyperscaleES | MaxText | LOBS5 B1 |
|------|-------------|---------|----------|
| **Mesh 维度** | 1D `('data',)` | 12D (data,fsdp,tensor,...) | 1D `('data',)` |
| **ICI/DCN 分离** | 无 | **有** | 无 |
| **FSDP** | 无 | **有** (默认节点内) | 无 |
| **梯度同步** | 不需要 (ES) | SPMD 自动 | SPMD 自动 |
| **编排 API** | `shard_map` | `jax.jit` + shardings | `jax.jit` + shardings |
| **数据构建** | `make_array_from_single_device_arrays` | 同左 | `make_array_from_process_local_data` |
| **Checkpoint** | 无 | Orbax async | Orbax sync (禁用 async) |
| **显式通信** | `process_allgather` | 无 | 无 |

---

### MaxText 核心文件引用路径

| 参考模式 | 文件路径 |
|---------|---------|
| Mesh 创建 | `/projects/s5e/quant/AlphaTrade/maxtext/src/MaxText/maxtext_utils.py` (L1027-1097) |
| Sharding 工具 | `/projects/s5e/quant/AlphaTrade/maxtext/src/MaxText/sharding.py` |
| 逻辑轴规则 | `/projects/s5e/quant/AlphaTrade/maxtext/src/MaxText/configs/base.yml` (L388-465) |
| JIT 签名 | `/projects/s5e/quant/AlphaTrade/maxtext/src/MaxText/train_utils.py` (L83-111) |
| train_step | `/projects/s5e/quant/AlphaTrade/maxtext/src/MaxText/train.py` (L215-330) |
| 训练循环 | `/projects/s5e/quant/AlphaTrade/maxtext/src/MaxText/train.py` (L369-497) |
| 多主机数据加载 | `/projects/s5e/quant/AlphaTrade/maxtext/src/MaxText/multihost_dataloading.py` |
| Checkpoint | `/projects/s5e/quant/AlphaTrade/maxtext/src/MaxText/checkpointing.py` |
| 教学示例 | `/projects/s5e/quant/AlphaTrade/maxtext/pedagogical_examples/shardings.py` |

---

### ★ Insight: MaxText 的 ICI/DCN 分离是多节点训练的最佳实践

在 GH200 集群上：
- **ICI (节点内)**: 4 GPU 通过高带宽互联 → 适合 FSDP (参数分片，通信量大但延迟低)
- **DCN (跨节点)**: Slingshot 网络，延迟高 → 只做 Data Parallel (梯度 allreduce，通信量相对小)

当前 B1 用 1D flat mesh 纯 DP，allreduce 通信量 = 全模型参数量。
改用 2D mesh (dcn_data=N, ici_fsdp=4) 可以减少跨节点通信。
但对于 75M 模型，纯 DP 已足够；FSDP 在 >1B 参数时才有显著收益。

### ★ Insight: B1 的做法已与 MaxText 核心架构一致

B1 的 `jax.jit + in/out_shardings` + `make_array_from_process_local_data` 模式
与 MaxText 的 SPMD 路径本质相同。两者都：
- 用全局 mesh 声明设备拓扑
- 用 NamedSharding + PartitionSpec 声明数据/参数分布
- 依赖 XLA 编译器自动插入 collective 通信
B1 的 5 个 multi-host bug fix 也都是这条路径上的实战经验。

### ★ Insight: HyperscaleES 的 shard_map 不适合 backprop 训练

`shard_map` 是更底层的 API，要求手动指定每个函数的 in_specs/out_specs。
ES 用它是因为需要精细控制 per-shard 的生成逻辑 + 确定性噪声重建。
对于标准 backprop 训练，`jax.jit + in/out_shardings` 更简洁且编译器能做更多优化。

### ★ Insight: FSDP 的本质是 "用通信换显存"

FSDP = Fully Sharded Data Parallel = ZeRO Stage 3。
DDP 每个 GPU 存完整模型（显存大，通信少），FSDP 每个 GPU 只存 1/N（显存降 N 倍，但 forward/backward 各需一次 all-gather）。
"Fully" = 参数+梯度+优化器状态三样全切，是最激进也最省显存的方案。
在 JAX 中不需要显式 FSDP wrapper，只需 mesh 中定义 fsdp 轴 + P('fsdp') 标注参数，XLA 编译器自动推断通信。

LOBS5 当前 75M 模型不需要 FSDP；1B+ 时需要；2.5B+ 时必须。
升级路径：mesh 从 1D ('data',) 改为 2D ('data','fsdp')，参考 MaxText 的 ici_fsdp=4。

---

---

## 状态评估 (2026-02-19)

**B1 (`exp/B1-ignore-times-v2`) DDP 架构已就位，但生产可用性尚未验证。**

### 已完成
- 架构与 MaxText (Google 参考实现) 一致 (jax.jit + in/out_shardings + NamedSharding)
- 12 个 multi-host bug 全部修复
- 3 epoch debug 版验证通过 (job 2368882, commit 9508198, block_until_ready 每步)

### 待验证 (PENDING jobs)
- ❌ **optax schedule 根因修复从未运行** (job 2370026, commit 4722ecf, PENDING)
- ❌ 20+ epoch 长训练稳定性未验证 (全局 mesh 下最多 3 epoch)
- ❌ BSZ=7 全局 mesh 未验证 (之前 BSZ sweep 是 local mesh)
- ❌ >2 节点 scale 未测试
- ❌ Test Loss 不上涨问题未确认 (最初触发调查的 bug)

**在 job 2370026 跑完并确认 20+ epoch 稳定之前，不能宣称 DDP 目标达成。**

### 下一步价值方向（不再是 DDP 本身）

| 优先级 | 方向 | 来源 | 收益 |
|--------|------|------|------|
| 高 | 长训练稳定性验证 (20+ epoch, BSZ=7) | B1 自身 | 确认生产可用 |
| 高 | 多节点 scalability (4/8/16 节点) | B1 自身 | 验证 scale |
| 中 | BF16 sandwich 移植 | ssm-stable | 显存减半 + Tensor Core |
| 中 | vmap→batch matmul 移植 | ssm-stable | SSM 计算加速 |
| 低 | 2D hybrid mesh (data+fsdp) | MaxText | >1B 参数时需要 |
| 低 | TBPTT / SWR 移植 | ssm-stable | 长序列训练 |

---

## 交叉引用

- B1 DDP 完整实现历程 (20 commits, 12 bugs): `tasks/B1_task/B1.18.feb/findings.md` §20-22
- B1 DDP 进度日志: `tasks/B1_task/B1.18.feb/progress.md`
