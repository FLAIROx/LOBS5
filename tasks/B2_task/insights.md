# B2 Scaletest Insights — 4N Performance Optimization (2026-02-20)

## 1. FSDP 的核心改动极其优雅
- 只需改 `create_state_shardings()` 一个函数——把 `P(None)` 改成 `P('data', None, ...)` for 2D+ params
- JAX/XLA 编译器会**自动推断**并插入 AllGather（forward 重组参数）和 ReduceScatter（backward 分片梯度）
- **不需要手动写任何通信代码**——这是声明式并行的威力
- FSDP 不减少通信总量（仍然 ~562MB），但把通信分散到每层，让 LHS 能和 compute 重叠
- 额外收益：每 GPU 显存从 900MB 降到 56MB (params+optim)，可以放大 batch size

## 2. Worktree 陷阱：WORKDIR 必须指向 worktree
- Batch script 里 `WORKDIR` 默认指向 git root，不是 worktree
- 从 worktree 提交 sbatch 时，计算节点的 `cd $WORKDIR` 会回到主目录
- 解决方案：`WORKDIR=<worktree_path> sbatch <worktree_path>/train.batch`
- 这也影响 C1 和 C3！但它们没改 Python 代码（只改 batch script），所以碰巧没出错

## 3. FSDP sharding mismatch 的根因链
1. `jax.device_get(state)` 把 jax.Array 转成 numpy.ndarray
2. `create_state_shardings` 用 `isinstance(leaf, jax.Array)` 检查 → numpy 不是 jax.Array → 全部 `P()`
3. `device_put(state, P())` 把参数以 replicated 放置
4. 但 `create_jit_train_step` 里同一个函数看到的是 jax.Array → 给 `P('data', None)`
5. jit 期望 `P('data', None)` 但实际是 `P()` → mismatch crash
6. Fix: `hasattr(leaf, 'ndim')` 兼容两种类型

## 4. FSDP 的可分性约束
- `PartitionSpec('data', None)` 要求第一个维度能被设备数整除
- 503 是质数，不能被 16 (4N × 4 GPU) 整除
- PyTorch FSDP 自动处理这种情况（padding），但 JAX PartitionSpec 要求严格整除
- 修复: 对不可整除的参数保持 replicated（不 shard）

## 5. 为什么加节点不等于扩展内存
- DDP = 每个 GPU 存**完整模型 + 完整 activation** → 加节点不省任何内存
- 75M 模型参数只占 1% GPU 内存，activation 占 99%
- FSDP 只 shard 参数 → 对 activation-bound 模型几乎无效
- 别人能扩展是因为他们用 Tensor Parallelism（把 activation 也切分到多 GPU）或 Pipeline Parallelism
- 对我们来说，梯度检查点 + BF16 在 2N 上就能解决 BSZ 问题

## 6. 2D Mesh 的本质区别
```
1D Mesh (原始):
  devices = [GPU0, GPU1, ..., GPU15]
  mesh = Mesh(16, 'data')
  AllReduce: 16-rank RING (91ms/op, 跨 4 节点)

2D Mesh (MaxText 方式):
  devices = [[GPU0-3],    # Node 0
             [GPU4-7],    # Node 1
             [GPU8-11],   # Node 2
             [GPU12-15]]  # Node 3
  mesh = Mesh((4,4), ('data','fsdp'))

  Forward:  AllGather on 'fsdp' (4 GPU, NVLink ~3ms)
  Backward: ReduceScatter on 'fsdp' (4 GPU, NVLink ~3ms)
            + AllReduce on 'data' (4 nodes, TREE ~1.4ms!)
```
- 关键: 跨节点 AllReduce 只有 4 个端点（每节点一个），而不是 16 个 GPU
- NCCL 在 4-rank 时选 TREE 算法，在 16-rank 时选 RING
- 这是 2N (1.43ms) vs 4N (91ms) 差距的根源

## 7. 2D Mesh 在 SSM 上失败的根因 (10.0 s/step, 比 1D 更慢)
- SSM 模型有很多小 parameter tensors（Lambda, B, C, D, 各种 kernel）
- 每个 tensor 在 2D mesh 上都需要独立的 AllGather/ReduceScatter
- NCCL 每个 op 的固定启动开销 (~0.1-0.3ms) × 300+ ops = 30-90ms 纯开销
- 1D mesh + combine 把这些全部合并成 2-3 个大包 → 固定开销仅 0.3-0.9ms
- MaxText 能用是因为 Transformer 每层只有几个大 tensor，op 数量少
- **结论**: 对于 small-param-many-tensor 的 SSM，1D + combine 是更优策略

## 8. 4N 性能墙的物理限制
- 4N→2N 的 12x 差距来自跨节点 NCCL 延迟，不是带宽瓶颈
- 每次 AllReduce 需要 ~91ms（跨 Slingshot 网络），但我们有 101 次/step
- XLA combine 合并到 2-3 次，LHS 让它和 compute overlap → 已是极限
- FSDP 理论上能进一步改善（per-layer overlap），但 XLA 编译开销巨大
- 真正的解决方案是减少通信次数——要么用更少的节点，要么用更粗粒度的并行策略

## 9. BlueConnect vs 2D Mesh FSDP — 为什么一个 flag 可能比整个架构改动更好
- **2D Mesh FSDP**: 改了参数 sharding → XLA 为每层插入 AllGather+ReduceScatter → 300+ NCCL ops → 10.0 s/step
- **BlueConnect**: 不改参数，让 XLA 编译器在 HLO 层面重写已有的 2-3 个合并 AllReduce → 每个变成 RS+AR+AG 三步
- 关键区别：combine threshold 先把 101 → 2-3 个大 op，然后 BlueConnect 把每个大 op 分解成分层的 3 步
- 所以总 NCCL op 数 = 2-3 × 3 = **6-9 个**（远少于 2D FSDP 的 300+）

## 10. P(None,None) 在 2D Mesh 上 ≡ P(None) 在 1D Mesh 上
- GSPMD 只看 PartitionSpec 语义，不看 mesh 拓扑
- `P(None, None)` = "在所有轴上都复制" = 和 `P(None)` 等价
- 所以把 mesh 改成 2D 但保持 replicated params → AllReduce 仍然是 flat 16-rank
- 要想分层 AllReduce，必须让 XLA 编译器主动分解（BlueConnect flag）

---

# FSDP 到底是什么 Parallel？

## 一句话答案

**FSDP = Data Parallel + Parameter Sharding（参数分片的数据并行）**

它本质是 **Data Parallel**，但在参数**存储**上借用了 Tensor Parallel 的思想。

## 三种并行的对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Parallel (DDP)                               │
│                                                                     │
│  GPU 0: [全部参数] + [数据切片 0] → 梯度 → AllReduce → 更新         │
│  GPU 1: [全部参数] + [数据切片 1] → 梯度 → AllReduce → 更新         │
│  GPU 2: [全部参数] + [数据切片 2] → 梯度 → AllReduce → 更新         │
│  GPU 3: [全部参数] + [数据切片 3] → 梯度 → AllReduce → 更新         │
│                                                                     │
│  特点: 每个 GPU 存完整模型副本, 只切分数据                            │
│  通信: AllReduce(梯度) — 一次, 在 backward 结束后                    │
│  内存: 每 GPU = 全部参数 + 全部优化器状态 + 激活值                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Tensor Parallel (TP)                              │
│                                                                     │
│  一个矩阵乘法 Y = X × W, W 形状 (4096, 4096):                      │
│                                                                     │
│  GPU 0: X × W[:, 0:1024]  → Y_0     ← 切分权重的列                  │
│  GPU 1: X × W[:, 1024:2048] → Y_1                                  │
│  GPU 2: X × W[:, 2048:3072] → Y_2                                  │
│  GPU 3: X × W[:, 3072:4096] → Y_3                                  │
│              → AllGather(Y_0..Y_3) → Y_full                         │
│                                                                     │
│  特点: 切分单个算子的权重矩阵, 每 GPU 算不同部分                     │
│  通信: AllGather / ReduceScatter — 每层, 在 forward 中               │
│  内存: 每 GPU = 1/N 参数 + 1/N 激活值 ← 激活值也被切了!              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    FSDP (Fully Sharded Data Parallel)                │
│                                                                     │
│  存储时 (参数分片, 类似 TP):                                         │
│  GPU 0: [参数切片 0] + [优化器切片 0] + [数据切片 0]                  │
│  GPU 1: [参数切片 1] + [优化器切片 1] + [数据切片 1]                  │
│  GPU 2: [参数切片 2] + [优化器切片 2] + [数据切片 2]                  │
│  GPU 3: [参数切片 3] + [优化器切片 3] + [数据切片 3]                  │
│                                                                     │
│  计算时 (重组参数, 回到 DP):                                         │
│  Forward 每层:                                                      │
│    AllGather(参数切片) → 临时拿到完整参数 → 计算 → 丢弃完整参数       │
│  Backward 每层:                                                     │
│    AllGather(参数切片) → 计算梯度 → ReduceScatter(梯度) → 只保留切片  │
│                                                                     │
│  特点: 存储像 TP (参数分片), 计算像 DP (每 GPU 算完整 forward)        │
│  通信: AllGather(forward) + ReduceScatter(backward) — 每层!          │
│  内存: 每 GPU = 1/N 参数 + 1/N 优化器 + 全部激活值                   │
│                             ↑ 省内存!       ↑ 激活值不省!            │
└─────────────────────────────────────────────────────────────────────┘
```

## FSDP vs TP 的关键区别

```
┌──────────────┬─────────────────────────────────┬──────────────────────────────┐
│              │           FSDP                  │          TP                  │
├──────────────┼─────────────────────────────────┼──────────────────────────────┤
│ 切什么       │ 参数存储（纯粹为了省内存）       │ 计算图（每 GPU 算不同部分）   │
├──────────────┼─────────────────────────────────┼──────────────────────────────┤
│ 计算方式     │ AllGather 后每 GPU 独立做        │ 每 GPU 做矩阵的一部分        │
│              │ 完整 forward (和 DDP 一样)       │ (真正的并行计算)             │
├──────────────┼─────────────────────────────────┼──────────────────────────────┤
│ 激活值       │ 不切分! 每 GPU 存完整激活值      │ 切分! 每 GPU 只存部分激活值   │
├──────────────┼─────────────────────────────────┼──────────────────────────────┤
│ 通信模式     │ AllGather + ReduceScatter        │ AllGather + ReduceScatter    │
│              │ (看起来一样!)                     │ (看起来一样!)               │
├──────────────┼─────────────────────────────────┼──────────────────────────────┤
│ 通信内容     │ 参数 (forward) + 梯度 (backward) │ 激活值 (forward + backward)  │
├──────────────┼─────────────────────────────────┼──────────────────────────────┤
│ 最佳场景     │ 参数/优化器占内存大头时           │ 激活值占内存大头时            │
│              │ (LLM: 175B × 12 bytes = 2.1 TB) │ (长序列, 大 batch)           │
├──────────────┼─────────────────────────────────┼──────────────────────────────┤
│ 我们的情况   │ 参数只占 1% → 几乎不省内存       │ 激活值占 99% → 这才有效!      │
└──────────────┴─────────────────────────────────┴──────────────────────────────┘
```

## FSDP 的名字解读

```
  F    S    D    P
  │    │    │    │
  │    │    │    └── Parallel     — 并行策略
  │    │    └─────── Data         — 本质是数据并行 (每 GPU 处理不同数据)
  │    └──────────── Sharded      — 参数被分片存储 (不是每 GPU 存完整副本)
  └───────────────── Fully        — 完全分片 (不只分片优化器, 连参数本身也分)

  演进路线 (DeepSpeed ZeRO 系列):
    DDP    = Data Parallel                    (参数复制, 数据切分)
    ZeRO-1 = Sharded Optimizer DP            (优化器分片, 参数复制)
    ZeRO-2 = Sharded Optimizer+Gradient DP   (优化器+梯度分片, 参数复制)
    ZeRO-3 = Fully Sharded DP = FSDP         (优化器+梯度+参数 全部分片)
```

## 为什么 FSDP 对 LOBS5 (75M) 没用

```
  LLaMA-70B:                           LOBS5 (75M):
  ┌─────────────┐                      ┌─────────────┐
  │ Params  70% │ ← FSDP 省这里!       │ Params   1% │ ← FSDP 省这里...
  │ Optim   20% │ ← FSDP 也省这里!     │ Optim    0% │    但只有 0.9 GB
  │ Activ   10% │                      │ Activ   99% │ ← 84 GB!
  └─────────────┘                      └─────────────┘
  FSDP 省 90% 内存 ✅                   FSDP 省 1% 内存 ❌
```

## 11. ★★★ 之前"2D Mesh 对 SSM 无效"的结论完全错误！(修正)

之前报告 2D mesh 10.0 s/step (比 1D 更慢) — 这是 **autotune 预热阶段** 的数据！

```
  XLA Autotune 预热期 vs 稳态速度:
  ┌──────────────────────────────────────────────────────┐
  │ Steps 1-57:  10.0 s/step  ← 之前错误地以为这是稳态   │
  │ Steps 58+:   0.70 s/step  ← 真正的稳态速度!          │
  └──────────────────────────────────────────────────────┘

  2D mesh 的 autotune 需要 ~57 步 (vs 1D 的 ~25 步)
  因为有更多通信模式需要优化:
  - AllGather on 'fsdp' (forward, reconstruct params)
  - ReduceScatter on 'fsdp' (backward, shard gradients)
  - AllReduce on 'data' (backward, cross-node sync)
```

修正后的性能表:
```
  ┌──────────────────┬──────────┬───────────────┬──────────────┐
  │      实验        │ 稳态     │  throughput    │ vs 2N 基线   │
  │                  │ s/step   │ (samples/s)   │              │
  ├──────────────────┼──────────┼───────────────┼──────────────┤
  │ 2N DDP (基线)    │ 0.55     │ 101.8         │ 1.0x         │
  │ 4N 1D + C3      │ 6.58     │ 17.0          │ 0.17x ❌      │
  │ 4N 2D Mesh+FSDP │ 0.70     │ 160.0         │ 1.57x ✅      │
  └──────────────────┴──────────┴───────────────┴──────────────┘
```

**教训**: 对 autotune 密集的分布式训练，必须等足够多步（>100 步）
才能判断稳态速度。25 步就下结论会得到完全错误的结果。

## 12. BlueConnect CRASH — AllGather rendezvous timeout
- `--xla_gpu_all_reduce_blueconnect_num_devices_per_host=4`
- XLA 把 AllReduce 分解成 AllGather + ReduceScatter
- 4-rank AllGather 超时 ("Expected 4 threads, only 1 arrived")
- **根因定位 (C4 隔离实验)**:
  - C4a: BlueConnect ONLY → 没 crash! JIT 15min 未完成 (TIMEOUT)
  - C4b: BlueConnect + 128MB combine → CRASH (AllGather 4→2)
  - C4c: BlueConnect + combine + LHS → CRASH (AllGather 4→2)
  - **结论: combine_threshold 是 crash 触发器，BC 本身和 JAX 多节点兼容**
  - combine 合并后的大 AllReduce 被 BC 分解时，子 op 间调度不一致
- C4a-retry (1h) 待结果 — 看 BC alone 的稳态速度

## 13. ★★★ 完整 Scaling Benchmark (1-32 GPU) — 节点内 vs 跨节点开销差异

```
  节点内扩展 (NVLink 478 GB/s):
    1→2 GPU: +0.020 s/step (+4.9%)   ← DDP AllReduce 几乎免费
    2→4 GPU: +0.025 s/step (+5.8%)   ← FSDP AllGather/ReduceScatter 也免费
    节点内效率: 90-100%

  跨节点扩展 (Slingshot 100 GB/s):
    4→8 GPU:  +0.110 s/step (+24.2%) ← 第一次跨节点通信
    8→16 GPU: +0.130 s/step (+23.0%) ← 每倍增加 ~24%
    16→32 GPU: +0.180 s/step (+25.9%)
    跨节点效率: 47-73%
```

**关键发现**: 每倍增 GPU 的通信开销比例恒定 (~24%)，说明 2D mesh 的
分层通信很有效——没有随节点数指数增长。32 GPU 仍能达到 15x 的吞吐量
提升 (vs 理想 32x)。

## 14. ★★★ 128 GPU (32 nodes) 扩展的四层问题链

```
  Layer 1: cuSolver handle contention
    ├── 症状: gpusolverDnCreate failed on ALL 32 nodes simultaneously
    ├── 原因: 128 CUDA contexts competing for cuSolver handles during init
    ├── Fix:  numpy.linalg.eigh() 替代 jax.numpy.linalg.eigh() in ssm_init.py
    └── 失败尝试: jax.default_device(cpu) → JAX_PLATFORMS=cuda 无 CPU backend

  Layer 2: XLA sharded autotuner DEVICE_TYPE_INVALID
    ├── 症状: "Autotuner could not find any supported configs" on node15
    ├── 原因: 32 节点 autotuner coordinator race condition
    ├── Fix:  --xla_gpu_shard_autotuning=false (各 host 独立 autotune)
    └── 阈值: 仅 32+ 节点! 16 节点 + shard_autotuning=false → JIT 卡死 30min+

  Layer 3: NCCL OOM
    ├── 症状: ncclGroupEnd failed: Cuda failure 2 'out of memory'
    ├── 原因: 32 nodes × 40 channels × 4MB buffer = NCCL 吃太多 GPU 内存
    ├── Fix:  NCCL_BUFFSIZE=2097152 (2MB) + PER_GPU_BSZ=4→7
    └── BSZ=7 最终也成功 (只需 NCCL_BUFFSIZE 降低)

  Layer 4: Scaling wall (物理极限)
    ├── 32 GPU (8N):  0.875 s/step, 256 samp/s, 47% eff ← PEAK
    ├── 64 GPU (16N): 2.130 s/step, 210 samp/s, 19% eff ← 比 32 GPU 还慢!
    └── 128 GPU (32N): 1.745 s/step, 514 samp/s, 23% eff ← 蛮力恢复
```

**教训**: 大规模分布式训练的每一层 infra 都可能有独立的 failure mode。
一层修复后暴露下一层——典型的 "peeling the onion" 调试模式。

## 15. ★ 64 GPU 异常 — sharded autotuning 可能生成劣质 kernel

64 GPU (16N) 使用默认 sharded autotuning, 得到 2.13 s/step。
128 GPU (32N) 使用 shard_autotuning=false (独立 autotune), 得到 1.745 s/step。
64 GPU 的通信开销理论上应该 < 128 GPU，但实际更慢。
可能原因: 16 节点 sharded autotuning 选择了质量较差的 CUDA kernel。
验证方法: 重跑 64 GPU + shard_autotuning=false (但 JIT 太慢, 已放弃)。
