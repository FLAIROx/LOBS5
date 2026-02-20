# Findings: 4N Scale Test (20 Feb 2026)

## ★ Insight: JAX 0.9.0.1 修复 autotuner crash
- XLA#36579: 修复 autotuner sharding 排序
- XLA#36755: 修复分布式迭代顺序确定性
- 升级后 2N+4N 都用 XLA 默认 flags 无 crash

## ★ Insight: NCCL_P2P_DISABLE=1 导致 13% 退化但非主因
- GH200 有 NV6 (478 GB/s NVLink) — P2P_DISABLE 是错误设置
- 但 P2P 只改善 11.3→9.8 (13%)，主要瓶颈在别处

## ★ Insight: NCCL 通信层完全可排除
- 2N 和 4N Phase 2 training comm 结构完全对称
- 都用 AWS Libfabric + CXI + GDRDMA (不是 TCP Socket!)
- 都有 8 coll channels, 8 p2p channels
- batch script L130 已加载 aws-ofi-nccl-1.8.1

## ★ Insight: 外部分析者的 #2 主张完全错误
- 他声称"缺少 NCCL_NET=AWS Libfabric → NCCL 走 TCP Socket"
- 实际: 脚本已通过 LD_LIBRARY_PATH 加载 aws-ofi-nccl 插件
- NCCL 日志明确显示: "Loaded net plugin AWS Libfabric (v7)"

## ★ Insight: 2N vs 4N 的通信模式差异
Phase 1 (bootstrap): 2N=4×nranks=2, 4N=4×nranks=4 (各 40 channels)
Phase 2 (training):  2N=1×nranks=8,  4N=1×nranks=16  (各 8 channels)
→ Channel 数完全相同，传输协议完全相同

## ★ Insight: TF_GPU_ALLOCATOR + gpu-bind 无效
- Job 2381259: 10.51 s/step (vs v7 的 9.78)
- 说明瓶颈不在 GPU 内存分配器或 NUMA affinity
- 已排除的因素增加到 8 个

## 已排除因素完整列表
| # | 排查项 | 证据 |
|---|--------|------|
| 1 | NCCL 走 TCP 而非 RDMA | AWS Libfabric + CXI confirmed |
| 2 | NCCL_P2P_DISABLE=1 | 只改善 13% (非主因) |
| 3 | NCCL channel 数差异 | 2N=4N=8 coll channels |
| 4 | NCCL_TIMEOUT 太短 | 3600s, 无 timeout 报错 |
| 5 | JAX_PROCESS_COUNT 未设置 | run_train.py 显式传参 |
| 6 | NCCL 拓扑错误 | PXN=1, NVLink+NET 正确 |
| 7 | TF_GPU_ALLOCATOR 缺失 | 无改善 (10.51 vs 9.78) |
| 8 | gpu-bind=none | 无改善 (10.51 vs 9.78) |

## ★★★ 根因确认: 100% 在 GPU sync (XLA 执行计划差异)
**Job 2381959 (2N) vs 2381960 (4N) timing breakdown (step 25-34 avg):**
```
               2N (8 GPU)    4N (16 GPU)   比率
prep           0.011s        0.010s        0.9x  ← 相同
dispatch       0.021s        0.020s        1.0x  ← 相同
sync           0.529s        9.503s       18.0x  ← ★★★ 根因!
total          0.560s        9.533s       17.0x
```
- prep/dispatch 完全排除了 host 端数据管线和 resharding barrier
- 100% 差异在 `loss.block_until_ready()` — GPU compute+comm 完成等待
- XLA 为 16-device mesh 编译了极差的 HLO 执行计划
- Profiler trace: `profiler_2381959/` (2N) + `profiler_2381960/` (4N)
- _prep_batch_par implicit reshard 假设已排除 (prep 只有 10ms)

## ★★★★ Profiler 根因: NCCL RING vs TREE 算法选择
**2N → TREE 为主 (3400 calls, 1.43ms/call), 4N → 全部 RING (4040 calls, 45.65ms/call)**

```
               2N (8 GPU)         4N (16 GPU)        比率
TREE AR        3400×1.43ms        0 calls !!!        N/A
RING AR        640×0.53ms         4040×45.65ms       86x 慢
AR Total       5,199ms            184,425ms          35x
GEMM Total     5,828ms            5,509ms            0.94x (完全一样)
```

- GEMM compute 完全一样，XLA 编译产物的 compute kernel 没问题
- 100% 差异在 NCCL AllReduce 算法选择: 4N 弃用 TREE 切到 RING
- RING 在 16-rank 跨 4 节点时延迟极高 (15 hop, 12 个跨节点)
- 修复尝试 1: `NCCL_ALGO=TREE` → crash! AllGather 不支持 TREE (job 2382073)
- 修复尝试 2: `NCCL_ALGO="allreduce:tree"` → 21.83 s/step! TREE 比 RING 慢 2.2x (job 2382095)
- 修复尝试 3: CSCS Alps CXI configs → 13.36 s/step! 比 baseline 慢 37% (job 2382150)
- **修复尝试 4: XLA combine 128MB + LHS → 6.9 s/step (29% 改善) (job 2382249)**
  - `--xla_gpu_all_reduce_combine_threshold_bytes=134217728` (128MB)
  - `--xla_gpu_enable_latency_hiding_scheduler=true`
  - `--xla_gpu_enable_highest_priority_async_stream=true`
  - 预期 ~1.0 s/step 但实际 6.9 — combine 有效但层间数据依赖限制了合并
- **修复尝试 5 (C3): 512MB+pipelined+double_buf → ~6.58 s/step (4.6% vs v13) (job 2382308)**
  - `--xla_gpu_all_reduce_combine_threshold_bytes=536870912` (512MB)
  - `--xla_gpu_enable_pipelined_all_reduce=true`
  - `--xla_gpu_enable_while_loop_double_buffering=true`
  - 小幅改善，但根本问题 (per-op NCCL 延迟) 不变
- **修复尝试 6 (C1): BF16 sandwich → 6.74 s/step (2.3% vs v13) (job 2382318)**
  - BF16 matmul (Dense, LayerNorm), FP32 SSM scan + decoder
  - 精确 benchmark: step 150=1221s, step 200=1558s → (1558-1221)/50 = 6.74 s/step
  - 梯度仍是 FP32，通信量 562MB 不变 → 改善仅来自 BF16 GEMM 加速
  - 代码: s5/layers.py, s5/seq_model.py, lob/lob_seq_model.py + USE_BF16=1
  - Commit: 06cb8ad (exp/C1-mixed-precision)
- **修复尝试 7 (C2): FSDP param sharding → JIT warmup 超时 (4 次提交)**
  - P(None) → P('data', None, ...) for 2D+ params
  - 消除 AllReduce, 改为 per-layer AllGather+ReduceScatter
  - 4 次失败: WORKDIR指向主目录(2382336), 503%16!=0(2382340), numpy/jax类型(2382352), JIT autotune>30min(2382384)
  - FSDP 创建远多于 DDP 的 XLA ops → autotune 无法在 30min 内完成

## 最终结论 (2026-02-20)

| 实验 | Job | s/step | vs 原始 4N | vs 2N |
|------|-----|--------|-----------|-------|
| 2N 参考 | — | 0.55 | — | 1.0x |
| 4N 原始 | 2381960 | 9.78 | 1.0x | 17.8x |
| v13 (128MB+LHS) | 2382249 | 6.9 | 0.71x | 12.5x |
| C1 (BF16) | 2382318 | 6.74 | 0.69x | 12.3x |
| C3 (512MB+pipelined+dblbuf) | 2382308 | 6.58 | 0.67x | 12.0x |
| C2 (FSDP) | 2382384 | N/A | JIT超时 | — |

~~**根本瓶颈**: 4N AllReduce 跨节点延迟 ~6.12s/step (占 93%)，软件优化已达极限。~~
~~**投产建议**: 生产训练用 2N (8 GPU)，4N 成本 24x 不合理。~~
**↑ 上述结论已被推翻！见下方 2D Mesh 修正结果。**

## ★★★★★★ 2D Mesh 突破 (job 2413027) — 0.70 s/step!! (修正之前的 10.0 s/step 错误)

**之前报告的 10.0 s/step 是 autotune 预热阶段（steps 1-57）的数据，不是稳态速度！**

2D mesh ('data'=4nodes, 'fsdp'=4GPUs/node) 分离节点内/跨节点通信:
- NCCL 确实选了 TREE ("Connected all trees")
- Autotune 预热: ~57 步 × 10 s/step（比 1D 的 ~25 步更长，因更多通信模式需优化）
- **稳态速度: 0.70 s/step** (step 150=339s, step 200=374s, (374-339)/50=0.70)
- 比 1D+C3 (6.58 s/step) 快 **9.4x**
- 吞吐量: 160 samples/s vs 2N 的 101.8 → **1.57x 近线性扩展!**
- Commit: 2d15170 (exp/C2-fsdp)
- W&B: https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/kh0i6ws3
- Bug: ZeroDivisionError in test loop (`pct = step / len_steps` where len_steps=0)

### 修正后的实验总表

| 实验 | Job | s/step | throughput | vs 2N |
|------|-----|--------|------------|-------|
| 2N DDP (基线) | — | 0.55 | 101.8 samp/s | 1.0x |
| 4N 1D (无优化) | 2381960 | 9.78 | 17.0 | 0.17x |
| 4N 1D + C3 (XLA flags) | 2382308 | 6.58 | 17.0 | 0.17x |
| **4N 2D Mesh + FSDP** | **2413027** | **0.70** | **160.0** | **1.57x** |
| 4N BlueConnect | 2418571 | CRASH | — | — |

**投产建议 (修正)**: 4N 2D mesh 达到 1.57x 近线性扩展，可以投产使用!
需修复: ZeroDivisionError in test loop, 并跑 40 epoch 完整验证。

## ★★★★★★★ 完整 Scaling Benchmark (1-128 GPU) — 2026-02-20

**方法**: CURTAIL_EPOCHS=400, PER_GPU_BSZ=7, 取 step 200-400 差值
- 2D mesh: ('data'=nodes, 'fsdp'=4 GPUs/node)
- 1D DDP: ('data',) 标准 DDP
- 128 GPU 需要: numpy eigh (cuSolver fix) + shard_autotuning=false + NCCL_BUFFSIZE=2MB

### 主表: 2D Mesh (FSDP) — 推荐方案

| GPU | Nodes | s/step | ms/step | samp/s | eff% | vs2N | comm_ms | comm% | 40ep_h | GPU·h | Job     | W&B      |
|-----|-------|--------|---------|--------|------|------|---------|-------|--------|-------|---------|----------|
| 1   | 1     | 0.410  | 410     | 17.1   | 100% | 0.17x| 0       | 0%    | 224.1  | 224   | 2419755 | b5zknywr |
| 2   | 1     | 0.430  | 430     | 32.6   | 95%  | 0.32x| 20      | 5%    | 117.5  | 235   | 2419756 | cjtgzfjm |
| 4   | 1     | 0.455  | 455     | 61.5   | 90%  | 0.60x| 45      | 10%   | 62.2   | 249   | 2419686 | u6qu8m6j |
| 8   | 2     | 0.565  | 565     | 99.1   | 73%  | 0.97x| 155     | 27%   | 38.6   | 309   | 2419687 | 4j2oehlk |
| 16  | 4     | 0.695  | 695     | 161.2  | 59%  | 1.58x| 285     | 41%   | 23.7   | 380   | 2419688 | owexzg82 |
| 32  | 8     | 0.875  | 875     | 256.0  | 47%  | 2.51x| 465     | 53%   | 14.9   | 478   | 2419690 | 5uljdw7r |
| 64  | 16    | 2.130  | 2130    | 210.3  | 19%  | 2.07x| 1720    | 81%   | 14.5   | 930   | 2421773 | i8ni4pn3 |
| 128 | 32    | 1.745  | 1745    | 513.5  | 23%  | 5.04x| 1335    | 77%   | 5.9    | 752   | 2421932 | kdeylyfr |

**注意**: 64 GPU 和 128 GPU 使用不同的 XLA 配置:
- 64 GPU: 默认 sharded autotuning → 2.13 s/step
- 128 GPU: shard_autotuning=false → 1.745 s/step
- 64 GPU 的 2.13 s/step 异常慢 (比 128 GPU 还慢!)，可能与 16 节点 sharded autotuning 的 kernel 质量有关
- 需要进一步调查: 重跑 64 GPU 对比有/无 shard_autotuning (v2 因 JIT 太慢而放弃)

### 对照表: 1D DDP — 已验证为稳态 (step 150-200 差值法)

| GPU | Nodes | XLA Flags          | s/step | samp/s | eff% | vs2N  | comm% | GPU·h  | Job     |
|-----|-------|--------------------|--------|--------|------|-------|-------|--------|---------|
| 8   | 2     | 默认(无flag)        | 0.460  | 121.7  | 89%  | 1.20x | 11%   | 251    | 2380062 |
| 8   | 2     | 128MB+LHS          | 0.550  | 101.8  | 75%  | 1.00x | 25%   | 301    | —       |
| 16  | 4     | 默认(无flag)        | 9.780  | 11.5   | 4%   | 0.11x | 96%   | 10689  | 2380063 |
| 16  | 4     | 128MB+LHS          | 7.180  | 15.6   | 6%   | 0.15x | 94%   | 7848   | 2419705 |
| 16  | 4     | 512MB+pipe+dbl     | 6.580  | 17.0   | 6%   | 0.17x | 94%   | 7192   | 2382308 |
| 16  | 4     | 128MB+LHS+BF16     | 6.740  | 16.6   | 6%   | 0.16x | 94%   | 7367   | 2382318 |
| 32  | 8     | 128MB+LHS          | 9.900  | 22.6   | 4%   | 0.22x | 96%   | 10820  | 2419706 |

### 2D Mesh vs 1D DDP 直接对比

| GPU | Nodes | 2D s/step | 2D samp/s | 1D s/step | 1D samp/s | 2D 加速比 |
|-----|-------|-----------|-----------|-----------|-----------|-----------|
| 8   | 2     | 0.565     | 99.1      | 0.550     | 101.8     | 1.0x      |
| 16  | 4     | 0.695     | 161.2     | 6.580     | 17.0      | **9.5x**  |
| 32  | 8     | 0.875     | 256.0     | 9.900     | 22.6      | **11.3x** |

### 通信开销分解 (2D Mesh)

| 扩展      | GPU变化 | s/step 变化     | 新增延迟 | 延迟占比 | 类型                    |
|-----------|---------|-----------------|----------|----------|-------------------------|
| 1→2 GPU   | 1→2     | 0.410→0.430     | +20ms    | 4.7%     | 节点内 DDP (NVLink)     |
| 2→4 GPU   | 2→4     | 0.430→0.455     | +25ms    | 5.5%     | 节点内 FSDP (NVLink)    |
| 4→8 GPU   | 4→8     | 0.455→0.565     | +110ms   | 19.5%    | 跨1节点 (Slingshot)     |
| 8→16 GPU  | 8→16    | 0.565→0.695     | +130ms   | 18.7%    | 跨3节点 (Slingshot)     |
| 16→32 GPU | 16→32   | 0.695→0.875     | +180ms   | 20.6%    | 跨7节点 (Slingshot)     |
| 32→64 GPU | 32→64   | 0.875→2.130     | +1255ms  | 58.9%    | 跨15节点 (★ 断崖式下降!) |
| 64→128 GPU| 64→128  | 2.130→1.745     | -385ms   | ???      | 128 用 shard_autotune=false |
| **总计**  | 1→32    | 0.410→0.875     | +465ms   | 53.1%    | 全路径 (最优区间)        |

**⚠ 64 GPU 异常**: 2.13 s/step 比 128 GPU (1.75 s/step) 还慢!
- 可能原因: 16 节点的 sharded autotuning 生成了质量较差的 CUDA kernel
- 128 GPU 用了 shard_autotuning=false (各 host 独立 autotune)，反而更快
- 需要进一步调查 (重跑 64 GPU 的 v2 因 JIT 太慢放弃了)

### 投产成本估算 (40 Epoch)

| 配置        | GPU | s/step | steps/ep | wall_40ep | GPU·h | 成本比 |
|-------------|-----|--------|----------|-----------|-------|--------|
| 2N 1D DDP   | 8   | 0.550  | 6148     | 37.6h     | 301   | 1.00x  |
| 4N 2D Mesh  | 16  | 0.695  | 3074     | 23.7h     | 380   | 1.26x  |
| 8N 2D Mesh  | 32  | 0.875  | 1537     | 14.9h     | 478   | 1.59x  |
| 4N 1D DDP   | 16  | 7.180  | 6148     | 490.5h    | 7848  | 26.1x  |
| 8N 1D DDP   | 32  | 9.900  | 3074     | 338.1h    | 10820 | 36.0x  |

### 关键结论

1. **1D DDP 稳态已验证**: 4N=7.18, 8N=9.90 — 之前测量正确，不是 autotune 偏差
2. **2D Mesh 是唯一可行的多节点方案**: 4N 加速 9.5x, 8N 加速 11.3x
3. **最优性价比**: 4N 2D Mesh (1.26x 成本 → 1.58x 吞吐)
4. **节点内扩展极好** (1→4, ~5%/倍增), **跨节点约 20%/倍增**

## BlueConnect 隔离实验 (C4 系列) — 2026-02-20

### 实验矩阵 (4N, 16 GPU, exp/C4-blueconnect commit 46eb251)

| 实验 | BlueConnect | combine | LHS | Job     | 结果    | 分析                         |
|------|-------------|---------|-----|---------|---------|------------------------------|
| C4a  | YES         | NONE    | NO  | 2421424 | TIMEOUT | 没crash! JIT 15min 未完成     |
| C4b  | YES         | 128MB   | NO  | 2421425 | CRASH   | AllGather 4→2 threads, 7:33  |
| C4c  | YES         | 128MB+  | YES | 2421426 | CRASH   | AllGather 4→2 threads, 7:45  |
| C3   | YES         | 512MB+  | YES+pipe+dbl | 2418571 | CRASH | AllGather 4→1 threads  |

### 根因定位

**combine_threshold 是 BlueConnect crash 的触发器，BlueConnect 本身和 JAX 多节点兼容！**

- C4a (BC only) 运行 15 分钟没有 crash → BC 本身正常
- C4b/C4c 加了 combine 后 crash → combine + BC 交互导致死锁
- 原因: combine 合并 101 小 AllReduce → 2-3 大 AllReduce → BC 分解成子 op →
  合并后的 buffer 有复杂依赖 → GPU 间调度不一致 → AllGather rendezvous 失败

### 待验证

- C4a-retry (job 2421754, 1h): BlueConnect ONLY + CURTAIL_EPOCHS=200, 等 JIT 完成看稳态速度
- 如果 C4a 稳态速度好 → 研究如何让 combine + BC 共存（可能需要 XLA 层面修复）

## ★★★★★ 真正根因: XLA AllReduce Combine Threshold
之前的 profiler 分析有误 — 640 vs 4040 不是每步调用次数差异，而是跨 GPU 聚合事件数。
深度分析 (subagent aaf4715) 发现:
- **每步 AllReduce 次数完全一样: 2N = 101, 4N = 101**
- **消息大小完全一样**
- **XLA 编译方式完全一样 (101 个独立 all-reduce-start)**
- 差异仅在 NCCL per-op 延迟: 2N TREE 2.4ms vs 4N RING 91ms (38x)
- 101 个 op 串行累加: 2N 0.22s vs 4N 9.2s
- XLA AllReduceCombiner pass 默认阈值极低 (~256 bytes)，不合并
- MaxText 设 128MB 阈值 → 101 → 2-3 个 AllReduce → 消除 launch latency 问题

## ★ Insight: NCCL 算法-Collective 兼容性
| Collective | Ring | Tree | CollNet | NVLS | PAT |
|------------|------|------|---------|------|-----|
| AllReduce | ✓ | ✓ | ✓ | ✓ | — |
| AllGather | ✓ | ✗ | ✗ | ✗ | ✓ |
| ReduceScatter | ✓ | ✗ | ✗ | ✗ | ✓ |
→ 全局 NCCL_ALGO=TREE 会让 AllGather 找不到可用算法

## ★ Insight: MaxText 的解决方案
- MaxText 用 NCCL Tuner Plugin (NCCL_TUNER_CONFIG_PATH) 而非 NCCL_ALGO
- 支持 per-message-size × per-collective 精细控制

## ★ Insight: CSCS Alps (同硬件) 缺失配置
我们缺少: NCCL_CROSS_NIC=1, NCCL_NET_GDR_LEVEL=PHB, NCCL_PROTO=^LL128, FI_CXI_* 等

## ★ Insight: ssm-stable 为什么没有 4N 性能问题
ssm-stable 用 jax.local_devices() 创建 LOCAL mesh → 不做跨节点通信
不是 checkpoint barrier 问题，而是 ssm-stable 的 DDP 本身就没有正确实现

## ★ Insight: C3 XLA Flags 天花板 (6.57 s/step)
- 512MB combine + pipelined AR + while_loop double_buf
- Benchmark: step150=1198s, step200=1526s → 6.56 s/step (job 2382308)
- 比 v13 (128MB) 省 4.8%，但仍 12x vs 2N → XLA flags 已到极限

## ★ Insight: 2N→4N cliff 三层根因假设
1. NCCL 算法切换: 2N=TREE(1.43ms/op), 4N=RING(45.65ms/op)
2. Channel 数骤降: 4-rank=40ch, 16-rank=8ch (5x 更少流水线)
3. Ring 跨网跳数: 2N=2hops, 4N=4hops (2x 延迟)
→ 需 3N 数据点确认 cliff 位置 (jobs 2382427+2382428 运行中)

## ★ Insight: S-Link/LB-Link 不存在于本系统
- Isambard 用 Quad GH200 (无 NVSwitch), MNNVL=0, NVLS=0
- S-Link/LB-Link 是 NVL32/NVL256 的 NVSwitch 互联术语
- 实际: NVLink 4.0 (159 GB/s) + Slingshot-11 (4×25 GB/s per node)




Kang  [6:28 PM]
Answering the meeting question: Why don’t we just use NCCL TREE for AllReduce?

  I actually made three attempts. Here’s a summary of those three TREE attempts:
  Attempt: Attempt 1
  Approach: NCCL_ALGO=TREE (global force)
  Result: Crashed immediately — because AllGather doesn’t support the TREE algorithm (see compatibility table in lines 124-130)

  Attempt: Attempt 2
  Approach: NCCL_ALGO=“allreduce:tree” (TREE only for AllReduce)
  Result: 21.83 s/step — 2.2x slower than original 9.78s!

  Attempt: Attempt 3
  Approach: CSCS Alps configuration (recommended config for the same hardware cluster)
  Result: 13.36 s/step — 37% slower than original
  Why is TREE actually slower on 4N?

  This is because the system is Quad GH200 (no NVSwitch):
  - No NVSwitch / MNNVL / NVLS
  - Actual interconnect is NVLink 4.0 (159 GB/s) + Slingshot-11 (4×25 GB/s per node)

  The TREE algorithm requires a root node to perform aggregation when crossing nodes. For hardware topologies without NVSwitch, cross-node.TREE is actually worse than RING. RING can at least utilize bidirectional bandwidth, whereas TREE’s root becomes a bottleneck.
Kang  [6:33 PM]
What about changing the NCCL collective type from AllReduce to AllGather/ReduceScatter, and then using NCCL_ALGO=TREE?

The TREE algorithm in NCCL is only implemented for AllReduce’s tree-reduce + tree-broadcast path. The data flow patterns of AllGather  and ReduceScatter (where each rank holds a different shard) have no corresponding efficient implementation on a tree structure, so NCCL simply hasn’t written this kernel.

  - DDP (AllReduce) → Can use TREE, but we tried it, and it’s 2.2x slower on current hardware
  - FSDP (AllGather + ReduceScatter) → Cannot use TREE, only RING or PAT available