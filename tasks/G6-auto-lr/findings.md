# G6 Auto-LR (Prodigy) — 最终分析报告

> 日期: 2026-02-23 | 分支: `exp/G6-auto-lr` (基于 `ignore-times-shard-map`)
> 模型: 75M S5 SSM | 数据: GOOG 2022 训练 + Jan 2023 测试

## 1. 实验目标

评估 **Prodigy**（optax.contrib.prodigy, D-Adaptation）能否替代 AdamW + 手动 LR tuning。
- Prodigy 是 learning-rate-free 的 Adam 变体，通过估计参数空间距离 d 自动设定 LR
- 代码改动仅 42 行（vs B3 分支的 474 行两阶段切换）
- 关键参数: `learning_rate=1.0, safeguard_warmup=True, weight_decay=0.05 (regular) / 0 (SSM)`

## 2. 全部实验总览

共 11 个 job，3 轮实验 + 32N sweep。**所有 job 均未完成 40 epoch**。

```
┌─────────┬──────────┬───────┬─────┬──────┬────────┬───────────┬──────────┬──────────┬──────┬────────────────┐
│ Job ID  │ Optim/LR │ Nodes │ BSZ │ gBSZ │ CURTAIL│ Best VLoss│ Best VAcc│ Best TAcc│ @Ep  │ 死因           │
├─────────┼──────────┼───────┼─────┼──────┼────────┼───────────┼──────────┼──────────┼──────┼────────────────┤
│         │ === Round 1: 2N BSZ=12 (MEM_FRAC=0.90) ===                                                    │
│ 2439386 │ Prodigy  │   2   │ 12  │  96  │  300   │ 1.893     │ 66.01%   │ 65.20%   │  1   │ OOM 80.15 GiB  │
│ 2439383 │ AdamW/3e3│   2   │ 12  │  96  │  300   │ 1.763     │ 67.37%   │ 65.95%   │  1   │ OOM 80.22 GiB  │
├─────────┼──────────┼───────┼─────┼──────┼────────┼───────────┼──────────┼──────────┼──────┼────────────────┤
│         │ === Round 2: 2N BSZ=8 (MEM_FRAC=0.85) ===                                                     │
│ 2439521 │ Prodigy  │   2   │  8  │  64  │  300   │ 1.312     │ 74.68%   │ 70.51%   │ 13   │ NCCL ep20 b44  │
│ 2439522 │ AdamW/3e3│   2   │  8  │  64  │  300   │ 1.808     │ 66.63%   │ 65.24%   │  1   │ NCCL ep13 b132 │
├─────────┼──────────┼───────┼─────┼──────┼────────┼───────────┼──────────┼──────────┼──────┼────────────────┤
│         │ === Round 3: 2N BSZ=8 LR Sweep (公平对比) ===                                                  │
│ 2439910 │ AdamW/5e4│   2   │  8  │  64  │  300   │ 1.313     │ 74.41%   │ 70.36%   │ 23   │ NCCL ep23 b16  │
│ 2439911 │ AdamW/1e3│   2   │  8  │  64  │  300   │ 1.325     │ 74.41%   │ 70.29%   │ 13   │ NCCL ep13 b264 │
│ 2439908 │ Prodigy  │   2   │  8  │  64  │  300   │ 1.418     │ 72.72%   │ 69.01%   │ 10   │ NCCL ep10 b144 │
│ 2439909 │ AdamW/3e4│   2   │  8  │  64  │  300   │ 1.372     │ 73.31%   │ 69.62%   │ 28   │ NCCL ep28 b48  │
├─────────┼──────────┼───────┼─────┼──────┼────────┼───────────┼──────────┼──────────┼──────┼────────────────┤
│         │ === 32N Production (BSZ=10, 无 CURTAIL) ===                                                    │
│ 2439976 │ Prodigy  │  32   │ 10  │ 1280 │  无    │ 1.524     │ 71.15%   │ 68.49%   │  9   │ NCCL ep9 b64   │
│ 2439703 │ AdamW/5e3│  32   │ 10  │ 1280 │  无    │ 1.628     │ 69.15%   │ 67.29%   │  1   │ NCCL + 爆炸    │
│ 2439704 │ AdamW/1e3│  32   │ 10  │ 1280 │  无    │ 1.412     │ 72.50%   │ 69.70%   │  5   │ NCCL ep5 b58   │
└─────────┴──────────┴───────┴─────┴──────┴────────┴───────────┴──────────┴──────────┴──────┴────────────────┘
```

## 3. 公平对比: 2N BSZ=8 排名

Round 2 的 AdamW/3e-3 因 LR 过高爆炸，不公平。Round 3 用多个 LR 做了公平对比：

```
┌──────┬─────────┬──────────────┬──────┬───────────┬──────────┬──────────┬───────────────────────────────┐
│ Rank │ Job     │ Optimizer/LR │ @Ep  │ Val Loss  │ Val Acc  │ Test Acc │ 备注                          │
├──────┼─────────┼──────────────┼──────┼───────────┼──────────┼──────────┼───────────────────────────────┤
│  1   │ 2439521 │ Prodigy/auto │  13  │ 1.312     │ 74.68%   │ 70.51%   │ R2, ep14 spike 后恢复中       │
│  2   │ 2439910 │ AdamW/5e-4   │  23  │ 1.313     │ 74.41%   │ 70.36%   │ R3, 最稳定, 存活最久之一      │
│  3   │ 2439911 │ AdamW/1e-3   │  13  │ 1.325     │ 74.41%   │ 70.29%   │ R3, 收敛快                    │
│  4   │ 2439909 │ AdamW/3e-4   │  28  │ 1.372     │ 73.31%   │ 69.62%   │ R3, 存活 28 ep (最久), 慢收敛 │
│  5   │ 2439908 │ Prodigy/auto │  10  │ 1.418     │ 72.72%   │ 69.01%   │ R3, 仅存活 10 ep              │
└──────┴─────────┴──────────────┴──────┴───────────┴──────────┴──────────┴───────────────────────────────┘
```

**关键发现**:
- R2 的 Prodigy (#1) 和 R3 的 Prodigy (#5) 差异大 → Prodigy 结果方差高
- Tuned AdamW (5e-4 ~ 1e-3) 与 Prodigy 最佳结果几乎打平
- AdamW 5e-4 最稳定（存活 23 ep，没有 loss spike）

## 4. 32N 排名

```
┌──────┬─────────┬──────────────┬──────┬───────────┬──────────┬──────────┬─────────────────────────────┐
│ Rank │ Job     │ Optimizer/LR │ @Ep  │ Val Loss  │ Val Acc  │ Test Acc │ 备注                        │
├──────┼─────────┼──────────────┼──────┼───────────┼──────────┼──────────┼─────────────────────────────┤
│  1   │ 2439704 │ AdamW/1e-3   │  5   │ 1.412     │ 72.50%   │ 69.70%   │ 稳定, 每 ep 改善            │
│  2   │ 2439976 │ Prodigy/auto │  9   │ 1.524     │ 71.15%   │ 68.49%   │ estim_lr=1.65e-3, 仍在改善  │
│  3   │ 2439703 │ AdamW/5e-3   │  1   │ 1.628     │ 69.15%   │ 67.29%   │ ep2 梯度爆炸 TrLoss=910     │
└──────┴─────────┴──────────────┴──────┴───────────┴──────────┴──────────┴─────────────────────────────┘
```

## 5. Prodigy estim_lr 分析

Prodigy 自动估计的 LR vs AdamW 最佳 LR：

```
┌────────────┬──────────┬──────────────────┬──────────────────┬───────────────────────────┐
│ 配置       │ gBSZ     │ Regular estim_lr │ SSM estim_lr     │ AdamW 最佳 LR             │
├────────────┼──────────┼──────────────────┼──────────────────┼───────────────────────────┤
│ 2N BSZ=8   │    64    │ 8.43e-4          │ 6.47e-4          │ 5e-4 ~ 1e-3              │
│ 32N BSZ=10 │  1280    │ 1.65e-3          │ 8.40e-4          │ 1e-3                     │
└────────────┴──────────┴──────────────────┴──────────────────┴───────────────────────────┘
```

- 2N: estim_lr=8.43e-4 精确落入 AdamW 最佳区间 [5e-4, 1e-3] ✓
- 32N: estim_lr=1.65e-3 略高于 AdamW 最佳 1e-3（1.65x）
- Batch size scaling: gBSZ 20x 增大 → estim_lr 仅 1.96x 增大（vs sqrt 理论预测 4.47x）
- **Prodigy 的 D-estimation 偏保守**，大 batch 下低估最优 LR

## 6. 速度对比

```
┌────────────┬──────────────┬──────────────┬────────┐
│ 配置       │ Prodigy      │ AdamW        │ 差异   │
├────────────┼──────────────┼──────────────┼────────┤
│ 2N BSZ=12  │ 0.234 s/step │ 0.234 s/step │ <1%    │
│ 2N BSZ=8   │ 0.51 s/step  │ 0.51 s/step  │ <1%    │
│ 32N BSZ=10 │ 1.08 s/step  │ 1.08 s/step  │ <1%    │
└────────────┴──────────────┴──────────────┴────────┘
```

**结论**: Prodigy 的 4x optimizer state (vs Adam 2x) 不影响训练吞吐。
75M 模型额外 ~600MB（bf16），在 85.5 GB GH200 上可忽略。

## 7. 稳定性分析

### Prodigy 的 Loss Spike
- R2 (2439521): epoch 14 Train Loss 1.39 → 5.48 (4x)，后逐步恢复
- R3 (2439908): 仅存活 10 epoch，未出现 spike 但最终指标偏低
- 32N (2439976): epoch 1→3 Val Loss 恶化 (1.607→1.813)，之后恢复并持续改善

### AdamW 的 LR 敏感性
- LR=3e-3 at gBSZ=64: epoch 2 Train Loss 爆炸 (→164)
- LR=5e-3 at gBSZ=1280: epoch 2 Train Loss 爆炸 (→910)
- LR=5e-4: 最稳定，存活 23 epoch，无 spike
- LR=3e-4: 最长存活 28 epoch，但收敛慢

## 8. 最终结论

### Prodigy 的价值
| 维度 | 评价 |
|------|------|
| **LR 估计准确性** | ✅ 2N: 精准 (8.4e-4 ∈ [5e-4, 1e-3])。32N: 偏高 (1.65e-3 vs 最佳 1e-3) |
| **收敛速度** | ✅ 2N R2: ep13 达到 AdamW 5e-4 在 ep23 才达到的水平 |
| **最终精度** | ⚠️ 公平对比下 Prodigy 落后 tuned AdamW ~1.7pp Val Acc |
| **稳定性** | ⚠️ Loss spike (ep14) + 结果方差高 (R2 vs R3 差 2pp) |
| **速度开销** | ✅ 零影响 (<1%) |
| **省去 LR sweep** | ✅ 最大价值：4 个 AdamW LR 实验 = 4x GPU 时间 |

### 推荐使用方式
1. **探索阶段**: 用 Prodigy 快速找到近似最优 LR（无需 sweep）
2. **Production 训练**: 读取 Prodigy 的 estim_lr，用该值作为 AdamW 的 LR 起点
3. **不建议直接用 Prodigy 做 production 训练**（稳定性风险 + 略逊于 tuned AdamW）

### 对 75M S5 SSM 的具体建议
- **2N (gBSZ=64)**: AdamW LR=5e-4 ~ 1e-3，cosine decay
- **32N (gBSZ=1280)**: AdamW LR=1e-3，cosine decay
- **SSM LR**: 与 regular LR 相同（Prodigy SSM estim_lr 接近 regular）

## W&B 链接

所有 run 在 project `lobs5-75M-G6`:
- R1 Prodigy BSZ=12: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/9yskmc85
- R1 AdamW BSZ=12: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/6q4n2ovb
- R2 Prodigy BSZ=8: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/w2ol0ivv
- R2 AdamW BSZ=8: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/rhmioz1b
- R3 Prodigy BSZ=8: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/y05mjk2i
- R3 AdamW/1e-3: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/g6apyxjs
- R3 AdamW/5e-4: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/5qdiwsxo
- R3 AdamW/3e-4: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/fkhhn6qz
- 32N Prodigy: https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/ywg1j08o
