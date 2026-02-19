# ssm-stable-08-feb 分析进度

## 会话记录

| 日期 | Session ID | 描述 | pwd | JSONL 路径 | 恢复命令 |
|------|-----------|------|-----|-----------|---------|
| 2026-02-18 | dc29e4b2 | ssm-stable 分支 DDP 分析 + 对比 B1 | /projects/s5e/quant/AlphaTrade/LOBS5 | /projects/s5e/quant/.claude/projects/-lus-lfs1aip2-projects-s5e-quant-AlphaTrade-LOBS5/dc29e4b2-3321-4030-8d65-d15ebd78b6ea.jsonl | `cd /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5 && claude --resume dc29e4b2` |

## 完成项

- [x] 分析 ssm-stable-08-feb 的 DDP 架构 (LOCAL mesh, 无跨节点梯度同步)
- [x] 对比 exp/B1-ignore-times-v2 的 GLOBAL mesh 实现
- [x] 记录 5 维度对比 + NCCL 配置差异 + B1 修复的 5 个 bug
- [x] 写入 findings.md
- [x] BF16 混合精度对比分析 (ssm-stable BF16 sandwich vs B1 全 FP32)
- [x] HyperscaleES 多节点模式分析 (shard_map + process_allgather + 确定性噪声)
- [x] MaxText 多节点模式分析 (ICI/DCN hybrid mesh + SPMD 自动 allreduce)
- [x] 三方对比表 + MaxText 核心文件引用 + ★ Insights
