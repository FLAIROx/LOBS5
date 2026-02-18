# LOBS5 运行文档

> **LOBS5** = Limit Order Book S5 — 基于 S5 (Simplified Structured State Space) 架构的 LOB 消息流 Token-Level 自回归生成模型
>
> **论文**: *Generative AI for End-to-End Limit Order Book Modelling* (Nagy, Frey, Sapora, Li, Calinescu, Zohren, Foerster)
>
> **分支**: `exp/B1-ignore-times-v2` (生产分支)

---

## 目录

1. [项目结构](#1-项目结构)
2. [数据源与权限](#2-数据源与权限)
3. [环境配置](#3-环境配置)
4. [运行流程](#4-运行流程)
5. [参数说明](#5-参数说明)
6. [Checkpoint 管理](#6-checkpoint-管理)
7. [关键文件路径表](#7-关键文件路径表)

---

## 1. 项目结构

```
LOBS5/
├── run_train.py              # 训练入口 (CLI 参数 + 多节点初始化)
├── run_eval.py               # 评估入口 (加载 checkpoint 验证)
├── run_inference.py          # 推理/生成入口
├── preproc.py                # 数据预处理 (LOBSTER CSV → .npy)
├── train_full_autoreg.batch  # SLURM 批处理脚本 (生产训练)
│
├── lob/                      # 核心 LOB 模型代码
│   ├── train.py              # 训练循环 (epoch, LR schedule, checkpoint)
│   ├── train_helpers.py      # train_step / eval_step / optimizer / LR schedule
│   ├── lob_seq_model.py      # 模型架构 (PaddedLobPredModel = 生产模型)
│   ├── encoding.py           # 消息 Tokenizer (Vocab, 编码/解码)
│   ├── dataloading.py        # Dataset 工厂函数
│   ├── lobster_dataloader.py # LOBSTER_Dataset, Sampler, 数据加载
│   ├── init_train.py         # 模型初始化, Checkpoint 加载/保存
│   ├── sharding_utils.py     # JAX mesh/sharding (多 GPU 并行)
│   ├── inference.py          # 自回归生成 (带 error correction)
│   └── evaluation.py         # 评估指标 (Wasserstein, mid-price loss 等)
│
├── s5/                       # S5 SSM 核心代码
│   ├── ssm.py                # S5SSM 模块 (离散化, parallel scan, RNN mode)
│   ├── ssm_init.py           # SSM 初始化 (HiPPO-LegS 矩阵)
│   ├── layers.py             # SequenceLayer (S5 + norm + activation + dropout)
│   └── seq_model.py          # StackedEncoderModel (embedding + S5 stack)
│
├── Alphatrade/               # Git submodule (Jax-LOB 模拟器, 仅推理用)
├── checkpoints/              # 保存的模型 checkpoint (Orbax 格式)
└── logs_lobs5/               # SLURM 训练日志
```

---

## 2. 数据源与权限

### 2.1 数据来源

| 数据集 | 来源 | 说明 |
|--------|------|------|
| LOBSTER | https://lobsterdata.com/ | NASDAQ Level 2 限价订单簿重构数据 |
| 股票 | GOOG (Google/Alphabet) | NASDAQ 上市 |
| 训练集 | 2022 全年 | 249 个交易日 |
| 测试集 | 2023 年 1 月 | 9 个交易日 |

### 2.2 数据路径 (硬编码)

```bash
# 训练数据 (249 天, GOOG 2022)
TRAIN_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022"

# 测试数据 (9 天, GOOG Jan 2023)
TEST_DIR="/projects/s5e/quant/JAN2023/GOOG_24tok_preproc"
```

> **⚠️ 禁止使用以下数据集:**
> - `GOOG_2018_2022_combined` — 混合了 2018-2022 + Jan 2023, 不可用
> - `GOOGJAN2023_encoded` — encoded 格式, 不是 preprocessed 格式

### 2.3 数据权限

- LOBSTER 数据需要 **商业许可证** 才能访问 NASDAQ Level 2 数据
- 当前数据由 Oxford 团队预购买并存放于 Isambard-AI 集群
- 数据目录权限: 项目组 `s5e` 成员可读
- **新成员需要**: 加入 `s5e` 项目组 + 确认 LOBSTER 数据使用条款

### 2.4 数据格式

每个交易日包含两个 `.npy` 文件:

```
GOOG_2022-01-03_34200000_57600000_message_10_proc.npy   # 消息序列 (N, 14)
GOOG_2022-01-03_34200000_57600000_orderbook_10_proc.npy  # 订单簿快照 (N, 503)
```

**消息格式 (14 字段)**:
```
[order_id, event_type, direction, price_abs, price_rel, size,
 delta_t_s, delta_t_ns, time_s, time_ns,
 price_ref, size_ref, time_s_ref, time_ns_ref]
```

**Token 化**: 每条消息 → 22 个 token, 序列长度 = `msg_seq_len × 22` = 11,000 tokens

**词表大小**: ~11,012 tokens

---

## 3. 环境配置

### 3.1 Conda 环境

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lob
```

环境路径: `/lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/envs/lob`

### 3.2 CUDA 模块

```bash
module load cuda/12.6
```

### 3.3 关键依赖

| 包 | 用途 |
|----|------|
| `jax` + `jaxlib` | 核心 ML 框架 (需独立安装) |
| `flax` | JAX 神经网络库 |
| `optax` | 优化器库 |
| `orbax-checkpoint==0.11.18` | Checkpoint 管理 (**必须固定版本**) |
| `wandb` | 实验追踪 |
| `torch` | 仅用于 DataLoader (不用 GPU) |
| `numpy`, `pandas`, `scipy` | 数据处理 |
| `gymnax` | RL 环境 (Jax-LOB 模拟器, 推理用) |

### 3.4 多节点通信 (LD_LIBRARY_PATH)

多节点训练需要 AWS OFI NCCL 插件 (Slingshot 网络):
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH"
```

完整路径见 `train_full_autoreg.batch` L124-128。

---

## 4. 运行流程

### 4.1 数据预处理 (一次性)

```bash
cd /projects/s5e/quant/AlphaTrade/LOBS5

python preproc.py \
    --data_dir /path/to/raw/LOBSTER/GOOG/2022/ \
    --save_dir /path/to/processed/GOOG/2022/ \
    --n_tick_range 500 \
    --use_raw_book_repr \
    --skip_existing
```

> **注意**: 当前训练/测试数据已预处理完成, 无需重新运行

### 4.2 快速测试 (单节点, 30min)

```bash
# 提交测试任务 (1 节点, 10 步/epoch, 30 分钟时限)
CURTAIL_EPOCHS=10 PER_GPU_BSZ=4 sbatch --nodes=1 --time=00:30:00 train_full_autoreg.batch
```

### 4.3 生产训练 (多节点)

```bash
# 8 节点生产训练 (~24h, 32 GPUs)
PER_GPU_BSZ=4 sbatch train_full_autoreg.batch
```

**多节点训练原理**:
```
┌─── Node 0 ───┐  ┌─── Node 1 ───┐       ┌─── Node 7 ───┐
│ GPU0 GPU1     │  │ GPU0 GPU1     │  ...  │ GPU0 GPU1     │
│ GPU2 GPU3     │  │ GPU2 GPU3     │       │ GPU2 GPU3     │
└───────────────┘  └───────────────┘       └───────────────┘
        ↕ Slingshot 网络 (NCCL + AWS OFI) ↕

1. SLURM 分配 N 个节点, 每节点 4 GPU
2. scontrol 获取 master 节点地址 → JAX_COORDINATOR_ADDRESS
3. 每节点 srun 启动 1 个进程 (4 GPU)
4. jax.distributed.initialize() 建立全局 mesh
5. 数据并行: 每 GPU 拿 batch 的 1/N_gpus
6. 梯度同步: JAX 自动 allreduce
7. Checkpoint: 仅 rank 0 保存
```

### 4.4 评估

```bash
python run_eval.py \
    --restore=checkpoints/<checkpoint_dir>/ \
    --restore_step=<epoch> \
    --dir_name="$TRAIN_DIR" \
    --bsz=16 \
    --num_devices=4 \
    --ignore_times=True \
    --epochs=1
```

### 4.5 推理/生成

```bash
python run_inference.py \
    --stock GOOG \
    --checkpoint_step 23 \
    --test_split 0 \
    --batch_size 32 \
    --n_sequences 1024 \
    --n_cond_msgs 500
```

> **注意**: `run_inference.py` 内有硬编码路径, 使用前需检查并更新

---

## 5. 参数说明

### 5.1 模型架构参数

| 参数 | 默认值 | 生产值 | 说明 |
|------|--------|--------|------|
| `--d_model` | 32 | **1024** | 隐藏维度 H (特征大小) |
| `--n_layers` | 6 | **12** | Fused S5 层数 (消息+订单簿融合后) |
| `--n_message_layers` | 2 | 2 | 消息编码器 S5 层数 |
| `--n_book_pre_layers` | 1 | 1 | 订单簿编码器预投影层数 |
| `--n_book_post_layers` | 1 | 1 | 订单簿编码器后投影层数 |
| `--ssm_size_base` | 32 | **1024** | SSM 状态大小 P |
| `--blocks` | 8 | **16** | HiPPO 对角块数 J |
| `--activation_fn` | `half_glu1` | `half_glu1` | 激活函数 |
| `--merging` | `projected` | **`padded`** | 消息+订单簿融合方式 |
| `--clip_eigs` | False | **True** | 约束特征值到左半平面 |
| `--conj_sym` | True | True | 共轭对称 (减半状态大小) |
| `--discretization` | `zoh` | `zoh` | 零阶保持离散化 |

**模型尺寸对照表**:

| 配置 | d_model | n_layers | ssm_size | blocks | 参数量 |
|------|---------|----------|----------|--------|--------|
| Tiny | 128 | 6 | 128 | 8 | ~2M |
| Small | 512 | 12 | 512 | 16 | ~20M |
| **Large (生产)** | **1024** | **12** | **1024** | **16** | **~75M** |

### 5.2 训练参数

| 参数 | 默认值 | 生产值 | 说明 |
|------|--------|--------|------|
| `--bsz` | 16 | 按 GPU 数量调整 | 每进程 batch size (分给本地 GPU) |
| `--epochs` | 100 | **40** | 训练 epoch 数 |
| `--ssm_lr_base` | 1e-3 | **5e-4** | SSM 参数学习率 (Lambda, B, log_step) |
| `--lr_factor` | 1 | 1 | 全局 LR = lr_factor × ssm_lr_base |
| `--weight_decay` | 0.05 | 0.05 | AdamW 权重衰减 (非 SSM 参数) |
| `--cosine_anneal` | True | True | 余弦退火 LR 调度 |
| `--warmup_end` | 1 | 1 | 线性预热结束 epoch |
| `--p_dropout` | 0.0 | 0.0 | Dropout 概率 |
| `--prenorm` | True | True | 使用 Pre-Normalization |
| `--batchnorm` | True | **False** | 生产用 LayerNorm 而非 BatchNorm |
| `--ignore_times` | False | **True** | 跳过时间 token 的 loss (重要!) |
| `--curtail_epochs` | None | None | 每 epoch 提前结束步数 (测试用) |

### 5.3 数据参数

| 参数 | 默认值 | 生产值 | 说明 |
|------|--------|--------|------|
| `--dir_name` | `./data/` | 见§2.2 | 训练数据目录 |
| `--test_dir_name` | None | 见§2.2 | 测试数据目录 (独立指定) |
| `--msg_seq_len` | 500 | 500 | 每序列消息数 (×22 = 11000 tokens) |
| `--use_book_data` | False | **True** | 使用订单簿数据 |
| `--book_transform` | False | **True** | 将 L2 转为 volume image |
| `--book_depth` | 500 | 500 | 订单簿价格层数 |
| `--masking` | `causal` | **`none`** | Masking 策略 (自回归用 none) |
| `--n_data_workers` | 0 | **12** | DataLoader worker 数 |
| `--random_offsets_train` | True | True | 每 epoch 随机化序列起始偏移 |

### 5.4 基础设施参数

| 参数 | 默认值 | 生产值 | 说明 |
|------|--------|--------|------|
| `--num_devices` | 1 | **4** | 每节点 GPU 数 |
| `--jax_seed` | 1919 | **42** | 随机种子 |
| `--USE_WANDB` | True | True | 启用 W&B 日志 |
| `--wandb_project` | `LOBS5v2` | `lobs5-75M-B1` | W&B 项目名 |
| `--wandb_entity` | `sasrey` | **`kang-oxford`** | W&B 实体 |
| `--restore` | None | - | Checkpoint 路径 (恢复训练) |
| `--restore_step` | None | - | 恢复的 checkpoint step |

### 5.5 环境变量覆盖 (train_full_autoreg.batch)

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PER_GPU_BSZ` | 8 | 每 GPU batch size |
| `CURTAIL_EPOCHS` | (无) | 每 epoch 提前结束步数 |
| `SSM_LR_BASE` | 0.0005 | SSM 学习率 |
| `LR_FACTOR` | 1 | 全局 LR 乘数 |
| `WARMUP_END` | 1 | 预热 epoch 数 |
| `EPOCHS` | 40 | 总 epoch 数 |
| `WANDB_PROJECT` | `lobs5-75M-B1` | W&B 项目名 |

---

## 6. Checkpoint 管理

### 6.1 格式

- **库**: Orbax CheckpointManager (**必须 v0.11.18**)
- **位置**: `checkpoints/{wandb_run_name}_{wandb_run_id}/`
- **内容**:
  - `state/`: Flax TrainState (params + optimizer state)
  - `metadata/_ROOT_METADATA`: JSON (config, metrics, training args)
- **保留策略**: 保留最近 10 个, 每第 5 个 epoch 永久保留

### 6.2 恢复训练

```bash
python run_train.py \
    --restore=checkpoints/<run_name>_<run_id>/ \
    --restore_step=<epoch> \
    ... # 其他参数保持一致
```

### 6.3 注意事项

- Checkpoint **仅 rank 0 保存** (多节点时)
- Orbax 异步保存在磁盘配额耗尽时 **静默失败** — 注意检查 project quota
- 检查配额命令:
  ```bash
  lfs quota -h -p $(lfs project -d /lus/lfs1aip2/projects/s5e 2>/dev/null | awk '{print $1}') /lus/lfs1aip2
  ```

---

## 7. 关键文件路径表

### 代码文件

| 文件 | 绝对路径 | 用途 |
|------|----------|------|
| 训练入口 | `/projects/s5e/quant/AlphaTrade/LOBS5/run_train.py` | CLI 参数 + 多节点初始化 |
| 训练循环 | `/projects/s5e/quant/AlphaTrade/LOBS5/lob/train.py` | Epoch 循环, 验证, Checkpoint |
| 训练工具 | `/projects/s5e/quant/AlphaTrade/LOBS5/lob/train_helpers.py` | train_step, eval_step, LR |
| 模型架构 | `/projects/s5e/quant/AlphaTrade/LOBS5/lob/lob_seq_model.py` | PaddedLobPredModel (生产) |
| S5 SSM | `/projects/s5e/quant/AlphaTrade/LOBS5/s5/ssm.py` | S5SSM, parallel scan |
| S5 层 | `/projects/s5e/quant/AlphaTrade/LOBS5/s5/layers.py` | SequenceLayer |
| 编码器栈 | `/projects/s5e/quant/AlphaTrade/LOBS5/s5/seq_model.py` | StackedEncoderModel |
| Tokenizer | `/projects/s5e/quant/AlphaTrade/LOBS5/lob/encoding.py` | Vocab, 编码/解码 |
| 数据加载 | `/projects/s5e/quant/AlphaTrade/LOBS5/lob/lobster_dataloader.py` | Dataset, Sampler |
| Sharding | `/projects/s5e/quant/AlphaTrade/LOBS5/lob/sharding_utils.py` | JAX mesh/sharding |
| 推理 | `/projects/s5e/quant/AlphaTrade/LOBS5/lob/inference.py` | 自回归生成 |
| 评估 | `/projects/s5e/quant/AlphaTrade/LOBS5/lob/evaluation.py` | Wasserstein 等指标 |
| 预处理 | `/projects/s5e/quant/AlphaTrade/LOBS5/preproc.py` | LOBSTER CSV → .npy |
| SLURM 脚本 | `/projects/s5e/quant/AlphaTrade/LOBS5/train_full_autoreg.batch` | 多节点训练提交 |

### 数据文件

| 数据集 | 绝对路径 | 说明 |
|--------|----------|------|
| 训练集 | `/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022` | 249 天 GOOG 2022 |
| 测试集 | `/projects/s5e/quant/JAN2023/GOOG_24tok_preproc` | 9 天 GOOG Jan 2023 |
| Checkpoint | `/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/` | Orbax 格式 |
| 训练日志 | `/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/` | SLURM 输出 |

### 配置文件

| 文件 | 绝对路径 | 用途 |
|------|----------|------|
| 依赖 | `/projects/s5e/quant/AlphaTrade/LOBS5/requirements.txt` | Python 包 |
| 经验教训 | `/projects/s5e/quant/AlphaTrade/LOBS5/learned_lessons.md` | 调试经验 |
| 项目说明 | `/projects/s5e/quant/AlphaTrade/LOBS5/CLAUDE.md.bak` | Claude 指令 |

---

## 附录: 常见问题

### Q1: 75M 模型每 GPU 最大 batch size?
**A**: 4 (GH200 96GB)。BSZ ≥ 5 会 OOM。BF16 下 bsz=4, FP32 下 bsz=2。

### Q2: 多节点训练 NCCL 报错?
**A**: 检查 `LD_LIBRARY_PATH` 是否包含 AWS OFI NCCL 路径。确认 `module load cuda/12.6`。

### Q3: Checkpoint 保存后加载失败?
**A**: 确认 `orbax-checkpoint==0.11.18` (版本必须固定)。检查磁盘配额是否耗尽 (Orbax 异步保存静默失败)。

### Q4: `curtail_epochs` 测试后训练卡住?
**A**: `curtail_epochs` 模式最多运行 ~30 分钟, 超时后 job 变僵尸。需及时 scancel。

### Q5: SLURM 用什么分区?
**A**: 始终使用 `workq` (不是 `batch`)。
