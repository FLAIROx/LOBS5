#!/usr/bin/env python3
"""
ES-LOBS5 主训练脚本 (Main Training Script)

所有生产作业都调用这个脚本，通过环境变量或命令行参数配置。

Usage:
    # 通过环境变量配置
    MODEL_PRESET=L1 python es_train_main.py

    # 通过命令行参数配置
    python es_train_main.py --preset L1
    python es_train_main.py --task_size 10 --noiser eggrollbs --n_steps 10

预设配置:
    D5: n_steps=10, bg_msgs=100, task_size=500, noiser=eggroll
    G5: n_steps=10, bg_msgs=100, task_size=500, noiser=eggroll
    H5: n_steps=100, bg_msgs=10, task_size=500, noiser=eggroll
    I5: n_steps=100, bg_msgs=10, task_size=100, noiser=eggroll
    J1-J4: n_steps=10, bg_msgs=10, task_size=500, noiser=eggroll (different file_idx)
    K1-K4: n_steps=10, bg_msgs=10, task_size=100, noiser=eggroll (different file_idx)
    L1-L4: n_steps=10, bg_msgs=10, task_size=10, noiser=eggrollbs (different file_idx)
"""

import os
import sys
import argparse
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# WandB configuration (must be set before import)
os.environ.setdefault('WANDB_MODE', 'online')
os.environ.setdefault('WANDB_BASE_URL', 'https://api.wandb.ai')
os.environ.setdefault('WANDB_INSECURE_DISABLE_SSL', 'True')

import jax
import jax.numpy as jnp
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not available")


# =============================================================================
# 配置预设 (Configuration Presets)
# =============================================================================
PRESETS = {
    # D series: baseline config
    'D5': {
        'n_steps': 10,
        'background_msgs_per_step': 100,
        'task_size': 500,
        'noiser': 'eggroll',
        'file_idx': 0,
        'group_size': 0,
    },
    # G series: same as D5
    'G5': {
        'n_steps': 10,
        'background_msgs_per_step': 100,
        'task_size': 500,
        'noiser': 'eggroll',
        'file_idx': 0,
        'group_size': 0,
    },
    # H series: more steps, less bg_msgs
    'H5': {
        'n_steps': 100,
        'background_msgs_per_step': 10,
        'task_size': 500,
        'noiser': 'eggroll',
        'file_idx': 0,
        'group_size': 0,
    },
    # I series: smaller task_size
    'I5': {
        'n_steps': 100,
        'background_msgs_per_step': 10,
        'task_size': 100,
        'noiser': 'eggroll',
        'file_idx': 0,
        'group_size': 0,
    },
    # J series: different file_idx for data diversity
    'J1': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 500, 'noiser': 'eggroll', 'file_idx': 0, 'group_size': 0},
    'J2': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 500, 'noiser': 'eggroll', 'file_idx': 50, 'group_size': 0},
    'J3': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 500, 'noiser': 'eggroll', 'file_idx': 100, 'group_size': 0},
    'J4': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 500, 'noiser': 'eggroll', 'file_idx': 150, 'group_size': 0},
    # K series: smaller task_size with data diversity
    'K1': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 100, 'noiser': 'eggroll', 'file_idx': 0, 'group_size': 0},
    'K2': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 100, 'noiser': 'eggroll', 'file_idx': 50, 'group_size': 0},
    'K3': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 100, 'noiser': 'eggroll', 'file_idx': 100, 'group_size': 0},
    'K4': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 100, 'noiser': 'eggroll', 'file_idx': 150, 'group_size': 0},
    # L series: EggRollBS with baseline subtraction, very small task_size
    'L1': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 10, 'noiser': 'eggrollbs', 'file_idx': 0, 'group_size': 8},
    'L2': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 10, 'noiser': 'eggrollbs', 'file_idx': 50, 'group_size': 8},
    'L3': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 10, 'noiser': 'eggrollbs', 'file_idx': 100, 'group_size': 8},
    'L4': {'n_steps': 10, 'background_msgs_per_step': 10, 'task_size': 10, 'noiser': 'eggrollbs', 'file_idx': 150, 'group_size': 8},
}


# =============================================================================
# 默认路径 (Default Paths)
# =============================================================================
DEFAULT_CHECKPOINT = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
DEFAULT_DATA_PATH = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
DEFAULT_OUTPUT_DIR = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints'


# =============================================================================
# 配置数据类 (Configuration Dataclass)
# =============================================================================
@dataclass
class ESConfig:
    """ES训练配置"""
    # Checkpoint and data paths
    lobs5_checkpoint: str = DEFAULT_CHECKPOINT
    replay_data_path: str = DEFAULT_DATA_PATH
    data_dir: str = DEFAULT_DATA_PATH

    # ES algorithm
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0
    group_size: int = 0  # For EggRollBS

    # Training scale
    n_perturbations: int = 1024
    n_epochs: int = 1000
    n_steps: int = 10
    n_warmup_msgs: int = 500
    background_msgs_per_step: int = 10

    # Token mode
    token_mode: int = 24
    background_mode: str = 'historical_replay'

    # Task
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100

    # Data selection
    file_idx: int = 0

    # Checkpointing
    checkpoint_dir: str = ''
    checkpoint_every: int = 10
    output_dir: str = DEFAULT_OUTPUT_DIR

    # Other
    seed: int = 42
    max_training_hours: float = 23.5

    # WandB
    wandb_project: str = 'ES-LOBS5'
    wandb_name: str = ''
    wandb_tags: list = field(default_factory=list)

    def apply_preset(self, preset_name: str):
        """应用预设配置"""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

        preset = PRESETS[preset_name]
        for key, value in preset.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Set default checkpoint_dir and wandb_name based on preset
        if not self.checkpoint_dir:
            self.checkpoint_dir = f'{DEFAULT_OUTPUT_DIR}/es_production_{preset_name.lower()}'
        if not self.wandb_name:
            self.wandb_name = f'{preset_name}_steps{self.n_steps}_bg{self.background_msgs_per_step}_task{self.task_size}'
            if self.noiser == 'eggrollbs':
                self.wandb_name += '_eggrollbs'
        if preset_name not in self.wandb_tags:
            self.wandb_tags.append(preset_name)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ES-LOBS5 Main Training Script')

    # Preset
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Configuration preset (D5, G5, H5, I5, J1-J4, K1-K4, L1-L4)')

    # ES algorithm
    parser.add_argument('--noiser', type=str, default=None,
                        choices=['eggroll', 'eggrollbs', 'open_es', 'sparse'])
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--group_size', type=int, default=None)

    # Training scale
    parser.add_argument('--n_perturbations', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--background_msgs_per_step', type=int, default=None)

    # Task
    parser.add_argument('--task_size', type=int, default=None)
    parser.add_argument('--file_idx', type=int, default=None)

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)

    # WandB
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')

    # Other
    parser.add_argument('--seed', type=int, default=None)

    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()

    # 也支持环境变量配置
    preset = args.preset or os.environ.get('MODEL_PRESET')

    # 创建配置
    config = ESConfig()

    # 应用预设
    if preset:
        config.apply_preset(preset)
        print(f"[CONFIG] Applied preset: {preset}")

    # 应用命令行参数覆盖
    if args.noiser: config.noiser = args.noiser
    if args.sigma: config.sigma = args.sigma
    if args.lr: config.lr = args.lr
    if args.group_size is not None: config.group_size = args.group_size
    if args.n_perturbations: config.n_perturbations = args.n_perturbations
    if args.n_epochs: config.n_epochs = args.n_epochs
    if args.n_steps: config.n_steps = args.n_steps
    if args.background_msgs_per_step: config.background_msgs_per_step = args.background_msgs_per_step
    if args.task_size: config.task_size = args.task_size
    if args.file_idx is not None: config.file_idx = args.file_idx
    if args.checkpoint_dir: config.checkpoint_dir = args.checkpoint_dir
    if args.data_path:
        config.replay_data_path = args.data_path
        config.data_dir = args.data_path
    if args.wandb_name: config.wandb_name = args.wandb_name
    if args.seed: config.seed = args.seed

    # 确保 checkpoint_dir 存在
    if not config.checkpoint_dir:
        config.checkpoint_dir = f'{DEFAULT_OUTPUT_DIR}/es_production_custom'
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 打印配置
    print("=" * 60)
    print("ES-LOBS5 Production Training")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Config:")
    print(f"  noiser: {config.noiser}")
    print(f"  n_perturbations: {config.n_perturbations}")
    print(f"  n_steps: {config.n_steps}")
    print(f"  background_msgs_per_step: {config.background_msgs_per_step}")
    print(f"  task_size: {config.task_size}")
    print(f"  file_idx: {config.file_idx}")
    print(f"  group_size: {config.group_size}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    print("=" * 60)

    # 初始化 WandB
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_name or f'run_{config.seed}',
            config=asdict(config),
            tags=config.wandb_tags + ['production', '24h']
        )
        print(f"WandB: {wandb_run.name}")

    # 初始化训练器
    print("\n[1/3] Initializing trainer...")
    t0 = time.time()
    from es_lobs5.training.es_trainer import ESTrainer
    trainer = ESTrainer(config)
    print(f"  Init time: {time.time() - t0:.1f}s")

    # 创建初始状态
    print("\n[2/3] Creating initial state...")
    t0 = time.time()
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  State time: {time.time() - t0:.1f}s")

    # 训练循环
    print("\n[3/3] Starting training loop...")
    print("=" * 60)

    key = jax.random.PRNGKey(config.seed)
    training_start = time.time()
    max_training_time = config.max_training_hours * 3600

    best_fitness = -float('inf')
    fitness_history = []

    for epoch in range(config.n_epochs):
        # 检查时间限制
        elapsed = time.time() - training_start
        if elapsed > max_training_time:
            print(f"\nTime limit reached ({elapsed/3600:.1f}h). Stopping.")
            break

        epoch_start = time.time()
        key, epoch_key = jax.random.split(key)

        # 训练一个 epoch
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )

        # 收集指标
        mf = float(mean_fitness.block_until_ready())
        sf = float(jnp.std(fitnesses))
        max_f = float(jnp.max(fitnesses))
        min_f = float(jnp.min(fitnesses))
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start

        fitness_history.append(mf)
        if mf > best_fitness:
            best_fitness = mf

        # 日志
        print(f"Epoch {epoch+1:4d} | Fitness: {mf:8.2f} +/- {sf:6.2f} | "
              f"Time: {epoch_time:5.1f}s | Total: {total_elapsed/3600:.2f}h")

        # WandB 日志
        if wandb_run:
            wandb.log({
                'epoch': epoch + 1,
                'fitness/mean': mf,
                'fitness/std': sf,
                'fitness/max': max_f,
                'fitness/min': min_f,
                'fitness/best': best_fitness,
                'time/epoch_seconds': epoch_time,
                'time/total_hours': total_elapsed / 3600,
            }, step=epoch + 1)

        # 定期保存
        if (epoch + 1) % config.checkpoint_every == 0:
            np.save(os.path.join(config.checkpoint_dir, 'fitness_history.npy'),
                    np.array(fitness_history))

    # 完成
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total epochs: {len(fitness_history)}")
    print(f"  Best fitness: {best_fitness:.4f}")
    print(f"  Final fitness: {fitness_history[-1] if fitness_history else 0:.4f}")

    # 保存最终结果
    np.save(os.path.join(config.checkpoint_dir, 'fitness_history.npy'),
            np.array(fitness_history))

    if wandb_run:
        wandb.finish()

    return 0


if __name__ == '__main__':
    sys.exit(main())
