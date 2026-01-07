#!/usr/bin/env python3
"""
ES-LOBS5 Production Training Script

Combines best practices from:
- A-E validation series: checkpoint loading, trainer initialization, checkpointing
- H series: shard_map multi-GPU distribution (92% GPU utilization achieved)
- I series: JAX compilation cache optimization
- HyperscaleES: mesh configuration patterns

Usage:
    # Quick test (single GPU, 5 epochs)
    python production_train.py --preset quick_test

    # Medium training (4 GPU, 100 epochs)
    python production_train.py --preset medium

    # Full production (4 GPU, 1000 epochs)
    python production_train.py --preset production

    # Custom configuration
    python production_train.py --n_threads 256 --n_epochs 500 --n_steps 100

    # Resume from checkpoint
    python production_train.py --preset production --resume_from /path/to/checkpoint

Author: ES-LOBS5 Team
"""

import os
import sys
import gc
import time
import argparse
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# JAX Configuration (MUST be before importing jax)
# ==============================================================================
# I1-I3: Persistent compilation cache for faster warm starts
_jax_cache_dir = os.path.expanduser("~/.cache/es_lobs5_jax_compilation")
os.makedirs(_jax_cache_dir, exist_ok=True)
os.environ.setdefault("XLA_FLAGS", f"--xla_dump_to=/tmp/xla_dump")

import jax
import jax.numpy as jnp

# Configure JAX compilation cache (from HyperscaleES)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)  # Cache all
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # Cache all

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[INFO] wandb not available, will skip W&B logging")


# ==============================================================================
# Configuration Presets
# ==============================================================================
PRESETS = {
    "quick_test": {
        "description": "Quick test - single GPU, minimal epochs",
        "n_threads": 32,
        "n_epochs": 5,
        "n_steps": 50,
        "background_msgs_per_step": 5,
        "checkpoint_every": 2,
    },
    "debug": {
        "description": "Debug - fast iteration, small scale",
        "n_threads": 64,
        "n_epochs": 20,
        "n_steps": 50,
        "background_msgs_per_step": 5,
        "checkpoint_every": 5,
    },
    "medium": {
        "description": "Medium - balanced training",
        "n_threads": 128,
        "n_epochs": 100,
        "n_steps": 100,
        "background_msgs_per_step": 5,
        "checkpoint_every": 20,
    },
    "production": {
        "description": "Production - full scale training",
        "n_threads": 256,
        "n_epochs": 1000,
        "n_steps": 100,
        "background_msgs_per_step": 10,
        "checkpoint_every": 50,
    },
    "large_scale": {
        "description": "Large scale - maximum throughput (4 GPU)",
        "n_threads": 512,
        "n_epochs": 2000,
        "n_steps": 100,
        "background_msgs_per_step": 10,
        "checkpoint_every": 100,
    },
}


# ==============================================================================
# Default Paths
# ==============================================================================
DEFAULT_CHECKPOINT = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
DEFAULT_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
DEFAULT_OUTPUT_DIR = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/es_checkpoints"


# ==============================================================================
# Configuration Dataclass
# ==============================================================================
@dataclass
class ProductionConfig:
    """Production ES training configuration."""

    # Checkpoint and data paths
    lobs5_checkpoint: str = DEFAULT_CHECKPOINT
    replay_data_path: str = DEFAULT_DATA_PATH
    data_dir: str = DEFAULT_DATA_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    checkpoint_dir: str = field(default_factory=lambda: os.path.join(DEFAULT_OUTPUT_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S")))

    # ES algorithm configuration
    noiser: str = 'eggroll'          # ES algorithm: eggroll, open_es, sparse
    sigma: float = 0.01              # Perturbation noise std
    lr: float = 0.001                # Learning rate
    lora_rank: int = 4               # LoRA adapter rank
    grad_clip: float = 1.0           # Gradient clipping norm

    # Training scale
    n_threads: int = 128             # Population size (must be divisible by n_devices for multi-GPU)
    n_epochs: int = 100              # Training epochs
    n_steps: int = 100               # Steps per episode
    background_msgs_per_step: int = 5     # Background messages per step

    # Token mode (must match checkpoint)
    token_mode: int = 24             # 22 or 24

    # Background mode
    background_mode: str = 'historical_replay'  # historical_replay or world_model

    # Task configuration
    task: str = 'sell'               # sell or buy
    task_size: int = 500             # Shares to execute
    tick_size: int = 100             # Tick size in cents

    # Checkpointing
    checkpoint_every: int = 50       # Save checkpoint every N epochs
    keep_last_n_checkpoints: int = 5 # Keep last N checkpoints (0 = keep all)

    # Wandb logging
    wandb_project: Optional[str] = 'es-lobs5-production'
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_name: Optional[str] = None

    # Other
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure checkpoint_dir is set properly
        if self.checkpoint_dir == DEFAULT_OUTPUT_DIR:
            self.checkpoint_dir = os.path.join(DEFAULT_OUTPUT_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))

        # Validate n_threads for multi-GPU
        n_devices = len(jax.devices())
        if n_devices > 1:
            if self.n_threads % n_devices != 0:
                old_threads = self.n_threads
                self.n_threads = (self.n_threads // n_devices) * n_devices
                print(f"[WARN] Adjusted n_threads from {old_threads} to {self.n_threads} for {n_devices}-GPU divisibility")

    def apply_preset(self, preset_name: str):
        """Apply a preset configuration."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

        preset = PRESETS[preset_name]
        for key, value in preset.items():
            if key != "description" and hasattr(self, key):
                setattr(self, key, value)

        # Re-validate after applying preset
        self.__post_init__()


# ==============================================================================
# Memory Utilities
# ==============================================================================
def get_memory_stats() -> Dict[str, Any]:
    """Get current GPU memory statistics."""
    stats = {}
    try:
        devices = jax.devices()
        for i, device in enumerate(devices):
            try:
                mem = device.memory_stats()
                if mem:
                    stats[f"gpu_{i}"] = {
                        "used_gb": mem.get('bytes_in_use', 0) / (1024**3),
                        "peak_gb": mem.get('peak_bytes_in_use', 0) / (1024**3),
                        "limit_gb": mem.get('bytes_limit', 0) / (1024**3),
                    }
            except Exception:
                pass
    except Exception:
        pass
    return stats


def format_memory_stats(stats: Dict[str, Any]) -> str:
    """Format memory stats as a string."""
    if not stats:
        return "N/A"
    parts = []
    for gpu_id, mem in stats.items():
        used = mem.get('used_gb', 0)
        limit = mem.get('limit_gb', 0)
        pct = 100 * used / limit if limit > 0 else 0
        parts.append(f"{gpu_id}:{used:.1f}GB({pct:.0f}%)")
    return ", ".join(parts)


# ==============================================================================
# Training Progress Logger
# ==============================================================================
class TrainingLogger:
    """Unified logging to console and wandb."""

    def __init__(self, config: ProductionConfig, wandb_run=None):
        self.config = config
        self.wandb_run = wandb_run
        self.start_time = time.time()
        self.epoch_times = []
        self.best_fitness = -float('inf')
        self.all_fitnesses = []

    def log_epoch(self, epoch: int, metrics: Dict[str, float], memory_stats: Dict = None):
        """Log epoch results."""
        mean_fitness = metrics.get('fitness_mean', 0)
        self.all_fitnesses.append(mean_fitness)

        if mean_fitness > self.best_fitness:
            self.best_fitness = mean_fitness

        epoch_time = metrics.get('epoch_time', 0)
        self.epoch_times.append(epoch_time)

        # Console output
        elapsed = time.time() - self.start_time
        eta = (elapsed / (epoch + 1)) * (self.config.n_epochs - epoch - 1)
        mem_str = format_memory_stats(memory_stats) if memory_stats else ""

        print(f"[Epoch {epoch+1:4d}/{self.config.n_epochs}] "
              f"fitness={mean_fitness:+.4f} (best={self.best_fitness:+.4f}) | "
              f"std={metrics.get('fitness_std', 0):.4f} | "
              f"pnl={metrics.get('pnl', 0):+.4f} | "
              f"time={epoch_time:.1f}s | "
              f"ETA={eta/60:.1f}min | "
              f"mem=[{mem_str}]")

        # Wandb logging
        if self.wandb_run:
            try:
                log_data = {
                    'epoch': epoch,
                    'fitness/mean': mean_fitness,
                    'fitness/best': self.best_fitness,
                    'fitness/std': metrics.get('fitness_std', 0),
                    'fitness/max': metrics.get('fitness_max', 0),
                    'fitness/min': metrics.get('fitness_min', 0),
                    'pnl/mean': metrics.get('pnl', 0),
                    'execution/agent_quantity': metrics.get('agent_quantity', 0),
                    'execution/agent_trades': metrics.get('agent_trades', 0),
                    'execution/total_trades': metrics.get('total_trades', 0),
                    'time/epoch_seconds': epoch_time,
                    'time/total_minutes': elapsed / 60,
                }

                # Add memory stats
                if memory_stats:
                    for gpu_id, mem in memory_stats.items():
                        log_data[f'memory/{gpu_id}_used_gb'] = mem.get('used_gb', 0)

                self.wandb_run.log(log_data)
            except Exception as e:
                print(f"[WARN] wandb log failed: {e}")

    def log_checkpoint(self, epoch: int, path: str, save_time: float):
        """Log checkpoint save."""
        print(f"  [CHECKPOINT] Saved epoch {epoch+1} to {path} ({save_time:.2f}s)")

        if self.wandb_run:
            try:
                self.wandb_run.log({
                    'checkpoint/epoch': epoch,
                    'checkpoint/save_time': save_time,
                })
            except Exception:
                pass

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'total_epochs': len(self.all_fitnesses),
            'total_time_min': (time.time() - self.start_time) / 60,
            'best_fitness': self.best_fitness,
            'final_fitness': self.all_fitnesses[-1] if self.all_fitnesses else 0,
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
        }


# ==============================================================================
# Main Training Function
# ==============================================================================
def run_production_training(config: ProductionConfig, resume_from: Optional[str] = None):
    """
    Run production ES training.

    Args:
        config: Training configuration
        resume_from: Optional checkpoint path to resume from

    Returns:
        TrainingLogger with all metrics
    """
    from es_lobs5.training.es_trainer import ESTrainer

    # Print header
    print("=" * 80)
    print("ES-LOBS5 Production Training")
    print("=" * 80)
    print(f"JAX devices:     {jax.devices()}")
    print(f"JAX backend:     {jax.default_backend()}")
    print(f"Compilation cache: {_jax_cache_dir}")
    print("-" * 80)
    print(f"Configuration:")
    print(f"  n_threads:     {config.n_threads}")
    print(f"  n_epochs:      {config.n_epochs}")
    print(f"  n_steps:       {config.n_steps}")
    print(f"  noiser:        {config.noiser}")
    print(f"  sigma:         {config.sigma}")
    print(f"  lr:            {config.lr}")
    print(f"  lora_rank:     {config.lora_rank}")
    print(f"  grad_clip:     {config.grad_clip}")
    print(f"  token_mode:    {config.token_mode}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    if resume_from:
        print(f"  resume_from:   {resume_from}")
    print("=" * 80)

    # Create output directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.checkpoint_dir, "config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(asdict(config), f)

    # Initialize wandb
    wandb_run = None
    if WANDB_AVAILABLE and config.wandb_project:
        try:
            run_name = config.wandb_name or f"es_n{config.n_threads}_s{config.seed}"
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                config=asdict(config),
                tags=config.wandb_tags + ['production'],
                resume='allow' if resume_from else None,
            )
            print(f"[WANDB] Run URL: {wandb_run.url}")
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    # Initialize logger
    logger = TrainingLogger(config, wandb_run)

    # Initialize trainer
    print("\n[1/4] Initializing ESTrainer...")
    init_start = time.time()
    trainer = ESTrainer(config)
    print(f"  Initialization time: {time.time() - init_start:.1f}s")

    # Create initial state
    print("\n[2/4] Creating initial simulation state...")
    state_start = time.time()
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"  State creation time: {time.time() - state_start:.1f}s")
    print(f"  msg_history shape: {initial_msg_history.shape}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        print(f"\n[3/4] Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)

        state_path = os.path.join(resume_from, 'training_state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            start_epoch = state.get('epoch', 0) + 1
            logger.best_fitness = state.get('best_fitness', -float('inf'))
            logger.all_fitnesses = state.get('fitnesses', [])
            print(f"  Resuming from epoch {start_epoch}, best_fitness={logger.best_fitness:.4f}")
    else:
        print("\n[3/4] Starting fresh training...")

    # Initial memory status
    mem_stats = get_memory_stats()
    print(f"\n[4/4] Initial memory: {format_memory_stats(mem_stats)}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training Loop")
    print("=" * 80 + "\n")

    key = jax.random.PRNGKey(config.seed)

    # Fast-forward key if resuming
    for _ in range(start_epoch):
        key, _ = jax.random.split(key)

    saved_checkpoints = []

    for epoch in range(start_epoch, config.n_epochs):
        key, epoch_key = jax.random.split(key)

        # Train epoch
        epoch_start = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        epoch_time = time.time() - epoch_start

        # Collect metrics
        metrics = {
            'fitness_mean': float(mean_fitness),
            'fitness_std': float(jnp.std(fitnesses)),
            'fitness_max': float(jnp.max(fitnesses)),
            'fitness_min': float(jnp.min(fitnesses)),
            'pnl': float(info['pnl']),
            'agent_quantity': float(info['agent_quantity']),
            'agent_trades': float(info['agent_trades']),
            'total_trades': float(info['total_trades']),
            'epoch_time': epoch_time,
        }

        # Log
        mem_stats = get_memory_stats()
        logger.log_epoch(epoch, metrics, mem_stats)

        # Checkpoint
        if (epoch + 1) % config.checkpoint_every == 0 or epoch == config.n_epochs - 1:
            ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch}")
            os.makedirs(ckpt_path, exist_ok=True)

            ckpt_start = time.time()
            trainer.save_checkpoint(ckpt_path)

            # Save training state
            training_state = {
                'epoch': epoch,
                'best_fitness': logger.best_fitness,
                'fitnesses': logger.all_fitnesses.copy(),
            }
            with open(os.path.join(ckpt_path, 'training_state.pkl'), 'wb') as f:
                pickle.dump(training_state, f)

            # Also save as 'latest'
            latest_path = os.path.join(config.checkpoint_dir, "latest")
            if os.path.exists(latest_path):
                import shutil
                shutil.rmtree(latest_path)
            import shutil
            shutil.copytree(ckpt_path, latest_path)

            ckpt_time = time.time() - ckpt_start
            logger.log_checkpoint(epoch, ckpt_path, ckpt_time)

            saved_checkpoints.append(ckpt_path)

            # Cleanup old checkpoints
            if config.keep_last_n_checkpoints > 0 and len(saved_checkpoints) > config.keep_last_n_checkpoints:
                old_ckpt = saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt) and "latest" not in old_ckpt:
                    import shutil
                    shutil.rmtree(old_ckpt)
                    print(f"  [CLEANUP] Removed old checkpoint: {old_ckpt}")

        # Periodic GC
        if epoch % 20 == 0:
            gc.collect()

    # Training complete
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    summary = logger.get_summary()
    print(f"\nSummary:")
    print(f"  Total epochs:      {summary['total_epochs']}")
    print(f"  Total time:        {summary['total_time_min']:.1f} min")
    print(f"  Best fitness:      {summary['best_fitness']:.4f}")
    print(f"  Final fitness:     {summary['final_fitness']:.4f}")
    print(f"  Avg epoch time:    {summary['avg_epoch_time']:.2f}s")
    print(f"  Checkpoints saved: {len(saved_checkpoints)}")
    print(f"  Output directory:  {config.checkpoint_dir}")

    # Finalize wandb
    if wandb_run:
        try:
            wandb_run.summary.update(summary)
            wandb_run.finish()
        except Exception:
            pass

    return logger


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='ES-LOBS5 Production Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  quick_test   - Quick test (32 threads, 5 epochs, 1 GPU)
  debug        - Debug mode (64 threads, 20 epochs)
  medium       - Medium training (128 threads, 100 epochs)
  production   - Production scale (256 threads, 1000 epochs)
  large_scale  - Maximum throughput (512 threads, 2000 epochs)

Examples:
  # Quick test
  python production_train.py --preset quick_test

  # Production with custom epochs
  python production_train.py --preset production --n_epochs 500

  # Resume training
  python production_train.py --preset production --resume_from ./es_checkpoints/run_xxx/latest
        """
    )

    # Preset
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Use a preset configuration')

    # Paths
    parser.add_argument('--lobs5_checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to LOBS5 model checkpoint')
    parser.add_argument('--replay_data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to historical replay data')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for checkpoints')

    # ES configuration
    parser.add_argument('--noiser', type=str, default='eggroll',
                        choices=['noop', 'open_es', 'eggroll', 'eggrollbs', 'sparse'])
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Training scale
    parser.add_argument('--n_threads', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--background_msgs_per_step', type=int, default=5)

    # Token mode
    parser.add_argument('--token_mode', type=int, default=24, choices=[22, 24])

    # Task
    parser.add_argument('--task', type=str, default='sell', choices=['sell', 'buy'])
    parser.add_argument('--task_size', type=int, default=500)
    parser.add_argument('--tick_size', type=int, default=100)

    # Checkpointing
    parser.add_argument('--checkpoint_every', type=int, default=50)
    parser.add_argument('--keep_last_n_checkpoints', type=int, default=5)
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint directory')

    # Wandb
    parser.add_argument('--wandb_project', type=str, default='es-lobs5-production')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')

    # Other
    parser.add_argument('--seed', type=int, default=42)

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Create config
    config = ProductionConfig(
        lobs5_checkpoint=args.lobs5_checkpoint,
        replay_data_path=args.replay_data_path,
        data_dir=args.replay_data_path,
        output_dir=args.output_dir,
        noiser=args.noiser,
        sigma=args.sigma,
        lr=args.lr,
        lora_rank=args.lora_rank,
        grad_clip=args.grad_clip,
        n_threads=args.n_threads,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        background_msgs_per_step=args.background_msgs_per_step,
        token_mode=args.token_mode,
        task=args.task,
        task_size=args.task_size,
        tick_size=args.tick_size,
        checkpoint_every=args.checkpoint_every,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        seed=args.seed,
    )

    # Apply preset if specified
    if args.preset:
        config.apply_preset(args.preset)
        print(f"[INFO] Applied preset: {args.preset} - {PRESETS[args.preset]['description']}")

    # Run training
    try:
        run_production_training(config, resume_from=args.resume_from)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
