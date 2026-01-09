#!/usr/bin/env python3
"""
ES Training Entry Point for LOBS5

Production-ready ES training script with wandb logging.

Usage:
    python es_training.py \
        --lobs5_checkpoint /path/to/checkpoint \
        --replay_data_path /path/to/data \
        --n_epochs 1000 \
        --wandb_project es-lobs5-production

Environment variables (for SLURM jobs):
    CHECKPOINT, DATA_DIR, N_EPOCHS, N_PERTURBATIONS, etc.
    See es_training.sh for full list.
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# =============================================================================
# CRITICAL: Multi-node distributed initialization MUST happen before any JAX
# imports that might initialize the XLA backend. This includes es_trainer.py
# which has jax.config.update() at module level.
# =============================================================================
def _init_distributed_if_needed():
    """Initialize JAX distributed if --coord_addr is provided."""
    # Quick check for distributed args without full argparse
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--coord_addr', type=str, default=None)
    parser.add_argument('--num_procs', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)
    args, _ = parser.parse_known_args()

    if args.coord_addr is not None:
        import jax
        print(f"[DIST] Initializing JAX distributed: coord={args.coord_addr}, procs={args.num_procs}, id={args.proc_id}")
        jax.distributed.initialize(args.coord_addr, args.num_procs, args.proc_id)
        print(f"[DIST] Process {jax.process_index()} of {jax.process_count()} initialized with {len(jax.devices())} devices")
        return True
    return False

_is_distributed = _init_distributed_if_needed()
# =============================================================================

# WandB configuration (must be set before wandb import)
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "True"
import wandb
from es_lobs5.training.es_trainer import ESTrainer, create_es_config


def main():
    start_time = time.time()

    # Parse arguments using ESTrainer's config parser
    parser = create_es_config()
    args = parser.parse_args()

    print("=" * 60)
    print(" ES Training - LOBS5")
    print("=" * 60)
    print(f"Checkpoint: {args.lobs5_checkpoint}")
    print(f"Data: {getattr(args, 'replay_data_path', 'N/A')}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Perturbations: {args.n_perturbations}")
    print(f"Steps/Episode: {args.n_steps}")
    print(f"Warmup msgs: {getattr(args, 'n_warmup_msgs', 500)}")
    print(f"BG msgs/step: {getattr(args, 'background_msgs_per_step', 10)}")
    file_idx = getattr(args, 'file_idx', None)
    print(f"File idx: {file_idx if file_idx is not None else 'random'}")
    print(f"Noiser: {args.noiser}")
    print(f"Sigma: {args.sigma}, LR: {args.lr}")
    print(f"Wandb: {args.wandb_project or 'disabled'}")
    print("=" * 60)

    # NOTE: wandb is initialized inside ESTrainer.train() to avoid duplicate init
    # The SSL config above ensures it works in HPC environments

    try:
        # Create trainer
        print("\n[TRAIN] Initializing ESTrainer...")
        trainer = ESTrainer(args)

        # Run training
        resume_from = getattr(args, 'resume_from', None)
        if resume_from:
            print(f"[TRAIN] Resuming from: {resume_from}")

        print(f"\n[TRAIN] Starting training for {args.n_epochs} epochs...")
        trainer.train(
            n_epochs=args.n_epochs,
            resume_from=resume_from,
        )

        elapsed = time.time() - start_time
        print(f"\n[TRAIN] Training completed in {elapsed/3600:.2f} hours")

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user")
        raise

    except Exception as e:
        print(f"\n[TRAIN] Error: {e}")
        raise


if __name__ == '__main__':
    main()
