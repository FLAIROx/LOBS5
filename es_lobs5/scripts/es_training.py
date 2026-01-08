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

    # Initialize wandb
    wandb_run = None
    if args.wandb_project:
        try:
            run_name = getattr(args, 'wandb_name', None) or \
                       f"es_n{args.n_perturbations}_s{args.seed}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=getattr(args, 'wandb_entity', None),
                name=run_name,
                config=vars(args),
                tags=['production', f'noiser-{args.noiser}', f'n{args.n_perturbations}'],
                resume='allow' if getattr(args, 'resume_from', None) else None,
            )
            print(f"[WANDB] Initialized: {wandb_run.url}")
        except Exception as e:
            print(f"[WANDB] Failed to initialize: {e}")
            print("[WANDB] Continuing without logging...")
            wandb.init(mode='disabled')
    else:
        print("[WANDB] Disabled (no --wandb_project specified)")
        wandb.init(mode='disabled')

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

        # Update wandb summary
        if wandb_run:
            wandb_run.summary.update({
                'total_time_hours': elapsed / 3600,
                'completed': True,
            })

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user")
        if wandb_run:
            wandb_run.summary.update({'interrupted': True})
        raise

    except Exception as e:
        print(f"\n[TRAIN] Error: {e}")
        if wandb_run:
            wandb_run.summary.update({'error': str(e)})
        raise

    finally:
        wandb.finish()
        print("[WANDB] Finished")


if __name__ == '__main__':
    main()
