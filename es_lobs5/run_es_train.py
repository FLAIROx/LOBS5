#!/usr/bin/env python
"""
ES Training Entry Point for LOBS5.

Usage:
    python -m es_lobs5.run_es_train --lobs5_checkpoint=... --noiser=eggroll ...

    # With historical replay:
    python -m es_lobs5.run_es_train \
        --lobs5_checkpoint=checkpoints/xxx \
        --background_mode=historical_replay \
        --replay_data_path=/path/to/data \
        --token_mode=22
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_parser():
    parser = argparse.ArgumentParser(description='ES Training for LOBS5')

    # Checkpoint
    parser.add_argument('--lobs5_checkpoint', type=str, required=True,
                        help='Path to LOBS5 checkpoint')

    # ES configuration
    parser.add_argument('--noiser', type=str, default='eggroll',
                        choices=['noop', 'open_es', 'eggroll', 'eggrollbs', 'sparse'])
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lora_rank', type=int, default=4)

    # Training configuration
    parser.add_argument('--n_threads', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--world_msgs_per_step', type=int, default=10)

    # Token mode
    parser.add_argument('--token_mode', type=int, default=22, choices=[22, 24])

    # Background mode
    parser.add_argument('--background_mode', type=str, default='world_model',
                        choices=['world_model', 'historical_replay'])
    parser.add_argument('--replay_data_path', type=str, default=None)

    # Task
    parser.add_argument('--task', type=str, default='sell', choices=['sell', 'buy'])
    parser.add_argument('--task_size', type=int, default=500)
    parser.add_argument('--tick_size', type=int, default=100)

    # Training stability
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./es_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint directory')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./es_checkpoints')

    # W&B
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("LOBS5 ES Training")
    print("=" * 60)
    print(f"Checkpoint: {args.lobs5_checkpoint}")
    print(f"Noiser: {args.noiser}")
    print(f"Token mode: {args.token_mode}")
    print(f"Background mode: {args.background_mode}")
    print(f"Threads: {args.n_threads}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Grad clip: {args.grad_clip}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
    print("=" * 60)

    # Validate
    if args.background_mode == 'historical_replay' and args.replay_data_path is None:
        parser.error("--replay_data_path required when background_mode=historical_replay")

    # Import trainer
    from es_lobs5.training.es_trainer import ESTrainer

    # Create and run trainer
    trainer = ESTrainer(args)
    trainer.train(resume_from=args.resume_from)

    print("=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
