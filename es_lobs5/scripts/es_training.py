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

# =============================================================================
# PERGPU_PERTURBATIONS Scaling Test Results (2026-01-17):
# ------------------------------------------------------
# Max Stable:  14,336 (Total 57,344) - Job 1921017 - RUNNING
# First Fail:  16,384 (Total 65,536) - Job 1920937 - FAILED (OOM/Aborted)
#
# Jobs 18,432+ all fail with OOM (RESOURCE_EXHAUSTED ~48-80GB allocation)
# The "Aborted" status indicates XLA runtime forced abort to prevent deadlock
# after one replica hit OOM during distributed computation.
#
# Related files:
#   - es_lobs5/scripts/es_training.sh     (batch script)
#   - es_lobs5/training/es_trainer.py     (core logic)
#
# Reference logs:
#   - logs/es_train_1921017.out/.err  (max stable run)
#   - logs/es_train_1920937.out/.err  (first OOM failure)
#   - logs/es_train_1921018.out/.err  (detailed OOM traceback)
# =============================================================================

import os
import sys
import time

# =============================================================================
# NOTE: DO NOT set CUDA_VISIBLE_DEVICES here!
# JAX distributed requires all processes to see all GPUs for topology discovery.
# Device assignment is handled by jax.local_devices() AFTER distributed init.
#
# With --ntasks-per-node=4 and jax.distributed.initialize():
# - Each process sees all 8 GPUs globally (jax.devices())
# - But only operates on 1 local GPU (jax.local_devices())
# =============================================================================

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
    parser.add_argument('--proc_id', type=int, default=None)  # Default None, will use env var
    args, _ = parser.parse_known_args()

    # Priority 1: Check standard JAX distributed environment variables
    coord_env = os.environ.get('JAX_COORDINATOR_ADDRESS')
    if coord_env:
        import jax
        pid = int(os.environ.get('JAX_PROCESS_INDEX', os.environ.get('SLURM_PROCID', '0')))
        pcnt = int(os.environ.get('JAX_PROCESS_COUNT', os.environ.get('SLURM_NNODES', '1')))
        
        # Determine local devices (critical for 1-process-per-node mode)
        # We assume 4 GPUs per node as per standard config, or check CUDA_VISIBLE_DEVICES
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cvd:
            n_local = len([d for d in cvd.split(',') if d.strip()])
        else:
            # Fallback/Default for our nodes
            n_local = 4
            
        local_device_ids = list(range(n_local))
        
        print(f"[DIST] Initializing JAX distributed (Env): coord={coord_env}, procs={pcnt}, id={pid}")
        print(f"[DIST] Using local_device_ids={local_device_ids} ({n_local} GPUs)")
        
        jax.distributed.initialize(
            coordinator_address=coord_env,
            num_processes=pcnt,
            process_id=pid,
            local_device_ids=local_device_ids
        )
        print(f"[DIST] Process {jax.process_index()} initialized. Global devices: {jax.device_count()}, Local devices: {jax.local_device_count()}")
        return True

    # Priority 2: Legacy argument parsing (fallback)
    if args.coord_addr is not None:
        import jax
        # Get proc_id from command line or SLURM environment variable
        proc_id = args.proc_id
        if proc_id is None or proc_id < 0:
            # Use SLURM_PROCID for global process ID (0 to ntasks-1)
            # With ntasks-per-node=4, this gives unique ID per GPU across all nodes
            proc_id = int(os.environ.get('SLURM_PROCID', 0))
        print(f"[DIST] Initializing JAX distributed: coord={args.coord_addr}, procs={args.num_procs}, id={proc_id}")
        print(f"[DIST] SLURM env: NODEID={os.environ.get('SLURM_NODEID', 'N/A')}, PROCID={os.environ.get('SLURM_PROCID', 'N/A')}")
        jax.distributed.initialize(args.coord_addr, args.num_procs, proc_id)
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
