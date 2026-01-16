"""
LOBMAX Training Loop

Pure Transformer training loop for LOB prediction.
"""

import os
import sys
import time
import gc
from datetime import datetime
import subprocess

import jax
from jax import random
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp

# Add MaxText to path
MAXTEXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'maxtext', 'src')
if MAXTEXT_PATH not in sys.path:
    sys.path.insert(0, MAXTEXT_PATH)

# WandB configuration
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "True"
import wandb

from lobmax.config import LOBMAXConfig
from lobmax.models import LOBMAXModel, get_model_summary, count_parameters
from lobmax.init_train import (
    create_lobmax_config,
    create_lobmax_optimizer,
    init_lobmax_train_state,
)


def log_with_timestamp(msg, prefix="*"):
    """Print log message with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{prefix}] {msg}")


def get_git_info():
    """Get current git branch and commit hash."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return branch, commit
    except:
        return "unknown", "unknown"


def train_lobmax(args):
    """
    Main training function for LOBMAX.
    """
    best_test_loss = float('inf')
    best_test_acc = -float('inf')

    # Multi-node handling
    is_distributed = getattr(args, 'is_distributed', False)
    process_rank = getattr(args, 'process_index', 0)
    is_main_process = (process_rank == 0)

    if is_distributed:
        log_with_timestamp(f"Distributed mode: rank {process_rank}, is_main={is_main_process}")

    # Initialize WandB
    if is_main_process and args.USE_WANDB:
        run = wandb.init(
            project=args.wandb_project,
            job_type='model_training',
            config=vars(args),
            entity=args.wandb_entity,
            settings=wandb.Settings(_disable_stats=False, _disable_meta=False)
        )
    else:
        run = wandb.init(mode='disabled')

    # =========================================================================
    # Create Dataset
    # =========================================================================
    from lob.dataloading import create_lobster_prediction_dataset
    from lob.lobster_dataloader import LOBSTER_Dataset

    mask_fn = None
    if args.masking == 'causal':
        mask_fn = LOBSTER_Dataset.causal_mask
    elif args.masking == 'random':
        mask_fn = LOBSTER_Dataset.random_mask
    elif args.masking == 'last_pos':
        mask_fn = LOBSTER_Dataset.last_pos_mask
    elif args.masking == 'none':
        mask_fn = LOBSTER_Dataset.no_mask

    log_with_timestamp("Creating dataset...")

    (lobster_dataset, trainloader, valloader, testloader, aux_dataloaders,
     n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size) = \
        create_lobster_prediction_dataset(
            args.dir_name,
            seed=args.jax_seed,
            mask_fn=mask_fn,
            msg_seq_len=args.msg_seq_len,
            global_bsz=args.global_bsz,
            use_book_data=args.use_book_data,
            use_simple_book=args.use_simple_book,
            book_transform=args.book_transform,
            book_depth=args.book_depth,
            token_mode=args.token_mode,
            test_dir_name=args.test_dir_name,
            n_data_workers=args.n_data_workers,
            shuffle_train=args.shuffle_train,
            rand_offset=args.random_offsets_train,
            debug_overfit=args.debug_overfit,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            use_distributed_sampler=is_distributed,
            process_rank=process_rank,
            process_count=getattr(args, 'process_count', 1),
        )

    log_with_timestamp(f"Dataset created: {train_size} training samples")
    log_with_timestamp(f"n_classes={n_classes}, seq_len={seq_len}, book_dim={book_dim}")

    # =========================================================================
    # Initialize Model
    # =========================================================================
    log_with_timestamp("Initializing LOBMAX model...")

    from lob.sharding_utils import initialize_mesh, get_global_mesh
    mesh = get_global_mesh()
    if mesh is None:
        mesh = initialize_mesh(args.num_devices)

    state, model, total_params = init_lobmax_train_state(
        args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        train_size=train_size,
        mesh=mesh,
        print_shapes=True
    )

    log_with_timestamp(f"Model initialized: {total_params:,} parameters ({total_params/1e9:.2f}B)")

    if is_main_process and args.USE_WANDB:
        wandb.log({"total_params": total_params, "total_params_B": total_params / 1e9})
        branch, commit = get_git_info()
        wandb.run.summary["git_branch"] = branch
        wandb.run.summary["git_commit"] = commit
        wandb.run.summary["global_batch_size"] = args.global_bsz
        wandb.run.summary["num_devices"] = args.num_devices
        wandb.run.summary["backend"] = "transformer"

    # =========================================================================
    # Create JIT-compiled train_step
    # Create JIT-compiled train_step
    # =========================================================================
    from lob.train_helpers import create_jit_train_step, create_jit_eval_step

    log_with_timestamp("Creating JIT-compiled training functions...")
    # NOTE: create_jit_*_step signatures changed, batchnorm arg removed from them
    jit_train_step_fn = create_jit_train_step(
        mesh, state, has_book_data=args.use_book_data
    )
    jit_eval_step_fn = create_jit_eval_step(
        mesh, state, has_book_data=args.use_book_data
    )
    log_with_timestamp("JIT compilation ready")

    # =========================================================================
    # Checkpoint Manager
    # =========================================================================
    if is_main_process:
        run_name = wandb.run.name if args.USE_WANDB else f"lobmax_{int(time.time())}"
        ckpt_dir = os.path.join("checkpoints", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        ckpt_mgr = ocp.CheckpointManager(
            os.path.abspath(ckpt_dir),
            item_names=('state', 'metadata'),
            options=ocp.CheckpointManagerOptions(max_to_keep=3),
        )
        log_with_timestamp(f"Checkpoints will be saved to: {ckpt_dir}")

    # =========================================================================
    # Training Loop
    # =========================================================================
    from lob.train_helpers import train_epoch, validate

    log_with_timestamp(f"Starting training for {args.epochs} epochs...")
    job_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Training epoch
        state, train_loss, ce_by_tok, interrupted_at_step = train_epoch(
            state,
            random.PRNGKey(args.jax_seed + epoch),
            trainloader,
            seq_len,
            args.batchnorm,
            args.num_devices,
            args.debug_loading,
            args.enable_profiler,
            args.curtail_epochs,
            None,  # init_hidden (not used for Transformer)
            epoch,
            args.ignore_times,
            args.log_ce_tables,
            jit_train_step_fn=jit_train_step_fn,
            model_params=total_params,
            batch_size=args.global_bsz,
            peak_tflops=1000.0,
            goodput_monitor=None,
            checkpoint_callback=None,
            checkpoint_every_n_steps=0,
            job_start_time=job_start_time,
            max_job_hours=args.max_job_hours,
            save_before_timeout_minutes=args.save_before_timeout_minutes,
            mesh=mesh,  # CRITICAL: Pass mesh for proper multi-GPU sharding
            # Transformer params for MFU
            num_layers=args.n_layers,
            num_heads=args.num_heads,
            d_model=args.d_model,
        )

        epoch_time = time.time() - epoch_start
        
        # Validation
        # NOTE: validate signature: (state, apply_fn, testloader, seq_len, in_dim, batchnorm, num_devices, epoch, ...)
        val_loss, val_acc = validate(
            state,
            state.apply_fn,  # Pass apply_fn!
            valloader,
            seq_len,
            book_dim, # in_dim
            args.batchnorm,
            args.num_devices,
            epoch,
            # Kwargs
            eval_step_fn=jit_eval_step_fn,
        )

        # Logging
        if is_main_process:
            log_with_timestamp(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}, time={epoch_time:.1f}s"
            )

            if args.USE_WANDB:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time": epoch_time,
                    "step": state.step,
                })

            # Save checkpoint
            if val_loss < best_test_loss:
                best_test_loss = val_loss
                # Save best model
                from lob.init_train import save_checkpoint, deduplicate_trainstate
                ckpt = {
                    'model': deduplicate_trainstate(state),
                    'config': vars(args),
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                    }
                }
                save_checkpoint(ckpt_mgr, ckpt, epoch)
                log_with_timestamp(f"Saved best checkpoint (val_loss={val_loss:.4f})")

        # Early stopping check
        if interrupted_at_step is not None:
            log_with_timestamp(f"Training interrupted at step {interrupted_at_step}")
            break

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    if testloader is not None:
        log_with_timestamp("Running final test evaluation...")
        test_loss, test_acc = validate(
            state,
            testloader,
            seq_len,
            args.batchnorm,
            args.num_devices,
            None,
            jit_eval_step_fn,
        )
        log_with_timestamp(f"Test Results: loss={test_loss:.4f}, acc={test_acc:.4f}")

        if is_main_process and args.USE_WANDB:
            wandb.run.summary["test_loss"] = test_loss
            wandb.run.summary["test_acc"] = test_acc

    log_with_timestamp("Training complete!")

    if is_main_process:
        wandb.finish()
