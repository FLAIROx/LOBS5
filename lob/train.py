import os
import time
import jax
from jax import random
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
# import wandb
import gc
from datetime import datetime
import subprocess

from lob.init_train import (
    init_train_state,
    init_train_state_with_prodigy,
    load_checkpoint,
    save_checkpoint,
    deduplicate_trainstate,
)
from lob.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader#, Datasets
from lob.lobster_dataloader import LOBSTER_Dataset
from lob.train_helpers import (
    reduce_lr_on_plateau, linear_warmup,
    cosine_annealing, constant_lr, train_epoch, validate,
    create_jit_train_step, create_jit_train_step_tbptt_ar_hidden,
    create_jit_eval_step, initialize_mesh, get_global_mesh,
    create_lobs5_learning_rate_schedule,
    # Prodigy LR estimation (Plan B)
    extract_prodigy_estimated_lr,
    switch_optimizer_after_prodigy_warmup,
)

# WandB configuration (must be set before wandb import)
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "True"
import wandb


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


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    # =========================================================================
    # Multi-node handling: Only rank 0 should run WandB and checkpointing
    # =========================================================================
    is_distributed = getattr(args, 'is_distributed', False)
    process_rank = getattr(args, 'process_index', 0)
    is_main_process = (process_rank == 0)

    if is_distributed:
        print(f"[Train] Distributed mode: rank {process_rank}, is_main_process={is_main_process}")

    #for parameter sweep: get args from wandb server
    if args is None:
        args = wandb.config
    else:
        # Only main process initializes WandB in online mode
        if is_main_process:
            if args.USE_WANDB:
                # Build wandb run name: d{d_model}_l{n_layers}_b{blocks}_bsz{micro}x{gpus}_seed{seed}_jid{job_id}
                slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
                micro_bsz = args.global_bsz // args.num_devices
                wandb_run_name = f"d{args.d_model}_l{args.n_layers}_b{args.blocks}_bsz{micro_bsz}x{args.num_devices}_seed{args.jax_seed}_jid{slurm_job_id}"

                # Make wandb config dictionary
                run = wandb.init(
                    name=wandb_run_name,
                    project=args.wandb_project,
                    job_type='model_training',
                    config=vars(args),
                    entity=args.wandb_entity,
                    settings=wandb.Settings(_disable_stats=False, _disable_meta=False)
                )
            else:
                run = wandb.init(mode='offline')
        else:
            # Non-main processes: use offline/disabled mode
            run = wandb.init(mode='disabled')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    if is_main_process:
        wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    ds = 'lobster-prediction'
    #create_dataset_fn =  Datasets[ds]

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    mask_fn=None
    if args.masking == 'causal':
        mask_fn = LOBSTER_Dataset.causal_mask
    elif args.masking == 'random':
        mask_fn = LOBSTER_Dataset.random_mask
    elif args.masking == 'last_pos':
         mask_fn = LOBSTER_Dataset.last_pos_mask
    elif args.masking == 'none':
         mask_fn = LOBSTER_Dataset.no_mask
    else:
        ValueError('Issue with mask function: logic for '+args.masking+' not implemented.')

    # Get distributed training parameters (set by run_train.py)
    is_distributed = getattr(args, 'is_distributed', False)
    process_rank = getattr(args, 'process_index', 0)
    process_count = getattr(args, 'process_count', 1)

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
            # Multi-node distributed training
            use_distributed_sampler=is_distributed,
            process_rank=process_rank,
            process_count=process_count,
        )



    log_with_timestamp(f"Starting S5 Training on {ds} =>> Initializing...")

    # ==================================================================
    # Prodigy LR Estimation Mode (Plan B)
    # ==================================================================
    prodigy_warmup_steps = getattr(args, 'prodigy_warmup_steps', 0)
    prodigy_lr_multiplier = getattr(args, 'prodigy_lr_multiplier', 1.0)
    prodigy_mode = prodigy_warmup_steps > 0
    prodigy_schedule_info = None  # Will be set if prodigy_mode is True

    if prodigy_mode:
        log_with_timestamp(f"[Prodigy Mode] LR estimation enabled for {prodigy_warmup_steps} steps")
        log_with_timestamp(f"[Prodigy Mode] LR multiplier: {prodigy_lr_multiplier}x")

    if args.debug_loading:
        state=None
        val_model=None
        init_hidden=None
        total_params=None
        # Dummy schedules for debug mode
        lr_schedule = lambda step: 0.0
        ssm_lr_schedule = lambda step: 0.0
    else:
        # ==================================================================
        # Initialize with Prodigy or standard optimizer
        # ==================================================================
        if prodigy_mode:
            # Phase 1: Initialize with Prodigy optimizer
            state, model_cls, total_params, prodigy_schedule_info = init_train_state_with_prodigy(
                args,
                n_classes=n_classes,
                seq_len=seq_len,
                book_dim=book_dim,
                book_seq_len=book_seq_len,
                train_size=train_size,
                print_shapes=True
            )
            # Use schedule_info for LR logging during warmup
            ssm_lr = prodigy_schedule_info['ssm_lr_base']
            lr = args.lr_factor * ssm_lr  # Will be replaced after warmup
            steps_per_epoch = prodigy_schedule_info['steps_per_epoch']
            total_steps = prodigy_schedule_info['total_steps']
            warmup_end_step = prodigy_schedule_info['warmup_end_step']
        else:
            # Standard initialization (no Prodigy)
            state, model_cls, total_params = init_train_state(
                args,
                n_classes=n_classes,
                seq_len=seq_len,
                book_dim=book_dim,
                book_seq_len=book_seq_len,
                train_size=train_size,  # NEW: for schedule calculation
                print_shapes=True
            )
            ssm_lr = args.ssm_lr_base
            lr = args.lr_factor * ssm_lr
            steps_per_epoch = train_size // args.global_bsz
            if hasattr(args, 'curtail_epochs') and args.curtail_epochs is not None:
                steps_per_epoch = min(steps_per_epoch, args.curtail_epochs + 1)
            total_steps = steps_per_epoch * args.epochs
            warmup_end_step = steps_per_epoch * args.warmup_end

        # ==================================================================
        # Create LR schedules for logging (mirrors init_train.py logic)
        # These are used to compute current LR from state.step for WandB logging
        # ==================================================================
        ssm_lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=ssm_lr,
            warmup_end_step=warmup_end_step,
            total_steps=total_steps,
            lr_min=args.lr_min,
            use_cosine_anneal=args.cosine_anneal,
        )
        lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=lr,
            warmup_end_step=warmup_end_step,
            total_steps=total_steps,
            lr_min=args.lr_min,
            use_cosine_anneal=args.cosine_anneal,
        )
        log_with_timestamp(f"LR schedules created for logging: base_lr={lr}, ssm_lr={ssm_lr}")
        # ==================================================================

        # Log BF16 status
        use_bf16 = os.environ.get('USE_BF16', '1') == '1'
        log_with_timestamp(f"Training precision: {'BF16 (mixed)' if use_bf16 else 'FP32'}")
        if use_bf16:
            log_with_timestamp("BF16 Mixed Precision enabled:")
            log_with_timestamp("  - Compute: BF16")
            log_with_timestamp("  - Parameters: BF16 (except Lambda/D/log_step)")
            log_with_timestamp("  - Optimizer states: FP32")
            log_with_timestamp("  - Decoder: FP32 (for numerical stability)")

        # Log to WandB (only main process)
        if args.USE_WANDB and is_main_process:
            wandb.log({
                "use_bf16": use_bf16,
                "precision_mode": "bf16_mixed" if use_bf16 else "fp32",
            })
            # Log git and batch size configuration to WandB
            branch, commit = get_git_info()
            wandb.run.summary["git_branch"] = branch
            wandb.run.summary["git_commit"] = commit
            wandb.run.summary["global_batch_size"] = args.global_bsz
            wandb.run.summary["micro_batch_size"] = args.global_bsz // args.num_devices
            wandb.run.summary["num_devices"] = args.num_devices

        if args.restore is not None and args.restore != '':
            print(f"[*] Restoring weights from {args.restore}")
            ckpt = load_checkpoint(
                state,
                args.restore,
                # args.__dict__,
                step=args.restore_step,
            )
            state = ckpt['model']
        
        val_model = model_cls(training=False, step_rescale=1)
        init_hidden=model_cls().initialize_carry(batch_size=args.global_bsz,
                                                hidden_size=(ssm_size // pow(2,int(args.conj_sym))),
                                                n_message_layers=args.n_message_layers,
                                                n_book_pre_layers=args.n_book_pre_layers ,
                                                n_book_post_layers=args.n_book_post_layers,
                                                n_fused_layers=args.n_layers,
                                                h_size_ema=ssm_size)

        # ====================================================================
        # New: Initialize mesh and JIT-compiled train_step (jax.jit + shardings migration)
        # ====================================================================
        log_with_timestamp("Initializing mesh and JIT-compiled functions...", prefix="Train")
        log_with_timestamp(f"Using {args.num_devices} devices for data parallelism", prefix="Train")

        # Mesh already initialized in create_train_state, get it here
        mesh = get_global_mesh()

        # Create JIT-compiled train_step
        # has_book_data parameter: set based on args.use_book_data
        # TBPTT parameters
        use_tbptt = getattr(args, 'use_tbptt', False)
        n_tbptt_chunks = getattr(args, 'n_tbptt_chunks', 4)
        tbptt_mode = getattr(args, 'tbptt_mode', 'gradient_chunking')
        tbptt_window_size = getattr(args, 'tbptt_window_size', 500)
        tbptt_reset_every_epoch = getattr(args, 'tbptt_reset_every_epoch', True)

        if use_tbptt:
            if tbptt_mode == "gradient_chunking":
                log_with_timestamp(f"TBPTT enabled: {n_tbptt_chunks} chunks", prefix="Train")
                log_with_timestamp(f"Expected memory reduction: ~{100 - 100/(n_tbptt_chunks**2):.1f}%", prefix="Train")
            elif tbptt_mode == "ar_hidden":
                if args.merging != "padded":
                    raise NotImplementedError("tbptt_mode=ar_hidden currently supports merging='padded' only")
                log_with_timestamp(f"TBPTT (AR hidden) enabled: window_size={tbptt_window_size} msgs", prefix="Train")
            else:
                raise ValueError(f"Unknown tbptt_mode: {tbptt_mode}")

        if use_tbptt and tbptt_mode == "ar_hidden":
            jit_train_step_fn = create_jit_train_step_tbptt_ar_hidden(
                mesh,
                state,
                has_book_data=args.use_book_data,
                tbptt_window_size=tbptt_window_size,
            )
        else:
            jit_train_step_fn = create_jit_train_step(
                mesh,
                state,
                has_book_data=args.use_book_data,
                use_tbptt=use_tbptt,
                n_tbptt_chunks=n_tbptt_chunks,
            )

        # Create JIT-compiled eval_step
        jit_eval_step_fn = create_jit_eval_step(
            mesh,
            state,
            has_book_data=args.use_book_data
        )

        log_with_timestamp("JIT compilation complete - ready to train!", prefix="Train")
        log_with_timestamp("Key optimizations enabled:", prefix="Train")
        log_with_timestamp("  - donate_argnums: Memory reuse for state", prefix="Train")
        log_with_timestamp(f"  - Data parallelism: {args.num_devices} devices", prefix="Train")
        # ====================================================================

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    # step variable removed - optax tracks step internally via state.step
    steps_per_epoch = int(train_size/args.global_bsz) if args.curtail_epochs is None else args.curtail_epochs+1

    # Log git information and batch size configuration
    branch, commit = get_git_info()
    log_with_timestamp(f"Git Branch: {branch}")
    log_with_timestamp(f"Git Commit: {commit}")
    global_batch_size = args.global_bsz
    micro_batch_size = args.global_bsz // args.num_devices
    log_with_timestamp(f"Global Batch Size (Gbs): {global_batch_size}")
    log_with_timestamp(f"Micro Batch Size (mbs/per_gpu_bsz): {micro_batch_size}")
    log_with_timestamp(f"Number of devices: {args.num_devices}")
    log_with_timestamp(f"Training dataset size: {train_size}")
    log_with_timestamp(f"Steps per epoch: {steps_per_epoch}")



    # Log DataLoader configuration                                                                                                                                
    log_with_timestamp("DataLoader Configuration:", prefix="Train")                                                                                               
    log_with_timestamp(f"  - num_workers: {args.n_data_workers}", prefix="Train")                                                                                 
    log_with_timestamp(f"  - pin_memory: {args.pin_memory}", prefix="Train")                                                                                      
    log_with_timestamp(f"  - prefetch_factor: {args.prefetch_factor}", prefix="Train")                                                                            
    log_with_timestamp(f"  - persistent_workers: {args.persistent_workers}", prefix="Train")                                                                      
                                                            
      
    # print("USING VERY INFREQUENT CHECKPOINTING FOR TINY EPOCH SIZE ")

    # Only main process creates checkpoint manager
    ckpt_mgr = None
    if is_main_process:
        mgr_options = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            create=True,
            max_to_keep=10,
            keep_period=5,
            # step_prefix=f'{run.name}_{run.id}',
            # enable_async_checkpointing=False,
        )
        ckpt_mgr = ocp.CheckpointManager(
            os.path.abspath(f'checkpoints/{run.name}_{run.id}/'),
            # ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
            # ocp.Checkpointer(ocp.StandardCheckpointHandler()),
            item_names=('state', 'metadata'),
            options=mgr_options,
            metadata=vars(args)
        )


    # ce_table only used by main process for logging
    ce_table = None
    if is_main_process:
        if args.ignore_times:
            # Removing the 5 abs time tokens from the length of the sequence.
            dt = [[x] for (x,) in zip([*range(seq_len-5*args.msg_seq_len)])]
        else:
            dt = [[x] for (x,) in zip([*range(seq_len)])]
        ce_table=wandb.Table(columns=["tok"] ,data=dt)

    ignore_times=args.ignore_times
    batchnorm=args.batchnorm

    # Initialize goodput monitor if enabled
    from lob.profiling_utils import GoodputMonitor
    goodput_monitor = GoodputMonitor() if args.enable_goodput_monitor else None

    # Track job start time for time-aware checkpointing
    job_start_time = time.time()

    # Handle "auto" checkpoint interval: use wall clock time
    # AUTO MODE: WANDB EVERY 10 MIN, CHECKPOINT EVERY 30 MIN
    checkpoint_every_n_steps = args.checkpoint_every_n_steps
    if checkpoint_every_n_steps == "auto":
        log_with_timestamp(f"Job started. Max duration: {args.max_job_hours}h, AUTO mode: wandb every 10min, checkpoint every 30min")
    else:
        log_with_timestamp(f"Job started. Max duration: {args.max_job_hours}h, checkpoint every {checkpoint_every_n_steps} steps")

    # =========================================================================
    # CALLBACK FOR WANDB LOGGING AND CHECKPOINT SAVING (DIFFERENT FREQUENCIES)
    # AUTO MODE:
    #   - WANDB LOSS LOGGING: EVERY 10 MINUTES
    #   - CHECKPOINT SAVING:  EVERY 30 MINUTES
    # =========================================================================
    def step_checkpoint_callback(state, epoch, step, loss, save_checkpoint_flag=True):
        """Log to wandb and optionally save checkpoint. Only main process does this."""
        # Only main process logs and saves checkpoints
        if not is_main_process:
            return

        global_step = int(state.step)

        # WANDB LOSS LOGGING (EVERY 10 MINUTES IN AUTO MODE)
        wandb.log({
            "step_loss": loss,
            "epoch": epoch + 1,
            "step_in_epoch": step + 1,
            "global_step": global_step,
        }, step=global_step)

        # CHECKPOINT SAVING (EVERY 30 MINUTES IN AUTO MODE)
        if save_checkpoint_flag and ckpt_mgr is not None:
            ckpt = {
                'model': deduplicate_trainstate(state),
                'config': vars(args),
                'metrics': {
                    'loss_train': float(loss),
                    'epoch': epoch,
                    'step': step,  # Step within epoch for resume
                }
            }
            save_checkpoint(ckpt_mgr, ckpt, global_step)
            log_with_timestamp(f"Checkpoint SAVED: epoch={epoch+1}, step={step+1}, global_step={global_step}, loss={loss:.4f}")
        else:
            log_with_timestamp(f"WandB logged: epoch={epoch+1}, step={step+1}, global_step={global_step}, loss={loss:.4f}")

    # Track resume state
    resume_from_step = getattr(args, 'resume_from_step', None)

    # Track Prodigy optimizer switch status
    prodigy_switched = False

    # Hidden carry for AR TBPTT
    tbptt_hidden = init_hidden if (use_tbptt and tbptt_mode == "ar_hidden") else None

    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        # LR scheduling now handled by optax schedules - no manual switching needed
        print(f"[*] Step {int(state.step)} - LR automatically managed by optax schedules")

        print('Training on', args.num_devices, 'devices.')
        train_rng, skey = random.split(train_rng)

        #Pass an initial hidden state to be used in case of the 'RNN' forward pass being used.
        state, train_loss, ce_by_tok, interrupted_at_step, tbptt_hidden_out = train_epoch(
            state,
            skey,
            trainloader,
            seq_len,
            batchnorm,
            # lr_params REMOVED - optax schedules handle LR
            args.num_devices,
            args.debug_loading,
            args.enable_profiler,
            args.curtail_epochs,
            init_hidden,
            epoch,
            ignore_times,
            args.log_ce_tables,
            jit_train_step_fn=jit_train_step_fn,
            # MFU tracking parameters
            model_params=total_params,
            batch_size=args.global_bsz,
            peak_tflops=1000.0,
            goodput_monitor=goodput_monitor,
            # Step-level checkpointing parameters
            checkpoint_callback=step_checkpoint_callback,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            job_start_time=job_start_time,
            max_job_hours=args.max_job_hours,
            save_before_timeout_minutes=args.save_before_timeout_minutes,
            # TBPTT parameters
            use_tbptt=use_tbptt,
            n_tbptt_chunks=n_tbptt_chunks,
            tbptt_mode=tbptt_mode,
            tbptt_window_size=tbptt_window_size,
            tbptt_hidden_in=tbptt_hidden,
            tbptt_reset_every_epoch=tbptt_reset_every_epoch,
        )

        # Note: tbptt_hidden is NOT carried across epochs
        # Each epoch starts with zero-initialized hidden state
        # This ensures samples from different time periods don't share history
        # (Removed cross-epoch hidden carry as it violates temporal independence)

        # Check if epoch was interrupted due to timeout
        if interrupted_at_step is not None:
            log_with_timestamp(f"Epoch {epoch+1} interrupted at step {interrupted_at_step} due to timeout")
            log_with_timestamp(f"To resume, use: --restore <checkpoint_path> --resume_from_step {interrupted_at_step}")
            break  # Exit training loop

        # ==================================================================
        # Prodigy Optimizer Switch Check (Plan B)
        # ==================================================================
        # After warmup steps, switch from Prodigy to AdamW + cosine annealing
        if prodigy_mode and not prodigy_switched and int(state.step) >= prodigy_warmup_steps:
            log_with_timestamp(f"[Prodigy] Reached {prodigy_warmup_steps} warmup steps, switching optimizer...")

            # Extract estimated LR from Prodigy
            estimated_lr = extract_prodigy_estimated_lr(state, lr_multiplier=prodigy_lr_multiplier)

            # Log to WandB (only main process)
            if args.USE_WANDB and is_main_process:
                wandb.log({
                    "prodigy_estimated_lr": estimated_lr,
                    "prodigy_switch_step": int(state.step),
                })
                wandb.run.summary["prodigy_estimated_lr"] = estimated_lr

            # Switch optimizer
            mesh = get_global_mesh()
            state = switch_optimizer_after_prodigy_warmup(
                state=state,
                estimated_lr=estimated_lr,
                ssm_lr_base=prodigy_schedule_info['ssm_lr_base'],
                warmup_end_step=prodigy_schedule_info['warmup_end_step'],
                total_steps=prodigy_schedule_info['total_steps'],
                lr_min=prodigy_schedule_info['lr_min'],
                use_cosine_anneal=prodigy_schedule_info['use_cosine_anneal'],
                weight_decay=prodigy_schedule_info['weight_decay'],
                opt_config=prodigy_schedule_info['opt_config'],
                dt_global=prodigy_schedule_info['dt_global'],
                mesh=mesh,
            )

            # Update LR schedules for logging
            lr = estimated_lr
            lr_schedule = create_lobs5_learning_rate_schedule(
                base_lr=estimated_lr,
                warmup_end_step=max(prodigy_schedule_info['warmup_end_step'], int(state.step)),
                total_steps=prodigy_schedule_info['total_steps'],
                lr_min=prodigy_schedule_info['lr_min'],
                use_cosine_anneal=prodigy_schedule_info['use_cosine_anneal'],
            )

            # Recreate JIT-compiled train_step with new optimizer
            log_with_timestamp("[Prodigy] Recreating JIT-compiled train_step...")
            jit_train_step_fn = create_jit_train_step(
                mesh,
                state,
                has_book_data=args.use_book_data
            )
            jit_eval_step_fn = create_jit_eval_step(
                mesh,
                state,
                has_book_data=args.use_book_data
            )

            prodigy_switched = True
            log_with_timestamp(f"[Prodigy] Switch complete! New base LR: {estimated_lr:.6f}")
        # ==================================================================

        if args.random_offsets_train:
            # reinit training loader, so that sequences are initialised with
            del trainloader
            # # different offsets
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                int(random.randint(skey, (1,), 0, 100000)[0]),
                args.global_bsz,
                num_workers=args.n_data_workers,
                reset_train_offsets=args.random_offsets_train,
                shuffle=args.shuffle_train,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers)
        print(f"val model hash: {val_model.__hash__()}")
        print(f"val model apply hash: {val_model.__hash__()}")

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation ") #on train set (With call)...
            (val_loss,
              val_acc,
                val_ce_means,
                val_acc_means) = validate(state,
                                        #model_cls,
                                        val_model.apply,
                                        valloader,
                                        seq_len,
                                        in_dim,
                                        batchnorm,
                                        args.num_devices,
                                        epoch,
                                        curtail_epoch=args.curtail_epochs,
                                        apply_method='__call_ar__',
                                        ignore_times=ignore_times,
                                        log_ce_tables=args.log_ce_tables,
                                        eval_step_fn=jit_eval_step_fn)

            # Print goodput statistics if enabled
            if goodput_monitor:
                print("\n[Goodput Monitor] Epoch statistics:")
                goodput_monitor.print_summary()
                print()

            print(f"[*] Running Epoch {epoch + 1} Test ") #on train set (With Call RNN)...
            (test_loss, test_acc,
              test_ce_means,test_acc_means) = validate(state,
                                           #model_cls,
                                           val_model.apply,
                                           testloader,
                                           seq_len,
                                           in_dim,
                                           batchnorm,
                                           args.num_devices,
                                           epoch,
                                           curtail_epoch=args.curtail_epochs,
                                           apply_method='__call_ar__',
                                           ignore_times=ignore_times,
                                           log_ce_tables=args.log_ce_tables,
                                           eval_step_fn=jit_eval_step_fn)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss (AR): {val_loss:.5f} --Test Loss (RNN): {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            # print("Testing on train data (diff offset) for debugging purposes")
            (test_loss, test_acc,
              test_ce_means,test_acc_means) = validate(state,
                                         #model_cls,
                                         val_model.apply,
                                         valloader,
                                         seq_len,
                                         in_dim,
                                         batchnorm,
                                         args.num_devices,
                                         epoch,
                                         curtail_epoch=args.curtail_epochs,
                                         ignore_times=ignore_times,
                                         log_ce_tables=args.log_ce_tables,
                                         eval_step_fn=jit_eval_step_fn)
            val_loss=test_loss
            val_acc=test_acc

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        #save checkpoint (only main process)
        if is_main_process and ckpt_mgr is not None:
            ckpt = {
                'model': deduplicate_trainstate(state),
                'config': vars(args),
                'metrics': {
                    'loss_train': float(train_loss),
                    'loss_val_ar': float(val_loss),
                    'loss_test_rnn': float(test_loss),
                    'acc_val_ar': float(val_acc),
                    'acc_test_rnn': float(test_acc),
                }
            }
            save_checkpoint(ckpt_mgr, ckpt, epoch)

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1



        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if args.log_ce_tables and is_main_process:
            ce_table.add_column(name="val_ce_"+str(epoch),data=val_ce_means.tolist())
            ce_table.add_column(name="test_ce_"+str(epoch),data=test_ce_means.tolist())
            ce_table.add_column(name="val_acc_"+str(epoch),data=val_acc_means.tolist())
            ce_table.add_column(name="test_acc_"+str(epoch),data=test_acc_means.tolist())
            ce_table.add_column(name="train_ce_"+str(epoch),data=ce_by_tok.tolist())
            ce_table=wandb.Table(columns=ce_table.columns,data=ce_table.data)


        # Compute learning rate from schedule using state.step
        # With optax schedules (MaxText way), LR is not stored in hyperparams but
        # computed on-the-fly from the schedule functions
        current_lr = lr_schedule(int(state.step))
        current_ssm_lr = ssm_lr_schedule(int(state.step))

        # Only main process logs to WandB
        if is_main_process:
            if valloader is not None:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": float(current_lr),
                        "ssm_lr": float(current_ssm_lr),
                        # "Training CE by token":ce_table
                    }
                )
            else:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": float(current_lr),
                        "ssm_lr": float(current_ssm_lr),
                        # "Training CE by token":ce_table
                    }
                )

            if args.log_ce_tables:
                wandb.log({"CE by token": ce_table})
            wandb.run.summary["Best Val Loss"] = best_loss
            wandb.run.summary["Best Val Accuracy"] = best_acc
            wandb.run.summary["Best Epoch"] = best_epoch
            wandb.run.summary["Best Test Loss"] = best_test_loss
            wandb.run.summary["Best Test Accuracy"] = best_test_acc
        # print("IGNORING EARLY STOPPING FOR TINY EPOCH SIZE ")
        # After each epoch
        gc.collect()
        # jax.clear_backends()
        jax.clear_caches()
        # jax.profiler.stop_trace()
        if count > args.early_stop_patience:
            break
