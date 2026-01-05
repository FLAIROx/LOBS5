import os
import jax
from jax import random
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
# import wandb
import gc
from datetime import datetime
import subprocess

from lob.init_train import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate
from lob.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader#, Datasets
from lob.lobster_dataloader import LOBSTER_Dataset
from lob.train_helpers import reduce_lr_on_plateau, linear_warmup, \
    cosine_annealing, constant_lr, train_epoch, validate, \
    create_jit_train_step, create_jit_eval_step, initialize_mesh, get_global_mesh, \
    create_lobs5_learning_rate_schedule  # New: JIT compilation functions + LR schedule

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

    #for parameter sweep: get args from wandb server
    if args is None:
        args = wandb.config
    else:
        if args.USE_WANDB:
            # Make wandb config dictionary
            run = wandb.init(
                project=args.wandb_project,
                job_type='model_training',
                config=vars(args),
                entity=args.wandb_entity,
                settings=wandb.Settings(_disable_stats=False, _disable_meta=False)
            )
        else:
            run = wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
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

    (lobster_dataset, trainloader, valloader, testloader, aux_dataloaders,
        n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size) = \
        create_lobster_prediction_dataset(
            args.dir_name,
            seed=args.jax_seed,
            mask_fn=mask_fn,
            msg_seq_len=args.msg_seq_len,
            bsz=args.bsz,
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
            persistent_workers=args.persistent_workers
        )



    log_with_timestamp(f"Starting S5 Training on {ds} =>> Initializing...")
    if args.debug_loading:
        state=None
        val_model=None
        init_hidden=None
        total_params=None
        # Dummy schedules for debug mode
        lr_schedule = lambda step: 0.0
        ssm_lr_schedule = lambda step: 0.0
    else:
        state, model_cls, total_params = init_train_state(
            args,
            n_classes=n_classes,
            seq_len=seq_len,
            book_dim=book_dim,
            book_seq_len=book_seq_len,
            train_size=train_size,  # NEW: for schedule calculation
            print_shapes=True
        )

        # ==================================================================
        # Create LR schedules for logging (mirrors init_train.py logic)
        # These are used to compute current LR from state.step for WandB logging
        # ==================================================================
        ssm_lr = args.ssm_lr_base
        lr = args.lr_factor * ssm_lr
        steps_per_epoch = train_size // args.bsz
        if hasattr(args, 'curtail_epochs') and args.curtail_epochs is not None:
            steps_per_epoch = min(steps_per_epoch, args.curtail_epochs + 1)
        total_steps = steps_per_epoch * args.epochs
        warmup_end_step = steps_per_epoch * args.warmup_end

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
        import os
        use_bf16 = os.environ.get('USE_BF16', '1') == '1'
        log_with_timestamp(f"Training precision: {'BF16 (mixed)' if use_bf16 else 'FP32'}")
        if use_bf16:
            log_with_timestamp("BF16 Mixed Precision enabled:")
            log_with_timestamp("  - Compute: BF16")
            log_with_timestamp("  - Parameters: BF16 (except Lambda/D/log_step)")
            log_with_timestamp("  - Optimizer states: FP32")
            log_with_timestamp("  - Decoder: FP32 (for numerical stability)")

        # Log to WandB
        if args.USE_WANDB:
            wandb.log({
                "use_bf16": use_bf16,
                "precision_mode": "bf16_mixed" if use_bf16 else "fp32",
            })
            # Log git and batch size configuration to WandB
            branch, commit = get_git_info()
            wandb.run.summary["git_branch"] = branch
            wandb.run.summary["git_commit"] = commit
            wandb.run.summary["global_batch_size"] = args.bsz
            wandb.run.summary["micro_batch_size"] = args.bsz // args.num_devices
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
        init_hidden=model_cls().initialize_carry(batch_size=args.bsz//args.num_devices,
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
        jit_train_step_fn = create_jit_train_step(
            mesh,
            state,
            has_book_data=args.use_book_data
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
    steps_per_epoch = int(train_size/args.bsz) if args.curtail_epochs is None else args.curtail_epochs+1

    # Log git information and batch size configuration
    branch, commit = get_git_info()
    log_with_timestamp(f"Git Branch: {branch}")
    log_with_timestamp(f"Git Commit: {commit}")
    global_batch_size = args.bsz
    micro_batch_size = args.bsz // args.num_devices
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

    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        # LR scheduling now handled by optax schedules - no manual switching needed
        print(f"[*] Step {int(state.step)} - LR automatically managed by optax schedules")

        print('Training on', args.num_devices, 'devices.')
        train_rng, skey = random.split(train_rng)

        #Pass an initial hidden state to be used in case of the 'RNN' forward pass being used.
        state, train_loss, ce_by_tok = train_epoch(
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
            batch_size=args.bsz,
            peak_tflops=1000.0,
            goodput_monitor=goodput_monitor,
        )

        if args.random_offsets_train:
            # reinit training loader, so that sequences are initialised with
            del trainloader
            # # different offsets
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                int(random.randint(skey, (1,), 0, 100000)[0]),
                args.bsz,
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

        #save checkpoint
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

        if args.log_ce_tables:
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
