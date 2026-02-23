import os
import sys
import time
import jax
from jax import random
from jax.experimental.multihost_utils import sync_global_devices
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
import wandb
import gc

from lob.init_train import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate
from lob.dataloading import create_lobster_prediction_dataset#, Datasets
from lob.lobster_dataloader import LOBSTER_Dataset
from lob.train_helpers import reduce_lr_on_plateau, train_epoch, validate, \
    create_jit_train_step, create_jit_eval_step, create_lobs5_learning_rate_schedule, \
    StepWatchdog, TIME_START_I, TIME_END_I
from lob.encoding import Message_Tokenizer
from lob.sharding_utils import initialize_mesh, create_state_shardings




def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0
    best_test_last_order_loss = 100000000
    best_test_last_order_acc = -10000.0
    best_test_last_order_ppl = 100000000

    #for parameter sweep: get args from wandb server
    is_main_process = getattr(args, 'process_index', 0) == 0
    if args is None:
        args = wandb.config
    else:
        if args.USE_WANDB and is_main_process:
            # Rank 0: online sync to wandb cloud
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
            wandb_name = f"j{slurm_job_id}" if slurm_job_id else None
            run = wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity, name=wandb_name)
        elif args.USE_WANDB:
            # Non-rank-0: local logging only, no duplicate cloud runs
            run = wandb.init(mode='offline')
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
            micro_bsz=args.micro_bsz,
            num_devices=args.num_devices,
            use_book_data=args.use_book_data,
            use_simple_book=args.use_simple_book,
            book_transform=args.book_transform,
            n_data_workers=args.n_data_workers,
            shuffle_train=args.shuffle_train,
            rand_offset=args.random_offsets_train,
            debug_overfit=args.debug_overfit,
            test_dir_name=getattr(args, 'test_dir_name', None),
            use_distributed_sampler=is_distributed,
            process_rank=process_rank,
            process_count=process_count,
        )

    

    print(f"[*] Starting S5 Training on {ds} =>> Initializing...")
    if args.debug_loading:
        state=None
        val_model=None
        init_hidden=None
    else:
        state, model_cls = init_train_state(
            args,
            n_classes=n_classes,
            seq_len=seq_len,
            book_dim=book_dim,
            book_seq_len=book_seq_len,
            train_size=train_size,
            print_shapes=True
        )

        # Initialize mesh first (needed for restore and JIT step functions)
        # Multi-node: global mesh over ALL devices for cross-node gradient sync
        # Single-node: local mesh over num_devices GPUs
        use_hierarchical = getattr(args, 'hierarchical', False)
        if jax.process_count() > 1:
            mesh = initialize_mesh(jax.device_count(), hierarchical=use_hierarchical)
        else:
            mesh = initialize_mesh(args.num_devices)

        restored_metrics = {}
        if args.restore is not None and args.restore != '':
            print(f"[*] Restoring weights from {args.restore}")
            ckpt = load_checkpoint(
                state,
                args.restore,
                # args.__dict__,
                step=args.restore_step,
                mesh=mesh,
                partial_restore=getattr(args, 'partial_restore', False),
            )
            state = ckpt['model']
            # Debug: verify restored state
            print(f"[Restore] state.step = {int(state.step)}")
            print(f"[Restore] Restored metrics: {ckpt.get('metrics', {})}")
            # Check optimizer momentum is non-zero (proves Adam state restored)
            ssm_inner = state.opt_state.inner_states['ssm'].inner_state
            adam_state = ssm_inner[0]  # ScaleByAdamState (optax schedule mode)
            mu_leaves = jax.tree_util.tree_leaves(adam_state.mu)
            nu_leaves = jax.tree_util.tree_leaves(adam_state.nu)
            mu_norms = [float(jnp.linalg.norm(m)) for m in mu_leaves[:3]]
            nu_norms = [float(jnp.linalg.norm(n)) for n in nu_leaves[:3]]
            print(f"[Restore] Adam mu norms (first 3 params): {mu_norms}")
            print(f"[Restore] Adam nu norms (first 3 params): {nu_norms}")
            schedule_count = ssm_inner[1].count  # ScaleByScheduleState
            print(f"[Restore] Schedule count = {int(schedule_count)}")
            restored_metrics = ckpt.get('metrics', {})

        val_model = model_cls(training=False, step_rescale=1)
        init_hidden=model_cls().initialize_carry(batch_size=args.micro_bsz,
                                                hidden_size=(ssm_size // pow(2,int(args.conj_sym))),
                                                n_message_layers=args.n_message_layers,
                                                n_book_pre_layers=args.n_book_pre_layers ,
                                                n_book_post_layers=args.n_book_post_layers,
                                                n_fused_layers=args.n_layers,
                                                h_size_ema=ssm_size)

        # Move state to host numpy (device-agnostic) then shard to global mesh.
        # This handles both init (jax array on local device) and restore (numpy from checkpoint).
        state = jax.device_get(state)
        state_shardings = create_state_shardings(state, mesh)
        state = jax.device_put(state, state_shardings)
        total_devices = jax.device_count() if jax.process_count() > 1 else args.num_devices
        print(f"[*] State distributed via sharding (replicated across {total_devices} devices)")

        local_sgd_k = getattr(args, 'local_sgd_k', 0)
        if local_sgd_k > 0:
            print(f"[*] Local SGD enabled: sync params every {local_sgd_k} steps "
                  f"(effective BSZ/step = {args.micro_bsz * args.num_devices})")

        jit_train_step = create_jit_train_step(
            mesh, state, has_book_data=args.use_book_data,
            hierarchical=use_hierarchical,
            batchnorm=args.batchnorm, ignore_times=args.ignore_times,
            local_sgd_k=local_sgd_k)
        jit_eval_step = create_jit_eval_step(mesh, state, has_book_data=args.use_book_data)

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = int(state.step)  # for per step learning rate decay (restored from checkpoint or 0)

    # Restore best metrics from checkpoint if available
    if restored_metrics:
        best_loss = restored_metrics.get('loss_val_ar', best_loss)
        best_acc = restored_metrics.get('acc_val_ar', best_acc)
        best_val_loss = restored_metrics.get('loss_val_ar', best_val_loss)
        best_test_loss = restored_metrics.get('loss_test_rnn', best_test_loss)
        best_test_acc = restored_metrics.get('acc_test_rnn', best_test_acc)
        print(f"[Restore] Best metrics restored: val_loss={best_loss:.5f}, val_acc={best_acc:.4f}, "
              f"test_loss={best_test_loss:.5f}, test_acc={best_test_acc:.4f}")
    steps_per_epoch = int(train_size / (args.micro_bsz * args.num_devices * process_count)) if args.curtail_epochs is None else args.curtail_epochs+1

    # Create LR schedule functions for wandb logging (optax manages LR inside JIT)
    total_steps = steps_per_epoch * args.epochs
    warmup_end_step = steps_per_epoch * args.warmup_end
    lr_schedule_fn = create_lobs5_learning_rate_schedule(
        base_lr=lr, warmup_end_step=warmup_end_step,
        total_steps=total_steps, lr_min=args.lr_min,
        use_cosine_anneal=args.cosine_anneal)
    ssm_lr_schedule_fn = create_lobs5_learning_rate_schedule(
        base_lr=ssm_lr, warmup_end_step=warmup_end_step,
        total_steps=total_steps, lr_min=args.lr_min,
        use_cosine_anneal=args.cosine_anneal)

    # print("USING VERY INFREQUENT CHECKPOINTING FOR TINY EPOCH SIZE ")

    # Global mesh: ALL ranks must create CheckpointManager so Orbax barriers work.
    # Use SLURM_JOB_ID for consistent path across ranks (wandb run names differ per rank).
    # Orbax primary_host=0 ensures only rank 0 writes; others just participate in barriers.
    slurm_jid = os.environ.get("SLURM_JOB_ID", "local")
    ckpt_dir = os.path.abspath(f'checkpoints/{run.name}_{run.id}_{slurm_jid}/') if is_main_process else \
               os.path.abspath(f'checkpoints/job_{slurm_jid}/')
    if process_count > 1:
        # Multi-node: broadcast rank 0's checkpoint dir to all ranks
        if is_main_process:
            # Encode path as fixed-length byte array
            path_bytes = ckpt_dir.encode('utf-8')
            path_arr = jnp.array(list(path_bytes) + [0] * (256 - len(path_bytes)), dtype=jnp.uint8)
        else:
            path_arr = jnp.zeros(256, dtype=jnp.uint8)
        path_arr = jax.make_array_from_process_local_data(
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), path_arr
        )
        path_arr = jax.experimental.multihost_utils.broadcast_one_to_all(path_arr)
        path_bytes = bytes(jax.device_get(path_arr).tolist()).rstrip(b'\x00')
        ckpt_dir = path_bytes.decode('utf-8')
        print(f"[Rank {process_rank}] Checkpoint dir: {ckpt_dir}")

    mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        create=True,
        max_to_keep=10,
        keep_period=5,
        # Disable async: forked subprocesses corrupt NCCL after cuInit in child
        enable_async_checkpointing=False,
    )
    ckpt_mgr = ocp.CheckpointManager(
        ckpt_dir,
        item_names=('state', 'metadata'),
        options=mgr_options,
        metadata=vars(args) if is_main_process else {}
    )


    if args.ignore_times:
        # Removing the 5 abs time tokens from the length of the sequence.  
        dt = [[x] for (x,) in zip([*range(seq_len-5*args.msg_seq_len)])]
    else:
        dt = [[x] for (x,) in zip([*range(seq_len)])]
    ce_table=wandb.Table(columns=["tok"] ,data=dt)

    ignore_times=args.ignore_times
    batchnorm=args.batchnorm
    tokens_per_msg = (Message_Tokenizer.MSG_LEN - (TIME_END_I - TIME_START_I + 1)
                      if ignore_times else Message_Tokenizer.MSG_LEN)

    start_epoch = 0
    if args.restore is not None and args.restore != '':
        # Always infer from state.step (works for both epoch-end and mid-epoch checkpoints)
        start_epoch = int(state.step) // max(steps_per_epoch, 1)
        print(f"[Restore] Resuming training from epoch {start_epoch} (of {args.epochs})")

    # Mid-epoch checkpoint: callback + resume state
    job_start_time = time.monotonic()
    resume_from_step = getattr(args, 'resume_from_step', None)

    # Pre-compiled reshard function for checkpoints (reuses single JIT cache entry)
    if is_distributed:
        _reshard_for_ckpt = jax.jit(lambda s: s, out_shardings=state_shardings)

    def step_checkpoint_callback(cb_state, cb_epoch, cb_batch_idx, cb_loss, save_flag=True):
        """Mid-epoch: log to wandb and optionally save checkpoint."""
        global_step = int(cb_state.step)
        if is_main_process and args.USE_WANDB:
            wandb.log({
                "step_loss": float(cb_loss),
                "epoch": cb_epoch + 1,
                "step_in_epoch": cb_batch_idx + 1,
                "global_step": global_step,
            }, step=global_step)
        if save_flag:
            if is_distributed:
                ckpt_st = _reshard_for_ckpt(cb_state)
            else:
                ckpt_st = deduplicate_trainstate(cb_state)
            ckpt = {
                'model': ckpt_st,
                'config': vars(args) if is_main_process else {},
                'metrics': {
                    'loss_train': float(cb_loss),
                    'epoch': cb_epoch,
                    'step_in_epoch': cb_batch_idx,
                }
            }
            try:
                save_checkpoint(ckpt_mgr, ckpt, global_step)
                if is_main_process:
                    print(f"[Checkpoint] Mid-epoch save: epoch={cb_epoch}, "
                          f"step={cb_batch_idx}, global_step={global_step}")
            except (OSError, ValueError) as e:
                print(f"[Checkpoint] WARNING: mid-epoch save failed: {e}")

    for epoch in range(start_epoch, args.epochs):
        # Free residual memory from previous epoch's val/test before training
        gc.collect()

        # Update DistributedSampler epoch for proper cross-epoch shuffling
        if hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)

        print(f"[*] Starting Training Epoch {epoch + 1}...")
        print(f"[*] Step {step} - LR managed by optax schedules")
        print('Training on', args.num_devices, 'devices.')
        train_rng, skey = random.split(train_rng)

        #Pass an initial hidden state to be used in case of the 'RNN' forward pass being used.
        state, train_loss, ce_by_tok, interrupted_at_step = train_epoch(state,
                                              skey,
                                              trainloader,
                                              seq_len,
                                              batchnorm,
                                              None,  # lr_params=None → optax schedules
                                              args.num_devices,
                                              args.debug_loading,
                                              args.enable_profiler,
                                              args.curtail_epochs,
                                              init_hidden,
                                              epoch,
                                              ignore_times,
                                              args.log_ce_tables,
                                              mesh=mesh,
                                              jit_train_step_fn=jit_train_step,
                                              checkpoint_callback=step_checkpoint_callback,
                                              checkpoint_every_n_steps=getattr(args, 'checkpoint_every_n_steps', 'auto'),
                                              job_start_time=job_start_time,
                                              max_job_hours=getattr(args, 'max_job_hours', 24.0),
                                              save_before_timeout_minutes=getattr(args, 'save_before_timeout_minutes', 30),
                                              resume_from_step=resume_from_step,
                                              )
        # resume_from_step only applies to the first epoch after restore
        resume_from_step = None
        step = int(state.step)

        # Handle timeout interrupt: skip validation + epoch-end checkpoint, exit
        if interrupted_at_step is not None:
            if is_main_process:
                print(f"[Train] Epoch {epoch+1} interrupted at step {interrupted_at_step} due to timeout")
                print(f"[Train] To resume: RESTORE_PATH={ckpt_dir} RESTORE_STEP={step} "
                      f"RESUME_FROM_STEP={interrupted_at_step}")
            break

        if args.random_offsets_train:
            # Refresh random offsets in-place without rebuilding DataLoader.
            # This keeps persistent workers alive across epochs.
            lobster_dataset.reset_train_offsets()
        print(f"val model hash: {val_model.__hash__()}")
        print(f"val model apply hash: {val_model.__hash__()}")

        # Barrier: sync all ranks before eval to prevent NCCL deadlock.
        # Without this, rank timing differences from GC/checkpoint I/O can cause
        # one rank to enter eval's NCCL collective while others are still delayed
        # → timeout → deadlock (observed in jobs 2438014, 2438369 at eval step 0).
        # Standard practice in JAX ecosystem (Orbax uses sync_global_processes).
        if is_distributed:
            sync_global_devices(f"pre_eval_epoch_{epoch}")

        if valloader is not None:
            # Eval watchdog: kill process if val/test eval hangs (e.g. NCCL deadlock).
            # 1200s (20min) timeout: 600s was too aggressive at 32N scale (job 2438369).
            eval_watchdog = StepWatchdog(timeout=1200)
            eval_watchdog.kick(epoch, 0)

            print(f"[*] Running Epoch {epoch + 1} Validation ") #on train set (With call)...
            (val_loss, val_acc,
                val_ce_means, val_acc_means,
                val_last_order_loss, val_last_order_acc,
                val_last_order_ppl) = validate(state,
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
                                        mesh=mesh,
                                        jit_eval_step_fn=jit_eval_step)

            eval_watchdog.kick(epoch, 1)  # reset timer before test eval
            print(f"[*] Running Epoch {epoch + 1} Test ")
            (test_loss, test_acc,
              test_ce_means, test_acc_means,
              test_last_order_loss, test_last_order_acc,
              test_last_order_ppl) = validate(state,
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
                                           mesh=mesh,
                                           jit_eval_step_fn=jit_eval_step)

            eval_watchdog.stop()
            val_log_msg_ppl = val_loss * tokens_per_msg
            test_log_msg_ppl = test_loss * tokens_per_msg

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss (AR): {val_loss:.5f} --Test Loss (RNN): {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
                f" Val Log Msg PPL: {val_log_msg_ppl:.4f}"
                f" Test Log Msg PPL: {test_log_msg_ppl:.4f}"
            )
            print(
                f"\tLast Order -- Test Loss: {test_last_order_loss:.5f}"
                f" Test Acc: {test_last_order_acc:.4f}"
                f" Test PPL: {test_last_order_ppl:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            eval_watchdog = StepWatchdog(timeout=1200)
            eval_watchdog.kick(epoch, 0)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            # print("Testing on train data (diff offset) for debugging purposes")
            (test_loss, test_acc,
              test_ce_means, test_acc_means,
              test_last_order_loss, test_last_order_acc,
              test_last_order_ppl) = validate(state,
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
                                         mesh=mesh,
                                         jit_eval_step_fn=jit_eval_step)
            eval_watchdog.stop()
            val_loss=test_loss
            val_acc=test_acc
            val_last_order_loss = test_last_order_loss
            val_last_order_acc = test_last_order_acc
            val_last_order_ppl = test_last_order_ppl
            test_log_msg_ppl = test_loss * tokens_per_msg
            val_log_msg_ppl = test_log_msg_ppl

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
                f" Test Log Msg PPL: {test_log_msg_ppl:.4f}"
            )
            print(
                f"\tLast Order -- Test Loss: {test_last_order_loss:.5f}"
                f" Test Acc: {test_last_order_acc:.4f}"
                f" Test PPL: {test_last_order_ppl:.4f}"
            )

        # Save checkpoint — ALL ranks must call save() for Orbax barrier sync.
        # Orbax primary_host=0 ensures only rank 0 writes to disk.
        # Multi-host: re-shard state to ensure consistent NamedSharding for Orbax.
        # Single-host: deduplicate to single device first.
        if is_distributed:
            ckpt_state = _reshard_for_ckpt(state)
        else:
            ckpt_state = deduplicate_trainstate(state)
        ckpt = {
            'model': ckpt_state,
            'config': vars(args) if is_main_process else {},
            'metrics': {
                'loss_train': float(train_loss),
                'loss_val_ar': float(val_loss),
                'loss_test_rnn': float(test_loss),
                'acc_val_ar': float(val_acc),
                'acc_test_rnn': float(test_acc),
            }
        }
        try:
            save_checkpoint(ckpt_mgr, ckpt, int(state.step))
            if is_main_process:
                print(f"[Checkpoint] Epoch-end save: epoch={epoch}, global_step={int(state.step)}")
        except (OSError, ValueError) as e:
            print(f"\n[FATAL] Checkpoint save failed at epoch {epoch}: {e}")
            print("[FATAL] Likely disk quota or serialization issue. Exiting.")
            if ckpt_mgr is not None:
                try:
                    ckpt_mgr.close()
                except Exception:
                    pass
            sys.exit(1)
        del ckpt  # Free GPU memory held by state copy
        del ckpt_state  # Free re-sharded state copy (~1.5 GiB)

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
            best_test_last_order_loss = test_last_order_loss
            best_test_last_order_acc = test_last_order_acc
            best_test_last_order_ppl = test_last_order_ppl

        # reduce_lr_on_plateau is informational only — LR managed by optax schedules
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        _, _, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        best_test_log_msg_ppl = best_test_loss * tokens_per_msg
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} -- Best Test Log Msg PPL: {best_test_log_msg_ppl:.4f}"
            f" at Epoch {best_epoch + 1}\n"
            f"\tBest Last Order -- Loss: {best_test_last_order_loss:.5f}"
            f" Acc: {best_test_last_order_acc:.4f}"
            f" PPL: {best_test_last_order_ppl:.4f}\n"
        )

        if args.log_ce_tables:
            ce_table.add_column(name="val_ce_"+str(epoch),data=val_ce_means.tolist())
            ce_table.add_column(name="test_ce_"+str(epoch),data=test_ce_means.tolist())
            ce_table.add_column(name="val_acc_"+str(epoch),data=val_acc_means.tolist())
            ce_table.add_column(name="test_acc_"+str(epoch),data=test_acc_means.tolist())
            ce_table.add_column(name="train_ce_"+str(epoch),data=ce_by_tok.tolist())
            ce_table=wandb.Table(columns=ce_table.columns,data=ce_table.data)
        

        # Compute LR from schedule for logging (optax manages LR inside JIT)
        current_lr = float(lr_schedule_fn(step))
        current_ssm_lr = float(ssm_lr_schedule_fn(step))

        if valloader is not None:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val loss": val_loss,
                    "Val Accuracy": val_acc,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    "Val Log Msg PPL": val_log_msg_ppl,
                    "Test Log Msg PPL": test_log_msg_ppl,
                    "Test Last Order Loss": test_last_order_loss,
                    "Test Last Order Accuracy": test_last_order_acc,
                    "Test Last Order PPL": test_last_order_ppl,
                    "Val Last Order Loss": val_last_order_loss,
                    "Val Last Order Accuracy": val_last_order_acc,
                    "Val Last Order PPL": val_last_order_ppl,
                    "count": count,
                    "Learning rate count": lr_count,
                    "Opt acc": opt_acc,
                    "lr": current_lr,
                    "ssm_lr": current_ssm_lr,
                }
            )
        else:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val loss": val_loss,
                    "Val Accuracy": val_acc,
                    "Val Log Msg PPL": val_log_msg_ppl,
                    "Test Last Order Loss": test_last_order_loss,
                    "Test Last Order Accuracy": test_last_order_acc,
                    "Test Last Order PPL": test_last_order_ppl,
                    "count": count,
                    "Learning rate count": lr_count,
                    "Opt acc": opt_acc,
                    "lr": current_lr,
                    "ssm_lr": current_ssm_lr,
                }
            )

        if args.log_ce_tables:
            wandb.log({"CE by token": ce_table})
        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc
        wandb.run.summary["Best Test Log Msg PPL"] = best_test_log_msg_ppl
        wandb.run.summary["Best Test Last Order Loss"] = best_test_last_order_loss
        wandb.run.summary["Best Test Last Order Accuracy"] = best_test_last_order_acc
        wandb.run.summary["Best Test Last Order PPL"] = best_test_last_order_ppl
        # print("IGNORING EARLY STOPPING FOR TINY EPOCH SIZE ")
        # After each epoch: clear_caches causes JIT recompilation (~193s/epoch) and
        # accumulates NCCL cliques, leading to epoch-3+ OOM. Remove it.
        # Instead, reduce PER_GPU_BSZ to lower train_step workspace from 71→35 GiB.
        gc.collect()
        # jax.clear_caches()  # Removed: recompiles each epoch + NCCL clique accumulation
        # jax.profiler.stop_trace()
        # Barrier: sync all ranks after epoch cleanup before next iteration.
        # gc.collect() timing varies across ranks; without this, fast ranks may
        # start next epoch's train while slow ranks are still in GC.
        if is_distributed:
            sync_global_devices(f"post_epoch_{epoch}")
        if count > args.early_stop_patience:
            break

    # Wait for async checkpoint writes to complete before exiting
    if ckpt_mgr is not None:
        ckpt_mgr.wait_until_finished()
        ckpt_mgr.close()
