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

from lob.init_train import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate, remap_train_state_step
from lob.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader
from lob.lobster_dataloader import LOBSTER_Dataset
from lob.train_helpers import reduce_lr_on_plateau, train_epoch, validate, \
    create_jit_train_step, create_jit_eval_step, create_lobs5_learning_rate_schedule, \
    StepWatchdog, TIME_START_I, TIME_END_I, LR_MIN_FRACTION
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
    best_test_last_order_nll = 100000000
    best_test_all_orders_nll = 100000000

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
    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)

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
            tickers=getattr(args, 'tickers', None),
            data_root=getattr(args, 'data_root', None),
            train_date_range=getattr(args, 'train_date_range', None),
            test_date_range=getattr(args, 'test_date_range', None),
            token_mode=getattr(args, 'token_mode', '24tok'),
        )

    # Extract per-ticker test loaders if available
    per_ticker_test_loaders = aux_dataloaders.get('per_ticker_test', {})

    

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
            # Handle both plain multi_transform and chain(clip, multi_transform) structures
            _opt = state.opt_state
            if isinstance(_opt, tuple):
                _opt = _opt[-1]  # unwrap chain → last element is MultiTransformState
            ssm_inner = _opt.inner_states['ssm'].inner_state
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

            # --- Elastic Resume: remap step when device count or grad_accum changes ---
            ckpt_config = ckpt.get('config', {})
            original_process_count = ckpt_config.get('process_count', process_count)
            original_grad_accum = ckpt_config.get('grad_accum_steps', 1)
            if original_process_count != process_count or original_grad_accum != grad_accum_steps:
                # Compute original optimizer_steps_per_epoch (respecting curtail if checkpoint used it)
                original_curtail = ckpt_config.get('curtail_epochs', None)
                raw_original_micro_spe = train_size // (args.micro_bsz * args.num_devices * original_process_count)
                original_micro_spe = min(raw_original_micro_spe, original_curtail + 1) if original_curtail is not None else raw_original_micro_spe
                original_spe = original_micro_spe // max(original_grad_accum, 1)

                # Compute new optimizer_steps_per_epoch (respecting current curtail setting)
                raw_new_micro_spe = train_size // (args.micro_bsz * args.num_devices * process_count)
                new_micro_spe = min(raw_new_micro_spe, args.curtail_epochs + 1) if args.curtail_epochs is not None else raw_new_micro_spe
                new_spe = new_micro_spe // max(grad_accum_steps, 1)

                restored_epoch = int(state.step) // max(original_spe, 1)
                step_within_epoch = int(state.step) % max(original_spe, 1)
                scaled_step_within = round(step_within_epoch * new_spe / original_spe) if original_spe > 0 else 0
                remapped_step = restored_epoch * new_spe + scaled_step_within
                print(f"[Elastic Resume] process_count changed: {original_process_count} → {process_count}")
                print(f"[Elastic Resume] grad_accum changed: {original_grad_accum} → {grad_accum_steps}")
                print(f"[Elastic Resume] optimizer_steps/epoch: {original_spe} → {new_spe}")
                print(f"[Elastic Resume] intra-epoch: {step_within_epoch}/{original_spe} → {scaled_step_within}/{new_spe} ({step_within_epoch/max(original_spe,1)*100:.1f}%)")
                print(f"[Elastic Resume] state.step {int(state.step)} → {remapped_step} (epoch {restored_epoch})")
                state = remap_train_state_step(state, remapped_step)

        val_model = model_cls(training=False, step_rescale=1)
        token_mode = getattr(args, 'token_mode', '24tok')
        if token_mode == '1tok':
            init_hidden = model_cls().initialize_carry(
                batch_size=args.micro_bsz,
                hidden_size=(ssm_size // pow(2, int(args.conj_sym))),
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                n_fused_layers=args.n_layers,
                h_size_ema=ssm_size)
        else:
            init_hidden = model_cls().initialize_carry(
                batch_size=args.micro_bsz,
                hidden_size=(ssm_size // pow(2, int(args.conj_sym))),
                n_message_layers=args.n_message_layers,
                n_book_pre_layers=args.n_book_pre_layers,
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

        local_steps_k = getattr(args, 'local_steps_k', 0)
        if local_steps_k > 0:
            assert use_hierarchical, \
                "Local Steps (--local_steps_k>0) requires --hierarchical=True"
            print(f"[*] Local Steps enabled: sync params every {local_steps_k} steps "
                  f"(inner optimizer unchanged, only intra-node grad sync per step)")

        jit_train_step = create_jit_train_step(
            mesh, state, has_book_data=args.use_book_data,
            hierarchical=use_hierarchical,
            batchnorm=args.batchnorm, ignore_times=args.ignore_times,
            local_steps_k=local_steps_k,
            grad_accum_steps=grad_accum_steps)
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
    micro_steps_per_epoch = int(train_size / (args.micro_bsz * args.num_devices * process_count)) if args.curtail_epochs is None else args.curtail_epochs+1
    # steps_per_epoch in optimizer updates (= micro_steps // K)
    steps_per_epoch = micro_steps_per_epoch // grad_accum_steps
    if grad_accum_steps > 1:
        print(f"[GradAccum] K={grad_accum_steps}: micro_steps/epoch={micro_steps_per_epoch}, "
              f"optimizer_steps/epoch={steps_per_epoch}, "
              f"effective_bsz={args.micro_bsz * args.num_devices * process_count * grad_accum_steps}")

    # Mini-epoch: split each data epoch into K sub-epochs for frequent eval
    mini_epochs = getattr(args, 'mini_epochs', 1)
    if mini_epochs > 1:
        steps_per_mini = steps_per_epoch // mini_epochs
        validate_every_n_steps = steps_per_mini
        print(f"[Schedule] mini_epochs={mini_epochs}, steps_per_mini={steps_per_mini}, "
              f"steps_per_epoch={steps_per_epoch}")
    else:
        validate_every_n_steps = 0

    # Create LR schedule functions for wandb logging (optax manages LR inside JIT)
    total_steps = steps_per_epoch * args.epochs
    warmup_end_step = int(steps_per_epoch * args.warmup_end)

    effective_lr_min = args.lr_min if args.lr_min > 0 else lr * LR_MIN_FRACTION
    effective_ssm_lr_min = args.lr_min if args.lr_min > 0 else ssm_lr * LR_MIN_FRACTION

    lr_schedule_fn = create_lobs5_learning_rate_schedule(
        base_lr=lr, warmup_end_step=warmup_end_step,
        total_steps=total_steps, lr_min=effective_lr_min,
        use_cosine_anneal=args.cosine_anneal)
    ssm_lr_schedule_fn = create_lobs5_learning_rate_schedule(
        base_lr=ssm_lr, warmup_end_step=warmup_end_step,
        total_steps=total_steps, lr_min=effective_ssm_lr_min,
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

    # ── Mini-epoch validation callback ──
    mini_epoch_counter = [0]  # mutable for closure

    def _save_mini_epoch_checkpoint(cb_state, cb_epoch, mini_idx, val_loss, test_loss, val_acc, test_acc, train_loss_avg):
        """Save checkpoint with val+test metrics at mini-epoch boundary."""
        global_step = int(cb_state.step)
        if is_distributed:
            ckpt_st = _reshard_for_ckpt(cb_state)
        else:
            ckpt_st = deduplicate_trainstate(cb_state)
        ckpt = {
            'model': ckpt_st,
            'config': vars(args) if is_main_process else {},
            'metrics': {
                'loss_train': float(train_loss_avg),
                'loss_val_ar': float(val_loss),
                'loss_test_rnn': float(test_loss),
                'acc_val_ar': float(val_acc),
                'acc_test_rnn': float(test_acc),
                'mini_epoch': mini_idx,
                'epoch': cb_epoch,
            }
        }
        try:
            save_checkpoint(ckpt_mgr, ckpt, global_step)
            if is_main_process:
                print(f"[Checkpoint] Mini-epoch save: epoch={cb_epoch}, mini={mini_idx}, "
                      f"global_step={global_step}")
        except (OSError, ValueError) as e:
            print(f"[Checkpoint] WARNING: mini-epoch save failed: {e}")

    def mini_epoch_validate(cb_state, cb_epoch, cb_batch_idx):
        """Run val + test + checkpoint at mini-epoch boundary. Returns True to stop."""
        nonlocal best_acc, best_loss, best_epoch, count, best_val_loss
        nonlocal best_test_loss, best_test_acc
        nonlocal best_test_last_order_loss, best_test_last_order_acc
        nonlocal best_test_last_order_nll, best_test_all_orders_nll
        nonlocal lr_count, opt_acc

        current_mini = mini_epoch_counter[0]
        mini_epoch_counter[0] += 1
        global_step = int(cb_state.step)

        print(f"\n[Mini-epoch {current_mini + 1}/{mini_epochs}] "
              f"Epoch {cb_epoch + 1}, step {cb_batch_idx + 1}")

        # Barrier before eval
        if is_distributed:
            sync_global_devices(f"pre_eval_mini_{cb_epoch}_{current_mini}")

        # === Validation ===
        eval_watchdog = StepWatchdog(timeout=1200)
        eval_watchdog.kick(cb_epoch, 0)

        (val_loss, val_acc,
            val_ce_means, val_acc_means,
            val_last_order_loss, val_last_order_acc,
            val_last_order_nll, val_all_orders_nll) = validate(cb_state,
                                        val_model.apply,
                                        valloader,
                                        seq_len,
                                        in_dim,
                                        batchnorm,
                                        args.num_devices,
                                        cb_epoch,
                                        curtail_epoch=args.curtail_epochs,
                                        apply_method='__call_ar__',
                                        ignore_times=ignore_times,
                                        log_ce_tables=args.log_ce_tables,
                                        mesh=mesh,
                                        jit_eval_step_fn=jit_eval_step,
                                        silent=True)

        # === Test ===
        eval_watchdog.kick(cb_epoch, 1)
        (test_loss, test_acc,
          test_ce_means, test_acc_means,
          test_last_order_loss, test_last_order_acc,
          test_last_order_nll, test_all_orders_nll) = validate(cb_state,
                                       val_model.apply,
                                       testloader,
                                       seq_len,
                                       in_dim,
                                       batchnorm,
                                       args.num_devices,
                                       cb_epoch,
                                       curtail_epoch=args.curtail_epochs,
                                       apply_method='__call_ar__',
                                       ignore_times=ignore_times,
                                       log_ce_tables=args.log_ce_tables,
                                       mesh=mesh,
                                       jit_eval_step_fn=jit_eval_step,
                                       silent=True)

        eval_watchdog.stop()

        # === Per-ticker test (if multi-ticker) ===
        per_ticker_metrics = {}
        if per_ticker_test_loaders:
            for ticker, ticker_loader in per_ticker_test_loaders.items():
                print(f"[*] Mini-epoch {current_mini + 1} Test [{ticker}]")
                (t_loss, t_acc, _, _, _, _, _, _) = validate(
                    cb_state, val_model.apply, ticker_loader,
                    seq_len, in_dim, batchnorm, args.num_devices, cb_epoch,
                    curtail_epoch=args.curtail_epochs,
                    apply_method='__call_ar__',
                    ignore_times=ignore_times,
                    log_ce_tables=False,
                    mesh=mesh,
                    jit_eval_step_fn=jit_eval_step,
                    silent=True)
                per_ticker_metrics[ticker] = {'loss': float(t_loss), 'acc': float(t_acc)}
                print(f"  [{ticker}] Loss: {t_loss:.5f}  Acc: {t_acc:.4f}")

        # === Print metrics ===
        # Compute train loss average so far this epoch
        # (batch_losses is in train_epoch scope, not accessible here; use step_loss wandb)
        print(f"\n=>> Mini-epoch {current_mini + 1}/{mini_epochs} Metrics ===")
        print(f"\tAll Orders -- Val Loss: {val_loss:.5f} Val Acc: {val_acc:.4f} "
              f"Val NLL: {val_all_orders_nll:.4f}")
        print(f"\t              Test Loss: {test_loss:.5f} Test Acc: {test_acc:.4f} "
              f"Test NLL: {test_all_orders_nll:.4f}")
        print(f"\tLast Order -- Val Loss: {val_last_order_loss:.5f} Val Acc: {val_last_order_acc:.4f} "
              f"Val NLL: {val_last_order_nll:.4f}")
        print(f"\t              Test Loss: {test_last_order_loss:.5f} Test Acc: {test_last_order_acc:.4f} "
              f"Test NLL: {test_last_order_nll:.4f}")

        # === Checkpoint with val/test metrics ===
        _save_mini_epoch_checkpoint(cb_state, cb_epoch, current_mini,
                                    val_loss, test_loss, val_acc, test_acc,
                                    float(val_loss))  # train_loss not available here, use val_loss

        # === Update best metrics + early stopping ===
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, cb_epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc
            best_test_last_order_loss = test_last_order_loss
            best_test_last_order_acc = test_last_order_acc
            best_test_last_order_nll = test_last_order_nll
            best_test_all_orders_nll = test_all_orders_nll

        # reduce_lr_on_plateau (informational only)
        input_rl = lr, ssm_lr, lr_count, val_acc, opt_acc
        _, _, lr_count, opt_acc = reduce_lr_on_plateau(
            input_rl, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best so far
        print(f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy: {best_acc:.4f}"
              f" at Epoch {best_epoch + 1}\n"
              f"\tBest All Orders -- Loss: {best_test_loss:.5f}"
              f" Acc: {best_test_acc:.4f}"
              f" NLL: {best_test_all_orders_nll:.4f}\n"
              f"\tBest Last Order -- Loss: {best_test_last_order_loss:.5f}"
              f" Acc: {best_test_last_order_acc:.4f}"
              f" NLL: {best_test_last_order_nll:.4f}\n")

        # === wandb logging ===
        current_lr = float(lr_schedule_fn(global_step))
        current_ssm_lr = float(ssm_lr_schedule_fn(global_step))

        ticker_wandb = {}
        if per_ticker_metrics:
            for ticker, tm in per_ticker_metrics.items():
                ticker_wandb[f"test/{ticker}/loss"] = tm['loss']
                ticker_wandb[f"test/{ticker}/accuracy"] = tm['acc']

        if is_main_process and args.USE_WANDB:
            wandb.log({
                "Val loss": val_loss,
                "Val Accuracy": val_acc,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "Val All Orders NLL": val_all_orders_nll,
                "Test All Orders NLL": test_all_orders_nll,
                "Test Last Order Loss": test_last_order_loss,
                "Test Last Order Accuracy": test_last_order_acc,
                "Test Last Order NLL": test_last_order_nll,
                "Val Last Order Loss": val_last_order_loss,
                "Val Last Order Accuracy": val_last_order_acc,
                "Val Last Order NLL": val_last_order_nll,
                "count": count,
                "Learning rate count": lr_count,
                "Opt acc": opt_acc,
                "lr": current_lr,
                "ssm_lr": current_ssm_lr,
                "mini_epoch": current_mini + 1,
                **ticker_wandb,
            }, step=global_step)

            wandb.run.summary["Best Val Loss"] = best_loss
            wandb.run.summary["Best Val Accuracy"] = best_acc
            wandb.run.summary["Best Epoch"] = best_epoch
            wandb.run.summary["Best Test Loss"] = best_test_loss
            wandb.run.summary["Best Test Accuracy"] = best_test_acc
            wandb.run.summary["Best Test All Orders NLL"] = best_test_all_orders_nll
            wandb.run.summary["Best Test Last Order Loss"] = best_test_last_order_loss
            wandb.run.summary["Best Test Last Order Accuracy"] = best_test_last_order_acc
            wandb.run.summary["Best Test Last Order NLL"] = best_test_last_order_nll

        # === Barrier after eval ===
        if is_distributed:
            sync_global_devices(f"post_eval_mini_{cb_epoch}_{current_mini}")

        # === Memory cleanup before returning to training ===
        # Eval + checkpoint + per-ticker tests fragment GPU memory.
        # Without cleanup, train_step can OOM trying to re-allocate its workspace.
        del per_ticker_metrics, ticker_wandb
        gc.collect()

        return count > args.early_stop_patience  # True → stop training

    for epoch in range(start_epoch, args.epochs):
        # Free residual memory from previous epoch's val/test before training
        gc.collect()
        mini_epoch_counter[0] = 0  # reset for each data epoch

        # Mid-epoch resume: rebuild trainloader with sampler-level skip
        # so DataLoader never calls __getitem__ for already-completed batches.
        if resume_from_step is not None and resume_from_step > 0:
            print(f"[Resume] Rebuilding trainloader with sampler skip: "
                  f"step={resume_from_step}, epoch={epoch}")
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                seed=args.jax_seed,
                per_process_bsz=args.micro_bsz * args.num_devices,
                num_workers=args.n_data_workers,
                reset_train_offsets=False,
                shuffle=args.shuffle_train,
                use_distributed_sampler=is_distributed,
                process_rank=process_rank,
                process_count=process_count,
                resume_from_step=resume_from_step,
                resume_epoch=epoch,
            )
        else:
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
                                              validate_callback=mini_epoch_validate if mini_epochs > 1 else None,
                                              validate_every_n_steps=validate_every_n_steps,
                                              )
        # resume_from_step only applies to the first epoch after restore.
        # If we rebuilt the trainloader with sampler skip, restore the normal
        # loader for subsequent epochs (so DistributedSampler.set_epoch works).
        if resume_from_step is not None:
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                seed=args.jax_seed,
                per_process_bsz=args.micro_bsz * args.num_devices,
                num_workers=args.n_data_workers,
                reset_train_offsets=False,
                shuffle=args.shuffle_train,
                use_distributed_sampler=is_distributed,
                process_rank=process_rank,
                process_count=process_count,
            )
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

        # Mini-epochs > 1: all validation/checkpoint/metrics handled by callback
        if mini_epochs > 1:
            # Handle trailing steps: if epoch didn't end exactly on a mini-epoch
            # boundary, run one final evaluation for the remaining steps.
            actual_steps = steps_per_epoch  # curtail already baked into steps_per_epoch
            if validate_every_n_steps > 0 and actual_steps % validate_every_n_steps != 0:
                last_batch_idx = actual_steps - 1
                print(f"[Mini-epoch] Trailing {actual_steps % validate_every_n_steps} steps — "
                      f"running final eval at step {actual_steps}")
                mini_epoch_validate(state, epoch, last_batch_idx)
            gc.collect()
            if is_distributed:
                sync_global_devices(f"post_epoch_{epoch}")
            if count > args.early_stop_patience:
                break
            continue  # skip epoch-end eval block

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
                val_last_order_nll, val_all_orders_nll) = validate(state,
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
              test_last_order_nll, test_all_orders_nll) = validate(state,
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

            # Per-ticker test eval (multi-ticker mode)
            per_ticker_metrics = {}
            if per_ticker_test_loaders:
                for ticker, ticker_loader in per_ticker_test_loaders.items():
                    print(f"[*] Running Epoch {epoch + 1} Test [{ticker}]")
                    (t_loss, t_acc, _, _, _, _, _, _) = validate(
                        state, val_model.apply, ticker_loader,
                        seq_len, in_dim, batchnorm, args.num_devices, epoch,
                        curtail_epoch=args.curtail_epochs,
                        apply_method='__call_ar__',
                        ignore_times=ignore_times,
                        log_ce_tables=False,
                        mesh=mesh,
                        jit_eval_step_fn=jit_eval_step)
                    per_ticker_metrics[ticker] = {'loss': float(t_loss), 'acc': float(t_acc)}
                    print(f"  [{ticker}] Loss: {t_loss:.5f}  Acc: {t_acc:.4f}")

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}"
            )
            print(
                f"\tAll Orders -- Val Loss: {val_loss:.5f} Val Acc: {val_acc:.4f} Val NLL: {val_all_orders_nll:.4f}"
                f" | Test Loss: {test_loss:.5f} Test Acc: {test_acc:.4f} Test NLL: {test_all_orders_nll:.4f}"
            )
            print(
                f"\tLast Order -- Val Loss: {val_last_order_loss:.5f} Val Acc: {val_last_order_acc:.4f} Val NLL: {val_last_order_nll:.4f}"
                f" | Test Loss: {test_last_order_loss:.5f} Test Acc: {test_last_order_acc:.4f} Test NLL: {test_last_order_nll:.4f}"
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
              test_last_order_nll, test_all_orders_nll) = validate(state,
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
            per_ticker_metrics = {}
            val_loss=test_loss
            val_acc=test_acc
            val_last_order_loss = test_last_order_loss
            val_last_order_acc = test_last_order_acc
            val_last_order_nll = test_last_order_nll
            val_all_orders_nll = test_all_orders_nll

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}"
            )
            print(
                f"\tAll Orders -- Test Loss: {test_loss:.5f} Test Acc: {test_acc:.4f}"
                f" Test NLL: {test_all_orders_nll:.4f}"
            )
            print(
                f"\tLast Order -- Test Loss: {test_last_order_loss:.5f}"
                f" Test Acc: {test_last_order_acc:.4f}"
                f" Test NLL: {test_last_order_nll:.4f}"
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
            best_test_last_order_nll = test_last_order_nll
            best_test_all_orders_nll = test_all_orders_nll

        # reduce_lr_on_plateau is informational only — LR managed by optax schedules
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        _, _, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest All Orders -- Loss: {best_test_loss:.5f}"
            f" Acc: {best_test_acc:.4f}"
            f" NLL: {best_test_all_orders_nll:.4f}"
            f" at Epoch {best_epoch + 1}\n"
            f"\tBest Last Order -- Loss: {best_test_last_order_loss:.5f}"
            f" Acc: {best_test_last_order_acc:.4f}"
            f" NLL: {best_test_last_order_nll:.4f}\n"
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

        # Per-ticker wandb metrics
        ticker_wandb = {}
        if per_ticker_metrics:
            for ticker, tm in per_ticker_metrics.items():
                ticker_wandb[f"test/{ticker}/loss"] = tm['loss']
                ticker_wandb[f"test/{ticker}/accuracy"] = tm['acc']

        if valloader is not None:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val loss": val_loss,
                    "Val Accuracy": val_acc,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    "Val All Orders NLL": val_all_orders_nll,
                    "Test All Orders NLL": test_all_orders_nll,
                    "Test Last Order Loss": test_last_order_loss,
                    "Test Last Order Accuracy": test_last_order_acc,
                    "Test Last Order NLL": test_last_order_nll,
                    "Val Last Order Loss": val_last_order_loss,
                    "Val Last Order Accuracy": val_last_order_acc,
                    "Val Last Order NLL": val_last_order_nll,
                    "count": count,
                    "Learning rate count": lr_count,
                    "Opt acc": opt_acc,
                    "lr": current_lr,
                    "ssm_lr": current_ssm_lr,
                    **ticker_wandb,
                }
            )
        else:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val loss": val_loss,
                    "Val Accuracy": val_acc,
                    "Val All Orders NLL": val_all_orders_nll,
                    "Test All Orders NLL": test_all_orders_nll,
                    "Test Last Order Loss": test_last_order_loss,
                    "Test Last Order Accuracy": test_last_order_acc,
                    "Test Last Order NLL": test_last_order_nll,
                    "count": count,
                    "Learning rate count": lr_count,
                    "Opt acc": opt_acc,
                    "lr": current_lr,
                    "ssm_lr": current_ssm_lr,
                    **ticker_wandb,
                }
            )

        if args.log_ce_tables:
            wandb.log({"CE by token": ce_table})
        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc
        wandb.run.summary["Best Test All Orders NLL"] = best_test_all_orders_nll
        wandb.run.summary["Best Test Last Order Loss"] = best_test_last_order_loss
        wandb.run.summary["Best Test Last Order Accuracy"] = best_test_last_order_acc
        wandb.run.summary["Best Test Last Order NLL"] = best_test_last_order_nll
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
