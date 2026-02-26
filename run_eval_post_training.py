"""Post-training evaluation script.

Loads a checkpoint, recreates the same data split (deterministic from seed + val_split),
and runs validation and test evaluation. All model/data parameters are read from
checkpoint metadata — only --restore is required.

Usage:
    python run_eval_post_training.py --restore /path/to/checkpoint [--restore_step N]

The val/test split is reproduced identically when:
  - Same data files exist on disk (same DATA_ROOT / dir_name)
  - Same val_split (default 0.01) and seed (default 42)
"""
import os
import sys
import json
import time

# Worker processes must not use GPU
if __name__ != "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-training eval: load checkpoint, run val/test")
    parser.add_argument("--restore", type=str, required=True,
                        help="Checkpoint directory path")
    parser.add_argument("--restore_step", type=int, default=None,
                        help="Checkpoint step (default: latest)")
    parser.add_argument("--micro_bsz", type=int, default=None,
                        help="Override per-GPU batch size (default: from metadata)")
    parser.add_argument("--val_split", type=float, default=None,
                        help="Override val_split (default: from metadata, fallback 0.01)")
    parser.add_argument("--ignore_times", type=str, default=None,
                        help="Override ignore_times (default: from metadata)")
    parser.add_argument("--eval_val", action="store_true", default=True,
                        help="Run validation eval (default: True)")
    parser.add_argument("--eval_test", action="store_true", default=True,
                        help="Run test eval (default: True)")
    parser.add_argument("--no_val", action="store_true",
                        help="Skip validation eval")
    parser.add_argument("--no_test", action="store_true",
                        help="Skip test eval")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file (default: auto in checkpoint dir)")
    parser.add_argument("--log_ce_tables", action="store_true",
                        help="Log per-token-position CE and accuracy tables")
    cli_args = parser.parse_args()

    # GPU setup
    _n_gpus = int(os.environ.get('GPUS_PER_NODE', '4'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(_n_gpus))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

    import jax
    import jax.numpy as jnp
    import numpy as onp

    num_devices = jax.local_device_count()
    print(f"[Eval] Devices: {num_devices} GPUs")
    print(f"[Eval] Checkpoint: {cli_args.restore}")

    # ── Step 1: Load metadata from checkpoint ──
    from lob.init_train import load_metadata, load_checkpoint, init_train_state
    args = load_metadata(cli_args.restore)
    print(f"[Eval] Loaded metadata: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"blocks={args.blocks}, ssm_size={args.ssm_size_base}")

    # Override with CLI args
    args.num_devices = num_devices
    args.process_count = 1
    args.process_index = 0
    args.is_distributed = False
    if cli_args.micro_bsz is not None:
        args.micro_bsz = cli_args.micro_bsz
    if cli_args.val_split is not None:
        val_split = cli_args.val_split
    else:
        val_split = getattr(args, 'val_split', 0.01)
    if cli_args.ignore_times is not None:
        args.ignore_times = cli_args.ignore_times.lower() in ('true', '1', 'yes')

    # Ensure required args have defaults
    if not hasattr(args, 'batchnorm'):
        args.batchnorm = False
    if not hasattr(args, 'ignore_times'):
        args.ignore_times = False

    print(f"[Eval] micro_bsz={args.micro_bsz}, val_split={val_split}, "
          f"ignore_times={args.ignore_times}")

    # ── Step 2: Create dataset with identical split ──
    from lob.dataloading import create_lobster_prediction_dataset
    from lob.lobster_dataloader import LOBSTER_Dataset

    mask_fn = LOBSTER_Dataset.no_mask
    masking = getattr(args, 'masking', 'none')
    if masking == 'causal':
        mask_fn = LOBSTER_Dataset.causal_mask
    elif masking == 'random':
        mask_fn = LOBSTER_Dataset.random_mask
    elif masking == 'last_pos':
        mask_fn = LOBSTER_Dataset.last_pos_mask

    dataset_kwargs = dict(
        cache_dir=args.dir_name,
        seed=args.jax_seed,
        mask_fn=mask_fn,
        msg_seq_len=args.msg_seq_len,
        micro_bsz=args.micro_bsz,
        num_devices=num_devices,
        use_book_data=getattr(args, 'use_book_data', True),
        use_simple_book=getattr(args, 'use_simple_book', False),
        book_transform=getattr(args, 'book_transform', True),
        n_data_workers=0,
        shuffle_train=False,
        rand_offset=False,
        debug_overfit=False,
        val_split=val_split,
        test_split=getattr(args, 'test_split', 0.1),
        test_dir_name=getattr(args, 'test_dir_name', None),
        use_distributed_sampler=False,
        process_rank=0,
        process_count=1,
    )

    # Multi-ticker support
    tickers = getattr(args, 'tickers', None)
    if tickers is not None:
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(',')]
        dataset_kwargs['tickers'] = tickers
        dataset_kwargs['data_root'] = getattr(args, 'data_root', None)
        train_dr = getattr(args, 'train_date_range', None)
        test_dr = getattr(args, 'test_date_range', None)
        if isinstance(train_dr, str):
            train_dr = tuple(train_dr.split(','))
        if isinstance(test_dr, str):
            test_dr = tuple(test_dr.split(','))
        dataset_kwargs['train_date_range'] = train_dr
        dataset_kwargs['test_date_range'] = test_dr

    print("[Eval] Creating dataset...")
    (lobster_dataset, trainloader, valloader, testloader, aux_dataloaders,
     n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size) = \
        create_lobster_prediction_dataset(**dataset_kwargs)

    print(f"[Eval] Dataset: n_classes={n_classes}, seq_len={seq_len}, "
          f"book_dim={book_dim}, book_seq_len={book_seq_len}")
    if valloader is not None:
        print(f"[Eval] Val batches: {len(valloader)}")
    if testloader is not None:
        print(f"[Eval] Test batches: {len(testloader)}")

    # ── Step 3: Init model + load checkpoint ──
    from lob.sharding_utils import initialize_mesh
    from lob.train_helpers import validate, create_jit_eval_step

    print("[Eval] Initializing model...")
    state, model_cls = init_train_state(
        args, n_classes, seq_len, book_dim, book_seq_len, train_size=0)

    mesh = initialize_mesh(num_devices)
    print(f"[Eval] Loading checkpoint (step={cli_args.restore_step or 'latest'})...")
    ckpt = load_checkpoint(
        state, cli_args.restore, step=cli_args.restore_step,
        train=True, mesh=mesh)
    state = ckpt['model']

    eval_model = model_cls(training=False, step_rescale=1.0)

    # JIT compile eval step
    print("[Eval] JIT compiling eval step...")
    jit_eval_step = create_jit_eval_step(
        mesh, state,
        has_book_data=getattr(args, 'use_book_data', True))

    # ── Step 4: Run evaluation ──
    results = {
        'checkpoint': cli_args.restore,
        'step': cli_args.restore_step or 'latest',
        'val_split': val_split,
        'seed': args.jax_seed,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'blocks': args.blocks,
        'ssm_size_base': args.ssm_size_base,
        'ignore_times': args.ignore_times,
    }

    if not cli_args.no_val and valloader is not None and len(valloader) > 0:
        print("\n" + "=" * 60)
        print("[Eval] Running VALIDATION...")
        print("=" * 60)
        t0 = time.time()
        (val_loss, val_acc, val_ce, val_acc_table,
         val_lo_loss, val_lo_acc, val_lo_nll, val_all_nll) = validate(
            state, eval_model.apply, valloader, seq_len, in_dim,
            args.batchnorm, num_devices, epoch=0,
            ignore_times=args.ignore_times,
            apply_method='__call_ar__',
            mesh=mesh, jit_eval_step_fn=jit_eval_step,
            log_ce_tables=cli_args.log_ce_tables)
        val_time = time.time() - t0

        results['val'] = {
            'loss': float(val_loss),
            'acc': float(val_acc),
            'last_order_loss': float(val_lo_loss),
            'last_order_acc': float(val_lo_acc),
            'last_order_nll': float(val_lo_nll),
            'all_orders_nll': float(val_all_nll),
            'eval_time_s': val_time,
        }
        print(f"\n[Val Results] Loss={val_loss:.4f}  Acc={val_acc:.4f}  "
              f"LO_Loss={val_lo_loss:.4f}  LO_Acc={val_lo_acc:.4f}")
        print(f"[Val Results] LO_NLL={val_lo_nll:.4f}  All_NLL={val_all_nll:.4f}  "
              f"Time={val_time:.1f}s")

    if not cli_args.no_test and testloader is not None and len(testloader) > 0:
        print("\n" + "=" * 60)
        print("[Eval] Running TEST...")
        print("=" * 60)
        t0 = time.time()
        (test_loss, test_acc, test_ce, test_acc_table,
         test_lo_loss, test_lo_acc, test_lo_nll, test_all_nll) = validate(
            state, eval_model.apply, testloader, seq_len, in_dim,
            args.batchnorm, num_devices, epoch=0,
            ignore_times=args.ignore_times,
            apply_method='__call_ar__',
            mesh=mesh, jit_eval_step_fn=jit_eval_step,
            log_ce_tables=cli_args.log_ce_tables)
        test_time = time.time() - t0

        results['test'] = {
            'loss': float(test_loss),
            'acc': float(test_acc),
            'last_order_loss': float(test_lo_loss),
            'last_order_acc': float(test_lo_acc),
            'last_order_nll': float(test_lo_nll),
            'all_orders_nll': float(test_all_nll),
            'eval_time_s': test_time,
        }
        print(f"\n[Test Results] Loss={test_loss:.4f}  Acc={test_acc:.4f}  "
              f"LO_Loss={test_lo_loss:.4f}  LO_Acc={test_lo_acc:.4f}")
        print(f"[Test Results] LO_NLL={test_lo_nll:.4f}  All_NLL={test_all_nll:.4f}  "
              f"Time={test_time:.1f}s")

    # Per-ticker test eval
    per_ticker_test_loaders = aux_dataloaders.get('per_ticker_test', {})
    if per_ticker_test_loaders and not cli_args.no_test:
        results['per_ticker_test'] = {}
        for ticker, tk_loader in per_ticker_test_loaders.items():
            print(f"\n[Eval] Running TEST for {ticker}...")
            t0 = time.time()
            (tk_loss, tk_acc, _, _, tk_lo_loss, tk_lo_acc, tk_lo_nll, tk_all_nll) = validate(
                state, eval_model.apply, tk_loader, seq_len, in_dim,
                args.batchnorm, num_devices, epoch=0,
                ignore_times=args.ignore_times,
                apply_method='__call_ar__',
                mesh=mesh, jit_eval_step_fn=jit_eval_step)
            tk_time = time.time() - t0
            results['per_ticker_test'][ticker] = {
                'loss': float(tk_loss), 'acc': float(tk_acc),
                'last_order_loss': float(tk_lo_loss), 'last_order_acc': float(tk_lo_acc),
                'last_order_nll': float(tk_lo_nll), 'all_orders_nll': float(tk_all_nll),
            }
            print(f"  {ticker}: Loss={tk_loss:.4f}  Acc={tk_acc:.4f}  "
                  f"LO_Acc={tk_lo_acc:.4f}  ({tk_time:.1f}s)")

    # ── Step 5: Save results ──
    if cli_args.output_json:
        output_path = cli_args.output_json
    else:
        step_str = cli_args.restore_step if cli_args.restore_step else 'latest'
        output_path = os.path.join(
            os.path.dirname(cli_args.restore) or '.',
            f'eval_results_step{step_str}.json')

    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Eval] Results saved to {output_path}")
    except (OSError, IOError) as e:
        print(f"[Eval] WARNING: Could not save results: {e}")
        print(json.dumps(results, indent=2))

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"blocks={args.blocks}, ssm={args.ssm_size_base}")
    print(f"Checkpoint: {cli_args.restore} (step={results['step']})")
    print(f"val_split={val_split}, seed={args.jax_seed}, "
          f"ignore_times={args.ignore_times}")
    print("-" * 60)
    if 'val' in results:
        v = results['val']
        print(f"VAL:  Loss={v['loss']:.4f}  Acc={v['acc']:.4f}  "
              f"LO_Loss={v['last_order_loss']:.4f}  LO_Acc={v['last_order_acc']:.4f}")
    if 'test' in results:
        t = results['test']
        print(f"TEST: Loss={t['loss']:.4f}  Acc={t['acc']:.4f}  "
              f"LO_Loss={t['last_order_loss']:.4f}  LO_Acc={t['last_order_acc']:.4f}")
    if 'per_ticker_test' in results:
        for ticker, t in results['per_ticker_test'].items():
            print(f"  {ticker}: Loss={t['loss']:.4f}  Acc={t['acc']:.4f}  "
                  f"LO_Acc={t['last_order_acc']:.4f}")
    print("=" * 60)
