#!/usr/bin/env python
"""
LOBMAX Training Entry Point

Pure Transformer-based LOB prediction using MaxText.
This replaces the S5-based run_train.py with a cleaner Transformer implementation.

Usage:
    python run_lobmax.py \
        --d_model=2048 \
        --n_layers=24 \
        --num_heads=16 \
        --USE_WANDB=True \
        ...
"""

import os
import sys

if __name__ == "__main__":
    pass
else:
    # Forces all generated worker processes to not run on GPU.
    #  Required because the worker spawn interface happens after CUDA init.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"

# Add MaxText to path
MAXTEXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'maxtext', 'src')
if MAXTEXT_PATH not in sys.path:
    sys.path.insert(0, MAXTEXT_PATH)

if __name__ == "__main__":
    import argparse
    import time
    from s5.utils.util import str2bool
    from lob.dataloading import Datasets

    # ============================================
    # Step 1: Detect Multi-Node Environment
    # ============================================
    is_slurm_multi_node = int(os.environ.get('SLURM_NNODES', '1')) > 1

    if is_slurm_multi_node:
        print(f"[*] Detected Slurm multi-node environment ({os.environ.get('SLURM_NNODES')} nodes)")
        print(f"[*] Using Slurm GPU allocation: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["NCCL_TIMEOUT"] = "600"

    # ============================================
    # Model Presets for Transformer
    # ============================================
    MODEL_PRESETS = {
        "3B": {
            "d_model": 2560, "n_layers": 32, "num_heads": 20, "mlp_dim": 10240,
            "micro_bsz": 1, "lr_base": 0.00008, "lr_factor": 1,
            "wandb_project": "lobmax-3B"
        },
        "1.3B": {
            "d_model": 2048, "n_layers": 24, "num_heads": 16, "mlp_dim": 8192,
            "micro_bsz": 2, "lr_base": 0.00015, "lr_factor": 1,
            "wandb_project": "lobmax-1.3B"
        },
        "360M": {
            "d_model": 1024, "n_layers": 24, "num_heads": 16, "mlp_dim": 4096,
            "micro_bsz": 4, "lr_base": 0.0003, "lr_factor": 1,
            "wandb_project": "lobmax-360M"
        },
        "125M": {
            "d_model": 768, "n_layers": 12, "num_heads": 12, "mlp_dim": 3072,
            "micro_bsz": 8, "lr_base": 0.0005, "lr_factor": 1,
            "wandb_project": "lobmax-125M"
        },
        # MoE Presets - ~10x total params, SAME activated params per token as dense
        # Naming: TotalParams-AActivatedParams (like Qwen3-30B-A3B)
        # Use num_experts=16, num_experts_per_tok=1 for ~10x total params with same activation
        "1.25B-A125M": {
            # 125M dense → ~1.25B total MoE, ~125M activated per token
            "d_model": 768, "n_layers": 12, "num_heads": 12, "mlp_dim": 3072,
            "num_experts": 16, "num_experts_per_tok": 1,
            "micro_bsz": 6, "lr_base": 0.0005, "lr_factor": 1,
            "wandb_project": "lobmax-1.25B-A125M"
        },
        "3.6B-A360M": {
            # 360M dense → ~3.6B total MoE, ~360M activated per token
            "d_model": 1024, "n_layers": 24, "num_heads": 16, "mlp_dim": 4096,
            "num_experts": 16, "num_experts_per_tok": 1,
            "micro_bsz": 3, "lr_base": 0.0003, "lr_factor": 1,
            "wandb_project": "lobmax-3.6B-A360M"
        },
        "13B-A1.3B": {
            # 1.3B dense → ~13B total MoE, ~1.3B activated per token
            "d_model": 2048, "n_layers": 24, "num_heads": 16, "mlp_dim": 8192,
            "num_experts": 16, "num_experts_per_tok": 1,
            "micro_bsz": 1, "lr_base": 0.00015, "lr_factor": 1,
            "wandb_project": "lobmax-13B-A1.3B"
        },
    }

    parser = argparse.ArgumentParser(description="LOBMAX: Transformer-based LOB Prediction")

    # === WandB and Data ===
    parser.add_argument("--USE_WANDB", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="lobmax")
    parser.add_argument("--wandb_entity", type=str, default="kang-oxford")
    parser.add_argument("--dir_name", type=str, default='./data/LOBS5v2Cached/')
    parser.add_argument("--test_dir_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=Datasets.keys(), default='lobster-prediction')

    # === Model Preset ===
    parser.add_argument("--model_preset", type=str, default=None,
                        choices=list(MODEL_PRESETS.keys()))

    # === Transformer Architecture ===
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--mlp_dim", type=int, default=None)
    parser.add_argument("--n_message_layers", type=int, default=2)
    parser.add_argument("--n_book_pre_layers", type=int, default=1)
    parser.add_argument("--n_book_post_layers", type=int, default=1)

    # === MoE (Mixture of Experts) Configuration ===
    parser.add_argument("--num_experts", type=int, default=1,
                        help="Number of experts in MoE layer. Set to 1 for dense model.")
    parser.add_argument("--num_experts_per_tok", type=int, default=1,
                        help="Number of experts activated per token (top-k routing).")
    parser.add_argument("--sparse_matmul", type=str2bool, default=True,
                        help="Use sparse matmul for MoE (efficient kernel).")
    parser.add_argument("--megablox", type=str2bool, default=True,
                        help="Use Megablox kernel for sparse MoE matmul.")
    parser.add_argument("--capacity_factor", type=float, default=-1.0,
                        help="Expert capacity factor. -1.0 for dropless MoE.")
    parser.add_argument("--load_balance_loss_weight", type=float, default=0.01,
                        help="Weight for load balancing auxiliary loss.")

    # === Attention Configuration ===
    parser.add_argument("--attention_kernel", type=str, default="flash",
                        choices=["dot_product", "flash", "cudnn_flash_te", "cudnn_flash_jax"])
    parser.add_argument("--rope_type", type=str, default="llama3.1",
                        choices=["default", "llama3.1", "yarn"])
    parser.add_argument("--fused_qkv", type=str2bool, default=False,
                        help="Fuse QKV projections for faster attention.")
    parser.add_argument("--fused_mlp", type=str2bool, default=False,
                        help="Fuse MLP projections for faster FFN.")
    parser.add_argument("--float32_qk_product", type=str2bool, default=True,
                        help="Compute QK product in float32 for stability.")
    parser.add_argument("--float32_logits", type=str2bool, default=True,
                        help="Cast attention logits to float32 before softmax.")

    # === Data Configuration ===
    parser.add_argument("--use_book_data", type=str2bool, default=True)
    parser.add_argument("--use_simple_book", type=str2bool, default=False)
    parser.add_argument("--book_transform", type=str2bool, default=True)
    parser.add_argument("--book_depth", type=int, default=500)
    parser.add_argument("--token_mode", type=int, choices=[22, 24], default=24)
    parser.add_argument("--msg_seq_len", type=int, default=500)
    parser.add_argument("--masking", type=str, default='none',
                        choices=['causal', 'random', 'last_pos', 'none'])
    parser.add_argument("--merging", type=str, default='padded',
                        choices=['projected', 'padded'])

    # === Training Configuration ===
    parser.add_argument("--global_bsz", type=int, default=16)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_base", type=float, default=0.0003)
    parser.add_argument("--lr_factor", type=float, default=1.0)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--warmup_end", type=int, default=1)
    parser.add_argument("--cosine_anneal", type=str2bool, default=True)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--p_dropout", type=float, default=0.0)

    # === Optimization ===
    parser.add_argument("--prenorm", type=str2bool, default=True)
    parser.add_argument("--batchnorm", type=str2bool, default=False)
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    parser.add_argument("--mode", type=str, default="none",
                        choices=["none", "pool", "last", "ema"])

    # === DataLoader ===
    parser.add_argument("--n_data_workers", type=int, default=12)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=6)
    parser.add_argument("--persistent_workers", type=str2bool, default=True)
    parser.add_argument("--shuffle_train", type=str2bool, default=True)
    parser.add_argument("--random_offsets_train", type=str2bool, default=True)

    # === Checkpointing ===
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--restore_step", type=int, default=None)
    parser.add_argument("--checkpoint_every_n_steps", type=str, default="auto")
    parser.add_argument("--max_job_hours", type=float, default=24.0)
    parser.add_argument("--save_before_timeout_minutes", type=int, default=30)

    # === Misc ===
    parser.add_argument("--jax_seed", type=int, default=42)
    parser.add_argument("--debug_loading", type=str2bool, default=False)
    parser.add_argument("--enable_profiler", type=str2bool, default=False)
    parser.add_argument("--curtail_epochs", type=int, default=None)
    parser.add_argument("--debug_overfit", type=str2bool, default=False)
    parser.add_argument("--log_ce_tables", type=str2bool, default=False)
    parser.add_argument("--enable_goodput_monitor", type=str2bool, default=True)
    parser.add_argument("--ignore_times", type=str2bool, default=False)
    parser.add_argument("--monitor_step_loss", type=str2bool, default=True,
                        help="Enable step-level loss synchronization (slower due to host-device barriers).")


    args = parser.parse_args()

    # ============================================
    # Apply Model Preset
    # ============================================
    if args.model_preset is not None:
        preset = MODEL_PRESETS[args.model_preset]
        print(f"[*] Using LOBMAX model preset: {args.model_preset}")
        for key, value in preset.items():
            if key == "micro_bsz":
                continue
            setattr(args, key, value)
        micro_bsz = preset["micro_bsz"]
        args.global_bsz = micro_bsz * args.num_devices
        print(f"    d_model={args.d_model}, n_layers={args.n_layers}, num_heads={args.num_heads}")
        print(f"    mlp_dim={args.mlp_dim}")
        print(f"    micro_bsz={micro_bsz}/GPU × {args.num_devices} devices = {args.global_bsz} (global_bsz)")
        print(f"    lr_base={args.lr_base}")

    # Override global_bsz with PER_GPU_BSZ environment variable (if set)
    # This allows sweeping batch sizes without changing the preset code
    if 'PER_GPU_BSZ' in os.environ:
        micro_bsz = int(os.environ['PER_GPU_BSZ'])
        args.global_bsz = micro_bsz * args.num_devices
        print(f"[*] Overriding with PER_GPU_BSZ={micro_bsz}")
        print(f"    New global_bsz = {args.global_bsz}")


    # ============================================
    # Fill in defaults
    # ============================================
    if args.num_kv_heads is None:
        args.num_kv_heads = args.num_heads
    if args.head_dim is None:
        args.head_dim = args.d_model // args.num_heads
    if args.mlp_dim is None:
        args.mlp_dim = 4 * args.d_model

    # Set environment
    os.environ['USE_BF16'] = '1' if args.use_bf16 else '0'
    args.backend = 'transformer'  # Force Transformer backend

    # Parse checkpoint interval
    if args.checkpoint_every_n_steps.lower() == "auto":
        args.checkpoint_every_n_steps = "auto"
    else:
        args.checkpoint_every_n_steps = int(args.checkpoint_every_n_steps)

    print(f"\n[*] LOBMAX Configuration:")
    print(f"    d_model={args.d_model}, n_layers={args.n_layers}")
    print(f"    num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}, head_dim={args.head_dim}")
    print(f"    mlp_dim={args.mlp_dim}")
    print(f"    attention_kernel={args.attention_kernel}, rope_type={args.rope_type}")
    print(f"    fused_qkv={args.fused_qkv}, fused_mlp={args.fused_mlp}")
    print(f"    float32_qk_product={args.float32_qk_product}, float32_logits={args.float32_logits}")
    print(f"    Message layers={args.n_message_layers}, Book pre/post={args.n_book_pre_layers}/{args.n_book_post_layers}")
    if args.num_experts > 1:
        print(f"    [MoE] num_experts={args.num_experts}, num_experts_per_tok={args.num_experts_per_tok}")
        print(f"    [MoE] sparse_matmul={args.sparse_matmul}, megablox={args.megablox}, capacity_factor={args.capacity_factor}")

    # ============================================
    # JAX Distributed Initialization
    # ============================================
    import jax
    from jax.experimental import multihost_utils

    if is_slurm_multi_node or os.environ.get('JAX_COORDINATOR_ADDRESS'):
        coord = os.environ.get('JAX_COORDINATOR_ADDRESS')
        pid = int(os.environ.get('JAX_PROCESS_INDEX', os.environ.get('SLURM_PROCID', '0')))
        pcnt = int(os.environ.get('JAX_PROCESS_COUNT', os.environ.get('SLURM_NNODES', '1')))

        if not coord:
            raise RuntimeError('JAX_COORDINATOR_ADDRESS is not set for multi-node run')

        cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cvd and cvd != '-1':
            try:
                n_local = len([d for d in cvd.split(',') if d.strip() != ''])
            except Exception:
                n_local = 1
        else:
            n_local = 1
        local_device_ids = list(range(n_local))

        print(f"\n[*] Initializing JAX distributed: coord={coord}, pid={pid}, pcnt={pcnt}")
        jax.distributed.initialize(
            coordinator_address=coord,
            num_processes=pcnt,
            process_id=pid,
            local_device_ids=local_device_ids
        )

        is_distributed = True
        process_index = jax.process_index()
        process_count = jax.process_count()
        args.num_devices = jax.local_device_count()

        print(f"[*] JAX distributed mode: Process {process_index}/{process_count}")
        print(f"    GPUs per process: {args.num_devices}")

        multihost_utils.sync_global_devices("jax_distributed_init")
    else:
        is_distributed = False
        process_index = 0
        process_count = 1
        print(f"\n[*] Single machine mode")

    args.is_distributed = is_distributed
    args.process_index = process_index
    args.process_count = process_count

    # ============================================
    # Start Training
    # ============================================
    import torch
    torch.multiprocessing.set_start_method('spawn')

    from lob.sharding_utils import initialize_mesh
    num_total_devices = jax.device_count()
    initialize_mesh(num_total_devices)

    from lobmax.train import train_lobmax
    train_lobmax(args)

    # Clean shutdown
    if is_distributed:
        multihost_utils.sync_global_devices("end-of-train")
        jax.distributed.shutdown()
