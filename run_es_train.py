# ES-LOBS5 Training Entry Point
# ============================================================================
# Similar to run_train.py, this is the main entry point for ES training.
# All ES production jobs should call this script.
# ============================================================================

import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Environment setup (before JAX import)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime

    # ============================================================================
    # ES Model Presets: Pre-configured ES Training Settings
    # ============================================================================
    # Similar to run_train.py MODEL_PRESETS, but for ES training configurations.
    #
    # Key parameters:
    #   - n_steps: Steps per episode (trading decisions per episode)
    #   - background_msgs_per_step: Market messages between policy actions
    #   - task_size: Shares to execute (smaller = easier task)
    #   - noiser: ES algorithm (eggroll, eggrollbs, open_es)
    #   - file_idx: Data file index (for data diversity across runs)
    #   - group_size: For EggRollBS baseline subtraction (0 = disabled)
    #
    # Usage:
    #   MODEL_PRESET=L1 python run_es_train.py
    #   python run_es_train.py --model_preset L1
    #   python run_es_train.py --model_preset L1 --task_size 20  # override
    # ============================================================================
    ES_PRESETS = {
        # D series: Baseline config (large task, many bg_msgs)
        "D5": {
            "n_steps": 10, "background_msgs_per_step": 100, "task_size": 500,
            "noiser": "eggroll", "file_idx": 0, "group_size": 0,
            "wandb_project": "ES-LOBS5-D",
        },
        # G series: Same as D5 (ablation baseline)
        "G5": {
            "n_steps": 10, "background_msgs_per_step": 100, "task_size": 500,
            "noiser": "eggroll", "file_idx": 0, "group_size": 0,
            "wandb_project": "ES-LOBS5-G",
        },
        # H series: More steps, fewer bg_msgs (longer episodes)
        "H5": {
            "n_steps": 100, "background_msgs_per_step": 10, "task_size": 500,
            "noiser": "eggroll", "file_idx": 0, "group_size": 0,
            "wandb_project": "ES-LOBS5-H",
        },
        # I series: Smaller task_size
        "I5": {
            "n_steps": 100, "background_msgs_per_step": 10, "task_size": 100,
            "noiser": "eggroll", "file_idx": 0, "group_size": 0,
            "wandb_project": "ES-LOBS5-I",
        },
        # J series: Data diversity sweep (different file_idx)
        "J1": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 500, "noiser": "eggroll", "file_idx": 0, "group_size": 0, "wandb_project": "ES-LOBS5-J"},
        "J2": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 500, "noiser": "eggroll", "file_idx": 50, "group_size": 0, "wandb_project": "ES-LOBS5-J"},
        "J3": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 500, "noiser": "eggroll", "file_idx": 100, "group_size": 0, "wandb_project": "ES-LOBS5-J"},
        "J4": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 500, "noiser": "eggroll", "file_idx": 150, "group_size": 0, "wandb_project": "ES-LOBS5-J"},
        # K series: Smaller task_size + data diversity
        "K1": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 100, "noiser": "eggroll", "file_idx": 0, "group_size": 0, "wandb_project": "ES-LOBS5-K"},
        "K2": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 100, "noiser": "eggroll", "file_idx": 50, "group_size": 0, "wandb_project": "ES-LOBS5-K"},
        "K3": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 100, "noiser": "eggroll", "file_idx": 100, "group_size": 0, "wandb_project": "ES-LOBS5-K"},
        "K4": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 100, "noiser": "eggroll", "file_idx": 150, "group_size": 0, "wandb_project": "ES-LOBS5-K"},
        # L series: EggRollBS with baseline subtraction (very small task)
        "L1": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 10, "noiser": "eggrollbs", "file_idx": 0, "group_size": 8, "wandb_project": "ES-LOBS5-L"},
        "L2": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 10, "noiser": "eggrollbs", "file_idx": 50, "group_size": 8, "wandb_project": "ES-LOBS5-L"},
        "L3": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 10, "noiser": "eggrollbs", "file_idx": 100, "group_size": 8, "wandb_project": "ES-LOBS5-L"},
        "L4": {"n_steps": 10, "background_msgs_per_step": 10, "task_size": 10, "noiser": "eggrollbs", "file_idx": 150, "group_size": 8, "wandb_project": "ES-LOBS5-L"},
    }

    # ============================================================================
    # Default Paths
    # ============================================================================
    DEFAULT_LOBS5_CHECKPOINT = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
    DEFAULT_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
    DEFAULT_OUTPUT_DIR = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints"

    # ============================================================================
    # Argument Parser
    # ============================================================================
    parser = argparse.ArgumentParser(
        description="ES-LOBS5 Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ES Model Presets:
  D5/G5    - n_steps=10, bg_msgs=100, task_size=500, eggroll
  H5       - n_steps=100, bg_msgs=10, task_size=500, eggroll
  I5       - n_steps=100, bg_msgs=10, task_size=100, eggroll
  J1-J4    - n_steps=10, bg_msgs=10, task_size=500, eggroll (data diversity)
  K1-K4    - n_steps=10, bg_msgs=10, task_size=100, eggroll (data diversity)
  L1-L4    - n_steps=10, bg_msgs=10, task_size=10, eggrollbs (baseline subtraction)

Examples:
  MODEL_PRESET=L1 python run_es_train.py
  python run_es_train.py --model_preset L1 --task_size 20
  python run_es_train.py --n_steps 50 --noiser eggroll
        """
    )

    # Model Preset
    parser.add_argument("--model_preset", type=str, default=None,
                        choices=list(ES_PRESETS.keys()),
                        help="Pre-configured ES training settings")

    # WandB
    parser.add_argument("--USE_WANDB", type=lambda x: x.lower() == 'true', default=True,
                        help="Log with wandb?")
    parser.add_argument("--wandb_project", type=str, default="ES-LOBS5",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="kang-oxford",
                        help="WandB entity/username")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="WandB run name (auto-generated if not specified)")

    # Paths
    parser.add_argument("--lobs5_checkpoint", type=str, default=DEFAULT_LOBS5_CHECKPOINT,
                        help="Path to LOBS5 model checkpoint")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to LOBSTER data directory (preproc format)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for ES checkpoints")

    # ES Algorithm
    parser.add_argument("--noiser", type=str, default="eggroll",
                        choices=["eggroll", "eggrollbs", "open_es", "sparse", "noop"],
                        help="ES algorithm: eggroll (default), eggrollbs (with baseline subtraction)")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Perturbation noise std")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="ES learning rate")
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="LoRA adapter rank")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--group_size", type=int, default=0,
                        help="Group size for EggRollBS (0=disabled)")

    # Training Scale
    parser.add_argument("--n_perturbations", type=int, default=1024,
                        help="Population size (must be divisible by num_devices)")
    parser.add_argument("--n_epochs", type=int, default=1000,
                        help="Max training epochs")
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Steps per episode (trading decisions)")
    parser.add_argument("--n_warmup_msgs", type=int, default=500,
                        help="Warmup messages for order book initialization")
    parser.add_argument("--background_msgs_per_step", type=int, default=10,
                        help="Background market messages between policy actions")

    # Token Mode
    parser.add_argument("--token_mode", type=int, default=24, choices=[22, 24],
                        help="Token encoding mode (must match checkpoint)")
    parser.add_argument("--background_mode", type=str, default="historical_replay",
                        choices=["historical_replay", "world_model"],
                        help="Background message generation mode")

    # Task Configuration
    parser.add_argument("--task", type=str, default="sell", choices=["sell", "buy"],
                        help="Execution task type")
    parser.add_argument("--task_size", type=int, default=500,
                        help="Shares to execute (smaller = easier)")
    parser.add_argument("--tick_size", type=int, default=100,
                        help="Tick size in cents")

    # Data Selection
    parser.add_argument("--file_idx", type=int, default=0,
                        help="Data file index (for data diversity)")

    # Checkpointing
    parser.add_argument("--checkpoint_every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_training_hours", type=float, default=23.5,
                        help="Max training time in hours (for 24h jobs)")

    args = parser.parse_args()

    # ============================================================================
    # Apply Model Preset (if specified)
    # ============================================================================
    # Also check MODEL_PRESET environment variable
    model_preset = args.model_preset or os.environ.get('MODEL_PRESET')

    if model_preset is not None:
        if model_preset not in ES_PRESETS:
            raise ValueError(f"Unknown preset: {model_preset}. Available: {list(ES_PRESETS.keys())}")
        preset = ES_PRESETS[model_preset]
        print(f"[*] Using ES preset: {model_preset}")
        for key, value in preset.items():
            if hasattr(args, key):
                setattr(args, key, value)
        print(f"    n_steps={args.n_steps}, bg_msgs={args.background_msgs_per_step}, "
              f"task_size={args.task_size}, noiser={args.noiser}")
        print(f"    file_idx={args.file_idx}, group_size={args.group_size}")
        print(f"    wandb_project={args.wandb_project}")

    # Override with environment variables (like PER_GPU_BSZ in run_train.py)
    if 'TASK_SIZE' in os.environ:
        args.task_size = int(os.environ['TASK_SIZE'])
        print(f"[*] Overriding: task_size={args.task_size}")
    if 'N_STEPS' in os.environ:
        args.n_steps = int(os.environ['N_STEPS'])
        print(f"[*] Overriding: n_steps={args.n_steps}")
    if 'FILE_IDX' in os.environ:
        args.file_idx = int(os.environ['FILE_IDX'])
        print(f"[*] Overriding: file_idx={args.file_idx}")

    # Set replay_data_path = data_dir (for ESTrainer compatibility)
    args.replay_data_path = args.data_dir

    # Generate checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preset_suffix = f"_{model_preset}" if model_preset else ""
    args.checkpoint_dir = os.path.join(
        args.output_dir,
        f"es{preset_suffix}_{timestamp}"
    )

    # Generate WandB run name if not specified
    if args.wandb_name is None:
        args.wandb_name = f"{model_preset or 'custom'}_steps{args.n_steps}_bg{args.background_msgs_per_step}_task{args.task_size}"
        if args.noiser == "eggrollbs":
            args.wandb_name += "_eggrollbs"

    # ============================================================================
    # Import JAX (after environment setup)
    # ============================================================================
    import jax
    import jax.numpy as jnp
    import numpy as np

    print("=" * 70)
    print("ES-LOBS5 Production Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Preset: {model_preset or 'custom'}")
    print("-" * 70)
    print(f"Config:")
    print(f"  noiser:                   {args.noiser}")
    print(f"  n_perturbations:          {args.n_perturbations}")
    print(f"  n_steps:                  {args.n_steps}")
    print(f"  background_msgs_per_step: {args.background_msgs_per_step}")
    print(f"  task_size:                {args.task_size}")
    print(f"  file_idx:                 {args.file_idx}")
    print(f"  group_size:               {args.group_size}")
    print(f"  data_dir:                 {args.data_dir}")
    print(f"  checkpoint_dir:           {args.checkpoint_dir}")
    print("=" * 70)

    # ============================================================================
    # Initialize WandB
    # ============================================================================
    if args.USE_WANDB:
        os.environ['WANDB_MODE'] = 'online'
        os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
        os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'True'
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
            tags=[model_preset or 'custom', 'production', '24h']
        )
        print(f"WandB: {run.name} ({run.url})")
    else:
        import wandb
        run = wandb.init(mode='disabled')

    # ============================================================================
    # Create Output Directory
    # ============================================================================
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ============================================================================
    # Initialize ESTrainer
    # ============================================================================
    print("\n[1/3] Initializing ESTrainer...")
    t0 = time.time()
    from es_lobs5.training.es_trainer import ESTrainer
    trainer = ESTrainer(args)
    print(f"      Init time: {time.time() - t0:.1f}s")

    # ============================================================================
    # Create Initial State
    # ============================================================================
    print("\n[2/3] Creating initial simulation state...")
    t0 = time.time()
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    print(f"      State time: {time.time() - t0:.1f}s")
    print(f"      msg_history shape: {initial_msg_history.shape}")

    # ============================================================================
    # Training Loop
    # ============================================================================
    print("\n[3/3] Starting training loop...")
    print("=" * 70)

    key = jax.random.PRNGKey(args.seed)
    training_start = time.time()
    max_training_time = args.max_training_hours * 3600

    best_fitness = -float('inf')
    fitness_history = []

    for epoch in range(args.n_epochs):
        # Check time limit
        elapsed = time.time() - training_start
        if elapsed > max_training_time:
            print(f"\n[!] Time limit reached ({elapsed/3600:.1f}h). Stopping.")
            break

        epoch_start = time.time()
        key, epoch_key = jax.random.split(key)

        # Train one epoch
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )

        # Collect metrics
        mf = float(mean_fitness.block_until_ready())
        sf = float(jnp.std(fitnesses))
        max_f = float(jnp.max(fitnesses))
        min_f = float(jnp.min(fitnesses))
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start

        fitness_history.append(mf)
        if mf > best_fitness:
            best_fitness = mf
            best_marker = " *BEST*"
        else:
            best_marker = ""

        # Console log
        print(f"Epoch {epoch+1:4d} | Fitness: {mf:8.2f} +/- {sf:6.2f} | "
              f"Time: {epoch_time:5.1f}s | Total: {total_elapsed/3600:.2f}h{best_marker}")

        # WandB log
        if args.USE_WANDB:
            wandb.log({
                'epoch': epoch + 1,
                'fitness/mean': mf,
                'fitness/std': sf,
                'fitness/max': max_f,
                'fitness/min': min_f,
                'fitness/best': best_fitness,
                'time/epoch_seconds': epoch_time,
                'time/total_hours': total_elapsed / 3600,
            }, step=epoch + 1)

        # Checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            np.save(os.path.join(args.checkpoint_dir, 'fitness_history.npy'),
                    np.array(fitness_history))

    # ============================================================================
    # Training Complete
    # ============================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total epochs:   {len(fitness_history)}")
    print(f"  Best fitness:   {best_fitness:.4f}")
    print(f"  Final fitness:  {fitness_history[-1] if fitness_history else 0:.4f}")
    print(f"  Total time:     {(time.time() - training_start)/3600:.2f}h")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 70)

    # Save final results
    np.save(os.path.join(args.checkpoint_dir, 'fitness_history.npy'),
            np.array(fitness_history))

    if args.USE_WANDB:
        wandb.finish()
