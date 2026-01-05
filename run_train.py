# CAVE: only for debugging purposes
import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=48'
# no GPU use at all
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# allocate and de-allocate memory as needed (SLOW)
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# TODO: change this if num_devices changes (is less than all of the available ones11)
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
# os.environ["NCCL_DEBUG"]="INFO"

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
if __name__ == "__main__":
	pass
else:
	# Forces all generated worker processes to not run on GPU.
	#  Required at this high level, because the init func in the 
	# worker spawn interface happens after init. of the CUDA process. 
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	os.environ["JAX_PLATFORMS"] = "cpu"

from lob.dataloading import Datasets

if __name__ == "__main__":
	import argparse
	from s5.utils.util import str2bool
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.9"
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
	os.environ["NCCL_TIMEOUT"] = "600"  # 10 minutes
	os.environ["NCCL_IB_DISABLE"] = "0"  # Disable InfiniBand if not used
	os.environ["NCCL_P2P_DISABLE"] = "0"  # Disable peer-to-peer if causing issues

	#physical_devices = tf.config.list_physical_devices('GPU')
	#tf.config.experimental.set_memory_growth(physical_devices[0], True)
	#tf.config.experimental.set_visible_devices([], "GPU")

	# ============================================
	# Model Presets: Pre-configured Model Settings
	# ============================================
	MODEL_PRESETS = {
		"2.5B-Wide": {
			"d_model": 5120, "n_layers": 32, "blocks": 64, "ssm_size_base": 5120, "bsz": 1,
			"ssm_lr_base": 0.00003, "lr_factor": 1,
			"wandb_project": "lobs5-2.5B-wide-d5120"
		},
		"2.5B-Deep": {
			"d_model": 4096, "n_layers": 48, "blocks": 64, "ssm_size_base": 4096, "bsz": 1,
			"ssm_lr_base": 0.00003, "lr_factor": 1,
			"wandb_project": "lobs5-2.5B-deep-d4096"
		},
		"2.2B-Wide": {
			"d_model": 4704, "n_layers": 32, "blocks": 56, "ssm_size_base": 4704, "bsz": 1,
			"ssm_lr_base": 0.000032, "lr_factor": 1,
			"wandb_project": "lobs5-2.2B-wide-d4704"
		},
		"2.2B-Deep": {
			"d_model": 4352, "n_layers": 48, "blocks": 64, "ssm_size_base": 4352, "bsz": 1,
			"ssm_lr_base": 0.000032, "lr_factor": 1,
			"wandb_project": "lobs5-2.2B-deep-d4352"
		},
		"2B-Wide": {
			"d_model": 4480, "n_layers": 32, "blocks": 56, "ssm_size_base": 4480, "bsz": 4,
			"ssm_lr_base": 0.000035, "lr_factor": 1,
			"wandb_project": "lobs5-2B-wide-d4480"
		},
		"2B-Deep": {
			"d_model": 4096, "n_layers": 44, "blocks": 64, "ssm_size_base": 4096, "bsz": 1,
			"ssm_lr_base": 0.000035, "lr_factor": 1,
			"wandb_project": "lobs5-2B-deep-d4096"
		},
		"1.8B-Wide": {
			"d_model": 4144, "n_layers": 32, "blocks": 56, "ssm_size_base": 4144, "bsz": 4,
			"ssm_lr_base": 0.00004, "lr_factor": 1,
			"wandb_project": "lobs5-1.8B-wide-d4144"
		},
		"1.8B-Deep": {
			"d_model": 3840, "n_layers": 40, "blocks": 60, "ssm_size_base": 3840, "bsz": 1,
			"ssm_lr_base": 0.00004, "lr_factor": 1,
			"wandb_project": "lobs5-1.8B-deep-d3840"
		},
		"1.4B": {
			"d_model": 3584, "n_layers": 32, "blocks": 56, "ssm_size_base": 3584, "bsz": 2,
			"ssm_lr_base": 0.00005, "lr_factor": 1,
			"wandb_project": "lobs5-1.4B-d3584"
		},
		"1B": {
			"d_model": 3072, "n_layers": 32, "blocks": 48, "ssm_size_base": 3072, "bsz": 2,
			"ssm_lr_base": 0.00010, "lr_factor": 1,
			"wandb_project": "lobs5-1B-d3072"
		},
		"360M": {
			"d_model": 2048, "n_layers": 24, "blocks": 32, "ssm_size_base": 2048, "bsz": 2,
			"ssm_lr_base": 0.00020, "lr_factor": 1,
			"wandb_project": "lobs5-300M-d2048"
		},
		"55M": {
			"d_model": 1024, "n_layers": 12, "blocks": 16, "ssm_size_base": 1024, "bsz": 12,
			"ssm_lr_base": 5e-5, "lr_factor": 1,
			"wandb_project": "lobs5-55M-d1024"
		},
	}

	parser = argparse.ArgumentParser()

	parser.add_argument("--USE_WANDB", type=str2bool, default=True,
						help="log with wandb?")
	parser.add_argument("--wandb_project", type=str, default="LOBS5v2",
						help="wandb project name")
	parser.add_argument("--wandb_entity", type=str, default="sasrey",
						help="wandb entity name, e.g. username")
	parser.add_argument("--dir_name", type=str, default='./data/LOBS5v2Cached/',
						help="name of directory where data is cached")
	parser.add_argument("--dataset", type=str, choices=Datasets.keys(),
						default='lobster-prediction',
						help="dataset name")
	parser.add_argument("--masking", type=str, choices={'causal', 'random','last_pos','none'},
						default='causal',  # random
						help="causal, random or last position masking of sequences")
	parser.add_argument("--use_book_data", type=str2bool, default=False,
		     			help="use book data in addition to message data")
	parser.add_argument("--merging", type=str, choices={'projected', 'padded'},
						default='projected', 
						help="Method for merging the book model with the message model. Cannot use RNN mode with projected mode.")
	parser.add_argument("--use_simple_book", type=str2bool, default=False,
		     			help="use raw price (-p0) and volume series instead of 'volume image representation'")
	parser.add_argument("--book_transform", type=str2bool, default=False,
		     			help="transform loaded book data to volume image repr. in dataloader")
	parser.add_argument("--book_depth", type=int, default=500,
		     			help="number of tick levels to use in book data [if book_transform=True]")
	parser.add_argument("--token_mode", type=int, choices=[22, 24],
		     			default=22,
		     			help="token encoding mode: 22 (default, base-10000 size) or 24 (base-100 size)")
	parser.add_argument("--test_dir_name", type=str, default=None,
		     			help="directory for test data (optional, uses --dir_name if not specified)")
	parser.add_argument("--restore", type=str,
		     			help="if given restore from given checkpoint dir")
	parser.add_argument("--restore_step", type=int)
	parser.add_argument("--msg_seq_len", type=int, default=500,  # 500
						help="How many past messages to include in each sample")
	parser.add_argument("--n_data_workers", type=int, default=0,
		     			help="number of workers used in DataLoader")
	parser.add_argument("--pin_memory", type=str2bool, default=True,
		     			help="enable pin_memory for DataLoader (faster GPU transfer)")
	parser.add_argument("--prefetch_factor", type=int, default=2,
		     			help="number of batches to prefetch per worker")
	parser.add_argument("--persistent_workers", type=str2bool, default=False,
		     			help="keep DataLoader workers alive between epochs")
	parser.add_argument("--enable_goodput_monitor", type=str2bool, default=False,
		     			help="enable goodput monitoring (data loading vs compute time)")

	# Model Preset (overrides individual model parameters if specified)
	parser.add_argument("--model_preset", type=str, default=None,
						choices=list(MODEL_PRESETS.keys()),
						help="Pre-configured model settings: 2.5B-Wide, 2.5B-Deep, 2B-Wide, 2B-Deep, 1.8B-Wide, 1.8B-Deep, 1.4B, 1B, 300M, 55M. Overrides d_model, n_layers, blocks, ssm_size_base, bsz, ssm_lr_base, lr_factor, wandb_project")

	# Model Parameters
	parser.add_argument("--n_message_layers", type=int, default=2,  # 2
						help="Number of layers after fusing message and book data")
	parser.add_argument("--n_book_pre_layers", type=int, default=1,  # 1
						help="Number of layers taking in raw book data (before projecting dimensions)")
	parser.add_argument("--n_book_post_layers", type=int, default=1,  # 1
						help="Number of book seq layers after projecting book data dimensions")
	parser.add_argument("--n_layers", type=int, default=6,  #6
						help="Number of layers after fusing message and book data")
	parser.add_argument("--d_model", type=int, default=32,  #128, 32, 16
						help="Number of features, i.e. H, "
							 "dimension of layer inputs/outputs")
	parser.add_argument("--ssm_size_base", type=int, default=32,  # 256
						help="SSM Latent size, i.e. P")
	parser.add_argument("--blocks", type=int, default=8,  # 8, 4
						help="How many blocks, J, to initialize with")
	parser.add_argument("--C_init", type=str, default="trunc_standard_normal",
						choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
						help="Options for initialization of C: \\"
							 "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ " \
							 "lecun_normal sample from lecun normal, then multiply by V\\ " \
							 "complex_normal: sample directly from complex standard normal")
	parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear"])
	parser.add_argument("--mode", type=str, default="none", choices=["none","pool", "last","ema"],
						help="options: (for classification tasks) \\" \
							 " none: no aggregation, raw output at decoder stage \\" \
							 " pool: mean pooling \\" \
							 "last: take last element \\" \
							 "ema : take exponential moving avg across all")
	parser.add_argument("--activation_fn", default="half_glu1", type=str,
						choices=["full_glu", "half_glu1", "half_glu2", "gelu"])
	parser.add_argument("--conj_sym", type=str2bool, default=True,
						help="whether to enforce conjugate symmetry")
	parser.add_argument("--clip_eigs", type=str2bool, default=False,
						help="whether to enforce the left-half plane condition")
	parser.add_argument("--bidirectional", type=str2bool, default=False,  #False,
						help="whether to use bidirectional model")
	parser.add_argument("--dt_min", type=float, default=0.001,
						help="min value to sample initial timescale params from")
	parser.add_argument("--dt_max", type=float, default=0.1,
						help="max value to sample initial timescale params from")

	# Optimization Parameters
	parser.add_argument("--prenorm", type=str2bool, default=True,
						help="True: use prenorm, False: use postnorm")
	parser.add_argument("--batchnorm", type=str2bool, default=True,
						help="True: use batchnorm, False: use layernorm")
	parser.add_argument("--bn_momentum", type=float, default=0.95,
						help="batchnorm momentum")
	parser.add_argument("--bsz", type=int, default=16, #64, (max 16 with full size)
						help="batch size")
	parser.add_argument("--num_devices", type=int, default=1,
		     			help="number of devices (GPUs) to use")
	parser.add_argument("--epochs", type=int, default=100,  #100, 20
						help="max number of epochs")
	parser.add_argument("--early_stop_patience", type=int, default=1000,
						help="number of epochs to continue training when val loss plateaus")
	parser.add_argument("--ssm_lr_base", type=float, default=1e-3,
						help="initial ssm learning rate")
	parser.add_argument("--lr_factor", type=float, default=1,
						help="global learning rate = lr_factor*ssm_lr_base")
	parser.add_argument("--dt_global", type=str2bool, default=False,
						help="Treat timescale parameter as global parameter or SSM parameter")
	parser.add_argument("--lr_min", type=float, default=0,
						help="minimum learning rate")
	parser.add_argument("--cosine_anneal", type=str2bool, default=True,
						help="whether to use cosine annealing schedule")
	parser.add_argument("--warmup_end", type=int, default=1,
						help="epoch to end linear warmup")
	parser.add_argument("--lr_patience", type=int, default=1000000,
						help="patience before decaying learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--reduce_factor", type=float, default=0.8,
						help="factor to decay learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--p_dropout", type=float, default=0.0,
						help="probability of dropout")
	parser.add_argument("--weight_decay", type=float, default=0.05,
						help="weight decay value")
	parser.add_argument("--opt_config", type=str, default="standard", choices=['standard',
																			   'BandCdecay',
																			   'BfastandCdecay',
																			   'noBCdecay'],
						help="Opt configurations: \\ " \
			   "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\" \
	  	       "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\")
	parser.add_argument("--jax_seed", type=int, default=1919,
						help="seed randomness")
	parser.add_argument("--debug_loading", type=str2bool, default=False,
						help="Set flag to True to skip any training and just run the loading process.")
	parser.add_argument("--enable_profiler", type=str2bool, default=False,
					help="Set flag to True to use the TB profiler.")
	parser.add_argument("--curtail_epochs", type=int, default=None,
				help="End epoch after n steps. Default is None, never. ")
	parser.add_argument("--random_offsets_train", type=str2bool, default=True,
				help="Whether or not the training data is offset randomly at each epoch.")
	parser.add_argument("--shuffle_train", type=str2bool, default=True,
				help="Whether or not the training data shuffled.")
	parser.add_argument("--ignore_times", type=str2bool, default=False,
                    help="Ignore the loss due to predicting the time.")
	parser.add_argument("--debug_overfit", type=str2bool, default=False,
				help="Runs the training loop in overfit mode on a single batch of data. Validation and testing are from the same set. ")
	parser.add_argument("--log_ce_tables", type=str2bool, default=False,
				help="Logs the CE values on a per token level to wandb. Memory intensive.")
	parser.add_argument("--use_bf16", type=str2bool, default=True,
				help="Use BF16 mixed precision training")

	# ============================================================================
	# Step-Level Checkpointing for Long-Running Jobs (12.5-14h epochs, 24h max)
	# ============================================================================
	#
	# This system provides mid-epoch checkpoint saves and time-aware auto-save:
	#
	# Usage:
	#   # Time-based (auto mode, using wall clock):
	#   #   - WANDB LOSS LOGGING: EVERY 10 MINUTES
	#   #   - CHECKPOINT SAVING:  EVERY 30 MINUTES
	#   python run_train.py --model_preset 55M --checkpoint_every_n_steps auto
	#
	#   # Manual interval (every 5000 steps):
	#   python run_train.py --model_preset 55M --checkpoint_every_n_steps 5000
	#
	#   # Resume from mid-epoch checkpoint:
	#   python run_train.py --model_preset 55M \
	#       --restore checkpoints/[run_name]/ \
	#       --restore_step [global_step]
	#
	# What gets saved in each checkpoint (verified):
	#   - state.step      - Training step counter (affects LR schedule)
	#   - opt_state.mu    - Adam first moment (momentum) ~130 params
	#   - opt_state.nu    - Adam second moment ~130 params
	#   - params          - Model parameters, 133 keys
	#   - Total: 686 keys, 553 optimizer state (80%)
	#
	# Why optimizer state matters:
	#   - Preserves Adam momentum for smooth training continuation
	#   - LR schedule continues from correct step (no restart)
	#   - No "cold start" penalty when resuming
	#
	# Time-aware auto-save:
	#   - Monitors elapsed time vs max_job_hours
	#   - Auto-saves checkpoint when save_before_timeout_minutes remaining
	#   - Prints resume command on timeout exit
	#
	# ============================================================================
	parser.add_argument("--checkpoint_every_n_steps", type=str, default="auto",
				help="'auto': wandb every 10min, checkpoint every 30min. Integer: both at N steps. 0: disable.")
	parser.add_argument("--max_job_hours", type=float, default=24.0,
				help="Maximum job duration in hours (default: 24.0). Used for time-aware checkpointing.")
	parser.add_argument("--save_before_timeout_minutes", type=int, default=30,
				help="Save checkpoint this many minutes before max_job_hours timeout (default: 30).")
	parser.add_argument("--resume_from_step", type=int, default=None,
				help="When restoring, skip to this step within the epoch. Used for mid-epoch resume.")

	args = parser.parse_args()

	# ============================================
	# Apply Model Preset (if specified)
	# ============================================
	if args.model_preset is not None:
		preset = MODEL_PRESETS[args.model_preset]
		print(f"[*] Using model preset: {args.model_preset}")
		for key, value in preset.items():
			setattr(args, key, value)
		print(f"    d_model={args.d_model}, n_layers={args.n_layers}, blocks={args.blocks}, ssm_size_base={args.ssm_size_base}")
		print(f"    bsz={args.bsz}, ssm_lr_base={args.ssm_lr_base}, lr_factor={args.lr_factor}")
		print(f"    wandb_project={args.wandb_project}")

	# Override bsz with PER_GPU_BSZ environment variable (if set)
	# This allows batch size sweeps while using model presets
	# Note: bsz is the GLOBAL batch size (across all devices)
	# PER_GPU_BSZ specifies the batch size per GPU, so we multiply by num_devices
	if 'PER_GPU_BSZ' in os.environ:
		per_gpu_bsz = int(os.environ['PER_GPU_BSZ'])
		args.bsz = per_gpu_bsz * args.num_devices
		print(f"[*] Overriding with PER_GPU_BSZ={per_gpu_bsz} × {args.num_devices} devices = {args.bsz} (global bsz)")

	# Set BF16 environment variable based on command-line argument
	os.environ['USE_BF16'] = '1' if args.use_bf16 else '0'

	# Parse checkpoint_every_n_steps: "auto", "0", or integer
	if args.checkpoint_every_n_steps.lower() == "auto":
		args.checkpoint_every_n_steps = "auto"  # Keep as string, train.py will calculate
	else:
		args.checkpoint_every_n_steps = int(args.checkpoint_every_n_steps)

	import torch
	torch.multiprocessing.set_start_method('spawn')

	from lob.train import train
	#import tensorflow as tf
	# import jax	
	# import cProfile

	#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
	train(args)
	#cProfile.run('train(parser.parse_args())')




'''

     ┌─────────────────────────────────────────────────────────────────────────────────┐
     │                   JAX/GPU Data Loading Pipeline (Multi-Node)                    │
     └─────────────────────────────────────────────────────────────────────────────────┘

       ┌──────────┐      ┌──────────┐      ┌────────────┐      ┌──────────┐      ┌──────────┐
       │   DISK   │ ──▶  │   RAM    │ ──▶  │ Pinned RAM │ ──▶  │   VRAM   │ ──▶  │  TRAIN   │
       │   (IO)   │      │  (CPU)   │      │  (CPU)     │      │  (GPU)   │      │  (GPU)   │
       └──────────┘      └──────────┘      └────────────┘      └──────────┘      └──────────┘
             │                 │                  │                  │                 │
             ▼                 ▼                  ▼                  ▼                 ▼
           .npy          preprocess          page-locked        device_put       jit train_step
        mmap read        tokenize         cudaHostAlloc()     CUDA async DMA     (with shardings)
                         batching           non-swappable       np.split()       global sharding
                         shuffle                               to local GPUs

             │                 │                  │                  │
             └────────┬────────┘                  │                  │
                      │                           │                  │
                      ▼                           ▼                  ▼
         ┌────────────────────────────┐  ┌─────────────────┐  ┌─────────────────┐
         │      num_workers = N       │  │  pin_memory=T   │  │ prefetch_factor │
         │  ┌────┐ ┌────┐ ┌────┐      │  │                 │  │      = M        │
         │  │ W0 │ │ W1 │ │... │      │  │  Page-locked    │  │                 │
         │  └────┘ └────┘ └────┘      │  │  memory         │  │  Prefetch M     │
         │  N subprocesses parallel   │  │  No swap to     │  │  batches ahead  │
         │  preloading data           │  │  disk allowed   │  │  per worker     │
         └────────────────────────────┘  │  Faster DMA     │  └─────────────────┘
                      │                  └─────────────────┘
                      ▼
         ┌──────────────────────────────────────────────────────────────────────────────┐
         │   persistent_workers=T:                                                      │
         │   Keep worker processes alive across epochs. Avoid restart overhead          │
         └──────────────────────────────────────────────────────────────────────────────┘
         
'''