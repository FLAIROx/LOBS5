# CAVE: only for debugging purposes
import os

# === XLA Memory Debugging Flags ===
# Uncomment to enable HLO dumping for memory analysis
# os.environ["XLA_FLAGS"] = (
#     "--xla_dump_hlo_as_text "
#     "--xla_dump_to=/tmp/xla_dump "
#     "--xla_dump_hlo_snapshots"
# )

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




# # I do not know if this makes the  stats = devices[0].memory_stats() returns none
# if __name__ == "__main__":
# 	pass
# else:
# 	# Forces all generated worker processes to not run on GPU.
# 	#  Required at this high level, because the init func in the 
# 	# worker spawn interface happens after init. of the CUDA process. 
# 	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 	os.environ["JAX_PLATFORMS"] = "cpu"
# # I do not know if this makes the  stats = devices[0].memory_stats() returns none, so I comment out

from lob.dataloading import Datasets

if __name__ == "__main__":
 
	# ······· choice 1 ······· STR
	import time
	time.sleep(1)
	# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.9"
	# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
	time.sleep(1)
	# ······· choice 1 ······· END
 
	# ······· choice 2 ······· STR
	# allocate and de-allocate memory as needed (SLOW)
	# NOTE: platform allocator does NOT support memory_stats() API!
	# Commenting out to enable GPU memory profiling
	# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
	# ······· choice 2 ······· END
 
 
	# ······· choice 3 ······· STR
	os.environ["JAX_DISABLE_JIT"] = "1"  # 完全关闭JIT编译
	# ······· choice 3 ······· END

	#physical_devices = tf.config.list_physical_devices('GPU')
	#tf.config.experimental.set_memory_growth(physical_devices[0], True)
	#tf.config.experimental.set_visible_devices([], "GPU")
 

	os.environ["NCCL_TIMEOUT"] = "600"  # 10 minutes
	os.environ["NCCL_IB_DISABLE"] = "0"  # Disable InfiniBand if not used
	os.environ["NCCL_P2P_DISABLE"] = "0"  # Disable peer-to-peer if causing issues
 
 
	os.environ["XLA_FLAGS"] = "--xla_gpu_enable_while_loop_unrolling=false"


	import argparse
	from s5.utils.util import str2bool
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


	parser = argparse.ArgumentParser()

	parser.add_argument("--USE_WANDB", type=str2bool, default=True,
						help="log with wandb?")
	parser.add_argument("--wandb_project", type=str, default="LOBS5v2",
						help="wandb project name")
	parser.add_argument("--wandb_entity", type=str, default="sasrey",
						help="wandb entity name, e.g. username")
	parser.add_argument("--dir_name", type=str, default='./data/LOBS5v2Cached/',
						help="name of directory where data is cached")
	parser.add_argument("--test_dir_name", type=str, default=None,
						help="separate test data directory (if None, split from train data)")
	parser.add_argument("--data_mode", type=str, choices=['preproc', 'encoded'],
						default='preproc',
						help="data loading mode: 'preproc' (encode on-the-fly) or 'encoded' (load pre-encoded data)")
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
	parser.add_argument("--restore", type=str,
		     			help="if given restore from given checkpoint dir")
	parser.add_argument("--restore_step", type=int)
	parser.add_argument("--msg_seq_len", type=int, default=500,  # 500
						help="How many past messages to include in each sample")
	parser.add_argument("--n_data_workers", type=int, default=0,
		     			help="number of workers used in DataLoader")

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
	
	args = parser.parse_args()


	import torch
	torch.multiprocessing.set_start_method('spawn')

	from lob.train import train
	#import tensorflow as tf
	# import jax	
	# import cProfile

	#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
	train(args)
	#cProfile.run('train(parser.parse_args())')
