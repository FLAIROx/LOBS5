from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
from s5.dataloading import make_data_loader
from .lobster_dataloader import LOBSTER, LOBSTER_Dataset, LOBSTER_Sampler
# from lob.encoding import Message_Tokenizer



DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')
DATA_DIR = Path('../data/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[LOBSTER, DataLoader, DataLoader, DataLoader, Dict, int, int, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


def create_lobster_prediction_dataset(
		cache_dir: Union[str, Path] = DATA_DIR,
		seed: int = 42,
		mask_fn = LOBSTER_Dataset.no_mask,
		msg_seq_len: int = 500,
		global_bsz: int=128,
		use_book_data: bool = False,
		use_simple_book: bool = False,
		book_transform: bool = False,
		book_depth: int = 500,
		token_mode: int = 22,
		test_dir_name: Union[str, Path, None] = None,
		n_data_workers: int = 16,  # DATA CORE PARAMS: 0→12, utilize multi-core CPU (GH200: 72 cores)
		return_raw_msgs: bool = False,
		shuffle_train=True,
		rand_offset=True,
		debug_overfit=False,
		pin_memory: bool = True,
		prefetch_factor: int = 8,  # DATA CORE PARAMS: 2→6, larger prefetch buffer (memory allows)
		persistent_workers: bool = True,  # DATA CORE PARAMS: keep workers alive across epochs
	) -> ReturnType:
	""" 
	"""
	if debug_overfit:
		rand_offset= False
		shuffle_train= False


	print("[*] Generating LOBSTER Prediction Dataset from", cache_dir)
	from .lobster_dataloader import LOBSTER
	name = 'lobster'

	dataset_obj = LOBSTER(
		name,
		data_dir=cache_dir,
		mask_fn=mask_fn,
		msg_seq_len=msg_seq_len,
		use_book_data=use_book_data,
		use_simple_book=use_simple_book,
		book_transform=book_transform,
		book_depth=book_depth,
		token_mode=token_mode,
		test_data_dir=test_dir_name,
		n_cache_files=250,  # large number to keep everything in cache
		return_raw_msgs=return_raw_msgs,
		rand_offset=rand_offset,
		debug_overfit=debug_overfit,
	)
	dataset_obj.setup()
 
	# breakpoint()

	print("Using mask function:", mask_fn)

	# use sampler to only get individual samples and automatic batching from dataloader
	#trn_sampler = LOBSTER_Sampler(
	#		dataset_obj.dataset_train, n_files_shuffle=5, batch_size=1, seed=seed)
	
	trn_loader = create_lobster_train_loader(
		dataset_obj, seed, global_bsz, n_data_workers, reset_train_offsets=rand_offset, shuffle=shuffle_train,
		pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
	# NOTE: drop_last=True recompiles the model for a smaller batch size
	val_loader = make_data_loader(
		dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=global_bsz,
		drop_last=True, shuffle=False, num_workers=n_data_workers,
		pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
	tst_loader = make_data_loader(
		dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=global_bsz,
		drop_last=True, shuffle=False, num_workers=n_data_workers,
		pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.L
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}

	BOOK_SEQ_LEN = dataset_obj.L_book
	BOOK_DIM = dataset_obj.d_book

	return (dataset_obj, trn_loader, val_loader, tst_loader, aux_loaders, 
	 		N_CLASSES, SEQ_LENGTH, IN_DIM, BOOK_SEQ_LEN, BOOK_DIM, TRAIN_SIZE)

def create_lobster_train_loader(dataset_obj, seed, global_bsz, num_workers, reset_train_offsets=False, shuffle=True,
								pin_memory=True, prefetch_factor=6, persistent_workers=True):  # DATA CORE PARAMS: optimized defaults
	if reset_train_offsets:
		dataset_obj.reset_train_offsets()
	# use sampler to only get individual samples and automatic batching from dataloader
	trn_loader = make_data_loader(
		dataset_obj.dataset_train,
		dataset_obj,
		seed=seed,
		batch_size=global_bsz,
		shuffle=shuffle,  # TODO: remove later
		num_workers=num_workers,
		worker_init_fn=force_cpu,
		pin_memory=pin_memory,
		prefetch_factor=prefetch_factor,
		persistent_workers=persistent_workers)
	return trn_loader

Datasets = {
	# financial data
	"lobster-prediction": create_lobster_prediction_dataset,
}


def force_cpu(index:int):
	import os
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	import jax
	jax.config.update('jax_platform_name', 'cpu')
	# print("turning off cuda")
	# time.sleep(3)
	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	# print("done")
	# time.sleep(3)