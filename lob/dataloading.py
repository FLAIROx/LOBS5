import torch
from pathlib import Path
import os
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
from s5.dataloading import make_data_loader
from .lobster_dataloader import LOBSTER, LOBSTER_Dataset, LOBSTER_Sampler


DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')
DATA_DIR = Path('../data/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[LOBSTER, DataLoader, DataLoader, DataLoader, Dict, int, int, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


# Global worker init function for DataLoader multiprocessing
def _lobster_worker_init_fn(worker_id):
	"""
	Initialize worker process to set correct _collate_arg_names.
	This is needed because class variables are not shared across processes.
	Also forces JAX to use CPU backend to avoid CUDA conflicts.
	"""
	import os
	# Force JAX to use CPU in worker processes to avoid CUDA device conflicts
	# Workers only load data (CPU task), main process handles GPU training
	os.environ['JAX_PLATFORMS'] = 'cpu'

	from .lobster_dataloader import LOBSTER
	# These will be set by create_lobster_prediction_dataset
	# Read from global config if set
	if hasattr(_lobster_worker_init_fn, 'use_book_data'):
		use_book = _lobster_worker_init_fn.use_book_data
		return_raw = _lobster_worker_init_fn.return_raw_msgs
		if use_book:
			if return_raw:
				LOBSTER._collate_arg_names = ['book_data', 'raw_msgs', 'book_l2_init']
			else:
				LOBSTER._collate_arg_names = ['book_data']
		else:
			if return_raw:
				LOBSTER._collate_arg_names = ['raw_msgs']
			else:
				LOBSTER._collate_arg_names = []


def create_lobster_prediction_dataset(
		cache_dir: Union[str, Path] = DATA_DIR,
		seed: int = 42,
		mask_fn = LOBSTER_Dataset.no_mask,
		msg_seq_len: int = 500,
		per_gpu_bsz: int = 32,
		num_devices: int = 1,
		use_book_data: bool = False,
		use_simple_book: bool = False,
		book_transform: bool = False,
		book_depth: int = 500,
		n_data_workers: int = 0,
		return_raw_msgs: bool = False,
		shuffle_train=True,
		rand_offset=True,
		debug_overfit=False,
		test_dir: Union[str, Path, None] = None,
		data_mode: str = 'preproc',
		use_distributed_sampler: bool = False,
		process_rank: int = 0,
		process_count: int = 1,
	) -> ReturnType:
	"""
	Create LOBSTER prediction dataset with DataLoaders.

	Args:
		per_gpu_bsz: Batch size per GPU (each device processes this many samples)
		num_devices: Number of GPUs to use

	Note:
		The DataLoader will create batches of size (effective_bsz = per_gpu_bsz × num_devices).
		During training, jax.pmap will automatically split this batch across devices,
		so each GPU receives exactly per_gpu_bsz samples.
	"""
	if debug_overfit:
		rand_offset = False
		shuffle_train = False

	print("[*] Generating LOBSTER Prediction Dataset from", cache_dir)
	if test_dir is not None:
		print("[*] Using separate test directory:", test_dir)
	from .lobster_dataloader import LOBSTER
	name = 'lobster'

	# Calculate effective batch size: total samples processed per training step
	# DataLoader creates batches of this size, which jax.pmap splits across devices
	effective_bsz = per_gpu_bsz * num_devices
	print(f"[*] Batch size: {per_gpu_bsz} per GPU × {num_devices} devices = {effective_bsz} effective")

	# Set global config for worker_init_fn
	_lobster_worker_init_fn.use_book_data = use_book_data
	_lobster_worker_init_fn.return_raw_msgs = return_raw_msgs

	dataset_obj = LOBSTER(
		name,
		data_dir=cache_dir,
		mask_fn=mask_fn,
		msg_seq_len=msg_seq_len,
		use_book_data=use_book_data,
		use_simple_book=use_simple_book,
		book_transform=book_transform,
		book_depth=book_depth,
		n_cache_files=1e7,  # large number to keep everything in cache
		return_raw_msgs=return_raw_msgs,
		rand_offset=rand_offset,
		debug_overfit=debug_overfit,
		test_data_dir=test_dir,
		data_mode=data_mode,
	)
	dataset_obj.setup()

	print("Using mask function:", mask_fn)

	# Use global worker_init_fn for all loaders
	worker_fn = _lobster_worker_init_fn if n_data_workers > 0 else None

	# Create training loader
	trn_loader = create_lobster_train_loader(
		dataset_obj, seed, effective_bsz, n_data_workers,
		reset_train_offsets=rand_offset, shuffle=shuffle_train, worker_init_fn=worker_fn,
		use_distributed_sampler=use_distributed_sampler,
		process_rank=process_rank, process_count=process_count)

	# Validation/Test loaders: follow the same distributed sharding logic as training
	if use_distributed_sampler and process_count > 1:
		from torch.utils.data.distributed import DistributedSampler

		print(f"[*] Using DistributedSampler for VAL: process {process_rank}/{process_count}")
		sam_val = DistributedSampler(
			dataset_obj.dataset_val,
			num_replicas=process_count,
			rank=process_rank,
			shuffle=False,
			seed=seed,
			drop_last=True,
		)
		val_loader = make_data_loader(
			dataset_obj.dataset_val,
			dataset_obj,
			seed=seed,
			batch_size=effective_bsz,
			shuffle=False,
			sampler=sam_val,
			num_workers=0,
			worker_init_fn=None)

		print(f"[*] Using DistributedSampler for TEST: process {process_rank}/{process_count}")
		sam_test = DistributedSampler(
			dataset_obj.dataset_test,
			num_replicas=process_count,
			rank=process_rank,
			shuffle=False,
			seed=seed,
			drop_last=True,
		)
		tst_loader = make_data_loader(
			dataset_obj.dataset_test,
			dataset_obj,
			seed=seed,
			batch_size=effective_bsz,
			shuffle=False,
			sampler=sam_test,
			num_workers=0,
			worker_init_fn=None)
	else:
		# Single-node: standard DataLoader
		val_loader = make_data_loader(
			dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=effective_bsz,
			drop_last=True, shuffle=False, num_workers=0, worker_init_fn=None)
		tst_loader = make_data_loader(
			dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=effective_bsz,
			drop_last=True, shuffle=False, num_workers=0, worker_init_fn=None)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.L
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}

	BOOK_SEQ_LEN = dataset_obj.L_book
	BOOK_DIM = dataset_obj.d_book

	return (dataset_obj, trn_loader, val_loader, tst_loader, aux_loaders,
	 		N_CLASSES, SEQ_LENGTH, IN_DIM, BOOK_SEQ_LEN, BOOK_DIM, TRAIN_SIZE)


def create_lobster_train_loader(dataset_obj, seed, bsz, num_workers,
								reset_train_offsets=False, shuffle=True, worker_init_fn=None,
								use_distributed_sampler=False, process_rank=0, process_count=1):
	"""
	Create LOBSTER training DataLoader.

	Args:
		use_distributed_sampler: If True, use DistributedSampler for multi-node training
		process_rank: Current process rank (0 to process_count-1)
		process_count: Total number of processes (nodes)
	"""
	if reset_train_offsets:
		dataset_obj.reset_train_offsets()

	# Use global worker_init_fn if not provided
	if worker_init_fn is None and num_workers > 0:
		worker_init_fn = _lobster_worker_init_fn

	# Multi-node training: use DistributedSampler to shard data across processes
	if use_distributed_sampler and process_count > 1:
		from torch.utils.data.distributed import DistributedSampler

		print(f"[*] Using DistributedSampler: process {process_rank}/{process_count}")
		print(f"    Each process will load 1/{process_count} of the data")

		sampler = DistributedSampler(
			dataset_obj.dataset_train,
			num_replicas=process_count,  # Total number of processes
			rank=process_rank,           # Current process ID
			shuffle=shuffle,
			seed=seed,
			drop_last=True  # Ensure consistent batch count across processes
		)

		print(f"    Dataset size: {len(dataset_obj.dataset_train)}")
		print(f"    Samples per process: {len(dataset_obj.dataset_train) // process_count}")
		print(f"    Batch size: {bsz}")

		# Calculate expected number of batches per process
		samples_per_process = len(dataset_obj.dataset_train) // process_count
		expected_batches = samples_per_process // bsz
		print(f"    Expected batches per process: {expected_batches} (with drop_last=True)")

		trn_loader = make_data_loader(
			dataset_obj.dataset_train,
			dataset_obj,
			seed=seed,
			batch_size=bsz,
			shuffle=False,  # DistributedSampler handles shuffling
			sampler=sampler,
			num_workers=num_workers,
			worker_init_fn=worker_init_fn,
			drop_last=True)  # Critical: ensure all batches have consistent size

		print(f"    Actual batches in dataloader: {len(trn_loader)}")
	else:
		# Single-node training: use standard DataLoader
		trn_loader = make_data_loader(
			dataset_obj.dataset_train,
			dataset_obj,
			seed=seed,
			batch_size=bsz,
			shuffle=shuffle,
			num_workers=num_workers,
			worker_init_fn=worker_init_fn)
	return trn_loader


Datasets = {
	# financial data
	"lobster-prediction": create_lobster_prediction_dataset,
}
