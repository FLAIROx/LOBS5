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
		bsz: int=128,
		use_book_data: bool = False,
		use_simple_book: bool = False,
		book_transform: bool = False,
		book_depth: int = 500,
		n_data_workers: int = 0,
		return_raw_msgs: bool = False,
		shuffle_train=True,
		rand_offset=True,
		debug_overfit=False,
		val_split: float = 0.1,
		test_split: float = 0.1,
		pin_memory: bool = True,
		prefetch_factor: int = 2,
		persistent_workers: bool = True,
		test_dir_name: Union[str, Path, None] = None,
		use_distributed_sampler: bool = False,
		process_rank: int = 0,
		process_count: int = 1,
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
		n_cache_files=250,  # large number to keep everything in cache
		return_raw_msgs=return_raw_msgs,
		rand_offset=rand_offset,
		debug_overfit=debug_overfit,
		val_split=val_split,
		test_split=test_split,
		test_data_dir=test_dir_name,
	)
	dataset_obj.setup()
 
	# breakpoint()

	print("Using mask function:", mask_fn)

	# use sampler to only get individual samples and automatic batching from dataloader
	#trn_sampler = LOBSTER_Sampler(
	#		dataset_obj.dataset_train, n_files_shuffle=5, batch_size=1, seed=seed)
	
	trn_loader = create_lobster_train_loader(
		dataset_obj, seed, bsz, n_data_workers, reset_train_offsets=rand_offset, shuffle=shuffle_train,
		pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
		use_distributed_sampler=use_distributed_sampler, process_rank=process_rank, process_count=process_count)
	# NOTE: drop_last=True recompiles the model for a smaller batch size
	val_sampler = None
	tst_sampler = None
	if use_distributed_sampler and process_count > 1:
		from torch.utils.data import DistributedSampler
		val_sampler = DistributedSampler(
			dataset_obj.dataset_val, num_replicas=process_count,
			rank=process_rank, shuffle=False, drop_last=True)
		tst_sampler = DistributedSampler(
			dataset_obj.dataset_test, num_replicas=process_count,
			rank=process_rank, shuffle=False, drop_last=True)
		print(f"[*] Val DistributedSampler: rank={process_rank}/{process_count}, "
			  f"samples_per_node={len(val_sampler)}")
		print(f"[*] Test DistributedSampler: rank={process_rank}/{process_count}, "
			  f"samples_per_node={len(tst_sampler)}")
	val_loader = make_data_loader(
		dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz,
		drop_last=True, shuffle=False, sampler=val_sampler, num_workers=n_data_workers,
		pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
	tst_loader = make_data_loader(
		dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz,
		drop_last=True, shuffle=False, sampler=tst_sampler, num_workers=n_data_workers,
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

def create_lobster_train_loader(dataset_obj, seed, bsz, num_workers, reset_train_offsets=False, shuffle=True,
								pin_memory=True, prefetch_factor=2, persistent_workers=True,
								use_distributed_sampler=False, process_rank=0, process_count=1):
	if reset_train_offsets:
		dataset_obj.reset_train_offsets()

	train_sampler = None
	if use_distributed_sampler and process_count > 1:
		from torch.utils.data import DistributedSampler
		train_sampler = DistributedSampler(
			dataset_obj.dataset_train,
			num_replicas=process_count,
			rank=process_rank,
			shuffle=shuffle,
			seed=seed,
			drop_last=True,
		)
		print(f"[*] DistributedSampler: rank={process_rank}/{process_count}, "
			  f"samples_per_node={len(train_sampler)}")
		shuffle = False  # sampler handles shuffling

	trn_loader = make_data_loader(
		dataset_obj.dataset_train,
		dataset_obj,
		seed=seed,
		batch_size=bsz,
		shuffle=shuffle,
		drop_last=True,
		sampler=train_sampler,
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
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	os.environ["JAX_PLATFORMS"] = "cpu"