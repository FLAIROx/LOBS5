from __future__ import annotations
import json
import os
from argparse import Namespace
from glob import glob
from functools import partial
from typing import Any, Optional, Tuple, Union
import jax
import jax.numpy as np
from jax import random
import flax
from flax import jax_utils
import orbax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from jax.scipy.linalg import block_diag
from flax.training import checkpoints
from flax import linen as nn
from orbax import checkpoint
from lob.encoding import Vocab
from lob.lob_seq_model import BatchFullLobPredModel, BatchLobPredModel, BatchPaddedLobPredModel,OldBatchPaddedLobPredModel, FullLobPredModel#, ParFullLobPredModel

#from lob.lob_seq_model import BatchLobPredModel
from lob.train_helpers import create_train_state, create_lobs5_learning_rate_schedule
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
# from s5.dataloading import make_data_loader
# from lob.lobster_dataloader import LOBSTER_Dataset, LOBSTER

import lob.validation_helpers as valh


def deduplicate_trainstate(
        state: TrainState,
    ) -> TrainState:
    """Extract a single copy of state for checkpointing.
    Moves state to host first (safe for any sharding topology),
    then places on local GPU 0.
    """
    host_state = jax.device_get(state)
    return jax.device_put(host_state, device=jax.local_devices()[0])


def remap_train_state_step(state: TrainState, new_step: int) -> TrainState:
    """Remap state.step and optimizer schedule counts for elastic resume.

    When resuming with a different number of nodes (different global BSZ),
    state.step and optimizer counts must be remapped so the LR schedule
    position matches the correct epoch in the new schedule.

    All optimizer count fields (ScaleByAdamState.count, ScaleByScheduleState.count)
    track the same global step. We replace all scalar int counts matching old_step
    with new_step. Adam bias correction (1/(1-beta^count)) converges for count>100,
    so remapping doesn't affect convergence — only schedule alignment matters.
    """
    import numpy as real_np  # init_train.py aliases jax.numpy as np
    old_step = int(state.step)
    new_step_val = real_np.int32(new_step)

    def remap_leaf(leaf):
        if hasattr(leaf, 'shape') and leaf.shape == () and hasattr(leaf, 'dtype'):
            if leaf.dtype in (real_np.int32, real_np.int64) and int(leaf) == old_step:
                return real_np.int32(new_step)
        return leaf

    new_opt_state = jax.tree_util.tree_map(remap_leaf, state.opt_state)
    return state.replace(step=new_step_val, opt_state=new_opt_state)


def load_args_from_checkpoint(
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> Namespace:

    """Load arguments from checkpoint"""
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    raw_restored = checkpoints.restore_checkpoint(
        checkpoint_path,
        None,
        step=step,
        orbax_checkpointer=orbax_checkpointer
    )
    args = Namespace(**raw_restored['config'])
    return args

def save_checkpoint(
        ckpt_mgr: ocp.CheckpointManager,
        ckpt: dict,
        step: int,
    ) -> bool:
    """Save checkpoint keyed by step (global_step for mid-epoch, or epoch for epoch-end)."""
    return ckpt_mgr.save(
        step,
        # args=ocp.args.PyTreeSave(ckpt)
        args=ocp.args.Composite(
            # train state
            state=ocp.args.StandardSave(ckpt['model']),
            # all other dict elements
            metadata=ocp.args.JsonSave({k: v for k, v in ckpt.items() if k != 'model'}),
        )
    )


# def load_checkpoint(
#         state: TrainState,
#         path: str,
#         config_dict: dict,
#         step: Optional[int] = None,
#     ) -> TrainState:
#     ckpt = {
#         'model': state,
#         'config': config_dict,
#         'metrics': {
#             'loss_train': np.nan,
#             'loss_val': np.nan,
#             'loss_test': np.nan,
#             'acc_val': np.nan,
#             'acc_test': np.nan,
#         }
#     }
#     orbax_checkpointer = checkpoint.PyTreeCheckpointer()
#     restored = checkpoints.restore_checkpoint(
#         path,
#         ckpt,
#         step=step,
#         orbax_checkpointer=orbax_checkpointer
#     )
#     return restored

def load_metadata(
        path: str,
    ) -> Namespace:

    json_path = path + '/metadata/_ROOT_METADATA'
    # load json path to dict
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    # Extract the actual parameters from the nested custom_metadata structure
    if 'custom_metadata' in metadata:
        return Namespace(**metadata['custom_metadata'])
    else:
        return Namespace(**metadata)

def load_checkpoint(
        state: TrainState,
        path: str,
        # config_dict: dict,
        step: Optional[int] = None,
        train: bool = True,
        mesh=None,
        partial_restore: bool = False,
    ) -> dict[str, Any]:

    mngr = ocp.CheckpointManager(
        os.path.abspath(path),
        item_names=('state', 'metadata'),
        options=ocp.CheckpointManagerOptions(),
        # metadata=ckpt['config']
    )

    if step is None:
        step = mngr.latest_step()

    print(f"[Checkpoint] Loading step={step} from {path} "
          f"(partial_restore={partial_restore})")

    loaded = mngr.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(
                # only stored trainstate from a single device (as they are all the same)
                deduplicate_trainstate(state),
                strict=(not partial_restore),
            ),
            metadata=ocp.args.JsonRestore()
        )
    )
    ckpt = loaded['metadata']
    # copy train state back to all devices
    if train:
        if mesh is not None:
            # Orbax restores to single device; move to numpy (device-agnostic)
            # then let caller shard to global mesh
            host_state = jax.device_get(loaded['state'])
            ckpt['model'] = host_state
        else:
            ckpt['model'] = jax_utils.replicate(loaded['state'])
    else:
        ckpt['model'] = loaded['state']
    return ckpt


def init_train_state(
        args: Namespace,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
        train_size: int = 0,
        print_shapes=False
    ) -> Tuple[TrainState, Union[partial[BatchLobPredModel],
                                  partial[BatchFullLobPredModel],
                                  partial[BatchPaddedLobPredModel],
                                  partial[OldBatchPaddedLobPredModel]]]:

    in_dim = n_classes

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)

    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    if print_shapes:
        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(V.shape))
        print("Vinv.shape={}".format(Vinv.shape))
        print("book_seq_len", book_seq_len)
        print("book_dim", book_dim)

    padded = False
    retrieval = False
    speech = False

    ssm_init_fn = init_S5SSM(
        H=args.d_model,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=args.C_init,
        discretization=args.discretization,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional
    )
    
    if args.use_book_data:
        # if args.num_devices > 1:
        #     model_cls = ParFullLobPredModel
        # else:
        #     model_cls = BatchFullLobPredModel
        

        if args.merging == 'projected':
            model_cls = partial(
                # projecting sequence lengths down has appeared better than padding
                BatchFullLobPredModel,
                #BatchPaddedLobPredModel,
                #model_cls,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,  # 2
                n_fused_layers=args.n_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
            )
        elif args.merging == 'padded': #i.e. 'padded'
            model_cls = partial(
                # projecting sequence lengths down has appeared better than padding
                BatchPaddedLobPredModel,
                #model_cls,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,  # 2
                n_fused_layers=args.n_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
                #args not adding to partial: training & rescale. 
            )
        else:
            raise ValueError("Merge method: " + args.merging + " is not valid (check spelling)")

    else:
        if args.num_devices > 1:
            raise NotImplementedError("Message only model not implemented for multi-device training")
        
        model_cls = partial(
            BatchLobPredModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    # Create learning rate schedules if train_size is available
    ssm_lr_schedule = None
    lr_schedule = None
    if train_size > 0:
        process_count = getattr(args, 'process_count', jax.process_count())
        grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        # args.micro_bsz is per-GPU BSZ; global BSZ = micro_bsz * num_devices * process_count
        micro_steps_per_epoch = train_size // (args.micro_bsz * args.num_devices * process_count)
        if hasattr(args, 'curtail_epochs') and args.curtail_epochs is not None:
            micro_steps_per_epoch = min(micro_steps_per_epoch, args.curtail_epochs + 1)
        # steps_per_epoch in optimizer updates (= micro_steps // K)
        steps_per_epoch = micro_steps_per_epoch // grad_accum_steps
        total_steps = steps_per_epoch * args.epochs
        warmup_end_step = int(steps_per_epoch * args.warmup_end)

        # lr_min = 5% of base LR unless explicitly overridden
        effective_lr_min = args.lr_min if args.lr_min > 0 else lr * 0.05
        effective_ssm_lr_min = args.lr_min if args.lr_min > 0 else ssm_lr * 0.05

        if print_shapes:
            print(f"[Schedule] steps_per_epoch: {steps_per_epoch}")
            print(f"[Schedule] total_steps: {total_steps}")
            print(f"[Schedule] warmup_end_step: {warmup_end_step}")
            print(f"[Schedule] Base SSM LR: {ssm_lr}, Base LR: {lr}")
            print(f"[Schedule] lr_min: {effective_lr_min}, ssm_lr_min: {effective_ssm_lr_min}")

        ssm_lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=ssm_lr,
            warmup_end_step=warmup_end_step,
            total_steps=total_steps,
            lr_min=effective_ssm_lr_min,
            use_cosine_anneal=args.cosine_anneal,
        )
        lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=lr,
            warmup_end_step=warmup_end_step,
            total_steps=total_steps,
            lr_min=effective_lr_min,
            use_cosine_anneal=args.cosine_anneal,
        )

    # initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        use_book_data=args.use_book_data,
        in_dim=1, # in_dim,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        micro_bsz=args.micro_bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        ssm_lr_schedule=ssm_lr_schedule,
        lr_schedule=lr_schedule,
        dt_global=args.dt_global,
        num_devices=args.num_devices,
    )

    return state, model_cls
