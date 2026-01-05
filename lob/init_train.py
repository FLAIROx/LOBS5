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
from lob.train_helpers import (
    create_train_state,
    create_lobs5_learning_rate_schedule,
    create_prodigy_optimizer,
    extract_prodigy_estimated_lr,
    switch_optimizer_after_prodigy_warmup,
)
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
# from s5.dataloading import make_data_loader
# from lob.lobster_dataloader import LOBSTER_Dataset, LOBSTER

import lob.validation_helpers as valh


def deduplicate_trainstate(
        state: TrainState,
    ) -> TrainState:
    """
    Extract state to single device for checkpoint saving.

    Old (pmap): State had device dimension, used x[0] to extract first device
    New (jit+shardings): State is replicated via sharding (no device dimension),
                        just need to put on single device
    """
    # With jit+shardings, arrays don't have device dimension
    # Just move to single device (no indexing needed)
    return jax.device_put(state, device=jax.devices('gpu')[0])

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
        epoch: int,
    ) -> bool:
    """
    """
    return ckpt_mgr.save(
        epoch,
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
    # Extract the actual parameters from the nested custom structure
    if 'custom' in metadata:
        return Namespace(**metadata['custom'])
    else:
        return Namespace(**metadata)

def load_checkpoint(
        state: TrainState,
        path: str,
        # config_dict: dict,
        step: Optional[int] = None,
        train: bool = True,
    ) -> dict[str, Any]:

    mngr = ocp.CheckpointManager(
        os.path.abspath(path),
        item_names=('state', 'metadata'),
        options=ocp.CheckpointManagerOptions(),
        # metadata=ckpt['config']
    )

    if step is None:
        step = mngr.latest_step()

    loaded = mngr.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(
                # only stored trainstate from a single device (as they are all the same)
                deduplicate_trainstate(state)
            ),
            metadata=ocp.args.JsonRestore()
        )
    )
    ckpt = loaded['metadata']
    # Copy train state back to all devices
    if train:
        # Old (pmap): Use jax_utils.replicate (adds device dimension)
        # ckpt['model'] = jax_utils.replicate(loaded['state'])

        # New (jit+shardings): Use sharding-based replication
        from lob.sharding_utils import get_global_mesh, create_state_shardings
        mesh = get_global_mesh()
        state_shardings = create_state_shardings(loaded['state'], mesh)
        ckpt['model'] = jax.device_put(loaded['state'], state_shardings)
    else:
        ckpt['model'] = loaded['state']
    return ckpt


def init_train_state(
        args: Namespace,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
        train_size: int,  # NEW: needed for schedule calculation
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

    # ===========================================================================
    # Create learning rate schedules (MaxText-style optax schedules)
    # ===========================================================================
    steps_per_epoch = train_size // args.global_bsz
    if hasattr(args, 'curtail_epochs') and args.curtail_epochs is not None:
        steps_per_epoch = min(steps_per_epoch, args.curtail_epochs + 1)

    total_steps = steps_per_epoch * args.epochs
    warmup_end_step = steps_per_epoch * args.warmup_end

    if print_shapes:
        print(f"[Schedule] steps_per_epoch: {steps_per_epoch}")
        print(f"[Schedule] total_steps: {total_steps}")
        print(f"[Schedule] warmup_end_step: {warmup_end_step}")
        print(f"[Schedule] Base SSM LR: {ssm_lr}, Base LR: {lr}")
        print(f"[Schedule] LR min: {args.lr_min}, Cosine anneal: {args.cosine_anneal}")

    # Create schedule for SSM parameters
    ssm_lr_schedule = create_lobs5_learning_rate_schedule(
        base_lr=ssm_lr,
        warmup_end_step=warmup_end_step,
        total_steps=total_steps,
        lr_min=args.lr_min,
        use_cosine_anneal=args.cosine_anneal,
    )

    # Create schedule for regular parameters
    lr_schedule = create_lobs5_learning_rate_schedule(
        base_lr=lr,
        warmup_end_step=warmup_end_step,
        total_steps=total_steps,
        lr_min=args.lr_min,
        use_cosine_anneal=args.cosine_anneal,
    )

    # Initialize training state with optax schedules
    state, total_params = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        use_book_data=args.use_book_data,
        in_dim=1,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        global_bsz=args.global_bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr_schedule=ssm_lr_schedule,  # Pass schedule, not scalar
        lr_schedule=lr_schedule,          # Pass schedule, not scalar
        dt_global=args.dt_global,
        num_devices=args.num_devices,
    )

    return state, model_cls, total_params


def init_train_state_with_prodigy(
        args: Namespace,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
        train_size: int,
        print_shapes=False
    ) -> Tuple[TrainState, Union[partial[BatchLobPredModel],
                                  partial[BatchFullLobPredModel],
                                  partial[BatchPaddedLobPredModel],
                                  partial[OldBatchPaddedLobPredModel]], int, dict]:
    """
    Initialize training state with Prodigy optimizer for LR estimation.

    This is Phase 1 of Plan B: Use Prodigy to estimate optimal learning rate.
    After warmup steps, call extract_prodigy_estimated_lr() and
    switch_optimizer_after_prodigy_warmup() to switch to AdamW + cosine.

    Returns:
        state: TrainState with Prodigy optimizer
        model_cls: Model class
        total_params: Total trainable parameters
        schedule_info: Dict with schedule parameters for Phase 2
    """
    from lob.train_helpers import create_prodigy_optimizer, map_nested_fn, create_train_state
    from lob.sharding_utils import initialize_mesh, get_global_mesh, create_state_shardings

    in_dim = n_classes

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

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

    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    if print_shapes:
        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(V.shape))
        print("Vinv.shape={}".format(Vinv.shape))
        print("book_seq_len", book_seq_len)
        print("book_dim", book_dim)
        print("[Prodigy Mode] Initializing with Prodigy optimizer for LR estimation")

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

    # Model class setup (same as normal init)
    if args.use_book_data:
        if args.merging == 'projected':
            model_cls = partial(
                BatchFullLobPredModel,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,
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
        elif args.merging == 'padded':
            model_cls = partial(
                BatchPaddedLobPredModel,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,
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
        else:
            raise ValueError("Merge method: " + args.merging + " is not valid")
    else:
        if args.num_devices > 1:
            raise NotImplementedError("Message only model not implemented for multi-device training")
        model_cls = partial(
            BatchLobPredModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=False,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    # Calculate schedule parameters (needed for Phase 2)
    steps_per_epoch = train_size // args.global_bsz
    if hasattr(args, 'curtail_epochs') and args.curtail_epochs is not None:
        steps_per_epoch = min(steps_per_epoch, args.curtail_epochs + 1)

    total_steps = steps_per_epoch * args.epochs
    warmup_end_step = steps_per_epoch * args.warmup_end

    schedule_info = {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'warmup_end_step': warmup_end_step,
        'ssm_lr_base': ssm_lr,
        'lr_min': args.lr_min,
        'use_cosine_anneal': args.cosine_anneal,
        'weight_decay': args.weight_decay,
        'opt_config': args.opt_config,
        'dt_global': args.dt_global,
    }

    if print_shapes:
        print(f"[Schedule] steps_per_epoch: {steps_per_epoch}")
        print(f"[Schedule] total_steps: {total_steps}")
        print(f"[Schedule] warmup_end_step: {warmup_end_step}")
        print(f"[Schedule] SSM LR base: {ssm_lr}")

    # Create SSM schedule for warmup phase
    ssm_lr_schedule = create_lobs5_learning_rate_schedule(
        base_lr=ssm_lr,
        warmup_end_step=warmup_end_step,
        total_steps=total_steps,
        lr_min=args.lr_min,
        use_cosine_anneal=args.cosine_anneal,
    )

    # Create Prodigy optimizer (regular params use Prodigy, SSM uses Adam)
    tx, ssm_fn = create_prodigy_optimizer(
        ssm_lr_schedule=ssm_lr_schedule,
        weight_decay=args.weight_decay,
        opt_config=args.opt_config,
        dt_global=args.dt_global,
    )

    # Initialize mesh
    try:
        mesh = get_global_mesh()
        print("[State] Using existing global mesh")
    except RuntimeError:
        mesh = initialize_mesh(args.num_devices)
        print("[State] Created new global mesh")

    # Initialize model
    micro_bsz = args.global_bsz // args.num_devices
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(init_rng, num=2)

    if args.use_book_data:
        dummy_input = (
            np.ones((micro_bsz, seq_len,), dtype=np.int32),
            np.ones((micro_bsz, seq_len, book_dim)),
        )
        integration_timesteps = (
            np.ones((micro_bsz, seq_len,)),
            np.ones((micro_bsz, seq_len,)),
        )
    else:
        dummy_input = (np.ones((micro_bsz, seq_len,), dtype=np.int32),)
        integration_timesteps = (np.ones((micro_bsz, seq_len,)),)

    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        *dummy_input, *integration_timesteps,
        method='__call_ar__'
    )

    if args.batchnorm:
        params = variables["params"]
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"]

    # Count parameters
    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    from lob.train_helpers import map_nested_fn
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    total_params = sum(jax.tree_util.tree_leaves(param_sizes))
    print(f"[*] Trainable Parameters: {total_params}")

    # Create train state with Prodigy
    if args.batchnorm:
        class TrainState(flax.training.train_state.TrainState):
            batch_stats: Any
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        state = flax.training.train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Apply sharding
    state_shardings = create_state_shardings(state, mesh)
    state = jax.device_put(state, state_shardings)

    print("[Prodigy Mode] State initialized with Prodigy optimizer")
    print("[Prodigy Mode] Run warmup steps, then call switch_optimizer_after_prodigy_warmup()")

    return state, model_cls, total_params, schedule_info
