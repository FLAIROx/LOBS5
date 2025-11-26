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
from lob.train_helpers import create_train_state#, eval_step, prep_batch, cross_entropy_loss, compute_accuracy
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
# from s5.dataloading import make_data_loader
# from lob.lobster_dataloader import LOBSTER_Dataset, LOBSTER

import lob.validation_helpers as valh


def deduplicate_trainstate(
        state: TrainState,
    ) -> TrainState:
    """
    Extract single copy of replicated state and place on local GPU 0.
    Uses jax.local_devices() for multi-node compatibility.
    """
    return jax.device_put(
        jax.tree.map(lambda x: x[0], state),
        device=jax.local_devices()[0]
    )

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
    Save checkpoint synchronously (waits for completion).
    """
    result = ckpt_mgr.save(
        epoch,
        # args=ocp.args.PyTreeSave(ckpt)
        args=ocp.args.Composite(
            # train state
            state=ocp.args.StandardSave(ckpt['model']),
            # all other dict elements
            metadata=ocp.args.JsonSave({k: v for k, v in ckpt.items() if k != 'model'}),
        )
    )
    # Wait for checkpoint to be fully written before continuing
    # This is critical for multi-node synchronization - prevents deadlock
    # where one process continues while another is still waiting for save
    ckpt_mgr.wait_until_finished()
    return result


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
    ) -> dict[str, Any]:

    mngr = ocp.CheckpointManager(
        os.path.abspath(path),
        item_names=('state', 'metadata'),
        options=ocp.CheckpointManagerOptions(),
        # metadata=ckpt['config']
    )

    if step is None:
        step = mngr.latest_step()

    # Restore checkpoint and update template state with restored params
    # This approach preserves the optimizer state structure from the template
    # and only updates params (and batch_stats if present)
    try:
        loaded = mngr.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(),  # No target - restore raw pytree
                metadata=ocp.args.JsonRestore()
            )
        )
        restored_state_dict = loaded['state']

        # Convert restored numpy arrays back to JAX arrays
        restored_state_dict = jax.tree.map(lambda x: np.asarray(x), restored_state_dict)

        # Get template state for structure preservation
        template_state = deduplicate_trainstate(state)

        # Update only params (and batch_stats if present) from checkpoint
        # Keep optimizer state from template - it will be re-initialized
        # This is safe because the optimizer state is just for continuing training
        if 'batch_stats' in restored_state_dict:
            restored_state = template_state.replace(
                step=restored_state_dict['step'],
                params=restored_state_dict['params'],
                batch_stats=restored_state_dict['batch_stats'],
            )
        else:
            restored_state = template_state.replace(
                step=restored_state_dict['step'],
                params=restored_state_dict['params'],
            )
        print(f"[*] Restored params from checkpoint (optimizer state re-initialized)")

    except Exception as e:
        print(f"[WARNING] Raw restore failed ({e}), trying with target structure...")
        # Fallback: restore with target structure
        loaded = mngr.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(
                    deduplicate_trainstate(state)
                ),
                metadata=ocp.args.JsonRestore()
            )
        )
        restored_state = loaded['state']

    ckpt = loaded['metadata']
    # copy train state back to all devices
    if train:
        ckpt['model'] = jax_utils.replicate(restored_state)
    else:
        ckpt['model'] = restored_state
    return ckpt


def init_train_state(
        args: Namespace,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
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
        bsz=args.effective_bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        dt_global=args.dt_global,
        num_devices=args.num_devices,
    )

    return state, model_cls
