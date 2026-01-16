from functools import partial
import numpy as onp
import jax
import jax.numpy as np
# from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
# from flax import jax_utils  # No longer needed - migrated to jax.jit + shardings
import optax
from typing import Any, Dict, Optional, Tuple, Union
from lob.encoding import Message_Tokenizer
import sys

import psutil
import os
import time

# New: Import sharding utilities (migrating from pmap to jax.jit + shardings)
from lob.sharding_utils import (
    create_simple_mesh,
    create_data_sharding,
    create_replicated_sharding,
    tree_replicate_to_devices,
    create_state_shardings,
    initialize_mesh,
    get_global_mesh,
    get_data_shardings_for_batch,
)
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from .profiling_utils import GoodputMonitor
# from lob.lob_seq_model import LobPredModel


TIME_START_I=9
TIME_END_I =13

# num_devices_global = 2
# global_devices = jax.local_devices()[0: num_devices_global]


# ==============================================================================
# Learning Rate Schedule Creation (MaxText-style optax schedules)
# ==============================================================================

def create_lobs5_learning_rate_schedule(
    base_lr: float,
    warmup_end_step: int,
    total_steps: int,
    lr_min: float = 0.0,
    use_cosine_anneal: bool = True,
) -> optax.Schedule:
    """
    Creates a learning rate schedule for LOBS5 training.

    This follows MaxText's approach: create an optax Schedule function that is
    passed directly to the optimizer, eliminating manual per-step LR updates.

    Schedule:
    1. Linear warmup from 0 to base_lr over [0, warmup_end_step]
    2. Cosine decay from base_lr to lr_min over [warmup_end_step, total_steps]
       (or constant base_lr if use_cosine_anneal=False)

    Args:
        base_lr: Peak learning rate (reached at end of warmup)
        warmup_end_step: Step at which warmup ends (steps_per_epoch * warmup_end_epochs)
        total_steps: Total training steps (steps_per_epoch * total_epochs)
        lr_min: Minimum learning rate at end of cosine decay
        use_cosine_anneal: If True, use cosine decay after warmup; if False, constant lr

    Returns:
        An optax Schedule function: step -> learning_rate
    """
    # Warmup schedule: 0 -> base_lr over warmup_end_step steps
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_end_step
    )

    if use_cosine_anneal:
        # Cosine decay after warmup
        cosine_steps = total_steps - warmup_end_step

        def make_cos_schedule(init_lr, final_lr, len_steps):
            """Custom cosine schedule matching LOBS5's original cosine_annealing."""
            def schedule(step):
                # step here is relative to start of cosine phase
                pct = step / len_steps
                pct = np.minimum(pct, 1.0)  # Clamp to [0, 1]
                cosine_decay = 0.5 * (1 + np.cos(np.pi * pct))
                lr = (init_lr - final_lr) * cosine_decay + final_lr
                return lr
            return schedule

        cosine_schedule = make_cos_schedule(base_lr, lr_min, cosine_steps)

        # Join warmup and cosine schedules
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[warmup_end_step]
        )
    else:
        # Constant LR after warmup
        constant_schedule = optax.constant_schedule(base_lr)
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, constant_schedule],
            boundaries=[warmup_end_step]
        )

    return schedule


# ==============================================================================
# Prodigy LR Estimation Functions (Plan B)
# ==============================================================================
#
# These functions support a two-phase training approach:
# Phase 1: Use Prodigy to estimate optimal learning rate
# Phase 2: Switch to AdamW + cosine annealing with estimated LR
#
# This preserves your existing schedule architecture while letting Prodigy
# find the optimal base learning rate automatically.
# ==============================================================================

def create_prodigy_optimizer(
    ssm_lr_schedule: optax.Schedule,
    weight_decay: float = 0.05,
    opt_config: str = "standard",
    dt_global: bool = False,
) -> Tuple[optax.GradientTransformation, callable]:
    """
    Create optimizer with Prodigy for 'regular' params and Adam for SSM params.

    Returns:
        tx: The multi_transform optimizer
        ssm_fn: The parameter labeling function (needed for recreation)
    """
    if dt_global:
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["B", "Lambda_re", "Lambda_im", "norm"]
            else ("none" if k in [] else "regular")
        )
    else:
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
            else ("none" if k in [] else "regular")
        )

    tx = optax.multi_transform(
        {
            "none": optax.sgd(learning_rate=0.0),
            "ssm": optax.adam(learning_rate=ssm_lr_schedule),
            "regular": optax.contrib.prodigy(
                learning_rate=1.0,  # Prodigy scales this automatically
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            ),
        },
        ssm_fn,
    )
    return tx, ssm_fn


def extract_prodigy_estimated_lr(
    state: train_state.TrainState,
    lr_multiplier: float = 1.0,
) -> float:
    """
    Extract the estimated learning rate from Prodigy optimizer state.

    Prodigy internally computes an optimal learning rate 'd' based on:
    - Gradient statistics
    - Parameter update magnitudes
    - Loss curvature estimates

    The estimated LR is stored in ProdigyState.estim_lr

    Args:
        state: Training state containing Prodigy optimizer
        lr_multiplier: Multiplier for the estimated LR (default 1.0)
                       Use <1.0 for conservative, >1.0 for aggressive

    Returns:
        Estimated optimal learning rate
    """
    # Navigate to Prodigy state within multi_transform
    # Structure: opt_state.inner_states['regular'] -> ProdigyState
    try:
        prodigy_state = state.opt_state.inner_states['regular']
        # ProdigyState fields: (exp_avg, exp_avg_sq, grad_sum, params0, estim_lr, numerator_weighted, count)
        estim_lr = float(prodigy_state.estim_lr)

        # Apply multiplier
        estimated_lr = estim_lr * lr_multiplier

        print(f"[Prodigy] Estimated LR: {estim_lr:.6f}")
        print(f"[Prodigy] With multiplier ({lr_multiplier}x): {estimated_lr:.6f}")

        return estimated_lr
    except AttributeError as e:
        raise RuntimeError(
            f"Failed to extract estim_lr from Prodigy state. "
            f"Ensure the optimizer was created with Prodigy. Error: {e}"
        )


def switch_optimizer_after_prodigy_warmup(
    state: train_state.TrainState,
    estimated_lr: float,
    ssm_lr_base: float,
    warmup_end_step: int,
    total_steps: int,
    lr_min: float,
    use_cosine_anneal: bool,
    weight_decay: float,
    opt_config: str,
    dt_global: bool,
    mesh,  # JAX Mesh for sharding
) -> train_state.TrainState:
    """
    Switch from Prodigy optimizer to AdamW + cosine annealing.

    This preserves:
    - Model parameters (params)
    - Training step counter (step)

    This resets:
    - Optimizer state (mu, nu) - fresh start with new optimizer

    Why reset optimizer state:
    - Prodigy's momentum is tuned for its adaptive LR algorithm
    - AdamW needs fresh momentum to properly converge with cosine schedule
    - Keeping Prodigy's momentum would cause training instability

    Args:
        state: Current training state with Prodigy optimizer
        estimated_lr: Learning rate estimated by Prodigy (used for 'regular' params)
        ssm_lr_base: Base LR for SSM params (unchanged from original)
        warmup_end_step: Step at which warmup ends
        total_steps: Total training steps
        lr_min: Minimum LR for cosine annealing
        use_cosine_anneal: Whether to use cosine annealing
        weight_decay: Weight decay for AdamW
        opt_config: Optimization config (standard, BandCdecay, etc.)
        dt_global: Whether dt is global parameter
        mesh: JAX Mesh for sharding

    Returns:
        New training state with AdamW + cosine schedule optimizer
    """
    from lob.sharding_utils import create_state_shardings

    # Get current step (we want to continue from here, not reset to 0)
    current_step = int(state.step)

    print(f"[Switch] Switching optimizer at step {current_step}")
    print(f"[Switch] estimated_lr (for regular params): {estimated_lr:.6f}")
    print(f"[Switch] ssm_lr_base (for SSM params): {ssm_lr_base:.6f}")

    # Create new schedules starting from current_step
    # Adjust warmup to account for Prodigy phase
    # Since we already did warmup via Prodigy, skip to cosine phase
    adjusted_warmup_end = max(warmup_end_step, current_step)

    # Create SSM schedule (uses original ssm_lr_base)
    ssm_lr_schedule = create_lobs5_learning_rate_schedule(
        base_lr=ssm_lr_base,
        warmup_end_step=adjusted_warmup_end,
        total_steps=total_steps,
        lr_min=lr_min,
        use_cosine_anneal=use_cosine_anneal,
    )

    # Create regular schedule (uses Prodigy-estimated LR)
    lr_schedule = create_lobs5_learning_rate_schedule(
        base_lr=estimated_lr,
        warmup_end_step=adjusted_warmup_end,
        total_steps=total_steps,
        lr_min=lr_min,
        use_cosine_anneal=use_cosine_anneal,
    )

    # Create new optimizer with schedules
    if dt_global:
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["B", "Lambda_re", "Lambda_im", "norm"]
            else ("none" if k in [] else "regular")
        )
    else:
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
            else ("none" if k in [] else "regular")
        )

    if opt_config in ["standard"]:
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "ssm": optax.adam(learning_rate=ssm_lr_schedule),
                "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        tx = optax.multi_transform(
            {
                "none": optax.adamw(learning_rate=ssm_lr_schedule, weight_decay=weight_decay),
                "ssm": optax.adam(learning_rate=ssm_lr_schedule),
                "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
            },
            ssm_fn,
        )
    else:
        # Default to standard
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "ssm": optax.adam(learning_rate=ssm_lr_schedule),
                "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
            },
            ssm_fn,
        )

    # Initialize new optimizer state
    new_opt_state = tx.init(state.params)

    # Create new TrainState preserving params and step
    new_state = state.replace(
        tx=tx,
        opt_state=new_opt_state,
        # step is preserved automatically
    )

    # Apply sharding
    state_shardings = create_state_shardings(new_state, mesh)
    new_state = jax.jit(lambda s: s, out_shardings=state_shardings)(new_state)

    print(f"[Switch] Optimizer switched successfully")
    print(f"[Switch] Current LR (regular): {lr_schedule(current_step):.6f}")
    print(f"[Switch] Current LR (SSM): {ssm_lr_schedule(current_step):.6f}")

    return new_state


# ==============================================================================
# Old LR schedulers (DEPRECATED - replaced by create_lobs5_learning_rate_schedule)
# ==============================================================================
# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step,  lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # # Update state
    # state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = \
    #     jax_utils.replicate(np.array(lr_val, dtype=np.float32))
        
    # state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']= \
    #     jax_utils.replicate(np.array(ssm_lr_val, dtype=np.float32))

    # if opt_config in ["BandCdecay"]:
    #     # In this case we are applying the ssm learning rate to B, even though
    #     # we are also using weight decay on B
    #     state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = \
    #         jax_utils.replicate(np.array(ssm_lr_val, dtype=np.float32))
    # BETTER WAY - reuse existing structure:
    # CRITICAL: Create separate arrays to avoid buffer aliasing with donate_argnums
    # Each assignment must get its own unique array to prevent "donate buffer twice" error
    lr_array = np.array(lr_val, dtype=np.float32)
    ssm_lr_array = np.array(ssm_lr_val, dtype=np.float32)
    ssm_lr_array_copy = np.array(ssm_lr_val, dtype=np.float32)  # Separate copy for 'none' optimizer
    
    # Update in place by creating new state with updated hyperparams
    # This avoids accumulating replicated tensors while preserving other hyperparameters
    state = state.replace(
        opt_state=state.opt_state._replace(
            inner_states={
                **state.opt_state.inner_states,
                'regular': state.opt_state.inner_states['regular']._replace(
                    inner_state=state.opt_state.inner_states['regular'].inner_state._replace(
                        hyperparams={
                            **state.opt_state.inner_states['regular'].inner_state.hyperparams,
                            # Old way (pmap): 'learning_rate': jax_utils.replicate(lr_array)
                            # New way (jit + shardings): lr_array already replicated via sharding
                            'learning_rate': lr_array
                        }
                    )
                ),
                'ssm': state.opt_state.inner_states['ssm']._replace(
                    inner_state=state.opt_state.inner_states['ssm'].inner_state._replace(
                        hyperparams={
                            **state.opt_state.inner_states['ssm'].inner_state.hyperparams,
                            # Old way (pmap): 'learning_rate': jax_utils.replicate(ssm_lr_array)
                            # New way (jit + shardings): ssm_lr_array already replicated via sharding
                            'learning_rate': ssm_lr_array
                        }
                    )
                ),
            }
        )
    )

    if opt_config in ["BandCdecay"]:
        state = state.replace(
            opt_state=state.opt_state._replace(
                inner_states={
                    **state.opt_state.inner_states,
                    'none': state.opt_state.inner_states['none']._replace(
                        inner_state=state.opt_state.inner_states['none'].inner_state._replace(
                            hyperparams={
                                **state.opt_state.inner_states['none'].inner_state.hyperparams,
                                # Old way (pmap): 'learning_rate': jax_utils.replicate(ssm_lr_array)
                                # New way (jit + shardings): Use separate copy to avoid buffer aliasing
                                'learning_rate': ssm_lr_array_copy  # Separate copy, not ssm_lr_array!
                            }
                        )
                    ),
                }
            )
        )
    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       use_book_data,
                       book_dim,
                       book_seq_len,
                       in_dim=1,
                       global_bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr_schedule=None,  # Changed: now accepts optax.Schedule
                       lr_schedule=None,      # Changed: now accepts optax.Schedule
                       dt_global=False,
                       num_devices=1,
                       ):
    """
    Initializes the training state using optax.

    IMPORTANT: ssm_lr_schedule and lr_schedule should be optax.Schedule functions,
    not scalar values. Use create_lobs5_learning_rate_schedule() to create them.

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param global_bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param dt_global:
    :return:
    """

    # global_bsz is total batch size across all devices
    # micro_bsz is per-GPU batch size
    assert global_bsz % num_devices == 0
    micro_bsz = global_bsz // num_devices

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*micro_bsz, seq_len, in_dim)), np.ones(2*micro_bsz))
            integration_timesteps = np.ones((2*micro_bsz, seq_len,))
        else:
            dummy_input = (np.ones((micro_bsz, seq_len, in_dim)), np.ones(micro_bsz))
            integration_timesteps = np.ones((micro_bsz, seq_len,))
    else:
        if use_book_data:
            dummy_input = (
                # np.ones((micro_bsz, seq_len, in_dim), dtype=np.int32),  # messages
                np.ones((micro_bsz, seq_len, ), dtype=np.int32),  # messages
                np.ones((micro_bsz, seq_len, book_dim)),  # books
            )
            integration_timesteps = (
                np.ones((micro_bsz, seq_len, )),
                np.ones((micro_bsz, seq_len, )),
            )
        else:
            # dummy_input = (np.ones((micro_bsz, seq_len, in_dim), dtype=np.int32) , )
            dummy_input = (np.ones((micro_bsz, seq_len, ), dtype=np.int32) , )
            integration_timesteps = (np.ones((micro_bsz, seq_len, )), )

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)

    # jax.debug.print("Dummy input shapes (msg,book) ({}, \n {})",dummy_input[0].shape,dummy_input[1].shape)
    #RNN mode and initialisation needs to go in here if we need it. 

    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           *dummy_input, *integration_timesteps,
                           method='__call_ar__' 
                           )
    
    if batchnorm:
        params = variables["params"]#.unfreeze()
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"]#.unfreeze()
        # Note: `unfreeze()` is for using Optax.

    print(params['message_encoder']['encoder']['embedding'].shape)

    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.

        Using optax schedules (MaxText way):
        - Schedules are passed directly to optimizers (no inject_hyperparams)
        - LR is automatically computed from state.step
        """
        print("configuring standard optimization setup (with optax schedules)")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "ssm": optax.adam(learning_rate=ssm_lr_schedule),
                "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.

        Using optax schedules (MaxText way):
        - "none" group (B): uses ssm_lr_schedule WITH weight decay
        """
        print("configuring optimization with B in AdamW setup (with optax schedules)")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.adamw(learning_rate=ssm_lr_schedule, weight_decay=weight_decay),
                "ssm": optax.adam(learning_rate=ssm_lr_schedule),
                "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply
           faster global learning rate to B also.

        Using optax schedules (MaxText way):
        - "none" group: constant 0.0 (disabled)
        - "ssm" group: uses ssm_lr_schedule
        - "regular" group: uses lr_schedule WITH weight decay
        """
        print("configuring optimization with B in AdamW setup with lr (with optax schedules)")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.adamw(learning_rate=0.0, weight_decay=0.0),
                "ssm": optax.adam(learning_rate=ssm_lr_schedule),
                "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included
            with the SSM parameters and uses ssm learning rate.

        Using optax schedules (MaxText way):
        - "none" group: constant 0.0 (disabled)
        - "ssm" group (B, C, D, Lambda, log_step, norm): uses ssm_lr_schedule, NO weight decay
        - "regular" group: uses lr_schedule WITH weight decay
         """
        print("configuring optimization with C not in AdamW setup (with optax schedules)")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "ssm": optax.adam(learning_rate=ssm_lr_schedule),
                "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    total_params = sum(jax.tree_util.tree_leaves(param_sizes))
    print(f"[*] Trainable Parameters: {total_params}")

    # Initialize mesh first
    try:
        mesh = get_global_mesh()
        print("[State] Using existing global mesh")
    except RuntimeError:
        mesh = initialize_mesh(num_devices)
        print("[State] Created new global mesh")

    # MaxText approach: Wrap state creation in JIT with out_shardings
    # This ensures each buffer is uniquely allocated, preventing XLA-level aliasing
    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any

        def create_state_fn():
            return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        def create_state_fn():
            return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # 1. Get abstract state shape to create shardings
    abstract_state = jax.eval_shape(create_state_fn)
    print(abstract_state.params['message_encoder']['encoder']['embedding'].shape)

    # 2. Create shardings based on abstract state
    state_shardings = create_state_shardings(abstract_state, mesh)

    # 3. JIT-compile state creation with explicit out_shardings
    # This forces JAX to allocate unique device buffers for each leaf
    print("[State] Creating state via JIT with out_shardings (ensures unique buffers)")
    state = jax.jit(
        create_state_fn,
        out_shardings=state_shardings,
    )()

    print(state.params['message_encoder']['encoder']['embedding'].shape)

    return state, total_params

def get_slices(dims):
    slices = []
    last_i = 0
    for d in dims:
        slices.append(slice(last_i, last_i+d))
        last_i += d
    return slices

# Train and eval steps
# @partial(np.vectorize, signature="(c),()->()")
# def cross_entropy_loss(logits, label):
#     one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[-1])
#     return -np.sum(one_hot_label * logits)

@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    return -np.sum(logits[label])


@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss_test(logits, label):
    return -np.sum(logits)

@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label

def prep_batch(
        batch: Union[
            Tuple[onp.ndarray, onp.ndarray, Dict[str, onp.ndarray]],
            Tuple[onp.ndarray, onp.ndarray]],
        seq_len: int,
        # in_dim: int,
        num_devices: int,
    ) -> Tuple[Tuple, np.ndarray, Tuple]:

    if len(batch) == 2:
        inputs, targets = batch
        book_data, timestep_msg, timestep_book = None, None, None
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
        book_data = aux_data.get("book_data", None)
        timestep_msg = aux_data.get("timesteps_msg", None)
        timestep_book = aux_data.get("timesteps_book", None)            
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # ========================================================================
    # Old (pmap): reshape to (num_devices, batch_per_device, ...) and use pmap
    # ========================================================================
    # inputs, targets, book_data, timestep_msg, timestep_book = device_reshape(...)
    # inputs, labels, integration_times = _prep_batch_par(...)

    # ========================================================================
    # New (jit+shardings): keep (global_batch, ...) shape, no device dimension
    # ========================================================================
    # Prepare batch data directly without device-specific reshaping
    # JAX sharding will automatically distribute data across devices

    assert inputs.shape[1] == seq_len, f'inputs: {inputs.shape} seq_len {seq_len}'

    # Compute integration timesteps
    if timestep_msg is not None:
        integration_timesteps = (np.diff(np.asarray(timestep_msg)), )
    else:
        integration_timesteps = (np.ones((len(inputs), seq_len)), )

    # Prepare full inputs (messages + optional book data)
    if book_data is not None:
        full_inputs = (inputs.astype(np.int32), book_data)
        if timestep_book is not None:
            integration_timesteps += (np.diff(timestep_book), )
        else:
            integration_timesteps += (np.ones((len(inputs), seq_len)), )
    else:
        full_inputs = (inputs.astype(np.int32), )

    # Prepare labels
    labels = np.squeeze(targets.astype(np.int32))

    return full_inputs, labels, integration_timesteps

# ============================================================================
# Old _prep_batch_par (pmap version) - NO LONGER NEEDED
# ============================================================================
# @partial(jax.pmap, axis_name="batch_devices", ...)
# def _prep_batch_par(inputs, targets, seq_len, ...):
#     """Prepare batch per device (pmap version)."""
#     # Logic now inlined in prep_batch above
#     ...

# Note: The batch preparation logic from _prep_batch_par has been inlined
# into prep_batch above, without the device dimension handling

# ============================================================================
# Old device_reshape (pmap version) - NO LONGER NEEDED
# ============================================================================
# @partial(jax.jit, static_argnums=(0,), backend='gpu')
# def device_reshape(num_devices, inputs, targets, ...):
#     """Reshape to (num_devices, batch_per_device, ...) for pmap."""
#     inputs = np.reshape(inputs, (num_devices, -1, *inputs.shape[1:]))
#     ...

# ============================================================================
# New: jit+shardings handles distribution automatically, no reshape needed
# ============================================================================


def print_memory_usage():
    """Print GPU and system memory usage"""
    process = psutil.Process(os.getpid())
    print(f"CPU Memory: {process.memory_info().rss / 1024 ** 3:.2f} GB")
    
    # JAX device memory
    for device in jax.local_devices()[:1]:
        try:
            stats = device.memory_stats()
            if stats:
                print(f"Device {device} Used: {stats['bytes_in_use'] / 1024**2:.2f} MB / {stats['bytes_limit'] / 1024**3:.2f} GB")
        except:
            pass

def print_memory_usage_tofile():
    """Print GPU and system memory usage to a file"""
    process = psutil.Process(os.getpid())
    with open('/tmp/memory_usage.txt', 'a') as f:
        f.write(f"CPU Memory: {process.memory_info().rss / 1024 ** 3:.2f} GB\n")
        
        # JAX device memory
        for device in jax.local_devices()[:1]:
            try:
                stats = device.memory_stats()
                if stats:
                    f.write(f"Device {device} Used: {stats['bytes_in_use'] / 1024**2:.2f} MB / {stats['bytes_limit'] / 1024**3:.2f} GB\n")
            except:
                pass


class MFUTracker:
    """
    MFU (Model FLOPs Utilization) Calculation Formulas
    ==================================================

    This document records the difference between the naive (linear only) FLOPs calculation
    and the accurate (Transformer-aware) FLOPs calculation used in LOBMAX.

    1. Naive / Linear-Only Formula (Original)
    -----------------------------------------
    Used for RNNs or models where linear projections dominate.
    Completely ignores the quadratic cost of Attention.

        FLOPS_Linear = 6 * N * B * S

        Where:
        - N: Number of Model Parameters
        - B: Global Batch Size
        - S: Sequence Length (e.g., 12000)
        - 6 factor: 2 FLOPs (mul+add) * 3 (Forward + Backward pass approximation)

        ISSUE: For long sequences (S=12000), this severely underestimates the compute load,
               leading to artificially low MFU numbers (e.g., 1.8%).

    2. Transformer-Aware Formula (Corrected)
    ----------------------------------------
    Used for Transformers (LOBMAX), accounting for Attention's O(S^2) complexity.

        FLOPS_Total = FLOPS_Linear + FLOPS_Attention

        FLOPS_Attention = 12 * L * H * Q * B * S^2

        Where:
        - L: Number of Layers (e.g., 12)
        - H: Number of Heads (e.g., 12)
        - Q: Head Dimension (d_model / H, e.g., 64)
        - 12 factor:
            - 4 * S^2 for Forward pass (QK^T and Softmax*V, roughly 2 matmuls of S*S)
            - 8 * S^2 for Backward pass (approx 2x Forward)

    IMPACT: For LOBMAX 125M with S=12000:
    - FLOPS_Linear: ~360 TFLOPS
    - FLOPS_Attention: ~510 TFLOPS
    - FLOPS_Total: ~870 TFLOPS
    
    The Attention mechanism adds >140% more FLOPs.
    The corrected MFU reflects true hardware utilization ~2.4x higher than the linear formula.
    """
    def __init__(self, model_params, batch_size, seq_len, num_devices, peak_tflops=1000.0, window=10,
                 num_layers=None, num_heads=None, d_model=None):
        
        # Base Linear FLOPs (6NBS)
        self.flops_per_step = 6 * batch_size * seq_len * model_params
        
        # Add Attention FLOPs if transformer params provided
        if num_layers is not None and num_heads is not None and d_model is not None:
            L = num_layers
            H = num_heads
            Q = d_model // num_heads
            B = batch_size
            S = seq_len
            
            # Attention FLOPs: 12 * L * H * Q * B * S^2
            att_flops = 12 * L * H * Q * B * (S**2)
            
            print(f"[MFUTracker] Transformer detected.")
            print(f"  Linear FLOPs: {self.flops_per_step/1e12:.2f} TFLOPS")
            print(f"  Attn FLOPs:   {att_flops/1e12:.2f} TFLOPS")
            print(f"  Total FLOPs:  {(self.flops_per_step + att_flops)/1e12:.2f} TFLOPS")
            
            self.flops_per_step += att_flops

        self.total_peak = peak_tflops * num_devices
        self.window = []
        self.window_size = window
        self.last_time = None
        self.step = 0

    def tick(self):
        """Call after each training step. Returns smoothed MFU% or None if not ready."""
        now = time.time()
        self.step += 1
        if self.last_time is None or self.step <= 1:
            self.last_time = now
            return None
        dt = now - self.last_time
        self.last_time = now
        if dt <= 0:
            return None
        mfu = (self.flops_per_step / dt / 1e12) / self.total_peak * 100
        self.window.append(mfu)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        return sum(self.window) / len(self.window)


def train_epoch(
        state,
        rng,
        trainloader,
        seq_len,
        batchnorm,
        # lr_params REMOVED - LR scheduling handled by optax
        num_devices,
        debug_loading,
        debug_profiler,
        curtail_epochs,
        init_hiddens,
        epoch,
        ignore_times,
        log_ce_tables,
        jit_train_step_fn=None,
        # MFU tracking parameters
        model_params=None,
        batch_size=None,
        peak_tflops=1000.0,
        goodput_monitor=None,
        # Transformer params for accurate MFU
        num_layers=None,
        num_heads=None,
        d_model=None,
        # Step-level checkpointing parameters
        checkpoint_callback=None,  # Callable: (state, epoch, step, loss) -> None
        checkpoint_every_n_steps=1000,  # Save every N steps, or "auto" for ~1 hour intervals
        job_start_time=None,  # Job start time for time-aware checkpointing
        max_job_hours=24.0,  # Maximum job duration in hours
        save_before_timeout_minutes=30,  # Save checkpoint this many minutes before timeout
        checkpoint_every_n_steps=1000,  # Save every N steps, or "auto" for ~1 hour intervals
        job_start_time=None,  # Job start time for time-aware checkpointing
        max_job_hours=24.0,  # Maximum job duration in hours
        save_before_timeout_minutes=30,  # Save checkpoint this many minutes before timeout
        mesh=None,  # JAX Mesh for multi-GPU sharding - CRITICAL for data parallelism
        monitor_step_loss=False,  # If True, block and print loss every step (slow)
    ):

    """
     ... (existing docstring) ...
    """
    
    # ... (existing setup code until loop) ...
    # Store Metrics
    batch_losses = []
    cross_entropies= [] #list of 1xNTok losses

    # Initialize MFU tracker if parameters provided
    mfu_tracker = None
    if model_params is not None and batch_size is not None:
        mfu_tracker = MFUTracker(
            model_params, batch_size, seq_len, num_devices, peak_tflops,
            num_layers=num_layers, num_heads=num_heads, d_model=d_model
        )

    # =========================================================================
    # AUTO MODE TIMING (WALL CLOCK):
    # =========================================================================
    auto_checkpoint_mode = checkpoint_every_n_steps == "auto"
    if auto_checkpoint_mode:
        checkpoint_every_n_steps = 0
        last_checkpoint_time = time.time()
        last_wandb_log_time = time.time()
        auto_checkpoint_interval_seconds = 1800
        auto_wandb_log_interval_seconds = 600

    pbar = tqdm(trainloader)
    for batch_idx, batch in enumerate(pbar):
        if not debug_loading:
            if (state.step>1) & (state.step<3) & debug_profiler:
                jax.profiler.start_trace("/tmp/tensorboard")

            if goodput_monitor:
                with goodput_monitor.record('prep_batch'):
                    inputs, labels, integration_times = prep_batch(batch, seq_len, num_devices)
            else:
                inputs, labels, integration_times = prep_batch(batch, seq_len, num_devices)
            
            if mesh is not None:
                inputs_sharding, labels_sharding, timesteps_sharding = get_data_shardings_for_batch(
                    mesh, has_book_data=(len(inputs) > 1)
                )
                inputs = tuple(jax.device_put(inp, sh) for inp, sh in zip(inputs, inputs_sharding))
                labels = jax.device_put(labels, labels_sharding)
                integration_times = tuple(jax.device_put(ts, sh) for ts, sh in zip(integration_times, timesteps_sharding))
            
            rng, drop_rng = jax.random.split(rng)
            if batch_idx % 1000 == 0:
                print(f"\n=== Epoch {epoch}, Batch {batch_idx} ===")
                print_memory_usage()
            
            train_fn = jit_train_step_fn if jit_train_step_fn is not None else train_step

            if goodput_monitor:
                with goodput_monitor.record('train_step'):
                    state, loss, ce, logits = train_fn(
                        state, drop_rng, inputs, labels, integration_times, batchnorm, ignore_times
                    )
            else:
                if mesh is not None:
                    with jax.set_mesh(mesh):
                        state, loss, ce, logits = train_fn(
                            state, drop_rng, inputs, labels, integration_times, batchnorm, ignore_times
                        )
                else:
                    state, loss, ce, logits = train_fn(
                        state, drop_rng, inputs, labels, integration_times, batchnorm, ignore_times
                    )

            if debug_profiler:
                loss.block_until_ready()

            # Old (pmap): loss had device dimension, needed loss[0]
            # New (jit+shardings): loss is already a scalar, no indexing needed
            batch_losses.append(loss)
            if log_ce_tables:
                cross_entropies.append(ce)

            # Update tqdm with MFU and goodput metrics
            postfix = {}

            # PERFORMANCE NOTE: Loss Synchronization Control
            # ---------------------------------------------
            # Fetching the loss value (e.g., loss.item() or computing str(loss)) requires transferring 
            # data from the accelerator (GPU/TPU) to the host CPU. This operation acts as a 
            # SYNCHRONIZATION BARRIER, forcing the host to wait for the device to finish the current 
            # step's computation before proceeding. 
            #
            # When monitor_step_loss is False (recommended for speed), we skip this fetch, allowing 
            # JAX to dispatch operations asynchronously. The device can process instructions ahead 
            # of the host ("compute overlap"), preventing device starvation and significantly 
            # improving training throughput (steps/sec).
            if monitor_step_loss:
                try:
                    postfix['loss'] = f'{loss:.4f}'
                except:
                    postfix['loss'] = "nan"
            
            # Periodically force sync for logging (every 100 steps) if not monitoring every step
            elif batch_idx % 100 == 0:
                 try:
                    postfix['loss'] = f'{loss:.4f}'
                 except:
                    pass

            if mfu_tracker is not None:
                mfu = mfu_tracker.tick()
                if mfu is not None:
                    postfix['MFU'] = f'{mfu:.1f}%'

            if goodput_monitor:
                prep_time = goodput_monitor.get_last('prep_batch')
                step_time = goodput_monitor.get_last('train_step')
                if prep_time is not None:
                    postfix['prep'] = f'{prep_time*1000:.1f}ms'
                if step_time is not None:
                    postfix['step'] = f'{step_time*1000:.0f}ms'

            if postfix:
                pbar.set_postfix(postfix)


            # No more manual LR updates - optax schedules handle this automatically!
            # No more buffer copying needed - eliminates donate_argnums aliasing

            # =========================================================================
            # TIMING LOGIC (AUTO MODE):
            #   - WANDB LOSS LOGGING: EVERY 10 MINUTES
            #   - CHECKPOINT SAVING:  EVERY 30 MINUTES
            # =========================================================================
            should_checkpoint = False
            should_wandb_log = False
            timeout_imminent = False

            # Check if we should save at regular step intervals (manual mode)
            if checkpoint_callback is not None and checkpoint_every_n_steps > 0:
                if (batch_idx + 1) % checkpoint_every_n_steps == 0:
                    should_checkpoint = True
                    should_wandb_log = True

            # Check if we should log/save based on wall clock time (auto mode)
            if checkpoint_callback is not None and auto_checkpoint_mode:
                now = time.time()
                # WANDB LOGGING: EVERY 10 MINUTES
                if now - last_wandb_log_time >= auto_wandb_log_interval_seconds:
                    should_wandb_log = True
                # CHECKPOINT SAVING: EVERY 30 MINUTES
                if now - last_checkpoint_time >= auto_checkpoint_interval_seconds:
                    should_checkpoint = True
                    should_wandb_log = True  # Also log when saving

            # Check if we're approaching the time limit
            if checkpoint_callback is not None and job_start_time is not None:
                elapsed_hours = (time.time() - job_start_time) / 3600.0
                remaining_hours = max_job_hours - elapsed_hours
                remaining_minutes = remaining_hours * 60

                if remaining_minutes <= save_before_timeout_minutes:
                    should_checkpoint = True
                    should_wandb_log = True
                    timeout_imminent = True

            # Execute callback (with save_checkpoint flag)
            if should_wandb_log or should_checkpoint:
                current_loss = float(loss)
                elapsed_mins = (time.time() - job_start_time) / 60.0 if job_start_time else 0
                if should_checkpoint:
                    print(f"\n[Checkpoint] SAVING at epoch {epoch+1}, step {batch_idx+1}, loss={current_loss:.4f}, elapsed={elapsed_mins:.1f}min")
                else:
                    print(f"\n[WandB Log] Logging at epoch {epoch+1}, step {batch_idx+1}, loss={current_loss:.4f}, elapsed={elapsed_mins:.1f}min")
                # CALLBACK SIGNATURE: (state, epoch, step, loss, save_checkpoint)
                checkpoint_callback(state, epoch, batch_idx, current_loss, should_checkpoint)

                # Update timing for auto mode
                if auto_checkpoint_mode:
                    if should_wandb_log:
                        last_wandb_log_time = time.time()
                    if should_checkpoint:
                        last_checkpoint_time = time.time()

                if timeout_imminent:
                    print(f"[Checkpoint] Timeout imminent! Saved checkpoint and exiting.")
                    print(f"[Checkpoint] Resume from: epoch={epoch}, step={batch_idx+1}")
                    # Return early with partial epoch results
                    loss_mean = np.mean(np.array(batch_losses)) if batch_losses else float('nan')
                    return state, loss_mean, None, batch_idx + 1  # Return step for resume

            if (state.step>20) & (state.step<=21) & debug_profiler:
                jax.profiler.stop_trace()
                break
            if (curtail_epochs is not None) and (batch_idx>=curtail_epochs):
                print("Ending epoch early due to curtail_epochs being ",curtail_epochs)
                break
        else:
            continue
        
    
        
    # Return average loss over batches
    if log_ce_tables:
        ce_means=np.mean(np.concatenate(cross_entropies,axis=0),axis=0)
    else:
        ce_means=None
    # jax.debug.print("CE of epoch by token: {}",ce_means.shape)
    loss_mean=np.mean(np.array(batch_losses))
    # No more returning step - optax tracks it internally via state.step
    # Return None for completed_step to indicate full epoch completed
    return state, loss_mean, ce_means, None


@partial(jax.vmap,in_axes=(0,0,None),out_axes=(0,0))
@partial(jax.jit,static_argnums=(2,))
def repeat_book(msg,book,shift_start):
    #DEFINITION OF START BOOK:
    # print("checking for compile in repeat_book")
    if msg.shape[0]>book.shape[0]:
        book = np.repeat(book, (msg.shape[0]) // book.shape[0], axis=0)
    # if shift_start:
    #     pad=book[:1]
    #     #FIXME: Wrong logic, needs to be the init book state.
    #     # book=np.concatenate([book[:1],book[1:]])
    #     book=np.concatenate([pad,book[:-1]])
    return (msg,book)

# ============================================================================
# Old train_step (pmap version) - Commented out
# ============================================================================
# @partial(
#     jax.pmap,
#     axis_name="batch_devices",
#     static_broadcasted_argnums=(5,6),  # TODO: revert to 5 for batchnorm in pmap
#     in_axes=(0, None, 0, 0, 0, None, None),
#     # out_axes=(0, 0),
#     # devices=global_devices
# )

# ============================================================================
# New train_step (jit + shardings version)
# ============================================================================
def train_step(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 1
        batch_inputs: Tuple[jax.Array, jax.Array], # 2
        batch_labels: jax.Array, # 3
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 4
        batchnorm: bool, # 5
        ignore_times:bool, #6
    ):
    """
    Training step function (jit + shardings version).

    Main changes:
    1. Removed pmap decorator, using jax.jit + in_shardings/out_shardings
    2. Removed jax.lax.pmean, automatic cross-device aggregation
    3. state no longer has device dimension (replicated via sharding)

    Why these changes:
    - pmap implicitly parallelizes over first axis, jit + shardings uses explicit sharding specs
    - pmap requires pmean for cross-device aggregation, jit + shardings handles this automatically
    - These changes make parallelism strategy more flexible (easy to add FSDP in future)
    """

    # Print hash values of static arguments
    # print(f"batchnorm hash: {batchnorm.__hash__()}")
    # print(f"ignore_times hash: {ignore_times.__hash__()}")
    # print('checking for compile in train_step')

    batch_inputs=repeat_book(*batch_inputs,True)
    # batch_integration_timesteps=repeat_book(*batch_integration_timesteps)

    def loss_fn(params):
        # print('checking for compile in loss_fn')
        if batchnorm:
            logits, mod_vars = state.apply_fn( 
                {"params": params, "batch_stats": state.batch_stats},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
                method='__call_ar__'
            )
        else:
            logits, mod_vars = state.apply_fn(
                {"params": params},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
                method='__call_ar__'
            )


        # jax.debug.print("Shape of Logits: {}",logits.shape)
        # jax.debug.print("Shape of Labels: {}", batch_labels.shape)

        
        ce=cross_entropy_loss(logits, batch_labels)
        if ignore_times:
            ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
            ce_1=ce[:,:,:TIME_START_I]
            ce_2=ce[:,:,(TIME_END_I+1):]
            ce=np.concatenate([ce_1,ce_2],axis=2)
            ce=ce.reshape(ce.shape[0],-1)

        ce=np.mean(ce,axis=0)
        # jax.debug.print("Shape of CE: {}", ce.shape)
        # average cross-ent loss
        loss = np.mean(ce)
        # jax.debug.print("Shape of loss: {}", loss.shape)
        return loss, (mod_vars, logits,ce)

    (loss, (mod_vars, logits,ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)



    # UPDATE
    # Old way (pmap): Use pmean for cross-device averaging
    # loss = jax.lax.pmean(loss, axis_name="batch_devices")
    # grads = jax.lax.pmean(grads, axis_name="batch_devices")
    # ce = jax.lax.pmean(ce, axis_name="batch_devices")

    # New way (jit + shardings with NamedSharding):
    # When using jax.jit with in_shardings/out_shardings:
    # - Input batch is sharded along 'data' axis (each device gets batch/n_devices samples)
    # - Forward pass computes on each shard independently
    # - Gradients are computed locally on each device's shard
    # - out_shardings=replicated forces JAX to all-reduce gradients automatically
    # - The key is that we compute mean over the LOCAL batch, then JAX averages across devices
    #
    # Note: jax.lax.psum requires axis_name from shard_map/pmap, which we don't have here.
    # Instead, the aggregation happens implicitly via sharding constraints in out_shardings.

    if batchnorm:
        # Old way: mod_vars = jax.lax.pmean(mod_vars, axis_name="batch_devices")
        # New way: batch_stats automatically aggregated
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    #return loss, mod_vars, grads, state
    return state, loss, ce, logits


# ============================================================================
# Create JIT-compiled train_step with shard_map for proper gradient sync
# ============================================================================
def create_jit_train_step(mesh: Mesh, state: train_state.TrainState, has_book_data: bool = True):
    """
    Create JIT-compiled train_step with shard_map for multi-GPU gradient sync.

    Key changes from simple jax.jit:
    - Uses shard_map to provide axis_name for psum
    - Enables proper gradient averaging across devices
    - Better MFU through explicit sharding control

    Args:
        mesh: JAX Mesh
        state: Example state (for inferring sharding)
        has_book_data: Whether book data is present

    Returns:
        JIT-compiled train_step function with gradient sync
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    
    # ========================================================================
    # DEBUG: Check for buffer aliasing in state before JIT compilation
    # ========================================================================
    print("\n[DEBUG] Checking for buffer aliasing in state...")
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(state)
    buffer_ids = {}
    for idx, (path, leaf) in enumerate(leaves_with_paths):
        if isinstance(leaf, jax.Array):
            buf_id = id(leaf)
            if buf_id not in buffer_ids:
                buffer_ids[buf_id] = []
            path_str = jax.tree_util.keystr(path)
            buffer_ids[buf_id].append((idx, path_str, leaf.shape))

    aliased = {bid: locs for bid, locs in buffer_ids.items() if len(locs) > 1}
    if aliased:
        print(f"[DEBUG] FOUND {len(aliased)} ALIASED BUFFERS:")
        for buf_id, locations in aliased.items():
            indices = [idx for idx, _, _ in locations]
            if 8 in indices and 18 in indices:
                print(f"\n[DEBUG] *** CULPRIT FOUND ***")
            print(f"  Buffer {buf_id} at indices {indices}:")
            for idx, path, shape in locations:
                print(f"    [{idx:3d}] {path} | shape={shape}")
    else:
        print(f"[DEBUG] No aliasing found - all buffers unique")
    print(f"[DEBUG] Total leaves: {len(leaves_with_paths)}, Unique buffers: {len(buffer_ids)}\n")
    # ========================================================================

    # 1. Create shardings for state (everything replicated)
    state_shardings = create_state_shardings(state, mesh)

    # 2. Create shardings for data
    inputs_shardings, labels_sharding, timesteps_shardings = get_data_shardings_for_batch(
        mesh, has_book_data=has_book_data
    )

    # 3. Define in_shardings
    in_shardings = (
        state_shardings,          # param 0: state - replicated
        None,                     # param 1: rng - replicated (None = default)
        inputs_shardings,         # param 2: batch_inputs - sharded
        labels_sharding,          # param 3: batch_labels - sharded
        timesteps_shardings,      # param 4: batch_integration_timesteps - sharded
    )

    # 4. Define out_shardings
    out_shardings = (
        state_shardings,          # state - replicated
        None,                     # loss - scalar, auto-handled
        None,                     # ce - small array, auto-handled
        None,                     # logits - inferred from inputs
    )

    # 5. Create JIT-compiled function
    # For single-device or when we want simpler code path, use standard jit
    num_devices = mesh.shape.get('data', 1)
    
    if num_devices > 1:
        print(f"[JIT] Multi-device mode ({num_devices} GPUs): Using gradient sync")
        
        # Create a wrapper that adds gradient sync after train_step
        def train_step_with_grad_sync(state, rng, batch_inputs, batch_labels, 
                                       batch_integration_timesteps, batchnorm, ignore_times):
            """Wrapper that adds gradient synchronization for multi-GPU."""
            # Run the base train_step
            new_state, loss, ce, logits = train_step(
                state, rng, batch_inputs, batch_labels,
                batch_integration_timesteps, batchnorm, ignore_times
            )
            # Note: With jax.jit + sharding, the gradient sync happens automatically
            # when out_shardings specifies replicated parameters.
            # The key is that we compute on sharded data and output to replicated state.
            return new_state, loss, ce, logits
        
        jit_train_step = jax.jit(
            train_step_with_grad_sync,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=(5, 6),
            donate_argnums=(0,),
        )
    else:
        print("[JIT] Single-device mode")
        jit_train_step = jax.jit(
            train_step,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=(5, 6),
            donate_argnums=(0,),
        )

    print("[JIT] Created JIT-compiled train_step")
    print(f"[JIT] in_shardings: state=replicated, data=sharded on 'data' axis")
    print(f"[JIT] out_shardings: state=replicated, metrics=auto")
    print(f"[JIT] donate_argnums: (0,) = state (memory optimization)")

    return jit_train_step


# ============================================================================
# Deleted: train_step_rnn and train_step_old (old pmap versions)
# These functions used jax.pmap and are no longer needed with jit+shardings
# ============================================================================


def validate(state,
             apply_fn,
             testloader,
             seq_len,
             in_dim,
             batchnorm,
             num_devices,
             epoch,
             curtail_epoch=None,
             ignore_times: bool =False,
             step_rescale=1.0,
             apply_method: str ='__call_ar__',
             init_hiddens=(np.array([0])),
             log_ce_tables : bool =False,
             eval_step_fn=None):
    """Validation function that loops over batches

    Args:
        eval_step_fn: Optional JIT-compiled eval_step function. If None, falls back
                      to un-JIT'd eval_step (slow - causes recompilation each batch).
    """
    # Use JIT-compiled eval_step if provided, otherwise fall back to raw eval_step
    _eval_step = eval_step_fn if eval_step_fn is not None else eval_step

    # losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    losses, accuracies, preds = [], [], []
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, num_devices)
        # print("eval step with method: ", apply_method)
        # print("Validataion: Inputs 0:5:", inputs[0][0,0:5,:])
        loss, acc, pred = _eval_step(
            inputs, labels, integration_timesteps, state, apply_fn, batchnorm,apply_method,init_hiddens,ignore_times)
        # losses = np.append(losses, loss)
        # accuracies = np.append(accuracies, acc)

        # if (batch_idx==0) & (epoch%100==0): 
        #     np.set_printoptions(threshold=sys.maxsize)
        #     with open(f'/data1/sascha/data/losses/losses_batch_{batch_idx}_testing_applying_{apply_method}.txt', 'w') as f:
        #         print(loss, file=f)
        #     print("Printing logits of shape ", pred.shape, " to file")
        #     with open(f'/data1/sascha/data/losses/logits_batch_{batch_idx}_testing_applying_{apply_method}.txt', 'w') as f:
        #         print(pred[0,0,0:44,:], file=f)
        #     np.set_printoptions()
        #     print("Done Printing")


        losses.append(loss)
        accuracies.append(acc)
        if curtail_epoch is not None and batch_idx>=curtail_epoch:
            print(f"Ending epoch early at step {batch_idx} due to curtail_epoch arg.")
            break

    concat_loss=np.concatenate(losses,axis=0)
    concat_acc=np.concatenate(accuracies,axis=0)
    print(f"Concat Loss is {concat_loss.shape}")
    print(f"Concat Acc is {concat_acc.shape}")
    if log_ce_tables:
        acc_means=np.mean(concat_acc,axis=(0,1))
        ce_means=np.mean(concat_loss,axis=(0,1))
    else:
        ce_means=None
        acc_means=None
    aveloss, aveaccu = np.mean(concat_loss), np.mean(np.asarray(accuracies))
    del losses, accuracies
    return aveloss, aveaccu, ce_means,acc_means

# ============================================================================
# Old eval_step (pmap version) - Commented out
# ============================================================================
# @partial(
#     jax.pmap,
#     axis_name="batch_devices",
#     static_broadcasted_argnums=(4,5,6,8),
#     in_axes=(0, 0, 0, 0, None, None, None,None,None),
#     # devices=global_devices
# )

# ============================================================================
# New eval_step (jit + shardings version)
# ============================================================================
def eval_step(
        batch_inputs,
        batch_labels,
        batch_integration_timesteps,
        state,
        #model,
        apply_fn,
        batchnorm,
        apply_method,
        init_hiddens,
        ignore_times,
    ):
    """
    Evaluation step function (jit + shardings version).

    Main changes:
    1. Removed pmap decorator, using jax.jit + in_shardings/out_shardings
    2. No pmean needed (eval_step originally had no cross-device aggregation)

    Why these changes:
    - Maintain consistent parallelism strategy with train_step
    - Use same sharding infrastructure
    """
    # print("checking for compile in eval_step function")


    batch_inputs=repeat_book(*batch_inputs,True)

    if apply_method == '__call_ar__':
        if batchnorm:
            logits = apply_fn({"params": state.params, "batch_stats": state.batch_stats},
                                *batch_inputs, *batch_integration_timesteps,
                                method=apply_method,
                                )
        else:
            logits = apply_fn({"params": state.params},
                                *batch_inputs, *batch_integration_timesteps,
                                method=apply_method,
                                )
    elif apply_method == '__call_rnn__':
        dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*3

        if batchnorm:
            hiddens,logits=apply_fn(
                        {"params": state.params, "batch_stats": state.batch_stats},
                        init_hiddens,
                        *batch_inputs,
                        *dones,
                        *batch_integration_timesteps,
                        method='__call_rnn__'
                    )
        else:
            hiddens,logits=apply_fn(
                        {"params": state.params},
                        init_hiddens,
                        *batch_inputs,
                        *dones,
                        *batch_integration_timesteps,
                        method='__call_rnn__'
                    )
    elif apply_method == 'scan_rnn':
        dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*3
        hiddens,logits=eval_rnn_scan(apply_fn,
                                     init_hiddens,
                                     state,
                                     batch_inputs,
                                     dones,
                                     batch_integration_timesteps,
                                     batchnorm)



    losses = cross_entropy_loss(logits, batch_labels)  
    if ignore_times:
        ce=losses
        ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
        ce_1=ce[:,:,:TIME_START_I]
        ce_2=ce[:,:,(TIME_END_I+1):]
        ce=np.concatenate([ce_1,ce_2],axis=2)
        ce=ce.reshape(ce.shape[0],-1)
        losses=ce
    accs = compute_accuracy(logits, batch_labels)
    if ignore_times:
        ce=accs
        ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
        ce_1=ce[:,:,:TIME_START_I]
        ce_2=ce[:,:,(TIME_END_I+1):]
        ce=np.concatenate([ce_1,ce_2],axis=2)
        ce=ce.reshape(ce.shape[0],-1)
        accs=ce

    return losses, accs, logits


# ============================================================================
# Create JIT-compiled eval_step
# ============================================================================
def create_jit_eval_step(mesh: Mesh, state: train_state.TrainState, has_book_data: bool = True):
    """
    Create JIT-compiled eval_step.

    Why needed:
    - eval_step also needs jax.jit + shardings for consistency
    - Though eval doesn't need donate_argnums (doesn't update state), still needs correct sharding

    Args:
        mesh: JAX Mesh
        state: Example state (for inferring sharding)
        has_book_data: Whether book data is present

    Returns:
        JIT-compiled eval_step function
    """
    # 1. Create shardings for state (everything replicated)
    state_shardings = create_state_shardings(state, mesh)

    # 2. Create shardings for data
    inputs_shardings, labels_sharding, timesteps_shardings = get_data_shardings_for_batch(
        mesh, has_book_data=has_book_data
    )

    # 3. Define in_shardings
    # IMPORTANT: in_shardings only includes NON-STATIC parameters!
    # Order corresponds to eval_step NON-STATIC parameters:
    # (batch_inputs, batch_labels, batch_integration_timesteps, state, init_hiddens)
    # apply_fn, batchnorm, apply_method, ignore_times are static - NOT included!
    # Note: init_hiddens is NOT static because JAX arrays are not hashable
    # Use None to let JAX infer sharding (init_hiddens can be array or complex pytree)
    in_shardings = (
        inputs_shardings,         # param 0: batch_inputs - sharded
        labels_sharding,          # param 1: batch_labels - sharded
        timesteps_shardings,      # param 2: batch_integration_timesteps - sharded
        state_shardings,          # param 3: state - replicated
        None,                     # param 7: init_hiddens - auto (None lets JAX handle pytree of arrays)
        # params 4-6, 8 (apply_fn, batchnorm, apply_method, ignore_times) are static!
    )

    # 4. Define out_shardings
    # Order corresponds to return values: (losses, accs, logits)
    out_shardings = (
        None,                     # losses - auto-handled
        None,                     # accs - auto-handled
        None,                     # logits - auto-handled
    )

    # 5. Create JIT-compiled function
    # Note: eval doesn't donate state because state is not modified
    jit_eval_step = jax.jit(
        eval_step,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=(4, 5, 6, 8),  # apply_fn, batchnorm, apply_method, ignore_times (NOT init_hiddens - JAX arrays not hashable)
        # Don't use donate_argnums because eval doesn't modify state
    )

    print("[JIT] Created JIT-compiled eval_step")
    print(f"[JIT] eval - No donate_argnums (state is read-only)")

    return jit_eval_step


def eval_rnn_scan(apply_fn,hiddens,state,batch_inputs,batch_dones,batch_inttimes,batchnorm):
    def apply_fn_scan(carry,x):
        (hiddens,state)=carry
        (batch_inputs,batch_dones,batch_inttimes)=x
        if batchnorm:
            hiddens,logits=apply_fn(
                    {"params": state.params, "batch_stats": state.batch_stats},
                    hiddens,
                    *batch_inputs,
                    *batch_dones,
                    *batch_inttimes,
                    method='__call_rnn__'
                )
        else:
            hiddens,logits=apply_fn(
                    {"params": state.params},
                    hiddens,
                    *batch_inputs,
                    *batch_dones,
                    *batch_inttimes,
                    method='__call_rnn__'
                )
        return (hiddens,state),logits
    #FIXME : Poor practice, but just for debugging purposes. 
    Ntoks=11000

    init=(hiddens,state)
    xs=(batch_inputs,batch_dones,batch_inttimes)
    
    xs=jax.tree_util.tree_map(partial(swap_leading,Ntoks),xs)
    
    carry_out,logits=jax.lax.scan(apply_fn_scan,init,xs)
    (hiddens,state)
    logits=np.concatenate(logits,axis=-2)
    return hiddens,logits
    
def swap_leading(targetsize,x):
    x=np.expand_dims(x,0)
    x=np.swapaxes(x,0,x.shape.index(targetsize))
    return x




