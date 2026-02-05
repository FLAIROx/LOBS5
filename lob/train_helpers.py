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

    import time
    if isinstance(dummy_input, tuple) and len(dummy_input) > 0:
         pass

    # Sleep to allow distributed environment to settle (though CPU hack below is the real fix)
    # time.sleep(1)

    # Force initialization on CPU to avoid potential distributed GPU hangs
    # Observation: On GH200 multi-node, model.init() on default GPU device hangs during distributed init.
    # Attempting to run on CPU (even if it fails) seems to unblock the subsequent attempt.
    try:
        # Try to use CPU device
        cpu_back = jax.local_devices(backend='cpu')[0] if len(jax.local_devices(backend='cpu')) > 0 else None
        if cpu_back:
            with jax.default_device(cpu_back):
                 variables = model.init({"params": init_rng,
                                        "dropout": dropout_rng},
                                       *dummy_input, *integration_timesteps,
                                       method='__call_ar__' 
                                       )
        else:
            raise RuntimeError("No CPU backend found")
            
    except Exception as e:
        # print(f"[DEBUG-HELPER] model.init on CPU failed with error: {e}")
        # Fallback to default device
        variables = model.init({"params": init_rng,
                                "dropout": dropout_rng},
                               *dummy_input, *integration_timesteps,
                               method='__call_ar__' 
                               )
                               
    # print(f"[DEBUG-HELPER] model.init returned.")
    
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
    """Track and compute MFU (Model FLOPs Utilization) with sliding window average."""
    def __init__(self, model_params, batch_size, seq_len, num_devices, peak_tflops=1000.0, window=10):
        self.flops_per_step = 6 * batch_size * seq_len * model_params
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
        # Step-level checkpointing parameters
        checkpoint_callback=None,  # Callable: (state, epoch, step, loss) -> None
        checkpoint_every_n_steps=1000,  # Save every N steps, or "auto" for ~1 hour intervals
        job_start_time=None,  # Job start time for time-aware checkpointing
        max_job_hours=24.0,  # Maximum job duration in hours
        save_before_timeout_minutes=30,  # Save checkpoint this many minutes before timeout
        # TBPTT parameters
        use_tbptt=False,  # Enable TBPTT
        n_tbptt_chunks=4,  # Number of chunks for TBPTT gradient chunking
        tbptt_mode: str = "gradient_chunking",
        tbptt_window_size: int = 500,
        tbptt_hidden_in=None,
        tbptt_reset_every_epoch: bool = True,
    ):

    """
    Training function for an epoch that loops over batches.

    With optax schedules:
    - Learning rate is automatically computed from state.step by the optimizer
    - No manual lr_params needed
    - No update_learning_rate_per_step() calls needed
    - No buffer copying needed (eliminates donate_argnums aliasing)

    Step-level checkpointing:
    - checkpoint_callback: Called at intervals and before timeout
    - checkpoint_every_n_steps: Save every N steps (default: 1000)
    - job_start_time: For time-aware checkpointing (detect 24hr limit)
    - max_job_hours: Maximum job duration (default: 24.0)
    - save_before_timeout_minutes: Save this many minutes before timeout (default: 30)
    """
    # Store Metrics
    batch_losses = []
    cross_entropies= [] #list of 1xNTok losses

    # Initialize MFU tracker if parameters provided
    mfu_tracker = None
    if model_params is not None and batch_size is not None:
        mfu_tracker = MFUTracker(model_params, batch_size, seq_len, num_devices, peak_tflops)

    # =========================================================================
    # AUTO MODE TIMING (WALL CLOCK):
    #   - WANDB LOSS LOGGING: EVERY 10 MINUTES
    #   - CHECKPOINT SAVING:  EVERY 30 MINUTES
    # =========================================================================
    auto_checkpoint_mode = checkpoint_every_n_steps == "auto"
    if auto_checkpoint_mode:
        checkpoint_every_n_steps = 0  # Disable step-based, use time-based instead
        last_checkpoint_time = time.time()  # Track when last checkpoint was saved
        last_wandb_log_time = time.time()   # Track when last wandb log happened
        auto_checkpoint_interval_seconds = 1800  # 30 MINUTES FOR CHECKPOINT
        auto_wandb_log_interval_seconds = 600    # 10 MINUTES FOR WANDB LOGGING

    # No more lr_params unpacking - optax handles LR scheduling internally
    # Step tracking is done via state.step (maintained by optax)
    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    use_ar_hidden = use_tbptt and tbptt_mode == "ar_hidden"
    tbptt_hidden = tbptt_hidden_in
    if use_ar_hidden:
        if tbptt_hidden is None or tbptt_reset_every_epoch:
            tbptt_hidden = init_hiddens
        if tbptt_hidden is None:
            raise ValueError("tbptt_mode=ar_hidden requires init_hiddens to be provided")

    pbar = tqdm(trainloader)
    for batch_idx, batch in enumerate(pbar):
        # print(f"train_epoch: Epoch {epoch} - Batch {batch_idx} / {len(trainloader)}")
        # print(f"train_epoch: Batch input shape: {batch[0].shape}, batch target shape: {batch[1].shape}")
        if not debug_loading:
            if (state.step>1) & (state.step<3) & debug_profiler:
                jax.profiler.start_trace("/tmp/tensorboard")

            # Monitor prep_batch time if goodput_monitor provided
            if goodput_monitor:
                with goodput_monitor.record('prep_batch'):
                    inputs, labels, integration_times = prep_batch(batch, seq_len, num_devices)
            else:
                inputs, labels, integration_times = prep_batch(batch, seq_len, num_devices)
            # print("train_epoch: Prepared batch inputs shape:", inputs[0].shape)
            # print("train_epoch: Prepared batch labels shape:", labels.shape)
            # print("train_epoch: Inputs 0:5:", inputs[0][0,0:5,:])
            rng, drop_rng = jax.random.split(rng)
            # Print memory every 1000 steps
            if batch_idx % 1000 == 0:
                print(f"\n=== Epoch {epoch}, Batch {batch_idx} ===")
                print_memory_usage()
            
            # state,loss=train_step_rnn(                
            #     state,
            #     drop_rng,
            #     inputs,
            #     labels,
            #     integration_times,
            #     batchnorm,
            #     init_hiddens)

            # print("Gets to train")
            if use_ar_hidden:
                # Use JIT-compiled TBPTT AR-hidden train_step if provided
                train_fn = jit_train_step_fn if jit_train_step_fn is not None else train_step_tbptt_ar_hidden

                if goodput_monitor:
                    with goodput_monitor.record('train_step'):
                        state, loss, ce, logits, tbptt_hidden = train_fn(
                            state,
                            drop_rng,
                            inputs,
                            labels,
                            integration_times,
                            tbptt_hidden,
                            batchnorm,
                            ignore_times,
                            tbptt_window_size,
                        )
                else:
                    state, loss, ce, logits, tbptt_hidden = train_fn(
                        state,
                        drop_rng,
                        inputs,
                        labels,
                        integration_times,
                        tbptt_hidden,
                        batchnorm,
                        ignore_times,
                        tbptt_window_size,
                    )
            else:
                # Use JIT-compiled train_step if provided
                train_fn = jit_train_step_fn if jit_train_step_fn is not None else train_step

                # Monitor train_step time if goodput_monitor provided
                if goodput_monitor:
                    with goodput_monitor.record('train_step'):
                        state, loss, ce, logits = train_fn(
                            state,
                            drop_rng,
                            inputs,
                            labels,
                            integration_times,
                            batchnorm,
                            ignore_times,
                            use_tbptt,       # TBPTT: enable gradient chunking
                            n_tbptt_chunks,  # TBPTT: number of chunks
                        )
                else:
                    state, loss, ce, logits = train_fn(
                        state,
                        drop_rng,
                        inputs,
                        labels,
                        integration_times,
                        batchnorm,
                        ignore_times,
                        use_tbptt,       # TBPTT: enable gradient chunking
                        n_tbptt_chunks,  # TBPTT: number of chunks
                    )
            if debug_profiler:
                loss.block_until_ready()

            if use_ar_hidden:
                tbptt_hidden = jax.lax.stop_gradient(tbptt_hidden)
                # Reset hidden between batches (samples are independent)
                # Each sample is 10k messages from different time periods
                # Use init_hiddens (not None) to maintain pytree structure for jax.lax.scan
                tbptt_hidden = init_hiddens  # Reset to zero-init state

            # print("completes train step")
            # if (batch_idx==0) & (epoch%100==0):
            #     np.set_printoptions(threshold=sys.maxsize)
            #     with open(f'/data1/sascha/data/losses/losses_batch_{batch_idx}_training.txt', 'w') as f:
            #         print( ce, file=f)
            #     print("Printing logits of shape ", logits.shape, " to file")
            #     with open(f'/data1/sascha/data/losses/logits_batch_{batch_idx}_training.txt', 'w') as f:
            #         print( logits[0,0,0:44,:], file=f)
            #     np.set_printoptions()
            #     print('Done Printing')

            # Old (pmap): loss had device dimension, needed loss[0]
            # New (jit+shardings): loss is already a scalar, no indexing needed
            batch_losses.append(loss)
            if log_ce_tables:
                cross_entropies.append(ce)

            # Update tqdm with MFU and goodput metrics
            # NOTE: Loss is NOT shown here to avoid GPU sync overhead
            # Loss is accumulated in batch_losses and averaged at epoch end
            postfix = {}

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
                    return state, loss_mean, None, batch_idx + 1, (tbptt_hidden if use_ar_hidden else None)  # Return step for resume

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
    return state, loss_mean, ce_means, None, (tbptt_hidden if use_ar_hidden else None)


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
# TBPTT (Truncated Backpropagation Through Time) Gradient Chunking
# ============================================================================
#
# TBPTT splits long sequences into chunks for gradient computation, reducing
# XLA compilation memory. This implementation uses gradient chunking (not
# hidden state TBPTT), where each chunk independently computes gradients that
# are then accumulated.
#
# Memory reduction: O(1/n_chunks²)
#   - 4 chunks → ~6.25% of original memory (~94% reduction)
#   - 8 chunks → ~1.56% of original memory (~98% reduction)
#
# ============================================================================

def split_sequence_for_tbptt(
    batch_inputs: Tuple[jax.Array, jax.Array],
    batch_labels: jax.Array,
    batch_integration_timesteps: Tuple[jax.Array, jax.Array],
    n_chunks: int = 4,
    msg_len: int = 24,
) -> Tuple:
    """
    Split sequence data into chunks for TBPTT gradient accumulation.

    This is the jit+shardings version: inputs have shape (global_batch, L, ...)
    without the device dimension that pmap required.

    Args:
        batch_inputs: (messages, books) tuple
            messages: (BSZ, L) - message tokens
            books: (BSZ, L, book_dim) - book states
        batch_labels: (BSZ, L) - labels
        batch_integration_timesteps: (msg_times, book_times) tuple
        n_chunks: Number of chunks to split into
        msg_len: Tokens per message (24 for alignment)

    Returns:
        chunked_inputs: ((msgs_chunked, books_chunked), ...)
            msgs_chunked: (n_chunks, BSZ, chunk_size)
            books_chunked: (n_chunks, BSZ, chunk_size, book_dim)
        chunked_labels: (n_chunks, BSZ, chunk_size)
        chunked_times: ((msg_t_chunked, book_t_chunked), ...)
        chunk_size: Size of each chunk (for reference)
    """
    messages, books = batch_inputs
    msg_times, book_times = batch_integration_timesteps

    BSZ, L = messages.shape[:2]

    # Compute chunk size aligned to message boundaries
    n_messages = L // msg_len
    messages_per_chunk = n_messages // n_chunks
    chunk_size = messages_per_chunk * msg_len

    # ========== Chunk messages ==========
    # (BSZ, L) → (BSZ, n_chunks, chunk_size) → (n_chunks, BSZ, chunk_size)
    messages_chunked = messages.reshape(BSZ, n_chunks, chunk_size)
    messages_chunked = np.transpose(messages_chunked, (1, 0, 2))

    # ========== Chunk labels ==========
    labels_chunked = batch_labels.reshape(BSZ, n_chunks, chunk_size)
    labels_chunked = np.transpose(labels_chunked, (1, 0, 2))

    # ========== Chunk message times ==========
    msg_times_chunked = msg_times.reshape(BSZ, n_chunks, chunk_size)
    msg_times_chunked = np.transpose(msg_times_chunked, (1, 0, 2))

    # ========== Chunk books ==========
    if len(books.shape) == 3:
        book_dim = books.shape[2]
        books_chunked = books.reshape(BSZ, n_chunks, chunk_size, book_dim)
        books_chunked = np.transpose(books_chunked, (1, 0, 2, 3))
    else:
        books_chunked = books.reshape(BSZ, n_chunks, chunk_size)
        books_chunked = np.transpose(books_chunked, (1, 0, 2))

    # ========== Chunk book times ==========
    if len(book_times.shape) == 3:
        book_time_dim = book_times.shape[2]
        book_times_chunked = book_times.reshape(BSZ, n_chunks, chunk_size, book_time_dim)
        book_times_chunked = np.transpose(book_times_chunked, (1, 0, 2, 3))
    else:
        book_times_chunked = book_times.reshape(BSZ, n_chunks, chunk_size)
        book_times_chunked = np.transpose(book_times_chunked, (1, 0, 2))

    return (
        (messages_chunked, books_chunked),
        labels_chunked,
        (msg_times_chunked, book_times_chunked),
        chunk_size,
    )

def split_sequence_for_tbptt_windows(
    batch_inputs: Tuple[jax.Array, jax.Array],
    batch_labels: jax.Array,
    batch_integration_timesteps: Tuple[jax.Array, jax.Array],
    window_size_msgs: int,
    msg_len: int = 24,
) -> Tuple:
    """Split sequence into fixed-size message windows for True TBPTT.

    Args:
        window_size_msgs: number of messages per window (e.g., 500)
    """
    messages, _ = batch_inputs
    L = messages.shape[1]
    n_messages = L // msg_len
    if n_messages % window_size_msgs != 0:
        raise ValueError(
            f"window_size_msgs={window_size_msgs} must divide n_messages={n_messages}"
        )
    n_chunks = n_messages // window_size_msgs
    return split_sequence_for_tbptt(
        batch_inputs,
        batch_labels,
        batch_integration_timesteps,
        n_chunks=n_chunks,
        msg_len=msg_len,
    )

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
# New train_step (jit + shardings version) with TBPTT support
# ============================================================================
def train_step(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 1
        batch_inputs: Tuple[jax.Array, jax.Array], # 2
        batch_labels: jax.Array, # 3
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 4
        batchnorm: bool, # 5
        ignore_times: bool, # 6
        use_tbptt: bool = False, # 7 - NEW: Enable TBPTT gradient chunking
        n_tbptt_chunks: int = 4, # 8 - NEW: Number of chunks
    ):
    """
    Training step function (jit + shardings version) with optional TBPTT.

    TBPTT Mode (use_tbptt=True):
    - Splits sequence into n_tbptt_chunks chunks
    - Each chunk computes forward pass and gradients independently
    - Gradients are accumulated and averaged
    - Reduces XLA compilation memory by O(1/n_chunks²)

    Memory savings:
    - 4 chunks: ~6.25% of original memory (~94% reduction)
    - 8 chunks: ~1.56% of original memory (~98% reduction)

    Main changes from pmap version:
    1. Removed pmap decorator, using jax.jit + in_shardings/out_shardings
    2. Removed jax.lax.pmean, automatic cross-device aggregation
    3. state no longer has device dimension (replicated via sharding)
    """

    batch_inputs = repeat_book(*batch_inputs, True)

    if use_tbptt:
        # ============== TBPTT Gradient Chunking Mode ==============
        # Split data into chunks
        (chunked_inputs, chunked_labels, chunked_times, chunk_size) = \
            split_sequence_for_tbptt(
                batch_inputs,
                batch_labels,
                batch_integration_timesteps,
                n_chunks=n_tbptt_chunks,
                msg_len=Message_Tokenizer.MSG_LEN,
            )

        messages_chunked, books_chunked = chunked_inputs
        msg_times_chunked, book_times_chunked = chunked_times

        # Define function to compute gradient for a single chunk
        def compute_chunk_gradient(carry, chunk_idx):
            """Compute gradient for a single chunk."""
            grads_accum, loss_accum, ce_accum, rng_carry = carry

            # Extract chunk data using dynamic indexing
            msgs_chunk = messages_chunked[chunk_idx]
            books_chunk = books_chunked[chunk_idx]
            labels_chunk = chunked_labels[chunk_idx]
            msg_t_chunk = msg_times_chunked[chunk_idx]
            book_t_chunk = book_times_chunked[chunk_idx]

            rng_carry, rng_chunk = jax.random.split(rng_carry)

            def loss_fn_chunk(params):
                """Loss function for a single chunk."""
                if batchnorm:
                    logits_chunk, mod_vars_chunk = state.apply_fn(
                        {"params": params, "batch_stats": state.batch_stats},
                        msgs_chunk, books_chunk, msg_t_chunk, book_t_chunk,
                        rngs={"dropout": rng_chunk},
                        mutable=["intermediates", "batch_stats"],
                        method='__call_ar__'
                    )
                else:
                    logits_chunk, mod_vars_chunk = state.apply_fn(
                        {"params": params},
                        msgs_chunk, books_chunk, msg_t_chunk, book_t_chunk,
                        rngs={"dropout": rng_chunk},
                        mutable=["intermediates"],
                        method='__call_ar__'
                    )

                ce_chunk = cross_entropy_loss(logits_chunk, labels_chunk)

                if ignore_times:
                    ce_chunk = ce_chunk.reshape(ce_chunk.shape[0], -1, Message_Tokenizer.MSG_LEN)
                    ce_1 = ce_chunk[:, :, :TIME_START_I]
                    ce_2 = ce_chunk[:, :, (TIME_END_I+1):]
                    ce_chunk = np.concatenate([ce_1, ce_2], axis=2)
                    ce_chunk = ce_chunk.reshape(ce_chunk.shape[0], -1)

                ce_chunk_mean = np.mean(ce_chunk, axis=0)
                loss_chunk = np.mean(ce_chunk_mean)

                return loss_chunk, (mod_vars_chunk, logits_chunk, ce_chunk_mean)

            (loss_chunk, (mod_vars_chunk, logits_chunk, ce_chunk)), grads_chunk = \
                jax.value_and_grad(loss_fn_chunk, has_aux=True)(state.params)

            # Accumulate gradients
            grads_accum = jax.tree_util.tree_map(
                lambda a, g: a + g, grads_accum, grads_chunk
            )
            loss_accum = loss_accum + loss_chunk
            ce_accum = ce_accum + ce_chunk

            new_carry = (grads_accum, loss_accum, ce_accum, rng_carry)
            return new_carry, logits_chunk

        # Initialize accumulation
        init_grads = jax.tree_util.tree_map(np.zeros_like, state.params)
        init_loss = np.array(0.0)
        # Initialize CE accumulator with correct shape
        if ignore_times:
            ce_size = chunk_size - (TIME_END_I - TIME_START_I + 1) * (chunk_size // Message_Tokenizer.MSG_LEN)
        else:
            ce_size = chunk_size
        init_ce = np.zeros((ce_size,))
        init_carry = (init_grads, init_loss, init_ce, rng)

        # Execute scan over chunks
        (final_grads, final_loss, final_ce, _), logits_chunks = jax.lax.scan(
            compute_chunk_gradient,
            init_carry,
            np.arange(n_tbptt_chunks),
            length=n_tbptt_chunks,
        )

        # Average gradients and loss
        grads = jax.tree_util.tree_map(lambda g: g / n_tbptt_chunks, final_grads)
        loss = final_loss / n_tbptt_chunks
        ce = final_ce / n_tbptt_chunks

        # Reconstruct logits (for compatibility)
        # logits_chunks: (n_chunks, BSZ, chunk_size, vocab_size)
        logits = np.transpose(logits_chunks, (1, 0, 2, 3))
        logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])

        # Apply gradients (TBPTT mode doesn't update batch_stats per chunk)
        state = state.apply_gradients(grads=grads)

    else:
        # ============== Standard (non-TBPTT) Mode ==============
        def loss_fn(params):
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

            ce = cross_entropy_loss(logits, batch_labels)
            if ignore_times:
                ce = ce.reshape(ce.shape[0], -1, Message_Tokenizer.MSG_LEN)
                ce_1 = ce[:, :, :TIME_START_I]
                ce_2 = ce[:, :, (TIME_END_I+1):]
                ce = np.concatenate([ce_1, ce_2], axis=2)
                ce = ce.reshape(ce.shape[0], -1)

            ce = np.mean(ce, axis=0)
            loss = np.mean(ce)
            return loss, (mod_vars, logits, ce)

        (loss, (mod_vars, logits, ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        if batchnorm:
            state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
        else:
            state = state.apply_gradients(grads=grads)

    return state, loss, ce, logits


# ============================================================================
# True TBPTT (AR-style with hidden carry)
# ============================================================================
def train_step_tbptt_ar_hidden(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,
        batch_inputs: Tuple[jax.Array, jax.Array],
        batch_labels: jax.Array,
        batch_integration_timesteps: Tuple[jax.Array, jax.Array],
        hiddens: Tuple,
        batchnorm: bool,
        ignore_times: bool,
        tbptt_window_size: int = 500,
    ):
    """True TBPTT using AR-style forward with hidden carry."""
    batch_inputs = repeat_book(*batch_inputs, True)

    (chunked_inputs, chunked_labels, chunked_times, chunk_size) = split_sequence_for_tbptt_windows(
        batch_inputs,
        batch_labels,
        batch_integration_timesteps,
        window_size_msgs=tbptt_window_size,
        msg_len=Message_Tokenizer.MSG_LEN,
    )

    messages_chunked, books_chunked = chunked_inputs
    msg_times_chunked, book_times_chunked = chunked_times
    n_windows = messages_chunked.shape[0]

    def compute_window(carry, window_idx):
        grads_accum, loss_accum, hiddens_carry, rng_carry = carry

        msgs_chunk = messages_chunked[window_idx]
        books_chunk = books_chunked[window_idx]
        labels_chunk = chunked_labels[window_idx]
        msg_t_chunk = msg_times_chunked[window_idx]
        book_t_chunk = book_times_chunked[window_idx]

        rng_carry, rng_window = jax.random.split(rng_carry)

        def loss_fn(params):
            if batchnorm:
                (new_hiddens, logits_chunk), _ = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    hiddens_carry,
                    msgs_chunk, books_chunk,
                    msg_t_chunk, book_t_chunk,
                    rngs={"dropout": rng_window},
                    mutable=["intermediates", "batch_stats"],
                    method='__call_ar_tbptt__'
                )
            else:
                (new_hiddens, logits_chunk), _ = state.apply_fn(
                    {"params": params},
                    hiddens_carry,
                    msgs_chunk, books_chunk,
                    msg_t_chunk, book_t_chunk,
                    rngs={"dropout": rng_window},
                    mutable=["intermediates"],
                    method='__call_ar_tbptt__'
                )

            ce_chunk = cross_entropy_loss(logits_chunk, labels_chunk)
            if ignore_times:
                ce_chunk = ce_chunk.reshape(ce_chunk.shape[0], -1, Message_Tokenizer.MSG_LEN)
                ce_1 = ce_chunk[:, :, :TIME_START_I]
                ce_2 = ce_chunk[:, :, (TIME_END_I+1):]
                ce_chunk = np.concatenate([ce_1, ce_2], axis=2)
                ce_chunk = ce_chunk.reshape(ce_chunk.shape[0], -1)

            ce_chunk = np.mean(ce_chunk, axis=0)
            loss_chunk = np.mean(ce_chunk)
            return loss_chunk, (new_hiddens, logits_chunk, ce_chunk)

        (loss_chunk, (new_hiddens, logits_chunk, ce_chunk)), grads_chunk = \
            jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # Truncate gradients between windows but keep hidden carry
        new_hiddens = jax.lax.stop_gradient(new_hiddens)

        grads_accum = jax.tree_util.tree_map(
            lambda a, g: a + g, grads_accum, grads_chunk
        )
        loss_accum = loss_accum + loss_chunk

        new_carry = (grads_accum, loss_accum, new_hiddens, rng_carry)
        return new_carry, (logits_chunk, ce_chunk)

    init_grads = jax.tree_util.tree_map(np.zeros_like, state.params)
    init_loss = np.array(0.0)
    init_carry = (init_grads, init_loss, hiddens, rng)

    (final_grads, final_loss, final_hiddens, _), (logits_chunks, ce_chunks) = jax.lax.scan(
        compute_window,
        init_carry,
        np.arange(n_windows),
        length=n_windows,
    )

    grads = jax.tree_util.tree_map(lambda g: g / n_windows, final_grads)
    loss = final_loss / n_windows

    logits = np.transpose(logits_chunks, (1, 0, 2, 3))
    logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
    ce = ce_chunks.reshape(-1)

    state = state.apply_gradients(grads=grads)
    return state, loss, ce, logits, final_hiddens

# ============================================================================
# Create JIT-compiled train_step
# ============================================================================
def create_jit_train_step(
    mesh: Mesh,
    state: train_state.TrainState,
    has_book_data: bool = True,
    use_tbptt: bool = False,
    n_tbptt_chunks: int = 4,
):
    """
    Create JIT-compiled train_step with optional TBPTT support.

    Why a separate function is needed:
    - jax.jit needs to know input/output shardings
    - We specify in_shardings and out_shardings here
    - donate_argnums tells JAX it can reuse state's memory

    TBPTT Mode:
    - When use_tbptt=True, train_step splits sequences into chunks
    - Each chunk computes gradients independently, then accumulates
    - Reduces XLA compilation memory by O(1/n_chunks²)

    Args:
        mesh: JAX Mesh
        state: Example state (for inferring sharding)
        has_book_data: Whether book data is present
        use_tbptt: Enable TBPTT gradient chunking
        n_tbptt_chunks: Number of chunks for TBPTT

    Returns:
        JIT-compiled train_step function
    """
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
    # IMPORTANT: in_shardings only includes NON-STATIC parameters!
    # Order corresponds to train_step NON-STATIC parameters:
    # (state, rng, batch_inputs, batch_labels, batch_integration_timesteps)
    # batchnorm and ignore_times are static_argnums=(5,6), NOT included here!
    in_shardings = (
        state_shardings,          # param 0: state - replicated
        None,                     # param 1: rng - replicated (None = default)
        inputs_shardings,         # param 2: batch_inputs - sharded
        labels_sharding,          # param 3: batch_labels - sharded
        timesteps_shardings,      # param 4: batch_integration_timesteps - sharded
        # params 5, 6 (batchnorm, ignore_times) are static - NOT in in_shardings!
    )

    # 4. Define out_shardings
    # Order corresponds to return values: (state, loss, ce, logits)
    out_shardings = (
        state_shardings,          # state - replicated
        None,                     # loss - scalar, auto-handled
        None,                     # ce - small array, auto-handled
        None,                     # logits - inferred from inputs
    )

    # 5. Create JIT-compiled function
    # static_argnums: (5, 6, 7, 8) = batchnorm, ignore_times, use_tbptt, n_tbptt_chunks
    jit_train_step = jax.jit(
        train_step,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=(5, 6, 7, 8),  # batchnorm, ignore_times, use_tbptt, n_tbptt_chunks
        donate_argnums=(0,),          # donate state (allows JAX to reuse memory)
    )

    print("[JIT] Created JIT-compiled train_step")
    print(f"[JIT] in_shardings: state=replicated, data=sharded on 'data' axis")
    print(f"[JIT] out_shardings: state=replicated, metrics=auto")
    print(f"[JIT] donate_argnums: (0,) = state (memory optimization)")
    if use_tbptt:
        print(f"[JIT] TBPTT enabled: {n_tbptt_chunks} chunks (memory reduction ~{100 - 100/(n_tbptt_chunks**2):.1f}%)")

    return jit_train_step


# ============================================================================
# Create JIT-compiled train_step for True TBPTT (AR + hidden carry)
# ============================================================================
def create_jit_train_step_tbptt_ar_hidden(
    mesh: Mesh,
    state: train_state.TrainState,
    has_book_data: bool = True,
    tbptt_window_size: int = 500,
):
    """
    Create JIT-compiled train_step for AR-style TBPTT with hidden carry.
    """
    # 1. Create shardings for state (everything replicated)
    state_shardings = create_state_shardings(state, mesh)

    # 2. Create shardings for data
    inputs_shardings, labels_sharding, timesteps_shardings = get_data_shardings_for_batch(
        mesh, has_book_data=has_book_data
    )

    # 3. Define in_shardings (non-static params only)
    in_shardings = (
        state_shardings,          # state
        None,                     # rng
        inputs_shardings,         # batch_inputs
        labels_sharding,          # batch_labels
        timesteps_shardings,      # batch_integration_timesteps
        None,                     # hiddens (pytree)
    )

    # 4. Define out_shardings
    out_shardings = (
        state_shardings,          # state
        None,                     # loss
        None,                     # ce
        None,                     # logits
        None,                     # hiddens_out
    )

    # 5. Create JIT-compiled function
    jit_train_step = jax.jit(
        train_step_tbptt_ar_hidden,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=(6, 7, 8),  # batchnorm, ignore_times, tbptt_window_size
        donate_argnums=(0,),       # donate state
    )

    print("[JIT] Created JIT-compiled train_step (TBPTT AR hidden)")
    print(f"[JIT] TBPTT window size (msgs): {tbptt_window_size}")
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
