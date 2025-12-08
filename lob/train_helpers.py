from functools import partial
import numpy as onp
import jax
import jax.numpy as np
# from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
from flax import jax_utils
import optax
from typing import Any, Dict, Optional, Tuple, Union
from lob.encoding import Message_Tokenizer
import sys
import json
import os

# from lob.lob_seq_model import LobPredModel


TIME_START_I=9
TIME_END_I =13

# num_devices_global = 2
# global_devices = jax.local_devices()[0: num_devices_global]


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

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = \
        jax_utils.replicate(np.array(lr_val, dtype=np.float32))
        
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = \
        jax_utils.replicate(np.array(ssm_lr_val, dtype=np.float32))

    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = \
            jax_utils.replicate(np.array(ssm_lr_val, dtype=np.float32))

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
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       dt_global=False,
                       num_devices=1,
                       ):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param dt_global:
    :return:
    """

    # batch size is given for data across all devices
    # i.e. batch is split between GPUs but dummy data is per GPU
    assert bsz % num_devices == 0
    bsz = bsz // num_devices

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*bsz, seq_len, in_dim)), np.ones(2*bsz))
            integration_timesteps = np.ones((2*bsz, seq_len,))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones((bsz, seq_len,))
    else:
        if use_book_data:
            dummy_input = (
                # np.ones((bsz, seq_len, in_dim), dtype=np.int32),  # messages
                np.ones((bsz, seq_len, ), dtype=np.int32),  # messages
                np.ones((bsz, seq_len, book_dim)),  # books
            )
            integration_timesteps = (
                np.ones((bsz, seq_len, )),
                np.ones((bsz, seq_len, )),
            )
        else:
            # dummy_input = (np.ones((bsz, seq_len, in_dim), dtype=np.int32) , )
            dummy_input = (np.ones((bsz, seq_len, ), dtype=np.int32) , )
            integration_timesteps = (np.ones((bsz, seq_len, )), )

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)

    # jax.debug.print("Dummy input shapes (msg,book) ({}, \n {})",dummy_input[0].shape,dummy_input[1].shape)
    print(f"[DEBUG] Dummy input shapes (msg,book): {dummy_input[0].shape}, {dummy_input[1].shape}")
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
        """
        print("configuring standard optimization setup")

        # # [DEBUG BF16] Test: Use single AdamW instead of multi_transform
        # # multi_transform + inject_hyperparams + mixed dtype (FP32+BF16) causes NaN in tx.update()
        # # Temporary workaround: use single optimizer for all params
        # use_single_optimizer = os.environ.get('USE_SINGLE_OPTIMIZER', '0') == '1'  # DEBUG BF16

        # if use_single_optimizer:  # DEBUG BF16
        #     print("[DEBUG] Using single AdamW optimizer (no multi_transform)")  # DEBUG BF16
        #     tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)  # DEBUG BF16
        # else:  # DEBUG BF16
        #     if dt_global:
        #         ssm_fn = map_nested_fn(
        #             lambda k, _: "ssm"
        #             if k in ["B", "Lambda_re", "Lambda_im", "norm"]
        #             else ("none" if k in [] else "regular")
        #         )

        #     else:
        #         ssm_fn = map_nested_fn(
        #             lambda k, _: "ssm"
        #             if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
        #             else ("none" if k in [] else "regular")
        #         )
        #     tx = optax.multi_transform(
        #         {
        #             "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
        #             "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
        #             "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
        #                                                              weight_decay=weight_decay),
        #         },
        #         ssm_fn,
        #     )
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
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
        
        
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
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
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply 
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with lr")
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
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
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
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    #print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    # ============================================================================
    # Full BF16 Training (Reference: OrderbookDiT implementation)
    # ============================================================================
    # Convert params to BF16 for storage (saves ~50% memory)
    # Optimizer states (Adam m, v) automatically remain FP32 in optax
    # Weight updates computed in FP32, then cast back to BF16
    # ============================================================================

    # # [DEBUG BF16] Check params BEFORE BF16 conversion
    # def check_nan(x):  # DEBUG BF16
    #     return np.any(np.isnan(x))  # DEBUG BF16
    # has_nan_before = jax.tree_util.tree_reduce(  # DEBUG BF16
    #     lambda a, b: a or b,  # DEBUG BF16
    #     jax.tree_util.tree_map(check_nan, params),  # DEBUG BF16
    #     False  # DEBUG BF16
    # )  # DEBUG BF16
    # if has_nan_before:  # DEBUG BF16
    #     print(f"[ERROR] Params have NaN BEFORE BF16 conversion!")  # DEBUG BF16
    #     # Find which params have NaN  # DEBUG BF16
    #     def find_nan_params(path, x):  # DEBUG BF16
    #         if np.any(np.isnan(x)):  # DEBUG BF16
    #             path_str = '/'.join(str(k.key) for k in path)  # DEBUG BF16
    #             print(f"  NaN in {path_str}: shape={x.shape}, dtype={x.dtype}, min={np.nanmin(x)}, max={np.nanmax(x)}")  # DEBUG BF16
    #     jax.tree_util.tree_map_with_path(lambda p, x: find_nan_params(p, x), params)  # DEBUG BF16
    # else:  # DEBUG BF16
    #     print(f"[*] ✓ No NaN in params before BF16 conversion")  # DEBUG BF16
    #     # [DEBUG BF16] Print initial param range
    #     params_min_init = jax.tree_util.tree_reduce(lambda a, b: np.minimum(a, b), jax.tree_util.tree_map(lambda p: np.min(p), params), np.inf)  # DEBUG BF16
    #     params_max_init = jax.tree_util.tree_reduce(lambda a, b: np.maximum(a, b), jax.tree_util.tree_map(lambda p: np.max(p), params), -np.inf)  # DEBUG BF16
    #     print(f"[*] Initial params range (FP32): [{params_min_init}, {params_max_init}]")  # DEBUG BF16

    #     # [DEBUG BF16] Find which param has extreme values
    #     def find_extreme_params(path, x):  # DEBUG BF16
    #         min_val = float(np.min(x))  # DEBUG BF16
    #         max_val = float(np.max(x))  # DEBUG BF16
    #         if abs(min_val) > 100 or abs(max_val) > 100:  # DEBUG BF16
    #             path_str = '/'.join(str(k.key) for k in path)  # DEBUG BF16
    #             print(f"  [EXTREME] {path_str}: range=[{min_val:.2f}, {max_val:.2f}], shape={x.shape}, dtype={x.dtype}")  # DEBUG BF16
    #     jax.tree_util.tree_map_with_path(lambda p, x: find_extreme_params(p, x), params)  # DEBUG BF16

    use_bf16 = os.environ.get('USE_BF16', '1') == '1'
    if use_bf16:
        def to_bf16_selective(path, x):
            """Convert float32 to bfloat16, but keep Lambda, D, log_step in FP32 for precision.

            Kept in FP32:
              - Lambda (eigenvalues): Lambda_im from HiPPO can be very large (e.g., -1303 to -83000)
                BF16 precision insufficient for large values + gradient updates
                Lambda used in discretization (exp(Lambda*Δ)) needs precision
              - D (feedthrough): Stored as FP32, cast to FP32 for computation
                Same treatment as B/C matrices (BF16 matmul → FP32 output)
              - log_step (timescale): Used in discretization (exp(log_step)) needs FP32 precision
                Step values are critical for SSM stability
            """
            path_str = '/'.join(str(k.key) for k in path)
            # Keep Lambda_re, Lambda_im, D, and log_step in FP32
            if 'Lambda_re' in path_str or 'Lambda_im' in path_str:
                return x  # Keep FP32 for Lambda
            if path_str.endswith('/D'):
                return x  # Keep FP32 for D vector
            if 'log_step' in path_str:
                return x  # Keep FP32 for log_step
            # Convert other float32 params to BF16
            if x.dtype == np.float32:
                return x.astype(np.bfloat16)
            return x
        params = jax.tree_util.tree_map_with_path(to_bf16_selective, params)
        print(f"[*] Selective BF16 Training: params=BF16, Lambda/D/log_step=FP32, optimizer_states=FP32")

        # # [DEBUG BF16] Check params AFTER BF16 conversion
        # has_nan_after = jax.tree_util.tree_reduce(  # DEBUG BF16
        #     lambda a, b: a or b,  # DEBUG BF16
        #     jax.tree_util.tree_map(check_nan, params),  # DEBUG BF16
        #     False  # DEBUG BF16
        # )  # DEBUG BF16
        # if has_nan_after:  # DEBUG BF16
        #     print(f"[ERROR] Params have NaN AFTER BF16 conversion!")  # DEBUG BF16
        #     def find_nan_params(path, x):  # DEBUG BF16
        #         if np.any(np.isnan(x)):  # DEBUG BF16
        #             path_str = '/'.join(str(k.key) for k in path)  # DEBUG BF16
        #             print(f"  NaN in {path_str}: shape={x.shape}, dtype={x.dtype}")  # DEBUG BF16
        #     jax.tree_util.tree_map_with_path(lambda p, x: find_nan_params(p, x), params)  # DEBUG BF16
        # else:  # DEBUG BF16
        #     print(f"[*] ✓ No NaN in params after BF16 conversion")  # DEBUG BF16
        # # [DEBUG BF16] Print BF16 param range
        # params_min_bf16 = jax.tree_util.tree_reduce(lambda a, b: np.minimum(a, b), jax.tree_util.tree_map(lambda p: np.min(p), params), np.inf)  # DEBUG BF16
        # params_max_bf16 = jax.tree_util.tree_reduce(lambda a, b: np.maximum(a, b), jax.tree_util.tree_map(lambda p: np.max(p), params), -np.inf)  # DEBUG BF16
        # print(f"[*] BF16 params range: [{params_min_bf16}, {params_max_bf16}]")  # DEBUG BF16

        # Print dtype distribution for verification
        dtype_counts = {}
        def count_dtype(x):
            dtype_str = str(x.dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            return x
        jax.tree_util.tree_map(count_dtype, params)
        print(f"[*] Param dtypes: {dtype_counts}")
    else:
        print(f"[*] Full FP32 training (BF16 disabled via USE_BF16=0)")

    # ========================================================================
    # Workaround: Initialize optimizer with FP32 params to avoid dtype mismatch
    # ========================================================================
    # Problem: If optimizer is initialized with BF16 params, its internal state may
    # expect BF16 even after we cast to FP32 during update
    # Solution: Initialize with FP32, then convert params to BF16 after TrainState.create()
    # ========================================================================
    if use_bf16:  # Workaround
        # Temporarily convert params back to FP32 for TrainState initialization
        def to_fp32_for_init(x):  # Workaround
            if x.dtype == np.bfloat16:  # Workaround
                return x.astype(np.float32)  # Workaround
            return x  # Workaround
        params_for_init = jax.tree_util.tree_map(to_fp32_for_init, params)  # Workaround
        print(f"[*] Initializing optimizer with FP32 params (will convert back to BF16)")  # DEBUG BF16
    else:  # Workaround
        params_for_init = params  # Workaround

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        state = TrainState.create(apply_fn=model.apply, params=params_for_init, tx=tx, batch_stats=batch_stats)
    else:
        state = train_state.TrainState.create(apply_fn=model.apply, params=params_for_init, tx=tx)

    # Convert params back to BF16 after optimizer initialization
    if use_bf16:  # Workaround
        state = state.replace(params=params)  # Use original BF16 params
        print(f"[*] Optimizer initialized with FP32, params converted back to selective BF16")  # DEBUG BF16

    # keep copy of state on each device
    print(state.params['message_encoder']['encoder']['embedding'].shape)

    # # [DEBUG BF16] Check state before replicate
    # print(f"[Before replicate] Lambda_im dtype: {state.params['message_encoder']['layers_1']['seq']['Lambda_im'].dtype}")  # DEBUG BF16
    # print(f"[Before replicate] B dtype: {state.params['message_encoder']['layers_1']['seq']['B'].dtype}")  # DEBUG BF16

    state = jax_utils.replicate(state)#, devices=global_devices)
    print(state.params['message_encoder']['encoder']['embedding'].shape)

    # [DEBUG BF16] Check state after replicate
    # print(f"[After replicate] Lambda_im dtype: {state.params['message_encoder']['layers_1']['seq']['Lambda_im'].dtype}")  # DEBUG BF16
    # print(f"[After replicate] Lambda_im has NaN: {jax.numpy.any(jax.numpy.isnan(state.params['message_encoder']['layers_1']['seq']['Lambda_im']))}")  # DEBUG BF16
    # print(f"[After replicate] B dtype: {state.params['message_encoder']['layers_1']['seq']['B'].dtype}")  # DEBUG BF16

    return state

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

    # reshape from large batch to multiple device batches
    inputs, targets, book_data, timestep_msg, timestep_book = device_reshape(
        num_devices,
        inputs,
        targets,
        book_data,
        timestep_msg,
        timestep_book,
    )
    # print('inputs shape (device_reshape):', inputs.shape)

    # split large batch into smaller device batches on the GPUs
    inputs, labels, integration_times = _prep_batch_par(
        inputs,
        targets,
        seq_len,
        # in_dim,
        book_data,
        timestep_msg,
        timestep_book,
    )
    # print('inputs (targets) shape (_prep_batch_par):', inputs[1].shape)

    return inputs, labels, integration_times

@partial(
#    jax.vmap,
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(2,),
    # in_axes=(0, 0, None, None, 0, 0, 0),
    in_axes=(0, 0, None, 0, 0, 0),
    # out_axes=(0, 0, 0),
    # devices=global_devices
)
def _prep_batch_par(
        inputs: jax.Array,
        targets: jax.Array,
        seq_len: int,
        # in_dim: int,
        book_data: Optional[jax.Array] = None,
        timestep_msg: Optional[jax.Array] = None,
        timestep_book: Optional[jax.Array] = None,
    ) -> Tuple[Tuple, np.ndarray, Tuple]:
    """
    Take a batch and convert it to a standard x/y format per device
    TODO: document this better for pmapped version
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """

    assert inputs.shape[1] == seq_len, f'inputs: {inputs.shape} seq_len {seq_len}'
    # inputs = one_hot(inputs, in_dim)

    # If there is an aux channel containing the integration times, then add that.
    if timestep_msg is not None:
        #timestep_msg = jax.device_put(timestep_msg, jax.devices()[0])
        integration_timesteps = (np.diff(np.asarray(timestep_msg)), )
    else:
        integration_timesteps = (np.ones((len(inputs), seq_len)), )

    if book_data is not None:
        #book_data = jax.device_put(book_data, jax.devices()[0])
        full_inputs = (inputs.astype(np.int32), book_data)
        if timestep_book is not None:
            #timestep_book = jax.device_put(timestep_book, jax.devices()[0])
            integration_timesteps += (np.diff(timestep_book), )
        else:
            integration_timesteps += (np.ones((len(inputs), seq_len)), )
    else:
        full_inputs = (inputs.astype(np.int32), )

    # CAVE: squeeze very important for training!
    return full_inputs, np.squeeze(targets.astype(np.int32)), integration_timesteps

@partial(jax.jit, static_argnums=(0,), backend='gpu')# backend='cpu')
def device_reshape(
        num_devices: int,
        inputs: jax.Array,
        targets: jax.Array,
        book_data: Optional[jax.Array] = None,
        timestep_msg: Optional[jax.Array] = None,
        timestep_book: Optional[jax.Array] = None,
    ) -> Tuple:
    """ 
    """
    inputs = np.reshape(inputs, (num_devices, -1, *inputs.shape[1:]))
    targets = np.reshape(targets, (num_devices, -1, *targets.shape[1:]))
    if book_data is not None:
        book_data = np.reshape(book_data, (num_devices, -1, *book_data.shape[1:]))
    if timestep_msg is not None:
        timestep_msg = np.reshape(timestep_msg, (num_devices, -1, *timestep_msg.shape[1:]))
    if timestep_book is not None:
        timestep_book = np.reshape(timestep_book, (num_devices, -1, *timestep_book.shape[1:]))
    return inputs, targets, book_data, timestep_msg, timestep_book


def train_epoch(
        state,
        rng,
        #model,
        trainloader,
        seq_len,
        # in_dim,
        batchnorm,
        lr_params,
        num_devices,
        debug_loading,
        debug_profiler,
        curtail_epochs,
        init_hiddens,
        epoch,
        ignore_times,
        log_ce_tables,
        use_wandb=False,
        process_index=0,
        max_batches=None,  # New: limit number of batches to train (for intra-epoch evaluation)
    ):

    """
    Training function for an epoch that loops over batches.

    Args:
        max_batches: If provided, stop training after processing this many batches.
                     Used for intra-epoch evaluation to train only a segment of the epoch.
    """
    # Store Metrics
    batch_losses = []
    cross_entropies= [] #list of 1xNTok losses

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params
    batches_processed = 0  # Track how many batches we've processed in this call

    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for batch_idx, batch in enumerate(tqdm(trainloader)):
        # print(f"train_epoch: Epoch {epoch} - Batch {batch_idx} / {len(trainloader)}")
        # print(f"train_epoch: Batch input shape: {batch[0].shape}, batch target shape: {batch[1].shape}")
        if not debug_loading:
            if (step>1) & (step<3) & debug_profiler:
                jax.profiler.start_trace("/tmp/tensorboard")
            inputs, labels, integration_times = prep_batch(batch, seq_len, num_devices)
            # print("train_epoch: Prepared batch inputs shape:", inputs[0].shape)
            # print("train_epoch: Prepared batch labels shape:", labels.shape)
            # print("train_epoch: Inputs 0:5:", inputs[0][0,0:5,:])
            rng, drop_rng = jax.random.split(rng)

            
            # state,loss=train_step_rnn(                
            #     state,
            #     drop_rng,
            #     inputs,
            #     labels,
            #     integration_times,
            #     batchnorm,
            #     init_hiddens)

            # print("Gets to train")
            state, loss, ce, logits = train_step(
                state,
                drop_rng,
                inputs,
                labels,
                integration_times,
                batchnorm,
                ignore_times,
            )
            if debug_profiler:
                loss.block_until_ready()
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

            # losses are already averaged across devices (--> should be all the same here)
            batch_losses.append(loss[0])
            if log_ce_tables:
                cross_entropies.append(ce)

            # DISABLED: Per-step wandb logging causes ~10% slowdown even at 1000-step intervals
            # Only using per-epoch logging in train.py instead
            # if use_wandb and process_index == 0 and step % 1000 == 0:
            #     import wandb
            #     current_lr = decay_function(step, lr, end_step, lr_min)
            #     wandb.log({
            #         "train/loss_step": float(loss[0]),
            #         "train/step": step,
            #         "train/epoch": epoch,
            #         "train/lr": float(current_lr),
            #     })

            lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
            state, step = update_learning_rate_per_step(lr_params, state)

            # Increment batch counter
            batches_processed += 1

            if (step>20) & (step<=21) & debug_profiler:
                jax.profiler.stop_trace()
                break

            # Check max_batches limit (for intra-epoch evaluation)
            if (max_batches is not None) and (batches_processed >= max_batches):
                print(f"[train_epoch] Reached max_batches={max_batches}, stopping segment")
                break

            # Original curtail_epochs check
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
    return state,loss_mean , ce_means,step


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

@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(5,6),  # TODO: revert to 5 for batchnorm in pmap
    in_axes=(0, None, 0, 0, 0, None, None),
    # out_axes=(0, 0),
    # devices=global_devices
)
def train_step(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 1
        batch_inputs: Tuple[jax.Array, jax.Array], # 2
        batch_labels: jax.Array, # 3
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 4
        batchnorm: bool, # 5
        ignore_times:bool, #6
    ):

    # Print hash values of static arguments
    # print(f"batchnorm hash: {batchnorm.__hash__()}")
    # print(f"ignore_times hash: {ignore_times.__hash__()}")
    # print('checking for compile in train_step')

    batch_inputs=repeat_book(*batch_inputs,True)
    # batch_integration_timesteps=repeat_book(*batch_integration_timesteps)

    def loss_fn(params):
        # print('checking for compile in loss_fn')
        # BF16 Mixed Precision: params are already BF16 from initialization
        # No need for tree_map here - direct use saves 30-40% overhead

        # # [DEBUG BF16] Check params dtype in loss_fn
        # def check_params_dtype_callback(params_tree):  # DEBUG BF16
        #     lambda_im = params_tree['message_encoder']['layers_1']['seq']['Lambda_im']  # DEBUG BF16
        #     b_param = params_tree['message_encoder']['layers_1']['seq']['B']  # DEBUG BF16
        #     print(f"[loss_fn] Lambda_im dtype: {lambda_im.dtype}, B dtype: {b_param.dtype}")  # DEBUG BF16
        # jax.debug.callback(check_params_dtype_callback, params)  # DEBUG BF16

        # # ===== NaN Detection Point 1: Input Parameters =====
        # params_has_nan = jax.tree_util.tree_reduce(  # mixed precision overflow debug
        #     lambda a, b: a | b,  # mixed precision overflow debug
        #     jax.tree_util.tree_map(lambda x: np.any(np.isnan(x)), params),  # mixed precision overflow debug
        #     False  # mixed precision overflow debug
        # )  # mixed precision overflow debug
        # jax.debug.print("[NaN Check 1] Params has NaN: {}", params_has_nan)  # mixed precision overflow debug

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

        # # ===== NaN Detection Point 2: Forward Output =====
        # logits_has_nan = np.any(np.isnan(logits))  # mixed precision overflow debug
        # jax.debug.print("[NaN Check 2] Logits has NaN: {}, dtype: {}, range: [{}, {}]", logits_has_nan, logits.dtype, np.min(logits), np.max(logits))  # DEBUG BF16

        # BF16 Mixed Precision: Cast logits back to FP32 for loss computation
        logits = logits.astype(np.float32)

        # jax.debug.print("Shape of Logits: {}",logits.shape)
        # jax.debug.print("Shape of Labels: {}", batch_labels.shape)

        # Ensure labels are int32
        batch_labels_int = batch_labels.astype(np.int32)
        ce=cross_entropy_loss(logits, batch_labels_int)
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

        # ===== NaN Detection Point 3: Loss =====
        # loss_has_nan = np.isnan(loss)  # mixed precision overflow debug
        # jax.debug.print("[NaN Check 3] Loss has NaN: {}, value: {:.6f}", loss_has_nan, loss)  # mixed precision overflow debug

        return loss, (mod_vars, logits,ce)

    (loss, (mod_vars, logits,ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # # [DEBUG BF16] Check grads dtype after backward
    # def check_grads_dtype_callback(grads_tree):  # DEBUG BF16
    #     lambda_im_grad = grads_tree['message_encoder']['layers_1']['seq']['Lambda_im']  # DEBUG BF16
    #     b_grad = grads_tree['message_encoder']['layers_1']['seq']['B']  # DEBUG BF16
    #     print(f"[After backward] Lambda_im_grad dtype: {lambda_im_grad.dtype}, B_grad dtype: {b_grad.dtype}")  # DEBUG BF16
    # jax.debug.callback(check_grads_dtype_callback, grads)  # DEBUG BF16

    # # ===== NaN Detection Point 4: Gradients =====
    # grads_has_nan = jax.tree_util.tree_reduce(  # mixed precision overflow debug
    #     lambda a, b: a | b,  # mixed precision overflow debug
    #     jax.tree_util.tree_map(lambda x: np.any(np.isnan(x)), grads),  # mixed precision overflow debug
    #     False  # mixed precision overflow debug
    # )  # mixed precision overflow debug
    # jax.debug.print("[NaN Check 4] Grads has NaN: {}", grads_has_nan)  # mixed precision overflow debug

    # # ===== NaN Detection Point 5: Gradient Norm =====
    # grad_norm = np.sqrt(jax.tree_util.tree_reduce(  # mixed precision overflow debug
    #     lambda a, b: a + b,  # mixed precision overflow debug
    #     jax.tree_util.tree_map(lambda x: np.sum(x.astype(np.float32) ** 2), grads),  # mixed precision overflow debug
    #     0.0  # mixed precision overflow debug
    # ))  # mixed precision overflow debug
    # jax.debug.print("[NaN Check 5] Grad norm: {:.6f}", grad_norm)  # mixed precision overflow debug

    # # [DEBUG BF16] Record all gradient norms to JSON file for detailed analysis
    # def record_all_grads_callback(step_val, global_norm_val, grads_pytree):  # DEBUG BF16
    #     """Record all gradient norms and identify large gradients"""  # DEBUG BF16
    #     import json  # DEBUG BF16

    #     # Compute per-parameter gradient norms
    #     grad_norms = {}  # DEBUG BF16
    #     large_grads = []  # DEBUG BF16

    #     def compute_grad_info(path, g):  # DEBUG BF16
    #         path_str = '/'.join(str(k.key) for k in path)  # DEBUG BF16
    #         g_norm = float(np.sqrt(np.sum(g.astype(np.float32) ** 2)))  # DEBUG BF16
    #         g_min = float(np.min(g))  # DEBUG BF16
    #         g_max = float(np.max(g))  # DEBUG BF16
    #         g_mean = float(np.mean(g.astype(np.float32)))  # DEBUG BF16

    #         grad_norms[path_str] = {  # DEBUG BF16
    #             'norm': g_norm,  # DEBUG BF16
    #             'min': g_min,  # DEBUG BF16
    #             'max': g_max,  # DEBUG BF16
    #             'mean': g_mean,  # DEBUG BF16
    #             'shape': str(g.shape),  # DEBUG BF16
    #             'dtype': str(g.dtype)  # DEBUG BF16
    #         }  # DEBUG BF16

    #         # Flag if large (>1e6)
    #         if g_norm > 1e6:  # DEBUG BF16
    #             large_grads.append(path_str)  # DEBUG BF16
    #             print(f"[LARGE GRAD] {path_str}: norm={g_norm:.2e}, range=[{g_min:.2e}, {g_max:.2e}]")  # DEBUG BF16

    #     jax.tree_util.tree_map_with_path(lambda p, g: compute_grad_info(p, g), grads_pytree)  # DEBUG BF16

    #     # Save to JSON
    #     record = {  # DEBUG BF16
    #         'step': int(step_val),  # DEBUG BF16
    #         'global_norm': float(global_norm_val),  # DEBUG BF16
    #         'num_large_grads': len(large_grads),  # DEBUG BF16
    #         'large_grad_params': large_grads,  # DEBUG BF16
    #         'all_grad_norms': grad_norms  # DEBUG BF16
    #     }  # DEBUG BF16

    #     with open('grad_analysis_debug.jsonl', 'a') as f:  # DEBUG BF16
    #         f.write(json.dumps(record) + '\n')  # DEBUG BF16

    # jax.debug.callback(record_all_grads_callback, state.step, grad_norm, grads)  # DEBUG BF16

    # ===== Layered Gradient Statistics (Write to JSON File) =====
    # # Calculate gradient norm for each leaf node
    # def compute_leaf_norm(grad):  # mixed precision overflow debug
    #     return np.sqrt(np.sum(grad.astype(np.float32) ** 2))  # mixed precision overflow debug

    # leaf_norms = jax.tree_util.tree_map(compute_leaf_norm, grads)  # mixed precision overflow debug
    # leaf_norms_with_path = jax.tree_util.tree_leaves_with_path(leaf_norms)  # mixed precision overflow debug

    # # Build record and write to file
    # def write_grad_stats(step_val, global_norm_val, leaf_norms_with_path_val):  # mixed precision overflow debug
    #     """Write gradient statistics to JSON file on host"""  # mixed precision overflow debug
    #     # Get precision mode from environment variable
    #     precision = os.environ.get('GRAD_STATS_PRECISION', 'bf16')  # mixed precision overflow debug

    #     # Convert to dict: path -> norm
    #     layer_norms_dict = {}  # mixed precision overflow debug
    #     for path, norm in leaf_norms_with_path_val:  # mixed precision overflow debug
    #         path_str = '/'.join(str(k.key) for k in path)  # mixed precision overflow debug
    #         layer_norms_dict[path_str] = float(norm)  # mixed precision overflow debug

    #     record = {  # mixed precision overflow debug
    #         'step': int(step_val),  # mixed precision overflow debug
    #         'precision': precision,  # mixed precision overflow debug
    #         'global_norm': float(global_norm_val),  # mixed precision overflow debug
    #         'layer_norms': layer_norms_dict,  # mixed precision overflow debug
    #     }  # mixed precision overflow debug

    #     filepath = f"grad_stats_{precision}.jsonl"  # mixed precision overflow debug
    #     with open(filepath, 'a') as f:  # mixed precision overflow debug
    #         f.write(json.dumps(record) + '\n')  # mixed precision overflow debug

    # # Use callback to execute file write on host
    # jax.debug.callback(write_grad_stats, state.step, grad_norm, leaf_norms_with_path)  # mixed precision overflow debug

    # # ===== Gradient Clipping - Must Enable to Prevent BF16 NaN =====
    # # BF16 training can enable: even if gradient norm is normal, BF16 updates may still produce NaN
    # # Reason: BF16 precision insufficient to handle small updates to large parameter values (e.g., Lambda_im=-1303)
    # MAX_GRAD_NORM = 1.0  # Standard value
    # clip_factor = np.minimum(1.0, MAX_GRAD_NORM / (grad_norm + 1e-6))  # Enable gradient clipping
    # grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)  # Enable gradient clipping
    # jax.debug.print("[Grad Clip] grad_norm: {}, clip_factor: {}, clipped_norm: {}", grad_norm, clip_factor, grad_norm * clip_factor)  # DEBUG BF16

    # UPDATE
    # calculate means over device dimension (first)
    loss = jax.lax.pmean(loss, axis_name="batch_devices")
    grads = jax.lax.pmean(grads, axis_name="batch_devices")
    ce=jax.lax.pmean(ce,axis_name="batch_devices")

    # # [DEBUG BF16] Check grads before apply_gradients
    # grads_min = jax.tree_util.tree_reduce(lambda a, b: np.minimum(a, b), jax.tree_util.tree_map(lambda g: np.min(g), grads), np.inf)  # DEBUG BF16
    # grads_max = jax.tree_util.tree_reduce(lambda a, b: np.maximum(a, b), jax.tree_util.tree_map(lambda g: np.max(g), grads), -np.inf)  # DEBUG BF16
    # jax.debug.print("[Pre-Update] Grads range: [{}, {}]", grads_min, grads_max)  # DEBUG BF16

    # # [DEBUG BF16] Check params before apply_gradients
    # params_min = jax.tree_util.tree_reduce(lambda a, b: np.minimum(a, b), jax.tree_util.tree_map(lambda p: np.min(p), state.params), np.inf)  # DEBUG BF16
    # params_max = jax.tree_util.tree_reduce(lambda a, b: np.maximum(a, b), jax.tree_util.tree_map(lambda p: np.max(p), state.params), -np.inf)  # DEBUG BF16
    # jax.debug.print("[Pre-Update] Params range: [{}, {}]", params_min, params_max)  # DEBUG BF16

    # # [DEBUG BF16] Check optimizer state (Adam m, v) before update
    # # Adam states should be FP32 even if params are BF16
    # def check_opt_state_callback(opt_state_tree):  # DEBUG BF16
    #     try:  # DEBUG BF16
    #         # Access Adam state for 'ssm' group (Lambda_im, log_step)
    #         ssm_state = opt_state_tree.inner_states['ssm'].inner_state  # DEBUG BF16
    #         # Check mu (first moment) dtype
    #         if hasattr(ssm_state, 'mu'):  # DEBUG BF16
    #             print(f"[Optimizer] ssm group mu dtype: {type(ssm_state.mu)}")  # DEBUG BF16
    #         elif hasattr(ssm_state, 'count'):  # DEBUG BF16
    #             print(f"[Optimizer] ssm group structure: {type(ssm_state)}")  # DEBUG BF16
    #     except Exception as e:  # DEBUG BF16
    #         print(f"[Optimizer] Cannot access structure: {e}")  # DEBUG BF16
    # jax.debug.callback(check_opt_state_callback, state.opt_state)  # DEBUG BF16

    # # [DEBUG BF16] Sample a few params and grads before update
    # def sample_params_grads(params_tree, grads_tree):  # DEBUG BF16
    #     # Sample message_encoder/layers_1/seq/Lambda_im (FP32 param that becomes NaN)
    #     try:  # DEBUG BF16
    #         lambda_im_param = params_tree['message_encoder']['layers_1']['seq']['Lambda_im']  # DEBUG BF16
    #         lambda_im_grad = grads_tree['message_encoder']['layers_1']['seq']['Lambda_im']  # DEBUG BF16
    #         print(f"[Sample Before Update] Lambda_im: param range=[{np.min(lambda_im_param):.4f}, {np.max(lambda_im_param):.4f}], grad range=[{np.min(lambda_im_grad):.6f}, {np.max(lambda_im_grad):.6f}]")  # DEBUG BF16
    #     except:  # DEBUG BF16
    #         pass  # DEBUG BF16
    # jax.debug.callback(sample_params_grads, state.params, grads)  # DEBUG BF16

    # ========================================================================
    # Workaround for multi_transform NaN bug with mixed dtype
    # ========================================================================
    # Problem: optax.multi_transform produces NaN updates when params tree has mixed dtype (FP32+BF16)
    # Solution: Cast all params/grads to FP32 before tx.update(), then cast back after
    # Reference: User suggestion - "输入optax前都cast成FP32，运算结束后再转回各自精度"
    # ========================================================================

    # Save original dtypes
    def get_dtype(x):  # Workaround for multi_transform NaN
        return x.dtype  # Workaround for multi_transform NaN
    original_param_dtypes = jax.tree_util.tree_map(get_dtype, state.params)  # Workaround for multi_transform NaN

    # Cast params and grads to FP32 for optimizer
    def to_fp32(x):  # Workaround for multi_transform NaN
        if x.dtype == np.bfloat16:  # Workaround for multi_transform NaN
            return x.astype(np.float32)  # Workaround for multi_transform NaN
        return x  # Workaround for multi_transform NaN
    params_fp32 = jax.tree_util.tree_map(to_fp32, state.params)  # Workaround for multi_transform NaN
    grads_fp32 = jax.tree_util.tree_map(to_fp32, grads)  # Workaround for multi_transform NaN

    jax.debug.print("[Optimizer Workaround] Cast params/grads to FP32 before tx.update()")  # DEBUG BF16

    # Run optimizer in FP32
    updates, new_opt_state = state.tx.update(grads_fp32, state.opt_state, params_fp32)  # Workaround for multi_transform NaN

    # # [DEBUG BF16] Check updates after tx.update() - per group
    # def check_updates_detailed(updates_tree):  # DEBUG BF16
    #     # Check ssm group (Lambda_im, should be updated by Adam)
    #     try:  # DEBUG BF16
    #         lambda_im_update = updates_tree['message_encoder']['layers_1']['seq']['Lambda_im']  # DEBUG BF16
    #         print(f"[After tx.update] Lambda_im (ssm group): range=[{np.min(lambda_im_update):.6f}, {np.max(lambda_im_update):.6f}], dtype={lambda_im_update.dtype}, has NaN={np.any(np.isnan(lambda_im_update))}")  # DEBUG BF16
    #     except Exception as e:  # DEBUG BF16
    #         print(f"[After tx.update] Error Lambda_im: {e}")  # DEBUG BF16

    #     # Check ssm group (B, BF16 complex param)
    #     try:  # DEBUG BF16
    #         b_update = updates_tree['message_encoder']['layers_1']['seq']['B']  # DEBUG BF16
    #         b_update_norm = np.sqrt(np.sum(b_update.astype(np.float32) ** 2))  # DEBUG BF16
    #         print(f"[After tx.update] B (ssm group): norm={b_update_norm:.6f}, dtype={b_update.dtype}, has NaN={np.any(np.isnan(b_update))}")  # DEBUG BF16
    #     except Exception as e:  # DEBUG BF16
    #         print(f"[After tx.update] Error B: {e}")  # DEBUG BF16

    #     # Check regular group (Dense kernel, BF16)
    #     try:  # DEBUG BF16
    #         kernel_update = updates_tree['message_encoder']['layers_1']['out2']['kernel']  # DEBUG BF16
    #         kernel_norm = np.sqrt(np.sum(kernel_update.astype(np.float32) ** 2))  # DEBUG BF16
    #         print(f"[After tx.update] out2/kernel (regular group): norm={kernel_norm:.6f}, dtype={kernel_update.dtype}, has NaN={np.any(np.isnan(kernel_update))}")  # DEBUG BF16
    #     except Exception as e:  # DEBUG BF16
    #         print(f"[After tx.update] Error kernel: {e}")  # DEBUG BF16
    # jax.debug.callback(check_updates_detailed, updates)  # DEBUG BF16

    # Step 2: optax.apply_updates() - apply updates to params (in FP32)
    new_params_fp32 = optax.apply_updates(params_fp32, updates)  # Workaround for multi_transform NaN

    # Cast new_params back to original dtypes (BF16 where needed)
    def cast_back_to_original_dtype(new_val, orig_dtype):  # Workaround for multi_transform NaN
        if orig_dtype == np.bfloat16:  # Workaround for multi_transform NaN
            return new_val.astype(np.bfloat16)  # Workaround for multi_transform NaN
        return new_val  # Keep FP32 (Lambda, D, log_step)  # Workaround for multi_transform NaN
    new_params = jax.tree_util.tree_map(cast_back_to_original_dtype, new_params_fp32, original_param_dtypes)  # Workaround for multi_transform NaN

    jax.debug.print("[Optimizer Workaround] Cast new_params back to original dtypes")  # DEBUG BF16

    # # [DEBUG BF16] Check params after apply_updates
    # def check_after_apply_updates(new_params_tree):  # DEBUG BF16
    #     try:  # DEBUG BF16
    #         lambda_im_new = new_params_tree['message_encoder']['layers_1']['seq']['Lambda_im']  # DEBUG BF16
    #         print(f"[After apply_updates] Lambda_im: range=[{np.min(lambda_im_new):.4f}, {np.max(lambda_im_new):.4f}], has NaN={np.any(np.isnan(lambda_im_new))}")  # DEBUG BF16
    #     except:  # DEBUG BF16
    #         pass  # DEBUG BF16
    # jax.debug.callback(check_after_apply_updates, new_params)  # DEBUG BF16

    # Step 3: Replace state
    if batchnorm:
        mod_vars = jax.lax.pmean(mod_vars, axis_name="batch_devices")
        state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)

    # # [DEBUG BF16] Sample after full update
    # def sample_params_after(params_tree):  # DEBUG BF16
    #     try:  # DEBUG BF16
    #         lambda_im_param = params_tree['message_encoder']['layers_1']['seq']['Lambda_im']  # DEBUG BF16
    #         print(f"[Sample After Update] Lambda_im: param range=[{np.min(lambda_im_param):.4f}, {np.max(lambda_im_param):.4f}], has NaN={np.any(np.isnan(lambda_im_param))}")  # DEBUG BF16
    #     except:  # DEBUG BF16
    #         pass  # DEBUG BF16
    # jax.debug.callback(sample_params_after, state.params)  # DEBUG BF16

    # ===== NaN Detection Point 6: Updated Parameters + Auto-stop Training =====
    new_params_has_nan = jax.tree_util.tree_reduce(  # NaN detection
        lambda a, b: a | b,  # NaN detection
        jax.tree_util.tree_map(lambda x: np.any(np.isnan(x)), state.params),  # NaN detection
        False  # NaN detection
    )  # NaN detection
    jax.debug.print("[NaN Check 6] Updated params has NaN: {}", new_params_has_nan)  # NaN detection

    # Auto-stop training on NaN (no JIT overhead, runs on host)
    def halt_on_nan(has_nan_val, step_val, params_pytree):  # NaN auto-stop
        if has_nan_val:  # NaN auto-stop
            print(f"\n{'='*70}")  # NaN auto-stop
            print(f"[FATAL] NaN detected in params after step {step_val}! Stopping training.")  # NaN auto-stop
            print(f"{'='*70}\n")  # NaN auto-stop
            # Find which params have NaN
            def check_nan(path, x):  # NaN auto-stop
                if np.any(np.isnan(x)):  # NaN auto-stop
                    path_str = '/'.join(str(k.key) for k in path)  # NaN auto-stop
                    print(f"  NaN in: {path_str}, shape={x.shape}, dtype={x.dtype}")  # NaN auto-stop
            jax.tree_util.tree_map_with_path(lambda p, x: check_nan(p, x), params_pytree)  # NaN auto-stop
            raise RuntimeError(f"Training stopped: NaN in params at step {step_val}")  # NaN auto-stop
    jax.debug.callback(halt_on_nan, new_params_has_nan, state.step, state.params)  # NaN auto-stop

    #return loss, mod_vars, grads, state
    return state, loss, ce, logits

@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(5,),  # TODO: revert to 5 for batchnorm in pmap
    in_axes=(0, None, 0, 0, 0, None, None),
    # out_axes=(0, 0),
    # devices=global_devices
)
def train_step_rnn(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 3
        batch_inputs: Tuple[jax.Array, jax.Array], # 4
        batch_labels: jax.Array, # 5
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 6
        batchnorm: bool, # 7
        init_hiddens: Tuple, 
    ):
    #print('tracing par_loss_and_grad')

    #Never reset the hidden states:
    
    batch_inputs=repeat_book(*batch_inputs,True)
    # batch_integration_timesteps=repeat_book(*batch_integration_timesteps)
    
    
    def loss_fn(params):
        def single_elem_loss(carry,xs):
            shapes=jax.tree_util.tree_map(lambda x: x.shape,xs)  # mixed precision overflow debug
            print("Shapes before using:",shapes)  # mixed precision overflow debug
            batch_inputs,batch_integration_timesteps,batch_labels=xs
            dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*len(hiddens)
            hiddens=carry
            if batchnorm:
                (hiddens,logits), mod_vars = state.apply_fn( 
                    {"params": params, "batch_stats": state.batch_stats},
                    hiddens,
                    *batch_inputs,
                    *dones,
                    *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates", "batch_stats"],
                    method='__call_rnn__'
                )
            else:
                (hiddens,logits), mod_vars = state.apply_fn(
                    {"params": params},
                    hiddens,
                    *batch_inputs,
                    *dones,
                    *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates"],
                    method='__call_rnn__'
                )
            
            
            ce=cross_entropy_loss(logits, batch_labels)
            # jax.debug.print("Shape of CE: {}", ce.shape)
            # average cross-ent loss
            ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
            ce=ce.at[:,:,TIME_START_I:TIME_END_I].set(0)
            ce=ce.reshape(ce.shape[0],-1)
            loss = np.mean(ce)
            return (hiddens),(loss,mod_vars)
        # jax.debug.print("Shape of loss: {}", loss.shape)
        xs=(batch_inputs,batch_integration_timesteps,batch_labels)
        xs=jax.tree_util.tree_map(lambda x: np.array(np.split(x,2,axis=1)),xs)
        hiddens,y=jax.lax.scan(single_elem_loss,init_hiddens,xs)
        losses,mod_vars=y
        loss=np.mean(losses)
        mod_vars=jax.tree_util.tree_map(np.mean,mod_vars)
        return loss, mod_vars

    (loss, mod_vars), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # UPDATE
    # calculate means over device dimension (first)
    loss = jax.lax.pmean(loss, axis_name="batch_devices")
    grads = jax.lax.pmean(grads, axis_name="batch_devices")

    if batchnorm:
        mod_vars = jax.lax.pmean(mod_vars, axis_name="batch_devices")
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    #return loss, mod_vars, grads, state
    return state, loss

@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(5,),  # TODO: revert to 5 for batchnorm in pmap
    in_axes=(0, None, 0, 0, 0, None),
    # out_axes=(0, 0),
    # devices=global_devices
)
def train_step_old(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 3
        batch_inputs: Tuple[jax.Array, jax.Array], # 4
        batch_labels: jax.Array, # 5
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 6
        batchnorm: bool, # 7
    ):
    #print('tracing par_loss_and_grad')
    def loss_fn(params):
        if batchnorm:
            logits, mod_vars = state.apply_fn( 
                {"params": params, "batch_stats": state.batch_stats},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = state.apply_fn(
                {"params": params},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )

        # average cross-ent loss
        loss = np.mean(cross_entropy_loss(logits, batch_labels))

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)



    # UPDATE
    # calculate means over device dimension (first)
    loss = jax.lax.pmean(loss, axis_name="batch_devices")
    grads = jax.lax.pmean(grads, axis_name="batch_devices")

    if batchnorm:
        mod_vars = jax.lax.pmean(mod_vars, axis_name="batch_devices")
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    #return loss, mod_vars, grads, state
    return state, loss


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
             log_ce_tables : bool =False):
    """Validation function that loops over batches"""
    # losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    losses, accuracies, preds = [], [], []
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, num_devices)
        # print("eval step with method: ", apply_method)
        # print("Validataion: Inputs 0:5:", inputs[0][0,0:5,:])
        loss, acc, pred = eval_step(
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

@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(4,5,6,8),
    in_axes=(0, 0, 0, 0, None, None, None,None,None),
    # devices=global_devices
)
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
    # print("checking for compile in eval_step function")

    batch_inputs=repeat_book(*batch_inputs,True)

    # BF16 Mixed Precision: params are already BF16 from initialization
    # No need for tree_map here - direct use saves overhead

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

        # BF16 Mixed Precision: Cast logits to FP32 for evaluation
        logits = logits.astype(np.float32)
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




