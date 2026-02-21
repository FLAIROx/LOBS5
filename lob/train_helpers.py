from functools import partial
import numpy as onp
import jax
import jax.numpy as np
# from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any, Dict, Optional, Tuple, Union
from lob.encoding import Message_Tokenizer
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import sys

import psutil
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
    Creates an optax Schedule: warmup -> cosine decay (or constant).
    Passed directly to the optimizer, eliminating manual per-step LR updates.
    """
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_end_step
    )

    if use_cosine_anneal:
        cosine_steps = total_steps - warmup_end_step

        def make_cos_schedule(init_lr, final_lr, len_steps):
            """Custom cosine schedule matching LOBS5's original cosine_annealing."""
            def schedule(step):
                pct = step / len_steps
                pct = np.minimum(pct, 1.0)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * pct))
                lr = (init_lr - final_lr) * cosine_decay + final_lr
                return lr
            return schedule

        cosine_schedule = make_cos_schedule(base_lr, lr_min, cosine_steps)
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[warmup_end_step]
        )
    else:
        constant_schedule = optax.constant_schedule(base_lr)
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, constant_schedule],
            boundaries=[warmup_end_step]
        )

    return schedule


def update_learning_rate_per_step(lr_params, state, mesh=None):
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
    # Multi-host: use globally-replicated JAX arrays to match train_step's
    # in_shardings. numpy arrays become SingleDeviceSharding which causes
    # NCCL deadlock when train_step tries to re-shard across hosts.
    if mesh is not None:
        from jax.sharding import NamedSharding, PartitionSpec as P
        replicated = NamedSharding(mesh, P())
        lr_array = jax.make_array_from_process_local_data(
            replicated, np.array(lr_val, dtype=np.float32))
        ssm_lr_array = jax.make_array_from_process_local_data(
            replicated, np.array(ssm_lr_val, dtype=np.float32))
    else:
        lr_array = np.array(lr_val, dtype=np.float32)
        ssm_lr_array = np.array(ssm_lr_val, dtype=np.float32)
    
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
                            'learning_rate': lr_array
                        }
                    )
                ),
                'ssm': state.opt_state.inner_states['ssm']._replace(
                    inner_state=state.opt_state.inner_states['ssm'].inner_state._replace(
                        hyperparams={
                            **state.opt_state.inner_states['ssm'].inner_state.hyperparams,
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
                                'learning_rate': ssm_lr_array
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
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       ssm_lr_schedule=None,
                       lr_schedule=None,
                       dt_global=False,
                       num_devices=1,
                       ):
    """
    Initializes the training state using optax.

    When ssm_lr_schedule/lr_schedule are provided (optax.Schedule functions),
    they are passed directly to the optimizer — no inject_hyperparams needed.
    When None, falls back to inject_hyperparams with scalar LRs (legacy mode).
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

    # Determine whether to use optax schedules (new) or inject_hyperparams (legacy)
    use_schedules = ssm_lr_schedule is not None and lr_schedule is not None
    if use_schedules:
        print("[Optimizer] Using optax schedules (LR managed inside JIT)")
        _ssm_lr = ssm_lr_schedule
        _lr = lr_schedule
    else:
        print("[Optimizer] Using inject_hyperparams (legacy scalar LR)")
        _ssm_lr = ssm_lr
        _lr = lr

    def _make_opt(optimizer_fn, learning_rate, **kwargs):
        """Create optimizer with or without inject_hyperparams."""
        if use_schedules:
            return optimizer_fn(learning_rate=learning_rate, **kwargs)
        else:
            return optax.inject_hyperparams(optimizer_fn)(learning_rate=learning_rate, **kwargs)

    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
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
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
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
                "none": _make_opt(optax.adamw, _ssm_lr, weight_decay=weight_decay),
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
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
                "none": optax.adamw(learning_rate=0.0, weight_decay=0.0),
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
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
                "none": optax.sgd(learning_rate=0.0),
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    #print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # jit+sharding: state replication handled in train.py via create_state_shardings
    print(f"[*] State params embedding shape: {state.params['message_encoder']['encoder']['embedding'].shape}")

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

    # jit+sharding: no device_reshape needed — sharding handles data distribution
    inputs, labels, integration_times = _prep_batch_par(
        inputs,
        targets,
        seq_len,
        book_data,
        timestep_msg,
        timestep_book,
    )

    return inputs, labels, integration_times

@partial(
    jax.jit,
    static_argnums=(2,),
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
        mesh=None,
        jit_train_step_fn=None,
    ):

    """
    Training function for an epoch that loops over batches.

    lr_params: If None, LR is managed by optax schedules (no manual update).
               If provided, legacy mode with update_learning_rate_per_step.
    """
    # Store Metrics
    batch_losses = []
    cross_entropies= [] #list of 1xNTok losses

    use_optax_schedules = lr_params is None
    if not use_optax_schedules:
        decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params
    else:
        step = int(state.step)
    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for batch_idx, batch in enumerate(tqdm(trainloader)):
        # print(f"train_epoch: Epoch {epoch} - Batch {batch_idx} / {len(trainloader)}")
        # print(f"train_epoch: Batch input shape: {batch[0].shape}, batch target shape: {batch[1].shape}")
        if not debug_loading:
            if (step>1) & (step<3) & debug_profiler:
                jax.profiler.start_trace("/tmp/tensorboard")
            inputs, labels, integration_times = prep_batch(batch, seq_len, num_devices)

            # jit+sharding: place data on devices with correct sharding
            # Use make_array_from_process_local_data for multi-host: each process
            # provides its local shard, JAX assembles the global array.
            if mesh is not None:
                from lob.sharding_utils import get_data_shardings_for_batch
                inputs_sh, labels_sh, times_sh = get_data_shardings_for_batch(mesh, has_book_data=(len(inputs) > 1))
                inputs = tuple(jax.make_array_from_process_local_data(sh, inp) for inp, sh in zip(inputs, inputs_sh))
                labels = jax.make_array_from_process_local_data(labels_sh, labels)
                integration_times = tuple(jax.make_array_from_process_local_data(sh, ts) for ts, sh in zip(integration_times, times_sh))

            rng, drop_rng = jax.random.split(rng)
            if batch_idx % 1000 == 0:
                print(f"\n=== Epoch {epoch}, Batch {batch_idx} ===")
                print_memory_usage()

            train_fn = jit_train_step_fn if jit_train_step_fn is not None else train_step
            state, loss, ce, logits = train_fn(
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

            # jit+sharding: loss is already a scalar (no device dimension)
            batch_losses.append(loss)
            if log_ce_tables:
                cross_entropies.append(ce)

            if use_optax_schedules:
                # LR managed by optax — no manual update needed
                step = int(state.step)
            else:
                # Legacy mode: manual per-step LR update
                lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
                state, step = update_learning_rate_per_step(lr_params, state, mesh=mesh)

            if (step>20) & (step<=21) & debug_profiler:
                jax.profiler.stop_trace()
                break
            if (curtail_epochs is not None) and (batch_idx>=curtail_epochs):
                print("Ending epoch early at step", step, "due to curtail_epoch arg.")
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



    # jit+sharding: no pmean needed — sharding handles cross-device aggregation
    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss, ce, logits

@partial(
    jax.jit,
    static_argnums=(5,),
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
            shapes=jax.tree_util.tree_map(lambda x: x.shape,xs)
            print("Shapes before using:",shapes)
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

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss

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



    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

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
             log_ce_tables : bool =False,
             mesh=None,
             jit_eval_step_fn=None):
    """Validation function that loops over batches"""
    losses, accuracies, preds = [], [], []
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, num_devices)

        # jit+sharding: place data on devices (multi-host compatible)
        if mesh is not None:
            from lob.sharding_utils import get_data_shardings_for_batch
            inputs_sh, labels_sh, times_sh = get_data_shardings_for_batch(mesh, has_book_data=(len(inputs) > 1))
            inputs = tuple(jax.make_array_from_process_local_data(sh, inp) for inp, sh in zip(inputs, inputs_sh))
            labels = jax.make_array_from_process_local_data(labels_sh, labels)
            integration_timesteps = tuple(jax.make_array_from_process_local_data(sh, ts) for ts, sh in zip(integration_timesteps, times_sh))

        eval_fn = jit_eval_step_fn if jit_eval_step_fn is not None else eval_step
        loss, acc, pred = eval_fn(
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


        # device_get → numpy before append: avoids XLA trace explosion
        # on 2D mesh when concatenating 200+ sharded arrays (C5 hang fix)
        losses.append(jax.device_get(loss))
        accuracies.append(jax.device_get(acc))
        if curtail_epoch is not None and batch_idx>=curtail_epoch:
            print(f"Ending epoch early at step {batch_idx} due to curtail_epoch arg.")
            break

    import numpy as onp
    concat_loss=onp.concatenate(losses,axis=0)
    concat_acc=onp.concatenate(accuracies,axis=0)
    print(f"Concat Loss is {concat_loss.shape}")
    print(f"Concat Acc is {concat_acc.shape}")
    if log_ce_tables:
        acc_means=onp.mean(concat_acc,axis=(0,1))
        ce_means=onp.mean(concat_loss,axis=(0,1))
    else:
        ce_means=None
        acc_means=None
    aveloss, aveaccu = onp.mean(concat_loss), onp.mean(onp.asarray(accuracies))
    del losses, accuracies
    return aveloss, aveaccu, ce_means,acc_means

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


# ============================================================================
# JIT-compiled step functions (replacement for @pmap decorators)
# ============================================================================

def create_jit_train_step(mesh, state, has_book_data=True, hierarchical=False,
                          batchnorm=False, ignore_times=True):
    """Create JIT-compiled train_step with explicit sharding.

    hierarchical=True: uses shard_map + explicit pmean('gpus') + pmean('nodes')
    for hierarchical AllReduce on 2D mesh. Requires mesh with ('nodes','gpus') axes.
    """
    if hierarchical:
        return _create_hierarchical_train_step(mesh, has_book_data, batchnorm, ignore_times)

    from lob.sharding_utils import create_state_shardings, get_data_shardings_for_batch

    state_shardings = create_state_shardings(state, mesh)
    inputs_shardings, labels_sharding, timesteps_shardings = get_data_shardings_for_batch(mesh, has_book_data=has_book_data)

    in_shardings = (
        state_shardings,    # state - replicated
        None,               # rng
        inputs_shardings,   # batch_inputs - sharded
        labels_sharding,    # batch_labels - sharded
        timesteps_shardings,# batch_integration_timesteps - sharded
    )
    out_shardings = (
        state_shardings,    # state
        None,               # loss
        None,               # ce
        None,               # logits
    )

    jit_train_step = jax.jit(
        train_step,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=(5, 6),  # batchnorm, ignore_times
        donate_argnums=(0,),    # donate state for memory reuse
    )
    print("[JIT] Created JIT-compiled train_step with sharding")
    return jit_train_step


def _create_hierarchical_train_step(mesh, has_book_data, batchnorm, ignore_times):
    """Create shard_map-based train_step with hierarchical AllReduce.

    Uses 2D mesh ('nodes', 'gpus') to decompose gradient reduction:
    1. pmean('gpus')  — intra-node via NVLink (478 GB/s)
    2. pmean('nodes') — inter-node via Slingshot

    XLA combine threshold merges each group's pmean ops independently,
    avoiding the BlueConnect + combine deadlock (C4 experiments).
    """
    from jax.experimental.shard_map import shard_map

    batch_axis = ('nodes', 'gpus')

    # in_specs must match pytree structure of each argument
    if has_book_data:
        in_data = (P(batch_axis, None), P(batch_axis, None))
        in_times = (P(batch_axis, None), P(batch_axis, None))
    else:
        in_data = (P(batch_axis, None),)
        in_times = (P(batch_axis, None),)

    in_specs = (
        P(),            # state — replicated
        P(),            # rng — replicated
        in_data,        # batch_inputs — sharded tuple
        P(batch_axis),  # batch_labels — sharded
        in_times,       # batch_integration_timesteps — sharded tuple
    )
    # state, loss, ce are replicated after pmean; logits replaced with dummy scalar
    out_specs = (P(), P(), P(), P())

    def sharded_step(state, rng, batch_inputs, batch_labels,
                     batch_integration_timesteps):
        """Per-shard train step with hierarchical gradient reduction."""
        batch_inputs = repeat_book(*batch_inputs, True)

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
                ce_2 = ce[:, :, (TIME_END_I + 1):]
                ce = np.concatenate([ce_1, ce_2], axis=2)
                ce = ce.reshape(ce.shape[0], -1)

            ce = np.mean(ce, axis=0)
            loss = np.mean(ce)
            return loss, (mod_vars, logits, ce)

        (loss, (mod_vars, logits, ce)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)

        # ── Hierarchical AllReduce ──
        # Level 1: NVLink within each node (4 GPUs, ~478 GB/s)
        grads = jax.lax.pmean(grads, axis_name='gpus')
        loss = jax.lax.pmean(loss, axis_name='gpus')
        ce = jax.lax.pmean(ce, axis_name='gpus')
        # Level 2: Slingshot across nodes (N nodes)
        grads = jax.lax.pmean(grads, axis_name='nodes')
        loss = jax.lax.pmean(loss, axis_name='nodes')
        ce = jax.lax.pmean(ce, axis_name='nodes')

        if batchnorm:
            state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
        else:
            state = state.apply_gradients(grads=grads)

        # Return dummy scalar for logits (not needed, avoids shipping per-shard data)
        return state, loss, ce, np.float32(0.0)

    mapped_fn = shard_map(sharded_step, mesh=mesh,
                          in_specs=in_specs, out_specs=out_specs,
                          check_rep=False)
    jitted_fn = jax.jit(mapped_fn, donate_argnums=(0,))

    # API-compatible wrapper: train_epoch passes batchnorm, ignore_times as args 6-7
    def compatible_fn(state, rng, batch_inputs, batch_labels,
                      batch_integration_timesteps, _batchnorm, _ignore_times):
        return jitted_fn(state, rng, batch_inputs, batch_labels,
                         batch_integration_timesteps)

    print(f"[JIT] Created hierarchical shard_map train_step "
          f"(2D mesh, pmean('gpus') + pmean('nodes'))")
    return compatible_fn


def create_jit_eval_step(mesh, state, has_book_data=True):
    """Create JIT-compiled eval_step with explicit sharding."""
    from lob.sharding_utils import create_state_shardings, get_data_shardings_for_batch

    state_shardings = create_state_shardings(state, mesh)
    inputs_shardings, labels_sharding, timesteps_shardings = get_data_shardings_for_batch(mesh, has_book_data=has_book_data)

    in_shardings = (
        inputs_shardings,   # batch_inputs
        labels_sharding,    # batch_labels
        timesteps_shardings,# batch_integration_timesteps
        state_shardings,    # state
        None,               # init_hiddens
    )
    out_shardings = (None, None, None)

    jit_eval_step = jax.jit(
        eval_step,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=(4, 5, 6, 8),  # apply_fn, batchnorm, apply_method, ignore_times
    )
    print("[JIT] Created JIT-compiled eval_step with sharding")
    return jit_eval_step

