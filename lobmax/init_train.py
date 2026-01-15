"""
LOBMAX Training Initialization

Functions to initialize LOBMAX model for training.
"""

from __future__ import annotations
import os
import sys
from argparse import Namespace
from functools import partial
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from flax.training.train_state import TrainState
import optax

# Add MaxText to path
MAXTEXT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'maxtext', 'src')
if MAXTEXT_PATH not in sys.path:
    sys.path.insert(0, MAXTEXT_PATH)

from lobmax.config import LOBMAXConfig
from lobmax.models import LOBMAXModel, BatchLOBMAXModel, get_model_summary, count_parameters


def create_lobmax_config(args: Namespace, n_classes: int, book_dim: int) -> LOBMAXConfig:
    """
    Create LOBMAX config from training arguments.
    
    Args:
        args: Training arguments namespace
        n_classes: Number of output classes
        book_dim: Book state dimension
        
    Returns:
        LOBMAXConfig instance
    """
    # Calculate head_dim from d_model and num_heads
    num_heads = getattr(args, 'num_heads', 16)
    d_model = getattr(args, 'd_model', 2048)
    head_dim = d_model // num_heads
    
    # Calculate sequence length
    token_mode = getattr(args, 'token_mode', 24)
    msg_seq_len = getattr(args, 'msg_seq_len', 500)
    max_target_length = token_mode * msg_seq_len + 256  # Add buffer
    
    return LOBMAXConfig(
        # Core parameters
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=getattr(args, 'num_kv_heads', num_heads),
        head_dim=head_dim,
        mlp_dim=getattr(args, 'mlp_dim', 4 * d_model),
        
        # Layer configuration
        n_message_layers=getattr(args, 'n_message_layers', 2),
        n_book_pre_layers=getattr(args, 'n_book_pre_layers', 1),
        n_book_post_layers=getattr(args, 'n_book_post_layers', 1),
        n_layers=getattr(args, 'n_layers', 24),
        
        # Vocabulary and dimensions
        vocab_size=n_classes,  # Message tokens
        d_book=book_dim,
        n_classes=n_classes,
        
        # Sequence length
        max_target_length=max_target_length,
        
        # Attention configuration
        attention=getattr(args, 'attention_kernel', 'flash'),
        float32_qk_product=True,
        float32_logits=True,
        
        # RoPE configuration
        rope_type=getattr(args, 'rope_type', 'llama3.1'),
        rope_max_timescale=10000,
        
        # Normalization and dropout
        prenorm=getattr(args, 'prenorm', True),
        dropout_rate=getattr(args, 'p_dropout', 0.0),
        enable_dropout=getattr(args, 'p_dropout', 0.0) > 0,
        
        # Data types
        dtype="bfloat16" if getattr(args, 'use_bf16', True) else "float32",
        weight_dtype="bfloat16" if getattr(args, 'use_bf16', True) else "float32",
        
        # Pooling mode
        mode=getattr(args, 'mode', 'none'),
        
        # FFN activation (SwiGLU)
        mlp_activations=("silu", "linear"),
    )


def create_lobmax_optimizer(
    lr_schedule,
    weight_decay: float = 0.05,
) -> optax.GradientTransformation:
    """
    Create optimizer for LOBMAX.
    
    Uses AdamW with learning rate schedule.
    
    Args:
        lr_schedule: Optax learning rate schedule
        weight_decay: Weight decay coefficient
        
    Returns:
        Optax optimizer
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay,
        ),
    )
    return optimizer


def init_lobmax_train_state(
    args: Namespace,
    n_classes: int,
    seq_len: int,
    book_dim: int,
    book_seq_len: int,
    train_size: int,
    mesh: Mesh,
    print_shapes: bool = False,
) -> Tuple[TrainState, LOBMAXModel, int]:
    """
    Initialize LOBMAX training state.
    
    Args:
        args: Training arguments
        n_classes: Number of output classes
        seq_len: Message sequence length
        book_dim: Book state dimension
        book_seq_len: Book sequence length
        train_size: Training set size (for schedule calculation)
        mesh: JAX device mesh
        print_shapes: Whether to print shape information
        
    Returns:
        Tuple of (TrainState, model, total_params)
    """
    # Create config
    config = create_lobmax_config(args, n_classes, book_dim)
    
    if print_shapes:
        print(get_model_summary(config))
    
    # Create model
    model = LOBMAXModel(
        config=config,
        mesh=mesh,
        training=True,
    )
    
    # Initialize with dummy inputs
    key = random.PRNGKey(getattr(args, 'jax_seed', 42))
    init_rng, _ = random.split(key)
    
    # Create dummy inputs for initialization
    batch_size = 1
    dummy_msg = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_book = jnp.zeros((batch_size, seq_len, book_dim), dtype=jnp.float32)
    dummy_timesteps = jnp.ones((batch_size, seq_len))
    
    # Initialize parameters
    variables = model.init(
        init_rng,
        x_m=dummy_msg,
        x_b=dummy_book,
        message_integration_timesteps=dummy_timesteps,
        book_integration_timesteps=dummy_timesteps,
    )
    params = variables['params']
    
    # Count parameters
    total_params = count_parameters(params)
    if print_shapes:
        print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    
    # Create learning rate schedule
    steps_per_epoch = train_size // args.global_bsz
    if hasattr(args, 'curtail_epochs') and args.curtail_epochs is not None:
        steps_per_epoch = min(steps_per_epoch, args.curtail_epochs + 1)
    
    total_steps = steps_per_epoch * args.epochs
    warmup_end_step = steps_per_epoch * getattr(args, 'warmup_end', 1)
    
    # Base learning rate
    lr_base = getattr(args, 'lr_base', getattr(args, 'ssm_lr_base', 0.0005))
    lr_factor = getattr(args, 'lr_factor', 1.0)
    lr = lr_base * lr_factor
    lr_min = getattr(args, 'lr_min', 1e-6)
    use_cosine = getattr(args, 'cosine_anneal', True)
    
    if print_shapes:
        print(f"[LOBMAX Schedule] steps_per_epoch: {steps_per_epoch}")
        print(f"[LOBMAX Schedule] total_steps: {total_steps}")
        print(f"[LOBMAX Schedule] warmup_end_step: {warmup_end_step}")
        print(f"[LOBMAX Schedule] Base LR: {lr}, LR min: {lr_min}")
    
    # Create schedule
    if use_cosine:
        # Warmup + Cosine annealing
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=lr,
                    transition_steps=warmup_end_step,
                ),
                optax.cosine_decay_schedule(
                    init_value=lr,
                    decay_steps=total_steps - warmup_end_step,
                    alpha=lr_min / lr,
                ),
            ],
            boundaries=[warmup_end_step],
        )
    else:
        # Just warmup + constant
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=lr,
                    transition_steps=warmup_end_step,
                ),
                optax.constant_schedule(lr),
            ],
            boundaries=[warmup_end_step],
        )
    
    # Create optimizer
    weight_decay = getattr(args, 'weight_decay', 0.05)
    optimizer = create_lobmax_optimizer(lr_schedule, weight_decay)
    
    # Create TrainState
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    
    # Create model_cls partial (compatible with S5 interface)
    # This allows: model = model_cls(training=False, step_rescale=1)
    model_cls = partial(
        LOBMAXModel,
        config=config,
        mesh=mesh,
    )
    
    return state, model_cls, total_params


def create_lobmax_model_cls(args: Namespace, n_classes: int, book_dim: int, mesh: Mesh):
    """
    Create a partial model class for LOBMAX (compatibility with S5 interface).
    
    This allows the existing training loop to work with LOBMAX.
    
    Args:
        args: Training arguments
        n_classes: Number of output classes
        book_dim: Book state dimension
        mesh: JAX device mesh
        
    Returns:
        Partial function that creates LOBMAXModel
    """
    config = create_lobmax_config(args, n_classes, book_dim)
    
    return partial(
        LOBMAXModel,
        config=config,
        mesh=mesh,
    )
