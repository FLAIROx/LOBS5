"""
Debug script to find buffer aliasing in TrainState that causes donate-twice error.

Error: "Attempt to donate the same buffer twice (flattened argument 18, first use: 8)"
This means the 18th and 8th leaves in the flattened state pytree share the same buffer.
"""

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade')

import jax
import jax.numpy as jnp
from lob.init_train import init_train_state
from argparse import Namespace


def analyze_state_aliasing(state):
    """
    Analyze a TrainState for buffer aliasing.

    Prints:
    - Total number of leaves
    - Which leaves share the same buffer (have same id)
    - Paths to aliased leaves
    """
    # Flatten the state with paths
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(state)

    print(f"\n{'='*80}")
    print(f"State Structure Analysis")
    print(f"{'='*80}")
    print(f"Total leaves in state: {len(leaves_with_paths)}")

    # Build a map of buffer id -> list of (index, path, value)
    buffer_map = {}
    for idx, (path, leaf) in enumerate(leaves_with_paths):
        if isinstance(leaf, jax.Array):
            buf_id = id(leaf)
            if buf_id not in buffer_map:
                buffer_map[buf_id] = []
            path_str = jax.tree_util.keystr(path)
            buffer_map[buf_id].append((idx, path_str, leaf))

    # Find aliased buffers (same id appears multiple times)
    aliased_buffers = {buf_id: paths for buf_id, paths in buffer_map.items() if len(paths) > 1}

    if aliased_buffers:
        print(f"\n{'='*80}")
        print(f"FOUND {len(aliased_buffers)} ALIASED BUFFERS!")
        print(f"{'='*80}")

        for buf_id, locations in aliased_buffers.items():
            print(f"\nBuffer ID {buf_id} appears {len(locations)} times:")
            for idx, path_str, leaf in locations:
                print(f"  [Index {idx:3d}] {path_str}")
                print(f"             Shape: {leaf.shape}, dtype: {leaf.dtype}")

            # Check if indices 8 and 18 are in this group
            indices = [idx for idx, _, _ in locations]
            if 8 in indices and 18 in indices:
                print(f"\n  *** THIS IS THE CULPRIT! Indices 8 and 18 share this buffer ***")
                print(f"  Index 8:  {locations[[i for i, (idx, _, _) in enumerate(locations) if idx == 8][0]][1]}")
                print(f"  Index 18: {locations[[i for i, (idx, _, _) in enumerate(locations) if idx == 18][0]][1]}")
    else:
        print(f"\n{'='*80}")
        print(f"No buffer aliasing found - all buffers are unique")
        print(f"{'='*80}")

    # Also print first 20 leaves for reference
    print(f"\n{'='*80}")
    print(f"First 20 leaves (to understand structure):")
    print(f"{'='*80}")
    for idx, (path, leaf) in enumerate(leaves_with_paths[:20]):
        path_str = jax.tree_util.keystr(path)
        if isinstance(leaf, jax.Array):
            print(f"[{idx:3d}] {path_str:60s} | Shape: {str(leaf.shape):20s} | dtype: {leaf.dtype}")
        else:
            print(f"[{idx:3d}] {path_str:60s} | Non-array: {type(leaf)}")


if __name__ == "__main__":
    # Create minimal args to initialize state
    args = Namespace(
        ssm_size_base=256,
        blocks=4,
        n_layers=4,
        n_message_layers=2,
        n_book_pre_layers=1,
        n_book_post_layers=1,
        bsz=128,
        msg_seq_len=500,
        use_book_data=True,
        use_simple_book=False,
        book_transform='default',
        book_depth=10,
        jax_seed=42,
        conj_sym=False,
        clip_eigs=False,
        bidirectional=False,
        dt_global=False,
        opt_config='standard',
        weight_decay=0.01,
        p_dropout=0.0,
        batchnorm=False,
        bn_momentum=0.9,
        mode='pool',
        prenorm=False,
        activation_fn='gelu',
        num_devices=4,
        warmup_end=1,
        epochs=100,
        lr_min=1e-6,
        cosine_anneal=True,
        ssm_lr_base=1e-3,
        lr_factor=1.0,
    )

    # Dummy values for initialization
    n_classes = 1000
    seq_len = 12000
    book_dim = 503
    book_seq_len = 500
    train_size = 2742400  # Example train size

    print("Initializing state...")
    state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        train_size=train_size,
        print_shapes=True
    )

    print("\nAnalyzing state for buffer aliasing...")
    analyze_state_aliasing(state)

    print(f"\n{'='*80}")
    print("DEBUG COMPLETE")
    print(f"{'='*80}")
