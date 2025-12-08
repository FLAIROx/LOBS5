#!/usr/bin/env python3
"""
Pre-encode LOBSTER data for faster training.

This script processes raw message data (shape: N,14) and encodes it
using the Vocab encoding to produce pre-encoded data (shape: N,24).
This eliminates the need to encode on-the-fly during training.

Usage:
    python pre_encode_data.py \
        --input_dir /path/to/GOOG2016TO2021 \
        --output_dir /path/to/GOOG2016TO2021_encoded \
        [--num_workers 4] \
        [--symlink_books]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import multiprocessing
from functools import partial

# Add parent directory to path to import lob modules
sys.path.insert(0, str(Path(__file__).parent))

from lob.encoding import Vocab, encode_msgs
# ✅ Use numpy version - simple, reliable, validated in training
# ⚠️ DO NOT use transform_L2_state (JAX) - complex and error-prone
from preproc import transform_L2_state_numpy


def encode_message_file(
    input_path: Path,
    output_path: Path,
    vocab_encoding: dict,
) -> tuple[bool, str]:
    """
    Encode a single message file.

    Args:
        input_path: Path to input .npy file (shape: N,14)
        output_path: Path to output .npy file (shape: N,24)
        vocab_encoding: The Vocab.ENCODING dictionary

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Load raw message data
        X_raw = np.load(input_path, mmap_mode='r')

        # Check shape
        if X_raw.ndim != 2 or X_raw.shape[1] != 14:
            return False, f"Invalid shape {X_raw.shape}, expected (N, 14)"

        # Convert to JAX array and encode
        X_raw_jax = jnp.array(X_raw)
        X_encoded = encode_msgs(X_raw_jax, vocab_encoding)

        # Convert back to numpy and save
        X_encoded_np = np.array(X_encoded, dtype=np.int32)

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save encoded data
        np.save(output_path, X_encoded_np)

        return True, f"Encoded {X_raw.shape[0]} messages -> {X_encoded_np.shape}"

    except Exception as e:
        return False, f"Error: {str(e)}"


def transform_orderbook_file(
    input_path: Path,
    output_path: Path,
    book_depth: int = 500,
    tick_size: int = 100
) -> tuple[bool, str]:
    """
    Transform a single orderbook file to volume image representation.

    Args:
        input_path: Path to input .npy file (raw L2 orderbook)
        output_path: Path to output .npy file (transformed volume image)
        book_depth: Number of price levels (default: 500)
        tick_size: Tick size for price discretization (default: 100)

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Load raw orderbook data
        book_raw = np.load(input_path, mmap_mode='r')

        # ✅ Transform using numpy version (simple, reliable, validated)
        # Note: JAX versions exist but are deprecated (see preproc.py warnings)
        book_transformed_np = transform_L2_state_numpy(book_raw, book_depth, tick_size)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save transformed data
        np.save(output_path, book_transformed_np)

        return True, f"Transformed {book_raw.shape} → {book_transformed_np.shape}"

    except Exception as e:
        return False, f"Error: {str(e)}"


def copy_or_symlink_file(
    input_path: Path,
    output_path: Path,
    use_symlink: bool = True
) -> tuple[bool, str]:
    """
    Copy or symlink orderbook files (no encoding needed).

    Args:
        input_path: Source file path
        output_path: Destination file path
        use_symlink: If True, create symlink; otherwise copy file

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if use_symlink:
            # Create relative symlink
            rel_path = os.path.relpath(input_path, output_path.parent)
            if output_path.exists():
                output_path.unlink()
            output_path.symlink_to(rel_path)
            return True, "Symlinked"
        else:
            # Copy file
            import shutil
            shutil.copy2(input_path, output_path)
            return True, "Copied"

    except Exception as e:
        return False, f"Error: {str(e)}"


def process_single_file(args_tuple):
    """Wrapper for multiprocessing."""
    input_path, output_path, vocab_encoding, is_message_file, book_depth = args_tuple

    if is_message_file:
        return input_path, encode_message_file(input_path, output_path, vocab_encoding)
    else:
        # Transform orderbook file
        return input_path, transform_orderbook_file(input_path, output_path, book_depth)


def pre_encode_directory(
    input_dir: str,
    output_dir: str,
    num_workers: int = 4,
    symlink_books: bool = True,
    vocab: Optional[Vocab] = None,
    skip_files: int = 0,
    max_files: Optional[int] = None
):
    """
    Pre-encode all message files in a directory tree.

    Args:
        input_dir: Input directory containing raw data
        output_dir: Output directory for encoded data
        num_workers: Number of parallel workers
        symlink_books: If True, symlink orderbook files; otherwise copy them
        vocab: Vocab instance (created if None)
        skip_files: Skip first N message files (for batch processing)
        max_files: Process at most N message files (for batch processing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Initialize vocab if not provided
    if vocab is None:
        print("Initializing Vocab...")
        vocab = Vocab()

    # Find all .npy files
    print(f"Scanning {input_dir} for .npy files...")
    all_files = list(input_path.rglob("*.npy"))

    if not all_files:
        print(f"Warning: No .npy files found in {input_dir}")
        return

    print(f"Found {len(all_files)} .npy files")

    # Separate message files from orderbook files
    message_files = []
    book_files = []

    for file_path in all_files:
        rel_path = file_path.relative_to(input_path)

        # Change suffix from _proc.npy to _encoded.npy
        out_name = file_path.name.replace('_proc.npy', '_encoded.npy')
        out_path = output_path / rel_path.parent / out_name

        if "message" in file_path.name:
            message_files.append((file_path, out_path, True))
        elif "orderbook" in file_path.name or "book" in file_path.name:
            book_files.append((file_path, out_path, False))
        else:
            # Assume it's a message file if unclear
            message_files.append((file_path, out_path, True))

    print(f"  - {len(message_files)} message files total")

    # Apply skip and max_files for batch processing
    if skip_files > 0 or max_files is not None:
        original_count = len(message_files)
        end_idx = skip_files + max_files if max_files is not None else len(message_files)
        message_files = message_files[skip_files:end_idx]
        print(f"  - Processing message files {skip_files} to {skip_files + len(message_files)} (batch of {len(message_files)})")

        # Only process orderbook files in the first batch (skip_files == 0)
        if skip_files == 0:
            print(f"  - {len(book_files)} orderbook files to transform (first batch only)")
        else:
            print(f"  - Skipping orderbook files (processed in first batch)")
            book_files = []  # Skip book files in subsequent batches
    else:
        print(f"  - {len(message_files)} message files to encode")
        print(f"  - {len(book_files)} orderbook files to transform")

    # Prepare arguments for multiprocessing
    args_list = []

    # Message files - need encoding
    for input_file, output_file, _ in message_files:
        args_list.append((input_file, output_file, vocab.ENCODING, True, None))

    # Book files - need transformation (only in first batch when batching)
    for input_file, output_file, _ in book_files:
        args_list.append((input_file, output_file, vocab.ENCODING, False, 500))

    # Process files
    print(f"\nProcessing with {num_workers} workers...")

    if num_workers > 1:
        # Use 'spawn' to avoid JAX multithreading + fork() deadlock
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, args_list),
                total=len(args_list),
                desc="Processing files"
            ))
    else:
        results = []
        for args in tqdm(args_list, desc="Processing files"):
            results.append(process_single_file(args))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successes = sum(1 for _, (success, _) in results if success)
    failures = sum(1 for _, (success, _) in results if not success)

    print(f"Successfully processed: {successes}/{len(results)} files")

    if failures > 0:
        print(f"\nFailed: {failures} files")
        print("\nFailed files:")
        for file_path, (success, msg) in results:
            if not success:
                print(f"  {file_path}: {msg}")

    print(f"\nOutput directory: {output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-encode LOBSTER message data for faster training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python pre_encode_data.py \\
        --input_dir /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021 \\
        --output_dir /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded

    # With custom number of workers and copying books instead of symlinking
    python pre_encode_data.py \\
        --input_dir /lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2 \\
        --output_dir /lus/lfs1aip2/home/s5e/kangli.s5e/GOOGJAN2023_encoded \\
        --num_workers 8 \\
        --copy_books
        """
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing raw message data (*.npy files with shape N,14)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for encoded data (*.npy files with shape N,24)"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4)"
    )

    parser.add_argument(
        "--copy_books",
        action="store_true",
        help="Copy orderbook files instead of creating symlinks (default: create symlinks)"
    )

    parser.add_argument(
        "--skip_files",
        type=int,
        default=0,
        help="Skip first N message files (for batch processing, default: 0)"
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Process at most N message files (for batch processing, default: all)"
    )

    args = parser.parse_args()

    print("="*70)
    print("Pre-encoding LOBSTER Data")
    print("="*70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers:          {args.num_workers}")
    print(f"Orderbook files:  Transform (Numpy - CPU only)")
    if args.skip_files > 0 or args.max_files is not None:
        print(f"Batch processing: Skip {args.skip_files}, Max {args.max_files if args.max_files else 'all'}")
    print("="*70 + "\n")

    try:
        pre_encode_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            symlink_books=not args.copy_books,
            skip_files=args.skip_files,
            max_files=args.max_files
        )
        print("\n✓ Pre-encoding completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
