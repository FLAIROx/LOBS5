#!/usr/bin/env python3
"""
A6: Historical Replay 数据加载验证
目标: 验证预编码数据能正确加载
"""

import sys
import os
import glob

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np


# Constants
DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG/2021"
TOKEN_MODE = 24
VOCAB_SIZE = 2112  # for token_mode=24


def test_historical_data_load():
    """验证预编码数据能正确加载"""

    print("=" * 60)
    print("A6: Historical Replay 数据加载验证")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print(f"Token mode: {TOKEN_MODE}")
    print(f"Expected vocab size: {VOCAB_SIZE}")

    # Find data files (only message files, not orderbook files)
    print("\n[1/5] Finding data files...")
    files = sorted(glob.glob(f"{DATA_PATH}/*_message_*.npy"))
    assert len(files) > 0, f"No message .npy files found in {DATA_PATH}"
    print(f"  ✓ Found {len(files)} data files")
    print(f"  First file: {os.path.basename(files[0])}")
    print(f"  Last file: {os.path.basename(files[-1])}")

    # Load first file
    print("\n[2/5] Loading first data file...")
    data = np.load(files[0])
    print(f"  ✓ Data loaded")
    print(f"    Shape: {data.shape}")
    print(f"    Dtype: {data.dtype}")

    # Verify shape
    print("\n[3/5] Verifying data shape...")
    # Expected: (n_messages, n_tokens_per_message) or (n_tokens,)
    if len(data.shape) == 2:
        n_messages, tokens_per_msg = data.shape
        print(f"  ✓ 2D data: {n_messages} messages × {tokens_per_msg} tokens/message")
    elif len(data.shape) == 1:
        n_tokens = data.shape[0]
        print(f"  ✓ 1D data: {n_tokens} total tokens")
    else:
        raise ValueError(f"Unexpected shape: {data.shape}")

    # Verify token range
    print("\n[4/5] Verifying token range...")
    min_token = data.min()
    max_token = data.max()
    print(f"  Min token: {min_token}")
    print(f"  Max token: {max_token}")

    assert min_token >= 0, f"Token values should be >= 0, got {min_token}"
    print("  ✓ Min token >= 0")

    assert max_token < VOCAB_SIZE, f"Token values should be < {VOCAB_SIZE}, got {max_token}"
    print(f"  ✓ Max token < {VOCAB_SIZE} (vocab size)")

    # Sample statistics
    print("\n[5/5] Sample statistics...")
    unique_tokens = len(np.unique(data))
    print(f"  Unique tokens in file: {unique_tokens}")
    print(f"  Coverage: {unique_tokens / VOCAB_SIZE * 100:.1f}% of vocab")

    # Load a few more files to verify consistency
    n_samples = min(5, len(files))
    print(f"\n  Checking {n_samples} more files for consistency...")
    for f in files[1:n_samples+1]:
        d = np.load(f)
        assert d.min() >= 0 and d.max() < VOCAB_SIZE, f"Invalid tokens in {f}"
    print("  ✓ All sampled files have valid token ranges")

    # Summary
    print("\n" + "=" * 60)
    print("✅ A6: Historical Replay 数据加载验证通过")
    print("=" * 60)

    return data


def main():
    """Main entry point"""
    data = test_historical_data_load()
    return 0


if __name__ == "__main__":
    sys.exit(main())
