#!/usr/bin/env python
"""
Integration test: Historical Replay mode data flow.

This test verifies the complete Historical Replay data pipeline:
1. Load real LOBSTER data from disk
2. Encode to tokens using Vocab
3. Create batches for training
4. Verify batch shapes are correct
5. Test sequence windowing

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_historical_replay_flow.py
"""

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
import numpy as np
import glob
import os

from lob.encoding import Vocab, encode_msgs


# Test configuration
DATA_DIR = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021/'
TOKEN_MODE = 24  # Use 24-token mode for this test


def test_load_trading_day():
    """Test loading one day's data correctly from LOBSTER preprocessed files."""
    print("\n" + "="*60)
    print("TEST: Load trading day data")
    print("="*60)

    # Find message files
    message_pattern = os.path.join(DATA_DIR, '*_message_*_proc.npy')
    message_files = sorted(glob.glob(message_pattern))

    assert len(message_files) > 0, f"No message files found in {DATA_DIR}"
    print(f"Found {len(message_files)} trading day files")

    # Load first trading day
    day_file = message_files[0]
    day_name = os.path.basename(day_file)
    print(f"Loading: {day_name}")

    day_data = np.load(day_file)

    # Validate shape: should be (N_messages, 14) for raw preprocessed data
    assert len(day_data.shape) == 2, f"Expected 2D array, got shape {day_data.shape}"
    assert day_data.shape[1] == 14, f"Expected 14 columns (raw message fields), got {day_data.shape[1]}"

    print(f"Day data shape: {day_data.shape}")
    print(f"Number of messages: {day_data.shape[0]}")
    print(f"Data dtype: {day_data.dtype}")

    # Validate data contains reasonable values
    # Event types should be 1-4
    event_types = np.unique(day_data[:, 1])
    print(f"Event types present: {event_types}")
    assert all(1 <= et <= 4 for et in event_types), f"Invalid event types: {event_types}"

    # Direction should be 0 or 1
    directions = np.unique(day_data[:, 2])
    print(f"Directions present: {directions}")
    assert all(d in [0, 1] for d in directions), f"Invalid directions: {directions}"

    print("PASSED: Trading day data loaded correctly")
    return day_data, day_file


def test_encode_day_data(day_data):
    """Test encoding a full day's data to tokens."""
    print("\n" + "="*60)
    print("TEST: Encode day data to tokens")
    print("="*60)

    vocab = Vocab(token_mode=TOKEN_MODE)
    vocab_size = len(vocab)

    print(f"Token mode: {TOKEN_MODE}")
    print(f"Vocab size: {vocab_size}")

    # Encode all messages
    n_messages = day_data.shape[0]
    print(f"Encoding {n_messages} messages...")

    # Convert to JAX array for encoding
    day_data_jax = jnp.array(day_data)

    # Encode in batches to avoid memory issues
    batch_size = 10000
    encoded_batches = []

    for start_idx in range(0, n_messages, batch_size):
        end_idx = min(start_idx + batch_size, n_messages)
        batch = day_data_jax[start_idx:end_idx]
        encoded_batch = encode_msgs(batch, vocab.ENCODING, token_mode=TOKEN_MODE)
        encoded_batches.append(np.array(encoded_batch))

    encoded_tokens = np.concatenate(encoded_batches, axis=0)

    # Validate shape
    expected_shape = (n_messages, TOKEN_MODE)
    assert encoded_tokens.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {encoded_tokens.shape}"
    print(f"Encoded tokens shape: {encoded_tokens.shape}")

    # Validate token range
    min_tok = encoded_tokens.min()
    max_tok = encoded_tokens.max()
    print(f"Token range: [{min_tok}, {max_tok}]")

    assert min_tok >= 0, f"Found negative token: {min_tok}"
    assert max_tok < vocab_size, f"Token {max_tok} exceeds vocab size {vocab_size}"

    # Show sample
    print(f"Sample encoded message (first):\n{encoded_tokens[0]}")

    print("PASSED: Day data encoded correctly")
    return encoded_tokens


def test_create_windows(encoded_tokens, window_size=128, stride=64):
    """Test creating fixed-length windows from the token sequence."""
    print("\n" + "="*60)
    print("TEST: Create windows from sequence")
    print("="*60)

    n_messages, tokens_per_msg = encoded_tokens.shape

    print(f"Input: {n_messages} messages, {tokens_per_msg} tokens each")
    print(f"Window size: {window_size} messages")
    print(f"Stride: {stride} messages")

    # Flatten tokens to sequence
    flat_tokens = encoded_tokens.reshape(-1)
    seq_len = len(flat_tokens)
    print(f"Flattened sequence length: {seq_len} tokens")

    # Create windows (message-level windowing)
    windows = []
    window_tokens = window_size * tokens_per_msg
    stride_tokens = stride * tokens_per_msg

    for start_idx in range(0, n_messages - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = encoded_tokens[start_idx:end_idx]
        windows.append(window)

    if len(windows) == 0:
        print(f"SKIP: Not enough messages ({n_messages}) for window size {window_size}")
        return None

    windows = np.array(windows)

    # Validate shape
    n_windows = windows.shape[0]
    expected_window_shape = (n_windows, window_size, tokens_per_msg)
    assert windows.shape == expected_window_shape, \
        f"Expected shape {expected_window_shape}, got {windows.shape}"

    print(f"Created {n_windows} windows")
    print(f"Windows array shape: {windows.shape}")

    # Verify window content is correct
    # First window should match first window_size messages
    assert np.array_equal(windows[0], encoded_tokens[:window_size]), \
        "First window content mismatch"

    # Second window should match messages starting at stride
    if n_windows > 1:
        assert np.array_equal(windows[1], encoded_tokens[stride:stride + window_size]), \
            "Second window content mismatch (stride not applied correctly)"

    print("PASSED: Windows created correctly")
    return windows


def test_batch_creation(windows, batch_size=32):
    """Test creating batches of windows for training."""
    print("\n" + "="*60)
    print("TEST: Create training batches")
    print("="*60)

    if windows is None:
        print("SKIP: No windows available")
        return None

    n_windows = windows.shape[0]
    window_size = windows.shape[1]
    tokens_per_msg = windows.shape[2]

    print(f"Input: {n_windows} windows")
    print(f"Batch size: {batch_size}")

    # Calculate number of complete batches
    n_batches = n_windows // batch_size

    if n_batches == 0:
        print(f"SKIP: Not enough windows ({n_windows}) for batch size {batch_size}")
        return None

    # Create batches
    batches = []
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = windows[start_idx:end_idx]
        batches.append(batch)

    batches = np.array(batches)

    # Validate shape
    expected_shape = (n_batches, batch_size, window_size, tokens_per_msg)
    assert batches.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {batches.shape}"

    print(f"Created {n_batches} batches")
    print(f"Batches array shape: {batches.shape}")
    print(f"  - {n_batches} batches")
    print(f"  - {batch_size} windows per batch")
    print(f"  - {window_size} messages per window")
    print(f"  - {tokens_per_msg} tokens per message")

    # Calculate total training examples
    total_examples = n_batches * batch_size
    print(f"Total training examples: {total_examples}")

    print("PASSED: Training batches created correctly")
    return batches


def test_shuffle_batches(windows, batch_size=32, seed=42):
    """Test that shuffling maintains data integrity."""
    print("\n" + "="*60)
    print("TEST: Shuffle batches maintains data integrity")
    print("="*60)

    if windows is None:
        print("SKIP: No windows available")
        return

    n_windows = windows.shape[0]

    if n_windows < batch_size:
        print(f"SKIP: Not enough windows ({n_windows}) for batch size {batch_size}")
        return

    print(f"Input: {n_windows} windows")
    print(f"Batch size: {batch_size}")

    # Create original order
    np.random.seed(seed)
    indices_original = np.arange(n_windows)

    # Shuffle indices
    indices_shuffled = np.random.permutation(n_windows)

    # Verify shuffle changed order (with high probability)
    is_same_order = np.array_equal(indices_original, indices_shuffled)
    print(f"Order changed after shuffle: {not is_same_order}")

    # Verify all indices are preserved
    assert len(np.unique(indices_shuffled)) == n_windows, \
        "Shuffle lost or duplicated indices"
    assert set(indices_shuffled) == set(indices_original), \
        "Shuffle changed the set of indices"
    print("All indices preserved after shuffle")

    # Create shuffled windows
    windows_shuffled = windows[indices_shuffled]

    # Verify shapes match
    assert windows_shuffled.shape == windows.shape, \
        f"Shuffled shape {windows_shuffled.shape} != original {windows.shape}"

    # Verify data integrity: sum of all values should be the same
    original_sum = windows.sum()
    shuffled_sum = windows_shuffled.sum()
    assert np.isclose(original_sum, shuffled_sum), \
        f"Data sum changed: {original_sum} vs {shuffled_sum}"
    print(f"Data integrity verified (sum: {original_sum})")

    # Verify we can recover original order
    inverse_indices = np.argsort(indices_shuffled)
    windows_recovered = windows_shuffled[inverse_indices]
    assert np.array_equal(windows_recovered, windows), \
        "Could not recover original order from shuffled data"
    print("Original order recoverable from shuffle")

    # Test batching after shuffle
    n_batches = n_windows // batch_size
    if n_batches > 0:
        batched_shuffled = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = windows_shuffled[start_idx:end_idx]
            batched_shuffled.append(batch)
        batched_shuffled = np.array(batched_shuffled)

        expected_shape = (n_batches, batch_size, windows.shape[1], windows.shape[2])
        assert batched_shuffled.shape == expected_shape, \
            f"Shuffled batch shape {batched_shuffled.shape} != expected {expected_shape}"
        print(f"Shuffled batches shape: {batched_shuffled.shape}")

    print("PASSED: Shuffle maintains data integrity")


def run_all_tests():
    """Run all Historical Replay flow integration tests."""
    print("="*60)
    print("Integration Test: Historical Replay Data Flow")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Token mode: {TOKEN_MODE}")

    try:
        # Test 1: Load trading day data
        day_data, day_file = test_load_trading_day()

        # Test 2: Encode day data to tokens
        encoded_tokens = test_encode_day_data(day_data)

        # Test 3: Create windows from sequence
        # Use smaller window for testing (full day may be very large)
        windows = test_create_windows(encoded_tokens, window_size=128, stride=64)

        # Test 4: Create training batches
        batches = test_batch_creation(windows, batch_size=32)

        # Test 5: Shuffle maintains data integrity
        test_shuffle_batches(windows, batch_size=32)

        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)

        # Summary statistics
        print("\nSummary:")
        print(f"  Trading day file: {os.path.basename(day_file)}")
        print(f"  Raw messages: {day_data.shape[0]}")
        print(f"  Encoded tokens shape: {encoded_tokens.shape}")
        if windows is not None:
            print(f"  Windows created: {windows.shape[0]}")
        if batches is not None:
            print(f"  Batches created: {batches.shape[0]}")

        return 0

    except FileNotFoundError as e:
        print(f"\nTEST SKIPPED: {e}")
        print("Data directory not found. This is expected in CI environments.")
        return 0  # Return 0 to not fail CI when data is not available

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
