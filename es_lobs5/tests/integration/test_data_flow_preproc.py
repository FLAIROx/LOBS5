#!/usr/bin/env python
"""
Integration test: Data flow from preprocessed LOBSTER data to token encoding.

This test verifies:
1. Loading real preprocessed data from disk
2. Encoding with both token_mode=22 (vocab=12012) and token_mode=24 (vocab=2112)
3. Correct output shapes: (N, 14) message -> (N, token_mode) tokens
4. All tokens within vocabulary bounds
5. Batch processing works correctly

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_data_flow_preproc.py
"""

import sys
import os
import glob
import numpy as np

# Add LOBS5 to path
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from lob.encoding import Vocab, encode_msgs


# Test configuration
DATA_DIR = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021/'
EXPECTED_INPUT_COLS = 14  # Number of columns in preprocessed message data


def load_preprocessed_data(data_dir, max_files=1):
    """Load preprocessed message data from .npy files."""
    message_files = sorted(glob.glob(os.path.join(data_dir, '*_message_*_proc.npy')))

    if not message_files:
        raise FileNotFoundError(f"No preprocessed message files found in {data_dir}")

    # Load up to max_files
    all_messages = []
    for f in message_files[:max_files]:
        data = np.load(f)
        all_messages.append(data)
        print(f"Loaded {f}: shape={data.shape}")

    return np.concatenate(all_messages, axis=0) if len(all_messages) > 1 else all_messages[0]


def test_load_preprocessed_data():
    """Test that preprocessed data can be loaded and has expected shape."""
    print("\n" + "="*60)
    print("TEST: Load preprocessed data")
    print("="*60)

    messages = load_preprocessed_data(DATA_DIR, max_files=1)

    # Check shape
    assert len(messages.shape) == 2, f"Expected 2D array, got shape {messages.shape}"
    assert messages.shape[1] == EXPECTED_INPUT_COLS, \
        f"Expected {EXPECTED_INPUT_COLS} columns, got {messages.shape[1]}"

    print(f"Data shape: {messages.shape}")
    print(f"Data dtype: {messages.dtype}")
    print(f"Sample message (first row):\n{messages[0]}")
    print("PASSED: Data loaded successfully with correct shape")

    return messages


def test_encoding_token_mode_22(messages):
    """Test encoding with token_mode=22 (vocab=12012)."""
    print("\n" + "="*60)
    print("TEST: Encoding with token_mode=22")
    print("="*60)

    vocab = Vocab(token_mode=22)
    vocab_size = len(vocab)
    expected_vocab_size = 12012

    print(f"Vocab size: {vocab_size}")
    assert vocab_size == expected_vocab_size, \
        f"Expected vocab size {expected_vocab_size}, got {vocab_size}"

    # Encode a batch of messages
    batch_size = min(1000, len(messages))
    batch = messages[:batch_size]

    encoded = encode_msgs(batch, vocab.ENCODING, token_mode=22)
    encoded_np = np.array(encoded)

    # Check output shape
    expected_shape = (batch_size, 22)
    assert encoded_np.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {encoded_np.shape}"
    print(f"Encoded shape: {encoded_np.shape}")

    # Check all tokens within vocab bounds
    min_tok = encoded_np.min()
    max_tok = encoded_np.max()
    print(f"Token range: [{min_tok}, {max_tok}]")

    assert min_tok >= 0, f"Found negative token: {min_tok}"
    assert max_tok < vocab_size, \
        f"Token {max_tok} exceeds vocab size {vocab_size}"

    print(f"Sample encoded message:\n{encoded_np[0]}")
    print("PASSED: token_mode=22 encoding works correctly")

    return encoded_np


def test_encoding_token_mode_24(messages):
    """Test encoding with token_mode=24 (vocab=2112)."""
    print("\n" + "="*60)
    print("TEST: Encoding with token_mode=24")
    print("="*60)

    vocab = Vocab(token_mode=24)
    vocab_size = len(vocab)
    expected_vocab_size = 2112

    print(f"Vocab size: {vocab_size}")
    assert vocab_size == expected_vocab_size, \
        f"Expected vocab size {expected_vocab_size}, got {vocab_size}"

    # Encode a batch of messages
    batch_size = min(1000, len(messages))
    batch = messages[:batch_size]

    encoded = encode_msgs(batch, vocab.ENCODING, token_mode=24)
    encoded_np = np.array(encoded)

    # Check output shape
    expected_shape = (batch_size, 24)
    assert encoded_np.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {encoded_np.shape}"
    print(f"Encoded shape: {encoded_np.shape}")

    # Check all tokens within vocab bounds
    min_tok = encoded_np.min()
    max_tok = encoded_np.max()
    print(f"Token range: [{min_tok}, {max_tok}]")

    assert min_tok >= 0, f"Found negative token: {min_tok}"
    assert max_tok < vocab_size, \
        f"Token {max_tok} exceeds vocab size {vocab_size}"

    print(f"Sample encoded message:\n{encoded_np[0]}")
    print("PASSED: token_mode=24 encoding works correctly")

    return encoded_np


def test_batch_processing(messages):
    """Test that batch processing works correctly for various batch sizes."""
    print("\n" + "="*60)
    print("TEST: Batch processing")
    print("="*60)

    vocab_22 = Vocab(token_mode=22)
    vocab_24 = Vocab(token_mode=24)

    batch_sizes = [1, 10, 100, 500]

    for batch_size in batch_sizes:
        if batch_size > len(messages):
            print(f"Skipping batch_size={batch_size} (not enough data)")
            continue

        batch = messages[:batch_size]

        # Test token_mode=22
        encoded_22 = encode_msgs(batch, vocab_22.ENCODING, token_mode=22)
        assert np.array(encoded_22).shape == (batch_size, 22), \
            f"token_mode=22: Wrong shape for batch_size={batch_size}"

        # Test token_mode=24
        encoded_24 = encode_msgs(batch, vocab_24.ENCODING, token_mode=24)
        assert np.array(encoded_24).shape == (batch_size, 24), \
            f"token_mode=24: Wrong shape for batch_size={batch_size}"

        print(f"Batch size {batch_size}: PASSED")

    print("PASSED: Batch processing works correctly")


def test_consistency_across_batches(messages):
    """Test that encoding is consistent whether done in batch or individually."""
    print("\n" + "="*60)
    print("TEST: Consistency across batches")
    print("="*60)

    vocab = Vocab(token_mode=22)

    # Encode 10 messages as a batch
    batch = messages[:10]
    batch_encoded = np.array(encode_msgs(batch, vocab.ENCODING, token_mode=22))

    # Encode same messages individually
    individual_encoded = []
    for i in range(10):
        single = messages[i:i+1]
        enc = np.array(encode_msgs(single, vocab.ENCODING, token_mode=22))
        individual_encoded.append(enc[0])
    individual_encoded = np.array(individual_encoded)

    # Compare
    assert np.array_equal(batch_encoded, individual_encoded), \
        "Batch and individual encoding produced different results"

    print("PASSED: Encoding is consistent across batches")


def test_special_tokens_handling():
    """Test that special tokens are properly defined."""
    print("\n" + "="*60)
    print("TEST: Special tokens handling")
    print("="*60)

    for token_mode in [22, 24]:
        vocab = Vocab(token_mode=token_mode)

        assert vocab.MASK_TOK == 0, f"MASK_TOK should be 0, got {vocab.MASK_TOK}"
        assert vocab.HIDDEN_TOK == 1, f"HIDDEN_TOK should be 1, got {vocab.HIDDEN_TOK}"
        assert vocab.NA_TOK == 2, f"NA_TOK should be 2, got {vocab.NA_TOK}"
        assert vocab.START_TOK == 3, f"START_TOK should be 3, got {vocab.START_TOK}"

        print(f"token_mode={token_mode}: Special tokens are correct")

    print("PASSED: Special tokens handling is correct")


def run_all_tests():
    """Run all integration tests."""
    print("="*60)
    print("Integration Test: Data Flow from Preprocessed to Encoded")
    print("="*60)

    try:
        # Test 1: Load data
        messages = test_load_preprocessed_data()

        # Test 2: Encoding with token_mode=22
        test_encoding_token_mode_22(messages)

        # Test 3: Encoding with token_mode=24
        test_encoding_token_mode_24(messages)

        # Test 4: Batch processing
        test_batch_processing(messages)

        # Test 5: Consistency
        test_consistency_across_batches(messages)

        # Test 6: Special tokens
        test_special_tokens_handling()

        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
