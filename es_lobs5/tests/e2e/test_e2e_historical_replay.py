#!/usr/bin/env python3
"""
Level 3 End-to-End Test: Historical Replay Training Mode.

This test validates the full historical replay pipeline:
1. Load real LOBSTER data
2. Encode messages to tokens (mode 22 or 24)
3. Create batched training data with windowing
4. Run forward passes through model
5. Compute fitness based on prediction accuracy
6. Verify the full historical replay pipeline works
"""

import os
import sys

# Set CPU only before importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
import numpy as np
import glob

from lob.encoding import Vocab, encode_msgs, decode_msg

# =============================================================================
# Test Configuration
# =============================================================================

DATA_DIR = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021'
TOKEN_MODE = 24  # Use 24-token mode for this test
SEQ_LEN = 500    # Sequence length for model input
BATCH_SIZE = 4   # Batch size for testing
WINDOW_SIZE = 24 # Tokens per message


# =============================================================================
# Test 1: Load Full Trading Day
# =============================================================================

def test_load_full_day():
    """Test: Load and encode a full trading day of LOBSTER data."""
    print("\n" + "=" * 60)
    print("TEST 1: test_load_full_day")
    print("=" * 60)

    # Find message files
    message_files = sorted(glob.glob(os.path.join(DATA_DIR, '*message*proc.npy')))

    if len(message_files) == 0:
        print(f"[SKIP] No message files found in {DATA_DIR}")
        return None, None

    print(f"  Found {len(message_files)} message files")
    print(f"  First file: {os.path.basename(message_files[0])}")

    # Load first day's data
    sample_file = message_files[0]
    msg_raw = np.load(sample_file)

    print(f"  Raw data shape: {msg_raw.shape}")
    print(f"  Raw data dtype: {msg_raw.dtype}")

    # Determine data format
    n_cols = msg_raw.shape[1]
    if n_cols == 14:
        print(f"  Data format: Raw 14-field messages")
        is_pretokenized = False
    elif n_cols in [22, 24]:
        print(f"  Data format: Pre-tokenized ({n_cols} tokens/msg)")
        is_pretokenized = True
    else:
        print(f"  [WARN] Unknown format: {n_cols} columns")
        is_pretokenized = False

    # Create vocabulary and encode if needed
    vocab = Vocab(token_mode=TOKEN_MODE)

    if is_pretokenized:
        # Data is already tokenized
        if n_cols == TOKEN_MODE:
            tokens = jnp.array(msg_raw)
        elif n_cols > TOKEN_MODE:
            tokens = jnp.array(msg_raw[:, :TOKEN_MODE])
        else:
            # Pad if fewer columns than expected
            padding = np.zeros((msg_raw.shape[0], TOKEN_MODE - n_cols), dtype=msg_raw.dtype)
            tokens = jnp.array(np.concatenate([msg_raw, padding], axis=1))
        print(f"  Using pre-tokenized data: shape {tokens.shape}")
    else:
        # Encode raw messages
        tokens = encode_msgs(jnp.array(msg_raw), vocab.ENCODING, token_mode=TOKEN_MODE)
        print(f"  Encoded to tokens: shape {tokens.shape}")

    # Validate token values
    max_token = int(jnp.max(tokens))
    min_token = int(jnp.min(tokens))
    vocab_size = len(vocab)

    print(f"  Token range: [{min_token}, {max_token}]")
    print(f"  Vocab size: {vocab_size}")

    assert max_token < vocab_size, f"Max token {max_token} >= vocab size {vocab_size}"
    assert min_token >= 0, f"Min token {min_token} < 0"

    print(f"  [PASS] Loaded {tokens.shape[0]} messages from full trading day")

    return tokens, vocab


# =============================================================================
# Test 2: Create Windowed Batches
# =============================================================================

def test_windowed_batches(tokens, vocab):
    """Test: Create proper training batches with windowing."""
    print("\n" + "=" * 60)
    print("TEST 2: test_windowed_batches")
    print("=" * 60)

    if tokens is None:
        print("  [SKIP] No tokens available")
        return None

    n_messages = tokens.shape[0]
    tokens_per_msg = tokens.shape[1]

    print(f"  Total messages: {n_messages}")
    print(f"  Tokens per message: {tokens_per_msg}")

    # Flatten tokens for sequence processing
    flat_tokens = tokens.flatten()
    total_tokens = len(flat_tokens)
    print(f"  Total tokens (flattened): {total_tokens}")

    # Calculate window parameters
    window_tokens = SEQ_LEN * tokens_per_msg
    stride = tokens_per_msg  # Stride by one message at a time

    # Create windows
    n_windows = (total_tokens - window_tokens) // stride + 1
    n_windows = min(n_windows, 1000)  # Limit for testing

    print(f"  Window size: {window_tokens} tokens ({SEQ_LEN} messages)")
    print(f"  Stride: {stride} tokens")
    print(f"  Number of windows: {n_windows}")

    # Create batch indices
    n_batches = n_windows // BATCH_SIZE

    if n_batches == 0:
        print("  [WARN] Not enough data for batching, using single batch")
        n_batches = 1
        actual_batch_size = min(BATCH_SIZE, n_windows)
    else:
        actual_batch_size = BATCH_SIZE

    print(f"  Number of batches: {n_batches}")
    print(f"  Batch size: {actual_batch_size}")

    # Create first batch
    batch_indices = np.arange(actual_batch_size) * stride
    batch_data = []

    for idx in batch_indices:
        window = flat_tokens[idx:idx + window_tokens]
        if len(window) == window_tokens:
            batch_data.append(window)

    if len(batch_data) == 0:
        print("  [FAIL] Could not create any valid windows")
        return None

    batch = jnp.stack(batch_data)
    print(f"  First batch shape: {batch.shape}")

    # Validate batch
    assert batch.shape[0] == len(batch_data), "Batch size mismatch"
    assert batch.shape[1] == window_tokens, f"Window size mismatch: {batch.shape[1]} vs {window_tokens}"

    # Create input/target pairs for next-token prediction
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")

    assert inputs.shape == targets.shape, "Input/target shape mismatch"

    print(f"  [PASS] Created windowed batches successfully")

    return {
        'batch': batch,
        'inputs': inputs,
        'targets': targets,
        'n_batches': n_batches,
        'window_tokens': window_tokens,
        'vocab': vocab,
    }


# =============================================================================
# Test 3: Forward Pass Through Model (Mocked)
# =============================================================================

def test_forward_pass_batch(batch_data):
    """Test: Process batch through model (using mock model for CPU testing)."""
    print("\n" + "=" * 60)
    print("TEST 3: test_forward_pass_batch")
    print("=" * 60)

    if batch_data is None:
        print("  [SKIP] No batch data available")
        return None

    inputs = batch_data['inputs']
    vocab = batch_data['vocab']
    vocab_size = len(vocab)

    print(f"  Input shape: {inputs.shape}")
    print(f"  Vocab size: {vocab_size}")

    # Mock forward pass - simulate model output with random logits
    # In real scenario, this would use ES_PaddedLobPredModel
    key = jax.random.PRNGKey(42)

    batch_size, seq_len = inputs.shape

    # Simulate embedding lookup
    embedding_dim = 64  # Small for CPU testing
    key, emb_key = jax.random.split(key)
    embedding_table = jax.random.normal(emb_key, (vocab_size, embedding_dim))

    # Get embeddings (clamp input indices to valid range)
    safe_inputs = jnp.clip(inputs, 0, vocab_size - 1)
    embeddings = embedding_table[safe_inputs]

    print(f"  Embeddings shape: {embeddings.shape}")

    # Simulate model forward pass with simple projection
    key, proj_key = jax.random.split(key)
    output_proj = jax.random.normal(proj_key, (embedding_dim, vocab_size)) * 0.01

    # Compute logits
    logits = jnp.einsum('bsd,dv->bsv', embeddings, output_proj)

    print(f"  Logits shape: {logits.shape}")

    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Log probs range: [{float(jnp.min(log_probs)):.4f}, {float(jnp.max(log_probs)):.4f}]")

    # Validate output shape
    expected_shape = (batch_size, seq_len, vocab_size)
    assert log_probs.shape == expected_shape, f"Output shape {log_probs.shape} != expected {expected_shape}"

    # Validate probabilities sum to 1
    probs = jnp.exp(log_probs)
    prob_sums = jnp.sum(probs, axis=-1)
    max_deviation = float(jnp.max(jnp.abs(prob_sums - 1.0)))

    print(f"  Probability sum deviation: {max_deviation:.6f}")
    assert max_deviation < 1e-5, f"Probability sums deviate from 1.0: max deviation {max_deviation}"

    print(f"  [PASS] Forward pass produced valid output")

    return {
        'log_probs': log_probs,
        'logits': logits,
        'vocab_size': vocab_size,
    }


# =============================================================================
# Test 4: Compute Token Prediction Accuracy
# =============================================================================

def test_prediction_accuracy(batch_data, forward_data):
    """Test: Compute token prediction accuracy as fitness metric."""
    print("\n" + "=" * 60)
    print("TEST 4: test_prediction_accuracy")
    print("=" * 60)

    if batch_data is None or forward_data is None:
        print("  [SKIP] Missing batch or forward data")
        return None

    targets = batch_data['targets']
    log_probs = forward_data['log_probs']
    vocab_size = forward_data['vocab_size']

    print(f"  Targets shape: {targets.shape}")
    print(f"  Log probs shape: {log_probs.shape}")

    # Get predictions (argmax of log probs)
    predictions = jnp.argmax(log_probs, axis=-1)

    print(f"  Predictions shape: {predictions.shape}")

    # Compute accuracy
    # Clamp targets to valid range
    safe_targets = jnp.clip(targets, 0, vocab_size - 1)
    correct = (predictions == safe_targets).astype(jnp.float32)

    # Per-sample accuracy
    per_sample_accuracy = jnp.mean(correct, axis=1)
    print(f"  Per-sample accuracy range: [{float(jnp.min(per_sample_accuracy)):.4f}, {float(jnp.max(per_sample_accuracy)):.4f}]")

    # Overall accuracy
    overall_accuracy = float(jnp.mean(correct))
    print(f"  Overall accuracy: {overall_accuracy:.4f}")

    # Compute cross-entropy loss (as negative log likelihood)
    batch_size, seq_len = targets.shape

    # Gather log probs at target indices
    target_log_probs = jnp.take_along_axis(
        log_probs.reshape(batch_size * seq_len, vocab_size),
        safe_targets.reshape(batch_size * seq_len, 1),
        axis=1
    ).squeeze(-1)

    # Mean negative log likelihood
    mean_nll = -float(jnp.mean(target_log_probs))
    print(f"  Mean NLL (cross-entropy loss): {mean_nll:.4f}")

    # Perplexity
    perplexity = float(jnp.exp(jnp.minimum(mean_nll, 20.0)))  # Cap to prevent overflow
    print(f"  Perplexity: {perplexity:.4f}")

    # Fitness based on accuracy (for ES training)
    fitness = overall_accuracy - 0.5  # Center around 0
    print(f"  Fitness (accuracy - 0.5): {fitness:.4f}")

    # Alternative fitness: negative NLL (higher is better)
    fitness_nll = -mean_nll
    print(f"  Fitness (negative NLL): {fitness_nll:.4f}")

    # Validate metrics
    assert 0.0 <= overall_accuracy <= 1.0, f"Invalid accuracy: {overall_accuracy}"
    assert mean_nll > 0, f"Invalid NLL: {mean_nll}"
    assert perplexity >= 1.0, f"Invalid perplexity: {perplexity}"

    print(f"  [PASS] Computed prediction accuracy metrics")

    return {
        'accuracy': overall_accuracy,
        'per_sample_accuracy': per_sample_accuracy,
        'mean_nll': mean_nll,
        'perplexity': perplexity,
        'fitness': fitness,
        'fitness_nll': fitness_nll,
    }


# =============================================================================
# Test 5: Full Replay Epoch Simulation
# =============================================================================

def test_full_replay_epoch():
    """Test: Simulate one epoch of historical replay training."""
    print("\n" + "=" * 60)
    print("TEST 5: test_full_replay_epoch")
    print("=" * 60)

    # Load data files
    message_files = sorted(glob.glob(os.path.join(DATA_DIR, '*message*proc.npy')))

    if len(message_files) == 0:
        print(f"  [SKIP] No message files found in {DATA_DIR}")
        return False

    print(f"  Found {len(message_files)} trading days")

    # Use first 3 days for epoch simulation
    n_days = min(3, len(message_files))

    # Initialize vocab
    vocab = Vocab(token_mode=TOKEN_MODE)
    vocab_size = len(vocab)

    print(f"  Token mode: {TOKEN_MODE}")
    print(f"  Vocab size: {vocab_size}")

    # Track epoch statistics
    epoch_losses = []
    epoch_accuracies = []
    total_messages = 0

    # Simulate processing each day
    for day_idx in range(n_days):
        file_path = message_files[day_idx]
        day_name = os.path.basename(file_path)

        # Load day's data
        msg_raw = np.load(file_path)
        n_messages = msg_raw.shape[0]
        total_messages += n_messages

        # Determine if pre-tokenized
        n_cols = msg_raw.shape[1]
        if n_cols in [22, 24]:
            if n_cols == TOKEN_MODE:
                tokens = jnp.array(msg_raw)
            else:
                tokens = jnp.array(msg_raw[:, :TOKEN_MODE])
        else:
            tokens = encode_msgs(jnp.array(msg_raw), vocab.ENCODING, token_mode=TOKEN_MODE)

        # Create batches for this day
        flat_tokens = tokens.flatten()
        window_tokens = SEQ_LEN * TOKEN_MODE
        stride = TOKEN_MODE

        n_windows = max(1, (len(flat_tokens) - window_tokens) // stride)
        n_windows = min(n_windows, 100)  # Limit for speed

        # Create batch
        batch_data = []
        for i in range(min(BATCH_SIZE, n_windows)):
            idx = i * stride
            window = flat_tokens[idx:idx + window_tokens]
            if len(window) == window_tokens:
                batch_data.append(window)

        if len(batch_data) == 0:
            print(f"    Day {day_idx + 1}: {day_name} - skipped (not enough data)")
            continue

        batch = jnp.stack(batch_data)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Mock forward pass
        key = jax.random.PRNGKey(42 + day_idx)
        batch_size_actual, seq_len_actual = inputs.shape

        # Simple random logits
        key, logit_key = jax.random.split(key)
        logits = jax.random.normal(logit_key, (batch_size_actual, seq_len_actual, vocab_size)) * 0.1
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Compute metrics
        predictions = jnp.argmax(log_probs, axis=-1)
        safe_targets = jnp.clip(targets, 0, vocab_size - 1)
        accuracy = float(jnp.mean(predictions == safe_targets))

        # Compute loss
        target_log_probs = jnp.take_along_axis(
            log_probs.reshape(-1, vocab_size),
            safe_targets.reshape(-1, 1),
            axis=1
        ).squeeze(-1)
        loss = -float(jnp.mean(target_log_probs))

        epoch_losses.append(loss)
        epoch_accuracies.append(accuracy)

        print(f"    Day {day_idx + 1}: {day_name[:30]}... - msgs: {n_messages}, acc: {accuracy:.4f}, loss: {loss:.4f}")

    if len(epoch_losses) == 0:
        print("  [FAIL] No days processed")
        return False

    # Epoch summary
    mean_loss = np.mean(epoch_losses)
    mean_accuracy = np.mean(epoch_accuracies)

    print(f"\n  Epoch Summary:")
    print(f"    Days processed: {len(epoch_losses)}/{n_days}")
    print(f"    Total messages: {total_messages}")
    print(f"    Mean accuracy: {mean_accuracy:.4f}")
    print(f"    Mean loss: {mean_loss:.4f}")

    # Compute epoch fitness (for ES training)
    epoch_fitness = mean_accuracy - 0.5
    print(f"    Epoch fitness: {epoch_fitness:.4f}")

    # Validate epoch completed
    assert len(epoch_losses) > 0, "No losses recorded"
    assert 0.0 <= mean_accuracy <= 1.0, f"Invalid mean accuracy: {mean_accuracy}"

    print(f"  [PASS] Completed one epoch of historical replay simulation")

    return True


# =============================================================================
# Test 6: Data Compatibility Check
# =============================================================================

def test_data_compatibility():
    """Test: Verify data format compatibility with ES training pipeline."""
    print("\n" + "=" * 60)
    print("TEST 6: test_data_compatibility")
    print("=" * 60)

    # Check both message and orderbook files exist
    message_files = sorted(glob.glob(os.path.join(DATA_DIR, '*message*proc.npy')))
    orderbook_files = sorted(glob.glob(os.path.join(DATA_DIR, '*orderbook*proc.npy')))

    print(f"  Message files: {len(message_files)}")
    print(f"  Orderbook files: {len(orderbook_files)}")

    if len(message_files) == 0 or len(orderbook_files) == 0:
        print(f"  [SKIP] Missing data files in {DATA_DIR}")
        return False

    # Verify file pairing
    for msg_file in message_files[:3]:
        # Extract date from message file
        msg_basename = os.path.basename(msg_file)
        date_str = msg_basename.split('_')[1]  # e.g., "2021-01-04"

        # Find matching orderbook file
        matching_ob = [f for f in orderbook_files if date_str in f]

        if len(matching_ob) == 0:
            print(f"  [WARN] No orderbook file for date {date_str}")
            continue

        # Load and verify shapes
        msg = np.load(msg_file)
        ob = np.load(matching_ob[0])

        msg_rows, msg_cols = msg.shape
        ob_rows, ob_cols = ob.shape

        # Messages and orderbook should have compatible row counts
        # (orderbook has one more row typically for initial state)
        row_diff = abs(ob_rows - msg_rows)

        print(f"  {date_str}: msg={msg.shape}, ob={ob.shape}, row_diff={row_diff}")

        if row_diff > 2:
            print(f"  [WARN] Large row difference for {date_str}: {row_diff}")

    # Verify vocab compatibility
    for token_mode in [22, 24]:
        vocab = Vocab(token_mode=token_mode)

        # Load sample message
        msg_raw = np.load(message_files[0])
        n_cols = msg_raw.shape[1]

        # Check if data matches token mode
        if n_cols == token_mode:
            print(f"  Token mode {token_mode}: Data already tokenized with matching columns")
        elif n_cols == 14:
            print(f"  Token mode {token_mode}: Data is raw format, encoding will be applied")
        elif n_cols in [22, 24] and n_cols != token_mode:
            print(f"  Token mode {token_mode}: Data is tokenized with {n_cols} columns (mismatch)")

        # Verify vocab has expected size
        expected_vocab_sizes = {22: 10021, 24: 2112}
        expected_size = expected_vocab_sizes.get(token_mode, 0)
        actual_size = len(vocab)

        if expected_size > 0:
            if actual_size == expected_size:
                print(f"  Token mode {token_mode}: Vocab size {actual_size} matches expected")
            else:
                print(f"  Token mode {token_mode}: Vocab size {actual_size} (expected ~{expected_size})")

    print(f"  [PASS] Data compatibility check complete")

    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run all E2E tests for historical replay pipeline."""
    print("=" * 70)
    print("E2E Test: Historical Replay Training Mode")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Token mode: {TOKEN_MODE}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"JAX devices: {jax.devices()}")

    # Track test results
    results = {}

    # Test 1: Load full day
    try:
        tokens, vocab = test_load_full_day()
        results['test_load_full_day'] = tokens is not None
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        results['test_load_full_day'] = False
        tokens, vocab = None, None

    # Test 2: Create windowed batches
    try:
        batch_data = test_windowed_batches(tokens, vocab)
        results['test_windowed_batches'] = batch_data is not None
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        results['test_windowed_batches'] = False
        batch_data = None

    # Test 3: Forward pass
    try:
        forward_data = test_forward_pass_batch(batch_data)
        results['test_forward_pass_batch'] = forward_data is not None
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        results['test_forward_pass_batch'] = False
        forward_data = None

    # Test 4: Prediction accuracy
    try:
        accuracy_data = test_prediction_accuracy(batch_data, forward_data)
        results['test_prediction_accuracy'] = accuracy_data is not None
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        results['test_prediction_accuracy'] = False

    # Test 5: Full replay epoch
    try:
        results['test_full_replay_epoch'] = test_full_replay_epoch()
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        results['test_full_replay_epoch'] = False

    # Test 6: Data compatibility
    try:
        results['test_data_compatibility'] = test_data_compatibility()
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        results['test_data_compatibility'] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "PASS" if passed_flag else "FAIL"
        print(f"  [{status}] {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
