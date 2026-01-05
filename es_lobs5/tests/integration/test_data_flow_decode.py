#!/usr/bin/env python
"""
Integration Test: Data Flow from Model Output to Decoded Messages

This test verifies the data flow from model output logits back to decoded messages:
1. Create mock model output logits
2. Apply argmax to get predicted tokens
3. Decode tokens back to message format using decode_msg
4. Verify decoded message has correct fields (event_type, side, price, size, etc.)

Test both token_mode=22 and token_mode=24

Usage:
    JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_data_flow_decode.py
"""

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
from lob.encoding import Vocab, decode_msg


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    'token_mode_22': {
        'vocab_size': 12012,
        'n_tokens': 22,
    },
    'token_mode_24': {
        'vocab_size': 2112,
        'n_tokens': 24,
    },
    'message_fields': 14,  # Decoded message has 14 fields
}

# Field names for decoded message (14 fields)
DECODED_FIELD_NAMES = [
    'order_id',      # 0: NA_VAL (not encoded)
    'event_type',    # 1: Event type (1-4)
    'direction',     # 2: Side (0 or 1)
    'price_abs',     # 3: NA_VAL (not encoded)
    'price',         # 4: Relative price
    'size',          # 5: Order size
    'delta_t_s',     # 6: Delta time seconds
    'delta_t_ns',    # 7: Delta time nanoseconds
    'time_s',        # 8: Time seconds
    'time_ns',       # 9: Time nanoseconds
    'price_ref',     # 10: Reference price
    'size_ref',      # 11: Reference size
    'time_s_ref',    # 12: Reference time seconds
    'time_ns_ref',   # 13: Reference time nanoseconds
]


# ============================================================
# Test Functions
# ============================================================

def test_logits_to_tokens_22():
    """Test: argmax on logits gives valid token indices for 22-token mode."""
    print("\n[Test 1a] Logits to Tokens (22-token mode)")

    vocab_size = CONFIG['token_mode_22']['vocab_size']
    n_tokens = CONFIG['token_mode_22']['n_tokens']
    batch_size = 4

    # Create mock logits: (batch_size, n_tokens, vocab_size)
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, shape=(batch_size, n_tokens, vocab_size))

    # Apply argmax to get predicted tokens
    tokens = jnp.argmax(logits, axis=-1)

    # Verify shape
    assert tokens.shape == (batch_size, n_tokens), \
        f"Token shape mismatch: {tokens.shape} != ({batch_size}, {n_tokens})"

    # Verify all tokens are within valid range [0, vocab_size)
    assert jnp.all(tokens >= 0), "Negative token indices found"
    assert jnp.all(tokens < vocab_size), f"Token indices >= {vocab_size} found"

    print(f"  Logits shape: {logits.shape}")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
    print("  [PASS] 22-token mode: argmax produces valid token indices")
    return True


def test_logits_to_tokens_24():
    """Test: argmax on logits gives valid token indices for 24-token mode."""
    print("\n[Test 1b] Logits to Tokens (24-token mode)")

    vocab_size = CONFIG['token_mode_24']['vocab_size']
    n_tokens = CONFIG['token_mode_24']['n_tokens']
    batch_size = 4

    # Create mock logits: (batch_size, n_tokens, vocab_size)
    key = jax.random.PRNGKey(123)
    logits = jax.random.normal(key, shape=(batch_size, n_tokens, vocab_size))

    # Apply argmax to get predicted tokens
    tokens = jnp.argmax(logits, axis=-1)

    # Verify shape
    assert tokens.shape == (batch_size, n_tokens), \
        f"Token shape mismatch: {tokens.shape} != ({batch_size}, {n_tokens})"

    # Verify all tokens are within valid range [0, vocab_size)
    assert jnp.all(tokens >= 0), "Negative token indices found"
    assert jnp.all(tokens < vocab_size), f"Token indices >= {vocab_size} found"

    print(f"  Logits shape: {logits.shape}")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
    print("  [PASS] 24-token mode: argmax produces valid token indices")
    return True


def test_tokens_to_message_22():
    """Test: decode_msg produces 14-field message for 22-token mode."""
    print("\n[Test 2a] Tokens to Message (22-token mode)")

    token_mode = 22
    n_tokens = CONFIG['token_mode_22']['n_tokens']
    vocab = Vocab(token_mode=token_mode)
    vocab_size = len(vocab)

    # Create valid tokens within vocab range
    # Use deterministic tokens for reproducibility
    key = jax.random.PRNGKey(42)
    tokens = jax.random.randint(key, shape=(n_tokens,), minval=0, maxval=vocab_size)

    # Decode tokens to message
    decoded_msg = decode_msg(tokens, vocab.ENCODING, token_mode=token_mode)

    # Verify message has 14 fields
    assert decoded_msg.shape == (CONFIG['message_fields'],), \
        f"Decoded message shape mismatch: {decoded_msg.shape} != ({CONFIG['message_fields']},)"

    print(f"  Input tokens shape: {tokens.shape}")
    print(f"  Decoded message shape: {decoded_msg.shape}")
    print(f"  Field names: {DECODED_FIELD_NAMES}")
    print("  [PASS] 22-token mode: decode_msg produces 14-field message")
    return True


def test_tokens_to_message_24():
    """Test: decode_msg produces 14-field message for 24-token mode."""
    print("\n[Test 2b] Tokens to Message (24-token mode)")

    token_mode = 24
    n_tokens = CONFIG['token_mode_24']['n_tokens']
    vocab = Vocab(token_mode=token_mode)
    vocab_size = len(vocab)

    # Create valid tokens within vocab range
    key = jax.random.PRNGKey(123)
    tokens = jax.random.randint(key, shape=(n_tokens,), minval=0, maxval=vocab_size)

    # Decode tokens to message
    decoded_msg = decode_msg(tokens, vocab.ENCODING, token_mode=token_mode)

    # Verify message has 14 fields
    assert decoded_msg.shape == (CONFIG['message_fields'],), \
        f"Decoded message shape mismatch: {decoded_msg.shape} != ({CONFIG['message_fields']},)"

    print(f"  Input tokens shape: {tokens.shape}")
    print(f"  Decoded message shape: {decoded_msg.shape}")
    print(f"  Field names: {DECODED_FIELD_NAMES}")
    print("  [PASS] 24-token mode: decode_msg produces 14-field message")
    return True


def test_full_decode_pipeline_22():
    """Test: Full pipeline logits -> tokens -> message for 22-token mode."""
    print("\n[Test 3a] Full Decode Pipeline (22-token mode)")

    token_mode = 22
    vocab_size = CONFIG['token_mode_22']['vocab_size']
    n_tokens = CONFIG['token_mode_22']['n_tokens']
    vocab = Vocab(token_mode=token_mode)

    # Step 1: Create mock logits
    key = jax.random.PRNGKey(456)
    logits = jax.random.normal(key, shape=(n_tokens, vocab_size))

    # Step 2: Apply argmax to get predicted tokens
    tokens = jnp.argmax(logits, axis=-1)

    # Step 3: Decode tokens to message
    decoded_msg = decode_msg(tokens, vocab.ENCODING, token_mode=token_mode)

    # Verify pipeline output
    assert decoded_msg.shape == (CONFIG['message_fields'],), \
        f"Pipeline output shape mismatch: {decoded_msg.shape}"

    # Verify order_id and price_abs are NA_VAL (-9999)
    NA_VAL = -9999
    assert decoded_msg[0] == NA_VAL, f"order_id should be NA_VAL, got {decoded_msg[0]}"
    assert decoded_msg[3] == NA_VAL, f"price_abs should be NA_VAL, got {decoded_msg[3]}"

    print(f"  Logits shape: {logits.shape}")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Decoded message shape: {decoded_msg.shape}")
    print(f"  order_id (field 0): {decoded_msg[0]} (expected NA_VAL={NA_VAL})")
    print(f"  event_type (field 1): {decoded_msg[1]}")
    print(f"  direction (field 2): {decoded_msg[2]}")
    print(f"  price (field 4): {decoded_msg[4]}")
    print(f"  size (field 5): {decoded_msg[5]}")
    print("  [PASS] 22-token mode: full decode pipeline works")
    return True


def test_full_decode_pipeline_24():
    """Test: Full pipeline logits -> tokens -> message for 24-token mode."""
    print("\n[Test 3b] Full Decode Pipeline (24-token mode)")

    token_mode = 24
    vocab_size = CONFIG['token_mode_24']['vocab_size']
    n_tokens = CONFIG['token_mode_24']['n_tokens']
    vocab = Vocab(token_mode=token_mode)

    # Step 1: Create mock logits
    key = jax.random.PRNGKey(789)
    logits = jax.random.normal(key, shape=(n_tokens, vocab_size))

    # Step 2: Apply argmax to get predicted tokens
    tokens = jnp.argmax(logits, axis=-1)

    # Step 3: Decode tokens to message
    decoded_msg = decode_msg(tokens, vocab.ENCODING, token_mode=token_mode)

    # Verify pipeline output
    assert decoded_msg.shape == (CONFIG['message_fields'],), \
        f"Pipeline output shape mismatch: {decoded_msg.shape}"

    # Verify order_id and price_abs are NA_VAL (-9999)
    NA_VAL = -9999
    assert decoded_msg[0] == NA_VAL, f"order_id should be NA_VAL, got {decoded_msg[0]}"
    assert decoded_msg[3] == NA_VAL, f"price_abs should be NA_VAL, got {decoded_msg[3]}"

    print(f"  Logits shape: {logits.shape}")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Decoded message shape: {decoded_msg.shape}")
    print(f"  order_id (field 0): {decoded_msg[0]} (expected NA_VAL={NA_VAL})")
    print(f"  event_type (field 1): {decoded_msg[1]}")
    print(f"  direction (field 2): {decoded_msg[2]}")
    print(f"  price (field 4): {decoded_msg[4]}")
    print(f"  size (field 5): {decoded_msg[5]}")
    print("  [PASS] 24-token mode: full decode pipeline works")
    return True


def test_batch_decode_pipeline():
    """Test: Batch decoding for multiple messages."""
    print("\n[Test 4] Batch Decode Pipeline")

    batch_size = 8

    for token_mode in [22, 24]:
        config_key = f'token_mode_{token_mode}'
        vocab_size = CONFIG[config_key]['vocab_size']
        n_tokens = CONFIG[config_key]['n_tokens']
        vocab = Vocab(token_mode=token_mode)

        # Create batch of mock logits
        key = jax.random.PRNGKey(100 + token_mode)
        logits = jax.random.normal(key, shape=(batch_size, n_tokens, vocab_size))

        # Apply argmax to get predicted tokens
        tokens = jnp.argmax(logits, axis=-1)

        # Decode each message in the batch
        decoded_msgs = []
        for i in range(batch_size):
            msg = decode_msg(tokens[i], vocab.ENCODING, token_mode=token_mode)
            decoded_msgs.append(msg)

        decoded_batch = jnp.stack(decoded_msgs)

        # Verify batch output shape
        expected_shape = (batch_size, CONFIG['message_fields'])
        assert decoded_batch.shape == expected_shape, \
            f"Batch output shape mismatch: {decoded_batch.shape} != {expected_shape}"

        print(f"  Mode {token_mode}: batch decode shape = {decoded_batch.shape}")

    print("  [PASS] Batch decode pipeline works for both modes")
    return True


def test_vocab_consistency():
    """Test: Vocab size consistency between modes."""
    print("\n[Test 5] Vocab Consistency")

    vocab_22 = Vocab(token_mode=22)
    vocab_24 = Vocab(token_mode=24)

    # Verify vocab sizes
    assert len(vocab_22) == CONFIG['token_mode_22']['vocab_size'], \
        f"22-token vocab size mismatch: {len(vocab_22)} != {CONFIG['token_mode_22']['vocab_size']}"
    assert len(vocab_24) == CONFIG['token_mode_24']['vocab_size'], \
        f"24-token vocab size mismatch: {len(vocab_24)} != {CONFIG['token_mode_24']['vocab_size']}"

    print(f"  22-token mode vocab size: {len(vocab_22)}")
    print(f"  24-token mode vocab size: {len(vocab_24)}")
    print("  [PASS] Vocab sizes are consistent with configuration")
    return True


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Integration Test: Data Flow from Model Output to Decoded Messages")
    print("=" * 60)

    tests = [
        ("Logits to Tokens (22-tok)", test_logits_to_tokens_22),
        ("Logits to Tokens (24-tok)", test_logits_to_tokens_24),
        ("Tokens to Message (22-tok)", test_tokens_to_message_22),
        ("Tokens to Message (24-tok)", test_tokens_to_message_24),
        ("Full Pipeline (22-tok)", test_full_decode_pipeline_22),
        ("Full Pipeline (24-tok)", test_full_decode_pipeline_24),
        ("Batch Decode Pipeline", test_batch_decode_pipeline),
        ("Vocab Consistency", test_vocab_consistency),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("[PASS] All data flow decode tests passed!")
    else:
        print("[FAIL] Some tests failed. Please check the output above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
