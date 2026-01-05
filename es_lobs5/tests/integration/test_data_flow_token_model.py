#!/usr/bin/env python
"""
Integration Test: Data Flow from Token Encoding to Model Input

This test verifies the data flow from token encoding to model input format,
ensuring tokens are correctly processed through embedding layers.

Usage:
    JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_data_flow_token_model.py
"""

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
from lob.encoding import Vocab


# ============================================================
# Constants
# ============================================================

VOCAB_SIZE_22 = 12012  # 22-token mode vocabulary size
VOCAB_SIZE_24 = 2112   # 24-token mode vocabulary size
TOKEN_MODE_22 = 22
TOKEN_MODE_24 = 24


# ============================================================
# Mock Model Components for Testing
# ============================================================

class MockEmbeddingLayer:
    """
    Mock embedding layer to test token-to-embedding flow.

    Simulates the embedding lookup that occurs in ES_PaddedLobPredModel.
    """

    def __init__(self, vocab_size: int, d_model: int, key: jax.Array):
        """
        Initialize embedding layer.

        Args:
            vocab_size: Size of vocabulary (12012 for 22-tok, 2112 for 24-tok)
            d_model: Model hidden dimension
            key: JAX random key for initialization
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize embedding matrix with small random values (GPT-style)
        self.embedding = jax.random.normal(key, (vocab_size, d_model)) * 0.02

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """
        Look up embeddings for tokens.

        Args:
            tokens: Token indices, shape (seq_len,) or (batch, seq_len)

        Returns:
            Embedded tokens, shape (seq_len, d_model) or (batch, seq_len, d_model)
        """
        return self.embedding[tokens]


class MockLOBModel:
    """
    Mock LOB model for testing data flow.

    Simplified model that tests:
    1. Token input -> Embedding lookup
    2. Embedding -> Sequence processing
    3. Sequence -> Output projection
    """

    def __init__(self, vocab_size: int, d_model: int, key: jax.Array):
        """
        Initialize mock model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model hidden dimension
            key: JAX random key
        """
        keys = jax.random.split(key, 3)

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layer
        self.embedding_layer = MockEmbeddingLayer(vocab_size, d_model, keys[0])

        # Simple linear layer to simulate sequence processing
        self.hidden_weight = jax.random.normal(keys[1], (d_model, d_model)) * 0.02
        self.hidden_bias = jnp.zeros(d_model)

        # Output projection
        self.output_weight = jax.random.normal(keys[2], (d_model, vocab_size)) * 0.02
        self.output_bias = jnp.zeros(vocab_size)

    def forward(self, tokens: jax.Array) -> jax.Array:
        """
        Forward pass through mock model.

        Args:
            tokens: Token indices, shape (seq_len,) or (batch, seq_len)

        Returns:
            Log probabilities, shape (vocab_size,) or (batch, vocab_size)
        """
        # Embedding lookup
        x = self.embedding_layer(tokens)

        # Simple hidden layer (simulate sequence processing)
        x = jnp.dot(x, self.hidden_weight) + self.hidden_bias
        x = jax.nn.gelu(x)

        # Mean pooling
        if x.ndim == 2:
            x = jnp.mean(x, axis=0)  # (d_model,)
        else:
            x = jnp.mean(x, axis=1)  # (batch, d_model)

        # Output projection
        logits = jnp.dot(x, self.output_weight) + self.output_bias

        return jax.nn.log_softmax(logits, axis=-1)


# ============================================================
# Test Functions
# ============================================================

def test_token_sequence_format_22():
    """Test: Create token sequences in correct format for 22-token mode."""
    print("\n[Test 1] Token Sequence Format (22-token mode)")

    vocab = Vocab(token_mode=TOKEN_MODE_22)

    # Verify vocab size
    if len(vocab) != VOCAB_SIZE_22:
        print(f"  FAIL: Expected vocab size {VOCAB_SIZE_22}, got {len(vocab)}")
        return False
    print(f"  Vocab size: {len(vocab)}")

    # Create sample token sequence (simulating encoded messages)
    # Shape should be (seq_len, token_mode) for multiple messages
    # or (token_mode,) for single message
    seq_len = 100  # Number of messages

    # Create random tokens within valid range
    key = jax.random.PRNGKey(42)
    tokens = jax.random.randint(key, (seq_len, TOKEN_MODE_22), 0, VOCAB_SIZE_22)

    if tokens.shape != (seq_len, TOKEN_MODE_22):
        print(f"  FAIL: Expected shape ({seq_len}, {TOKEN_MODE_22}), got {tokens.shape}")
        return False
    print(f"  Token sequence shape: {tokens.shape}")

    # Flatten for model input (typical format)
    flat_tokens = tokens.ravel()
    if flat_tokens.shape != (seq_len * TOKEN_MODE_22,):
        print(f"  FAIL: Flattened shape mismatch")
        return False
    print(f"  Flattened token shape: {flat_tokens.shape}")

    print("  PASS: Token sequence format is correct for 22-token mode")
    return True


def test_token_sequence_format_24():
    """Test: Create token sequences in correct format for 24-token mode."""
    print("\n[Test 2] Token Sequence Format (24-token mode)")

    vocab = Vocab(token_mode=TOKEN_MODE_24)

    # Verify vocab size
    if len(vocab) != VOCAB_SIZE_24:
        print(f"  FAIL: Expected vocab size {VOCAB_SIZE_24}, got {len(vocab)}")
        return False
    print(f"  Vocab size: {len(vocab)}")

    # Create sample token sequence
    seq_len = 100
    key = jax.random.PRNGKey(42)
    tokens = jax.random.randint(key, (seq_len, TOKEN_MODE_24), 0, VOCAB_SIZE_24)

    if tokens.shape != (seq_len, TOKEN_MODE_24):
        print(f"  FAIL: Expected shape ({seq_len}, {TOKEN_MODE_24}), got {tokens.shape}")
        return False
    print(f"  Token sequence shape: {tokens.shape}")

    # Flatten for model input
    flat_tokens = tokens.ravel()
    if flat_tokens.shape != (seq_len * TOKEN_MODE_24,):
        print(f"  FAIL: Flattened shape mismatch")
        return False
    print(f"  Flattened token shape: {flat_tokens.shape}")

    print("  PASS: Token sequence format is correct for 24-token mode")
    return True


def test_model_accepts_tokenized_input_22():
    """Test: Verify model can accept tokenized input for 22-token mode."""
    print("\n[Test 3] Model Accepts Tokenized Input (22-token mode)")

    key = jax.random.PRNGKey(123)
    d_model = 64
    seq_len = 50

    # Create mock model
    model = MockLOBModel(VOCAB_SIZE_22, d_model, key)

    # Create token input
    keys = jax.random.split(key, 2)
    tokens = jax.random.randint(keys[0], (seq_len * TOKEN_MODE_22,), 0, VOCAB_SIZE_22)

    print(f"  Input tokens shape: {tokens.shape}")
    print(f"  Input tokens dtype: {tokens.dtype}")
    print(f"  Token value range: [{tokens.min()}, {tokens.max()}]")

    # Forward pass
    try:
        log_probs = model.forward(tokens)
        print(f"  Output shape: {log_probs.shape}")

        if log_probs.shape != (VOCAB_SIZE_22,):
            print(f"  FAIL: Expected output shape ({VOCAB_SIZE_22},), got {log_probs.shape}")
            return False

        if jnp.any(jnp.isnan(log_probs)):
            print("  FAIL: NaN values in output")
            return False

        print("  PASS: Model accepts tokenized input for 22-token mode")
        return True

    except Exception as e:
        print(f"  FAIL: Exception during forward pass: {e}")
        return False


def test_model_accepts_tokenized_input_24():
    """Test: Verify model can accept tokenized input for 24-token mode."""
    print("\n[Test 4] Model Accepts Tokenized Input (24-token mode)")

    key = jax.random.PRNGKey(456)
    d_model = 64
    seq_len = 50

    # Create mock model
    model = MockLOBModel(VOCAB_SIZE_24, d_model, key)

    # Create token input
    keys = jax.random.split(key, 2)
    tokens = jax.random.randint(keys[0], (seq_len * TOKEN_MODE_24,), 0, VOCAB_SIZE_24)

    print(f"  Input tokens shape: {tokens.shape}")
    print(f"  Input tokens dtype: {tokens.dtype}")
    print(f"  Token value range: [{tokens.min()}, {tokens.max()}]")

    # Forward pass
    try:
        log_probs = model.forward(tokens)
        print(f"  Output shape: {log_probs.shape}")

        if log_probs.shape != (VOCAB_SIZE_24,):
            print(f"  FAIL: Expected output shape ({VOCAB_SIZE_24},), got {log_probs.shape}")
            return False

        if jnp.any(jnp.isnan(log_probs)):
            print("  FAIL: NaN values in output")
            return False

        print("  PASS: Model accepts tokenized input for 24-token mode")
        return True

    except Exception as e:
        print(f"  FAIL: Exception during forward pass: {e}")
        return False


def test_embedding_lookup_22():
    """Test: Verify embedding lookup works correctly for 22-token mode."""
    print("\n[Test 5] Embedding Lookup (22-token mode)")

    key = jax.random.PRNGKey(789)
    d_model = 128

    # Create embedding layer
    embedding = MockEmbeddingLayer(VOCAB_SIZE_22, d_model, key)

    # Test single token
    single_token = jnp.array(100)  # Token index 100
    single_embedding = embedding(single_token)

    if single_embedding.shape != (d_model,):
        print(f"  FAIL: Single token embedding shape {single_embedding.shape}, expected ({d_model},)")
        return False
    print(f"  Single token embedding shape: {single_embedding.shape}")

    # Test sequence of tokens
    seq_len = 22  # One message worth of tokens
    tokens = jax.random.randint(key, (seq_len,), 0, VOCAB_SIZE_22)
    seq_embedding = embedding(tokens)

    if seq_embedding.shape != (seq_len, d_model):
        print(f"  FAIL: Sequence embedding shape {seq_embedding.shape}, expected ({seq_len}, {d_model})")
        return False
    print(f"  Sequence embedding shape: {seq_embedding.shape}")

    # Verify lookup consistency
    for i in range(min(5, seq_len)):
        individual = embedding(tokens[i])
        if not jnp.allclose(individual, seq_embedding[i]):
            print(f"  FAIL: Embedding lookup inconsistent at position {i}")
            return False

    print("  PASS: Embedding lookup works correctly for 22-token mode")
    return True


def test_embedding_lookup_24():
    """Test: Verify embedding lookup works correctly for 24-token mode."""
    print("\n[Test 6] Embedding Lookup (24-token mode)")

    key = jax.random.PRNGKey(101112)
    d_model = 128

    # Create embedding layer
    embedding = MockEmbeddingLayer(VOCAB_SIZE_24, d_model, key)

    # Test single token
    single_token = jnp.array(500)  # Token index 500
    single_embedding = embedding(single_token)

    if single_embedding.shape != (d_model,):
        print(f"  FAIL: Single token embedding shape {single_embedding.shape}, expected ({d_model},)")
        return False
    print(f"  Single token embedding shape: {single_embedding.shape}")

    # Test sequence of tokens
    seq_len = 24  # One message worth of tokens
    tokens = jax.random.randint(key, (seq_len,), 0, VOCAB_SIZE_24)
    seq_embedding = embedding(tokens)

    if seq_embedding.shape != (seq_len, d_model):
        print(f"  FAIL: Sequence embedding shape {seq_embedding.shape}, expected ({seq_len}, {d_model})")
        return False
    print(f"  Sequence embedding shape: {seq_embedding.shape}")

    # Verify lookup consistency
    for i in range(min(5, seq_len)):
        individual = embedding(tokens[i])
        if not jnp.allclose(individual, seq_embedding[i]):
            print(f"  FAIL: Embedding lookup inconsistent at position {i}")
            return False

    print("  PASS: Embedding lookup works correctly for 24-token mode")
    return True


def test_sequence_length_handling():
    """Test: Verify correct handling of various sequence lengths."""
    print("\n[Test 7] Sequence Length Handling")

    key = jax.random.PRNGKey(131415)
    d_model = 64

    # Test different sequence lengths
    test_lengths = [1, 22, 24, 100, 500, 1000]

    for token_mode, vocab_size in [(TOKEN_MODE_22, VOCAB_SIZE_22),
                                    (TOKEN_MODE_24, VOCAB_SIZE_24)]:
        print(f"  Testing {token_mode}-token mode:")
        model = MockLOBModel(vocab_size, d_model, key)

        for seq_len in test_lengths:
            tokens = jax.random.randint(key, (seq_len,), 0, vocab_size)

            try:
                log_probs = model.forward(tokens)

                if log_probs.shape != (vocab_size,):
                    print(f"    FAIL: seq_len={seq_len}, wrong output shape {log_probs.shape}")
                    return False

                if jnp.any(jnp.isnan(log_probs)):
                    print(f"    FAIL: seq_len={seq_len}, NaN in output")
                    return False

            except Exception as e:
                print(f"    FAIL: seq_len={seq_len}, exception: {e}")
                return False

        print(f"    All sequence lengths handled correctly")

    print("  PASS: Sequence length handling works for all tested lengths")
    return True


def test_batch_token_input():
    """Test: Verify batch processing of tokenized input."""
    print("\n[Test 8] Batch Token Input")

    key = jax.random.PRNGKey(161718)
    d_model = 64
    batch_size = 4
    seq_len = 100

    for token_mode, vocab_size in [(TOKEN_MODE_22, VOCAB_SIZE_22),
                                    (TOKEN_MODE_24, VOCAB_SIZE_24)]:
        print(f"  Testing {token_mode}-token mode:")

        # Create embedding layer
        embedding = MockEmbeddingLayer(vocab_size, d_model, key)

        # Create batch of tokens
        tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
        print(f"    Input batch shape: {tokens.shape}")

        # Batch embedding lookup
        batch_embeddings = embedding(tokens)
        expected_shape = (batch_size, seq_len, d_model)

        if batch_embeddings.shape != expected_shape:
            print(f"    FAIL: Expected shape {expected_shape}, got {batch_embeddings.shape}")
            return False

        print(f"    Output embedding shape: {batch_embeddings.shape}")

    print("  PASS: Batch token input processed correctly")
    return True


def test_token_embedding_shape_transformation():
    """Test: Verify shape transformation from tokens to embedded representation."""
    print("\n[Test 9] Token to Embedding Shape Transformation")

    key = jax.random.PRNGKey(192021)

    test_cases = [
        # (token_mode, vocab_size, batch_size, seq_len, d_model)
        (22, VOCAB_SIZE_22, None, 100, 64),
        (22, VOCAB_SIZE_22, None, 22, 128),
        (24, VOCAB_SIZE_24, None, 100, 64),
        (24, VOCAB_SIZE_24, None, 24, 128),
        (22, VOCAB_SIZE_22, 4, 50, 64),
        (24, VOCAB_SIZE_24, 8, 100, 256),
    ]

    for token_mode, vocab_size, batch_size, seq_len, d_model in test_cases:
        embedding = MockEmbeddingLayer(vocab_size, d_model, key)

        if batch_size is None:
            # Unbatched
            tokens = jax.random.randint(key, (seq_len,), 0, vocab_size)
            expected_shape = (seq_len, d_model)
        else:
            # Batched
            tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
            expected_shape = (batch_size, seq_len, d_model)

        embeddings = embedding(tokens)

        if embeddings.shape != expected_shape:
            print(f"  FAIL: tokens {tokens.shape} -> embeddings {embeddings.shape}, expected {expected_shape}")
            return False

        print(f"    tokens {tokens.shape} -> embeddings {embeddings.shape}")

    print("  PASS: All shape transformations correct")
    return True


def test_special_token_handling():
    """Test: Verify special tokens (MASK, HIDDEN, NA, START) are handled."""
    print("\n[Test 10] Special Token Handling")

    # Special token indices from Vocab class
    MASK_TOK = 0
    HIDDEN_TOK = 1
    NA_TOK = 2
    START_TOK = 3

    key = jax.random.PRNGKey(222324)
    d_model = 64

    for token_mode, vocab_size in [(TOKEN_MODE_22, VOCAB_SIZE_22),
                                    (TOKEN_MODE_24, VOCAB_SIZE_24)]:
        print(f"  Testing {token_mode}-token mode:")

        embedding = MockEmbeddingLayer(vocab_size, d_model, key)
        model = MockLOBModel(vocab_size, d_model, key)

        # Create sequence with special tokens
        special_tokens = jnp.array([MASK_TOK, HIDDEN_TOK, NA_TOK, START_TOK])

        # Verify embedding lookup works
        special_embeddings = embedding(special_tokens)
        if special_embeddings.shape != (4, d_model):
            print(f"    FAIL: Special token embedding shape wrong")
            return False

        # Verify each special token has unique embedding
        for i in range(4):
            for j in range(i + 1, 4):
                if jnp.allclose(special_embeddings[i], special_embeddings[j]):
                    print(f"    FAIL: Special tokens {i} and {j} have same embedding")
                    return False

        # Verify model can process sequence with special tokens
        seq_with_special = jnp.concatenate([
            special_tokens,
            jax.random.randint(key, (20,), 4, vocab_size)  # Regular tokens
        ])

        try:
            log_probs = model.forward(seq_with_special)
            if jnp.any(jnp.isnan(log_probs)):
                print("    FAIL: NaN when processing special tokens")
                return False
        except Exception as e:
            print(f"    FAIL: Exception with special tokens: {e}")
            return False

        print(f"    Special tokens processed correctly")

    print("  PASS: Special token handling works correctly")
    return True


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Integration Test: Data Flow from Token Encoding to Model Input")
    print("=" * 60)

    results = []
    results.append(("Token Format 22-tok", test_token_sequence_format_22()))
    results.append(("Token Format 24-tok", test_token_sequence_format_24()))
    results.append(("Model Input 22-tok", test_model_accepts_tokenized_input_22()))
    results.append(("Model Input 24-tok", test_model_accepts_tokenized_input_24()))
    results.append(("Embedding Lookup 22-tok", test_embedding_lookup_22()))
    results.append(("Embedding Lookup 24-tok", test_embedding_lookup_24()))
    results.append(("Sequence Length", test_sequence_length_handling()))
    results.append(("Batch Input", test_batch_token_input()))
    results.append(("Shape Transform", test_token_embedding_shape_transformation()))
    results.append(("Special Tokens", test_special_token_handling()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All integration tests passed!")
    else:
        print("Some tests failed.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
