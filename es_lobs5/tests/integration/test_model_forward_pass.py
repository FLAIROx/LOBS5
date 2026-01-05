"""
Integration Test: Full Model Forward Pass
File: es_lobs5/tests/integration/test_model_forward_pass.py

This test verifies the full model forward pass integration using mock modules.
Tests:
- test_embedding_forward: tokens -> embeddings
- test_encoder_forward: embeddings -> encoded sequence
- test_decoder_forward: encoded -> logits
- test_full_forward: tokens -> logits end-to-end
- test_output_shape: output has shape (batch, seq_len, vocab_size)

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_model_forward_pass.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Any
from functools import partial


# ==============================================================================
# Mock Model Configuration
# ==============================================================================

class ModelConfig(NamedTuple):
    """Minimal LOB model configuration for testing."""
    vocab_size: int = 12012  # 22-token mode vocabulary
    d_model: int = 64
    d_ff: int = 256  # Feed-forward hidden dim
    n_layers: int = 2
    n_heads: int = 4
    max_seq_len: int = 512
    dropout_rate: float = 0.0


# ==============================================================================
# Mock Model Modules
# ==============================================================================

class MockEmbedding:
    """Mock embedding layer: tokens -> embeddings."""

    def __init__(self, vocab_size: int, d_model: int, key: jax.Array):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize embedding matrix
        self.embedding_matrix = jax.random.normal(key, (vocab_size, d_model)) * 0.02

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """
        Forward pass: tokens -> embeddings.

        Args:
            tokens: Integer token indices of shape (batch, seq_len) or (seq_len,)

        Returns:
            embeddings: Float array of shape (..., seq_len, d_model)
        """
        return self.embedding_matrix[tokens]


class MockEncoderBlock:
    """Mock encoder block: embeddings -> encoded."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, key: jax.Array):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        # Initialize parameters
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Self-attention weights (simplified)
        self.attn_weight = jax.random.normal(k1, (d_model, d_model)) * 0.02

        # Feed-forward weights
        self.ff_up = jax.random.normal(k2, (d_model, d_ff)) * 0.02
        self.ff_down = jax.random.normal(k3, (d_ff, d_model)) * 0.02

        # Layer norm parameters
        self.ln_scale = jnp.ones(d_model)
        self.ln_bias = jnp.zeros(d_model)

    def layer_norm(self, x: jax.Array) -> jax.Array:
        """Simple layer normalization."""
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True) + 1e-6
        return self.ln_scale * (x - mean) / std + self.ln_bias

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass: embeddings -> encoded.

        Args:
            x: Input of shape (..., seq_len, d_model)

        Returns:
            encoded: Output of shape (..., seq_len, d_model)
        """
        # Simplified self-attention (linear projection for testing)
        attn_out = jnp.matmul(x, self.attn_weight)
        x = self.layer_norm(x + attn_out)

        # Feed-forward
        ff_hidden = jax.nn.gelu(jnp.matmul(x, self.ff_up))
        ff_out = jnp.matmul(ff_hidden, self.ff_down)
        x = self.layer_norm(x + ff_out)

        return x


class MockEncoder:
    """Mock encoder: stacks multiple encoder blocks."""

    def __init__(self, config: ModelConfig, key: jax.Array):
        self.config = config
        keys = jax.random.split(key, config.n_layers)
        self.blocks = [
            MockEncoderBlock(config.d_model, config.d_ff, config.n_heads, k)
            for k in keys
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward through all encoder blocks."""
        for block in self.blocks:
            x = block(x)
        return x


class MockDecoder:
    """Mock decoder: encoded -> logits."""

    def __init__(self, d_model: int, vocab_size: int, key: jax.Array):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Output projection
        self.output_proj = jax.random.normal(key, (d_model, vocab_size)) * 0.02

    def __call__(self, encoded: jax.Array) -> jax.Array:
        """
        Forward pass: encoded -> logits.

        Args:
            encoded: Input of shape (..., seq_len, d_model)

        Returns:
            logits: Output of shape (..., seq_len, vocab_size)
        """
        return jnp.matmul(encoded, self.output_proj)


class MockFullModel:
    """Mock full model: tokens -> logits end-to-end."""

    def __init__(self, config: ModelConfig, key: jax.Array):
        self.config = config

        k1, k2, k3 = jax.random.split(key, 3)
        self.embedding = MockEmbedding(config.vocab_size, config.d_model, k1)
        self.encoder = MockEncoder(config, k2)
        self.decoder = MockDecoder(config.d_model, config.vocab_size, k3)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """
        Full forward pass: tokens -> logits.

        Args:
            tokens: Integer token indices of shape (batch, seq_len)

        Returns:
            logits: Output of shape (batch, seq_len, vocab_size)
        """
        embeddings = self.embedding(tokens)
        encoded = self.encoder(embeddings)
        logits = self.decoder(encoded)
        return logits


# ==============================================================================
# Test Functions
# ==============================================================================

def test_embedding_forward():
    """Test: tokens -> embeddings."""
    print("Running test_embedding_forward...")

    key = jax.random.PRNGKey(42)
    config = ModelConfig()

    # Create embedding layer
    embedding = MockEmbedding(config.vocab_size, config.d_model, key)

    # Create batch of tokens
    batch_size = 4
    seq_len = 32
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    # Forward pass
    embeddings = embedding(tokens)

    # Verify output shape
    expected_shape = (batch_size, seq_len, config.d_model)
    assert embeddings.shape == expected_shape, \
        f"Embedding shape wrong: {embeddings.shape} vs {expected_shape}"

    # Verify no NaNs
    assert not jnp.any(jnp.isnan(embeddings)), "NaN in embeddings"

    # Verify embeddings are not all zeros
    assert jnp.abs(embeddings).max() > 0, "Embeddings are all zeros"

    print(f"  [PASS] Embedding forward: input {tokens.shape} -> output {embeddings.shape}")
    return True


def test_encoder_forward():
    """Test: embeddings -> encoded sequence."""
    print("Running test_encoder_forward...")

    key = jax.random.PRNGKey(123)
    config = ModelConfig()

    # Create encoder
    encoder = MockEncoder(config, key)

    # Create batch of embeddings
    batch_size = 4
    seq_len = 32
    embeddings = jax.random.normal(key, (batch_size, seq_len, config.d_model))

    # Forward pass
    encoded = encoder(embeddings)

    # Verify output shape (should be same as input)
    expected_shape = (batch_size, seq_len, config.d_model)
    assert encoded.shape == expected_shape, \
        f"Encoder output shape wrong: {encoded.shape} vs {expected_shape}"

    # Verify no NaNs
    assert not jnp.any(jnp.isnan(encoded)), "NaN in encoded output"

    # Verify output differs from input (transformation happened)
    assert not jnp.allclose(encoded, embeddings, atol=1e-5), \
        "Encoder output is identical to input"

    print(f"  [PASS] Encoder forward: input {embeddings.shape} -> output {encoded.shape}")
    return True


def test_decoder_forward():
    """Test: encoded -> logits."""
    print("Running test_decoder_forward...")

    key = jax.random.PRNGKey(456)
    config = ModelConfig()

    # Create decoder
    decoder = MockDecoder(config.d_model, config.vocab_size, key)

    # Create batch of encoded sequences
    batch_size = 4
    seq_len = 32
    encoded = jax.random.normal(key, (batch_size, seq_len, config.d_model))

    # Forward pass
    logits = decoder(encoded)

    # Verify output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, \
        f"Decoder output shape wrong: {logits.shape} vs {expected_shape}"

    # Verify no NaNs
    assert not jnp.any(jnp.isnan(logits)), "NaN in logits"

    # Verify logits have reasonable range
    assert jnp.abs(logits).max() < 100, f"Logits too large: {jnp.abs(logits).max()}"

    print(f"  [PASS] Decoder forward: input {encoded.shape} -> output {logits.shape}")
    return True


def test_full_forward():
    """Test: tokens -> logits end-to-end."""
    print("Running test_full_forward...")

    key = jax.random.PRNGKey(789)
    config = ModelConfig()

    # Create full model
    model = MockFullModel(config, key)

    # Create batch of tokens
    batch_size = 4
    seq_len = 32
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    # Forward pass
    logits = model(tokens)

    # Verify output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, \
        f"Full model output shape wrong: {logits.shape} vs {expected_shape}"

    # Verify no NaNs
    assert not jnp.any(jnp.isnan(logits)), "NaN in full model output"

    # Verify softmax produces valid probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    prob_sums = probs.sum(axis=-1)
    assert jnp.allclose(prob_sums, 1.0, atol=1e-5), \
        f"Softmax probabilities don't sum to 1: {prob_sums.mean()}"

    print(f"  [PASS] Full forward: tokens {tokens.shape} -> logits {logits.shape}")
    return True


def test_output_shape():
    """Test: output has shape (batch, seq_len, vocab_size)."""
    print("Running test_output_shape...")

    key = jax.random.PRNGKey(101112)

    # Test various configurations
    test_cases = [
        {"batch_size": 1, "seq_len": 16, "vocab_size": 1000, "d_model": 32},
        {"batch_size": 8, "seq_len": 64, "vocab_size": 12012, "d_model": 64},
        {"batch_size": 4, "seq_len": 128, "vocab_size": 2112, "d_model": 128},
        {"batch_size": 2, "seq_len": 256, "vocab_size": 5000, "d_model": 256},
    ]

    for i, tc in enumerate(test_cases):
        config = ModelConfig(
            vocab_size=tc["vocab_size"],
            d_model=tc["d_model"],
            d_ff=tc["d_model"] * 4,
            n_layers=1,  # Use 1 layer for speed
        )

        # Create model
        k = jax.random.fold_in(key, i)
        model = MockFullModel(config, k)

        # Create tokens
        tokens = jax.random.randint(k, (tc["batch_size"], tc["seq_len"]), 0, config.vocab_size)

        # Forward pass
        logits = model(tokens)

        # Verify shape
        expected_shape = (tc["batch_size"], tc["seq_len"], tc["vocab_size"])
        assert logits.shape == expected_shape, \
            f"Case {i}: shape wrong: {logits.shape} vs {expected_shape}"

        print(f"  Case {i}: batch={tc['batch_size']}, seq={tc['seq_len']}, "
              f"vocab={tc['vocab_size']} -> OK")

    print(f"  [PASS] Output shape verification for {len(test_cases)} configurations")
    return True


def test_gradient_flow():
    """Test: gradients flow through the full model."""
    print("Running test_gradient_flow...")

    key = jax.random.PRNGKey(131415)
    config = ModelConfig(d_model=32, d_ff=128, n_layers=1)

    # Create model
    model = MockFullModel(config, key)

    # Create batch of tokens
    batch_size = 2
    seq_len = 16
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    targets = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    # Define loss function
    def loss_fn(embedding_matrix):
        # Create model with modified embedding
        model_copy = MockFullModel(config, key)
        model_copy.embedding.embedding_matrix = embedding_matrix
        logits = model_copy(tokens)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Cross-entropy loss
        one_hot = jax.nn.one_hot(targets, config.vocab_size)
        loss = -jnp.sum(one_hot * log_probs) / (batch_size * seq_len)
        return loss

    # Compute gradients
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(model.embedding.embedding_matrix)

    # Verify gradients exist and are not all zeros
    assert grads.shape == model.embedding.embedding_matrix.shape, \
        f"Gradient shape wrong: {grads.shape}"
    assert not jnp.any(jnp.isnan(grads)), "NaN in gradients"
    assert jnp.abs(grads).max() > 0, "Gradients are all zeros"

    print(f"  [PASS] Gradient flow: grad shape {grads.shape}, "
          f"max abs grad {jnp.abs(grads).max():.6f}")
    return True


def test_batch_consistency():
    """Test: batched forward pass gives same results as individual forward passes."""
    print("Running test_batch_consistency...")

    key = jax.random.PRNGKey(161718)
    config = ModelConfig(d_model=32, d_ff=128, n_layers=1)

    # Create model
    model = MockFullModel(config, key)

    # Create batch of tokens
    batch_size = 4
    seq_len = 16
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    # Batched forward
    batched_logits = model(tokens)

    # Individual forward passes
    individual_logits = jnp.stack([
        model(tokens[i:i+1])[0]  # Remove batch dim, then re-stack
        for i in range(batch_size)
    ])

    # Verify consistency
    assert jnp.allclose(batched_logits, individual_logits, atol=1e-5), \
        "Batched and individual forward passes differ"

    print(f"  [PASS] Batch consistency: max diff {jnp.abs(batched_logits - individual_logits).max():.6e}")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Integration Test: Full Model Forward Pass")
    print("=" * 60)
    print()

    tests = [
        test_embedding_forward,
        test_encoder_forward,
        test_decoder_forward,
        test_full_forward,
        test_output_shape,
        test_gradient_flow,
        test_batch_consistency,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test_fn.__name__}: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

    print("All integration tests passed!")
    return True


if __name__ == "__main__":
    run_all_tests()
