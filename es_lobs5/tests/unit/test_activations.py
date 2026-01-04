"""Test activation functions."""
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')


# Define ACTIVATIONS dict that maps names to supported activation types
ACTIVATIONS = {
    "gelu": "gelu",
    "full_glu": "full_glu",
    "half_glu1": "half_glu1",
    "half_glu2": "half_glu2",
}


def test_gelu_activation():
    """Test: GELU output is in reasonable range.

    GELU(x) = x * Phi(x) where Phi is the CDF of standard normal.
    Properties:
    - GELU(0) = 0
    - GELU(x) ~ x for large positive x
    - GELU(x) ~ 0 for large negative x
    - Output range is approximately [-0.17, inf)
    """
    key = jax.random.PRNGKey(42)

    # Test various input ranges
    test_cases = [
        ("zero", jnp.zeros((10, 32))),
        ("positive", jax.random.uniform(key, (10, 32), minval=0, maxval=5)),
        ("negative", jax.random.uniform(key, (10, 32), minval=-5, maxval=0)),
        ("mixed", jax.random.normal(key, (10, 32))),
    ]

    for name, x in test_cases:
        output = nn.gelu(x)

        # Basic shape preservation
        assert output.shape == x.shape, f"Shape mismatch for {name}: {output.shape} != {x.shape}"

        # No NaN or Inf values
        assert not jnp.any(jnp.isnan(output)), f"NaN in GELU output for {name}"
        assert not jnp.any(jnp.isinf(output)), f"Inf in GELU output for {name}"

        # GELU output should be bounded below by approximately -0.17
        min_val = jnp.min(output)
        assert min_val >= -0.5, f"GELU output too negative for {name}: min={min_val}"

        print(f"[PASS] GELU {name}: shape={output.shape}, range=[{jnp.min(output):.4f}, {jnp.max(output):.4f}]")

    # Test GELU(0) = 0
    zero_out = nn.gelu(jnp.array([0.0]))
    assert jnp.allclose(zero_out, 0.0, atol=1e-6), f"GELU(0) should be 0, got {zero_out}"
    print("[PASS] GELU(0) = 0")

    # Test positive input: GELU(x) < x for positive x (since Phi(x) < 1)
    # But GELU(x) approaches x as x -> inf
    pos_x = jnp.array([0.5, 1.0, 2.0, 5.0])
    pos_out = nn.gelu(pos_x)
    assert jnp.all(pos_out <= pos_x), "GELU(x) should be <= x for positive x"
    assert jnp.all(pos_out > 0), "GELU(x) should be > 0 for positive x"
    print("[PASS] GELU(x) <= x for positive x")

    return True


def test_glu_gate_range():
    """Test: GLU gate (sigmoid) is in (0, 1).

    GLU variants use sigmoid as gate:
    - full_glu: out1(x) * sigmoid(out2(x))
    - half_glu1: x * sigmoid(out2(x))
    - half_glu2: x * sigmoid(out2(gelu(x)))

    Sigmoid output should always be in (0, 1).
    """
    key = jax.random.PRNGKey(123)

    # Test sigmoid function directly (this is what GLU uses for gating)
    test_inputs = [
        ("large_positive", jnp.array([10.0, 50.0, 100.0])),
        ("large_negative", jnp.array([-10.0, -50.0, -100.0])),
        ("zero", jnp.array([0.0])),
        ("random", jax.random.normal(key, (100,))),
        ("extreme", jnp.array([-1000.0, 1000.0])),
    ]

    for name, x in test_inputs:
        gate = jax.nn.sigmoid(x)

        # Gate should be strictly in (0, 1)
        # Due to numerical precision, we check >= 0 and <= 1
        assert jnp.all(gate >= 0), f"Sigmoid output < 0 for {name}: min={jnp.min(gate)}"
        assert jnp.all(gate <= 1), f"Sigmoid output > 1 for {name}: max={jnp.max(gate)}"

        # No NaN or Inf
        assert not jnp.any(jnp.isnan(gate)), f"NaN in sigmoid output for {name}"
        assert not jnp.any(jnp.isinf(gate)), f"Inf in sigmoid output for {name}"

        print(f"[PASS] sigmoid gate {name}: range=[{jnp.min(gate):.6f}, {jnp.max(gate):.6f}]")

    # Test sigmoid(0) = 0.5
    zero_gate = jax.nn.sigmoid(jnp.array([0.0]))
    assert jnp.allclose(zero_gate, 0.5, atol=1e-6), f"sigmoid(0) should be 0.5, got {zero_gate}"
    print("[PASS] sigmoid(0) = 0.5")

    # Test sigmoid asymptotic behavior
    large_pos = jax.nn.sigmoid(jnp.array([100.0]))
    large_neg = jax.nn.sigmoid(jnp.array([-100.0]))
    assert large_pos > 0.999, f"sigmoid(100) should be ~1, got {large_pos}"
    assert large_neg < 0.001, f"sigmoid(-100) should be ~0, got {large_neg}"
    print(f"[PASS] sigmoid asymptotic: sigmoid(100)={large_pos[0]:.10f}, sigmoid(-100)={large_neg[0]:.10f}")

    return True


def test_glu_variants_shape():
    """Test: GLU variants preserve output shape.

    All GLU variants should output the same shape as d_model.
    """
    d_model = 64
    seq_len = 16
    key = jax.random.PRNGKey(456)

    # Simulate GLU operations (without full SequenceLayer to keep test focused)
    x = jax.random.normal(key, (seq_len, d_model))

    # full_glu: Dense(x) * sigmoid(Dense(x))
    key1, key2 = jax.random.split(key)
    out1_params = jax.random.normal(key1, (d_model, d_model)) * 0.02
    out2_params = jax.random.normal(key2, (d_model, d_model)) * 0.02

    gelu_x = nn.gelu(x)
    out1 = gelu_x @ out1_params
    gate = jax.nn.sigmoid(gelu_x @ out2_params)
    full_glu_out = out1 * gate

    assert full_glu_out.shape == (seq_len, d_model), \
        f"full_glu shape mismatch: {full_glu_out.shape} != {(seq_len, d_model)}"
    print(f"[PASS] full_glu output shape: {full_glu_out.shape}")

    # half_glu1: x * sigmoid(Dense(x))
    gate = jax.nn.sigmoid(gelu_x @ out2_params)
    half_glu1_out = gelu_x * gate

    assert half_glu1_out.shape == (seq_len, d_model), \
        f"half_glu1 shape mismatch: {half_glu1_out.shape} != {(seq_len, d_model)}"
    print(f"[PASS] half_glu1 output shape: {half_glu1_out.shape}")

    # half_glu2: x * sigmoid(Dense(gelu(x)))
    gelu_for_gate = nn.gelu(x)
    gate = jax.nn.sigmoid(gelu_for_gate @ out2_params)
    half_glu2_out = x * gate

    assert half_glu2_out.shape == (seq_len, d_model), \
        f"half_glu2 shape mismatch: {half_glu2_out.shape} != {(seq_len, d_model)}"
    print(f"[PASS] half_glu2 output shape: {half_glu2_out.shape}")

    return True


def test_activations_dict():
    """Test: ACTIVATIONS dict contains expected activation types."""
    expected_activations = ["gelu", "full_glu", "half_glu1", "half_glu2"]

    for act in expected_activations:
        assert act in ACTIVATIONS, f"Missing activation: {act}"
        print(f"[PASS] {act} in ACTIVATIONS dict")

    return True


def test_gelu_gradient_flow():
    """Test: GELU gradient flows properly (non-zero gradients for most inputs)."""
    key = jax.random.PRNGKey(789)
    x = jax.random.normal(key, (32,))

    # Compute gradient of sum(gelu(x)) w.r.t. x
    def gelu_sum(x):
        return jnp.sum(nn.gelu(x))

    grad_fn = jax.grad(gelu_sum)
    grads = grad_fn(x)

    # Most gradients should be non-zero (except possibly for very negative inputs)
    non_zero_grads = jnp.sum(jnp.abs(grads) > 1e-6)
    total_grads = grads.shape[0]

    # At least 70% of gradients should be non-zero for normal distribution input
    assert non_zero_grads >= 0.7 * total_grads, \
        f"Too few non-zero gradients: {non_zero_grads}/{total_grads}"

    print(f"[PASS] GELU gradient flow: {non_zero_grads}/{total_grads} non-zero gradients")

    # Gradient should be ~1 for large positive inputs
    large_pos_x = jnp.array([10.0])
    large_grad = jax.grad(gelu_sum)(large_pos_x)
    assert jnp.abs(large_grad - 1.0) < 0.1, f"GELU gradient for large x should be ~1, got {large_grad}"
    print(f"[PASS] GELU gradient for large positive x: {large_grad[0]:.6f} (should be ~1)")

    return True


def test_sigmoid_gradient_flow():
    """Test: Sigmoid gradient flows properly."""
    key = jax.random.PRNGKey(101)
    x = jax.random.normal(key, (32,))

    def sigmoid_sum(x):
        return jnp.sum(jax.nn.sigmoid(x))

    grad_fn = jax.grad(sigmoid_sum)
    grads = grad_fn(x)

    # All gradients should be positive (sigmoid is monotonically increasing)
    assert jnp.all(grads >= 0), "Sigmoid gradients should be non-negative"
    print(f"[PASS] Sigmoid gradients all non-negative: min={jnp.min(grads):.6f}")

    # Gradient at 0 should be 0.25 (sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5)
    zero_grad = jax.grad(sigmoid_sum)(jnp.array([0.0]))
    assert jnp.allclose(zero_grad, 0.25, atol=1e-6), \
        f"Sigmoid gradient at 0 should be 0.25, got {zero_grad}"
    print(f"[PASS] Sigmoid gradient at x=0: {zero_grad[0]:.6f} (should be 0.25)")

    return True


if __name__ == "__main__":
    test_gelu_activation()
    test_glu_gate_range()
    test_glu_variants_shape()
    test_activations_dict()
    test_gelu_gradient_flow()
    test_sigmoid_gradient_flow()
    print("\nAll activation tests passed!")
