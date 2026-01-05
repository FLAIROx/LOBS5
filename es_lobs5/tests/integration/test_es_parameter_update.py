"""
Test: ES parameter update step integration
File: es_lobs5/tests/integration/test_es_parameter_update.py

This test verifies ES parameter update step:
1. Create mock params and ES gradient
2. Apply Adam optimizer update
3. Verify params are updated correctly
4. Test optimizer state is maintained
5. Test gradient clipping integration
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp

# Mock optax-like optimizer to avoid optax dependency
class MockAdamState:
    """Mock Adam optimizer state."""
    def __init__(self, count, mu, nu):
        self.count = count
        self.mu = mu
        self.nu = nu

class MockAdam:
    """Mock Adam optimizer matching optax interface."""
    def __init__(self, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def init(self, params):
        mu = jax.tree.map(jnp.zeros_like, params)
        nu = jax.tree.map(jnp.zeros_like, params)
        return MockAdamState(0, mu, nu)

    def update(self, grads, state, params=None):
        count = state.count + 1
        mu = jax.tree.map(lambda m, g: self.b1 * m + (1 - self.b1) * g, state.mu, grads)
        nu = jax.tree.map(lambda v, g: self.b2 * v + (1 - self.b2) * (g ** 2), state.nu, grads)
        mu_hat = jax.tree.map(lambda m: m / (1 - self.b1 ** count), mu)
        nu_hat = jax.tree.map(lambda v: v / (1 - self.b2 ** count), nu)
        updates = jax.tree.map(lambda m, v: -self.lr * m / (jnp.sqrt(v) + self.eps), mu_hat, nu_hat)
        return updates, MockAdamState(count, mu, nu)

def mock_clip_by_global_norm(max_norm):
    """Mock gradient clipping."""
    def clip_fn(grads):
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
        return jax.tree.map(lambda g: g * scale, grads)
    return clip_fn

def mock_apply_updates(params, updates):
    """Apply updates to params."""
    return jax.tree.map(lambda p, u: p + u, params, updates)

class MockChain:
    """Mock optax.chain for combining transforms."""
    def __init__(self, *transforms):
        self.transforms = transforms
        self._optimizer = None
        self._clipper = None
        for t in transforms:
            if isinstance(t, MockAdam):
                self._optimizer = t
            elif callable(t):
                self._clipper = t

    def init(self, params):
        return self._optimizer.init(params)

    def update(self, grads, state, params=None):
        if self._clipper:
            grads = self._clipper(grads)
        return self._optimizer.update(grads, state, params)

# Create mock optax module
class MockOptax:
    @staticmethod
    def adam(learning_rate):
        return MockAdam(learning_rate)

    @staticmethod
    def clip_by_global_norm(max_norm):
        return mock_clip_by_global_norm(max_norm)

    @staticmethod
    def chain(*transforms):
        return MockChain(*transforms)

    @staticmethod
    def apply_updates(params, updates):
        return mock_apply_updates(params, updates)

optax = MockOptax()


def test_adam_update():
    """Test: params_new = params - lr * adam(grad)."""
    # Create mock parameters
    params = {
        'W1': jnp.ones((32, 64)) * 0.5,
        'W2': jnp.ones((64, 32)) * 0.3,
        'b': jnp.zeros(32),
    }

    # Create ES gradient (pseudo-gradient from ES)
    grads = {
        'W1': jnp.ones((32, 64)) * 0.01,
        'W2': jnp.ones((64, 32)) * 0.02,
        'b': jnp.ones(32) * 0.005,
    }

    # Initialize Adam optimizer
    lr = 0.001
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Apply update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)

    # Verify params are updated (not equal to original)
    for key in params:
        diff = jnp.max(jnp.abs(params_new[key] - params[key]))
        assert diff > 0, f"Param {key} should be updated, but diff={diff}"

    # Verify update direction is opposite to gradient (descent)
    # For Adam, first step with zero momentum: update ≈ -lr * grad / (sqrt(grad^2) + eps)
    # = -lr * sign(grad) for uniform gradients
    for key in params:
        # With positive gradients, params should decrease
        assert jnp.all(params_new[key] <= params[key] + 1e-6), \
            f"Param {key} should decrease with positive gradient"

    print(f"[PASS] Adam update: params correctly updated")
    return True


def test_optimizer_state():
    """Test: optimizer state is updated correctly."""
    params = {
        'W': jnp.ones((16, 16)) * 0.5,
    }

    grads = {
        'W': jnp.ones((16, 16)) * 0.01,
    }

    lr = 0.001
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Extract initial state (MockAdamState with count, mu, nu)
    initial_count = opt_state.count
    initial_mu = jax.tree.leaves(opt_state.mu)[0]
    initial_nu = jax.tree.leaves(opt_state.nu)[0]

    assert initial_count == 0, f"Initial count should be 0, got {initial_count}"
    assert jnp.allclose(initial_mu, 0), "Initial mu should be zeros"
    assert jnp.allclose(initial_nu, 0), "Initial nu should be zeros"

    # Apply first update
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Check state is updated
    new_count = opt_state.count
    new_mu = jax.tree.leaves(opt_state.mu)[0]
    new_nu = jax.tree.leaves(opt_state.nu)[0]

    assert new_count == 1, f"Count should be 1 after first update, got {new_count}"
    assert not jnp.allclose(new_mu, 0), "Mu should be updated"
    assert not jnp.allclose(new_nu, 0), "Nu should be updated"

    # Apply second update
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    final_count = opt_state.count
    assert final_count == 2, f"Count should be 2 after second update, got {final_count}"

    print(f"[PASS] Optimizer state correctly maintained: count=0->1->2")
    return True


def test_gradient_clipping():
    """Test: gradients are clipped before update."""
    params = {
        'W': jnp.ones((16, 16)) * 0.5,
    }

    # Large gradient that should be clipped
    large_grads = {
        'W': jnp.ones((16, 16)) * 100.0,  # Very large gradient
    }

    lr = 0.001
    max_grad_norm = 1.0

    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=lr),
    )
    opt_state = optimizer.init(params)

    # Compute expected clipped gradient norm
    grad_norm = jnp.sqrt(jnp.sum(large_grads['W'] ** 2))
    expected_scale = max_grad_norm / grad_norm

    # Apply update
    updates, _ = optimizer.update(large_grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)

    # Compute actual change
    param_diff = jnp.max(jnp.abs(params_new['W'] - params['W']))

    # Without clipping, change would be much larger
    # With clipping, the gradient is scaled down significantly
    # Expected: gradient clipped to norm 1.0, then Adam applied
    assert param_diff < 0.1, f"Gradient clipping should limit update, but diff={param_diff}"

    # Compare with unclipped optimizer
    unclipped_optimizer = optax.adam(learning_rate=lr)
    unclipped_state = unclipped_optimizer.init(params)
    unclipped_updates, _ = unclipped_optimizer.update(large_grads, unclipped_state, params)
    unclipped_params = optax.apply_updates(params, unclipped_updates)
    unclipped_diff = jnp.max(jnp.abs(unclipped_params['W'] - params['W']))

    # Clipped update should be smaller or equal
    assert param_diff <= unclipped_diff + 1e-6, \
        f"Clipped update ({param_diff}) should be <= unclipped ({unclipped_diff})"

    print(f"[PASS] Gradient clipping: clipped_diff={param_diff:.6f} <= unclipped_diff={unclipped_diff:.6f}")
    return True


def test_update_magnitude():
    """Test: ||params_new - params|| is bounded by lr."""
    params = {
        'W1': jnp.ones((32, 64)) * 0.5,
        'W2': jnp.ones((64, 32)) * 0.3,
    }

    # Unit norm gradient
    grads = {
        'W1': jax.random.normal(jax.random.PRNGKey(42), (32, 64)),
        'W2': jax.random.normal(jax.random.PRNGKey(43), (64, 32)),
    }

    # Normalize gradients to unit norm per param
    for key in grads:
        norm = jnp.sqrt(jnp.sum(grads[key] ** 2))
        grads[key] = grads[key] / (norm + 1e-8)

    lr = 0.01
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr),
    )
    opt_state = optimizer.init(params)

    # Apply update
    updates, _ = optimizer.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)

    # Check update magnitude for each param
    for key in params:
        diff_norm = jnp.sqrt(jnp.sum((params_new[key] - params[key]) ** 2))
        param_size = params[key].size

        # Per-element update should be bounded
        max_per_element = jnp.max(jnp.abs(params_new[key] - params[key]))

        # Adam first step: update ≈ lr * grad / sqrt(grad^2 + eps)
        # For normalized grad, this is approximately lr
        assert max_per_element < lr * 2, \
            f"Per-element update for {key} should be bounded, got {max_per_element}"

    print(f"[PASS] Update magnitude bounded by learning rate")
    return True


def test_multiple_updates():
    """Test: multiple update steps work correctly."""
    key = jax.random.PRNGKey(0)

    params = {
        'W': jax.random.normal(key, (32, 32)) * 0.1,
    }

    lr = 0.001
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr),
    )
    opt_state = optimizer.init(params)

    # Track param history
    param_history = [params['W'].copy()]

    # Multiple updates with consistent gradient direction
    num_steps = 10
    for step in range(num_steps):
        key, subkey = jax.random.split(key)

        # Consistent gradient direction with some noise
        grads = {
            'W': jnp.ones((32, 32)) * 0.01 + jax.random.normal(subkey, (32, 32)) * 0.001,
        }

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        param_history.append(params['W'].copy())

    # Verify params continue to change
    total_change = jnp.sqrt(jnp.sum((param_history[-1] - param_history[0]) ** 2))
    assert total_change > 0, f"Params should change over multiple steps, total_change={total_change}"

    # Verify optimizer count
    # MockChain returns MockAdamState directly
    final_count = opt_state.count
    assert final_count == num_steps, f"Count should be {num_steps}, got {final_count}"

    # Verify Adam momentum is built up (mu should be non-zero and growing)
    final_mu = jax.tree.leaves(opt_state.mu)[0]
    mu_norm = jnp.sqrt(jnp.sum(final_mu ** 2))
    assert mu_norm > 0, f"Momentum should be accumulated, mu_norm={mu_norm}"

    print(f"[PASS] Multiple updates: {num_steps} steps, total_change={total_change:.6f}, mu_norm={mu_norm:.6f}")
    return True


def test_nested_params_update():
    """Test: nested parameter structures are updated correctly."""
    # Nested params similar to real models
    params = {
        'encoder': {
            'W': jnp.ones((16, 32)) * 0.5,
            'b': jnp.zeros(32),
        },
        'decoder': {
            'W': jnp.ones((32, 16)) * 0.3,
            'b': jnp.zeros(16),
        },
    }

    grads = {
        'encoder': {
            'W': jnp.ones((16, 32)) * 0.01,
            'b': jnp.ones(32) * 0.005,
        },
        'decoder': {
            'W': jnp.ones((32, 16)) * 0.02,
            'b': jnp.ones(16) * 0.01,
        },
    }

    lr = 0.001
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Apply update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)

    # Verify all nested params are updated
    for group in ['encoder', 'decoder']:
        for key in params[group]:
            diff = jnp.max(jnp.abs(params_new[group][key] - params[group][key]))
            assert diff > 0, f"Param {group}/{key} should be updated, but diff={diff}"

    print(f"[PASS] Nested params correctly updated")
    return True


def test_zero_gradient():
    """Test: zero gradient results in minimal param change."""
    params = {
        'W': jnp.ones((16, 16)) * 0.5,
    }

    zero_grads = {
        'W': jnp.zeros((16, 16)),
    }

    lr = 0.001
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Apply update with zero gradient
    updates, _ = optimizer.update(zero_grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)

    # Params should remain unchanged with zero gradient
    diff = jnp.max(jnp.abs(params_new['W'] - params['W']))
    assert diff < 1e-10, f"Zero gradient should result in no update, but diff={diff}"

    print(f"[PASS] Zero gradient: no param change (diff={diff:.2e})")
    return True


def test_learning_rate_schedule():
    """Test: learning rate schedule affects update magnitude."""
    params = {
        'W': jnp.ones((16, 16)) * 0.5,
    }

    grads = {
        'W': jnp.ones((16, 16)) * 0.01,
    }

    # High learning rate
    high_lr = 0.1
    high_lr_optimizer = optax.adam(learning_rate=high_lr)
    high_lr_state = high_lr_optimizer.init(params)
    high_updates, _ = high_lr_optimizer.update(grads, high_lr_state, params)
    high_params = optax.apply_updates(params, high_updates)
    high_diff = jnp.max(jnp.abs(high_params['W'] - params['W']))

    # Low learning rate
    low_lr = 0.001
    low_lr_optimizer = optax.adam(learning_rate=low_lr)
    low_lr_state = low_lr_optimizer.init(params)
    low_updates, _ = low_lr_optimizer.update(grads, low_lr_state, params)
    low_params = optax.apply_updates(params, low_updates)
    low_diff = jnp.max(jnp.abs(low_params['W'] - params['W']))

    # Higher lr should result in larger update
    assert high_diff > low_diff, \
        f"Higher lr should give larger update: high={high_diff}, low={low_diff}"

    # Ratio should be approximately lr ratio
    ratio = high_diff / low_diff
    expected_ratio = high_lr / low_lr
    # Allow some tolerance due to Adam's adaptive nature
    assert ratio > expected_ratio * 0.5, \
        f"Update ratio ({ratio}) should reflect lr ratio ({expected_ratio})"

    print(f"[PASS] LR schedule: high_diff={high_diff:.6f}, low_diff={low_diff:.6f}, ratio={ratio:.1f}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ES Parameter Update Integration Tests")
    print("=" * 60)

    test_adam_update()
    test_optimizer_state()
    test_gradient_clipping()
    test_update_magnitude()
    test_multiple_updates()
    test_nested_params_update()
    test_zero_gradient()
    test_learning_rate_schedule()

    print("=" * 60)
    print("All ES parameter update tests passed!")
    print("=" * 60)
