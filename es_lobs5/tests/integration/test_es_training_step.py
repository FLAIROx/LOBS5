"""
Test: Complete ES Training Step Integration
File: es_lobs5/tests/integration/test_es_training_step.py

This test verifies a complete ES training step integration:
1. Create mock model params
2. Create batch of data
3. Run one ES training step:
   - Generate perturbations for N workers
   - Forward pass for each worker
   - Compute fitness
   - Estimate gradient
   - Update params
4. Verify params are updated
5. Verify training metrics are computed

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_es_training_step.py
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


# ============================================================================
# Mock ES Training Step Implementation
# ============================================================================

def create_mock_params(key: jax.random.PRNGKey, layer_sizes: list = None) -> dict:
    """Create mock model parameters.

    Args:
        key: JAX random key
        layer_sizes: List of layer dimensions, default [(64, 128), (128, 64), (64, 32)]

    Returns:
        Nested parameter dictionary
    """
    if layer_sizes is None:
        layer_sizes = [(64, 128), (128, 64), (64, 32)]

    params = {}
    for i, (in_dim, out_dim) in enumerate(layer_sizes):
        key, k1, k2 = jax.random.split(key, 3)
        params[f'layer_{i}'] = {
            'weights': jax.random.normal(k1, (in_dim, out_dim)) * 0.1,
            'bias': jax.random.normal(k2, (out_dim,)) * 0.01,
        }
    return params


def create_mock_batch(key: jax.random.PRNGKey, batch_size: int, input_dim: int) -> jnp.ndarray:
    """Create mock input batch.

    Args:
        key: JAX random key
        batch_size: Number of samples
        input_dim: Input feature dimension

    Returns:
        Input array of shape (batch_size, input_dim)
    """
    return jax.random.normal(key, (batch_size, input_dim))


def generate_perturbations(key: jax.random.PRNGKey, params: dict,
                           n_workers: int, sigma: float) -> dict:
    """Generate random perturbations for ES.

    Args:
        key: JAX random key
        params: Parameter dictionary
        n_workers: Number of ES workers
        sigma: Noise standard deviation

    Returns:
        Perturbation dictionary with same structure as params,
        each leaf has shape (n_workers, *param_shape)
    """
    def gen_noise_for_leaf(k, leaf):
        return sigma * jax.random.normal(k, (n_workers,) + leaf.shape)

    # Flatten params to generate keys for each leaf
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(flat_params))

    flat_noise = [gen_noise_for_leaf(k, p) for k, p in zip(keys, flat_params)]
    return jax.tree_util.tree_unflatten(tree_def, flat_noise)


def apply_perturbation(params: dict, perturbations: dict, worker_idx: int) -> dict:
    """Apply perturbation for a specific worker.

    Args:
        params: Original parameters
        perturbations: Perturbation dictionary with worker dimension first
        worker_idx: Index of the worker

    Returns:
        Perturbed parameters for this worker
    """
    def add_noise(param, noise):
        return param + noise[worker_idx]

    return jax.tree.map(add_noise, params, perturbations)


def mock_forward_pass(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Simple MLP forward pass for testing.

    Args:
        params: Model parameters
        x: Input array

    Returns:
        Output after forward pass
    """
    h = x
    layer_keys = sorted([k for k in params.keys() if k.startswith('layer_')])

    for i, layer_key in enumerate(layer_keys):
        w = params[layer_key]['weights']
        b = params[layer_key]['bias']
        h = jnp.dot(h, w) + b
        # ReLU activation except for last layer
        if i < len(layer_keys) - 1:
            h = jax.nn.relu(h)

    return h


def compute_fitness(params: dict, x: jnp.ndarray, target: jnp.ndarray) -> float:
    """Compute fitness score (negative loss).

    Args:
        params: Model parameters
        x: Input batch
        target: Target values

    Returns:
        Fitness score (higher is better)
    """
    output = mock_forward_pass(params, x)
    # Negative MSE as fitness (higher is better)
    mse = jnp.mean((output - target) ** 2)
    return -mse


def estimate_es_gradient(fitness_scores: jnp.ndarray, perturbations: dict,
                         sigma: float) -> dict:
    """Estimate gradient using ES formula.

    Args:
        fitness_scores: Array of fitness for each worker, shape (n_workers,)
        perturbations: Perturbation dictionary
        sigma: Noise standard deviation

    Returns:
        Gradient estimate dictionary with same structure as params
    """
    n_workers = fitness_scores.shape[0]

    # Normalize fitness to advantages (mean=0, std=1)
    mean = jnp.mean(fitness_scores)
    std = jnp.std(fitness_scores) + 1e-8
    advantages = (fitness_scores - mean) / std

    def compute_grad_leaf(noise):
        # noise shape: (n_workers, *param_shape)
        # advantages shape: (n_workers,)
        broadcast_shape = (n_workers,) + (1,) * (noise.ndim - 1)
        weighted = advantages.reshape(broadcast_shape) * noise
        return jnp.sum(weighted, axis=0) / (n_workers * sigma)

    return jax.tree.map(compute_grad_leaf, perturbations)


def es_training_step(key: jax.random.PRNGKey, params: dict, opt_state, optimizer,
                     x: jnp.ndarray, target: jnp.ndarray, n_workers: int,
                     sigma: float) -> tuple:
    """Complete ES training step.

    Args:
        key: JAX random key
        params: Current model parameters
        opt_state: Optimizer state
        optimizer: Optax optimizer
        x: Input batch
        target: Target values
        n_workers: Number of ES workers
        sigma: Noise standard deviation

    Returns:
        Tuple of (new_params, new_opt_state, metrics_dict)
    """
    # Step 1: Generate perturbations for all workers
    key, perturb_key = jax.random.split(key)
    perturbations = generate_perturbations(perturb_key, params, n_workers, sigma)

    # Step 2: Evaluate fitness for each worker
    fitness_scores = []
    for worker_idx in range(n_workers):
        perturbed_params = apply_perturbation(params, perturbations, worker_idx)
        fitness = compute_fitness(perturbed_params, x, target)
        fitness_scores.append(fitness)
    fitness_scores = jnp.array(fitness_scores)

    # Step 3: Estimate gradient
    gradient = estimate_es_gradient(fitness_scores, perturbations, sigma)

    # Step 4: Apply optimizer update (note: ES maximizes fitness, so we negate gradient)
    # Actually for ES: we update in direction of gradient (not descent)
    # But optax expects gradients for minimization, so we negate
    neg_gradient = jax.tree.map(lambda g: -g, gradient)
    updates, new_opt_state = optimizer.update(neg_gradient, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Compute gradient norm for metrics
    grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gradient)
    ))

    # Metrics
    metrics = {
        'mean_fitness': jnp.mean(fitness_scores),
        'max_fitness': jnp.max(fitness_scores),
        'min_fitness': jnp.min(fitness_scores),
        'std_fitness': jnp.std(fitness_scores),
        'grad_norm': grad_norm,
    }

    return new_params, new_opt_state, metrics


def es_training_step_vmapped(key: jax.random.PRNGKey, params: dict, opt_state,
                             optimizer, x: jnp.ndarray, target: jnp.ndarray,
                             n_workers: int, sigma: float) -> tuple:
    """Complete ES training step using vmap for parallel evaluation.

    This is more efficient than the loop-based version.
    """
    # Step 1: Generate perturbations
    key, perturb_key = jax.random.split(key)
    perturbations = generate_perturbations(perturb_key, params, n_workers, sigma)

    # Step 2: Evaluate fitness for all workers using vmap
    def eval_worker(worker_idx):
        perturbed_params = apply_perturbation(params, perturbations, worker_idx)
        return compute_fitness(perturbed_params, x, target)

    worker_indices = jnp.arange(n_workers)
    fitness_scores = jax.vmap(eval_worker)(worker_indices)

    # Step 3: Estimate gradient
    gradient = estimate_es_gradient(fitness_scores, perturbations, sigma)

    # Step 4: Apply optimizer update
    neg_gradient = jax.tree.map(lambda g: -g, gradient)
    updates, new_opt_state = optimizer.update(neg_gradient, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Compute metrics
    grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gradient)
    ))

    metrics = {
        'mean_fitness': jnp.mean(fitness_scores),
        'max_fitness': jnp.max(fitness_scores),
        'min_fitness': jnp.min(fitness_scores),
        'std_fitness': jnp.std(fitness_scores),
        'grad_norm': grad_norm,
    }

    return new_params, new_opt_state, metrics


# ============================================================================
# Integration Tests
# ============================================================================

def test_single_training_step():
    """Test: One complete ES step works end-to-end."""
    print("test_single_training_step...")

    key = jax.random.PRNGKey(42)

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(32, 64), (64, 32)])
    x = create_mock_batch(k2, batch_size=16, input_dim=32)
    target = jax.random.normal(k3, (16, 32))

    n_workers = 8
    sigma = 0.1
    lr = 0.01

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Run one training step
    key, step_key = jax.random.split(key)
    new_params, new_opt_state, metrics = es_training_step(
        step_key, params, opt_state, optimizer, x, target, n_workers, sigma
    )

    # Verify step completed without error
    assert new_params is not None, "new_params should not be None"
    assert new_opt_state is not None, "new_opt_state should not be None"
    assert metrics is not None, "metrics should not be None"

    # Verify params have same structure
    original_leaves = jax.tree_util.tree_leaves(params)
    new_leaves = jax.tree_util.tree_leaves(new_params)
    assert len(original_leaves) == len(new_leaves), "Param structure should be preserved"

    for orig, new in zip(original_leaves, new_leaves):
        assert orig.shape == new.shape, f"Shape mismatch: {orig.shape} vs {new.shape}"

    # Verify metrics are computed
    required_metrics = ['mean_fitness', 'max_fitness', 'min_fitness', 'std_fitness', 'grad_norm']
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert jnp.isfinite(metrics[metric]), f"Metric {metric} is not finite"

    print(f"  n_workers={n_workers}, sigma={sigma}")
    print(f"  mean_fitness={float(metrics['mean_fitness']):.4f}")
    print(f"  grad_norm={float(metrics['grad_norm']):.4f}")
    print("[PASS] test_single_training_step")
    return True


def test_params_updated():
    """Test: Parameters differ after training step."""
    print("test_params_updated...")

    key = jax.random.PRNGKey(123)

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(16, 32), (32, 16)])
    x = create_mock_batch(k2, batch_size=8, input_dim=16)
    target = jax.random.normal(k3, (8, 16))

    n_workers = 16
    sigma = 0.1
    lr = 0.01

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Run training step
    key, step_key = jax.random.split(key)
    new_params, _, _ = es_training_step(
        step_key, params, opt_state, optimizer, x, target, n_workers, sigma
    )

    # Verify params are actually updated
    params_changed = False
    max_diff = 0.0

    for layer_name in params:
        for param_name in params[layer_name]:
            old_val = params[layer_name][param_name]
            new_val = new_params[layer_name][param_name]
            diff = jnp.max(jnp.abs(old_val - new_val))
            max_diff = max(max_diff, float(diff))
            if diff > 1e-10:
                params_changed = True

    assert params_changed, "At least some params should be updated"
    assert max_diff > 1e-8, f"Max diff too small: {max_diff}"
    assert max_diff < 1.0, f"Max diff too large (exploding): {max_diff}"

    print(f"  max_param_diff={max_diff:.6f}")
    print("[PASS] test_params_updated")
    return True


def test_metrics_computed():
    """Test: Fitness and gradient norm are correctly computed."""
    print("test_metrics_computed...")

    key = jax.random.PRNGKey(456)

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(32, 64), (64, 32)])
    x = create_mock_batch(k2, batch_size=16, input_dim=32)
    target = jax.random.normal(k3, (16, 32))

    n_workers = 32
    sigma = 0.05
    lr = 0.001

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Run training step
    key, step_key = jax.random.split(key)
    _, _, metrics = es_training_step(
        step_key, params, opt_state, optimizer, x, target, n_workers, sigma
    )

    # Verify fitness statistics
    assert metrics['max_fitness'] >= metrics['mean_fitness'], "max >= mean"
    assert metrics['min_fitness'] <= metrics['mean_fitness'], "min <= mean"
    assert metrics['std_fitness'] >= 0, "std >= 0"

    # Verify gradient norm is positive (gradients should not be zero with varied fitness)
    assert metrics['grad_norm'] > 0, f"grad_norm should be > 0, got {metrics['grad_norm']}"
    assert jnp.isfinite(metrics['grad_norm']), "grad_norm should be finite"

    # Verify all values are reasonable (not NaN/Inf)
    for metric_name, value in metrics.items():
        assert jnp.isfinite(value), f"{metric_name} is not finite: {value}"

    print(f"  mean_fitness={float(metrics['mean_fitness']):.6f}")
    print(f"  std_fitness={float(metrics['std_fitness']):.6f}")
    print(f"  grad_norm={float(metrics['grad_norm']):.6f}")
    print("[PASS] test_metrics_computed")
    return True


def test_multiple_steps():
    """Test: Multiple training steps don't error."""
    print("test_multiple_steps...")

    key = jax.random.PRNGKey(789)

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(16, 32), (32, 16)])
    x = create_mock_batch(k2, batch_size=8, input_dim=16)
    target = jax.random.normal(k3, (8, 16))

    n_workers = 8
    sigma = 0.1
    lr = 0.01
    n_steps = 10

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Track fitness over steps
    fitness_history = []

    for step in range(n_steps):
        key, step_key = jax.random.split(key)
        params, opt_state, metrics = es_training_step(
            step_key, params, opt_state, optimizer, x, target, n_workers, sigma
        )
        fitness_history.append(float(metrics['mean_fitness']))

        # Verify no NaN/Inf
        for leaf in jax.tree_util.tree_leaves(params):
            assert jnp.all(jnp.isfinite(leaf)), f"Params contain NaN/Inf at step {step}"

    # Verify optimizer state is updated (MockChain returns MockAdamState directly)
    assert opt_state.count == n_steps, f"Optimizer count should be {n_steps}, got {opt_state.count}"

    # Verify fitness is tracked
    assert len(fitness_history) == n_steps, "Should have fitness for each step"

    print(f"  n_steps={n_steps}")
    print(f"  initial_fitness={fitness_history[0]:.4f}")
    print(f"  final_fitness={fitness_history[-1]:.4f}")
    print("[PASS] test_multiple_steps")
    return True


def test_step_determinism():
    """Test: Same seed gives same result."""
    print("test_step_determinism...")

    seed = 12345

    def run_step_with_seed(seed):
        key = jax.random.PRNGKey(seed)

        key, k1, k2, k3 = jax.random.split(key, 4)
        params = create_mock_params(k1, [(16, 32), (32, 16)])
        x = create_mock_batch(k2, batch_size=8, input_dim=16)
        target = jax.random.normal(k3, (8, 16))

        n_workers = 8
        sigma = 0.1
        lr = 0.01

        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params)

        key, step_key = jax.random.split(key)
        new_params, _, metrics = es_training_step(
            step_key, params, opt_state, optimizer, x, target, n_workers, sigma
        )

        return new_params, metrics

    # Run twice with same seed
    params1, metrics1 = run_step_with_seed(seed)
    params2, metrics2 = run_step_with_seed(seed)

    # Run with different seed
    params3, metrics3 = run_step_with_seed(seed + 1)

    # Same seed should give identical results
    for leaf1, leaf2 in zip(jax.tree_util.tree_leaves(params1),
                            jax.tree_util.tree_leaves(params2)):
        diff = jnp.max(jnp.abs(leaf1 - leaf2))
        assert diff < 1e-10, f"Same seed should give identical params, diff={diff}"

    assert jnp.allclose(metrics1['mean_fitness'], metrics2['mean_fitness']), \
        "Same seed should give identical metrics"

    # Different seed should give different results
    different = False
    for leaf1, leaf3 in zip(jax.tree_util.tree_leaves(params1),
                            jax.tree_util.tree_leaves(params3)):
        diff = jnp.max(jnp.abs(leaf1 - leaf3))
        if diff > 1e-3:
            different = True
            break

    assert different, "Different seeds should give different results"

    print(f"  same_seed_diff=0.0 (exact)")
    print(f"  different_seed_diff > 1e-3")
    print("[PASS] test_step_determinism")
    return True


def test_vmapped_vs_loop():
    """Test: Vmapped version gives same results as loop version."""
    print("test_vmapped_vs_loop...")

    key = jax.random.PRNGKey(999)

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(16, 32), (32, 16)])
    x = create_mock_batch(k2, batch_size=8, input_dim=16)
    target = jax.random.normal(k3, (8, 16))

    n_workers = 8
    sigma = 0.1
    lr = 0.01

    optimizer = optax.adam(learning_rate=lr)
    opt_state1 = optimizer.init(params)
    opt_state2 = optimizer.init(params)

    # Run both versions with same key
    key, step_key = jax.random.split(key)

    new_params1, _, metrics1 = es_training_step(
        step_key, params, opt_state1, optimizer, x, target, n_workers, sigma
    )

    new_params2, _, metrics2 = es_training_step_vmapped(
        step_key, params, opt_state2, optimizer, x, target, n_workers, sigma
    )

    # Results should match
    for leaf1, leaf2 in zip(jax.tree_util.tree_leaves(new_params1),
                            jax.tree_util.tree_leaves(new_params2)):
        diff = jnp.max(jnp.abs(leaf1 - leaf2))
        assert diff < 1e-5, f"Vmapped and loop versions should match, diff={diff}"

    # Metrics should match
    for key in metrics1:
        diff = jnp.abs(metrics1[key] - metrics2[key])
        assert diff < 1e-5, f"Metric {key} should match, diff={diff}"

    print("[PASS] test_vmapped_vs_loop")
    return True


def test_gradient_clipping_integration():
    """Test: Gradient clipping works with ES training step."""
    print("test_gradient_clipping_integration...")

    key = jax.random.PRNGKey(111)

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(16, 32), (32, 16)])
    x = create_mock_batch(k2, batch_size=8, input_dim=16)
    target = jax.random.normal(k3, (8, 16)) * 100  # Large target to induce large gradients

    n_workers = 16
    sigma = 0.01  # Small sigma to get larger gradients
    lr = 0.01
    max_grad_norm = 1.0

    # With gradient clipping
    optimizer_clipped = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=lr),
    )
    opt_state_clipped = optimizer_clipped.init(params)

    # Without gradient clipping
    optimizer_unclipped = optax.adam(learning_rate=lr)
    opt_state_unclipped = optimizer_unclipped.init(params)

    key, step_key = jax.random.split(key)

    new_params_clipped, _, metrics_clipped = es_training_step(
        step_key, params, opt_state_clipped, optimizer_clipped,
        x, target, n_workers, sigma
    )

    new_params_unclipped, _, metrics_unclipped = es_training_step(
        step_key, params, opt_state_unclipped, optimizer_unclipped,
        x, target, n_workers, sigma
    )

    # Both should complete without error
    assert new_params_clipped is not None
    assert new_params_unclipped is not None

    # Compute param change magnitudes
    def compute_param_change(old_params, new_params):
        total_sq = 0.0
        for old_leaf, new_leaf in zip(jax.tree_util.tree_leaves(old_params),
                                      jax.tree_util.tree_leaves(new_params)):
            total_sq += float(jnp.sum((old_leaf - new_leaf) ** 2))
        return jnp.sqrt(total_sq)

    change_clipped = compute_param_change(params, new_params_clipped)
    change_unclipped = compute_param_change(params, new_params_unclipped)

    print(f"  clipped_change={change_clipped:.6f}")
    print(f"  unclipped_change={change_unclipped:.6f}")
    print(f"  grad_norm={float(metrics_clipped['grad_norm']):.6f}")

    # Clipped should have smaller or equal change when gradient is large
    if metrics_clipped['grad_norm'] > max_grad_norm:
        assert change_clipped <= change_unclipped + 1e-6, \
            "Clipped update should be <= unclipped when gradient exceeds max_norm"

    print("[PASS] test_gradient_clipping_integration")
    return True


def test_antithetic_sampling():
    """Test: Antithetic sampling (paired +/- perturbations) works."""
    print("test_antithetic_sampling...")

    key = jax.random.PRNGKey(222)

    def generate_antithetic_perturbations(key, params, n_pairs, sigma):
        """Generate antithetic perturbation pairs."""
        def gen_noise_for_leaf(k, leaf):
            base_noise = sigma * jax.random.normal(k, (n_pairs,) + leaf.shape)
            # Stack positive and negative perturbations
            return jnp.concatenate([base_noise, -base_noise], axis=0)

        flat_params, tree_def = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(key, len(flat_params))
        flat_noise = [gen_noise_for_leaf(k, p) for k, p in zip(keys, flat_params)]
        return jax.tree_util.tree_unflatten(tree_def, flat_noise)

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(16, 32), (32, 16)])
    x = create_mock_batch(k2, batch_size=8, input_dim=16)
    target = jax.random.normal(k3, (8, 16))

    n_pairs = 8
    n_workers = 2 * n_pairs  # Total workers (positive + negative)
    sigma = 0.1

    # Generate antithetic perturbations
    key, perturb_key = jax.random.split(key)
    perturbations = generate_antithetic_perturbations(perturb_key, params, n_pairs, sigma)

    # Verify antithetic property
    for leaf in jax.tree_util.tree_leaves(perturbations):
        # First half should be negative of second half
        first_half = leaf[:n_pairs]
        second_half = leaf[n_pairs:]
        diff = jnp.max(jnp.abs(first_half + second_half))
        assert diff < 1e-10, f"Antithetic pairs should cancel: diff={diff}"

    # Evaluate fitness
    fitness_scores = []
    for worker_idx in range(n_workers):
        perturbed_params = apply_perturbation(params, perturbations, worker_idx)
        fitness = compute_fitness(perturbed_params, x, target)
        fitness_scores.append(fitness)
    fitness_scores = jnp.array(fitness_scores)

    # Compute gradient
    gradient = estimate_es_gradient(fitness_scores, perturbations, sigma)

    # Gradient should be computed correctly
    grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gradient)
    ))
    assert jnp.isfinite(grad_norm), "Gradient should be finite"

    print(f"  n_pairs={n_pairs}, n_workers={n_workers}")
    print(f"  grad_norm={float(grad_norm):.6f}")
    print("[PASS] test_antithetic_sampling")
    return True


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("ES Training Step Integration Tests")
    print("=" * 60)

    tests = [
        test_single_training_step,
        test_params_updated,
        test_metrics_computed,
        test_multiple_steps,
        test_step_determinism,
        test_vmapped_vs_loop,
        test_gradient_clipping_integration,
        test_antithetic_sampling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
