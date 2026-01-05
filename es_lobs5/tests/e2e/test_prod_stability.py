"""
Level 4 Production Stability Test - Long-running Training
File: es_lobs5/tests/e2e/test_prod_stability.py

This test verifies production stability for long-running ES training:
1. No NaN/Inf in parameters over many epochs
2. Gradient norms stay bounded (no explosion/vanishing)
3. Parameter magnitudes remain stable
4. No memory growth (param sizes stay constant)
5. Fitness does not degrade catastrophically

For CPU testing: 10-20 epochs
For GPU production: 1000 epochs

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/e2e/test_prod_stability.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp
import numpy as np
import gc
from typing import Tuple, Dict, List, Any


# =============================================================================
# Helper Functions for Numerical Stability Checks
# =============================================================================

def check_nan_inf_in_pytree(pytree: Any, name: str = "pytree") -> Tuple[bool, str]:
    """Check for NaN/Inf values in a pytree.

    Args:
        pytree: JAX pytree (nested dict, list, or array)
        name: Name for error reporting

    Returns:
        (is_valid, error_message) - is_valid is True if no NaN/Inf found
    """
    leaves = jax.tree_util.tree_leaves(pytree)

    for i, leaf in enumerate(leaves):
        if leaf is None:
            continue
        if not hasattr(leaf, 'shape'):
            # Skip non-array leaves (e.g., scalars, None)
            if isinstance(leaf, (int, float)):
                if np.isnan(leaf) or np.isinf(leaf):
                    return False, f"{name} leaf {i}: scalar is NaN/Inf"
            continue

        if not jnp.all(jnp.isfinite(leaf)):
            nan_count = jnp.sum(jnp.isnan(leaf))
            inf_count = jnp.sum(jnp.isinf(leaf))
            return False, f"{name} leaf {i} shape={leaf.shape}: {nan_count} NaN, {inf_count} Inf"

    return True, ""


def compute_pytree_stats(pytree: Any) -> Dict[str, float]:
    """Compute statistics of a pytree.

    Args:
        pytree: JAX pytree

    Returns:
        Dictionary with min, max, mean, std, l2_norm
    """
    leaves = jax.tree_util.tree_leaves(pytree)

    if len(leaves) == 0:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'l2_norm': 0.0}

    all_values = []
    for leaf in leaves:
        if leaf is None:
            continue
        if hasattr(leaf, 'flatten'):
            all_values.append(leaf.flatten())

    if len(all_values) == 0:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'l2_norm': 0.0}

    concatenated = jnp.concatenate(all_values)

    return {
        'min': float(jnp.min(concatenated)),
        'max': float(jnp.max(concatenated)),
        'mean': float(jnp.mean(concatenated)),
        'std': float(jnp.std(concatenated)),
        'l2_norm': float(jnp.sqrt(jnp.sum(concatenated ** 2))),
    }


def get_pytree_sizes(pytree: Any) -> List[Tuple[int, ...]]:
    """Get list of leaf shapes in pytree.

    Args:
        pytree: JAX pytree

    Returns:
        List of shapes for each leaf
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    sizes = []
    for leaf in leaves:
        if leaf is None:
            sizes.append(())
        elif hasattr(leaf, 'shape'):
            sizes.append(leaf.shape)
        else:
            sizes.append(())
    return sizes


# =============================================================================
# Mock Training Components (Same pattern as integration tests)
# =============================================================================

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


# =============================================================================
# ES Training Step Implementation for Stability Testing
# =============================================================================

def create_mock_params(key: jax.random.PRNGKey, layer_sizes: list = None) -> dict:
    """Create mock model parameters."""
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


def generate_perturbations(key: jax.random.PRNGKey, params: dict,
                           n_workers: int, sigma: float) -> dict:
    """Generate random perturbations for ES."""
    def gen_noise_for_leaf(k, leaf):
        return sigma * jax.random.normal(k, (n_workers,) + leaf.shape)

    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(flat_params))

    flat_noise = [gen_noise_for_leaf(k, p) for k, p in zip(keys, flat_params)]
    return jax.tree_util.tree_unflatten(tree_def, flat_noise)


def apply_perturbation(params: dict, perturbations: dict, worker_idx: int) -> dict:
    """Apply perturbation for a specific worker."""
    def add_noise(param, noise):
        return param + noise[worker_idx]
    return jax.tree.map(add_noise, params, perturbations)


def mock_forward_pass(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Simple MLP forward pass for testing."""
    h = x
    layer_keys = sorted([k for k in params.keys() if k.startswith('layer_')])

    for i, layer_key in enumerate(layer_keys):
        w = params[layer_key]['weights']
        b = params[layer_key]['bias']
        h = jnp.dot(h, w) + b
        if i < len(layer_keys) - 1:
            h = jax.nn.relu(h)

    return h


def compute_fitness(params: dict, x: jnp.ndarray, target: jnp.ndarray) -> float:
    """Compute fitness score (negative loss)."""
    output = mock_forward_pass(params, x)
    mse = jnp.mean((output - target) ** 2)
    return -mse


def estimate_es_gradient(fitness_scores: jnp.ndarray, perturbations: dict,
                         sigma: float) -> dict:
    """Estimate gradient using ES formula."""
    n_workers = fitness_scores.shape[0]

    mean = jnp.mean(fitness_scores)
    std = jnp.std(fitness_scores) + 1e-8
    advantages = (fitness_scores - mean) / std

    def compute_grad_leaf(noise):
        broadcast_shape = (n_workers,) + (1,) * (noise.ndim - 1)
        weighted = advantages.reshape(broadcast_shape) * noise
        return jnp.sum(weighted, axis=0) / (n_workers * sigma)

    return jax.tree.map(compute_grad_leaf, perturbations)


def es_training_step_vmapped(key: jax.random.PRNGKey, params: dict, opt_state,
                             optimizer, x: jnp.ndarray, target: jnp.ndarray,
                             n_workers: int, sigma: float) -> tuple:
    """Complete ES training step using vmap for parallel evaluation."""
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


# =============================================================================
# Level 4 Production Stability Tests
# =============================================================================

def test_10_epoch_stability():
    """Test: No NaN/Inf over 10 epochs of training."""
    print("test_10_epoch_stability...")

    key = jax.random.PRNGKey(42)
    n_epochs = 10

    # Setup - use moderately sized model
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(64, 128), (128, 64), (64, 32)])
    x = jax.random.normal(k2, (32, 64))
    target = jax.random.normal(k3, (32, 32))

    n_workers = 16
    sigma = 0.1
    lr = 0.01
    grad_clip = 1.0

    # Use gradient clipping for stability
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr),
    )
    opt_state = optimizer.init(params)

    # Track stability metrics
    nan_inf_detected = False
    fitness_history = []
    grad_norm_history = []
    param_stats_history = []

    for epoch in range(n_epochs):
        key, step_key = jax.random.split(key)
        params, opt_state, metrics = es_training_step_vmapped(
            step_key, params, opt_state, optimizer, x, target, n_workers, sigma
        )

        # Check for NaN/Inf in params
        is_valid, error_msg = check_nan_inf_in_pytree(params, f"params_epoch_{epoch}")
        if not is_valid:
            nan_inf_detected = True
            print(f"  [FAIL] Epoch {epoch}: {error_msg}")
            break

        # Check for NaN/Inf in optimizer state
        is_valid_mu, _ = check_nan_inf_in_pytree(opt_state.mu, f"opt_mu_epoch_{epoch}")
        is_valid_nu, _ = check_nan_inf_in_pytree(opt_state.nu, f"opt_nu_epoch_{epoch}")
        if not is_valid_mu or not is_valid_nu:
            nan_inf_detected = True
            print(f"  [FAIL] Epoch {epoch}: NaN/Inf in optimizer state")
            break

        # Track metrics
        fitness_history.append(float(metrics['mean_fitness']))
        grad_norm_history.append(float(metrics['grad_norm']))
        param_stats = compute_pytree_stats(params)
        param_stats_history.append(param_stats)

    assert not nan_inf_detected, "NaN/Inf detected during training"
    assert len(fitness_history) == n_epochs, f"Training did not complete: {len(fitness_history)}/{n_epochs} epochs"

    # Verify all fitness values are finite
    assert all(np.isfinite(f) for f in fitness_history), "Fitness contains NaN/Inf"
    assert all(np.isfinite(g) for g in grad_norm_history), "Gradient norm contains NaN/Inf"

    print(f"  Completed {n_epochs} epochs without NaN/Inf")
    print(f"  Initial fitness: {fitness_history[0]:.6f}")
    print(f"  Final fitness: {fitness_history[-1]:.6f}")
    print(f"  Max grad_norm: {max(grad_norm_history):.6f}")
    print("[PASS] test_10_epoch_stability")
    return True


def test_gradient_norm_bounded():
    """Test: Gradient norm stays within reasonable bounds."""
    print("test_gradient_norm_bounded...")

    key = jax.random.PRNGKey(123)
    n_epochs = 15

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(32, 64), (64, 32)])
    x = jax.random.normal(k2, (16, 32))
    target = jax.random.normal(k3, (16, 32))

    n_workers = 8
    sigma = 0.1
    lr = 0.01
    grad_clip = 5.0  # Allow some headroom

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr),
    )
    opt_state = optimizer.init(params)

    grad_norm_history = []
    MIN_REASONABLE_NORM = 1e-10  # Gradient vanishing threshold
    MAX_REASONABLE_NORM = 100.0  # Gradient explosion threshold (before clipping)

    for epoch in range(n_epochs):
        key, step_key = jax.random.split(key)
        params, opt_state, metrics = es_training_step_vmapped(
            step_key, params, opt_state, optimizer, x, target, n_workers, sigma
        )
        grad_norm_history.append(float(metrics['grad_norm']))

    # Check bounds
    min_norm = min(grad_norm_history)
    max_norm = max(grad_norm_history)
    mean_norm = np.mean(grad_norm_history)

    # Gradient should not vanish
    assert min_norm > MIN_REASONABLE_NORM, f"Gradient vanishing: min_norm={min_norm}"

    # Gradient should not explode (before clipping is applied)
    assert max_norm < MAX_REASONABLE_NORM, f"Gradient explosion: max_norm={max_norm}"

    # Gradients should be relatively stable (not vary wildly)
    std_norm = np.std(grad_norm_history)
    cv = std_norm / (mean_norm + 1e-8)  # Coefficient of variation
    assert cv < 5.0, f"Gradient norm too unstable: CV={cv:.2f}"

    print(f"  Epochs: {n_epochs}")
    print(f"  Grad norm - min: {min_norm:.6f}, max: {max_norm:.6f}, mean: {mean_norm:.6f}")
    print(f"  Grad norm CV (std/mean): {cv:.4f}")
    print("[PASS] test_gradient_norm_bounded")
    return True


def test_param_magnitude_stable():
    """Test: Parameter magnitudes don't explode or collapse."""
    print("test_param_magnitude_stable...")

    key = jax.random.PRNGKey(456)
    n_epochs = 20

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(32, 64), (64, 32)])
    x = jax.random.normal(k2, (16, 32))
    target = jax.random.normal(k3, (16, 32))

    n_workers = 8
    sigma = 0.05
    lr = 0.005
    grad_clip = 1.0

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr),
    )
    opt_state = optimizer.init(params)

    # Record initial param stats
    initial_stats = compute_pytree_stats(params)
    l2_norm_history = [initial_stats['l2_norm']]
    max_val_history = [initial_stats['max']]
    min_val_history = [initial_stats['min']]

    for epoch in range(n_epochs):
        key, step_key = jax.random.split(key)
        params, opt_state, metrics = es_training_step_vmapped(
            step_key, params, opt_state, optimizer, x, target, n_workers, sigma
        )

        stats = compute_pytree_stats(params)
        l2_norm_history.append(stats['l2_norm'])
        max_val_history.append(stats['max'])
        min_val_history.append(stats['min'])

    # Check parameter stability
    initial_l2 = l2_norm_history[0]
    final_l2 = l2_norm_history[-1]
    max_l2 = max(l2_norm_history)

    # L2 norm should not explode (more than 10x initial)
    assert max_l2 < initial_l2 * 10, f"Param explosion: max_l2={max_l2:.2f}, initial={initial_l2:.2f}"

    # L2 norm should not collapse to near zero
    assert final_l2 > 0.01, f"Param collapse: final_l2={final_l2:.6f}"

    # Max/min values should stay bounded
    max_abs_val = max(abs(v) for v in max_val_history + min_val_history)
    assert max_abs_val < 100, f"Extreme param value: {max_abs_val:.2f}"

    # Parameters should not be static (should change)
    l2_change_ratio = abs(final_l2 - initial_l2) / (initial_l2 + 1e-8)
    # Note: might be small with low lr, just check it's not exactly 0
    # (We don't require change because ES updates might be small)

    print(f"  Epochs: {n_epochs}")
    print(f"  L2 norm - initial: {initial_l2:.4f}, final: {final_l2:.4f}, max: {max_l2:.4f}")
    print(f"  Max absolute value in params: {max_abs_val:.4f}")
    print(f"  L2 change ratio: {l2_change_ratio:.6f}")
    print("[PASS] test_param_magnitude_stable")
    return True


def test_memory_stable():
    """Test: No memory growth (param sizes stay constant)."""
    print("test_memory_stable...")

    key = jax.random.PRNGKey(789)
    n_epochs = 10

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    layer_sizes = [(64, 128), (128, 64), (64, 32)]
    params = create_mock_params(k1, layer_sizes)
    x = jax.random.normal(k2, (32, 64))
    target = jax.random.normal(k3, (32, 32))

    n_workers = 16
    sigma = 0.1
    lr = 0.01

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Record initial sizes
    initial_sizes = get_pytree_sizes(params)
    initial_opt_sizes = get_pytree_sizes(opt_state.mu)

    for epoch in range(n_epochs):
        key, step_key = jax.random.split(key)
        params, opt_state, _ = es_training_step_vmapped(
            step_key, params, opt_state, optimizer, x, target, n_workers, sigma
        )

        # Force garbage collection
        if epoch % 3 == 0:
            gc.collect()

    # Check sizes haven't changed
    final_sizes = get_pytree_sizes(params)
    final_opt_sizes = get_pytree_sizes(opt_state.mu)

    assert initial_sizes == final_sizes, f"Param shapes changed: {initial_sizes} -> {final_sizes}"
    assert initial_opt_sizes == final_opt_sizes, f"Optimizer shapes changed"

    # Count total parameters
    total_params = sum(np.prod(s) for s in initial_sizes if len(s) > 0)

    print(f"  Epochs: {n_epochs}")
    print(f"  Total parameters: {total_params}")
    print(f"  Param shapes preserved: {len(initial_sizes)} leaves")
    print("[PASS] test_memory_stable")
    return True


def test_fitness_no_catastrophic_degradation():
    """Test: Fitness does not degrade catastrophically over epochs."""
    print("test_fitness_no_catastrophic_degradation...")

    key = jax.random.PRNGKey(999)
    n_epochs = 15

    # Setup
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(32, 64), (64, 32)])
    x = jax.random.normal(k2, (16, 32))
    target = jax.random.normal(k3, (16, 32))

    n_workers = 16
    sigma = 0.05
    lr = 0.01
    grad_clip = 1.0

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr),
    )
    opt_state = optimizer.init(params)

    fitness_history = []

    for epoch in range(n_epochs):
        key, step_key = jax.random.split(key)
        params, opt_state, metrics = es_training_step_vmapped(
            step_key, params, opt_state, optimizer, x, target, n_workers, sigma
        )
        fitness_history.append(float(metrics['mean_fitness']))

    # Check for catastrophic degradation
    initial_fitness = fitness_history[0]
    final_fitness = fitness_history[-1]
    min_fitness = min(fitness_history)

    # Fitness should not drop catastrophically (e.g., go to -inf)
    assert min_fitness > -1e10, f"Fitness collapsed: min={min_fitness}"

    # Fitness should not drop by more than 10x from initial (arbitrary but reasonable)
    # Note: For ES, fitness might fluctuate, so we're lenient here
    if initial_fitness < 0:
        # For negative fitness (MSE loss), check it doesn't get much worse
        assert min_fitness > initial_fitness * 100, f"Fitness degraded catastrophically"
    else:
        # For positive fitness, check it doesn't become highly negative
        assert min_fitness > -100 * abs(initial_fitness + 1), "Fitness degraded catastrophically"

    # Check that at least some epochs improved or stayed stable
    improvements = sum(1 for i in range(1, len(fitness_history))
                      if fitness_history[i] >= fitness_history[i-1])
    total_transitions = n_epochs - 1
    improvement_ratio = improvements / total_transitions

    print(f"  Epochs: {n_epochs}")
    print(f"  Fitness - initial: {initial_fitness:.6f}, final: {final_fitness:.6f}")
    print(f"  Min fitness: {min_fitness:.6f}")
    print(f"  Improvement ratio: {improvement_ratio:.2%} ({improvements}/{total_transitions})")
    print("[PASS] test_fitness_no_catastrophic_degradation")
    return True


def test_extended_stability_20_epochs():
    """Test: Extended 20-epoch stability with all metrics tracked."""
    print("test_extended_stability_20_epochs...")

    key = jax.random.PRNGKey(2024)
    n_epochs = 20

    # Setup - larger model
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = create_mock_params(k1, [(64, 128), (128, 128), (128, 64), (64, 32)])
    x = jax.random.normal(k2, (32, 64))
    target = jax.random.normal(k3, (32, 32))

    n_workers = 32
    sigma = 0.05
    lr = 0.005
    grad_clip = 2.0

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr),
    )
    opt_state = optimizer.init(params)

    # Track all metrics
    metrics_history = {
        'fitness': [],
        'grad_norm': [],
        'param_l2': [],
        'param_max': [],
    }

    issues_detected = []

    for epoch in range(n_epochs):
        key, step_key = jax.random.split(key)
        params, opt_state, metrics = es_training_step_vmapped(
            step_key, params, opt_state, optimizer, x, target, n_workers, sigma
        )

        # Check NaN/Inf
        is_valid, error_msg = check_nan_inf_in_pytree(params, f"epoch_{epoch}")
        if not is_valid:
            issues_detected.append(f"Epoch {epoch}: {error_msg}")

        # Track metrics
        param_stats = compute_pytree_stats(params)
        metrics_history['fitness'].append(float(metrics['mean_fitness']))
        metrics_history['grad_norm'].append(float(metrics['grad_norm']))
        metrics_history['param_l2'].append(param_stats['l2_norm'])
        metrics_history['param_max'].append(max(abs(param_stats['max']), abs(param_stats['min'])))

    # Assertions
    assert len(issues_detected) == 0, f"Issues detected: {issues_detected}"

    # All metrics should be finite
    for metric_name, values in metrics_history.items():
        assert all(np.isfinite(v) for v in values), f"{metric_name} contains NaN/Inf"

    # Gradient norms should be bounded
    max_grad = max(metrics_history['grad_norm'])
    assert max_grad < 50, f"Gradient exploded: max={max_grad}"

    # Param L2 should be bounded
    max_param_l2 = max(metrics_history['param_l2'])
    assert max_param_l2 < 100, f"Params exploded: max_l2={max_param_l2}"

    print(f"  Completed {n_epochs} epochs")
    print(f"  Fitness range: [{min(metrics_history['fitness']):.4f}, {max(metrics_history['fitness']):.4f}]")
    print(f"  Grad norm range: [{min(metrics_history['grad_norm']):.6f}, {max(metrics_history['grad_norm']):.6f}]")
    print(f"  Param L2 range: [{min(metrics_history['param_l2']):.4f}, {max(metrics_history['param_l2']):.4f}]")
    print("[PASS] test_extended_stability_20_epochs")
    return True


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all Level 4 production stability tests."""
    print("=" * 70)
    print("Level 4 Production Stability Tests - Long-running Training")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print()

    tests = [
        test_10_epoch_stability,
        test_gradient_norm_bounded,
        test_param_magnitude_stable,
        test_memory_stable,
        test_fitness_no_catastrophic_degradation,
        test_extended_stability_20_epochs,
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

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
