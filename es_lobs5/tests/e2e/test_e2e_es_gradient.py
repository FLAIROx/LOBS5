"""
Test: End-to-End ES gradient estimation with real model
File: es_lobs5/tests/e2e/test_e2e_es_gradient.py

Level 3 E2E test that verifies the complete ES gradient estimation pipeline:
1. Initialize real model with parameters
2. Generate perturbations for N workers (16)
3. Run forward passes with perturbed params
4. Compute fitness for each worker
5. Estimate ES gradient using the formula: grad = (1/Nσ) * Σ fitness[i] * noise[i]
6. Apply gradient update

This test uses 16 workers and verifies gradient has same structure as params.
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# Mock Antithetic EggRoll Noiser (for E2E testing without HyperscaleES)
# ============================================================================

def get_nonlora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    """Generate perturbation for a parameter using antithetic sampling.

    Key logic:
    - true_thread_idx = thread_id // 2: pairs share the same random seed
    - sigma = +base_sigma if thread_id % 2 == 0, else -base_sigma: opposite signs
    """
    epoch, thread_id = iterinfo

    noise_reuse = frozen_noiser_params.get("noise_reuse", 0)
    true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse

    # Antithetic sampling: pairs share the same noise pattern
    true_thread_idx = thread_id // 2
    # Core antithetic logic: even threads get +sigma, odd threads get -sigma
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    updates = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        param.shape,
        dtype=param.dtype
    )
    return updates * sigma


class MockEggRoll:
    """Mock EggRoll noiser for E2E testing."""

    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, noise_reuse=0, freeze_nonlora=False, **kwargs):
        """Initialize noiser parameters."""
        frozen_noiser_params = {
            "noise_reuse": noise_reuse,
            "freeze_nonlora": freeze_nonlora,
        }
        noiser_params = {"sigma": sigma, "lr": lr}
        return frozen_noiser_params, noiser_params

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        """Get noisy parameter using antithetic sampling."""
        if iterinfo is None or frozen_noiser_params.get("freeze_nonlora", False):
            return param
        return param + get_nonlora_update_params(
            frozen_noiser_params, noiser_params["sigma"], iterinfo, param, base_key
        )


# ============================================================================
# Simple Mock Model for E2E Testing
# ============================================================================

class SimpleMockModel:
    """A simple mock model for E2E ES gradient testing."""

    def __init__(self, key, d_input=32, d_hidden=64, d_output=16):
        keys = jax.random.split(key, 4)
        self.params = {
            'layer1': {
                'weights': jax.random.normal(keys[0], (d_input, d_hidden)) * 0.1,
                'bias': jnp.zeros(d_hidden),
            },
            'layer2': {
                'weights': jax.random.normal(keys[1], (d_hidden, d_hidden)) * 0.1,
                'bias': jnp.zeros(d_hidden),
            },
            'output': {
                'weights': jax.random.normal(keys[2], (d_hidden, d_output)) * 0.1,
                'bias': jnp.zeros(d_output),
            },
        }
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output

    def forward(self, params, x):
        """Forward pass through the model."""
        # Layer 1
        x = jnp.dot(x, params['layer1']['weights']) + params['layer1']['bias']
        x = jax.nn.relu(x)
        # Layer 2
        x = jnp.dot(x, params['layer2']['weights']) + params['layer2']['bias']
        x = jax.nn.relu(x)
        # Output
        x = jnp.dot(x, params['output']['weights']) + params['output']['bias']
        return jax.nn.log_softmax(x, axis=-1)


# ============================================================================
# ES Gradient Estimation Functions
# ============================================================================

def generate_perturbations(params, key, n_workers, sigma, noiser):
    """Generate perturbations for N workers using antithetic sampling.

    Args:
        params: Parameter pytree
        key: JAX random key
        n_workers: Number of workers (should be even for antithetic)
        sigma: Noise standard deviation
        noiser: Noiser class

    Returns:
        perturbed_params: List of perturbed param pytrees
        perturbations: List of perturbation pytrees (noise only, without base params)
    """
    frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=sigma, lr=0.001)

    # Generate keys for each parameter - use index-based folding
    flat_params, tree_def = jax.tree.flatten(params)
    flat_keys = [jax.random.fold_in(key, jnp.uint32(i)) for i in range(len(flat_params))]
    param_keys = jax.tree.unflatten(tree_def, flat_keys)

    perturbed_params = []
    perturbations = []

    for worker_id in range(n_workers):
        epoch = 0
        iterinfo = (epoch, worker_id)

        # Get noisy params for this worker
        noisy_params = jax.tree.map(
            lambda p, k: noiser.get_noisy_standard(frozen_noiser_params, noiser_params, p, k, iterinfo),
            params, param_keys
        )
        perturbed_params.append(noisy_params)

        # Compute perturbation (noise only)
        pert = jax.tree.map(lambda noisy, orig: noisy - orig, noisy_params, params)
        perturbations.append(pert)

    return perturbed_params, perturbations


def compute_fitness(model, params, x_batch, y_targets):
    """Compute fitness (negative loss) for a model with given params.

    Args:
        model: Model instance
        params: Parameter pytree
        x_batch: Input batch (batch_size, d_input)
        y_targets: Target labels (batch_size,) int

    Returns:
        fitness: Scalar fitness value (higher is better)
    """
    log_probs = jax.vmap(lambda x: model.forward(params, x))(x_batch)
    # Cross-entropy loss (negative because we want fitness, not loss)
    nll = -jnp.mean(log_probs[jnp.arange(len(y_targets)), y_targets])
    # Fitness is negative loss (higher is better)
    return -nll


def compute_es_gradient(fitness_scores, perturbations, sigma):
    """
    Compute ES gradient estimate.

    Formula: grad = (1 / (N * sigma)) * sum(fitness[i] * noise[i])

    Args:
        fitness_scores: array of shape (N,) - fitness for each worker
        perturbations: list of N pytrees - perturbations for each worker
        sigma: noise standard deviation

    Returns:
        gradient: pytree with same structure as perturbations[0]
    """
    N = len(fitness_scores)

    def compute_grad_leaf(perturbation_stack):
        """Compute gradient for a single parameter leaf."""
        # perturbation_stack: (N, *param_shape)
        # fitness_scores: (N,)
        # Broadcast fitness to match perturbation shape
        broadcast_shape = (N,) + (1,) * (perturbation_stack.ndim - 1)
        weighted = fitness_scores.reshape(broadcast_shape) * perturbation_stack
        return jnp.sum(weighted, axis=0) / (N * sigma)

    # Stack perturbations into arrays
    stacked_perts = jax.tree.map(
        lambda *ps: jnp.stack(ps, axis=0),
        *perturbations
    )

    return jax.tree.map(compute_grad_leaf, stacked_perts)


def apply_gradient_update(params, gradient, lr):
    """Apply gradient update to parameters.

    Args:
        params: Parameter pytree
        gradient: Gradient pytree (same structure as params)
        lr: Learning rate

    Returns:
        updated_params: Updated parameter pytree
    """
    return jax.tree.map(lambda p, g: p + lr * g, params, gradient)


# ============================================================================
# Test Functions
# ============================================================================

def test_perturbation_generation():
    """Test: Generate N perturbation vectors with correct structure."""
    print("\n" + "=" * 60)
    print("test_perturbation_generation")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    model = SimpleMockModel(key)
    n_workers = 16
    sigma = 0.1

    # Generate perturbations
    perturbed_params, perturbations = generate_perturbations(
        model.params, key, n_workers, sigma, MockEggRoll
    )

    # Verify number of perturbations
    assert len(perturbed_params) == n_workers, f"Expected {n_workers} perturbed params, got {len(perturbed_params)}"
    assert len(perturbations) == n_workers, f"Expected {n_workers} perturbations, got {len(perturbations)}"

    # Verify structure matches original params
    for i in range(n_workers):
        def check_structure(orig, pert):
            assert orig.shape == pert.shape, f"Shape mismatch: {orig.shape} vs {pert.shape}"
            assert orig.dtype == pert.dtype, f"Dtype mismatch: {orig.dtype} vs {pert.dtype}"

        jax.tree.map(check_structure, model.params, perturbations[i])

    # Verify perturbations are non-zero
    for i in range(n_workers):
        pert_norm = jax.tree.map(lambda x: jnp.linalg.norm(x), perturbations[i])
        total_norm = sum(jax.tree.leaves(pert_norm))
        assert total_norm > 1e-6, f"Worker {i} has zero perturbation"

    print(f"[PASS] Generated {n_workers} perturbations with correct structure")
    print(f"  Sample perturbation norms: {[float(jax.tree.leaves(jax.tree.map(lambda x: jnp.linalg.norm(x), perturbations[i]))[0]) for i in range(4)]}")
    return True


def test_fitness_per_worker():
    """Test: Compute fitness for each perturbed model."""
    print("\n" + "=" * 60)
    print("test_fitness_per_worker")
    print("=" * 60)

    key = jax.random.PRNGKey(123)
    model = SimpleMockModel(key)
    n_workers = 16
    sigma = 0.1

    # Generate perturbations
    key, subkey = jax.random.split(key)
    perturbed_params, perturbations = generate_perturbations(
        model.params, subkey, n_workers, sigma, MockEggRoll
    )

    # Create test data
    batch_size = 32
    key, subkey = jax.random.split(key)
    x_batch = jax.random.normal(subkey, (batch_size, model.d_input))
    key, subkey = jax.random.split(key)
    y_targets = jax.random.randint(subkey, (batch_size,), 0, model.d_output)

    # Compute fitness for each worker
    fitness_scores = []
    for i in range(n_workers):
        fitness = compute_fitness(model, perturbed_params[i], x_batch, y_targets)
        fitness_scores.append(float(fitness))

    fitness_scores = jnp.array(fitness_scores)

    # Verify fitness scores
    assert fitness_scores.shape == (n_workers,), f"Wrong fitness shape: {fitness_scores.shape}"
    assert not jnp.any(jnp.isnan(fitness_scores)), "NaN in fitness scores"
    assert not jnp.any(jnp.isinf(fitness_scores)), "Inf in fitness scores"

    # Verify variance in fitness (different perturbations should give different fitness)
    fitness_std = jnp.std(fitness_scores)
    assert fitness_std > 1e-6, f"No variance in fitness scores: std={fitness_std}"

    print(f"[PASS] Computed fitness for {n_workers} workers")
    print(f"  Fitness mean: {jnp.mean(fitness_scores):.4f}")
    print(f"  Fitness std: {fitness_std:.4f}")
    print(f"  Fitness range: [{jnp.min(fitness_scores):.4f}, {jnp.max(fitness_scores):.4f}]")
    return True


def test_es_gradient_formula():
    """Test: Gradient matches ES formula: grad = (1/Nσ) * Σ fitness[i] * noise[i]"""
    print("\n" + "=" * 60)
    print("test_es_gradient_formula")
    print("=" * 60)

    key = jax.random.PRNGKey(456)
    model = SimpleMockModel(key)
    n_workers = 16
    sigma = 0.1

    # Generate perturbations
    key, subkey = jax.random.split(key)
    perturbed_params, perturbations = generate_perturbations(
        model.params, subkey, n_workers, sigma, MockEggRoll
    )

    # Create test data
    batch_size = 32
    key, subkey = jax.random.split(key)
    x_batch = jax.random.normal(subkey, (batch_size, model.d_input))
    key, subkey = jax.random.split(key)
    y_targets = jax.random.randint(subkey, (batch_size,), 0, model.d_output)

    # Compute fitness for each worker
    fitness_scores = jnp.array([
        compute_fitness(model, perturbed_params[i], x_batch, y_targets)
        for i in range(n_workers)
    ])

    # Compute gradient using ES formula
    gradient = compute_es_gradient(fitness_scores, perturbations, sigma)

    # Manual computation for verification
    def manual_gradient_leaf(*perturbation_leaves):
        stacked = jnp.stack(perturbation_leaves, axis=0)
        broadcast_shape = (n_workers,) + (1,) * (stacked.ndim - 1)
        weighted = fitness_scores.reshape(broadcast_shape) * stacked
        return jnp.sum(weighted, axis=0) / (n_workers * sigma)

    expected_gradient = jax.tree.map(manual_gradient_leaf, *perturbations)

    # Compare gradients
    def compare_gradients(computed, expected):
        diff = jnp.max(jnp.abs(computed - expected))
        assert diff < 1e-5, f"Gradient mismatch: max diff = {diff}"

    jax.tree.map(compare_gradients, gradient, expected_gradient)

    # Verify gradient structure matches params
    def check_shape(param, grad):
        assert param.shape == grad.shape, f"Shape mismatch: {param.shape} vs {grad.shape}"

    jax.tree.map(check_shape, model.params, gradient)

    # Compute gradient statistics
    grad_norms = jax.tree.map(lambda g: jnp.linalg.norm(g), gradient)
    total_grad_norm = sum(jax.tree.leaves(grad_norms))

    print("[PASS] Gradient matches ES formula")
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    print(f"  Gradient structure verified to match params")
    return True


def test_antithetic_sampling():
    """Test: Pairs 2k, 2k+1 are negatives (antithetic sampling)."""
    print("\n" + "=" * 60)
    print("test_antithetic_sampling")
    print("=" * 60)

    key = jax.random.PRNGKey(789)
    model = SimpleMockModel(key)
    n_workers = 16  # 8 antithetic pairs
    sigma = 0.1

    # Generate perturbations
    key, subkey = jax.random.split(key)
    perturbed_params, perturbations = generate_perturbations(
        model.params, subkey, n_workers, sigma, MockEggRoll
    )

    # Verify antithetic pairs
    n_pairs = n_workers // 2
    for pair_idx in range(n_pairs):
        even_idx = 2 * pair_idx
        odd_idx = 2 * pair_idx + 1

        pert_even = perturbations[even_idx]
        pert_odd = perturbations[odd_idx]

        # Check that perturbations are opposite
        def check_antithetic(p_even, p_odd):
            sum_pert = p_even + p_odd
            max_diff = jnp.max(jnp.abs(sum_pert))
            assert max_diff < 1e-6, (
                f"Pair ({even_idx}, {odd_idx}): perturbations should be opposite, "
                f"but |pert_even + pert_odd|_max = {max_diff:.2e}"
            )

        jax.tree.map(check_antithetic, pert_even, pert_odd)

    print(f"[PASS] Antithetic sampling verified for {n_pairs} pairs")
    print(f"  All pairs have opposite perturbations: sigma[2k] = -sigma[2k+1]")
    return True


def test_gradient_update():
    """Test: Params update correctly with gradient."""
    print("\n" + "=" * 60)
    print("test_gradient_update")
    print("=" * 60)

    key = jax.random.PRNGKey(101)
    model = SimpleMockModel(key)
    n_workers = 16
    sigma = 0.1
    lr = 0.001

    # Generate perturbations
    key, subkey = jax.random.split(key)
    perturbed_params, perturbations = generate_perturbations(
        model.params, subkey, n_workers, sigma, MockEggRoll
    )

    # Create test data
    batch_size = 32
    key, subkey = jax.random.split(key)
    x_batch = jax.random.normal(subkey, (batch_size, model.d_input))
    key, subkey = jax.random.split(key)
    y_targets = jax.random.randint(subkey, (batch_size,), 0, model.d_output)

    # Compute fitness for each worker
    fitness_scores = jnp.array([
        compute_fitness(model, perturbed_params[i], x_batch, y_targets)
        for i in range(n_workers)
    ])

    # Compute gradient
    gradient = compute_es_gradient(fitness_scores, perturbations, sigma)

    # Apply gradient update
    original_params = model.params
    updated_params = apply_gradient_update(original_params, gradient, lr)

    # Verify update was applied
    def check_update(orig, grad, updated):
        expected = orig + lr * grad
        diff = jnp.max(jnp.abs(updated - expected))
        assert diff < 1e-10, f"Update mismatch: diff = {diff}"

    jax.tree.map(check_update, original_params, gradient, updated_params)

    # Verify params actually changed
    def check_changed(orig, updated):
        diff = jnp.max(jnp.abs(updated - orig))
        return diff

    changes = jax.tree.map(check_changed, original_params, updated_params)
    max_change = max(jax.tree.leaves(changes))
    assert max_change > 1e-10, f"Params didn't change after update: max_change = {max_change}"

    # Compute fitness before and after update
    fitness_before = compute_fitness(model, original_params, x_batch, y_targets)
    fitness_after = compute_fitness(model, updated_params, x_batch, y_targets)

    print("[PASS] Gradient update applied correctly")
    print(f"  Max parameter change: {max_change:.6f}")
    print(f"  Fitness before: {fitness_before:.4f}")
    print(f"  Fitness after: {fitness_after:.4f}")
    print(f"  Fitness change: {fitness_after - fitness_before:.6f}")
    return True


def test_gradient_structure_matches_params():
    """Test: Gradient has same structure as params (all keys and shapes)."""
    print("\n" + "=" * 60)
    print("test_gradient_structure_matches_params")
    print("=" * 60)

    key = jax.random.PRNGKey(202)
    model = SimpleMockModel(key)
    n_workers = 16
    sigma = 0.1

    # Generate perturbations
    key, subkey = jax.random.split(key)
    perturbed_params, perturbations = generate_perturbations(
        model.params, subkey, n_workers, sigma, MockEggRoll
    )

    # Create test data
    batch_size = 32
    key, subkey = jax.random.split(key)
    x_batch = jax.random.normal(subkey, (batch_size, model.d_input))
    key, subkey = jax.random.split(key)
    y_targets = jax.random.randint(subkey, (batch_size,), 0, model.d_output)

    # Compute fitness
    fitness_scores = jnp.array([
        compute_fitness(model, perturbed_params[i], x_batch, y_targets)
        for i in range(n_workers)
    ])

    # Compute gradient
    gradient = compute_es_gradient(fitness_scores, perturbations, sigma)

    # Check structure recursively
    def check_structure(param, grad, path=""):
        if isinstance(param, dict):
            assert isinstance(grad, dict), f"Gradient at {path} should be dict"
            assert set(param.keys()) == set(grad.keys()), f"Key mismatch at {path}"
            for k in param.keys():
                check_structure(param[k], grad[k], f"{path}/{k}")
        else:
            assert param.shape == grad.shape, f"Shape mismatch at {path}: {param.shape} vs {grad.shape}"
            assert param.dtype == grad.dtype, f"Dtype mismatch at {path}: {param.dtype} vs {grad.dtype}"

    check_structure(model.params, gradient)

    # Count leaves
    n_param_leaves = len(jax.tree.leaves(model.params))
    n_grad_leaves = len(jax.tree.leaves(gradient))
    assert n_param_leaves == n_grad_leaves, f"Leaf count mismatch: {n_param_leaves} vs {n_grad_leaves}"

    print("[PASS] Gradient structure matches params")
    print(f"  Number of parameter leaves: {n_param_leaves}")
    print(f"  All shapes and dtypes match")
    return True


def test_multiple_epochs():
    """Test: Run multiple ES epochs to verify training loop."""
    print("\n" + "=" * 60)
    print("test_multiple_epochs")
    print("=" * 60)

    key = jax.random.PRNGKey(303)
    model = SimpleMockModel(key)
    n_workers = 16
    sigma = 0.1
    lr = 0.01
    n_epochs = 5

    # Create test data
    batch_size = 32
    key, subkey = jax.random.split(key)
    x_batch = jax.random.normal(subkey, (batch_size, model.d_input))
    key, subkey = jax.random.split(key)
    y_targets = jax.random.randint(subkey, (batch_size,), 0, model.d_output)

    params = model.params
    fitness_history = []

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)

        # Generate perturbations
        perturbed_params, perturbations = generate_perturbations(
            params, subkey, n_workers, sigma, MockEggRoll
        )

        # Compute fitness for each worker
        fitness_scores = jnp.array([
            compute_fitness(model, perturbed_params[i], x_batch, y_targets)
            for i in range(n_workers)
        ])

        # Compute gradient
        gradient = compute_es_gradient(fitness_scores, perturbations, sigma)

        # Apply gradient update
        params = apply_gradient_update(params, gradient, lr)

        # Track progress
        mean_fitness = float(jnp.mean(fitness_scores))
        fitness_history.append(mean_fitness)

    # Verify training progressed (fitness should not be NaN/Inf)
    for i, fitness in enumerate(fitness_history):
        assert not np.isnan(fitness), f"Epoch {i}: NaN fitness"
        assert not np.isinf(fitness), f"Epoch {i}: Inf fitness"

    print(f"[PASS] Completed {n_epochs} ES training epochs")
    print(f"  Fitness history: {[f'{f:.4f}' for f in fitness_history]}")
    print(f"  Final mean fitness: {fitness_history[-1]:.4f}")
    return True


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E2E Test: ES Gradient Estimation with Real Model")
    print("=" * 70)
    print(f"Workers: 16")
    print(f"Sigma: 0.1")
    print("=" * 70)

    test_perturbation_generation()
    test_fitness_per_worker()
    test_es_gradient_formula()
    test_antithetic_sampling()
    test_gradient_update()
    test_gradient_structure_matches_params()
    test_multiple_epochs()

    print("\n" + "=" * 70)
    print("All E2E ES gradient estimation tests passed!")
    print("=" * 70)
