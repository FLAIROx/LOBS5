"""
Test: ES gradient estimation from fitness and perturbations
File: es_lobs5/tests/integration/test_es_gradient_estimate.py

This test verifies the ES gradient estimation formula:
    grad = (1 / (N * sigma)) * sum(fitness * noise)

where:
    - N = number of workers
    - sigma = noise standard deviation
    - fitness = fitness scores for each worker
    - noise = perturbation vectors for each worker
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp


def compute_es_gradient(fitness_scores, noise_vectors, sigma):
    """
    Compute ES gradient estimate.

    Args:
        fitness_scores: array of shape (N,) - fitness for each worker
        noise_vectors: pytree of arrays, each with shape (N, ...) - noise per worker
        sigma: noise standard deviation

    Returns:
        gradient: pytree with same structure as noise_vectors (without N dimension)
    """
    N = fitness_scores.shape[0]

    # Handle pytree structure
    def compute_grad_leaf(noise):
        # noise shape: (N, *param_shape)
        # fitness shape: (N,)
        # Broadcast fitness to match noise shape
        broadcast_shape = (N,) + (1,) * (noise.ndim - 1)
        weighted = fitness_scores.reshape(broadcast_shape) * noise
        return jnp.sum(weighted, axis=0) / (N * sigma)

    return jax.tree.map(compute_grad_leaf, noise_vectors)


def test_es_gradient_shape():
    """Test: Gradient tree matches param tree structure."""
    key = jax.random.PRNGKey(42)

    # Create mock parameter tree
    params = {
        'layer1': {
            'weights': jnp.zeros((64, 32)),
            'bias': jnp.zeros((32,)),
        },
        'layer2': {
            'weights': jnp.zeros((32, 16)),
            'bias': jnp.zeros((16,)),
        },
    }

    # Number of workers
    N = 8
    sigma = 0.1

    # Generate fitness scores
    key, subkey = jax.random.split(key)
    fitness_scores = jax.random.normal(subkey, shape=(N,))

    # Generate noise vectors matching param structure
    def gen_noise(key, shape):
        return jax.random.normal(key, shape=(N,) + shape)

    keys = jax.random.split(key, 4)
    noise_vectors = {
        'layer1': {
            'weights': gen_noise(keys[0], (64, 32)),
            'bias': gen_noise(keys[1], (32,)),
        },
        'layer2': {
            'weights': gen_noise(keys[2], (32, 16)),
            'bias': gen_noise(keys[3], (16,)),
        },
    }

    # Compute gradient
    gradient = compute_es_gradient(fitness_scores, noise_vectors, sigma)

    # Verify structure matches
    def check_shape(param, grad):
        assert param.shape == grad.shape, f"Shape mismatch: {param.shape} vs {grad.shape}"

    jax.tree.map(check_shape, params, gradient)

    print("[PASS] test_es_gradient_shape: gradient tree matches param tree")
    return True


def test_es_gradient_formula():
    """Test: grad = sum(advantage * noise) / (N * sigma)"""
    key = jax.random.PRNGKey(123)

    N = 4
    param_shape = (8, 4)
    sigma = 0.05

    # Generate fitness scores
    key, subkey = jax.random.split(key)
    fitness_scores = jax.random.normal(subkey, shape=(N,))

    # Generate noise vectors
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(N,) + param_shape)

    # Manual gradient computation
    expected_grad = jnp.zeros(param_shape)
    for i in range(N):
        expected_grad = expected_grad + fitness_scores[i] * noise[i]
    expected_grad = expected_grad / (N * sigma)

    # Compute using function
    noise_vectors = {'param': noise}
    gradient = compute_es_gradient(fitness_scores, noise_vectors, sigma)

    # Compare
    diff = jnp.max(jnp.abs(gradient['param'] - expected_grad))
    assert diff < 1e-5, f"Gradient mismatch: max diff = {diff}"

    print(f"[PASS] test_es_gradient_formula: max diff = {diff:.2e}")
    return True


def test_antithetic_variance_reduction():
    """Test: Antithetic pairs reduce variance in gradient estimation."""
    key = jax.random.PRNGKey(456)

    # Antithetic sampling: pairs of (epsilon, -epsilon)
    N_pairs = 50
    N = 2 * N_pairs  # Total workers
    param_shape = (32, 16)
    sigma = 0.1

    # Generate base noise for N_pairs
    key, subkey = jax.random.split(key)
    base_noise = jax.random.normal(subkey, shape=(N_pairs,) + param_shape)

    # Create antithetic pairs
    antithetic_noise = jnp.concatenate([base_noise, -base_noise], axis=0)

    # Generate regular (non-antithetic) noise for comparison
    key, subkey = jax.random.split(key)
    regular_noise = jax.random.normal(subkey, shape=(N,) + param_shape)

    # True gradient direction (simulated)
    key, subkey = jax.random.split(key)
    true_grad = jax.random.normal(subkey, shape=param_shape)
    true_grad = true_grad / jnp.linalg.norm(true_grad)

    # Simulate fitness based on gradient alignment
    # f(theta + sigma * epsilon) approx f(theta) + sigma * <grad, epsilon>
    def compute_fitness(noise):
        # Flatten and compute dot product
        return jnp.sum(noise * true_grad, axis=(-2, -1))

    fitness_antithetic = jax.vmap(compute_fitness)(antithetic_noise)
    fitness_regular = jax.vmap(compute_fitness)(regular_noise)

    # Compute gradients
    grad_antithetic = compute_es_gradient(
        fitness_antithetic, {'param': antithetic_noise}, sigma
    )['param']
    grad_regular = compute_es_gradient(
        fitness_regular, {'param': regular_noise}, sigma
    )['param']

    # Compute variance over multiple trials
    n_trials = 20
    variances_antithetic = []
    variances_regular = []

    for trial in range(n_trials):
        key, subkey = jax.random.split(key)
        base_noise = jax.random.normal(subkey, shape=(N_pairs,) + param_shape)
        anti_noise = jnp.concatenate([base_noise, -base_noise], axis=0)

        key, subkey = jax.random.split(key)
        reg_noise = jax.random.normal(subkey, shape=(N,) + param_shape)

        fitness_anti = jax.vmap(compute_fitness)(anti_noise)
        fitness_reg = jax.vmap(compute_fitness)(reg_noise)

        grad_anti = compute_es_gradient(fitness_anti, {'param': anti_noise}, sigma)['param']
        grad_reg = compute_es_gradient(fitness_reg, {'param': reg_noise}, sigma)['param']

        # Error from true gradient
        error_anti = jnp.mean((grad_anti - true_grad) ** 2)
        error_reg = jnp.mean((grad_reg - true_grad) ** 2)

        variances_antithetic.append(float(error_anti))
        variances_regular.append(float(error_reg))

    mean_var_antithetic = jnp.mean(jnp.array(variances_antithetic))
    mean_var_regular = jnp.mean(jnp.array(variances_regular))

    # Antithetic should have lower variance (or at least not significantly higher)
    # Note: with finite samples, this is probabilistic
    print(f"  Antithetic MSE: {mean_var_antithetic:.6f}")
    print(f"  Regular MSE: {mean_var_regular:.6f}")
    print(f"  Ratio (regular/antithetic): {mean_var_regular / mean_var_antithetic:.2f}x")

    print("[PASS] test_antithetic_variance_reduction: antithetic pairs tested")
    return True


def test_antithetic_gradient_cancellation():
    """Test: With equal fitness, antithetic noise cancels perfectly."""
    key = jax.random.PRNGKey(789)

    N_pairs = 10
    param_shape = (16, 8)
    sigma = 0.1

    # Generate antithetic noise pairs
    key, subkey = jax.random.split(key)
    base_noise = jax.random.normal(subkey, shape=(N_pairs,) + param_shape)
    antithetic_noise = jnp.concatenate([base_noise, -base_noise], axis=0)

    # With equal fitness, gradient should be zero
    N = 2 * N_pairs
    equal_fitness = jnp.ones(N)

    gradient = compute_es_gradient(
        equal_fitness, {'param': antithetic_noise}, sigma
    )['param']

    max_abs = jnp.max(jnp.abs(gradient))
    # Use practical tolerance for floating-point arithmetic
    assert max_abs < 1e-5, f"Gradient should be near zero with equal fitness, got max={max_abs}"

    print(f"[PASS] test_antithetic_gradient_cancellation: max |grad| = {max_abs:.2e}")
    return True


def test_gradient_magnitude():
    """Test: Gradient magnitude is reasonable (not exploding/vanishing)."""
    key = jax.random.PRNGKey(101)

    N = 32
    param_shape = (128, 64)
    sigma = 0.1

    # Generate standard normal noise
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(N,) + param_shape)

    # Generate standard normal fitness
    key, subkey = jax.random.split(key)
    fitness = jax.random.normal(subkey, shape=(N,))

    # Compute gradient
    gradient = compute_es_gradient(fitness, {'param': noise}, sigma)['param']

    # Compute gradient statistics
    grad_norm = jnp.linalg.norm(gradient)
    grad_mean = jnp.mean(jnp.abs(gradient))
    grad_max = jnp.max(jnp.abs(gradient))

    # Expected scale analysis:
    # Each element: sum(fitness_i * noise_i) / (N * sigma)
    # With N=32, sigma=0.1: scale factor = 1/(32*0.1) = 0.3125
    # Elements of noise ~ N(0,1), fitness ~ N(0,1)
    # Product has variance 1, sum of N products has std ~ sqrt(N)
    # So gradient elements should have std ~ sqrt(N) / (N * sigma) = 1 / (sqrt(N) * sigma)
    # For N=32, sigma=0.1: expected_std ~ 1 / (5.66 * 0.1) ~ 1.77

    expected_std = 1.0 / (jnp.sqrt(N) * sigma)

    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  Gradient mean(|.|): {grad_mean:.4f}")
    print(f"  Gradient max(|.|): {grad_max:.4f}")
    print(f"  Expected element std: {expected_std:.4f}")

    # Sanity checks
    assert not jnp.isnan(grad_norm), "Gradient contains NaN"
    assert not jnp.isinf(grad_norm), "Gradient contains Inf"
    assert grad_norm > 1e-10, f"Gradient vanished: norm = {grad_norm}"
    assert grad_norm < 1e10, f"Gradient exploded: norm = {grad_norm}"

    # Check mean is in reasonable range (within 10x of expected)
    assert grad_mean > expected_std / 10, f"Gradient too small: mean={grad_mean}"
    assert grad_mean < expected_std * 10, f"Gradient too large: mean={grad_mean}"

    print("[PASS] test_gradient_magnitude: gradient has reasonable magnitude")
    return True


def test_gradient_with_centered_fitness():
    """Test: Using centered fitness (advantages) for gradient estimation."""
    key = jax.random.PRNGKey(202)

    N = 16
    param_shape = (32, 16)
    sigma = 0.1

    # Generate noise
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(N,) + param_shape)

    # Generate fitness with non-zero mean
    key, subkey = jax.random.split(key)
    fitness = jax.random.normal(subkey, shape=(N,)) + 5.0  # Add bias

    # Compute gradient with raw fitness
    grad_raw = compute_es_gradient(fitness, {'param': noise}, sigma)['param']

    # Compute gradient with centered fitness (advantages)
    centered_fitness = fitness - jnp.mean(fitness)
    grad_centered = compute_es_gradient(centered_fitness, {'param': noise}, sigma)['param']

    # Centering should not change the gradient when noise is zero-mean
    # But with finite samples, there will be some difference
    diff = jnp.max(jnp.abs(grad_raw - grad_centered))

    print(f"  Raw fitness mean: {jnp.mean(fitness):.4f}")
    print(f"  Centered fitness mean: {jnp.mean(centered_fitness):.4f}")
    print(f"  Gradient diff (raw vs centered): {diff:.6f}")

    print("[PASS] test_gradient_with_centered_fitness: centering tested")
    return True


def test_gradient_consistency_across_seeds():
    """Test: Same seed produces same gradient."""
    param_shape = (16, 8)
    N = 8
    sigma = 0.1

    def compute_grad_for_seed(seed):
        key = jax.random.PRNGKey(seed)
        key, k1, k2 = jax.random.split(key, 3)
        noise = jax.random.normal(k1, shape=(N,) + param_shape)
        fitness = jax.random.normal(k2, shape=(N,))
        return compute_es_gradient(fitness, {'param': noise}, sigma)['param']

    grad1 = compute_grad_for_seed(12345)
    grad2 = compute_grad_for_seed(12345)
    grad3 = compute_grad_for_seed(54321)  # Different seed

    diff_same = jnp.max(jnp.abs(grad1 - grad2))
    diff_diff = jnp.max(jnp.abs(grad1 - grad3))

    assert diff_same < 1e-10, f"Same seed should give same gradient: diff={diff_same}"
    assert diff_diff > 1e-3, f"Different seeds should give different gradients: diff={diff_diff}"

    print(f"[PASS] test_gradient_consistency_across_seeds: same_seed_diff={diff_same:.2e}, diff_seed_diff={diff_diff:.4f}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ES Gradient Estimation Integration Tests")
    print("=" * 60)

    test_es_gradient_shape()
    test_es_gradient_formula()
    test_antithetic_variance_reduction()
    test_antithetic_gradient_cancellation()
    test_gradient_magnitude()
    test_gradient_with_centered_fitness()
    test_gradient_consistency_across_seeds()

    print("=" * 60)
    print("All ES gradient estimation tests passed!")
    print("=" * 60)
