"""Integration test for ES fitness computation from rollout results.

This test verifies the complete fitness computation pipeline:
1. Mock rollout results (PnL, positions, inventory)
2. Fitness function computation
3. Fitness normalization (z-score: mean=0, std=1)
4. Fitness advantage computation
5. Gradient differentiability through fitness

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_es_fitness_compute.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp


# ============================================================================
# Fitness computation utilities (mimicking ES pipeline)
# ============================================================================

def compute_pnl_fitness(total_revenue: jnp.ndarray) -> jnp.ndarray:
    """Compute fitness from PnL values (identity for simple case)."""
    return total_revenue


def zscore_normalize(fitness: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Z-score normalization: (x - mean) / sqrt(var + eps).

    Args:
        fitness: Array of fitness scores, shape (n_workers,)
        eps: Small constant for numerical stability

    Returns:
        Normalized fitness with mean ~0 and std ~1
    """
    mean = jnp.mean(fitness)
    var = jnp.var(fitness)
    return (fitness - mean) / jnp.sqrt(var + eps)


def compute_fitness_advantage(fitness: jnp.ndarray) -> jnp.ndarray:
    """Compute advantage: normalized fitness centered at mean.

    In ES, advantage = (fitness - mean) / std
    This is equivalent to z-score normalization.

    Args:
        fitness: Raw fitness scores, shape (n_workers,)

    Returns:
        Advantage scores with mean=0 and std=1
    """
    mean = jnp.mean(fitness)
    std = jnp.std(fitness) + 1e-8  # Numerical stability
    return (fitness - mean) / std


def create_mock_rollout_results(
    key: jax.random.PRNGKey,
    n_workers: int,
    base_pnl: float = 1000.0,
    pnl_std: float = 100.0
) -> dict:
    """Create mock rollout results simulating ES evaluation.

    Args:
        key: JAX random key
        n_workers: Number of ES workers/perturbations
        base_pnl: Base PnL around which variations occur
        pnl_std: Standard deviation of PnL variations

    Returns:
        Dictionary with mock rollout data:
        - pnl: (n_workers,) total PnL per worker
        - positions: (n_workers,) final positions
        - inventory: (n_workers,) remaining inventory
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Generate PnL with some variation across workers
    pnl = base_pnl + pnl_std * jax.random.normal(k1, (n_workers,))

    # Generate positions (can be positive or negative)
    positions = jax.random.randint(k2, (n_workers,), -50, 51).astype(jnp.float32)

    # Generate inventory (remaining shares to execute)
    inventory = jax.random.randint(k3, (n_workers,), 0, 100).astype(jnp.float32)

    return {
        'pnl': pnl,
        'positions': positions,
        'inventory': inventory,
    }


# ============================================================================
# Integration Tests
# ============================================================================

def test_pnl_to_fitness():
    """Test: PnL values are correctly converted to fitness scores.

    In ES-JaxLOB, fitness = execution quality = total PnL.
    This test verifies the basic PnL -> fitness mapping.
    """
    print("test_pnl_to_fitness...")

    key = jax.random.PRNGKey(42)
    n_workers = 16

    # Create mock rollout results
    rollout = create_mock_rollout_results(key, n_workers)
    pnl = rollout['pnl']

    # Compute fitness from PnL
    fitness = compute_pnl_fitness(pnl)

    # Fitness should equal PnL for basic fitness function
    assert jnp.allclose(fitness, pnl), "Fitness should equal PnL"

    # Fitness should preserve ordering
    pnl_argsort = jnp.argsort(pnl)
    fitness_argsort = jnp.argsort(fitness)
    assert jnp.array_equal(pnl_argsort, fitness_argsort), "Fitness ordering should match PnL ordering"

    # Test with specific values
    specific_pnl = jnp.array([100.0, 200.0, 150.0, 50.0])
    specific_fitness = compute_pnl_fitness(specific_pnl)
    assert jnp.allclose(specific_fitness, specific_pnl), "Specific PnL values should map directly to fitness"

    print(f"  n_workers={n_workers}")
    print(f"  PnL range: [{float(pnl.min()):.2f}, {float(pnl.max()):.2f}]")
    print(f"  Fitness range: [{float(fitness.min()):.2f}, {float(fitness.max()):.2f}]")
    print("[PASS] test_pnl_to_fitness")
    return True


def test_fitness_normalization():
    """Test: Normalized fitness has mean=0 and std=1.

    Z-score normalization: normalized = (x - mean) / sqrt(var + eps)
    After normalization:
    - Mean should be ~0 (within numerical tolerance)
    - Std should be ~1 (when variance >> eps)
    """
    print("test_fitness_normalization...")

    key = jax.random.PRNGKey(123)

    test_cases = [
        # (n_workers, base_pnl, pnl_std, description)
        (16, 1000.0, 100.0, "typical ES batch"),
        (64, 5000.0, 500.0, "larger batch with higher variance"),
        (8, 100.0, 10.0, "small batch"),
        (128, 0.0, 1000.0, "zero mean, high variance"),
    ]

    for n_workers, base_pnl, pnl_std, desc in test_cases:
        key, subkey = jax.random.split(key)
        rollout = create_mock_rollout_results(subkey, n_workers, base_pnl, pnl_std)
        fitness = compute_pnl_fitness(rollout['pnl'])

        # Normalize fitness
        normalized = zscore_normalize(fitness)

        # Check mean ~= 0
        mean = jnp.mean(normalized)
        assert jnp.abs(mean) < 1e-5, f"{desc}: mean={mean:.6f}, expected ~0"

        # Check std ~= 1 (variance should be close to 1)
        std = jnp.std(normalized)
        assert jnp.abs(std - 1.0) < 0.1, f"{desc}: std={std:.6f}, expected ~1"

        print(f"  {desc}: mean={float(mean):.6f}, std={float(std):.6f}")

    print("[PASS] test_fitness_normalization")
    return True


def test_fitness_advantage():
    """Test: Advantage = (fitness - mean) / std.

    The fitness advantage is used in ES gradient estimation:
    grad ~ sum(advantage_i * perturbation_i)

    Properties:
    - Mean of advantages = 0
    - Std of advantages = 1
    - Positive advantage = better than average
    - Negative advantage = worse than average
    """
    print("test_fitness_advantage...")

    key = jax.random.PRNGKey(456)
    n_workers = 32

    rollout = create_mock_rollout_results(key, n_workers)
    fitness = compute_pnl_fitness(rollout['pnl'])

    # Compute advantage
    advantage = compute_fitness_advantage(fitness)

    # Check mean = 0
    mean = jnp.mean(advantage)
    assert jnp.abs(mean) < 1e-5, f"Advantage mean={mean:.6f}, expected ~0"

    # Check std = 1
    std = jnp.std(advantage)
    assert jnp.abs(std - 1.0) < 0.1, f"Advantage std={std:.6f}, expected ~1"

    # Check that advantage preserves relative ordering
    fitness_argsort = jnp.argsort(fitness)
    advantage_argsort = jnp.argsort(advantage)
    assert jnp.array_equal(fitness_argsort, advantage_argsort), "Advantage should preserve fitness ordering"

    # Check positive/negative advantage alignment with above/below mean
    fitness_mean = jnp.mean(fitness)
    above_mean_mask = fitness > fitness_mean
    positive_advantage_mask = advantage > 0

    # Most above-mean fitness should have positive advantage (allow some tolerance for ties)
    agreement = jnp.mean((above_mean_mask == positive_advantage_mask).astype(jnp.float32))
    assert agreement > 0.9, f"Advantage sign agreement with above-mean={agreement:.2f}, expected >0.9"

    print(f"  n_workers={n_workers}")
    print(f"  advantage mean={float(mean):.6f}, std={float(std):.6f}")
    print(f"  sign agreement with above-mean: {float(agreement):.2%}")
    print("[PASS] test_fitness_advantage")
    return True


def test_fitness_shape():
    """Test: Fitness has shape (n_workers,).

    In ES, each worker evaluates one perturbation and produces a single scalar fitness.
    The fitness array should have shape (n_workers,) - one scalar per worker.
    """
    print("test_fitness_shape...")

    key = jax.random.PRNGKey(789)

    test_cases = [
        8,    # Small batch
        16,   # Typical batch
        64,   # Larger batch
        128,  # Large batch
    ]

    for n_workers in test_cases:
        key, subkey = jax.random.split(key)
        rollout = create_mock_rollout_results(subkey, n_workers)

        # Check raw fitness shape
        fitness = compute_pnl_fitness(rollout['pnl'])
        assert fitness.shape == (n_workers,), f"Expected shape ({n_workers},), got {fitness.shape}"

        # Check normalized fitness shape
        normalized = zscore_normalize(fitness)
        assert normalized.shape == (n_workers,), f"Normalized shape mismatch: {normalized.shape}"

        # Check advantage shape
        advantage = compute_fitness_advantage(fitness)
        assert advantage.shape == (n_workers,), f"Advantage shape mismatch: {advantage.shape}"

        # Verify each element is scalar-like (0-dimensional when indexed)
        for i in range(n_workers):
            assert fitness[i].shape == (), f"Fitness[{i}] should be scalar"
            assert normalized[i].shape == (), f"Normalized[{i}] should be scalar"
            assert advantage[i].shape == (), f"Advantage[{i}] should be scalar"

        print(f"  n_workers={n_workers}: fitness.shape={fitness.shape} [OK]")

    print("[PASS] test_fitness_shape")
    return True


def test_fitness_differentiable():
    """Test: Can compute gradients through fitness computation.

    In ES, we don't typically backprop through fitness, but the fitness function
    should still be differentiable for potential hybrid approaches or debugging.

    This test verifies:
    1. Fitness function is JAX-traceable
    2. Gradients can be computed through normalization
    3. Gradients are finite and reasonable
    """
    print("test_fitness_differentiable...")

    key = jax.random.PRNGKey(101)
    n_workers = 16

    # Create a function that takes raw PnL and returns normalized fitness
    def fitness_pipeline(pnl: jnp.ndarray) -> jnp.ndarray:
        """Complete fitness computation pipeline."""
        fitness = compute_pnl_fitness(pnl)
        normalized = zscore_normalize(fitness)
        return normalized

    # Create a scalar loss from fitness (e.g., sum of squares)
    def fitness_loss(pnl: jnp.ndarray) -> jnp.ndarray:
        """Scalar loss from fitness for gradient testing."""
        normalized = fitness_pipeline(pnl)
        # Use sum of squares as a differentiable scalar
        return jnp.sum(normalized ** 2)

    # Generate test PnL
    pnl = 1000.0 + 100.0 * jax.random.normal(key, (n_workers,))

    # Test 1: Compute gradients
    grad_fn = jax.grad(fitness_loss)
    gradients = grad_fn(pnl)

    # Check gradient shape matches input shape
    assert gradients.shape == pnl.shape, f"Gradient shape mismatch: {gradients.shape} vs {pnl.shape}"

    # Check gradients are finite
    assert jnp.all(jnp.isfinite(gradients)), "Gradients should be finite"

    # Check gradients are not all zeros
    assert jnp.any(gradients != 0), "Gradients should not all be zero"

    # Test 2: JIT compilation works
    jit_grad_fn = jax.jit(grad_fn)
    jit_gradients = jit_grad_fn(pnl)
    assert jnp.allclose(gradients, jit_gradients, rtol=1e-5), "JIT gradients should match eager gradients"

    # Test 3: Value and grad work together
    value_and_grad_fn = jax.value_and_grad(fitness_loss)
    value, grads = value_and_grad_fn(pnl)

    assert jnp.isfinite(value), "Loss value should be finite"
    assert jnp.allclose(grads, gradients, rtol=1e-5), "Value-and-grad should match grad-only"

    # Test 4: Jacobian of fitness pipeline
    jacobian_fn = jax.jacobian(fitness_pipeline)
    jacobian = jacobian_fn(pnl)

    # Jacobian should be (n_workers, n_workers) - each output depends on all inputs
    assert jacobian.shape == (n_workers, n_workers), f"Jacobian shape: {jacobian.shape}"
    assert jnp.all(jnp.isfinite(jacobian)), "Jacobian should be finite"

    print(f"  n_workers={n_workers}")
    print(f"  gradient norm: {float(jnp.linalg.norm(gradients)):.6f}")
    print(f"  jacobian shape: {jacobian.shape}")
    print(f"  loss value: {float(value):.6f}")
    print("[PASS] test_fitness_differentiable")
    return True


def test_fitness_with_inventory_penalty():
    """Test: Fitness with inventory penalty for incomplete execution.

    In trading, remaining inventory is typically penalized because:
    - Unfilled orders mean missed execution opportunities
    - Inventory risk from holding positions overnight

    Fitness = PnL - penalty * remaining_inventory
    """
    print("test_fitness_with_inventory_penalty...")

    key = jax.random.PRNGKey(202)
    n_workers = 16
    penalty_weight = 10.0  # Penalty per unit of remaining inventory

    rollout = create_mock_rollout_results(key, n_workers)
    pnl = rollout['pnl']
    inventory = rollout['inventory']

    # Compute fitness with inventory penalty
    def fitness_with_penalty(pnl, inventory, penalty_weight):
        base_fitness = compute_pnl_fitness(pnl)
        penalty = penalty_weight * inventory
        return base_fitness - penalty

    fitness = fitness_with_penalty(pnl, inventory, penalty_weight)

    # Check shape
    assert fitness.shape == (n_workers,), f"Fitness shape mismatch: {fitness.shape}"

    # Check that penalty reduces fitness
    fitness_no_penalty = compute_pnl_fitness(pnl)
    expected_penalty = penalty_weight * inventory
    expected_fitness = fitness_no_penalty - expected_penalty

    assert jnp.allclose(fitness, expected_fitness), "Fitness should equal PnL - penalty"

    # Workers with zero inventory should have same fitness
    zero_inventory_mask = inventory == 0
    if jnp.any(zero_inventory_mask):
        zero_inv_fitness = fitness[zero_inventory_mask]
        zero_inv_pnl = pnl[zero_inventory_mask]
        assert jnp.allclose(zero_inv_fitness, zero_inv_pnl), "Zero inventory workers should have fitness = PnL"

    print(f"  n_workers={n_workers}")
    print(f"  penalty_weight={penalty_weight}")
    print(f"  avg inventory: {float(jnp.mean(inventory)):.2f}")
    print(f"  avg penalty: {float(jnp.mean(expected_penalty)):.2f}")
    print("[PASS] test_fitness_with_inventory_penalty")
    return True


def test_fitness_batch_consistency():
    """Test: Fitness computation is consistent across batch processing.

    Verifies that computing fitness for the whole batch at once produces
    the same results as computing individually (vectorization correctness).
    """
    print("test_fitness_batch_consistency...")

    key = jax.random.PRNGKey(303)
    n_workers = 32

    rollout = create_mock_rollout_results(key, n_workers)
    pnl = rollout['pnl']

    # Batch computation
    batch_fitness = compute_pnl_fitness(pnl)
    batch_normalized = zscore_normalize(batch_fitness)

    # Individual computation (simulate per-worker processing)
    individual_fitness = jnp.array([compute_pnl_fitness(pnl[i]) for i in range(n_workers)])

    # Fitness computation should be consistent
    assert jnp.allclose(batch_fitness, individual_fitness), "Batch vs individual fitness mismatch"

    # Note: Normalization requires global statistics, so it's inherently batch-dependent
    # But if we normalize the same batch, results should match
    individual_normalized = zscore_normalize(individual_fitness)
    assert jnp.allclose(batch_normalized, individual_normalized), "Normalized fitness should match"

    # Verify vmap produces same results
    vmap_fitness = jax.vmap(lambda x: compute_pnl_fitness(x))(pnl)
    assert jnp.allclose(batch_fitness, vmap_fitness), "vmap fitness should match batch"

    print(f"  n_workers={n_workers}")
    print(f"  batch-individual max diff: {float(jnp.max(jnp.abs(batch_fitness - individual_fitness))):.2e}")
    print("[PASS] test_fitness_batch_consistency")
    return True


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("ES Fitness Computation Integration Tests")
    print("=" * 60)

    tests = [
        test_pnl_to_fitness,
        test_fitness_normalization,
        test_fitness_advantage,
        test_fitness_shape,
        test_fitness_differentiable,
        test_fitness_with_inventory_penalty,
        test_fitness_batch_consistency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
