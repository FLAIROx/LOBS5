"""Test fitness normalization.

This module tests z-score normalization used in Evolution Strategies.
It does not depend on HyperscaleES or external ES frameworks.
"""
import jax
import jax.numpy as jnp


def zscore_normalize(x):
    """Z-score normalization with numerical stability.

    Args:
        x: Input array of fitness scores.

    Returns:
        Normalized array with mean ~0 and std ~1 (when var >> 1e-5).
    """
    return (x - jnp.mean(x)) / jnp.sqrt(jnp.var(x) + 1e-5)


def test_zscore_normalization():
    """Test: normalized fitness mean ~ 0, std ~ 1.

    The zscore_normalize function performs z-score normalization:
        normalized = (raw - mean) / sqrt(var + 1e-5)

    After normalization, the output should have:
        - mean ~ 0
        - std ~ 1 (when var >> 1e-5)
    """
    key = jax.random.PRNGKey(42)

    # Test with various raw fitness distributions
    # Note: The formula uses sqrt(var + 1e-5) for numerical stability
    # So std ~= 1 only when var >> 1e-5
    test_cases = [
        # (raw_scores, check_std) - check_std=False for small variance cases
        (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), True),              # simple ascending
        (jnp.array([-10.0, 0.0, 10.0, 20.0]), True),               # with negatives
        (jnp.array([100.0, 200.0, 300.0, 400.0, 500.0]), True),    # large scale
        (jnp.array([0.001, 0.002, 0.003, 0.004]), False),          # small scale (var << 1e-5)
        (jax.random.normal(key, (128,)), True),                    # random normal
        (jax.random.uniform(key, (64,)) * 100 - 50, True),         # random uniform
    ]

    for i, (raw_scores, check_std) in enumerate(test_cases):
        normalized = zscore_normalize(raw_scores)

        mean = jnp.mean(normalized)
        std = jnp.std(normalized)
        raw_var = jnp.var(raw_scores)

        # mean should be approximately 0
        assert jnp.abs(mean) < 1e-5, f"Case {i}: mean={mean:.6f}, expected ~0"

        # std should be approximately 1 (only when variance >> 1e-5)
        if check_std:
            assert jnp.abs(std - 1.0) < 0.1, f"Case {i}: std={std:.6f}, expected ~1"

        print(f"  Case {i}: mean={mean:.6f}, std={std:.6f}, raw_var={raw_var:.2e}")

    print("[PASS] test_zscore_normalization: mean ~ 0, std ~ 1")
    return True


def test_gradient_estimation():
    """Test: ES gradient approximates sum(f_i * epsilon_i) / sigma.

    In Evolution Strategies, the gradient is estimated as:
        grad ~ (1 / n*sigma) * sum(f_i * epsilon_i)

    Where:
        - f_i is the normalized fitness for perturbation i
        - epsilon_i is the noise perturbation for thread i
        - sigma is the noise standard deviation

    This test verifies the gradient estimation by:
    1. Creating antithetic perturbations (+epsilon, -epsilon)
    2. Assigning fitness proportional to perturbation direction
    3. Verifying that the gradient points in the correct direction
    """
    key = jax.random.PRNGKey(123)

    # Simple 2D parameter for easy verification
    in_dim, out_dim = 4, 4
    param = jax.random.normal(key, (out_dim, in_dim)) * 0.1

    sigma = 0.1
    n_threads = 8  # Must be even for antithetic pairs
    rank = 4

    # Generate a base key for perturbations
    base_key = jax.random.PRNGKey(999)

    # Generate LORA perturbations (A, B pairs) for each thread
    # Mimicking the EggRoll noiser's perturbation generation
    epoch = 0
    perturbations = []
    for tid in range(n_threads):
        # Mimic the LORA perturbation generation from eggroll.py
        true_thread_idx = tid // 2
        sign = 1.0 if tid % 2 == 0 else -1.0

        a, b = param.shape
        lora_key = jax.random.fold_in(jax.random.fold_in(base_key, epoch), true_thread_idx)
        lora_params = jax.random.normal(lora_key, (a + b, rank), dtype=param.dtype)
        B = lora_params[:b]  # b x r
        A = lora_params[b:]  # a x r

        # The effective perturbation is A @ B.T scaled by sigma/sqrt(rank) * sign
        perturbation = (A @ B.T) * (sigma / jnp.sqrt(rank)) * sign
        perturbations.append(perturbation)

    perturbations = jnp.stack(perturbations, axis=0)  # (n_threads, out_dim, in_dim)

    # Create raw fitnesses: higher fitness for positive perturbation direction
    # This simulates a simple optimization landscape where we want param to increase
    direction = jnp.ones_like(param)
    raw_fitnesses = jnp.array([
        jnp.sum(perturbations[i] * direction) for i in range(n_threads)
    ])

    # Normalize fitnesses using z-score
    normalized_fitnesses = zscore_normalize(raw_fitnesses)

    # Verify antithetic property: paired threads should have opposite normalized fitness
    for pair_idx in range(n_threads // 2):
        even_idx = pair_idx * 2
        odd_idx = pair_idx * 2 + 1

        # The perturbations are opposite, so if raw fitness is proportional to perturbation,
        # the normalized fitnesses should be opposite too
        f_even = normalized_fitnesses[even_idx]
        f_odd = normalized_fitnesses[odd_idx]

        # For antithetic pairs with fitness = sum(perturbation * direction):
        # f_even and f_odd should have opposite signs (approximately)
        # because epsilon_odd = -epsilon_even
        assert jnp.sign(f_even) != jnp.sign(f_odd) or jnp.abs(f_even - f_odd) < 0.1, \
            f"Pair {pair_idx}: f_even={f_even:.4f}, f_odd={f_odd:.4f} should be opposite"

    # Compute the gradient estimate: weighted sum of perturbations
    # grad ~ (1/n) * sum(f_i * epsilon_i / sigma^2)
    # For EGGROLL: grad ~ mean(f_i * epsilon_i)
    weighted_perturbations = jnp.einsum('n,nij->ij', normalized_fitnesses, perturbations)
    grad_estimate = weighted_perturbations / n_threads

    # The gradient should point in the positive direction (same as target direction)
    # because we assigned higher fitness to perturbations aligned with direction
    grad_alignment = jnp.sum(grad_estimate * direction)

    # Gradient should be positively aligned with target direction
    assert grad_alignment > 0, \
        f"Gradient should align with target direction, got alignment={grad_alignment:.6f}"

    print(f"  n_threads={n_threads}, sigma={sigma}")
    print(f"  raw_fitness range: [{float(jnp.min(raw_fitnesses)):.4f}, {float(jnp.max(raw_fitnesses)):.4f}]")
    print(f"  normalized mean: {float(jnp.mean(normalized_fitnesses)):.6f}")
    print(f"  gradient alignment: {float(grad_alignment):.6f}")

    print("[PASS] test_gradient_estimation: ES gradient ~ sum(f_i * epsilon_i) / sigma")
    return True


if __name__ == "__main__":
    test_zscore_normalization()
    test_gradient_estimation()
    print("All fitness normalization tests passed!")
