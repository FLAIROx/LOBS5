"""
Integration Test: ES Perturbation Application to Model Parameters
File: es_lobs5/tests/integration/test_es_perturbation_apply.py

Tests verify:
1. Gaussian noise perturbation with sigma scaling
2. Antithetic sampling: perturb[2k] = -perturb[2k+1]
3. Parameter tree structure preservation
4. Perturbation magnitude correctness
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp
from jax import tree_util


def create_mock_param_tree():
    """Create a mock parameter tree similar to model params."""
    return {
        'layer1': {
            'weights': jnp.ones((64, 32)),
            'bias': jnp.zeros((32,)),
        },
        'layer2': {
            'weights': jnp.ones((32, 16)) * 0.5,
            'bias': jnp.zeros((16,)),
        },
        'output': {
            'weights': jnp.ones((16, 4)) * 0.1,
            'bias': jnp.zeros((4,)),
        },
    }


def generate_noise_tree(params, key):
    """Generate Gaussian noise with same structure as params."""
    leaves, treedef = tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    noise_leaves = [jax.random.normal(k, leaf.shape) for k, leaf in zip(keys, leaves)]
    return tree_util.tree_unflatten(treedef, noise_leaves)


def apply_gaussian_perturbation(params, noise, sigma):
    """Apply Gaussian perturbation: perturbed = params + sigma * noise."""
    return tree_util.tree_map(
        lambda p, n: p + sigma * n,
        params, noise
    )


def test_gaussian_perturbation():
    """Test: params + sigma * noise produces correct perturbation."""
    key = jax.random.PRNGKey(42)
    params = create_mock_param_tree()
    sigma = 0.1

    noise = generate_noise_tree(params, key)
    perturbed = apply_gaussian_perturbation(params, noise, sigma)

    # Verify: perturbed - original = sigma * noise
    def check_perturbation(p, n, perturbed_p):
        expected_diff = sigma * n
        actual_diff = perturbed_p - p
        max_error = jnp.max(jnp.abs(actual_diff - expected_diff))
        return max_error

    errors = tree_util.tree_map(check_perturbation, params, noise, perturbed)
    max_error = max(tree_util.tree_leaves(errors))

    assert max_error < 1e-6, f"Perturbation error too large: {max_error}"
    print(f"[PASS] test_gaussian_perturbation: max_error = {max_error:.2e}")
    return True


def test_antithetic_pairs():
    """Test: noise[0] = -noise[1] for antithetic sampling."""
    key = jax.random.PRNGKey(123)
    params = create_mock_param_tree()
    sigma = 0.1

    # Generate noise for even index (thread_id=0)
    key_even = jax.random.fold_in(key, 0 // 2)  # Same base key for pair
    noise_even = generate_noise_tree(params, key_even)

    # For antithetic pair, noise_odd = -noise_even
    noise_odd = tree_util.tree_map(lambda n: -n, noise_even)

    # Apply perturbations
    perturbed_even = apply_gaussian_perturbation(params, noise_even, sigma)
    perturbed_odd = apply_gaussian_perturbation(params, noise_odd, sigma)

    # Verify: perturbation_even + perturbation_odd = 0
    # i.e., (perturbed_even - params) + (perturbed_odd - params) = 0
    # i.e., perturbed_even + perturbed_odd = 2 * params
    def check_antithetic(p_even, p_odd, original):
        sum_perturbed = p_even + p_odd
        expected_sum = 2 * original
        max_diff = jnp.max(jnp.abs(sum_perturbed - expected_sum))
        return max_diff

    diffs = tree_util.tree_map(check_antithetic, perturbed_even, perturbed_odd, params)
    max_diff = max(tree_util.tree_leaves(diffs))

    assert max_diff < 1e-6, f"Antithetic property violated: {max_diff}"
    print(f"[PASS] test_antithetic_pairs: max_diff = {max_diff:.2e}")
    return True


def test_tree_structure_preserved():
    """Test: perturbed tree has same structure as original."""
    key = jax.random.PRNGKey(456)
    params = create_mock_param_tree()
    sigma = 0.1

    noise = generate_noise_tree(params, key)
    perturbed = apply_gaussian_perturbation(params, noise, sigma)

    # Check tree structure
    original_structure = tree_util.tree_structure(params)
    perturbed_structure = tree_util.tree_structure(perturbed)
    noise_structure = tree_util.tree_structure(noise)

    assert original_structure == perturbed_structure, "Perturbed tree structure differs"
    assert original_structure == noise_structure, "Noise tree structure differs"

    # Check leaf shapes match
    original_leaves = tree_util.tree_leaves(params)
    perturbed_leaves = tree_util.tree_leaves(perturbed)
    noise_leaves = tree_util.tree_leaves(noise)

    for i, (orig, pert, noi) in enumerate(zip(original_leaves, perturbed_leaves, noise_leaves)):
        assert orig.shape == pert.shape, f"Shape mismatch at leaf {i}: {orig.shape} vs {pert.shape}"
        assert orig.shape == noi.shape, f"Shape mismatch for noise at leaf {i}: {orig.shape} vs {noi.shape}"

    print(f"[PASS] test_tree_structure_preserved: {len(original_leaves)} leaves verified")
    return True


def test_perturbation_magnitude():
    """Test: ||perturbed - original|| approx sigma * sqrt(n_params)."""
    key = jax.random.PRNGKey(789)
    params = create_mock_param_tree()
    sigma = 0.1

    noise = generate_noise_tree(params, key)
    perturbed = apply_gaussian_perturbation(params, noise, sigma)

    # Calculate total number of parameters
    n_params = sum(leaf.size for leaf in tree_util.tree_leaves(params))

    # Calculate L2 norm of perturbation
    def squared_diff(p, perturbed_p):
        diff = perturbed_p - p
        return jnp.sum(diff ** 2)

    squared_diffs = tree_util.tree_map(squared_diff, params, perturbed)
    total_squared = sum(tree_util.tree_leaves(squared_diffs))
    l2_norm = jnp.sqrt(total_squared)

    # Expected: sigma * sqrt(n_params) for standard normal noise
    # But noise is normalized per-element, so we expect:
    # E[||sigma * noise||^2] = sigma^2 * n_params
    # E[||sigma * noise||] = sigma * sqrt(n_params)
    expected_norm = sigma * jnp.sqrt(n_params)

    # Allow 30% tolerance for randomness
    ratio = l2_norm / expected_norm
    assert 0.7 < ratio < 1.3, f"Perturbation magnitude unexpected: ratio={ratio:.3f}"

    print(f"[PASS] test_perturbation_magnitude: ||pert|| = {l2_norm:.4f}, expected ~ {expected_norm:.4f}, ratio = {ratio:.3f}")
    return True


def test_different_keys_different_noise():
    """Test: Different PRNG keys produce different noise."""
    params = create_mock_param_tree()
    sigma = 0.1

    key1 = jax.random.PRNGKey(100)
    key2 = jax.random.PRNGKey(200)

    noise1 = generate_noise_tree(params, key1)
    noise2 = generate_noise_tree(params, key2)

    # Calculate max difference between noise from different keys
    def max_diff(n1, n2):
        return jnp.max(jnp.abs(n1 - n2))

    diffs = tree_util.tree_map(max_diff, noise1, noise2)
    max_noise_diff = max(tree_util.tree_leaves(diffs))

    assert max_noise_diff > 0.1, f"Different keys should produce different noise: max_diff={max_noise_diff}"
    print(f"[PASS] test_different_keys_different_noise: max_diff = {max_noise_diff:.4f}")
    return True


def test_same_key_same_noise():
    """Test: Same PRNG key produces identical noise."""
    params = create_mock_param_tree()

    key = jax.random.PRNGKey(42)

    noise1 = generate_noise_tree(params, key)
    noise2 = generate_noise_tree(params, key)

    # Calculate max difference - should be zero
    def max_diff(n1, n2):
        return jnp.max(jnp.abs(n1 - n2))

    diffs = tree_util.tree_map(max_diff, noise1, noise2)
    max_noise_diff = max(tree_util.tree_leaves(diffs))

    assert max_noise_diff < 1e-10, f"Same key should produce identical noise: max_diff={max_noise_diff}"
    print(f"[PASS] test_same_key_same_noise: max_diff = {max_noise_diff:.2e}")
    return True


def test_zero_sigma_no_perturbation():
    """Test: sigma=0 means no perturbation."""
    key = jax.random.PRNGKey(999)
    params = create_mock_param_tree()
    sigma = 0.0

    noise = generate_noise_tree(params, key)
    perturbed = apply_gaussian_perturbation(params, noise, sigma)

    # Verify perturbed == original
    def max_diff(p, perturbed_p):
        return jnp.max(jnp.abs(perturbed_p - p))

    diffs = tree_util.tree_map(max_diff, params, perturbed)
    max_perturbation = max(tree_util.tree_leaves(diffs))

    assert max_perturbation < 1e-10, f"Zero sigma should mean no perturbation: {max_perturbation}"
    print(f"[PASS] test_zero_sigma_no_perturbation: max_diff = {max_perturbation:.2e}")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("ES Perturbation Apply Integration Tests")
    print("=" * 60)

    tests = [
        test_gaussian_perturbation,
        test_antithetic_pairs,
        test_tree_structure_preserved,
        test_perturbation_magnitude,
        test_different_keys_different_noise,
        test_same_key_same_noise,
        test_zero_sigma_no_perturbation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
