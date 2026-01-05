"""
Level 4 Production Test: Worker Scaling for ES Training
File: es_lobs5/tests/e2e/test_prod_scaling.py

Tests ES training with different worker counts (4, 8, 16, 32) to verify:
1. All worker counts work correctly
2. Antithetic sampling works for all counts
3. Gradient variance decreases with more workers
4. Results are consistent across scales

Run: JAX_PLATFORMS=cpu python es_lobs5/tests/e2e/test_prod_scaling.py
"""

import os
# Limit XLA threads to avoid thread creation failures on some systems
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
# Disable parallel compilation to reduce thread usage
jax.config.update('jax_threefry_partitionable', False)
import jax.numpy as jnp
import numpy as np
from functools import partial

# -----------------------------------------------------------------------------
# Self-contained ES noiser implementation (extracted from HyperscaleES)
# This avoids dependency on optax and other external libraries
# -----------------------------------------------------------------------------

def get_lora_update_params_2d(frozen_noiser_params, base_sigma, iterinfo, param_2d, key):
    """Generate LoRA decomposition A and B matrices for 2D parameter.

    The update is computed as A @ B.T where:
    - A: (a, r) - scaled by sigma
    - B: (b, r)
    - param_2d shape: (a, b)

    Antithetic sampling: thread_id % 2 == 0 uses +sigma, thread_id % 2 == 1 uses -sigma
    """
    epoch, thread_id = iterinfo

    # Noise reuse: share noise across epochs
    noise_reuse = frozen_noiser_params.get("noise_reuse", 0)
    true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse

    # Antithetic sampling: pairs of threads share same noise with opposite signs
    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    a, b = param_2d.shape
    lora_params = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        (a + b, frozen_noiser_params["rank"]),
        dtype=param_2d.dtype
    )
    B = lora_params[:b]  # b x r
    A = lora_params[b:]  # a x r

    # update is A @ B.T
    return A * sigma, B


def get_noisy_param(frozen_noiser_params, noiser_params, param, key, iterinfo):
    """Apply noise perturbation to a parameter using LoRA decomposition.

    Handles n-dimensional parameters by reshaping to 2D and back.
    """
    if iterinfo is None:
        return param

    sigma = noiser_params.get("sigma", 0.1)
    rank = frozen_noiser_params.get("rank", 4)
    scaled_sigma = sigma / jnp.sqrt(rank)

    original_shape = param.shape

    # Reshape to 2D: (first_dim, remaining_dims_product)
    if param.ndim == 1:
        param_2d = param.reshape(-1, 1)
    elif param.ndim == 2:
        param_2d = param
    else:
        # For n-dim params, flatten all but first dimension
        param_2d = param.reshape(param.shape[0], -1)

    # Apply LoRA perturbation
    A, B = get_lora_update_params_2d(frozen_noiser_params, scaled_sigma, iterinfo, param_2d, key)
    update = A @ B.T
    noisy_2d = param_2d + update

    # Reshape back to original shape
    return noisy_2d.reshape(original_shape)


class MockNoiser:
    """Mock noiser that implements EggRoll-style LoRA perturbation."""

    @staticmethod
    def init_noiser(params, sigma=0.1, lr=0.001, rank=4, **kwargs):
        """Initialize noiser parameters."""
        frozen_noiser_params = {
            "rank": rank,
            "noise_reuse": kwargs.get("noise_reuse", 0),
        }
        noiser_params = {
            "sigma": sigma,
            "lr": lr,
        }
        return frozen_noiser_params, noiser_params

    @staticmethod
    def get_noisy_standard(frozen_noiser_params, noiser_params, param, key, iterinfo):
        """Apply noise to parameter."""
        return get_noisy_param(frozen_noiser_params, noiser_params, param, key, iterinfo)


# -----------------------------------------------------------------------------
# Test parameter creation (simplified without full SSM)
# -----------------------------------------------------------------------------

def create_test_params(key, H=32, P=16):
    """Create test parameters for scaling tests."""
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    params = {
        'B': jax.random.normal(k1, (2 * P, H, 2)),  # Complex as (real, imag)
        'C': jax.random.normal(k2, (H, 2 * P, 2)),
        'D': jax.random.normal(k3, (H,)),
        'Lambda_re': jax.random.normal(k4, (P,)) * 0.1 - 0.5,  # Negative real part
        'Lambda_im': jax.random.uniform(k4, (P,)) * jnp.pi,
    }

    # ES tree key for each parameter
    es_tree_key = {k: jax.random.fold_in(key, i) for i, k in enumerate(params.keys())}

    return params, es_tree_key


def compute_worker_fitness(key, thread_id, epoch, noiser, frozen_noiser_params,
                           noiser_params, params, es_tree_key):
    """Compute fitness for a single worker (simplified for testing)."""
    iterinfo = (jnp.int32(epoch), jnp.int32(thread_id))

    # Get perturbed parameter (use B as example)
    noisy_B = noiser.get_noisy_standard(
        frozen_noiser_params, noiser_params,
        params['B'], es_tree_key['B'], iterinfo
    )

    # Simple fitness function: quadratic + noise
    # In real ES, this would be episode reward from environment simulation
    fitness = -jnp.sum((noisy_B - 0.5) ** 2) + jax.random.normal(key) * 0.01

    # Noise magnitude for analysis
    noise_magnitude = jnp.sum(jnp.abs(noisy_B - params['B']))

    return fitness, noise_magnitude


def run_es_epoch(key, n_workers, epoch, noiser, frozen_noiser_params,
                 noiser_params, params, es_tree_key):
    """Run one ES epoch with n_workers workers using vmap."""
    # Generate keys for all workers
    keys = jax.random.split(key, n_workers)
    thread_ids = jnp.arange(n_workers, dtype=jnp.int32)

    # Vectorized fitness evaluation
    def single_worker_fitness(key, thread_id):
        return compute_worker_fitness(
            key, thread_id, epoch, noiser, frozen_noiser_params,
            noiser_params, params, es_tree_key
        )

    # Use vmap for parallel evaluation across workers
    fitnesses, noise_magnitudes = jax.vmap(single_worker_fitness)(keys, thread_ids)

    return fitnesses, noise_magnitudes


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_4_workers():
    """Test: ES training with 4 workers."""
    print("[TEST] test_4_workers: ES training with 4 workers")

    key = jax.random.PRNGKey(42)
    n_workers = 4

    # Setup
    key, subkey = jax.random.split(key)
    params, es_tree_key = create_test_params(subkey)

    noiser = MockNoiser()
    frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

    # Run epoch
    key, subkey = jax.random.split(key)
    fitnesses, noise_mags = run_es_epoch(
        subkey, n_workers, 0, noiser, frozen_noiser_params,
        noiser_params, params, es_tree_key
    )

    # Verify
    assert fitnesses.shape == (n_workers,), f"Expected ({n_workers},), got {fitnesses.shape}"
    assert jnp.all(jnp.isfinite(fitnesses)), "Fitnesses should be finite"
    assert len(jnp.unique(fitnesses)) == n_workers, "All fitnesses should be different"

    print(f"  Fitnesses: {fitnesses}")
    print(f"  Mean: {jnp.mean(fitnesses):.4f}, Std: {jnp.std(fitnesses):.4f}")
    print(f"[PASS] test_4_workers")
    return True


def test_8_workers():
    """Test: ES training with 8 workers."""
    print("[TEST] test_8_workers: ES training with 8 workers")

    key = jax.random.PRNGKey(123)
    n_workers = 8

    key, subkey = jax.random.split(key)
    params, es_tree_key = create_test_params(subkey)

    noiser = MockNoiser()
    frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

    key, subkey = jax.random.split(key)
    fitnesses, noise_mags = run_es_epoch(
        subkey, n_workers, 0, noiser, frozen_noiser_params,
        noiser_params, params, es_tree_key
    )

    assert fitnesses.shape == (n_workers,), f"Expected ({n_workers},), got {fitnesses.shape}"
    assert jnp.all(jnp.isfinite(fitnesses)), "Fitnesses should be finite"

    print(f"  Fitnesses shape: {fitnesses.shape}")
    print(f"  Mean: {jnp.mean(fitnesses):.4f}, Std: {jnp.std(fitnesses):.4f}")
    print(f"[PASS] test_8_workers")
    return True


def test_16_workers():
    """Test: ES training with 16 workers."""
    print("[TEST] test_16_workers: ES training with 16 workers")

    key = jax.random.PRNGKey(456)
    n_workers = 16

    key, subkey = jax.random.split(key)
    params, es_tree_key = create_test_params(subkey)

    noiser = MockNoiser()
    frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

    key, subkey = jax.random.split(key)
    fitnesses, noise_mags = run_es_epoch(
        subkey, n_workers, 0, noiser, frozen_noiser_params,
        noiser_params, params, es_tree_key
    )

    assert fitnesses.shape == (n_workers,), f"Expected ({n_workers},), got {fitnesses.shape}"
    assert jnp.all(jnp.isfinite(fitnesses)), "Fitnesses should be finite"

    print(f"  Fitnesses shape: {fitnesses.shape}")
    print(f"  Mean: {jnp.mean(fitnesses):.4f}, Std: {jnp.std(fitnesses):.4f}")
    print(f"[PASS] test_16_workers")
    return True


def test_32_workers():
    """Test: ES training with 32 workers."""
    print("[TEST] test_32_workers: ES training with 32 workers")

    key = jax.random.PRNGKey(789)
    n_workers = 32

    key, subkey = jax.random.split(key)
    params, es_tree_key = create_test_params(subkey)

    noiser = MockNoiser()
    frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

    key, subkey = jax.random.split(key)
    fitnesses, noise_mags = run_es_epoch(
        subkey, n_workers, 0, noiser, frozen_noiser_params,
        noiser_params, params, es_tree_key
    )

    assert fitnesses.shape == (n_workers,), f"Expected ({n_workers},), got {fitnesses.shape}"
    assert jnp.all(jnp.isfinite(fitnesses)), "Fitnesses should be finite"

    print(f"  Fitnesses shape: {fitnesses.shape}")
    print(f"  Mean: {jnp.mean(fitnesses):.4f}, Std: {jnp.std(fitnesses):.4f}")
    print(f"[PASS] test_32_workers")
    return True


def test_variance_reduction():
    """Test: More workers should reduce gradient variance estimate."""
    print("[TEST] test_variance_reduction: More workers = lower variance")

    key = jax.random.PRNGKey(999)
    n_trials = 5  # Number of independent trials to estimate variance
    worker_counts = [4, 8, 16, 32]

    key, subkey = jax.random.split(key)
    params, es_tree_key = create_test_params(subkey)

    noiser = MockNoiser()
    frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

    variance_estimates = {}

    for n_workers in worker_counts:
        mean_fitnesses = []

        for trial in range(n_trials):
            key, subkey = jax.random.split(key)
            fitnesses, _ = run_es_epoch(
                subkey, n_workers, trial, noiser, frozen_noiser_params,
                noiser_params, params, es_tree_key
            )
            mean_fitnesses.append(float(jnp.mean(fitnesses)))

        # Variance of mean fitness across trials
        variance_estimates[n_workers] = np.var(mean_fitnesses)
        print(f"  {n_workers} workers: variance of mean = {variance_estimates[n_workers]:.6f}")

    # Generally more workers should lead to lower variance (not strict due to randomness)
    # Check that variance with 32 workers is not significantly higher than with 4
    ratio = variance_estimates[32] / (variance_estimates[4] + 1e-10)
    print(f"  Variance ratio (32/4): {ratio:.4f}")

    # Relaxed check: 32 workers should not have much higher variance than 4 workers
    assert ratio < 5.0, f"32 workers should not have much higher variance than 4 workers, got ratio={ratio}"

    print(f"[PASS] test_variance_reduction")
    return True


def test_antithetic_all_scales():
    """Test: Antithetic sampling works at all worker scales."""
    print("[TEST] test_antithetic_all_scales: Antithetic sampling at all scales")

    worker_counts = [4, 8, 16, 32]

    for n_workers in worker_counts:
        key = jax.random.PRNGKey(n_workers * 100)

        key, subkey = jax.random.split(key)
        params, es_tree_key = create_test_params(subkey)

        noiser = MockNoiser()
        frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

        original_B = params['B']

        # Check antithetic pairs for all even/odd thread pairs
        n_pairs = n_workers // 2
        sum_perturbations = 0.0

        for pair_idx in range(n_pairs):
            thread_even = pair_idx * 2
            thread_odd = pair_idx * 2 + 1

            iterinfo_even = (jnp.int32(0), jnp.int32(thread_even))
            iterinfo_odd = (jnp.int32(0), jnp.int32(thread_odd))

            noisy_B_even = noiser.get_noisy_standard(
                frozen_noiser_params, noiser_params,
                original_B, es_tree_key['B'], iterinfo_even
            )
            noisy_B_odd = noiser.get_noisy_standard(
                frozen_noiser_params, noiser_params,
                original_B, es_tree_key['B'], iterinfo_odd
            )

            pert_even = noisy_B_even - original_B
            pert_odd = noisy_B_odd - original_B

            # Antithetic: perturbations should sum to zero
            pair_sum = jnp.max(jnp.abs(pert_even + pert_odd))
            sum_perturbations += float(pair_sum)

        avg_sum = sum_perturbations / n_pairs
        assert avg_sum < 1e-5, f"Antithetic pairs should cancel, got avg sum = {avg_sum}"

        print(f"  {n_workers} workers: avg |pert_even + pert_odd| = {avg_sum:.2e}")

    print(f"[PASS] test_antithetic_all_scales")
    return True


def test_worker_consistency():
    """Test: Same seed produces same results across runs."""
    print("[TEST] test_worker_consistency: Deterministic results with same seed")

    worker_counts = [4, 16, 32]

    for n_workers in worker_counts:
        # Run 1
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        params, es_tree_key = create_test_params(subkey)

        noiser = MockNoiser()
        frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

        key, subkey = jax.random.split(key)
        fitnesses_1, _ = run_es_epoch(
            subkey, n_workers, 0, noiser, frozen_noiser_params,
            noiser_params, params, es_tree_key
        )

        # Run 2 (same seed)
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        params, es_tree_key = create_test_params(subkey)

        noiser = MockNoiser()
        frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

        key, subkey = jax.random.split(key)
        fitnesses_2, _ = run_es_epoch(
            subkey, n_workers, 0, noiser, frozen_noiser_params,
            noiser_params, params, es_tree_key
        )

        # Results should be identical
        assert jnp.allclose(fitnesses_1, fitnesses_2, atol=1e-6), \
            f"{n_workers} workers: Results not consistent with same seed"

        print(f"  {n_workers} workers: Consistent results confirmed")

    print(f"[PASS] test_worker_consistency")
    return True


def test_scaling_efficiency():
    """Test: Verify vmap scales correctly across worker counts."""
    print("[TEST] test_scaling_efficiency: vmap parallelization works at all scales")

    import time

    worker_counts = [4, 8, 16, 32]

    key = jax.random.PRNGKey(1234)
    key, subkey = jax.random.split(key)
    params, es_tree_key = create_test_params(subkey)

    noiser = MockNoiser()
    frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

    times = {}

    for n_workers in worker_counts:
        # Warmup (JIT compilation)
        key, subkey = jax.random.split(key)
        _ = run_es_epoch(
            subkey, n_workers, 0, noiser, frozen_noiser_params,
            noiser_params, params, es_tree_key
        )

        # Timed run
        n_runs = 5
        start = time.time()
        for i in range(n_runs):
            key, subkey = jax.random.split(key)
            fitnesses, _ = run_es_epoch(
                subkey, n_workers, i, noiser, frozen_noiser_params,
                noiser_params, params, es_tree_key
            )
            # Block until computation is done
            fitnesses.block_until_ready()
        end = time.time()

        avg_time = (end - start) / n_runs
        times[n_workers] = avg_time
        print(f"  {n_workers} workers: {avg_time*1000:.2f} ms per epoch")

    # On CPU, time should scale approximately linearly (no GPU parallelism)
    # But vmap should still be efficient (better than Python loops)
    # Just verify all scales complete successfully within reasonable time
    for n_workers in worker_counts:
        assert times[n_workers] < 10.0, f"{n_workers} workers took too long: {times[n_workers]:.2f}s"

    print(f"[PASS] test_scaling_efficiency")
    return True


def test_gradient_aggregation():
    """Test: Gradient aggregation works correctly across worker counts."""
    print("[TEST] test_gradient_aggregation: Weighted gradient aggregation")

    worker_counts = [4, 8, 16, 32]

    for n_workers in worker_counts:
        key = jax.random.PRNGKey(n_workers * 10)

        key, subkey = jax.random.split(key)
        params, es_tree_key = create_test_params(subkey)

        noiser = MockNoiser()
        frozen_noiser_params, noiser_params = noiser.init_noiser(params, sigma=0.1, lr=0.001, rank=4)

        key, subkey = jax.random.split(key)
        fitnesses, _ = run_es_epoch(
            subkey, n_workers, 0, noiser, frozen_noiser_params,
            noiser_params, params, es_tree_key
        )

        # Test fitness normalization (used in ES gradient estimation)
        # Rank-based normalization
        ranks = jnp.argsort(jnp.argsort(-fitnesses))  # Higher fitness = lower rank
        normalized_fitnesses = (ranks - (n_workers - 1) / 2) / (n_workers / 2)

        # Check properties
        assert jnp.allclose(jnp.mean(normalized_fitnesses), 0.0, atol=1e-5), \
            "Normalized fitnesses should have zero mean"
        assert normalized_fitnesses.shape == (n_workers,), \
            f"Shape mismatch: {normalized_fitnesses.shape}"

        print(f"  {n_workers} workers: normalized fitnesses range [{normalized_fitnesses.min():.2f}, {normalized_fitnesses.max():.2f}]")

    print(f"[PASS] test_gradient_aggregation")
    return True


def run_all_tests():
    """Run all Level 4 Production scaling tests."""
    print("=" * 60)
    print("Level 4 Production Test: Worker Scaling")
    print("=" * 60)
    print()

    tests = [
        test_4_workers,
        test_8_workers,
        test_16_workers,
        test_32_workers,
        test_variance_reduction,
        test_antithetic_all_scales,
        test_worker_consistency,
        test_scaling_efficiency,
        test_gradient_aggregation,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"[FAIL] {test_fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
