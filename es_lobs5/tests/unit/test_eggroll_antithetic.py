"""Test Eggroll antithetic sampling."""
import jax
import jax.numpy as jnp
import pytest

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.utils.import_utils import get_all_noisers


def test_antithetic_pairs():
    """Test: thread pairs have opposite perturbations: sigma[2k] = -sigma[2k+1].

    In antithetic sampling, thread pairs (0,1), (2,3), (4,5), etc. should have
    opposite perturbations. This reduces variance in the gradient estimate.

    The EggRoll implementation uses:
    - true_thread_idx = thread_id // 2 (so pairs share the same noise pattern)
    - sigma = +base_sigma if thread_id % 2 == 0, else -base_sigma

    Therefore: perturbation[2k] = -perturbation[2k+1]
    """
    noisers = get_all_noisers()
    FixedEggRoll = noisers['eggroll']

    # Create a test parameter
    key = jax.random.PRNGKey(42)
    param = jax.random.normal(key, (64, 32), dtype=jnp.float32)

    # Initialize noiser
    frozen_noiser_params, noiser_params = FixedEggRoll.init_noiser(
        {'test': param}, sigma=0.1, lr=0.001, rank=4
    )

    # Create parameter key
    param_key = jax.random.PRNGKey(123)

    # Test multiple antithetic pairs
    for pair_idx in range(5):
        thread_even = 2 * pair_idx      # 0, 2, 4, 6, 8
        thread_odd = 2 * pair_idx + 1   # 1, 3, 5, 7, 9

        epoch = 0

        # Get noisy parameters for even and odd threads
        noisy_even = FixedEggRoll.get_noisy_standard(
            frozen_noiser_params, noiser_params,
            param, param_key,
            (epoch, thread_even)
        )

        noisy_odd = FixedEggRoll.get_noisy_standard(
            frozen_noiser_params, noiser_params,
            param, param_key,
            (epoch, thread_odd)
        )

        # Compute perturbations
        pert_even = noisy_even - param
        pert_odd = noisy_odd - param

        # Antithetic property: pert_even + pert_odd should be zero
        sum_pert = pert_even + pert_odd
        max_diff = jnp.max(jnp.abs(sum_pert))

        assert max_diff < 1e-6, (
            f"Pair ({thread_even}, {thread_odd}): perturbations should be opposite, "
            f"but |pert_even + pert_odd|_max = {max_diff:.2e}"
        )

    print("[PASS] test_antithetic_pairs: sigma[2k] = -sigma[2k+1] verified for 5 pairs")


def test_antithetic_same_magnitude():
    """Test: |sigma[2k]| == |sigma[2k+1]|.

    Antithetic pairs should have the same magnitude of perturbation,
    just opposite signs. This ensures variance reduction while maintaining
    the same exploration radius.
    """
    noisers = get_all_noisers()
    FixedEggRoll = noisers['eggroll']

    # Create a test parameter
    key = jax.random.PRNGKey(42)
    param = jax.random.normal(key, (64, 32), dtype=jnp.float32)

    # Initialize noiser
    frozen_noiser_params, noiser_params = FixedEggRoll.init_noiser(
        {'test': param}, sigma=0.1, lr=0.001, rank=4
    )

    # Create parameter key
    param_key = jax.random.PRNGKey(456)

    # Test multiple antithetic pairs
    for pair_idx in range(5):
        thread_even = 2 * pair_idx      # 0, 2, 4, 6, 8
        thread_odd = 2 * pair_idx + 1   # 1, 3, 5, 7, 9

        epoch = 0

        # Get noisy parameters for even and odd threads
        noisy_even = FixedEggRoll.get_noisy_standard(
            frozen_noiser_params, noiser_params,
            param, param_key,
            (epoch, thread_even)
        )

        noisy_odd = FixedEggRoll.get_noisy_standard(
            frozen_noiser_params, noiser_params,
            param, param_key,
            (epoch, thread_odd)
        )

        # Compute perturbations
        pert_even = noisy_even - param
        pert_odd = noisy_odd - param

        # Compute magnitudes
        mag_even = jnp.abs(pert_even)
        mag_odd = jnp.abs(pert_odd)

        # Magnitudes should be equal
        mag_diff = jnp.max(jnp.abs(mag_even - mag_odd))

        assert mag_diff < 1e-6, (
            f"Pair ({thread_even}, {thread_odd}): magnitudes should be equal, "
            f"but |mag_even - mag_odd|_max = {mag_diff:.2e}"
        )

        # Also verify that perturbations are non-zero
        max_pert = jnp.max(mag_even)
        assert max_pert > 1e-6, (
            f"Pair ({thread_even}, {thread_odd}): perturbations should be non-zero"
        )

    print("[PASS] test_antithetic_same_magnitude: |sigma[2k]| == |sigma[2k+1]| verified for 5 pairs")


def test_different_pairs_different_noise():
    """Test: different antithetic pairs have different noise patterns.

    While threads within a pair share the same noise pattern (opposite signs),
    different pairs should have different noise patterns.
    """
    noisers = get_all_noisers()
    FixedEggRoll = noisers['eggroll']

    # Create a test parameter
    key = jax.random.PRNGKey(42)
    param = jax.random.normal(key, (64, 32), dtype=jnp.float32)

    # Initialize noiser
    frozen_noiser_params, noiser_params = FixedEggRoll.init_noiser(
        {'test': param}, sigma=0.1, lr=0.001, rank=4
    )

    # Create parameter key
    param_key = jax.random.PRNGKey(789)
    epoch = 0

    # Get perturbations for pair 0 (threads 0, 1)
    noisy_pair0 = FixedEggRoll.get_noisy_standard(
        frozen_noiser_params, noiser_params,
        param, param_key,
        (epoch, 0)
    )
    pert_pair0 = noisy_pair0 - param

    # Get perturbations for pair 1 (threads 2, 3)
    noisy_pair1 = FixedEggRoll.get_noisy_standard(
        frozen_noiser_params, noiser_params,
        param, param_key,
        (epoch, 2)
    )
    pert_pair1 = noisy_pair1 - param

    # Different pairs should have different noise
    diff = jnp.max(jnp.abs(pert_pair0 - pert_pair1))

    assert diff > 1e-6, (
        f"Different pairs should have different noise patterns, "
        f"but max diff = {diff:.2e}"
    )

    print("[PASS] test_different_pairs_different_noise: different pairs have different noise")


if __name__ == "__main__":
    test_antithetic_pairs()
    test_antithetic_same_magnitude()
    test_different_pairs_different_noise()
    print("\nAll Eggroll antithetic sampling tests passed!")
