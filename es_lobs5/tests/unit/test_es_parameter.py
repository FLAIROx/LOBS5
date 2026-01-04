"""Test ES_Parameter noise injection."""
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.models.common import (
    ES_Parameter, PARAM, EXCLUDED, CommonParams, simple_es_tree_key
)
from es_lobs5.utils.import_utils import get_all_noisers


def _build_common_params_with_noiser(es_init, key, noiser_name='eggroll', sigma=0.1, iterinfo=(0, 0)):
    """Helper to build CommonParams with specified noiser and iterinfo."""
    noisers = get_all_noisers()
    noiser = noisers[noiser_name]

    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params, sigma=sigma, lr=0.001, rank=4
    )

    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)

    return CommonParams(
        noiser=noiser,
        frozen_noiser_params=frozen_noiser_params,
        noiser_params=noiser_params,
        frozen_params=es_init.frozen_params,
        params=es_init.params,
        es_tree_key=es_tree_key,
        iterinfo=iterinfo,
    )


def test_parameter_noise_applied():
    """Test: ES perturbation produces noisy != base params for PARAM type."""
    key = jax.random.PRNGKey(42)
    shape = (64, 32)

    # Initialize ES_Parameter with default es_type=PARAM
    key, subkey = jax.random.split(key)
    es_init = ES_Parameter.rand_init(subkey, shape=shape, scale=1.0, es_type=PARAM)

    # Verify es_map is PARAM
    assert es_init.es_map == PARAM, f"Expected es_map=PARAM({PARAM}), got {es_init.es_map}"

    # Store original params for comparison
    original_params = es_init.params.copy()

    # Build common_params with noiser and non-None iterinfo
    key, subkey = jax.random.split(key)
    common_params = _build_common_params_with_noiser(
        es_init, subkey, noiser_name='eggroll', sigma=0.1, iterinfo=(0, 0)
    )

    # Forward pass applies noise via get_noisy_standard
    noisy_params = ES_Parameter._forward(common_params)

    # Noisy params should differ from original
    diff = jnp.max(jnp.abs(noisy_params - original_params))
    assert diff > 1e-6, f"Noisy params should differ from base, but max diff = {diff:.2e}"

    print(f"[PASS] test_parameter_noise_applied: max diff = {diff:.6f}")
    return True


def test_parameter_excluded_unchanged():
    """Test: EXCLUDED type parameter noisy == base params (with freeze_nonlora=True)."""
    key = jax.random.PRNGKey(123)
    shape = (32, 16)

    # Initialize ES_Parameter with es_type=EXCLUDED
    key, subkey = jax.random.split(key)
    es_init = ES_Parameter.rand_init(subkey, shape=shape, scale=1.0, es_type=EXCLUDED)

    # Verify es_map is EXCLUDED
    assert es_init.es_map == EXCLUDED, f"Expected es_map=EXCLUDED({EXCLUDED}), got {es_init.es_map}"

    # Store original params for comparison
    original_params = es_init.params.copy()

    # Build common_params with noiser, freeze_nonlora=True to skip standard param perturbation
    noisers = get_all_noisers()
    noiser = noisers['eggroll']

    key, subkey = jax.random.split(key)
    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params, sigma=0.1, lr=0.001, rank=4, freeze_nonlora=True
    )

    es_tree_key = simple_es_tree_key(es_init.params, subkey, es_init.scan_map)

    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=frozen_noiser_params,
        noiser_params=noiser_params,
        frozen_params=es_init.frozen_params,
        params=es_init.params,
        es_tree_key=es_tree_key,
        iterinfo=(0, 0),  # Non-None iterinfo
    )

    # Forward pass - with freeze_nonlora=True, params should not be perturbed
    noisy_params = ES_Parameter._forward(common_params)

    # Noisy params should be identical to original
    diff = jnp.max(jnp.abs(noisy_params - original_params))
    assert diff < 1e-10, f"EXCLUDED params should not be perturbed, but max diff = {diff:.2e}"

    print(f"[PASS] test_parameter_excluded_unchanged: max diff = {diff:.2e}")
    return True


if __name__ == "__main__":
    test_parameter_noise_applied()
    test_parameter_excluded_unchanged()
    print("All ES_Parameter tests passed!")
