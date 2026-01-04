"""Test call_submodule utility."""
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.models.common import (
    call_submodule, merge_inits, simple_es_tree_key,
    ES_Parameter, CommonParams, PARAM,
)
from es_lobs5.utils.import_utils import load_module, get_hyperscalees_path
import os

# Load only base_noiser to avoid optax dependency
_hyperscalees_path = get_hyperscalees_path()
_base_noiser = load_module(
    'hyperscalees.noiser.base_noiser',
    os.path.join(_hyperscalees_path, 'hyperscalees/noiser/base_noiser.py')
)
NoopNoiser = _base_noiser.Noiser


def _build_common_params(es_init, key):
    """Helper to build CommonParams with noop noiser (no optax dependency)."""
    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)

    return CommonParams(
        noiser=NoopNoiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=es_init.frozen_params,
        params=es_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )


def test_call_submodule_extracts_params():
    """Test: call_submodule correctly extracts submodule params from nested structure."""
    key = jax.random.PRNGKey(42)

    # Create nested structure with two submodules
    key, k1, k2 = jax.random.split(key, 3)
    weight_init = ES_Parameter.rand_init(k1, shape=(16, 8), scale=1.0, es_type=PARAM)
    bias_init = ES_Parameter.rand_init(k2, shape=(8,), scale=0.1, es_type=PARAM)

    # Merge into parent structure
    merged_init = merge_inits(weight=weight_init, bias=bias_init)

    # Verify merged structure has both submodules
    assert 'weight' in merged_init.params, "Merged params should have 'weight' key"
    assert 'bias' in merged_init.params, "Merged params should have 'bias' key"

    # Build common_params
    key, subkey = jax.random.split(key)
    common_params = _build_common_params(merged_init, subkey)

    # Verify that common_params.params contains nested structure
    assert isinstance(common_params.params, dict), "params should be a dict"
    assert jnp.allclose(common_params.params['weight'], weight_init.params), \
        "Parent params['weight'] should match weight_init.params"
    assert jnp.allclose(common_params.params['bias'], bias_init.params), \
        "Parent params['bias'] should match bias_init.params"

    # Call submodule for 'weight' - this should extract params['weight']
    weight_result = call_submodule(ES_Parameter, 'weight', common_params)

    # Weight result should be noisy version of weight_init.params
    assert weight_result.shape == (16, 8), f"Weight shape mismatch: {weight_result.shape}"

    # Call submodule for 'bias'
    bias_result = call_submodule(ES_Parameter, 'bias', common_params)
    assert bias_result.shape == (8,), f"Bias shape mismatch: {bias_result.shape}"

    print("[PASS] test_call_submodule_extracts_params")
    return True


def test_call_submodule_updates_key():
    """Test: es_tree_key path is correctly updated when calling submodule."""
    key = jax.random.PRNGKey(123)

    # Create parent structure with two submodules
    key, k1, k2 = jax.random.split(key, 3)
    sub_a_init = ES_Parameter.rand_init(k1, shape=(4,), scale=1.0, es_type=PARAM)
    sub_b_init = ES_Parameter.rand_init(k2, shape=(4,), scale=1.0, es_type=PARAM)

    merged_init = merge_inits(sub_a=sub_a_init, sub_b=sub_b_init)

    # Build common_params with es_tree_key
    key, subkey = jax.random.split(key)
    common_params = _build_common_params(merged_init, subkey)

    # Verify es_tree_key has nested structure
    assert isinstance(common_params.es_tree_key, dict), "es_tree_key should be a dict"
    assert 'sub_a' in common_params.es_tree_key, "es_tree_key should have 'sub_a'"
    assert 'sub_b' in common_params.es_tree_key, "es_tree_key should have 'sub_b'"

    # The sub-keys should be different for different submodules
    key_a = common_params.es_tree_key['sub_a']
    key_b = common_params.es_tree_key['sub_b']
    assert not jnp.allclose(key_a, key_b), \
        "Different submodules should have different es_tree_keys"

    # Call submodules with same common_params
    result_a = call_submodule(ES_Parameter, 'sub_a', common_params)
    result_b = call_submodule(ES_Parameter, 'sub_b', common_params)

    # Results should differ due to different es_tree_keys
    assert not jnp.allclose(result_a, result_b), \
        "Different submodules should produce different noisy outputs"

    # Calling same submodule twice should produce same result (deterministic)
    result_a2 = call_submodule(ES_Parameter, 'sub_a', common_params)
    assert jnp.allclose(result_a, result_a2), \
        "Same submodule called twice should produce identical results"

    print("[PASS] test_call_submodule_updates_key")
    return True


if __name__ == "__main__":
    test_call_submodule_extracts_params()
    test_call_submodule_updates_key()
    print("All call_submodule tests passed!")
