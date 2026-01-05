"""Test ES_Parameter initialization and es_map classification.

Simplified tests that do NOT require HyperscaleES noiser imports.
We only verify ES_Parameter initialization, es_map values, and scan_map behavior.
"""
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.models.common import (
    ES_Parameter, PARAM, EXCLUDED, MM_PARAM, EMB_PARAM
)


def test_es_parameter_param_type():
    """Test: ES_Parameter with es_type=PARAM has correct es_map."""
    key = jax.random.PRNGKey(42)
    shape = (64, 32)

    es_init = ES_Parameter.rand_init(key, shape=shape, scale=1.0, es_type=PARAM)

    # Verify es_map is PARAM
    assert es_init.es_map == PARAM, f"Expected es_map=PARAM({PARAM}), got {es_init.es_map}"
    # Verify params shape
    assert es_init.params.shape == shape, f"Expected shape {shape}, got {es_init.params.shape}"
    # Verify frozen_params is None (PARAM type should not have frozen params by default)
    assert es_init.frozen_params is None, f"Expected frozen_params=None for PARAM type"
    # Verify scan_map is empty tuple by default (not scanned)
    assert es_init.scan_map == (), f"Expected scan_map=(), got {es_init.scan_map}"

    print(f"[PASS] test_es_parameter_param_type: es_map={es_init.es_map}, shape={es_init.params.shape}")


def test_es_parameter_excluded_type():
    """Test: ES_Parameter with es_type=EXCLUDED has correct es_map."""
    key = jax.random.PRNGKey(123)
    shape = (32, 16)

    es_init = ES_Parameter.rand_init(key, shape=shape, scale=1.0, es_type=EXCLUDED)

    # Verify es_map is EXCLUDED
    assert es_init.es_map == EXCLUDED, f"Expected es_map=EXCLUDED({EXCLUDED}), got {es_init.es_map}"
    # Verify params shape
    assert es_init.params.shape == shape, f"Expected shape {shape}, got {es_init.params.shape}"
    # Verify scan_map is empty tuple by default
    assert es_init.scan_map == (), f"Expected scan_map=(), got {es_init.scan_map}"

    print(f"[PASS] test_es_parameter_excluded_type: es_map={es_init.es_map}, shape={es_init.params.shape}")


def test_es_parameter_mm_emb_types():
    """Test: ES_Parameter with MM_PARAM and EMB_PARAM es_types."""
    key = jax.random.PRNGKey(456)

    # MM_PARAM type
    key, subkey = jax.random.split(key)
    mm_param = ES_Parameter.rand_init(subkey, shape=(64, 32), scale=0.1, es_type=MM_PARAM)
    assert mm_param.es_map == MM_PARAM, f"Expected es_map=MM_PARAM({MM_PARAM}), got {mm_param.es_map}"

    # EMB_PARAM type
    key, subkey = jax.random.split(key)
    emb_param = ES_Parameter.rand_init(subkey, shape=(1000, 64), scale=0.1, es_type=EMB_PARAM)
    assert emb_param.es_map == EMB_PARAM, f"Expected es_map=EMB_PARAM({EMB_PARAM}), got {emb_param.es_map}"

    print(f"[PASS] test_es_parameter_mm_emb_types: MM_PARAM={mm_param.es_map}, EMB_PARAM={emb_param.es_map}")


def test_es_parameter_raw_value():
    """Test: ES_Parameter with raw_value initialization."""
    key = jax.random.PRNGKey(789)
    raw_value = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    es_init = ES_Parameter.rand_init(key, raw_value=raw_value, es_type=PARAM)

    # Verify params match raw_value
    assert jnp.allclose(es_init.params, raw_value), "Params should match raw_value"
    # Verify es_map is PARAM
    assert es_init.es_map == PARAM, f"Expected es_map=PARAM({PARAM}), got {es_init.es_map}"

    print(f"[PASS] test_es_parameter_raw_value: params match raw_value")


def test_es_parameter_dtype():
    """Test: ES_Parameter with different dtype."""
    key = jax.random.PRNGKey(111)
    shape = (32, 32)

    # Test float16
    es_init_f16 = ES_Parameter.rand_init(key, shape=shape, scale=1.0, dtype=jnp.float16)
    assert es_init_f16.params.dtype == jnp.float16, f"Expected float16, got {es_init_f16.params.dtype}"

    # Test bfloat16
    es_init_bf16 = ES_Parameter.rand_init(key, shape=shape, scale=1.0, dtype=jnp.bfloat16)
    assert es_init_bf16.params.dtype == jnp.bfloat16, f"Expected bfloat16, got {es_init_bf16.params.dtype}"

    print(f"[PASS] test_es_parameter_dtype: float16 and bfloat16 dtypes work")


def test_es_map_constants():
    """Test: Verify es_map constants have expected values."""
    # These constants should be: PARAM=0, MM_PARAM=1, EMB_PARAM=2, EXCLUDED=3
    assert PARAM == 0, f"Expected PARAM=0, got {PARAM}"
    assert MM_PARAM == 1, f"Expected MM_PARAM=1, got {MM_PARAM}"
    assert EMB_PARAM == 2, f"Expected EMB_PARAM=2, got {EMB_PARAM}"
    assert EXCLUDED == 3, f"Expected EXCLUDED=3, got {EXCLUDED}"

    print(f"[PASS] test_es_map_constants: PARAM={PARAM}, MM_PARAM={MM_PARAM}, EMB_PARAM={EMB_PARAM}, EXCLUDED={EXCLUDED}")


if __name__ == "__main__":
    test_es_parameter_param_type()
    test_es_parameter_excluded_type()
    test_es_parameter_mm_emb_types()
    test_es_parameter_raw_value()
    test_es_parameter_dtype()
    test_es_map_constants()
    print("\nAll ES_Parameter tests passed!")
