"""Test ES_LayerNorm.

Tests:
1. test_layernorm_output_normalized: output mean ~ 0, var ~ 1
2. test_layernorm_excluded: LayerNorm params are EXCLUDED type by default
"""
import jax
import jax.numpy as jnp
import numpy as np
import sys

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.models.common import (
    ES_LayerNorm, ES_Parameter, CommonParams, simple_es_tree_key, EXCLUDED
)
from es_lobs5.utils.import_utils import get_all_noisers


def _build_common_params(es_init, key):
    """Helper to build CommonParams with noop noiser."""
    noisers = get_all_noisers()
    noiser = noisers['noop']
    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)
    return CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=es_init.frozen_params,
        params=es_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )


def test_layernorm_output_normalized():
    """Test: LayerNorm output has mean ~ 0 and var ~ 1 along last axis."""
    key = jax.random.PRNGKey(42)
    dim = 64

    # Initialize ES_LayerNorm (with bias, so affine transform applied)
    key, subkey = jax.random.split(key)
    es_init = ES_LayerNorm.rand_init(subkey, dim=dim, use_bias=True)
    common_params = _build_common_params(es_init, key)

    # Create random input with non-zero mean and varying variance
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (16, 32, dim)) * 5.0 + 3.0  # mean~3, std~5

    # Forward pass
    output = ES_LayerNorm._forward(common_params, x)

    # Check output statistics along last axis
    # Since weight=1 and bias=0 by default, output should be normalized
    out_mean = jnp.mean(output, axis=-1)
    out_var = jnp.var(output, axis=-1)

    mean_close_to_zero = jnp.allclose(out_mean, 0.0, atol=1e-5)
    var_close_to_one = jnp.allclose(out_var, 1.0, atol=1e-5)

    assert mean_close_to_zero, f"Mean not close to 0: max|mean|={jnp.max(jnp.abs(out_mean))}"
    assert var_close_to_one, f"Var not close to 1: max|var-1|={jnp.max(jnp.abs(out_var - 1.0))}"

    print(f"[PASS] LayerNorm output normalized: max|mean|={jnp.max(jnp.abs(out_mean)):.2e}, max|var-1|={jnp.max(jnp.abs(out_var - 1.0)):.2e}")
    return True


def test_layernorm_shapes():
    """Test: LayerNorm preserves input/output shape."""
    key = jax.random.PRNGKey(123)
    dim = 128

    # Initialize without bias
    key, subkey = jax.random.split(key)
    es_init_no_bias = ES_LayerNorm.rand_init(subkey, dim=dim, use_bias=False)
    common_params_no_bias = _build_common_params(es_init_no_bias, key)

    # Initialize with bias
    key, subkey = jax.random.split(key)
    es_init_with_bias = ES_LayerNorm.rand_init(subkey, dim=dim, use_bias=True)
    common_params_with_bias = _build_common_params(es_init_with_bias, key)

    # Test various input shapes
    test_shapes = [
        (dim,),           # 1D
        (8, dim),         # 2D (seq_len, dim)
        (4, 16, dim),     # 3D (batch, seq_len, dim)
        (2, 4, 8, dim),   # 4D
    ]

    for shape in test_shapes:
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape)

        out_no_bias = ES_LayerNorm._forward(common_params_no_bias, x)
        out_with_bias = ES_LayerNorm._forward(common_params_with_bias, x)

        assert out_no_bias.shape == shape, f"Shape mismatch (no_bias): {out_no_bias.shape} vs {shape}"
        assert out_with_bias.shape == shape, f"Shape mismatch (with_bias): {out_with_bias.shape} vs {shape}"

    print(f"[PASS] LayerNorm shapes preserved for {len(test_shapes)} test cases")
    return True


def test_layernorm_excluded():
    """Test: LayerNorm params are EXCLUDED type by default."""
    key = jax.random.PRNGKey(99)
    dim = 64

    # Initialize with default es_type (should be EXCLUDED)
    key, subkey = jax.random.split(key)
    es_init = ES_LayerNorm.rand_init(subkey, dim=dim, use_bias=True)

    # Check es_map for weight and bias
    weight_es_type = es_init.es_map['weight']
    bias_es_type = es_init.es_map['bias']

    assert weight_es_type == EXCLUDED, f"Weight es_type should be EXCLUDED, got {weight_es_type}"
    assert bias_es_type == EXCLUDED, f"Bias es_type should be EXCLUDED, got {bias_es_type}"

    # Also test without bias
    key, subkey = jax.random.split(key)
    es_init_no_bias = ES_LayerNorm.rand_init(subkey, dim=dim, use_bias=False)
    weight_es_type_no_bias = es_init_no_bias.es_map['weight']

    assert weight_es_type_no_bias == EXCLUDED, f"Weight (no_bias) es_type should be EXCLUDED, got {weight_es_type_no_bias}"
    assert 'bias' not in es_init_no_bias.es_map, "No bias expected when use_bias=False"

    print("[PASS] LayerNorm params have EXCLUDED es_type by default")
    return True


if __name__ == "__main__":
    test_layernorm_output_normalized()
    test_layernorm_excluded()
    test_layernorm_shapes()
    print("All ES_LayerNorm tests passed!")
