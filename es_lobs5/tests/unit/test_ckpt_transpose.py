"""Test checkpoint kernel transpose."""

import numpy as np
import jax.numpy as jnp
import pytest

from es_lobs5.adapters.checkpoint_adapter import (
    _convert_dense_to_linear,
    _convert_layernorm,
)


class TestKernelTranspose:
    """Test Flax kernel (in,out) -> ES weight (out,in) transpose."""

    def test_kernel_transpose_basic(self):
        """Verify kernel.T == weight for Dense layer conversion."""
        in_dim, out_dim = 64, 128
        flax_kernel = np.random.randn(in_dim, out_dim).astype(np.float32)
        flax_bias = np.random.randn(out_dim).astype(np.float32)

        flax_dense = {
            'kernel': jnp.array(flax_kernel),
            'bias': jnp.array(flax_bias),
        }

        es_linear = _convert_dense_to_linear(flax_dense)

        # Verify transpose: kernel.T == weight
        np.testing.assert_array_almost_equal(
            np.array(es_linear['weight']),
            flax_kernel.T,
            decimal=6,
            err_msg="kernel.T should equal weight"
        )

        # Verify bias unchanged
        np.testing.assert_array_almost_equal(
            np.array(es_linear['bias']),
            flax_bias,
            decimal=6,
            err_msg="bias should remain unchanged"
        )

    def test_kernel_transpose_shapes(self):
        """Verify shape transformation: (in, out) -> (out, in)."""
        in_dim, out_dim = 256, 512
        flax_kernel = np.random.randn(in_dim, out_dim).astype(np.float32)

        flax_dense = {'kernel': jnp.array(flax_kernel)}
        es_linear = _convert_dense_to_linear(flax_dense)

        # Verify shape is transposed
        assert es_linear['weight'].shape == (out_dim, in_dim), \
            f"Expected shape {(out_dim, in_dim)}, got {es_linear['weight'].shape}"

    def test_kernel_transpose_without_bias(self):
        """Test conversion when bias is not present."""
        in_dim, out_dim = 32, 64
        flax_kernel = np.random.randn(in_dim, out_dim).astype(np.float32)

        flax_dense = {'kernel': jnp.array(flax_kernel)}
        es_linear = _convert_dense_to_linear(flax_dense)

        # Verify weight exists and is transposed
        assert 'weight' in es_linear
        np.testing.assert_array_almost_equal(
            np.array(es_linear['weight']),
            flax_kernel.T,
            decimal=6
        )

        # Verify bias is not added when not present
        assert 'bias' not in es_linear


class TestLayerNormMapping:
    """Test Flax LayerNorm scale -> ES weight, bias -> bias mapping."""

    def test_layernorm_mapping_basic(self):
        """Verify scale -> weight and bias -> bias mapping."""
        dim = 256
        flax_scale = np.random.randn(dim).astype(np.float32)
        flax_bias = np.random.randn(dim).astype(np.float32)

        flax_norm = {
            'scale': jnp.array(flax_scale),
            'bias': jnp.array(flax_bias),
        }

        es_norm = _convert_layernorm(flax_norm)

        # Verify scale -> weight
        np.testing.assert_array_almost_equal(
            np.array(es_norm['weight']),
            flax_scale,
            decimal=6,
            err_msg="Flax scale should map to ES weight"
        )

        # Verify bias -> bias
        np.testing.assert_array_almost_equal(
            np.array(es_norm['bias']),
            flax_bias,
            decimal=6,
            err_msg="Flax bias should map to ES bias"
        )

    def test_layernorm_shapes_preserved(self):
        """Verify shapes are preserved after conversion."""
        dim = 512
        flax_scale = np.ones(dim, dtype=np.float32)
        flax_bias = np.zeros(dim, dtype=np.float32)

        flax_norm = {
            'scale': jnp.array(flax_scale),
            'bias': jnp.array(flax_bias),
        }

        es_norm = _convert_layernorm(flax_norm)

        assert es_norm['weight'].shape == (dim,), \
            f"Expected weight shape {(dim,)}, got {es_norm['weight'].shape}"
        assert es_norm['bias'].shape == (dim,), \
            f"Expected bias shape {(dim,)}, got {es_norm['bias'].shape}"

    def test_layernorm_output_keys(self):
        """Verify output dict has correct keys (weight, bias)."""
        dim = 128
        flax_norm = {
            'scale': jnp.ones(dim),
            'bias': jnp.zeros(dim),
        }

        es_norm = _convert_layernorm(flax_norm)

        assert set(es_norm.keys()) == {'weight', 'bias'}, \
            f"Expected keys {{'weight', 'bias'}}, got {set(es_norm.keys())}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
