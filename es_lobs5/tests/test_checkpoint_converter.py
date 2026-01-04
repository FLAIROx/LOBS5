"""
Test: Checkpoint conversion round-trip and output consistency
File: es_lobs5/tests/test_checkpoint_converter.py
"""
import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.adapters.checkpoint_adapter import (
    _convert_dense_to_linear,
    _convert_layernorm,
    _convert_ssm,
    convert_flax_to_es,
)


def test_dense_transpose():
    """Test: Dense kernel is correctly transposed."""
    # Flax format: kernel (in_dim, out_dim)
    flax_dense = {
        'kernel': jnp.array([[1, 2, 3], [4, 5, 6]]),  # (2, 3)
        'bias': jnp.array([0.1, 0.2, 0.3]),  # (3,)
    }

    es_linear = _convert_dense_to_linear(flax_dense)

    # ES format: weight (out_dim, in_dim)
    expected_weight = jnp.array([[1, 4], [2, 5], [3, 6]])  # (3, 2)

    assert jnp.allclose(es_linear['weight'], expected_weight), \
        f"Transpose failed: {es_linear['weight']} vs {expected_weight}"
    assert jnp.allclose(es_linear['bias'], flax_dense['bias']), \
        "Bias should be unchanged"

    print("[PASS] Dense transpose test")
    return True


def test_layernorm_conversion():
    """Test: LayerNorm scale -> weight rename."""
    flax_norm = {
        'scale': jnp.ones(64),
        'bias': jnp.zeros(64),
    }

    es_norm = _convert_layernorm(flax_norm)

    assert 'weight' in es_norm, "Missing 'weight' key"
    assert 'scale' not in es_norm, "Should not have 'scale' key"
    assert jnp.allclose(es_norm['weight'], flax_norm['scale'])
    assert jnp.allclose(es_norm['bias'], flax_norm['bias'])

    print("[PASS] LayerNorm conversion test")
    return True


def test_ssm_conversion_unchanged():
    """Test: SSM params are copied without modification."""
    key = jax.random.PRNGKey(42)

    flax_ssm = {
        'Lambda_re': jax.random.normal(key, (32,)),
        'Lambda_im': jax.random.normal(key, (32,)),
        'B': jax.random.normal(key, (32, 64, 2)),
        'C': jax.random.normal(key, (64, 32, 2)),
        'D': jax.random.normal(key, (64,)),
        'log_step': jax.random.normal(key, (32, 1)),
    }

    es_ssm = _convert_ssm(flax_ssm)

    for k in flax_ssm:
        assert jnp.allclose(es_ssm[k], flax_ssm[k]), f"SSM param {k} changed!"

    print("[PASS] SSM conversion unchanged test")
    return True


def test_full_model_conversion():
    """Test: Full model conversion with layer naming."""
    key = jax.random.PRNGKey(123)

    # Create mock Flax params structure
    d_model = 64
    H = 64
    P = 32

    # Mock SSM params
    def make_ssm():
        return {
            'Lambda_re': jax.random.normal(key, (P,)),
            'Lambda_im': jax.random.normal(key, (P,)),
            'B': jax.random.normal(key, (P, H, 2)),
            'C': jax.random.normal(key, (H, P, 2)),
            'D': jax.random.normal(key, (H,)),
            'log_step': jax.random.normal(key, (P, 1)),
        }

    # Mock sequence layer
    def make_seq_layer():
        return {
            'seq': make_ssm(),  # Flax uses 'seq'
            'norm': {
                'scale': jnp.ones(d_model),
                'bias': jnp.zeros(d_model),
            },
            'out2': {  # half_glu1 has out2
                'kernel': jax.random.normal(key, (d_model, d_model)),
                'bias': jnp.zeros(d_model),
            },
        }

    flax_params = {
        'message_encoder': {
            'encoder': {'embedding': jax.random.normal(key, (12012, d_model))},
            'layers_0': make_seq_layer(),
            'layers_1': make_seq_layer(),
        },
        'book_encoder': {
            'pre_layers_0': make_seq_layer(),
            'projection': {
                'kernel': jax.random.normal(key, (503, d_model)),
                'bias': jnp.zeros(d_model),
            },
            'post_layers_0': make_seq_layer(),
        },
        'fused_s5': {
            'encoder': {
                'kernel': jax.random.normal(key, (2*d_model, d_model)),
                'bias': jnp.zeros(d_model),
            },
            'layers_0': make_seq_layer(),
            'layers_1': make_seq_layer(),
        },
        'decoder': {
            'kernel': jax.random.normal(key, (d_model, 12012)),
            'bias': jnp.zeros(12012),
        },
    }

    config = {
        'n_message_layers': 2,
        'n_fused_layers': 2,
        'n_book_pre_layers': 1,
        'n_book_post_layers': 1,
        'activation': 'half_glu1',
    }

    es_params = convert_flax_to_es(flax_params, config)

    # Verify key structure
    assert 'message_encoder' in es_params
    assert 'book_encoder' in es_params
    assert 'fused_encoder' in es_params  # fused_s5 -> fused_encoder
    assert 'decoder' in es_params

    # Verify layer naming
    assert 'layer_0' in es_params['message_encoder']  # layers_0 -> layer_0
    assert 'layer_1' in es_params['message_encoder']
    assert 'ssm' in es_params['message_encoder']['layer_0']  # seq -> ssm
    assert 'weight' in es_params['message_encoder']['layer_0']['norm']  # scale -> weight

    # Verify book encoder
    assert 'pre_layer_0' in es_params['book_encoder']  # pre_layers_0 -> pre_layer_0
    assert 'proj' in es_params['book_encoder']  # projection -> proj
    assert 'post_layer_0' in es_params['book_encoder']

    # Verify fused encoder
    assert 'input_proj' in es_params['fused_encoder']  # encoder -> input_proj

    # Verify decoder transpose
    expected_decoder_shape = (12012, d_model)  # Transposed
    assert es_params['decoder']['weight'].shape == expected_decoder_shape

    print("[PASS] Full model conversion test")
    return True


if __name__ == "__main__":
    test_dense_transpose()
    test_layernorm_conversion()
    test_ssm_conversion_unchanged()
    test_full_model_conversion()
    print("All checkpoint converter tests passed!")
