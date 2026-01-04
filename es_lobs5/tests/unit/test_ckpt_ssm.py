"""
Test SSM params unchanged in checkpoint conversion.

This module tests that SSM parameters (Lambda, B, C, D) remain unchanged
during Flax <-> ES checkpoint conversion, ensuring model equivalence.

Tests:
    1. test_ssm_params_unchanged: Lambda, B, C, D unchanged in Flax<->ES conversion
    2. test_layer_naming: layers_N -> layer_N naming conversion
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import pytest
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.adapters.checkpoint_adapter import (
    _convert_ssm,
    _convert_sequence_layer,
    _convert_message_encoder,
    _convert_book_encoder,
    _convert_stacked_encoder,
    convert_flax_to_es,
    convert_es_to_flax,
)


class TestSSMParamsUnchanged:
    """Test that SSM parameters remain unchanged during conversion."""

    def test_lambda_unchanged(self):
        """Lambda_re and Lambda_im should be identical after conversion."""
        key = jax.random.PRNGKey(42)
        P = 64  # State size

        flax_ssm = {
            'Lambda_re': jax.random.normal(key, (P,)),
            'Lambda_im': jax.random.normal(jax.random.split(key)[0], (P,)),
            'B': jax.random.normal(key, (P, 128, 2)),
            'C': jax.random.normal(key, (128, P, 2)),
            'D': jax.random.normal(key, (128,)),
            'log_step': jax.random.normal(key, (P, 1)),
        }

        es_ssm = _convert_ssm(flax_ssm)

        # Lambda_re unchanged
        assert jnp.allclose(es_ssm['Lambda_re'], flax_ssm['Lambda_re']), (
            f"Lambda_re changed! max_diff={jnp.max(jnp.abs(es_ssm['Lambda_re'] - flax_ssm['Lambda_re']))}"
        )

        # Lambda_im unchanged
        assert jnp.allclose(es_ssm['Lambda_im'], flax_ssm['Lambda_im']), (
            f"Lambda_im changed! max_diff={jnp.max(jnp.abs(es_ssm['Lambda_im'] - flax_ssm['Lambda_im']))}"
        )

    def test_b_unchanged(self):
        """B matrix should be identical after conversion."""
        key = jax.random.PRNGKey(123)
        P, H = 64, 128

        flax_ssm = {
            'Lambda_re': jax.random.normal(key, (P,)),
            'Lambda_im': jax.random.normal(key, (P,)),
            'B': jax.random.normal(key, (P, H, 2)),  # (P, H, 2) for complex
            'C': jax.random.normal(key, (H, P, 2)),
            'D': jax.random.normal(key, (H,)),
            'log_step': jax.random.normal(key, (P, 1)),
        }

        es_ssm = _convert_ssm(flax_ssm)

        assert jnp.allclose(es_ssm['B'], flax_ssm['B']), (
            f"B changed! max_diff={jnp.max(jnp.abs(es_ssm['B'] - flax_ssm['B']))}"
        )

    def test_c_unchanged(self):
        """C matrix should be identical after conversion."""
        key = jax.random.PRNGKey(456)
        P, H = 64, 128

        flax_ssm = {
            'Lambda_re': jax.random.normal(key, (P,)),
            'Lambda_im': jax.random.normal(key, (P,)),
            'B': jax.random.normal(key, (P, H, 2)),
            'C': jax.random.normal(key, (H, P, 2)),  # (H, P, 2) for complex
            'D': jax.random.normal(key, (H,)),
            'log_step': jax.random.normal(key, (P, 1)),
        }

        es_ssm = _convert_ssm(flax_ssm)

        assert jnp.allclose(es_ssm['C'], flax_ssm['C']), (
            f"C changed! max_diff={jnp.max(jnp.abs(es_ssm['C'] - flax_ssm['C']))}"
        )

    def test_d_unchanged(self):
        """D (feedthrough) should be identical after conversion."""
        key = jax.random.PRNGKey(789)
        P, H = 64, 128

        flax_ssm = {
            'Lambda_re': jax.random.normal(key, (P,)),
            'Lambda_im': jax.random.normal(key, (P,)),
            'B': jax.random.normal(key, (P, H, 2)),
            'C': jax.random.normal(key, (H, P, 2)),
            'D': jax.random.normal(key, (H,)),  # Feedthrough
            'log_step': jax.random.normal(key, (P, 1)),
        }

        es_ssm = _convert_ssm(flax_ssm)

        assert jnp.allclose(es_ssm['D'], flax_ssm['D']), (
            f"D changed! max_diff={jnp.max(jnp.abs(es_ssm['D'] - flax_ssm['D']))}"
        )

    def test_log_step_unchanged(self):
        """log_step (discretization step) should be identical after conversion."""
        key = jax.random.PRNGKey(999)
        P, H = 64, 128

        flax_ssm = {
            'Lambda_re': jax.random.normal(key, (P,)),
            'Lambda_im': jax.random.normal(key, (P,)),
            'B': jax.random.normal(key, (P, H, 2)),
            'C': jax.random.normal(key, (H, P, 2)),
            'D': jax.random.normal(key, (H,)),
            'log_step': jax.random.normal(key, (P, 1)),  # Learnable step
        }

        es_ssm = _convert_ssm(flax_ssm)

        assert jnp.allclose(es_ssm['log_step'], flax_ssm['log_step']), (
            f"log_step changed! max_diff={jnp.max(jnp.abs(es_ssm['log_step'] - flax_ssm['log_step']))}"
        )

    def test_roundtrip_ssm_params(self):
        """Flax -> ES -> Flax roundtrip should preserve all SSM params exactly."""
        key = jax.random.PRNGKey(42)
        P, H = 64, 128
        d_model = 256

        def make_ssm():
            return {
                'Lambda_re': jax.random.normal(key, (P,)),
                'Lambda_im': jax.random.normal(key, (P,)),
                'B': jax.random.normal(key, (P, H, 2)),
                'C': jax.random.normal(key, (H, P, 2)),
                'D': jax.random.normal(key, (H,)),
                'log_step': jax.random.normal(key, (P, 1)),
            }

        def make_seq_layer():
            return {
                'seq': make_ssm(),
                'norm': {
                    'scale': jnp.ones(d_model),
                    'bias': jnp.zeros(d_model),
                },
                'out2': {
                    'kernel': jax.random.normal(key, (d_model, d_model)),
                    'bias': jnp.zeros(d_model),
                },
            }

        # Create Flax params with full structure
        original_flax = {
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

        # Flax -> ES
        es_params = convert_flax_to_es(original_flax, config)

        # ES -> Flax
        roundtrip_flax = convert_es_to_flax(es_params, config)

        # Verify SSM params in message_encoder layer_0
        orig_ssm = original_flax['message_encoder']['layers_0']['seq']
        rt_ssm = roundtrip_flax['message_encoder']['layers_0']['seq']

        for param_name in ['Lambda_re', 'Lambda_im', 'B', 'C', 'D', 'log_step']:
            assert jnp.allclose(orig_ssm[param_name], rt_ssm[param_name]), (
                f"Roundtrip changed {param_name}! "
                f"max_diff={jnp.max(jnp.abs(orig_ssm[param_name] - rt_ssm[param_name]))}"
            )

        # Verify SSM params in fused_s5 layer_0
        orig_fused_ssm = original_flax['fused_s5']['layers_0']['seq']
        rt_fused_ssm = roundtrip_flax['fused_s5']['layers_0']['seq']

        for param_name in ['Lambda_re', 'Lambda_im', 'B', 'C', 'D', 'log_step']:
            assert jnp.allclose(orig_fused_ssm[param_name], rt_fused_ssm[param_name]), (
                f"Roundtrip changed fused_s5 {param_name}! "
                f"max_diff={jnp.max(jnp.abs(orig_fused_ssm[param_name] - rt_fused_ssm[param_name]))}"
            )


class TestLayerNaming:
    """Test layers_N -> layer_N naming conversion."""

    def test_layers_to_layer_message_encoder(self):
        """layers_0, layers_1 -> layer_0, layer_1 in message_encoder."""
        key = jax.random.PRNGKey(42)
        d_model = 64
        P, H = 32, 64

        def make_ssm():
            return {
                'Lambda_re': jax.random.normal(key, (P,)),
                'Lambda_im': jax.random.normal(key, (P,)),
                'B': jax.random.normal(key, (P, H, 2)),
                'C': jax.random.normal(key, (H, P, 2)),
                'D': jax.random.normal(key, (H,)),
                'log_step': jax.random.normal(key, (P, 1)),
            }

        def make_seq_layer():
            return {
                'seq': make_ssm(),
                'norm': {'scale': jnp.ones(d_model), 'bias': jnp.zeros(d_model)},
            }

        flax_encoder = {
            'encoder': {'embedding': jax.random.normal(key, (1000, d_model))},
            'layers_0': make_seq_layer(),
            'layers_1': make_seq_layer(),
            'layers_2': make_seq_layer(),
        }

        es_encoder = _convert_message_encoder(flax_encoder, n_layers=3, activation='half_glu1')

        # Verify naming conversion
        assert 'layer_0' in es_encoder, "Missing layer_0"
        assert 'layer_1' in es_encoder, "Missing layer_1"
        assert 'layer_2' in es_encoder, "Missing layer_2"
        assert 'layers_0' not in es_encoder, "layers_0 should be renamed to layer_0"
        assert 'layers_1' not in es_encoder, "layers_1 should be renamed to layer_1"
        assert 'layers_2' not in es_encoder, "layers_2 should be renamed to layer_2"

    def test_layers_to_layer_fused_encoder(self):
        """layers_N -> layer_N in fused encoder (StackedEncoderModel)."""
        key = jax.random.PRNGKey(123)
        d_model = 64
        P, H = 32, 64

        def make_ssm():
            return {
                'Lambda_re': jax.random.normal(key, (P,)),
                'Lambda_im': jax.random.normal(key, (P,)),
                'B': jax.random.normal(key, (P, H, 2)),
                'C': jax.random.normal(key, (H, P, 2)),
                'D': jax.random.normal(key, (H,)),
                'log_step': jax.random.normal(key, (P, 1)),
            }

        def make_seq_layer():
            return {
                'seq': make_ssm(),
                'norm': {'scale': jnp.ones(d_model), 'bias': jnp.zeros(d_model)},
            }

        flax_encoder = {
            'encoder': {
                'kernel': jax.random.normal(key, (d_model * 2, d_model)),
                'bias': jnp.zeros(d_model),
            },
            'layers_0': make_seq_layer(),
            'layers_1': make_seq_layer(),
            'layers_2': make_seq_layer(),
            'layers_3': make_seq_layer(),
        }

        es_encoder = _convert_stacked_encoder(flax_encoder, n_layers=4, activation='half_glu1')

        # Verify naming
        for i in range(4):
            assert f'layer_{i}' in es_encoder, f"Missing layer_{i}"
            assert f'layers_{i}' not in es_encoder, f"layers_{i} should be renamed"

    def test_pre_post_layers_naming(self):
        """pre_layers_N -> pre_layer_N, post_layers_N -> post_layer_N in book_encoder."""
        key = jax.random.PRNGKey(456)
        d_model = 64
        P, H = 32, 64

        def make_ssm():
            return {
                'Lambda_re': jax.random.normal(key, (P,)),
                'Lambda_im': jax.random.normal(key, (P,)),
                'B': jax.random.normal(key, (P, H, 2)),
                'C': jax.random.normal(key, (H, P, 2)),
                'D': jax.random.normal(key, (H,)),
                'log_step': jax.random.normal(key, (P, 1)),
            }

        def make_seq_layer():
            return {
                'seq': make_ssm(),
                'norm': {'scale': jnp.ones(d_model), 'bias': jnp.zeros(d_model)},
            }

        flax_book = {
            'pre_layers_0': make_seq_layer(),
            'pre_layers_1': make_seq_layer(),
            'projection': {
                'kernel': jax.random.normal(key, (503, d_model)),
                'bias': jnp.zeros(d_model),
            },
            'post_layers_0': make_seq_layer(),
        }

        es_book = _convert_book_encoder(
            flax_book, n_pre_layers=2, n_post_layers=1, activation='half_glu1'
        )

        # Verify pre_layers naming
        assert 'pre_layer_0' in es_book, "Missing pre_layer_0"
        assert 'pre_layer_1' in es_book, "Missing pre_layer_1"
        assert 'pre_layers_0' not in es_book, "pre_layers_0 should be renamed"
        assert 'pre_layers_1' not in es_book, "pre_layers_1 should be renamed"

        # Verify post_layers naming
        assert 'post_layer_0' in es_book, "Missing post_layer_0"
        assert 'post_layers_0' not in es_book, "post_layers_0 should be renamed"

        # Verify projection -> proj
        assert 'proj' in es_book, "Missing proj (from projection)"
        assert 'projection' not in es_book, "projection should be renamed to proj"

    def test_seq_to_ssm_naming(self):
        """seq -> ssm naming in sequence layers."""
        key = jax.random.PRNGKey(789)
        d_model = 64
        P, H = 32, 64

        flax_seq_layer = {
            'seq': {  # Flax uses 'seq'
                'Lambda_re': jax.random.normal(key, (P,)),
                'Lambda_im': jax.random.normal(key, (P,)),
                'B': jax.random.normal(key, (P, H, 2)),
                'C': jax.random.normal(key, (H, P, 2)),
                'D': jax.random.normal(key, (H,)),
                'log_step': jax.random.normal(key, (P, 1)),
            },
            'norm': {'scale': jnp.ones(d_model), 'bias': jnp.zeros(d_model)},
        }

        es_seq_layer = _convert_sequence_layer(flax_seq_layer, activation='half_glu1')

        assert 'ssm' in es_seq_layer, "Missing ssm (from seq)"
        assert 'seq' not in es_seq_layer, "seq should be renamed to ssm"

    def test_full_model_naming(self):
        """Full model conversion should have correct naming throughout."""
        key = jax.random.PRNGKey(42)
        d_model = 64
        P, H = 32, 64

        def make_ssm():
            return {
                'Lambda_re': jax.random.normal(key, (P,)),
                'Lambda_im': jax.random.normal(key, (P,)),
                'B': jax.random.normal(key, (P, H, 2)),
                'C': jax.random.normal(key, (H, P, 2)),
                'D': jax.random.normal(key, (H,)),
                'log_step': jax.random.normal(key, (P, 1)),
            }

        def make_seq_layer():
            return {
                'seq': make_ssm(),
                'norm': {'scale': jnp.ones(d_model), 'bias': jnp.zeros(d_model)},
                'out2': {
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

        # Check top-level naming
        assert 'message_encoder' in es_params
        assert 'book_encoder' in es_params
        assert 'fused_encoder' in es_params  # fused_s5 -> fused_encoder
        assert 'decoder' in es_params
        assert 'fused_s5' not in es_params  # Should be renamed

        # Check message_encoder
        msg_enc = es_params['message_encoder']
        assert 'layer_0' in msg_enc
        assert 'layer_1' in msg_enc
        assert 'ssm' in msg_enc['layer_0']  # seq -> ssm
        assert 'norm' in msg_enc['layer_0']

        # Check book_encoder
        book_enc = es_params['book_encoder']
        assert 'pre_layer_0' in book_enc
        assert 'proj' in book_enc  # projection -> proj
        assert 'post_layer_0' in book_enc

        # Check fused_encoder
        fused_enc = es_params['fused_encoder']
        assert 'input_proj' in fused_enc  # encoder -> input_proj
        assert 'layer_0' in fused_enc
        assert 'layer_1' in fused_enc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
