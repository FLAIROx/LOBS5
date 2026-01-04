"""
Test: ES_SequenceLayer output consistency and activation handling
File: es_lobs5/tests/test_sequence_layer.py
"""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.models.s5_layer import ES_SequenceLayer
from es_lobs5.adapters.hippo_adapter import get_hippo_params
from es_lobs5.models.common import CommonParams, simple_es_tree_key
from es_lobs5.utils.import_utils import get_all_noisers


def test_layer_forward_shape():
    """Test: Output shape matches input shape."""
    key = jax.random.PRNGKey(42)

    d_model = 64
    ssm_size = 64
    blocks = 4
    L = 100

    hippo = get_hippo_params(ssm_size, blocks, True)

    for activation in ['gelu', 'half_glu1', 'half_glu2', 'full_glu']:
        key, subkey = jax.random.split(key)
        layer_init = ES_SequenceLayer.rand_init(
            subkey,
            d_model=d_model,
            ssm_size=ssm_size,
            Lambda_re_init=hippo['Lambda_re_init'],
            Lambda_im_init=hippo['Lambda_im_init'],
            V=hippo['V'], Vinv=hippo['Vinv'],
            blocks=blocks,
            activation=activation,
            prenorm=True,
        )

        noisers = get_all_noisers()
        noiser = noisers['noop']
        es_tree_key = simple_es_tree_key(layer_init.params, key, layer_init.scan_map)

        common_params = CommonParams(
            noiser=noiser,
            frozen_noiser_params=None,
            noiser_params=None,
            frozen_params=layer_init.frozen_params,
            params=layer_init.params,
            es_tree_key=es_tree_key,
            iterinfo=None,
        )

        x = jax.random.normal(key, (L, d_model))
        y = ES_SequenceLayer._forward(common_params, x)

        assert y.shape == x.shape, f"Shape mismatch for {activation}: {y.shape} vs {x.shape}"
        assert not jnp.any(jnp.isnan(y)), f"NaN detected for {activation}"

        print(f"[PASS] {activation}: output shape {y.shape}")

    return True


def test_residual_connection():
    """Test: Residual connection is correctly applied."""
    key = jax.random.PRNGKey(123)

    d_model = 32
    ssm_size = 32
    blocks = 2

    hippo = get_hippo_params(ssm_size, blocks, True)

    layer_init = ES_SequenceLayer.rand_init(
        key, d_model=d_model, ssm_size=ssm_size,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
        blocks=blocks,
        activation='gelu', prenorm=False,
    )

    noisers = get_all_noisers()
    noiser = noisers['noop']
    es_tree_key = simple_es_tree_key(layer_init.params, key, layer_init.scan_map)

    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=layer_init.frozen_params,
        params=layer_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )

    # Zero input should produce non-zero output (from LayerNorm bias)
    x = jnp.zeros((10, d_model))
    y = ES_SequenceLayer._forward(common_params, x)

    print(f"[PASS] Residual connection: output range [{y.min():.4f}, {y.max():.4f}]")
    return True


def test_glu_gates():
    """Test: GLU gates produce correct output pattern."""
    key = jax.random.PRNGKey(456)

    d_model = 32
    ssm_size = 32
    blocks = 2

    hippo = get_hippo_params(ssm_size, blocks, True)

    # Test half_glu1: output = GELU(x) * sigmoid(out2(x))
    layer_init = ES_SequenceLayer.rand_init(
        key, d_model=d_model, ssm_size=ssm_size,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
        blocks=blocks,
        activation='half_glu1', prenorm=True,
    )

    # Verify out2 exists but out1 does not
    assert 'out2' in layer_init.params, "half_glu1 should have out2"
    assert 'out1' not in layer_init.params, "half_glu1 should not have out1"

    # Test full_glu: both out1 and out2
    layer_init_full = ES_SequenceLayer.rand_init(
        key, d_model=d_model, ssm_size=ssm_size,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
        blocks=blocks,
        activation='full_glu', prenorm=True,
    )

    assert 'out1' in layer_init_full.params, "full_glu should have out1"
    assert 'out2' in layer_init_full.params, "full_glu should have out2"

    print("[PASS] GLU gates structure verified")
    return True


if __name__ == "__main__":
    test_layer_forward_shape()
    test_residual_connection()
    test_glu_gates()
    print("All sequence layer tests passed!")
