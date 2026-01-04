"""
Test: ES_PaddedLobPredModel end-to-end forward pass
File: es_lobs5/tests/test_lob_model.py
"""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.models.lob_model import ES_PaddedLobPredModel
from es_lobs5.models.common import CommonParams, simple_es_tree_key
from es_lobs5.utils.import_utils import get_all_noisers


def test_model_forward_pass():
    """Test: End-to-end forward pass produces valid output."""
    key = jax.random.PRNGKey(42)

    # Model config (small for testing)
    d_output = 12012  # 22-token mode vocab
    d_model = 64
    d_book = 503  # 3 + 500 depth
    n_message_layers = 2
    n_fused_layers = 2
    ssm_size = 64
    blocks = 4

    # Initialize model
    model_init = ES_PaddedLobPredModel.rand_init(
        key,
        d_output=d_output,
        d_model=d_model,
        d_book=d_book,
        n_message_layers=n_message_layers,
        n_fused_layers=n_fused_layers,
        ssm_size=ssm_size,
        blocks=blocks,
        activation='half_glu1',
        mode='pool',
    )

    # Setup noiser (noop for testing)
    noisers = get_all_noisers()
    noiser = noisers['noop']
    es_tree_key = simple_es_tree_key(model_init.params, key, model_init.scan_map)

    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=model_init.frozen_params,
        params=model_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )

    # Create inputs
    L_m = 500  # Message sequence length
    L_b = 1    # Book sequence length (typically 1)

    x_m = jax.random.randint(key, (L_m,), 0, d_output)  # Token indices
    x_b = jax.random.normal(key, (L_b, d_book))  # Book features

    # Forward pass
    log_probs = ES_PaddedLobPredModel._forward(common_params, x_m, x_b)

    # Verify output
    assert log_probs.shape == (d_output,), f"Output shape wrong: {log_probs.shape}"
    assert not jnp.any(jnp.isnan(log_probs)), "NaN in output"

    # Verify log probabilities sum to ~1 (log-space)
    probs_sum = jnp.exp(log_probs).sum()
    assert jnp.abs(probs_sum - 1.0) < 0.01, f"Probabilities don't sum to 1: {probs_sum}"

    print(f"[PASS] Forward pass: output shape {log_probs.shape}, prob sum {probs_sum:.4f}")
    return True


def test_model_modes():
    """Test: Different pooling modes produce expected output shapes."""
    key = jax.random.PRNGKey(123)

    d_output = 2112  # 24-token mode vocab
    d_model = 32
    d_book = 103
    ssm_size = 32
    blocks = 2

    for mode in ['pool', 'last', 'ema']:
        model_init = ES_PaddedLobPredModel.rand_init(
            key, d_output=d_output, d_model=d_model, d_book=d_book,
            n_message_layers=1, n_fused_layers=1,
            ssm_size=ssm_size, blocks=blocks,
            mode=mode,
        )

        noisers = get_all_noisers()
        noiser = noisers['noop']
        es_tree_key = simple_es_tree_key(model_init.params, key, model_init.scan_map)

        common_params = CommonParams(
            noiser=noiser,
            frozen_noiser_params=None,
            noiser_params=None,
            frozen_params=model_init.frozen_params,
            params=model_init.params,
            es_tree_key=es_tree_key,
            iterinfo=None,
        )

        x_m = jax.random.randint(key, (100,), 0, d_output)
        x_b = jax.random.normal(key, (1, d_book))

        log_probs = ES_PaddedLobPredModel._forward(common_params, x_m, x_b)

        # All pooling modes should produce single prediction
        assert log_probs.shape == (d_output,), f"Mode {mode}: wrong shape {log_probs.shape}"

        print(f"[PASS] Mode '{mode}': output shape {log_probs.shape}")

    return True


def test_autoregressive_forward():
    """Test: Autoregressive forward returns per-token predictions."""
    key = jax.random.PRNGKey(456)

    d_output = 12012
    d_model = 32
    d_book = 103
    L_m = 100
    ssm_size = 32
    blocks = 2

    model_init = ES_PaddedLobPredModel.rand_init(
        key, d_output=d_output, d_model=d_model, d_book=d_book,
        n_message_layers=1, n_fused_layers=1,
        ssm_size=ssm_size, blocks=blocks,
        mode='none',  # No pooling for AR
    )

    noisers = get_all_noisers()
    noiser = noisers['noop']
    es_tree_key = simple_es_tree_key(model_init.params, key, model_init.scan_map)

    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=model_init.frozen_params,
        params=model_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )

    x_m = jax.random.randint(key, (L_m,), 0, d_output)
    x_b = jax.random.normal(key, (1, d_book))

    # Use _forward_ar for autoregressive predictions
    log_probs = ES_PaddedLobPredModel._forward_ar(common_params, x_m, x_b)

    # Should return (L_m, d_output) for per-token predictions
    expected_shape = (L_m, d_output)
    assert log_probs.shape == expected_shape, f"AR shape wrong: {log_probs.shape} vs {expected_shape}"

    print(f"[PASS] Autoregressive: output shape {log_probs.shape}")
    return True


if __name__ == "__main__":
    test_model_forward_pass()
    test_model_modes()
    test_autoregressive_forward()
    print("All LOB model tests passed!")
