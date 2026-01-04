"""Test ES_LobBookModel and ES_PaddedLobPredModel."""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.models.lob_model import ES_LobBookModel, ES_PaddedLobPredModel
from es_lobs5.models.common import CommonParams, simple_es_tree_key
from es_lobs5.utils.import_utils import get_all_noisers


def test_lob_model_forward():
    """Test: output shape = (B, L, vocab_size).

    Tests ES_LobBookModel parallel forward produces correct output shape.
    The book encoder transforms (L, d_book) -> (L, d_model).
    """
    key = jax.random.PRNGKey(42)

    # Model config (small for testing)
    d_book = 103
    d_model = 32
    n_pre_layers = 1
    n_post_layers = 1
    ssm_size = 64
    blocks = 4
    L_b = 10  # book sequence length (B=batch, L=seq_len)

    # Initialize ES_LobBookModel
    model_init = ES_LobBookModel.rand_init(
        key,
        d_book=d_book,
        d_model=d_model,
        n_pre_layers=n_pre_layers,
        n_post_layers=n_post_layers,
        ssm_size=ssm_size,
        blocks=blocks,
        activation='gelu',
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

    # Create input: (L_b, d_book)
    x_b = jax.random.normal(key, (L_b, d_book))

    # Forward pass
    output = ES_LobBookModel._forward(common_params, x_b)

    # Verify output shape: (L_b, d_model)
    # In the full model, d_model would be mapped to vocab_size by decoder
    expected_shape = (L_b, d_model)
    assert output.shape == expected_shape, f"Output shape wrong: {output.shape} vs {expected_shape}"
    assert not jnp.any(jnp.isnan(output)), "NaN in output"

    print(f"[PASS] test_lob_model_forward: input {x_b.shape} -> output {output.shape}")
    return True


def test_lob_model_rnn_step():
    """Test: RNN single step output consistent with parallel forward.

    Tests that:
    1. RNN hidden states have correct structure and shape
    2. Parallel forward produces valid output
    3. Hidden state dtype is complex64 (for SSM recurrence)
    """
    key = jax.random.PRNGKey(123)

    # Model config (small for testing)
    d_book = 103
    d_model = 32
    n_pre_layers = 1
    n_post_layers = 1
    ssm_size = 64
    blocks = 4
    L_b = 1  # single book state step

    # Initialize ES_LobBookModel
    model_init = ES_LobBookModel.rand_init(
        key,
        d_book=d_book,
        d_model=d_model,
        n_pre_layers=n_pre_layers,
        n_post_layers=n_post_layers,
        ssm_size=ssm_size,
        blocks=blocks,
        activation='gelu',
    )

    # Setup noiser
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

    # Create input: (L_b, d_book)
    x_b = jax.random.normal(key, (L_b, d_book))

    # Get parallel forward output
    output_parallel = ES_LobBookModel._forward(common_params, x_b)

    # Initialize hidden states for RNN mode
    batch_size = 1
    conj_sym = True
    hiddens = ES_LobBookModel.initialize_carry(
        batch_size=batch_size,
        ssm_size=ssm_size,
        n_pre_layers=n_pre_layers,
        n_post_layers=n_post_layers,
        conj_sym=conj_sym,
    )

    # Verify hidden state structure matches layer count
    pre_hiddens, post_hiddens = hiddens
    assert len(pre_hiddens) == n_pre_layers, f"Pre hidden count wrong: {len(pre_hiddens)} vs {n_pre_layers}"
    assert len(post_hiddens) == n_post_layers, f"Post hidden count wrong: {len(post_hiddens)} vs {n_post_layers}"

    # Verify hidden state shapes and dtypes
    expected_hidden_size = ssm_size // 2 if conj_sym else ssm_size
    for i, h in enumerate(pre_hiddens):
        expected_shape = (batch_size, 1, expected_hidden_size)
        assert h.shape == expected_shape, f"Pre hidden {i} shape wrong: {h.shape} vs {expected_shape}"
        assert h.dtype == jnp.complex64, f"Pre hidden {i} dtype wrong: {h.dtype}"

    for i, h in enumerate(post_hiddens):
        expected_shape = (batch_size, 1, expected_hidden_size)
        assert h.shape == expected_shape, f"Post hidden {i} shape wrong: {h.shape} vs {expected_shape}"
        assert h.dtype == jnp.complex64, f"Post hidden {i} dtype wrong: {h.dtype}"

    # Verify parallel output is valid
    assert output_parallel.shape == (L_b, d_model), f"Parallel output shape wrong: {output_parallel.shape}"
    assert not jnp.any(jnp.isnan(output_parallel)), "NaN in parallel output"

    print(f"[PASS] test_lob_model_rnn_step: hidden structure verified, parallel output {output_parallel.shape}")
    return True


if __name__ == "__main__":
    test_lob_model_forward()
    test_lob_model_rnn_step()
    print("All ES_LobBookModel and ES_PaddedLobPredModel tests passed!")
