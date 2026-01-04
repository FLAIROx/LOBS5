"""Test ES_StackedEncoder."""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.models.encoder import ES_StackedEncoder
from es_lobs5.models.common import CommonParams, simple_es_tree_key


# Simple noop noiser to avoid optax dependency
class NoopNoiser:
    """Minimal noiser that returns params unchanged (for testing)."""
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, **kwargs):
        return {}, {}

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return x @ param.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return x @ param

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return param[x]

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        return param

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        return raw_scores

    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        return noiser_params, params


def test_stacked_encoder_layers():
    """Test: N layers produce N hidden states in RNN mode."""
    key = jax.random.PRNGKey(42)

    # Test with different layer counts
    d_input = 16
    d_model = 32
    ssm_size = 64
    blocks = 4

    for n_layers in [1, 2, 4]:
        # Initialize encoder
        encoder_init = ES_StackedEncoder.rand_init(
            key,
            d_input=d_input,
            d_model=d_model,
            n_layers=n_layers,
            ssm_size=ssm_size,
            blocks=blocks,
        )

        # Initialize hidden states
        batch_size = 2
        hiddens = ES_StackedEncoder.initialize_carry(
            batch_size=batch_size,
            ssm_size=ssm_size,
            n_layers=n_layers,
            conj_sym=True,
        )

        # Verify N layers produce N hidden states
        assert len(hiddens) == n_layers, f"Expected {n_layers} hidden states, got {len(hiddens)}"

        # Verify each hidden state has correct shape
        # For conj_sym=True, hidden_size = ssm_size // 2
        expected_hidden_size = ssm_size // 2
        for i, h in enumerate(hiddens):
            expected_shape = (batch_size, 1, expected_hidden_size)
            assert h.shape == expected_shape, f"Hidden {i} shape {h.shape} != {expected_shape}"

        print(f"[PASS] test_stacked_encoder_layers: n_layers={n_layers} -> {len(hiddens)} hidden states")

    return True


def test_stacked_encoder_output_shape():
    """Test: (L, d_input) -> (L, d_model) shape transformation."""
    key = jax.random.PRNGKey(123)

    # Test dimensions
    seq_len = 64
    d_input = 16
    d_model = 32
    n_layers = 3
    ssm_size = 64
    blocks = 4

    # Initialize encoder
    encoder_init = ES_StackedEncoder.rand_init(
        key,
        d_input=d_input,
        d_model=d_model,
        n_layers=n_layers,
        ssm_size=ssm_size,
        blocks=blocks,
    )

    # Setup noiser (noop for testing)
    noiser = NoopNoiser
    es_tree_key = simple_es_tree_key(encoder_init.params, key, encoder_init.scan_map)

    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=encoder_init.frozen_params,
        params=encoder_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )

    # Create input: (L, d_input)
    x = jax.random.normal(key, (seq_len, d_input))

    # Forward pass
    out = ES_StackedEncoder._forward(common_params, x)

    # Verify output shape: (L, d_model)
    expected_shape = (seq_len, d_model)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
    assert not jnp.any(jnp.isnan(out)), "NaN in output"

    print(f"[PASS] test_stacked_encoder_output_shape: input {x.shape} -> output {out.shape}")
    return True


if __name__ == "__main__":
    test_stacked_encoder_layers()
    test_stacked_encoder_output_shape()
    print("All ES_StackedEncoder tests passed!")
