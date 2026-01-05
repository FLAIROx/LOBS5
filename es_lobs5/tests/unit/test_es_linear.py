"""Test ES_Linear layer."""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.models.common import ES_Linear, CommonParams, simple_es_tree_key


# Local NoopNoiser mock - avoids HyperscaleES dependency (requires optax, etc.)
class NoopNoiser:
    """Noop noiser for testing: returns params unchanged, no noise."""
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None, **kwargs):
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


def test_linear_shapes():
    """Test: (B, L, in) -> (B, L, out) shape transformation."""
    key = jax.random.PRNGKey(42)

    # Test dimensions
    batch_size = 4
    seq_len = 16
    in_dim = 32
    out_dim = 64

    # Initialize layer
    linear_init = ES_Linear.rand_init(key, in_dim=in_dim, out_dim=out_dim, use_bias=True)

    # Setup noiser (noop for testing)
    noiser = NoopNoiser
    es_tree_key = simple_es_tree_key(linear_init.params, key, linear_init.scan_map)

    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=linear_init.frozen_params,
        params=linear_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )

    # Create input: (B, L, in_dim)
    x = jax.random.normal(key, (batch_size, seq_len, in_dim))

    # Forward pass
    out = ES_Linear._forward(common_params, x)

    # Verify output shape: (B, L, out_dim)
    expected_shape = (batch_size, seq_len, out_dim)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
    assert not jnp.any(jnp.isnan(out)), "NaN in output"

    print(f"[PASS] test_linear_shapes: input {x.shape} -> output {out.shape}")
    return True


def test_linear_with_bias():
    """Test: bias is correctly added to the output."""
    key = jax.random.PRNGKey(123)

    in_dim = 16
    out_dim = 8

    # Initialize with bias
    linear_with_bias = ES_Linear.rand_init(key, in_dim=in_dim, out_dim=out_dim, use_bias=True)

    # Initialize without bias
    linear_no_bias = ES_Linear.rand_init(key, in_dim=in_dim, out_dim=out_dim, use_bias=False)

    # Verify bias parameter exists/not exists
    assert 'bias' in linear_with_bias.params, "Bias should be in params when use_bias=True"
    assert 'bias' not in linear_no_bias.params, "Bias should not be in params when use_bias=False"

    # Setup noiser
    noiser = NoopNoiser

    # Test forward with bias
    es_tree_key = simple_es_tree_key(linear_with_bias.params, key, linear_with_bias.scan_map)
    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=linear_with_bias.frozen_params,
        params=linear_with_bias.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )

    # Zero input should produce bias as output
    x_zero = jnp.zeros((1, in_dim))
    out_zero = ES_Linear._forward(common_params, x_zero)

    # The output for zero input should be the bias (since weight @ 0 = 0)
    # Bias is initialized to zeros, so output should be zeros
    assert out_zero.shape == (1, out_dim), f"Output shape wrong: {out_zero.shape}"
    assert jnp.allclose(out_zero, 0.0, atol=1e-6), f"Zero input should give bias (zeros): {out_zero}"

    # Test forward without bias
    es_tree_key_no_bias = simple_es_tree_key(linear_no_bias.params, key, linear_no_bias.scan_map)
    common_params_no_bias = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=linear_no_bias.frozen_params,
        params=linear_no_bias.params,
        es_tree_key=es_tree_key_no_bias,
        iterinfo=None,
    )

    out_no_bias = ES_Linear._forward(common_params_no_bias, x_zero)
    assert out_no_bias.shape == (1, out_dim), f"Output shape wrong: {out_no_bias.shape}"
    assert jnp.allclose(out_no_bias, 0.0, atol=1e-6), "Zero input with no bias should give zeros"

    print(f"[PASS] test_linear_with_bias: bias correctly added")
    return True


if __name__ == "__main__":
    test_linear_shapes()
    test_linear_with_bias()
    print("All ES_Linear tests passed!")
