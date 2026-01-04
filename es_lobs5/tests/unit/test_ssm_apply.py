"""Test apply_ssm function."""
import pytest
import jax
import jax.numpy as jnp

from es_lobs5.adapters.ssm_adapter import apply_ssm, apply_ssm_rnn


class TestApplySSM:
    """Tests for apply_ssm and apply_ssm_rnn functions."""

    @pytest.fixture
    def ssm_params(self):
        """Create SSM parameters for testing."""
        L = 64  # sequence length
        H = 32  # hidden dimension
        P = 16  # state dimension

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        # Create discretized SSM parameters
        # Lambda_bar: (P,) complex - discretized state transition
        Lambda_bar = 0.9 * jnp.exp(1j * jax.random.uniform(keys[0], (P,)) * 2 * jnp.pi)

        # B_bar: (P, H) complex - discretized input matrix
        B_bar = (jax.random.normal(keys[1], (P, H)) +
                 1j * jax.random.normal(keys[2], (P, H))) * 0.1

        # C_tilde: (H, P) complex - output matrix
        C_tilde = (jax.random.normal(keys[3], (H, P)) +
                   1j * jax.random.normal(jax.random.split(keys[3])[0], (H, P))) * 0.1

        return {
            'Lambda_bar': Lambda_bar,
            'B_bar': B_bar,
            'C_tilde': C_tilde,
            'L': L,
            'H': H,
            'P': P,
        }

    def test_apply_ssm_output_shape(self, ssm_params):
        """Test: output shape = (L, H)"""
        L = ssm_params['L']
        H = ssm_params['H']

        # Create input sequence
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (L, H))

        # Test without conjugate symmetry, unidirectional
        y = apply_ssm(
            ssm_params['Lambda_bar'],
            ssm_params['B_bar'],
            ssm_params['C_tilde'],
            x,
            conj_sym=False,
            bidirectional=False,
        )

        assert y.shape == (L, H), f"Expected shape ({L}, {H}), got {y.shape}"
        assert y.dtype == jnp.float32, f"Expected dtype float32, got {y.dtype}"

        # Test with conjugate symmetry
        y_conj = apply_ssm(
            ssm_params['Lambda_bar'],
            ssm_params['B_bar'],
            ssm_params['C_tilde'],
            x,
            conj_sym=True,
            bidirectional=False,
        )

        assert y_conj.shape == (L, H), f"Expected shape ({L}, {H}), got {y_conj.shape}"

    def test_apply_ssm_rnn_equivalence(self, ssm_params):
        """Test: parallel vs RNN mode output max diff < 1e-5"""
        L = ssm_params['L']
        H = ssm_params['H']
        P = ssm_params['P']

        # Create input sequence
        key = jax.random.PRNGKey(123)
        x = jax.random.normal(key, (L, H))

        # Parallel mode (apply_ssm)
        y_parallel = apply_ssm(
            ssm_params['Lambda_bar'],
            ssm_params['B_bar'],
            ssm_params['C_tilde'],
            x,
            conj_sym=False,
            bidirectional=False,
        )

        # RNN mode (apply_ssm_rnn) with zero initial hidden state
        hidden_init = jnp.zeros((1, P), dtype=jnp.complex64)
        hidden_out, y_rnn = apply_ssm_rnn(
            ssm_params['Lambda_bar'],
            ssm_params['B_bar'],
            ssm_params['C_tilde'],
            hidden_init,
            x,
            resets=None,
            conj_sym=False,
            bidirectional=False,
        )

        # Check output shape
        assert y_rnn.shape == (L, H), f"RNN output shape mismatch: {y_rnn.shape}"
        assert hidden_out.shape == (1, P), f"Hidden state shape mismatch: {hidden_out.shape}"

        # Check equivalence: max diff < 1e-5
        max_diff = jnp.max(jnp.abs(y_parallel - y_rnn))
        assert max_diff < 1e-5, f"Parallel vs RNN max diff = {max_diff}, expected < 1e-5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
