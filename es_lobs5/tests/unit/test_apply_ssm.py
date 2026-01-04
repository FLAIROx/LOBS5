"""Test apply_ssm function."""
import pytest
import jax
import jax.numpy as jnp

from es_lobs5.adapters.ssm_adapter import apply_ssm, apply_ssm_rnn, discretize_zoh


class TestApplySSM:
    """Tests for apply_ssm function with discretize_zoh preprocessing."""

    @pytest.fixture
    def ssm_params(self):
        """Create SSM parameters for testing.

        Returns continuous-time parameters that need discretization via discretize_zoh.
        """
        L = 64  # sequence length
        H = 32  # hidden dimension
        P = 16  # state dimension

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        # Continuous-time SSM parameters
        # Lambda: (P,) complex - diagonal state matrix eigenvalues (stable: Re < 0)
        Lambda_re = -jax.random.uniform(keys[0], (P,), minval=0.1, maxval=1.0)
        Lambda_im = jax.random.uniform(keys[1], (P,), minval=-3.14, maxval=3.14)
        Lambda = Lambda_re + 1j * Lambda_im

        # B_tilde: (P, H) complex - input matrix
        B_tilde = (jax.random.normal(keys[2], (P, H)) +
                   1j * jax.random.normal(keys[3], (P, H))) * 0.1

        # C_tilde: (H, P) complex - output matrix
        C_tilde = (jax.random.normal(keys[4], (H, P)) +
                   1j * jax.random.normal(jax.random.split(keys[4])[0], (H, P))) * 0.1

        # Discretization step (Delta)
        Delta = jnp.ones(P) * 0.01  # uniform step size

        return {
            'Lambda': Lambda,
            'B_tilde': B_tilde,
            'C_tilde': C_tilde,
            'Delta': Delta,
            'L': L,
            'H': H,
            'P': P,
        }

    def test_apply_ssm_output_shape(self, ssm_params):
        """Test: input (L, H), output (L, H)."""
        L = ssm_params['L']
        H = ssm_params['H']

        # Discretize continuous-time SSM using ZOH
        Lambda_bar, B_bar = discretize_zoh(
            ssm_params['Lambda'],
            ssm_params['B_tilde'],
            ssm_params['Delta']
        )

        # Create input sequence
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (L, H))

        # Apply SSM (unidirectional, no conjugate symmetry)
        y = apply_ssm(
            Lambda_bar,
            B_bar,
            ssm_params['C_tilde'],
            x,
            conj_sym=False,
            bidirectional=False,
        )

        assert y.shape == (L, H), f"Expected shape ({L}, {H}), got {y.shape}"
        assert y.dtype == jnp.float32, f"Expected dtype float32, got {y.dtype}"

    def test_apply_ssm_rnn_equivalence(self, ssm_params):
        """Test: parallel mode vs RNN mode output max diff < 1e-5."""
        L = ssm_params['L']
        H = ssm_params['H']
        P = ssm_params['P']

        # Discretize continuous-time SSM using ZOH
        Lambda_bar, B_bar = discretize_zoh(
            ssm_params['Lambda'],
            ssm_params['B_tilde'],
            ssm_params['Delta']
        )

        # Create input sequence
        key = jax.random.PRNGKey(123)
        x = jax.random.normal(key, (L, H))

        # Parallel mode (apply_ssm)
        y_parallel = apply_ssm(
            Lambda_bar,
            B_bar,
            ssm_params['C_tilde'],
            x,
            conj_sym=False,
            bidirectional=False,
        )

        # RNN mode (apply_ssm_rnn) with zero initial hidden state
        hidden_init = jnp.zeros((1, P), dtype=jnp.complex64)
        hidden_out, y_rnn = apply_ssm_rnn(
            Lambda_bar,
            B_bar,
            ssm_params['C_tilde'],
            hidden_init,
            x,
            resets=None,
            conj_sym=False,
            bidirectional=False,
        )

        # Check output shapes
        assert y_rnn.shape == (L, H), f"RNN output shape mismatch: {y_rnn.shape}"
        assert hidden_out.shape == (1, P), f"Hidden state shape mismatch: {hidden_out.shape}"

        # Check equivalence: max diff < 1e-5
        max_diff = jnp.max(jnp.abs(y_parallel - y_rnn))
        assert max_diff < 1e-5, f"Parallel vs RNN max diff = {max_diff}, expected < 1e-5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
