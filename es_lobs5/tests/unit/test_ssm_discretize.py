"""Test discretize_zoh function."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import pytest
import jax
import jax.numpy as jnp

from es_lobs5.adapters.ssm_adapter import discretize_zoh


class TestDiscretizeZoh:
    """Tests for the discretize_zoh function."""

    def test_discretize_zoh_shape(self):
        """Lambda_bar shape = (P,), B_bar shape = (P, H)."""
        P = 64
        H = 128

        # Create test inputs
        Lambda = jnp.ones(P, dtype=jnp.float32) * (-0.5) + 1j * jnp.linspace(0.1, 1.0, P)
        B_tilde = jax.random.normal(jax.random.PRNGKey(0), (P, H)) + 1j * jax.random.normal(jax.random.PRNGKey(1), (P, H))
        Delta = jnp.ones(P, dtype=jnp.float32) * 0.01

        Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, Delta)

        assert Lambda_bar.shape == (P,), f"Expected Lambda_bar shape (P,)=({P},), got {Lambda_bar.shape}"
        assert B_bar.shape == (P, H), f"Expected B_bar shape (P, H)=({P}, {H}), got {B_bar.shape}"

        # Verify complex dtype
        assert jnp.issubdtype(Lambda_bar.dtype, jnp.complexfloating), f"Expected complex dtype, got {Lambda_bar.dtype}"
        assert jnp.issubdtype(B_bar.dtype, jnp.complexfloating), f"Expected complex dtype, got {B_bar.dtype}"

    def test_discretize_zoh_stability(self):
        """|Lambda_bar| < 1 for stable system (Re(Lambda) < 0)."""
        P = 128
        H = 256

        # Create stable continuous-time system: Re(Lambda) < 0
        # Eigenvalues in left-half plane guarantee stability
        Lambda_re = -jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (P,))) - 0.01  # Ensure negative
        Lambda_im = jax.random.normal(jax.random.PRNGKey(3), (P,))
        Lambda = Lambda_re + 1j * Lambda_im

        B_tilde = jax.random.normal(jax.random.PRNGKey(4), (P, H)) + 1j * jax.random.normal(jax.random.PRNGKey(5), (P, H))
        Delta = jnp.ones(P, dtype=jnp.float32) * 0.01

        Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, Delta)

        # For stable continuous system, discrete eigenvalues must satisfy |Lambda_bar| < 1
        # ZOH discretization: Lambda_bar = exp(Lambda * Delta)
        # If Re(Lambda) < 0 and Delta > 0, then |exp(Lambda * Delta)| = exp(Re(Lambda) * Delta) < 1
        Lambda_bar_abs = jnp.abs(Lambda_bar)

        assert jnp.all(Lambda_bar_abs < 1.0), (
            f"Stability violated: max |Lambda_bar| = {jnp.max(Lambda_bar_abs):.6f}, "
            f"should be < 1.0 for stable system"
        )

        # Additional check: verify the formula Lambda_bar = exp(Lambda * Delta)
        expected_Lambda_bar = jnp.exp(Lambda * Delta)
        assert jnp.allclose(Lambda_bar, expected_Lambda_bar, rtol=1e-5), (
            "Lambda_bar does not match expected exp(Lambda * Delta)"
        )

    def test_discretize_zoh_no_nan(self):
        """Verify no NaN values in output."""
        P = 64
        H = 128

        Lambda = jnp.ones(P, dtype=jnp.float32) * (-0.1) + 1j * jnp.ones(P) * 0.5
        B_tilde = jnp.ones((P, H), dtype=jnp.complex64)
        Delta = jnp.ones(P, dtype=jnp.float32) * 0.01

        Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, Delta)

        assert not jnp.any(jnp.isnan(Lambda_bar)), "Lambda_bar contains NaN"
        assert not jnp.any(jnp.isnan(B_bar)), "B_bar contains NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
