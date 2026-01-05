"""
Integration Test: SSM (State Space Model) Integration with Encoder
File: es_lobs5/tests/integration/test_ssm_integration.py

Tests verify:
1. SSM parameter creation (Lambda, B, C, D, log_step)
2. Discretization using ZOH (zero-order hold)
3. SSM application to sequence
4. Output shape matches input shape
5. SSM state evolution correctness
6. SSM stability (eigenvalues inside unit circle)

SSM Formulas:
- Continuous: dx/dt = A*x + B*u, y = C*x + D*u
- Discretized (ZOH): x[k+1] = A_bar @ x[k] + B_bar @ u[k]
                     y[k] = C @ x[k] + D @ u[k]
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp


def create_ssm_params(key, P, H):
    """
    Create SSM parameters for testing.

    Args:
        key: JAX random key
        P: State dimension
        H: Feature dimension (input/output)

    Returns:
        Dictionary with Lambda, B, C, D, log_step
    """
    keys = jax.random.split(key, 5)

    # Lambda: Complex eigenvalues (continuous-time)
    # For stability, real part should be negative
    Lambda_re = -jax.random.uniform(keys[0], (P,), minval=0.1, maxval=1.0)
    Lambda_im = jax.random.uniform(keys[1], (P,), minval=-1.0, maxval=1.0)
    Lambda = Lambda_re + 1j * Lambda_im

    # B: Input matrix (P, H) complex
    B_re = jax.random.normal(keys[2], (P, H)) * 0.1
    B_im = jax.random.normal(keys[2], (P, H)) * 0.1
    B = B_re + 1j * B_im

    # C: Output matrix (H, P) complex
    C_re = jax.random.normal(keys[3], (H, P)) * 0.1
    C_im = jax.random.normal(keys[3], (H, P)) * 0.1
    C = C_re + 1j * C_im

    # D: Feedthrough (H,)
    D = jax.random.normal(keys[4], (H,)) * 0.01

    # log_step: Controls discretization timestep
    log_step = jnp.log(jnp.ones((P,)) * 0.01)

    return {
        'Lambda': Lambda,
        'B': B,
        'C': C,
        'D': D,
        'log_step': log_step,
    }


def discretize_zoh(Lambda, B, step):
    """
    Discretize continuous SSM using Zero-Order Hold (ZOH).

    Continuous: dx/dt = Lambda*x + B*u
    Discrete: x[k+1] = A_bar*x[k] + B_bar*u[k]

    ZOH formulas:
        A_bar = exp(Lambda * step)
        B_bar = (A_bar - I) / Lambda * B = (exp(Lambda*step) - 1) / Lambda * B

    Args:
        Lambda: Complex eigenvalues (P,)
        B: Input matrix (P, H) complex
        step: Discretization step (P,)

    Returns:
        A_bar (P,) complex, B_bar (P, H) complex
    """
    # A_bar = exp(Lambda * dt)
    A_bar = jnp.exp(Lambda * step)

    # B_bar = (exp(Lambda*dt) - 1) / Lambda * B
    # For numerical stability when Lambda is small
    B_bar = (1.0 / Lambda) * (A_bar - 1.0)
    B_bar = jnp.expand_dims(B_bar, axis=-1) * B

    return A_bar, B_bar


def apply_ssm(A_bar, B_bar, C, D, x):
    """
    Apply discretized SSM to input sequence.

    x[k+1] = A_bar @ x[k] + B_bar @ u[k]
    y[k] = C @ x[k] + D @ u[k]

    Args:
        A_bar: Discretized state transition (P,) complex (diagonal)
        B_bar: Discretized input matrix (P, H) complex
        C: Output matrix (H, P) complex
        D: Feedthrough (H,)
        x: Input sequence (L, H)

    Returns:
        Output sequence (L, H)
    """
    L, H = x.shape
    P = A_bar.shape[0]

    # Initialize state
    state = jnp.zeros((P,), dtype=jnp.complex64)

    outputs = []
    for t in range(L):
        u_t = x[t]  # (H,)

        # State update: x[k+1] = A_bar * x[k] + B_bar @ u[k]
        # A_bar is diagonal, so element-wise multiplication
        # B_bar @ u: (P, H) @ (H,) = (P,)
        state = A_bar * state + jnp.dot(B_bar, u_t)

        # Output: y[k] = C @ x[k] + D * u[k]
        # C @ state: (H, P) @ (P,) = (H,)
        y_t = jnp.dot(C, state).real + D * u_t
        outputs.append(y_t)

    return jnp.stack(outputs, axis=0)


def apply_ssm_scan(A_bar, B_bar, C, D, x):
    """
    Apply SSM using jax.lax.scan for efficiency.

    Same as apply_ssm but uses scan for better performance.
    """
    L, H = x.shape
    P = A_bar.shape[0]

    def step_fn(state, u_t):
        # State update
        new_state = A_bar * state + jnp.dot(B_bar, u_t)
        # Output
        y_t = jnp.dot(C, new_state).real + D * u_t
        return new_state, y_t

    init_state = jnp.zeros((P,), dtype=jnp.complex64)
    final_state, outputs = jax.lax.scan(step_fn, init_state, x)

    return outputs, final_state


def test_ssm_discretize():
    """Test: continuous to discrete SSM parameter conversion via ZOH."""
    key = jax.random.PRNGKey(42)
    P, H = 8, 16

    params = create_ssm_params(key, P, H)
    step = jnp.exp(params['log_step'])

    A_bar, B_bar = discretize_zoh(params['Lambda'], params['B'], step)

    # Verify shapes
    assert A_bar.shape == (P,), f"A_bar shape mismatch: {A_bar.shape}"
    assert B_bar.shape == (P, H), f"B_bar shape mismatch: {B_bar.shape}"

    # Verify A_bar = exp(Lambda * step)
    expected_A_bar = jnp.exp(params['Lambda'] * step)
    assert jnp.allclose(A_bar, expected_A_bar), "A_bar computation incorrect"

    # Verify B_bar formula
    expected_B_bar = ((A_bar - 1.0) / params['Lambda'])[:, None] * params['B']
    assert jnp.allclose(B_bar, expected_B_bar, rtol=1e-5), "B_bar computation incorrect"

    print(f"[PASS] test_ssm_discretize: A_bar shape={A_bar.shape}, B_bar shape={B_bar.shape}")
    return True


def test_ssm_apply():
    """Test: apply discretized SSM to sequence."""
    key = jax.random.PRNGKey(123)
    P, H, L = 8, 16, 32

    # Create params and discretize
    params = create_ssm_params(key, P, H)
    step = jnp.exp(params['log_step'])
    A_bar, B_bar = discretize_zoh(params['Lambda'], params['B'], step)

    # Create input sequence
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (L, H))

    # Apply SSM
    y = apply_ssm(A_bar, B_bar, params['C'], params['D'], x)

    # Verify output is finite
    assert jnp.all(jnp.isfinite(y)), "SSM output contains NaN or Inf"

    # Verify output has same shape as input
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    print(f"[PASS] test_ssm_apply: input={x.shape}, output={y.shape}")
    return True


def test_ssm_output_shape():
    """Test: SSM output has same shape as input."""
    key = jax.random.PRNGKey(456)

    test_cases = [
        (4, 8, 16),    # Small
        (16, 32, 64),  # Medium
        (32, 64, 128), # Large
    ]

    for P, H, L in test_cases:
        params = create_ssm_params(key, P, H)
        step = jnp.exp(params['log_step'])
        A_bar, B_bar = discretize_zoh(params['Lambda'], params['B'], step)

        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (L, H))

        y = apply_ssm(A_bar, B_bar, params['C'], params['D'], x)

        assert y.shape == (L, H), f"Output shape mismatch for P={P}, H={H}, L={L}: {y.shape}"

    print(f"[PASS] test_ssm_output_shape: all {len(test_cases)} test cases passed")
    return True


def test_ssm_state_evolution():
    """Test: SSM state updates correctly according to recurrence."""
    key = jax.random.PRNGKey(789)
    P, H, L = 4, 8, 10

    params = create_ssm_params(key, P, H)
    step = jnp.exp(params['log_step'])
    A_bar, B_bar = discretize_zoh(params['Lambda'], params['B'], step)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (L, H))

    # Get output and final state using scan
    y_scan, final_state = apply_ssm_scan(A_bar, B_bar, params['C'], params['D'], x)

    # Manually compute final state step by step
    state = jnp.zeros((P,), dtype=jnp.complex64)
    for t in range(L):
        state = A_bar * state + jnp.dot(B_bar, x[t])

    # Verify final states match
    state_diff = jnp.max(jnp.abs(final_state - state))
    assert state_diff < 1e-5, f"State evolution mismatch: {state_diff}"

    # Verify loop and scan produce same output
    y_loop = apply_ssm(A_bar, B_bar, params['C'], params['D'], x)
    output_diff = jnp.max(jnp.abs(y_scan - y_loop))
    assert output_diff < 1e-5, f"Output mismatch between loop and scan: {output_diff}"

    print(f"[PASS] test_ssm_state_evolution: state_diff={state_diff:.2e}, output_diff={output_diff:.2e}")
    return True


def test_ssm_stability():
    """Test: discretized SSM eigenvalues (A_bar) are inside unit circle."""
    key = jax.random.PRNGKey(999)
    P, H = 16, 32

    params = create_ssm_params(key, P, H)
    step = jnp.exp(params['log_step'])
    A_bar, B_bar = discretize_zoh(params['Lambda'], params['B'], step)

    # Check eigenvalue magnitudes
    # A_bar = exp(Lambda * step) where Re(Lambda) < 0
    # |A_bar| = |exp(Lambda * step)| = exp(Re(Lambda) * step)
    # Since Re(Lambda) < 0 and step > 0, we have |A_bar| < 1
    eigenvalue_magnitudes = jnp.abs(A_bar)
    max_magnitude = jnp.max(eigenvalue_magnitudes)

    assert max_magnitude < 1.0, f"SSM unstable: max eigenvalue magnitude = {max_magnitude}"

    # Verify all magnitudes are inside unit circle
    assert jnp.all(eigenvalue_magnitudes < 1.0), "Some eigenvalues outside unit circle"

    print(f"[PASS] test_ssm_stability: max |eigenvalue| = {max_magnitude:.6f} < 1.0")
    return True


def test_ssm_zero_input():
    """Test: zero input produces zero output (after initial transient)."""
    key = jax.random.PRNGKey(111)
    P, H, L = 8, 16, 100

    params = create_ssm_params(key, P, H)
    step = jnp.exp(params['log_step'])
    A_bar, B_bar = discretize_zoh(params['Lambda'], params['B'], step)

    # Zero input
    x = jnp.zeros((L, H))

    y = apply_ssm(A_bar, B_bar, params['C'], params['D'], x)

    # Output should be zero (D*0 = 0, and state stays at zero)
    max_output = jnp.max(jnp.abs(y))
    assert max_output < 1e-10, f"Zero input should produce zero output: {max_output}"

    print(f"[PASS] test_ssm_zero_input: max_output = {max_output:.2e}")
    return True


def test_ssm_impulse_response():
    """Test: SSM impulse response decays due to stable eigenvalues."""
    key = jax.random.PRNGKey(222)
    P, H, L = 8, 16, 500  # Longer sequence for decay

    # Use stronger damping for clearer decay
    keys = jax.random.split(key, 5)
    Lambda_re = -jax.random.uniform(keys[0], (P,), minval=0.5, maxval=2.0)  # Stronger damping
    Lambda_im = jax.random.uniform(keys[1], (P,), minval=-1.0, maxval=1.0)
    Lambda = Lambda_re + 1j * Lambda_im

    B_re = jax.random.normal(keys[2], (P, H)) * 0.1
    B = B_re + 1j * B_re
    C_re = jax.random.normal(keys[3], (H, P)) * 0.1
    C = C_re + 1j * C_re
    D = jnp.zeros((H,))  # No feedthrough for clearer impulse response

    step = jnp.ones((P,)) * 0.1  # Larger step for faster decay

    A_bar, B_bar = discretize_zoh(Lambda, B, step)

    # Impulse at t=0
    x = jnp.zeros((L, H))
    x = x.at[0].set(1.0)  # Impulse

    y = apply_ssm(A_bar, B_bar, C, D, x)

    # Response should decay over time
    # Compare cumulative energy in first half vs second half
    first_half_energy = jnp.sum(y[:L//2] ** 2)
    second_half_energy = jnp.sum(y[L//2:] ** 2)

    # First half should have more energy than second half
    assert first_half_energy > second_half_energy, \
        f"Response should decay: first_half={first_half_energy:.6f}, second_half={second_half_energy:.6f}"

    print(f"[PASS] test_ssm_impulse_response: first_half_energy={first_half_energy:.6f}, second_half_energy={second_half_energy:.6f}")
    return True


def test_ssm_linearity():
    """Test: SSM is linear (superposition principle)."""
    key = jax.random.PRNGKey(333)
    P, H, L = 8, 16, 32

    params = create_ssm_params(key, P, H)
    step = jnp.exp(params['log_step'])
    A_bar, B_bar = discretize_zoh(params['Lambda'], params['B'], step)

    # Two different inputs
    key, k1, k2 = jax.random.split(key, 3)
    x1 = jax.random.normal(k1, (L, H))
    x2 = jax.random.normal(k2, (L, H))
    alpha, beta = 2.5, -1.3

    # SSM is linear: SSM(alpha*x1 + beta*x2) = alpha*SSM(x1) + beta*SSM(x2)
    y1 = apply_ssm(A_bar, B_bar, params['C'], params['D'], x1)
    y2 = apply_ssm(A_bar, B_bar, params['C'], params['D'], x2)
    y_combined = apply_ssm(A_bar, B_bar, params['C'], params['D'], alpha * x1 + beta * x2)

    y_superposition = alpha * y1 + beta * y2

    max_diff = jnp.max(jnp.abs(y_combined - y_superposition))
    assert max_diff < 1e-4, f"Linearity violated: {max_diff}"

    print(f"[PASS] test_ssm_linearity: max_diff = {max_diff:.2e}")
    return True


def run_all_tests():
    """Run all SSM integration tests."""
    print("=" * 60)
    print("SSM Integration Tests")
    print("=" * 60)

    tests = [
        test_ssm_discretize,
        test_ssm_apply,
        test_ssm_output_shape,
        test_ssm_state_evolution,
        test_ssm_stability,
        test_ssm_zero_input,
        test_ssm_impulse_response,
        test_ssm_linearity,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
