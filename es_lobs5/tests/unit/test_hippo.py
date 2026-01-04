"""Test HiPPO initialization."""
import jax.numpy as jnp
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.adapters.hippo_adapter import get_hippo_params


def test_hippo_eigenvalues_stable():
    """Test: Lambda.real < 0 (all eigenvalues in left half-plane for stability)."""
    # Test with various configurations
    configs = [
        (64, 4, True),   # ssm_size=64, blocks=4, conj_sym=True
        (128, 8, True),  # ssm_size=128, blocks=8, conj_sym=True
        (256, 8, True),  # ssm_size=256, blocks=8, conj_sym=True
        (64, 4, False),  # ssm_size=64, blocks=4, conj_sym=False
    ]

    for ssm_size, blocks, conj_sym in configs:
        params = get_hippo_params(ssm_size, blocks, conj_sym)
        Lambda_re = params['Lambda_re_init']

        # All eigenvalues should have negative real part (stable system)
        max_real = jnp.max(Lambda_re)
        assert max_real < 0, (
            f"Eigenvalues not stable for config ({ssm_size}, {blocks}, {conj_sym}): "
            f"max(Lambda.real) = {max_real} >= 0"
        )

        print(f"[PASS] eigenvalues stable: ssm_size={ssm_size}, blocks={blocks}, "
              f"conj_sym={conj_sym}, max(Lambda.real)={max_real:.6f}")

    return True


def test_conj_sym_reduction():
    """Test: conj_sym=True results in P_eff = ssm_size // 2."""
    ssm_size = 128
    blocks = 8

    # With conj_sym=True
    params_conj = get_hippo_params(ssm_size, blocks, conj_sym=True)
    P_eff_conj = params_conj['Lambda_re_init'].shape[0]
    expected_P_conj = ssm_size // 2

    assert P_eff_conj == expected_P_conj, (
        f"conj_sym=True: expected P_eff={expected_P_conj}, got {P_eff_conj}"
    )

    # With conj_sym=False
    params_full = get_hippo_params(ssm_size, blocks, conj_sym=False)
    P_eff_full = params_full['Lambda_re_init'].shape[0]
    expected_P_full = ssm_size

    assert P_eff_full == expected_P_full, (
        f"conj_sym=False: expected P_eff={expected_P_full}, got {P_eff_full}"
    )

    # Verify V matrix shapes
    assert params_conj['V'].shape == (ssm_size, ssm_size // 2), (
        f"conj_sym=True: V shape mismatch, got {params_conj['V'].shape}"
    )
    assert params_full['V'].shape == (ssm_size, ssm_size), (
        f"conj_sym=False: V shape mismatch, got {params_full['V'].shape}"
    )

    print(f"[PASS] conj_sym=True: P_eff={P_eff_conj} (expected {expected_P_conj})")
    print(f"[PASS] conj_sym=False: P_eff={P_eff_full} (expected {expected_P_full})")

    return True


if __name__ == "__main__":
    test_hippo_eigenvalues_stable()
    test_conj_sym_reduction()
    print("All HiPPO tests passed!")
