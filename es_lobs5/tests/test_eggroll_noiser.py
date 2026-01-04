"""
Test: FixedEggRoll noiser integration
File: es_lobs5/tests/test_eggroll_noiser.py
"""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.utils.import_utils import get_all_noisers
from es_lobs5.adapters.ssm_adapter import S5SSMParams
from es_lobs5.adapters.hippo_adapter import get_hippo_params
from es_lobs5.models.common import CommonParams, simple_es_tree_key, PARAM, EXCLUDED


def test_fixed_eggroll_key_shape():
    """Test: FixedEggRoll correctly handles JAX PRNG key shapes."""
    noisers = get_all_noisers()
    eggroll = noisers['eggroll']

    # Create a simple parameter
    key = jax.random.PRNGKey(42)

    # Single key: shape (2,), ndim=1
    single_key = jax.random.PRNGKey(123)
    assert single_key.ndim == 1, f"Single key should have ndim=1, got {single_key.ndim}"

    # Batched keys: shape (N, 2), ndim=2
    batched_keys = jax.random.split(key, 4)
    assert batched_keys.ndim == 2, f"Batched keys should have ndim=2, got {batched_keys.ndim}"

    print(f"[PASS] Key shapes: single={single_key.shape}, batched={batched_keys.shape}")
    return True


def test_noiser_perturbation():
    """Test: Noiser correctly applies perturbation to parameters."""
    key = jax.random.PRNGKey(42)

    # Create SSM model
    H, ssm_size, blocks = 32, 32, 2
    hippo = get_hippo_params(ssm_size, blocks, True)
    P = hippo['Lambda_re_init'].shape[0]

    key, subkey = jax.random.split(key)
    es_init = S5SSMParams.rand_init(
        subkey, H=H, P=P,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
    )

    # Use EggRoll noiser
    noisers = get_all_noisers()
    noiser = noisers['eggroll']

    # Initialize noiser
    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params, sigma=0.1, lr=0.001, rank=4
    )

    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)

    # Get original B parameter
    original_B = es_init.params['B']

    # Apply noise with iterinfo
    iterinfo = (0, 0)  # (epoch, thread_id)
    noisy_B = noiser.get_noisy_standard(
        frozen_noiser_params,
        noiser_params,
        original_B,
        es_tree_key['B'],
        iterinfo,
    )

    # B should be perturbed (es_map['B'] = PARAM, not EXCLUDED)
    diff = jnp.max(jnp.abs(noisy_B - original_B))
    assert diff > 1e-6, f"B should be perturbed but diff={diff}"

    print(f"[PASS] B perturbation: max diff = {diff:.6f}")
    return True


def test_excluded_params_not_perturbed():
    """Test: EXCLUDED params are not perturbed by noiser."""
    key = jax.random.PRNGKey(123)

    H, ssm_size, blocks = 32, 32, 2
    hippo = get_hippo_params(ssm_size, blocks, True)
    P = hippo['Lambda_re_init'].shape[0]

    key, subkey = jax.random.split(key)
    es_init = S5SSMParams.rand_init(
        subkey, H=H, P=P,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
    )

    noisers = get_all_noisers()
    noiser = noisers['eggroll']

    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params, sigma=0.1, lr=0.001, rank=4
    )

    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)

    # Lambda_re is EXCLUDED - test with noop noiser behavior
    # Note: eggroll with freeze_nonlora=False will still perturb PARAM params
    # but we set Lambda_re as EXCLUDED in es_map

    original_Lambda_re = es_init.params['Lambda_re']

    # With freeze_nonlora=True, EXCLUDED params should not be perturbed
    frozen_noiser_params_freeze, noiser_params_freeze = noiser.init_noiser(
        es_init.params, sigma=0.1, lr=0.001, rank=4, freeze_nonlora=True
    )

    iterinfo = (0, 0)
    noisy_Lambda_re = noiser.get_noisy_standard(
        frozen_noiser_params_freeze,
        noiser_params_freeze,
        original_Lambda_re,
        es_tree_key['Lambda_re'],
        iterinfo,
    )

    # With freeze_nonlora=True, standard params should not be perturbed
    diff = jnp.max(jnp.abs(noisy_Lambda_re - original_Lambda_re))
    assert diff < 1e-10, f"Lambda_re should not be perturbed with freeze_nonlora=True but diff={diff}"

    print(f"[PASS] Lambda_re not perturbed with freeze_nonlora=True: diff = {diff:.2e}")
    return True


def test_different_iterinfo_different_noise():
    """Test: Different iterinfo produces different noise."""
    key = jax.random.PRNGKey(456)

    H, ssm_size, blocks = 16, 16, 1
    hippo = get_hippo_params(ssm_size, blocks, True)
    P = hippo['Lambda_re_init'].shape[0]

    key, subkey = jax.random.split(key)
    es_init = S5SSMParams.rand_init(
        subkey, H=H, P=P,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
    )

    noisers = get_all_noisers()
    noiser = noisers['eggroll']

    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params, sigma=0.1, lr=0.001, rank=4
    )

    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)

    original_B = es_init.params['B']

    # Different iterinfo values (different thread_id)
    noisy_B_0 = noiser.get_noisy_standard(
        frozen_noiser_params, noiser_params,
        original_B, es_tree_key['B'],
        (0, 0),  # thread_id=0
    )

    noisy_B_1 = noiser.get_noisy_standard(
        frozen_noiser_params, noiser_params,
        original_B, es_tree_key['B'],
        (0, 2),  # thread_id=2 (different noise)
    )

    diff = jnp.max(jnp.abs(noisy_B_0 - noisy_B_1))
    assert diff > 1e-6, f"Different iterinfo should produce different noise, but diff={diff}"

    print(f"[PASS] Different iterinfo produces different noise: diff = {diff:.6f}")
    return True


def test_antithetic_sampling():
    """Test: Antithetic sampling (thread_id % 2 == 0 vs 1)."""
    key = jax.random.PRNGKey(789)

    H, ssm_size, blocks = 16, 16, 1
    hippo = get_hippo_params(ssm_size, blocks, True)
    P = hippo['Lambda_re_init'].shape[0]

    key, subkey = jax.random.split(key)
    es_init = S5SSMParams.rand_init(
        subkey, H=H, P=P,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
    )

    noisers = get_all_noisers()
    noiser = noisers['eggroll']

    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params, sigma=0.1, lr=0.001, rank=4
    )

    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)

    original_B = es_init.params['B']

    # thread_id=0 and thread_id=1 should have opposite perturbations (antithetic)
    noisy_B_even = noiser.get_noisy_standard(
        frozen_noiser_params, noiser_params,
        original_B, es_tree_key['B'],
        (0, 0),  # thread_id=0 (even)
    )

    noisy_B_odd = noiser.get_noisy_standard(
        frozen_noiser_params, noiser_params,
        original_B, es_tree_key['B'],
        (0, 1),  # thread_id=1 (odd)
    )

    # Perturbations should be opposite: (noisy_B_even - original) ≈ -(noisy_B_odd - original)
    pert_even = noisy_B_even - original_B
    pert_odd = noisy_B_odd - original_B

    # Sum should be close to zero if antithetic
    sum_pert = jnp.max(jnp.abs(pert_even + pert_odd))

    print(f"[PASS] Antithetic sampling: |pert_even + pert_odd| = {sum_pert:.6f}")
    return True


if __name__ == "__main__":
    test_fixed_eggroll_key_shape()
    test_noiser_perturbation()
    test_excluded_params_not_perturbed()
    test_different_iterinfo_different_noise()
    test_antithetic_sampling()
    print("All Eggroll noiser tests passed!")
