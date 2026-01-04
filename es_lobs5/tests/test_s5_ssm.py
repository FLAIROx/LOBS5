"""
Test: ES_S5SSM output consistency with Flax S5SSM
File: es_lobs5/tests/test_s5_ssm.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Import ES version
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.adapters.ssm_adapter import S5SSMParams
from es_lobs5.adapters.hippo_adapter import get_hippo_params
from es_lobs5.models.common import CommonParams, simple_es_tree_key
from es_lobs5.utils.import_utils import get_all_noisers

# Import Flax version
from s5.ssm import discretize_zoh, apply_ssm


def test_ssm_forward_consistency():
    """Test: ES SSM produces same output as Flax SSM (without noise)."""
    key = jax.random.PRNGKey(42)

    # Model config
    H = 64       # Feature dimension
    ssm_size = 64
    blocks = 4
    conj_sym = True

    # Initialize HiPPO matrices
    hippo = get_hippo_params(ssm_size, blocks, conj_sym)
    Lambda_re_init = hippo['Lambda_re_init']
    Lambda_im_init = hippo['Lambda_im_init']
    V = hippo['V']
    Vinv = hippo['Vinv']
    P = Lambda_re_init.shape[0]

    # Initialize ES model
    key, subkey = jax.random.split(key)
    es_init = S5SSMParams.rand_init(
        subkey, H=H, P=P,
        Lambda_re_init=Lambda_re_init,
        Lambda_im_init=Lambda_im_init,
        V=V, Vinv=Vinv,
        conj_sym=conj_sym,
        clip_eigs=True,
    )

    # Get noop noiser (no perturbation)
    noisers = get_all_noisers()
    noiser = noisers['noop']

    # Build CommonParams without noise (iterinfo=None)
    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)
    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=es_init.frozen_params,
        params=es_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,  # No noise applied
    )

    # Create random input
    key, subkey = jax.random.split(key)
    L = 100  # Sequence length
    x = jax.random.normal(subkey, (L, H))

    # Forward pass (ES version)
    es_output = S5SSMParams._forward(common_params, x)

    # Manual forward (same as Flax would do)
    Lambda = es_init.params['Lambda_re'] + 1j * es_init.params['Lambda_im']
    Lambda = jnp.clip(Lambda.real, None, -1e-4) + 1j * Lambda.imag
    B_tilde = es_init.params['B'][..., 0] + 1j * es_init.params['B'][..., 1]
    C_tilde = es_init.params['C'][..., 0] + 1j * es_init.params['C'][..., 1]
    step = jnp.exp(es_init.params['log_step'][:, 0])

    Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
    ys = apply_ssm(Lambda_bar, B_bar, C_tilde, x.astype(jnp.float32), conj_sym, False)
    Du = jax.vmap(lambda u: es_init.params['D'] * u)(x.astype(jnp.float32))
    manual_output = ys + Du

    # Assert consistency
    assert es_output.shape == manual_output.shape, f"Shape mismatch: {es_output.shape} vs {manual_output.shape}"

    max_diff = jnp.max(jnp.abs(es_output - manual_output))
    assert max_diff < 1e-5, f"Output difference too large: {max_diff}"

    print(f"[PASS] SSM forward consistency: max_diff = {max_diff}")
    return True


def test_ssm_with_noise():
    """Test: ES SSM with noiser produces different output for different iterinfo."""
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

    # Use eggroll noiser
    noisers = get_all_noisers()
    noiser = noisers['eggroll']

    # Initialize noiser params
    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_init.params, sigma=0.01, lr=0.001, rank=4
    )

    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)

    # Input
    x = jax.random.normal(key, (50, H))

    # Forward with two different iterinfos
    outputs = []
    for thread_id in [0, 2]:  # Different threads get different noise
        iterinfo = (0, thread_id)  # (epoch, thread_id)
        common_params = CommonParams(
            noiser=noiser,
            frozen_noiser_params=frozen_noiser_params,
            noiser_params=noiser_params,
            frozen_params=es_init.frozen_params,
            params=es_init.params,
            es_tree_key=es_tree_key,
            iterinfo=iterinfo,
        )
        outputs.append(S5SSMParams._forward(common_params, x))

    # Different noise should produce different outputs
    diff = jnp.max(jnp.abs(outputs[0] - outputs[1]))
    assert diff > 1e-6, f"Different iterinfo should produce different output, but diff={diff}"

    print(f"[PASS] SSM with noise: output difference = {diff}")
    return True


if __name__ == "__main__":
    test_ssm_forward_consistency()
    test_ssm_with_noise()
    print("All SSM tests passed!")
