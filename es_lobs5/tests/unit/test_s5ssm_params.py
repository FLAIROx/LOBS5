"""Test S5SSMParams ES wrapper."""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.adapters.ssm_adapter import S5SSMParams
from es_lobs5.adapters.hippo_adapter import get_hippo_params
from es_lobs5.models.common import CommonParams, simple_es_tree_key, PARAM, EXCLUDED
from es_lobs5.utils.import_utils import load_module, get_hyperscalees_path
import os


def get_noop_noiser():
    """Get noop noiser without loading all noisers (avoids optax dependency)."""
    hyperscalees_path = get_hyperscalees_path()
    base_noiser = load_module(
        'hyperscalees.noiser.base_noiser',
        os.path.join(hyperscalees_path, 'hyperscalees/noiser/base_noiser.py')
    )
    return base_noiser.Noiser


def test_s5ssm_es_map():
    """Test: Lambda, log_step -> EXCLUDED; B, C, D -> PARAM."""
    key = jax.random.PRNGKey(42)

    # Model config
    H = 32
    ssm_size = 32
    blocks = 2
    conj_sym = True

    # Initialize HiPPO matrices
    hippo = get_hippo_params(ssm_size, blocks, conj_sym)
    P = hippo['Lambda_re_init'].shape[0]

    # Initialize S5SSMParams
    es_init = S5SSMParams.rand_init(
        key, H=H, P=P,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
        conj_sym=conj_sym,
    )

    es_map = es_init.es_map

    # Verify EXCLUDED parameters (stability critical)
    assert es_map['Lambda_re'] == EXCLUDED, f"Lambda_re should be EXCLUDED, got {es_map['Lambda_re']}"
    assert es_map['Lambda_im'] == EXCLUDED, f"Lambda_im should be EXCLUDED, got {es_map['Lambda_im']}"
    assert es_map['log_step'] == EXCLUDED, f"log_step should be EXCLUDED, got {es_map['log_step']}"

    # Verify PARAM parameters (safe to perturb)
    assert es_map['B'] == PARAM, f"B should be PARAM, got {es_map['B']}"
    assert es_map['C'] == PARAM, f"C should be PARAM, got {es_map['C']}"
    assert es_map['D'] == PARAM, f"D should be PARAM, got {es_map['D']}"

    print("[PASS] test_s5ssm_es_map: Lambda, log_step -> EXCLUDED; B, C, D -> PARAM")
    return True


def test_s5ssm_forward_shape():
    """Test: input (L, H) -> output (L, H)."""
    key = jax.random.PRNGKey(123)

    # Model config
    H = 64
    ssm_size = 64
    blocks = 4
    conj_sym = True
    L = 100  # Sequence length

    # Initialize HiPPO matrices
    hippo = get_hippo_params(ssm_size, blocks, conj_sym)
    P = hippo['Lambda_re_init'].shape[0]

    # Initialize S5SSMParams
    key, subkey = jax.random.split(key)
    es_init = S5SSMParams.rand_init(
        subkey, H=H, P=P,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
        conj_sym=conj_sym,
        clip_eigs=True,
    )

    # Setup noiser (noop for testing)
    noiser = get_noop_noiser()
    es_tree_key = simple_es_tree_key(es_init.params, key, es_init.scan_map)

    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=es_init.frozen_params,
        params=es_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,
    )

    # Create input: (L, H)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (L, H))

    # Forward pass
    output = S5SSMParams._forward(common_params, x)

    # Verify output shape: (L, H)
    expected_shape = (L, H)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
    assert not jnp.any(jnp.isnan(output)), "NaN in output"
    assert not jnp.any(jnp.isinf(output)), "Inf in output"

    print(f"[PASS] test_s5ssm_forward_shape: input {x.shape} -> output {output.shape}")
    return True


if __name__ == "__main__":
    test_s5ssm_es_map()
    test_s5ssm_forward_shape()
    print("All S5SSMParams tests passed!")
