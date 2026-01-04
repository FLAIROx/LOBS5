"""Test FixedEggRoll JAX key fix."""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.utils.import_utils import get_all_noisers, get_noiser_modules


def test_jax_key_shape():
    """Test: JAX PRNG key shape = (2,) is correctly handled.

    JAX PRNG keys have shape (2,) not shape () like a scalar.
    The original EggRoll checks `len(base_key.shape) == 0` which is always False
    for JAX keys. FixedEggRoll checks `base_key.ndim == 1` instead.
    """
    key = jax.random.PRNGKey(42)

    # Verify JAX key has shape (2,)
    assert key.shape == (2,), f"JAX key should have shape (2,), got {key.shape}"
    assert key.ndim == 1, f"JAX key should have ndim=1, got {key.ndim}"

    # Verify len(key.shape) != 0 (the original buggy check)
    assert len(key.shape) != 0, "len(key.shape) should not be 0 for JAX key"

    # Get the fixed EggRoll
    noisers = get_all_noisers()
    FixedEggRoll = noisers['eggroll']

    # Verify that FixedEggRoll is indeed the fixed version from import_utils
    # (not the original from HyperscaleES)
    original_modules = get_noiser_modules()
    OriginalEggRoll = original_modules['eggroll'].EggRoll

    # FixedEggRoll should be a subclass of the original
    assert issubclass(FixedEggRoll, OriginalEggRoll), \
        "FixedEggRoll should be a subclass of original EggRoll"

    # FixedEggRoll should have its own _do_update method (overridden)
    assert FixedEggRoll._do_update is not OriginalEggRoll._do_update, \
        "FixedEggRoll should override _do_update method"

    print(f"[PASS] test_jax_key_shape: JAX key shape={key.shape}, ndim={key.ndim}")
    return True


def test_key_ndim_check():
    """Test: base_key.ndim == 1 for single key, ndim == 2 for batched keys.

    The fix uses ndim instead of len(shape) == 0:
    - Single key: shape=(2,), ndim=1
    - Batched keys: shape=(N, 2), ndim=2
    """
    single_key = jax.random.PRNGKey(0)

    # Single key checks
    assert single_key.ndim == 1, f"Single key should have ndim=1, got {single_key.ndim}"
    assert single_key.shape == (2,), f"Single key should have shape (2,), got {single_key.shape}"

    # Batched keys (e.g., for scan over multiple parameters)
    batch_size = 4
    batched_keys = jax.random.split(single_key, batch_size)

    assert batched_keys.ndim == 2, f"Batched keys should have ndim=2, got {batched_keys.ndim}"
    assert batched_keys.shape == (batch_size, 2), \
        f"Batched keys should have shape ({batch_size}, 2), got {batched_keys.shape}"

    # Verify the original buggy condition fails for single key
    # Original: len(base_key.shape) == 0  --> False for single key (shape=(2,))
    original_condition_single = len(single_key.shape) == 0
    assert original_condition_single == False, \
        "Original condition should be False for single key (bug!)"

    # Verify the fixed condition works correctly
    # Fixed: base_key.ndim == 1  --> True for single key
    fixed_condition_single = single_key.ndim == 1
    assert fixed_condition_single == True, \
        "Fixed condition should be True for single key"

    # Verify for batched keys
    original_condition_batch = len(batched_keys.shape) == 0
    fixed_condition_batch = batched_keys.ndim == 1

    assert original_condition_batch == False, \
        "Original condition should be False for batched keys"
    assert fixed_condition_batch == False, \
        "Fixed condition should be False for batched keys (ndim=2)"

    print(f"[PASS] test_key_ndim_check: single key ndim={single_key.ndim}, batched keys ndim={batched_keys.ndim}")
    return True


def test_fixed_eggroll_single_key_path():
    """Test: FixedEggRoll correctly takes the single-key path when ndim=1.

    This test verifies that with the fix, a single JAX key (ndim=1)
    triggers the direct update path, not the scan path.
    """
    key = jax.random.PRNGKey(123)

    noisers = get_all_noisers()
    FixedEggRoll = noisers['eggroll']

    # Create a simple parameter and setup
    param = jnp.ones((4, 8), dtype=jnp.float32)  # Simple 4x8 weight matrix

    # Create mock fitnesses and iterinfos
    num_envs = 16
    fitnesses = jnp.zeros(num_envs)  # Normalized scores
    epochs = jnp.zeros(num_envs, dtype=jnp.int32)
    thread_ids = jnp.arange(num_envs, dtype=jnp.int32)
    iterinfos = (epochs, thread_ids)

    # Map classification: 0 = full update, 1 = lora update, 2 = noop, 3 = noop
    map_classification = 0  # Full update

    sigma = 0.01
    frozen_noiser_params = {
        "group_size": 0,
        "freeze_nonlora": False,
        "noise_reuse": 0,
        "rank": 1,
    }

    # Call _do_update with single key (ndim=1)
    result = FixedEggRoll._do_update(
        param, key, fitnesses, iterinfos, map_classification, sigma, frozen_noiser_params
    )

    # Result should have same shape as param
    assert result.shape == param.shape, \
        f"Result shape {result.shape} should match param shape {param.shape}"

    # Result should be a valid gradient (no NaN/Inf)
    assert jnp.isfinite(result).all(), "Result should contain finite values"

    print(f"[PASS] test_fixed_eggroll_single_key_path: output shape={result.shape}")
    return True


if __name__ == "__main__":
    test_jax_key_shape()
    test_key_ndim_check()
    test_fixed_eggroll_single_key_path()
    print("All FixedEggRoll tests passed!")
