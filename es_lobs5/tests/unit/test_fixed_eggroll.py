"""Test JAX PRNG key ndim check logic for EggRoll fix."""
import jax


def test_jax_key_ndim():
    """Test: JAX PRNG key ndim check.

    JAX PRNG keys have shape (2,), ndim=1, not shape () like a scalar.
    The original EggRoll checks `len(base_key.shape) == 0` which is always False
    for JAX keys. The fix uses `base_key.ndim == 1` instead.
    """
    key = jax.random.PRNGKey(0)

    # Fixed check: ndim == 1 for single key
    assert key.ndim == 1, f"JAX key should have ndim=1, got {key.ndim}"

    # Original buggy check: len(shape) == 0 is always False for JAX keys
    assert len(key.shape) != 0, "len(key.shape) should not be 0 for JAX key"

    # Verify the original condition fails (bug)
    original_condition = len(key.shape) == 0
    assert original_condition == False, "Original condition is buggy for JAX keys"

    # Verify the fixed condition works
    fixed_condition = key.ndim == 1
    assert fixed_condition == True, "Fixed condition should work for JAX keys"


def test_batched_keys_ndim():
    """Test: Batched JAX keys have ndim=2."""
    key = jax.random.PRNGKey(0)
    batched_keys = jax.random.split(key, 4)

    # Batched keys: shape=(N, 2), ndim=2
    assert batched_keys.ndim == 2, f"Batched keys should have ndim=2, got {batched_keys.ndim}"
    assert batched_keys.shape == (4, 2), f"Batched keys should have shape (4, 2), got {batched_keys.shape}"

    # Both original and fixed conditions should be False for batched keys
    assert len(batched_keys.shape) != 0, "len(shape) != 0 for batched keys"
    assert batched_keys.ndim != 1, "ndim != 1 for batched keys (should take scan path)"


if __name__ == "__main__":
    test_jax_key_ndim()
    test_batched_keys_ndim()
    print("All JAX key ndim tests passed!")
