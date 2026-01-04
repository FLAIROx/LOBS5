"""Test ES tree key generation."""

import jax
import jax.numpy as jnp
import pytest

from es_lobs5.models.common import (
    simple_es_tree_key,
    merge_inits,
    MM,
    ES_Parameter,
)


class TestSimpleEsTreeKey:
    """Tests for simple_es_tree_key function."""

    def test_simple_es_tree_key(self):
        """Test that simple_es_tree_key generates correct nested key structure."""
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)

        # Create a nested parameter structure
        merged = merge_inits(
            weight=MM.rand_init(key1, 16, 32, jnp.float32),
            bias=ES_Parameter.rand_init(
                key2, raw_value=jnp.zeros(32), dtype=jnp.float32
            ),
            scale=ES_Parameter.rand_init(
                key3, raw_value=jnp.ones(32), dtype=jnp.float32
            ),
        )

        # Generate ES tree keys
        base_key = jax.random.PRNGKey(123)
        es_tree_key = simple_es_tree_key(merged.params, base_key, merged.scan_map)

        # Verify structure matches params structure
        assert 'weight' in es_tree_key
        assert 'bias' in es_tree_key
        assert 'scale' in es_tree_key

        # Verify each key is a valid PRNG key (shape (2,) for old-style keys)
        # or shape () for new-style keys with dtype key<...>
        weight_key = es_tree_key['weight']
        bias_key = es_tree_key['bias']
        scale_key = es_tree_key['scale']

        # Keys should be JAX PRNG keys (arrays with shape (2,) or typed keys)
        assert hasattr(weight_key, 'shape')
        assert hasattr(bias_key, 'shape')
        assert hasattr(scale_key, 'shape')

    def test_es_tree_key_unique(self):
        """Test that each parameter path gets a unique key."""
        key = jax.random.PRNGKey(42)
        key1, key2, key3, key4 = jax.random.split(key, 4)

        # Create a nested parameter structure with multiple leaves
        merged = merge_inits(
            layer1=merge_inits(
                weight=MM.rand_init(key1, 8, 16, jnp.float32),
                bias=ES_Parameter.rand_init(
                    key2, raw_value=jnp.zeros(16), dtype=jnp.float32
                ),
            ),
            layer2=merge_inits(
                weight=MM.rand_init(key3, 16, 32, jnp.float32),
                bias=ES_Parameter.rand_init(
                    key4, raw_value=jnp.zeros(32), dtype=jnp.float32
                ),
            ),
        )

        # Generate ES tree keys
        base_key = jax.random.PRNGKey(456)
        es_tree_key = simple_es_tree_key(merged.params, base_key, merged.scan_map)

        # Flatten both trees to get all leaf keys
        flat_keys, _ = jax.tree.flatten(es_tree_key)

        # Verify we have the expected number of leaf keys (4 parameters)
        assert len(flat_keys) == 4

        # Verify all keys are unique by comparing pairs
        for i in range(len(flat_keys)):
            for j in range(i + 1, len(flat_keys)):
                # Keys should not be equal
                key_i = jnp.asarray(flat_keys[i])
                key_j = jnp.asarray(flat_keys[j])
                assert not jnp.array_equal(key_i, key_j), \
                    f"Keys at indices {i} and {j} are identical"

    def test_es_tree_key_deterministic(self):
        """Test that same base_key produces same tree keys."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key, 2)

        merged = merge_inits(
            weight=MM.rand_init(key1, 16, 32, jnp.float32),
            bias=ES_Parameter.rand_init(
                key2, raw_value=jnp.zeros(32), dtype=jnp.float32
            ),
        )

        base_key = jax.random.PRNGKey(789)

        # Generate keys twice with same base_key
        es_tree_key1 = simple_es_tree_key(merged.params, base_key, merged.scan_map)
        es_tree_key2 = simple_es_tree_key(merged.params, base_key, merged.scan_map)

        # Flatten and compare
        flat_keys1, _ = jax.tree.flatten(es_tree_key1)
        flat_keys2, _ = jax.tree.flatten(es_tree_key2)

        for k1, k2 in zip(flat_keys1, flat_keys2):
            assert jnp.array_equal(jnp.asarray(k1), jnp.asarray(k2))

    def test_es_tree_key_different_base(self):
        """Test that different base_key produces different tree keys."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key, 2)

        merged = merge_inits(
            weight=MM.rand_init(key1, 16, 32, jnp.float32),
            bias=ES_Parameter.rand_init(
                key2, raw_value=jnp.zeros(32), dtype=jnp.float32
            ),
        )

        # Generate keys with different base keys
        base_key1 = jax.random.PRNGKey(100)
        base_key2 = jax.random.PRNGKey(200)

        es_tree_key1 = simple_es_tree_key(merged.params, base_key1, merged.scan_map)
        es_tree_key2 = simple_es_tree_key(merged.params, base_key2, merged.scan_map)

        # Flatten and compare - at least one key should differ
        flat_keys1, _ = jax.tree.flatten(es_tree_key1)
        flat_keys2, _ = jax.tree.flatten(es_tree_key2)

        any_different = False
        for k1, k2 in zip(flat_keys1, flat_keys2):
            if not jnp.array_equal(jnp.asarray(k1), jnp.asarray(k2)):
                any_different = True
                break

        assert any_different, "Different base keys should produce different tree keys"
