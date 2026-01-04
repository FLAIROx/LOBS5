"""Test merge_inits and merge_frozen utilities."""

import jax
import jax.numpy as jnp
import pytest

from es_lobs5.models.common import (
    merge_inits,
    merge_frozen,
    MM,
    ES_Parameter,
    PARAM,
)


class TestMergeInits:
    """Tests for merge_inits function."""

    def test_merge_inits_structure(self):
        """Test that merged params contain all submodules."""
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)

        # Create individual inits
        weight_init = MM.rand_init(key1, 16, 32, jnp.float32)
        bias_init = ES_Parameter.rand_init(
            key2, raw_value=jnp.zeros(32), dtype=jnp.float32
        )
        scale_init = ES_Parameter.rand_init(
            key3, raw_value=jnp.ones(32), dtype=jnp.float32
        )

        # Merge them
        merged = merge_inits(
            weight=weight_init,
            bias=bias_init,
            scale=scale_init,
        )

        # Verify structure: params should contain all submodule keys
        assert 'weight' in merged.params
        assert 'bias' in merged.params
        assert 'scale' in merged.params

        # Verify scan_map has same keys
        assert 'weight' in merged.scan_map
        assert 'bias' in merged.scan_map
        assert 'scale' in merged.scan_map

        # Verify es_map has same keys
        assert 'weight' in merged.es_map
        assert 'bias' in merged.es_map
        assert 'scale' in merged.es_map

        # Verify param shapes are preserved
        assert merged.params['weight'].shape == (32, 16)
        assert merged.params['bias'].shape == (32,)
        assert merged.params['scale'].shape == (32,)


class TestMergeFrozen:
    """Tests for merge_frozen function."""

    def test_merge_frozen_adds_config(self):
        """Test that frozen_params contain additional config."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key, 2)

        # Create a merged init without frozen params
        merged = merge_inits(
            weight=MM.rand_init(key1, 16, 32, jnp.float32),
            bias=ES_Parameter.rand_init(
                key2, raw_value=jnp.zeros(32), dtype=jnp.float32
            ),
        )

        # Initially frozen_params should be None (no submodule has frozen_params)
        assert merged.frozen_params is None

        # Add frozen config
        with_frozen = merge_frozen(
            merged,
            activation='gelu',
            use_bias=True,
            hidden_dim=256,
        )

        # Verify frozen_params now contains the config
        assert with_frozen.frozen_params is not None
        assert with_frozen.frozen_params['activation'] == 'gelu'
        assert with_frozen.frozen_params['use_bias'] is True
        assert with_frozen.frozen_params['hidden_dim'] == 256

        # Verify other fields are unchanged
        assert 'weight' in with_frozen.params
        assert 'bias' in with_frozen.params
        assert with_frozen.params['weight'].shape == (32, 16)

    def test_merge_frozen_preserves_existing(self):
        """Test that merge_frozen preserves existing frozen_params."""
        key = jax.random.PRNGKey(42)

        merged = merge_inits(
            weight=MM.rand_init(key, 16, 32, jnp.float32),
        )

        # Add first config
        with_config1 = merge_frozen(merged, activation='relu')

        # Add second config
        with_config2 = merge_frozen(with_config1, dropout=0.1)

        # Both configs should be present
        assert with_config2.frozen_params['activation'] == 'relu'
        assert with_config2.frozen_params['dropout'] == 0.1
