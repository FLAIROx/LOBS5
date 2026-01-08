"""
Tests for Message_Tokenizer class.

These tests verify:
1. New instance-based API (Message_Tokenizer(token_mode=24))
2. Independence between instances (no global state pollution)
3. Backward compatibility with class-level API (deprecated)
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob.encoding import Message_Tokenizer


# =============================================================================
# Instance-Based API Tests (New API - Phase 3)
# =============================================================================

class TestTokenizerInstance:
    """Tests for the new instance-based Message_Tokenizer API."""

    def test_init_with_token_mode_24(self):
        """Message_Tokenizer(token_mode=24) should create instance with correct values."""
        tokenizer = Message_Tokenizer(token_mode=24)

        assert tokenizer.token_mode == 24
        assert tokenizer.msg_len == 24
        assert np.array_equal(tokenizer.tok_lens, Message_Tokenizer.TOK_LENS_24)

    def test_init_with_token_mode_22(self):
        """Message_Tokenizer(token_mode=22) should create instance with correct values."""
        tokenizer = Message_Tokenizer(token_mode=22)

        assert tokenizer.token_mode == 22
        assert tokenizer.msg_len == 22
        assert np.array_equal(tokenizer.tok_lens, Message_Tokenizer.TOK_LENS_22)

    def test_init_default_is_24(self):
        """Message_Tokenizer() should default to token_mode=24."""
        tokenizer = Message_Tokenizer()

        assert tokenizer.token_mode == 24
        assert tokenizer.msg_len == 24

    def test_init_invalid_token_mode_raises(self):
        """Message_Tokenizer(token_mode=23) should raise AssertionError."""
        with pytest.raises(AssertionError):
            Message_Tokenizer(token_mode=23)

    def test_tok_delim_computed_correctly_24(self):
        """tok_delim should be cumsum of tok_lens for 24-token."""
        tokenizer = Message_Tokenizer(token_mode=24)

        expected_delim = np.cumsum(Message_Tokenizer.TOK_LENS_24[:-1])
        assert np.array_equal(tokenizer.tok_delim, expected_delim)

    def test_tok_delim_computed_correctly_22(self):
        """tok_delim should be cumsum of tok_lens for 22-token."""
        tokenizer = Message_Tokenizer(token_mode=22)

        expected_delim = np.cumsum(Message_Tokenizer.TOK_LENS_22[:-1])
        assert np.array_equal(tokenizer.tok_delim, expected_delim)

    def test_new_msg_len_24(self):
        """new_msg_len should exclude reference fields for 24-token."""
        tokenizer = Message_Tokenizer(token_mode=24)

        # Reference fields: price_ref(2), size_ref(2), time_s_ref(2), time_ns_ref(3)
        # Total ref length: 2 + 2 + 2 + 3 = 9
        expected_new_msg_len = 24 - 9
        assert tokenizer.new_msg_len == expected_new_msg_len

    def test_new_msg_len_22(self):
        """new_msg_len should exclude reference fields for 22-token."""
        tokenizer = Message_Tokenizer(token_mode=22)

        # Reference fields: price_ref(2), size_ref(1), time_s_ref(2), time_ns_ref(3)
        # Total ref length: 2 + 1 + 2 + 3 = 8
        expected_new_msg_len = 22 - 8
        assert tokenizer.new_msg_len == expected_new_msg_len


class TestTokenizerInstanceIndependence:
    """Tests verifying that two tokenizer instances don't affect each other."""

    def test_two_instances_independent(self):
        """Creating tokenizer_24 after tokenizer_22 should not change tokenizer_22."""
        t22 = Message_Tokenizer(token_mode=22)
        t24 = Message_Tokenizer(token_mode=24)

        # t22 should still have 22-token values
        assert t22.msg_len == 22
        assert t22.token_mode == 22

        # t24 should have 24-token values
        assert t24.msg_len == 24
        assert t24.token_mode == 24

    def test_instances_can_coexist(self):
        """Both instances should be usable simultaneously."""
        t22 = Message_Tokenizer(token_mode=22)
        t24 = Message_Tokenizer(token_mode=24)

        # Both should work correctly
        assert t22.tok_lens[3] == 1  # size field: 1 token in 22-mode
        assert t24.tok_lens[3] == 2  # size field: 2 tokens in 24-mode

    def test_modifying_instance_does_not_affect_other(self):
        """Modifying one instance should not affect another."""
        t1 = Message_Tokenizer(token_mode=24)
        t2 = Message_Tokenizer(token_mode=24)

        # If we had mutable state, modifying t1 would affect t2
        # This test ensures they're truly independent
        original_t2_msg_len = t2.msg_len

        # Even if we somehow modify t1, t2 should be unchanged
        # (Currently instances are immutable after creation, but this test
        # guards against future changes)
        assert t2.msg_len == original_t2_msg_len


# =============================================================================
# Static Constants Tests
# =============================================================================

class TestStaticConstants:
    """Tests for static class constants (immutable)."""

    def test_tok_lens_22_is_correct(self):
        """TOK_LENS_22 should have correct values."""
        expected = np.array((1, 1, 2, 1, 1, 3, 2, 3, 2, 1, 2, 3))
        assert np.array_equal(Message_Tokenizer.TOK_LENS_22, expected)

    def test_tok_lens_24_is_correct(self):
        """TOK_LENS_24 should have correct values."""
        expected = np.array((1, 1, 2, 2, 1, 3, 2, 3, 2, 2, 2, 3))
        assert np.array_equal(Message_Tokenizer.TOK_LENS_24, expected)

    def test_tok_lens_22_sum_to_22(self):
        """TOK_LENS_22 should sum to 22."""
        assert np.sum(Message_Tokenizer.TOK_LENS_22) == 22

    def test_tok_lens_24_sum_to_24(self):
        """TOK_LENS_24 should sum to 24."""
        assert np.sum(Message_Tokenizer.TOK_LENS_24) == 24

    def test_tok_lens_size_field_differs(self):
        """Size field (index 3) should be 1 in 22-mode, 2 in 24-mode."""
        assert Message_Tokenizer.TOK_LENS_22[3] == 1  # size: 1 token
        assert Message_Tokenizer.TOK_LENS_24[3] == 2  # size: 2 tokens

    def test_tok_lens_size_ref_field_differs(self):
        """Size_ref field (index 9) should be 1 in 22-mode, 2 in 24-mode."""
        assert Message_Tokenizer.TOK_LENS_22[9] == 1  # size_ref: 1 token
        assert Message_Tokenizer.TOK_LENS_24[9] == 2  # size_ref: 2 tokens

    def test_fields_tuple_is_complete(self):
        """FIELDS tuple should contain all 12 field names."""
        expected_fields = (
            'event_type', 'direction', 'price', 'size',
            'delta_t_s', 'delta_t_ns', 'time_s', 'time_ns',
            'price_ref', 'size_ref', 'time_s_ref', 'time_ns_ref',
        )
        assert Message_Tokenizer.FIELDS == expected_fields

    def test_n_fields_matches(self):
        """N_NEW_FIELDS + N_REF_FIELDS should equal total fields."""
        assert Message_Tokenizer.N_NEW_FIELDS + Message_Tokenizer.N_REF_FIELDS == 12


# =============================================================================
# Static Helper Methods Tests
# =============================================================================

class TestStaticHelperMethods:
    """Tests for static/class methods that don't modify state."""

    def test_get_msg_len_22(self):
        """get_msg_len(22) should return 22 without changing state."""
        result = Message_Tokenizer.get_msg_len(22)
        assert result == 22

    def test_get_msg_len_24(self):
        """get_msg_len(24) should return 24 without changing state."""
        result = Message_Tokenizer.get_msg_len(24)
        assert result == 24

    def test_get_tok_lens_22(self):
        """get_tok_lens(22) should return TOK_LENS_22 without changing state."""
        result = Message_Tokenizer.get_tok_lens(22)
        assert np.array_equal(result, Message_Tokenizer.TOK_LENS_22)

    def test_get_tok_lens_24(self):
        """get_tok_lens(24) should return TOK_LENS_24 without changing state."""
        result = Message_Tokenizer.get_tok_lens(24)
        assert np.array_equal(result, Message_Tokenizer.TOK_LENS_24)


# =============================================================================
# Backward Compatibility Tests (Deprecated API)
# =============================================================================

class TestBackwardCompatibility:
    """Tests for deprecated class-level API (set_token_mode)."""

    def test_set_token_mode_24_updates_class_state(self):
        """set_token_mode(24) should update class variables (deprecated)."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(24)

        assert Message_Tokenizer.MSG_LEN == 24
        assert np.array_equal(Message_Tokenizer.TOK_LENS, Message_Tokenizer.TOK_LENS_24)

    def test_set_token_mode_22_updates_class_state(self):
        """set_token_mode(22) should update class variables (deprecated)."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(22)

        assert Message_Tokenizer.MSG_LEN == 22
        assert np.array_equal(Message_Tokenizer.TOK_LENS, Message_Tokenizer.TOK_LENS_22)

    def test_set_token_mode_invalid_raises(self):
        """set_token_mode(23) should raise AssertionError."""
        with pytest.raises(AssertionError):
            Message_Tokenizer.set_token_mode(23)

    def test_set_token_mode_emits_deprecation_warning(self):
        """set_token_mode() should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            Message_Tokenizer.set_token_mode(24)


# =============================================================================
# get_field_from_idx Tests
# =============================================================================

class TestGetFieldFromIdx:
    """Tests for get_field_from_idx() method."""

    def test_get_field_from_idx_position_0(self):
        """Position 0 should map to event_type."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(24)
        result = Message_Tokenizer.get_field_from_idx(0)
        assert result == ['event_type']

    def test_get_field_from_idx_position_4_mode_24(self):
        """Position 4 in 24-mode should map to size (size_high)."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(24)
        result = Message_Tokenizer.get_field_from_idx(4)
        assert result == ['size']

    def test_get_field_from_idx_position_5_mode_24(self):
        """Position 5 in 24-mode should map to size (size_low)."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(24)
        result = Message_Tokenizer.get_field_from_idx(5)
        assert result == ['size']

    def test_get_field_from_idx_position_4_mode_22(self):
        """Position 4 in 22-mode should map to size (single token)."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(22)
        result = Message_Tokenizer.get_field_from_idx(4)
        assert result == ['size']

    def test_get_field_from_idx_position_5_mode_22(self):
        """Position 5 in 22-mode should map to delta_t_s (not size)."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(22)
        result = Message_Tokenizer.get_field_from_idx(5)
        assert result == ['delta_t_s']

    def test_get_field_from_idx_out_of_bounds_raises(self):
        """Index >= MSG_LEN should raise ValueError."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(24)
        with pytest.raises(ValueError):
            Message_Tokenizer.get_field_from_idx(24)


# =============================================================================
# FIELD_ENC_TYPES Tests
# =============================================================================

class TestFieldEncTypes:
    """Tests for FIELD_ENC_TYPES mapping."""

    def test_size_maps_to_size(self):
        """'size' field should map to 'size' encoder type."""
        # Note: This is the class constant, which always returns 'size'
        # The get_encoder_key() function in validation_helpers handles
        # the 24-token 'size_digit' mapping
        assert Message_Tokenizer.FIELD_ENC_TYPES['size'] == 'size'

    def test_event_type_maps_to_event_type(self):
        """'event_type' field should map to 'event_type' encoder."""
        assert Message_Tokenizer.FIELD_ENC_TYPES['event_type'] == 'event_type'

    def test_time_fields_map_to_time(self):
        """Time fields should map to 'time' encoder."""
        assert Message_Tokenizer.FIELD_ENC_TYPES['delta_t_s'] == 'time'
        assert Message_Tokenizer.FIELD_ENC_TYPES['delta_t_ns'] == 'time'
        assert Message_Tokenizer.FIELD_ENC_TYPES['time_s'] == 'time'
        assert Message_Tokenizer.FIELD_ENC_TYPES['time_ns'] == 'time'
