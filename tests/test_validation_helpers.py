"""
Tests for validation_helpers module.

These tests verify:
1. syntax_validation_matrix produces correct shapes
2. Correct encoder mapping for size fields (size vs size_digit)
3. Special tokens are handled correctly
"""
import pytest
import warnings
import jax.numpy as jnp
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob.encoding import Vocab, Message_Tokenizer
from lob.validation_helpers import syntax_validation_matrix, get_encoder_key


# Helper to suppress deprecation warnings when using set_token_mode
@pytest.fixture
def suppress_deprecation():
    """Suppress DeprecationWarning for legacy API tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


# =============================================================================
# get_encoder_key Tests
# =============================================================================

class TestGetEncoderKey:
    """Tests for get_encoder_key function."""

    def test_size_returns_size_for_22(self):
        """'size' field should return 'size' for token_mode=22."""
        result = get_encoder_key('size', 22)
        assert result == 'size'

    def test_size_returns_size_digit_for_24(self):
        """'size' field should return 'size_digit' for token_mode=24."""
        result = get_encoder_key('size', 24)
        assert result == 'size_digit'

    def test_size_ref_returns_size_for_22(self):
        """'size_ref' field should return 'size' for token_mode=22."""
        result = get_encoder_key('size_ref', 22)
        assert result == 'size'

    def test_size_ref_returns_size_digit_for_24(self):
        """'size_ref' field should return 'size_digit' for token_mode=24."""
        result = get_encoder_key('size_ref', 24)
        assert result == 'size_digit'

    def test_event_type_same_for_both_modes(self):
        """'event_type' should return 'event_type' for both modes."""
        assert get_encoder_key('event_type', 22) == 'event_type'
        assert get_encoder_key('event_type', 24) == 'event_type'

    def test_price_same_for_both_modes(self):
        """'price' should return 'price' for both modes."""
        assert get_encoder_key('price', 22) == 'price'
        assert get_encoder_key('price', 24) == 'price'

    def test_time_fields_return_time(self):
        """Time-related fields should return 'time'."""
        time_fields = ['delta_t_s', 'delta_t_ns', 'time_s', 'time_ns',
                       'time_s_ref', 'time_ns_ref']
        for field in time_fields:
            assert get_encoder_key(field, 24) == 'time'


# =============================================================================
# syntax_validation_matrix Shape Tests
# =============================================================================

class TestSyntaxValidationMatrixShape:
    """Tests for syntax_validation_matrix shape."""

    def test_shape_22(self, vocab_22):
        """Matrix shape should be (22, vocab_size) for 22-token mode."""
        mask = syntax_validation_matrix(vocab_22)
        assert mask.shape == (22, len(vocab_22))

    def test_shape_24(self, vocab_24):
        """Matrix shape should be (24, vocab_size) for 24-token mode."""
        mask = syntax_validation_matrix(vocab_24)
        assert mask.shape == (24, len(vocab_24))

    def test_returns_boolean_array(self, vocab_24):
        """Matrix should be boolean (True=valid, False=invalid)."""
        mask = syntax_validation_matrix(vocab_24)
        assert mask.dtype == jnp.bool_


# =============================================================================
# syntax_validation_matrix Content Tests
# =============================================================================

class TestSyntaxValidationMatrixContent:
    """Tests for syntax_validation_matrix content."""

    def test_event_type_position_has_4_valid_tokens(self, vocab_24):
        """Position 0 (event_type) should have exactly 4 valid non-special tokens."""
        mask = syntax_validation_matrix(vocab_24)

        # Get event_type encoder values
        keys, vals = vocab_24.ENCODING['event_type']
        # Count valid tokens at position 0 (excluding special tokens 0-3)
        non_special_mask = mask[0, 4:]  # Skip special tokens
        valid_count = jnp.sum(non_special_mask)

        # Should allow exactly 4 event types (1, 2, 3, 4)
        assert valid_count == 4

    def test_direction_position_has_2_valid_tokens(self, vocab_24):
        """Position 1 (direction) should have exactly 2 valid non-special tokens."""
        mask = syntax_validation_matrix(vocab_24)

        # Count valid at position 1
        non_special_mask = mask[1, 4:]
        valid_count = jnp.sum(non_special_mask)

        assert valid_count == 2

    def test_size_position_4_allows_size_digit_tokens_24(self, vocab_24):
        """Position 4 in 24-mode should allow size_digit tokens (100 values)."""
        mask = syntax_validation_matrix(vocab_24)

        # Count valid at position 4
        non_special_mask = mask[4, 4:]
        valid_count = jnp.sum(non_special_mask)

        # Should allow 100 size_digit values (0-99)
        assert valid_count == 100

    def test_size_position_5_allows_size_digit_tokens_24(self, vocab_24):
        """Position 5 in 24-mode should allow size_digit tokens (100 values)."""
        mask = syntax_validation_matrix(vocab_24)

        # Count valid at position 5
        non_special_mask = mask[5, 4:]
        valid_count = jnp.sum(non_special_mask)

        # Should allow 100 size_digit values
        assert valid_count == 100

    def test_size_position_4_allows_size_tokens_22(self, vocab_22):
        """Position 4 in 22-mode should allow size tokens (10000 values)."""
        mask = syntax_validation_matrix(vocab_22)

        # Count valid at position 4
        non_special_mask = mask[4, 4:]
        valid_count = jnp.sum(non_special_mask)

        # Should allow 10000 size values (0-9999)
        assert valid_count == 10000

    def test_position_5_is_delta_t_s_in_22(self, vocab_22):
        """Position 5 in 22-mode should be delta_t_s (time encoder, 1000 values)."""
        mask = syntax_validation_matrix(vocab_22)

        # Count valid at position 5
        non_special_mask = mask[5, 4:]
        valid_count = jnp.sum(non_special_mask)

        # delta_t_s uses time encoder (0-999)
        assert valid_count == 1000


# =============================================================================
# Special Token Tests
# =============================================================================

class TestSpecialTokensInMatrix:
    """Tests for special token handling in syntax_validation_matrix."""

    def test_mask_token_disabled_everywhere(self, vocab_24):
        """MASK token (0) should be invalid at all positions."""
        mask = syntax_validation_matrix(vocab_24)
        assert not mask[:, Vocab.MASK_TOK].any()

    def test_hidden_token_disabled_everywhere(self, vocab_24):
        """HIDDEN token (1) should be invalid at all positions."""
        mask = syntax_validation_matrix(vocab_24)
        assert not mask[:, Vocab.HIDDEN_TOK].any()

    def test_na_token_only_in_ref_fields(self, vocab_24):
        """NA token (2) should only be valid in reference field positions."""
        mask = syntax_validation_matrix(vocab_24)

        # Sync tokenizer to get correct NEW_MSG_LEN
        Message_Tokenizer.set_token_mode(24)
        new_msg_len = Message_Tokenizer.NEW_MSG_LEN

        # NA should be False in non-ref positions (0 to new_msg_len-1)
        assert not mask[:new_msg_len, Vocab.NA_TOK].any()

        # NA should be True in ref positions (new_msg_len onwards)
        assert mask[new_msg_len:, Vocab.NA_TOK].all()

    def test_start_token_disabled_everywhere(self, vocab_24):
        """START token (3) should be invalid at all positions."""
        mask = syntax_validation_matrix(vocab_24)
        assert not mask[:, Vocab.START_TOK].any()


# =============================================================================
# Message_Tokenizer State Sync Tests
# =============================================================================

class TestTokenizerStateSync:
    """Tests verifying syntax_validation_matrix syncs Message_Tokenizer state."""

    def test_syncs_tokenizer_to_24(self, vocab_24):
        """Calling with vocab_24 should sync Message_Tokenizer to 24-token mode."""
        # Set to 22 first (suppress deprecation warning)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(22)
        assert Message_Tokenizer.MSG_LEN == 22

        # Call syntax_validation_matrix with vocab_24
        # Note: syntax_validation_matrix also uses set_token_mode internally
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            mask = syntax_validation_matrix(vocab_24)

        # Message_Tokenizer should now be synced to 24
        assert Message_Tokenizer.MSG_LEN == 24

    def test_syncs_tokenizer_to_22(self, vocab_22):
        """Calling with vocab_22 should sync Message_Tokenizer to 22-token mode."""
        # Set to 24 first (suppress deprecation warning)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            Message_Tokenizer.set_token_mode(24)
        assert Message_Tokenizer.MSG_LEN == 24

        # Call syntax_validation_matrix with vocab_22
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            mask = syntax_validation_matrix(vocab_22)

        # Message_Tokenizer should now be synced to 22
        assert Message_Tokenizer.MSG_LEN == 22


# =============================================================================
# Encoded Message Validation Tests
# =============================================================================

class TestEncodedMessageValidation:
    """Tests verifying encoded messages pass syntax validation."""

    def test_encoded_msg_passes_validation_24(self, vocab_24, sample_raw_msg):
        """Encoded message should pass syntax validation in 24-token mode."""
        from lob.encoding import encode_msg_24

        encoded = encode_msg_24(sample_raw_msg, vocab_24.ENCODING)
        mask = syntax_validation_matrix(vocab_24)

        # Every token in the encoded message should be valid at its position
        for pos in range(24):
            token = int(encoded[pos])
            assert mask[pos, token], f"Token {token} invalid at position {pos}"

    def test_encoded_msg_passes_validation_22(self, vocab_22, sample_raw_msg):
        """Encoded message should pass syntax validation in 22-token mode."""
        from lob.encoding import encode_msg_22

        encoded = encode_msg_22(sample_raw_msg, vocab_22.ENCODING)
        mask = syntax_validation_matrix(vocab_22)

        # Every token in the encoded message should be valid at its position
        for pos in range(22):
            token = int(encoded[pos])
            assert mask[pos, token], f"Token {token} invalid at position {pos}"


# =============================================================================
# Default Vocab Tests
# =============================================================================

class TestDefaultVocab:
    """Tests for syntax_validation_matrix with default Vocab."""

    def test_none_vocab_defaults_to_current_mode(self):
        """Passing v=None should create default Vocab."""
        # This test documents current behavior
        # After Phase 1 fixes, default should be token_mode=24
        mask = syntax_validation_matrix(None)
        # Shape should match the default token_mode
        # Current default is Vocab() which defaults to token_mode=24
        expected_shape = (24, 2112)  # 24 tokens, 2112 vocab size
        assert mask.shape == expected_shape
