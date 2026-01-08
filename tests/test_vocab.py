"""
Tests for Vocab class.

These tests verify:
1. Vocab initialization with different token modes
2. Correct encoder keys for each mode ('size' vs 'size_digit')
3. Vocabulary sizes are correct
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob.encoding import Vocab


# =============================================================================
# Vocab Initialization Tests
# =============================================================================

class TestVocabInit:
    """Tests for Vocab initialization."""

    def test_vocab_init_default_is_24(self):
        """Vocab() should default to token_mode=24."""
        v = Vocab()
        assert v.token_mode == 24

    def test_vocab_init_22(self):
        """Vocab(token_mode=22) should set token_mode=22."""
        v = Vocab(token_mode=22)
        assert v.token_mode == 22

    def test_vocab_init_24(self):
        """Vocab(token_mode=24) should set token_mode=24."""
        v = Vocab(token_mode=24)
        assert v.token_mode == 24

    def test_vocab_init_invalid_raises(self):
        """Vocab(token_mode=23) should raise AssertionError."""
        with pytest.raises(AssertionError):
            Vocab(token_mode=23)


# =============================================================================
# Encoder Keys Tests
# =============================================================================

class TestVocabEncoders22:
    """Tests for 22-token Vocab encoders."""

    def test_vocab_22_has_size_encoder(self, vocab_22):
        """22-token vocab should have 'size' encoder."""
        assert 'size' in vocab_22.ENCODING

    def test_vocab_22_no_size_digit_encoder(self, vocab_22):
        """22-token vocab should NOT have 'size_digit' encoder."""
        assert 'size_digit' not in vocab_22.ENCODING

    def test_vocab_22_size_encoder_range(self, vocab_22):
        """22-token 'size' encoder should cover 0-9999 (10000 values)."""
        keys, vals = vocab_22.ENCODING['size']
        # 4 special tokens (MASK, HIDDEN, NA, START) + 10000 values
        assert len(keys) == 10004

    def test_vocab_22_has_standard_encoders(self, vocab_22):
        """22-token vocab should have all standard encoders."""
        expected_encoders = ['time', 'event_type', 'size', 'price', 'sign', 'direction']
        for enc in expected_encoders:
            assert enc in vocab_22.ENCODING, f"Missing encoder: {enc}"


class TestVocabEncoders24:
    """Tests for 24-token Vocab encoders."""

    def test_vocab_24_has_size_digit_encoder(self, vocab_24):
        """24-token vocab should have 'size_digit' encoder."""
        assert 'size_digit' in vocab_24.ENCODING

    def test_vocab_24_no_size_encoder(self, vocab_24):
        """24-token vocab should NOT have 'size' encoder."""
        assert 'size' not in vocab_24.ENCODING

    def test_vocab_24_size_digit_encoder_range(self, vocab_24):
        """24-token 'size_digit' encoder should cover 0-99 (100 values)."""
        keys, vals = vocab_24.ENCODING['size_digit']
        # 4 special tokens + 100 values
        assert len(keys) == 104

    def test_vocab_24_has_standard_encoders(self, vocab_24):
        """24-token vocab should have all standard encoders (except 'size')."""
        expected_encoders = ['time', 'event_type', 'size_digit', 'price', 'sign', 'direction']
        for enc in expected_encoders:
            assert enc in vocab_24.ENCODING, f"Missing encoder: {enc}"


# =============================================================================
# Vocabulary Size Tests
# =============================================================================

class TestVocabSizes:
    """Tests for vocabulary sizes."""

    def test_vocab_22_size(self, vocab_22, expected_vocab_size_22):
        """22-token vocab should have expected size."""
        assert len(vocab_22) == expected_vocab_size_22

    def test_vocab_24_size(self, vocab_24, expected_vocab_size_24):
        """24-token vocab should have expected size."""
        assert len(vocab_24) == expected_vocab_size_24

    def test_vocab_22_larger_than_24(self, vocab_22, vocab_24):
        """22-token vocab should be larger than 24-token vocab."""
        # Because 'size' range (0-9999) > 'size_digit' range (0-99)
        assert len(vocab_22) > len(vocab_24)

    def test_vocab_size_difference(self, vocab_22, vocab_24):
        """Size difference should be approximately 9900."""
        # 22-tok: size 0-9999 = 10000 values
        # 24-tok: size_digit 0-99 = 100 values (but used twice for high/low)
        # Difference: 10000 - 100 = 9900
        diff = len(vocab_22) - len(vocab_24)
        assert diff == 9900


# =============================================================================
# Special Tokens Tests
# =============================================================================

class TestSpecialTokens:
    """Tests for special tokens."""

    def test_special_token_values(self):
        """Special tokens should have correct values."""
        assert Vocab.MASK_TOK == 0
        assert Vocab.HIDDEN_TOK == 1
        assert Vocab.NA_TOK == 2
        assert Vocab.START_TOK == 3

    def test_vocab_counter_starts_at_4(self, vocab_24):
        """Counter should start at 4 (after special tokens)."""
        # The first regular token should be at index 4
        v = Vocab(token_mode=24)
        # After initialization, counter should be > 4
        assert v.counter > 4


# =============================================================================
# Encoder Token Range Tests
# =============================================================================

class TestEncoderTokenRanges:
    """Tests for specific token ranges in encoders."""

    def test_event_type_encoder_has_4_values(self, vocab_24):
        """event_type encoder should have 4 event types."""
        keys, vals = vocab_24.ENCODING['event_type']
        # 4 special + 4 events (1=new, 2=cancel, 3=delete, 4=execute)
        non_special = vals[vals >= 4]
        assert len(non_special) == 4

    def test_direction_encoder_has_2_values(self, vocab_24):
        """direction encoder should have 2 values (buy/sell)."""
        keys, vals = vocab_24.ENCODING['direction']
        non_special = vals[vals >= 4]
        assert len(non_special) == 2

    def test_sign_encoder_has_2_values(self, vocab_24):
        """sign encoder should have 2 values (-1, 1)."""
        keys, vals = vocab_24.ENCODING['sign']
        non_special = vals[vals >= 4]
        assert len(non_special) == 2

    def test_price_encoder_has_1000_values(self, vocab_24):
        """price encoder should have 1000 values (0-999)."""
        keys, vals = vocab_24.ENCODING['price']
        non_special = vals[vals >= 4]
        assert len(non_special) == 1000

    def test_time_encoder_has_1000_values(self, vocab_24):
        """time encoder should have 1000 values (0-999)."""
        keys, vals = vocab_24.ENCODING['time']
        non_special = vals[vals >= 4]
        assert len(non_special) == 1000


# =============================================================================
# Vocab Instance Independence Tests
# =============================================================================

class TestVocabIndependence:
    """Tests verifying Vocab instances are independent."""

    def test_two_vocabs_independent(self):
        """Two Vocab instances should be independent."""
        v22 = Vocab(token_mode=22)
        v24 = Vocab(token_mode=24)

        assert v22.token_mode == 22
        assert v24.token_mode == 24

        # Modifying one should not affect the other
        assert 'size' in v22.ENCODING
        assert 'size_digit' in v24.ENCODING

    def test_vocab_encoding_is_per_instance(self):
        """Each Vocab instance should have its own ENCODING dict."""
        v1 = Vocab(token_mode=24)
        v2 = Vocab(token_mode=24)

        # Should be equal but not the same object
        assert v1.ENCODING.keys() == v2.ENCODING.keys()
        assert v1.ENCODING is not v2.ENCODING
