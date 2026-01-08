"""
Pytest fixtures for token mode compatibility tests.

These fixtures provide reusable test data for both 22-token and 24-token modes.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob.encoding import Vocab, Message_Tokenizer


# =============================================================================
# Vocab Fixtures
# =============================================================================

@pytest.fixture
def vocab_22():
    """22-token vocabulary (base-10000 size encoding)."""
    return Vocab(token_mode=22)


@pytest.fixture
def vocab_24():
    """24-token vocabulary (base-100 size encoding)."""
    return Vocab(token_mode=24)


@pytest.fixture(params=[22, 24])
def token_mode(request):
    """Parametrize tests to run for both token modes."""
    return request.param


@pytest.fixture
def vocab(token_mode):
    """Get vocab for parametrized token_mode."""
    return Vocab(token_mode=token_mode)


# =============================================================================
# Message_Tokenizer Fixtures (for new instance-based API)
# =============================================================================

@pytest.fixture
def tokenizer_22():
    """22-token Message_Tokenizer instance."""
    return Message_Tokenizer(token_mode=22)


@pytest.fixture
def tokenizer_24():
    """24-token Message_Tokenizer instance."""
    return Message_Tokenizer(token_mode=24)


@pytest.fixture
def tokenizer(token_mode):
    """Get tokenizer for parametrized token_mode."""
    return Message_Tokenizer(token_mode=token_mode)


# =============================================================================
# Sample Message Fixtures
# =============================================================================

@pytest.fixture
def sample_raw_msg():
    """Sample 14-field raw message (before encoding).

    Fields: [order_id, event_type, direction, price_abs, price, size,
             delta_t_s, delta_t_ns, time_s, time_ns,
             price_ref, size_ref, time_s_ref, time_ns_ref]
    """
    return jnp.array([
        12345,      # order_id (not encoded)
        1,          # event_type: NEW (1=new, 2=cancel, 3=delete, 4=execute)
        1,          # direction: BUY (0=sell, 1=buy)
        100000,     # price_abs (not encoded, used for simulation)
        50,         # price (relative to mid, range ±999)
        2345,       # size (will be 23*100+45 in 24-tok, or 2345 in 22-tok)
        0,          # delta_t_s (seconds since last msg)
        1000000,    # delta_t_ns (nanoseconds)
        36000,      # time_s (10:00:00 = 36000 seconds)
        500000000,  # time_ns (500ms)
        -9999,      # price_ref (NA for new orders)
        -9999,      # size_ref (NA)
        -9999,      # time_s_ref (NA)
        -9999,      # time_ns_ref (NA)
    ], dtype=jnp.int32)


@pytest.fixture
def sample_raw_msg_with_ref():
    """Sample message with reference fields (for cancel/delete/execute)."""
    return jnp.array([
        12345,      # order_id
        3,          # event_type: DELETE
        0,          # direction: SELL
        100000,     # price_abs
        -30,        # price (relative)
        500,        # size
        0,          # delta_t_s
        500000,     # delta_t_ns
        36001,      # time_s
        0,          # time_ns
        -30,        # price_ref (same as price for delete)
        500,        # size_ref (same as size)
        36000,      # time_s_ref (reference time)
        999000000,  # time_ns_ref
    ], dtype=jnp.int32)


@pytest.fixture
def sample_batch_msgs():
    """Batch of sample messages for vectorized tests."""
    return jnp.array([
        [0, 1, 1, 0, 50, 500, 0, 0, 36000, 0, -9999, -9999, -9999, -9999],
        [0, 2, 0, 0, -30, 100, 1, 0, 36001, 0, 50, 100, 36000, 0],
        [0, 4, 1, 0, 0, 200, 0, 1000, 36001, 500000, 0, 200, 36001, 0],
    ], dtype=jnp.int32)


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def max_size_msg():
    """Message with maximum size (9999)."""
    return jnp.array([
        0, 1, 1, 0, 0, 9999, 0, 0, 36000, 0, -9999, -9999, -9999, -9999
    ], dtype=jnp.int32)


@pytest.fixture
def min_size_msg():
    """Message with minimum size (1)."""
    return jnp.array([
        0, 1, 1, 0, 0, 1, 0, 0, 36000, 0, -9999, -9999, -9999, -9999
    ], dtype=jnp.int32)


@pytest.fixture
def zero_size_msg():
    """Message with zero size (edge case)."""
    return jnp.array([
        0, 1, 1, 0, 0, 0, 0, 0, 36000, 0, -9999, -9999, -9999, -9999
    ], dtype=jnp.int32)


# =============================================================================
# Expected Values
# =============================================================================

@pytest.fixture
def expected_vocab_size_22():
    """Expected vocabulary size for 22-token mode."""
    # size range: 0-9999 = 10000 values
    # Other fields contribute fixed amounts
    return 12012


@pytest.fixture
def expected_vocab_size_24():
    """Expected vocabulary size for 24-token mode."""
    # size_digit range: 0-99 = 100 values (used twice for high/low)
    return 2112


# =============================================================================
# Test Environment Setup
# =============================================================================

@pytest.fixture(autouse=True)
def reset_tokenizer_state():
    """Reset Message_Tokenizer class state before each test.

    This ensures tests don't affect each other through global state.
    Note: This fixture will be less important after Phase 3 refactoring.
    """
    import warnings

    # Store original state
    original_tok_lens = Message_Tokenizer.TOK_LENS.copy()
    original_msg_len = Message_Tokenizer.MSG_LEN
    original_new_msg_len = Message_Tokenizer.NEW_MSG_LEN
    original_tok_delim = Message_Tokenizer.TOK_DELIM.copy()

    yield

    # Restore original state without triggering deprecation warning
    # (Direct assignment to class attributes)
    Message_Tokenizer.TOK_LENS = original_tok_lens
    Message_Tokenizer.TOK_DELIM = original_tok_delim
    Message_Tokenizer.MSG_LEN = original_msg_len
    Message_Tokenizer.NEW_MSG_LEN = original_new_msg_len
