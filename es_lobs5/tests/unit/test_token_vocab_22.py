"""Test token mode 22 vocabulary."""
import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from lob.encoding import Vocab, encode_msgs


def test_vocab_size_22():
    """Test: vocab_size = 12012 for token mode 22."""
    vocab = Vocab(token_mode=22)
    vocab_size = len(vocab)

    assert vocab_size == 12012, f"Expected vocab_size=12012, got {vocab_size}"
    print(f"[PASS] test_vocab_size_22: vocab_size = {vocab_size}")
    return True


def test_token_shape_22():
    """Test: encode returns shape (N, 22) for token mode 22."""
    vocab = Vocab(token_mode=22)
    encoding = vocab.ENCODING

    # Create synthetic message data with correct shape
    # Message format: [OID, event_type, direction, price_abs, price, size,
    #                  delta_t_s, delta_t_ns, time_s, time_ns,
    #                  price_ref, size_ref, time_s_ref, time_ns_ref]
    # Total 14 fields
    N = 10  # number of messages
    msgs = np.zeros((N, 14), dtype=np.int32)

    # Set valid values for each field
    msgs[:, 0] = 0  # OID (not used in encoding)
    msgs[:, 1] = 1  # event_type (1-4)
    msgs[:, 2] = 1  # direction (0 or 1)
    msgs[:, 3] = 10000  # price_abs (not used in encoding)
    msgs[:, 4] = 100  # price (relative, -999 to 999)
    msgs[:, 5] = 500  # size (0-9999)
    msgs[:, 6] = 0  # delta_t_s
    msgs[:, 7] = 1000000  # delta_t_ns
    msgs[:, 8] = 36000  # time_s
    msgs[:, 9] = 500000000  # time_ns
    msgs[:, 10] = 50  # price_ref
    msgs[:, 11] = 200  # size_ref
    msgs[:, 12] = 35000  # time_s_ref
    msgs[:, 13] = 250000000  # time_ns_ref

    msgs_jnp = jnp.array(msgs, dtype=jnp.int32)

    # Encode messages
    encoded = encode_msgs(msgs_jnp, encoding, token_mode=22)

    # Check shape
    expected_shape = (N, 22)
    assert encoded.shape == expected_shape, f"Expected shape {expected_shape}, got {encoded.shape}"
    print(f"[PASS] test_token_shape_22: encoded shape = {encoded.shape}")
    return True


def test_max_token_22():
    """Test: max token < 12012 for token mode 22."""
    vocab = Vocab(token_mode=22)
    vocab_size = len(vocab)
    encoding = vocab.ENCODING

    # Create synthetic message data
    N = 10
    msgs = np.zeros((N, 14), dtype=np.int32)

    # Set valid values
    msgs[:, 0] = 0  # OID
    msgs[:, 1] = np.random.randint(1, 5, size=N)  # event_type (1-4)
    msgs[:, 2] = np.random.randint(0, 2, size=N)  # direction (0 or 1)
    msgs[:, 3] = 10000  # price_abs
    msgs[:, 4] = np.random.randint(-999, 1000, size=N)  # price
    msgs[:, 5] = np.random.randint(0, 10000, size=N)  # size (0-9999)
    msgs[:, 6] = np.random.randint(0, 10, size=N)  # delta_t_s
    msgs[:, 7] = np.random.randint(0, 1000000000, size=N)  # delta_t_ns
    msgs[:, 8] = np.random.randint(34200, 57600, size=N)  # time_s
    msgs[:, 9] = np.random.randint(0, 1000000000, size=N)  # time_ns
    msgs[:, 10] = np.random.randint(-999, 1000, size=N)  # price_ref
    msgs[:, 11] = np.random.randint(0, 10000, size=N)  # size_ref
    msgs[:, 12] = np.random.randint(34200, 57600, size=N)  # time_s_ref
    msgs[:, 13] = np.random.randint(0, 1000000000, size=N)  # time_ns_ref

    msgs_jnp = jnp.array(msgs, dtype=jnp.int32)

    # Encode messages
    encoded = encode_msgs(msgs_jnp, encoding, token_mode=22)

    # Check max token is within vocab
    max_token = int(jnp.max(encoded))
    assert max_token < vocab_size, f"Max token {max_token} >= vocab_size {vocab_size}"
    print(f"[PASS] test_max_token_22: max_token = {max_token} < {vocab_size}")
    return True


if __name__ == "__main__":
    test_vocab_size_22()
    test_token_shape_22()
    test_max_token_22()
    print("All token vocab 22 tests passed!")
