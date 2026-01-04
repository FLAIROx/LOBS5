"""Test token mode 24 vocabulary."""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
import numpy as np
from lob.encoding import Vocab, encode_msgs


def test_vocab_size_24():
    """Test: vocab_size = 2112 for token_mode 24."""
    vocab = Vocab(token_mode=24)
    assert len(vocab) == 2112, f"Expected vocab size 2112, got {len(vocab)}"
    print("[PASS] test_vocab_size_24: vocab_size = 2112")
    return True


def test_token_shape_24():
    """Test: encode returns shape (N, 24) for token_mode 24."""
    vocab = Vocab(token_mode=24)

    # Create sample messages (14 fields per message):
    # [order_id, event_type, direction, price_abs, price, size,
    #  delta_t_s, delta_t_ns, time_s, time_ns,
    #  price_ref, size_ref, time_s_ref, time_ns_ref]
    N = 5  # Number of messages
    msgs = jnp.array([
        [0, 1, 0, 10000, 50, 100, 0, 1000, 34200, 0, -9999, -9999, -9999, -9999],
        [1, 1, 1, 10100, -50, 200, 0, 2000, 34200, 1000, -9999, -9999, -9999, -9999],
        [2, 2, 0, 10000, 0, 50, 1, 500000, 34201, 500000, 50, 100, 34200, 0],
        [3, 3, 1, 10200, 100, 150, 0, 3000, 34200, 3000, -50, 200, 34200, 1000],
        [4, 4, 0, 10000, 50, 75, 2, 100000, 34202, 100000, 50, 100, 34200, 0],
    ], dtype=jnp.int32)

    # Encode messages
    encoded = encode_msgs(msgs, vocab.ENCODING, token_mode=24)

    assert encoded.shape == (N, 24), f"Expected shape ({N}, 24), got {encoded.shape}"
    print(f"[PASS] test_token_shape_24: encode returns shape {encoded.shape}")
    return True


def test_max_token_24():
    """Test: max token value < 2112 for token_mode 24."""
    vocab = Vocab(token_mode=24)

    # Create sample messages with various values
    N = 10
    msgs = jnp.array([
        [i, (i % 4) + 1, i % 2, 10000 + i * 100, (i - 5) * 10, (i + 1) * 100,
         i % 10, i * 100000, 34200 + i, i * 10000,
         -9999 if (i % 4) == 0 else (i - 5) * 10,
         -9999 if (i % 4) == 0 else (i + 1) * 50,
         -9999 if (i % 4) == 0 else 34200,
         -9999 if (i % 4) == 0 else i * 1000]
        for i in range(N)
    ], dtype=jnp.int32)

    # Encode messages
    encoded = encode_msgs(msgs, vocab.ENCODING, token_mode=24)

    max_token = int(jnp.max(encoded))
    assert max_token < 2112, f"Max token {max_token} >= 2112"
    print(f"[PASS] test_max_token_24: max token = {max_token} < 2112")
    return True


if __name__ == "__main__":
    test_vocab_size_24()
    test_token_shape_24()
    test_max_token_24()
    print("\nAll token vocab 24 tests passed!")
