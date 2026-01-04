"""Test token encode/decode round-trip."""
import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from lob.encoding import encode_msgs, decode_msg, Vocab, Message_Tokenizer


def _create_test_message(event_type=1, direction=1, price=50, size=100,
                         delta_t_s=0, delta_t_ns=1000000, time_s=36000, time_ns=500000000,
                         price_ref=-9999, size_ref=-9999, time_s_ref=-9999, time_ns_ref=-9999):
    """Create a test message with the 14-field format.

    Fields: [order_id, event_type, direction, price_abs, price, size,
             delta_t_s, delta_t_ns, time_s, time_ns,
             price_ref, size_ref, time_s_ref, time_ns_ref]
    """
    return jnp.array([
        -9999,          # order_id (NA)
        event_type,     # event_type: 1=new, 2=cancel, 3=delete, 4=execute
        direction,      # direction: 0=bid, 1=ask
        -9999,          # price_abs (NA)
        price,          # price (relative)
        size,           # size
        delta_t_s,      # delta_t_s
        delta_t_ns,     # delta_t_ns
        time_s,         # time_s
        time_ns,        # time_ns
        price_ref,      # price_ref
        size_ref,       # size_ref
        time_s_ref,     # time_s_ref
        time_ns_ref,    # time_ns_ref
    ], dtype=jnp.int32)


def test_roundtrip_event_type():
    """Test: encode -> decode event_type consistent for all event types."""
    print("\n=== test_roundtrip_event_type ===")

    for token_mode in [22, 24]:
        vocab = Vocab(token_mode=token_mode)
        Message_Tokenizer.set_token_mode(token_mode)

        # Test all valid event types: 1, 2, 3, 4
        for event_type in [1, 2, 3, 4]:
            msg = _create_test_message(event_type=event_type)
            msgs = msg.reshape(1, -1)

            # Encode
            encoded = encode_msgs(msgs, vocab.ENCODING, token_mode=token_mode)

            # Decode
            decoded = decode_msg(encoded[0], vocab.ENCODING, token_mode=token_mode)

            # Check event_type (index 1 in decoded message)
            original_event_type = int(msg[1])
            decoded_event_type = int(decoded[1])

            assert original_event_type == decoded_event_type, (
                f"mode={token_mode}, event_type mismatch: "
                f"original={original_event_type}, decoded={decoded_event_type}"
            )

        print(f"[PASS] mode={token_mode}: event_type roundtrip OK for types [1,2,3,4]")

    return True


def test_roundtrip_side():
    """Test: encode -> decode side (direction) consistent."""
    print("\n=== test_roundtrip_side ===")

    for token_mode in [22, 24]:
        vocab = Vocab(token_mode=token_mode)
        Message_Tokenizer.set_token_mode(token_mode)

        # Test both directions: 0=bid, 1=ask
        for direction in [0, 1]:
            msg = _create_test_message(direction=direction)
            msgs = msg.reshape(1, -1)

            # Encode
            encoded = encode_msgs(msgs, vocab.ENCODING, token_mode=token_mode)

            # Decode
            decoded = decode_msg(encoded[0], vocab.ENCODING, token_mode=token_mode)

            # Check direction (index 2 in decoded message)
            original_direction = int(msg[2])
            decoded_direction = int(decoded[2])

            assert original_direction == decoded_direction, (
                f"mode={token_mode}, direction mismatch: "
                f"original={original_direction}, decoded={decoded_direction}"
            )

        print(f"[PASS] mode={token_mode}: side (direction) roundtrip OK for [0,1]")

    return True


def test_roundtrip_both_modes():
    """Test: mode 22 and 24 both pass full roundtrip."""
    print("\n=== test_roundtrip_both_modes ===")

    # Test various combinations of parameters
    test_cases = [
        # (event_type, direction, price, size)
        (1, 0, 50, 100),
        (1, 1, -30, 500),
        (2, 0, 100, 1000),
        (3, 1, 0, 50),
        (4, 0, -99, 9999),
        (1, 1, 999, 1),
    ]

    for token_mode in [22, 24]:
        vocab = Vocab(token_mode=token_mode)
        Message_Tokenizer.set_token_mode(token_mode)

        for event_type, direction, price, size in test_cases:
            msg = _create_test_message(
                event_type=event_type,
                direction=direction,
                price=price,
                size=size
            )
            msgs = msg.reshape(1, -1)

            # Encode
            encoded = encode_msgs(msgs, vocab.ENCODING, token_mode=token_mode)

            # Verify encoded shape
            expected_tokens = 22 if token_mode == 22 else 24
            assert encoded.shape == (1, expected_tokens), (
                f"mode={token_mode}: expected encoded shape (1, {expected_tokens}), "
                f"got {encoded.shape}"
            )

            # Decode
            decoded = decode_msg(encoded[0], vocab.ENCODING, token_mode=token_mode)

            # Check key fields match
            assert int(decoded[1]) == event_type, (
                f"mode={token_mode}: event_type mismatch"
            )
            assert int(decoded[2]) == direction, (
                f"mode={token_mode}: direction mismatch"
            )
            assert int(decoded[4]) == price, (
                f"mode={token_mode}: price mismatch, expected {price}, got {int(decoded[4])}"
            )
            assert int(decoded[5]) == size, (
                f"mode={token_mode}: size mismatch, expected {size}, got {int(decoded[5])}"
            )

        print(f"[PASS] mode={token_mode}: full roundtrip OK for all test cases")

    return True


if __name__ == "__main__":
    test_roundtrip_event_type()
    test_roundtrip_side()
    test_roundtrip_both_modes()
    print("\nAll token roundtrip tests passed!")
