#!/usr/bin/env python
"""
ES-LOBS5 Data Compatibility Test

Validates that Historical Replay data is compatible with ES training.

Usage:
    JAX_PLATFORMS=cpu python es_lobs5/tests/test_data_compatibility.py
"""

import os
import sys
import glob
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lob.encoding import Vocab, encode_msgs, decode_msg


# ============================================================
# Configuration
# ============================================================

DATA_DIRS = {
    'preproc': '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG',
    'encoded_24': '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG',
}

EXPECTED = {
    'msg_cols': 14,
    'ob_cols': 43,
    'event_types': {1, 2, 3, 4},
    'sides': {0, 1},
    'vocab_22': 12012,
    'vocab_24': 2112,
}


# ============================================================
# Test Functions
# ============================================================

def test_data_format(data_dir: str, year: str = '2021') -> bool:
    """Test message and orderbook data format."""
    print(f"\n[Test 1] Data Format ({year})")

    year_dir = os.path.join(data_dir, year)
    if not os.path.exists(year_dir):
        print(f"  ❌ Directory not found: {year_dir}")
        return False

    # Find files
    msg_files = sorted(glob.glob(os.path.join(year_dir, '*message*proc.npy')))
    ob_files = sorted(glob.glob(os.path.join(year_dir, '*orderbook*proc.npy')))

    print(f"  Found {len(msg_files)} message files, {len(ob_files)} orderbook files")

    if len(msg_files) == 0:
        print("  ❌ No message files found")
        return False

    # Check first file
    msg_data = np.load(msg_files[0])

    if msg_data.shape[1] != EXPECTED['msg_cols']:
        print(f"  ❌ Message columns: {msg_data.shape[1]} (expected {EXPECTED['msg_cols']})")
        return False

    print(f"  ✅ Message shape: {msg_data.shape}")

    # Check orderbook if available
    if len(ob_files) > 0:
        ob_data = np.load(ob_files[0])
        if ob_data.shape[1] != EXPECTED['ob_cols']:
            print(f"  ❌ Orderbook columns: {ob_data.shape[1]} (expected {EXPECTED['ob_cols']})")
            return False
        print(f"  ✅ Orderbook shape: {ob_data.shape}")

    return True


def test_data_values(data_dir: str, year: str = '2021') -> bool:
    """Test message data value ranges."""
    print(f"\n[Test 2] Data Values ({year})")

    year_dir = os.path.join(data_dir, year)
    msg_files = sorted(glob.glob(os.path.join(year_dir, '*message*proc.npy')))

    if len(msg_files) == 0:
        print("  ❌ No message files")
        return False

    msg_data = np.load(msg_files[0])

    # Check event types
    event_types = set(np.unique(msg_data[:, 1]))
    if not event_types.issubset(EXPECTED['event_types']):
        print(f"  ❌ Event types: {event_types} (expected subset of {EXPECTED['event_types']})")
        return False
    print(f"  ✅ Event types: {event_types}")

    # Check sides
    sides = set(np.unique(msg_data[:, 2]))
    if not sides.issubset(EXPECTED['sides']):
        print(f"  ❌ Sides: {sides} (expected subset of {EXPECTED['sides']})")
        return False
    print(f"  ✅ Sides: {sides}")

    # Check price range (relative prices can be negative)
    prices = msg_data[:, 4]
    print(f"  ✅ Price range: [{prices.min()}, {prices.max()}]")

    # Check size range (must be positive)
    sizes = msg_data[:, 5]
    if sizes.min() < 0:
        print(f"  ❌ Negative sizes found: min={sizes.min()}")
        return False
    print(f"  ✅ Size range: [{sizes.min()}, {sizes.max()}]")

    return True


def test_token_encoding(data_dir: str, year: str = '2021') -> bool:
    """Test token encoding for both modes."""
    print(f"\n[Test 3] Token Encoding ({year})")

    year_dir = os.path.join(data_dir, year)
    msg_files = sorted(glob.glob(os.path.join(year_dir, '*message*proc.npy')))

    if len(msg_files) == 0:
        print("  ❌ No message files")
        return False

    msg_data = np.load(msg_files[0])[:100]  # First 100 messages

    for token_mode in [22, 24]:
        vocab = Vocab(token_mode=token_mode)
        tokens = encode_msgs(msg_data, vocab.ENCODING, token_mode=token_mode)

        expected_vocab = EXPECTED[f'vocab_{token_mode}']
        actual_max = tokens.max() + 1

        if tokens.shape[1] != token_mode:
            print(f"  ❌ Mode {token_mode}: wrong token count {tokens.shape[1]}")
            return False

        if actual_max > expected_vocab:
            print(f"  ❌ Mode {token_mode}: token {actual_max} > vocab {expected_vocab}")
            return False

        print(f"  ✅ Mode {token_mode}: shape={tokens.shape}, max_token={tokens.max()}")

    return True


def test_encode_decode_roundtrip(data_dir: str, year: str = '2021') -> bool:
    """Test encode/decode round-trip consistency."""
    print(f"\n[Test 4] Encode/Decode Round-trip ({year})")

    year_dir = os.path.join(data_dir, year)
    msg_files = sorted(glob.glob(os.path.join(year_dir, '*message*proc.npy')))

    if len(msg_files) == 0:
        print("  ❌ No message files")
        return False

    msg_data = np.load(msg_files[0])[:10]  # First 10 messages

    for token_mode in [22, 24]:
        vocab = Vocab(token_mode=token_mode)
        tokens = encode_msgs(msg_data, vocab.ENCODING, token_mode=token_mode)

        # Decode and check key fields
        for i in range(min(5, len(tokens))):
            decoded = decode_msg(tokens[i], vocab.ENCODING, token_mode=token_mode)

            # Check event_type (column 1)
            orig_event = msg_data[i, 1]
            decoded_event = decoded[1]

            if orig_event != decoded_event:
                print(f"  ❌ Mode {token_mode}: event mismatch at row {i}")
                print(f"     Original: {orig_event}, Decoded: {decoded_event}")
                return False

        print(f"  ✅ Mode {token_mode}: round-trip OK for event_type")

    return True


def test_data_coverage() -> bool:
    """Test data coverage across years."""
    print("\n[Test 5] Data Coverage")

    data_dir = DATA_DIRS['preproc']
    years = ['2016', '2017', '2018', '2019', '2020', '2021']

    total_days = 0
    for year in years:
        year_dir = os.path.join(data_dir, year)
        if os.path.exists(year_dir):
            files = glob.glob(os.path.join(year_dir, '*message*proc.npy'))
            print(f"  {year}: {len(files)} trading days")
            total_days += len(files)
        else:
            print(f"  {year}: NOT FOUND")

    print(f"  Total: {total_days} trading days")

    if total_days < 1000:
        print("  ⚠️  Less than 1000 days of data")
        return False

    print("  ✅ Sufficient data coverage")
    return True


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("ES-LOBS5 Data Compatibility Test")
    print("=" * 60)

    data_dir = DATA_DIRS['preproc']

    results = []
    results.append(("Data Format", test_data_format(data_dir)))
    results.append(("Data Values", test_data_values(data_dir)))
    results.append(("Token Encoding", test_token_encoding(data_dir)))
    results.append(("Round-trip", test_encode_decode_roundtrip(data_dir)))
    results.append(("Data Coverage", test_data_coverage()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✅ All tests passed! Data is ready for Historical Replay.")
    else:
        print("❌ Some tests failed. Please check the data.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
