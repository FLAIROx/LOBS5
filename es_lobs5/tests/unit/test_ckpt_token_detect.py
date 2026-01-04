"""Test checkpoint token mode detection."""

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.training.token_mode_check import infer_token_mode_from_d_output


def test_detect_mode_22():
    """d_output=12012 -> token_mode=22"""
    result = infer_token_mode_from_d_output(12012)
    assert result == 22, f"Expected 22, got {result}"
    print("[PASS] d_output=12012 -> token_mode=22")


def test_detect_mode_24():
    """d_output=2112 -> token_mode=24"""
    result = infer_token_mode_from_d_output(2112)
    assert result == 24, f"Expected 24, got {result}"
    print("[PASS] d_output=2112 -> token_mode=24")


if __name__ == "__main__":
    test_detect_mode_22()
    test_detect_mode_24()
    print("All checkpoint token detection tests passed!")
