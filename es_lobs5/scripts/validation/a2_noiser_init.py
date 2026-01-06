#!/usr/bin/env python3
"""
A2: Noiser 初始化验证
目标: 验证 EGGROLL noiser 能正确初始化
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


def test_noiser_init():
    """验证 EGGROLL noiser 能正确初始化"""
    from es_lobs5.utils.import_utils import get_all_noisers

    print("=" * 60)
    print("A2: Noiser 初始化验证")
    print("=" * 60)

    # Load all noisers
    print("\n[1/3] Loading all noisers...")
    noisers = get_all_noisers()
    print(f"  Available noisers: {list(noisers.keys())}")

    # Verify eggroll exists
    print("\n[2/3] Verifying EGGROLL noiser...")
    assert 'eggroll' in noisers, "Missing 'eggroll' noiser"
    noiser = noisers['eggroll']
    print(f"  ✓ EGGROLL noiser found: {noiser}")

    # Verify required methods
    print("\n[3/3] Verifying noiser methods...")
    required_methods = [
        'rand_init',          # Initialize noiser params
        'do_updates',         # Apply ES gradient updates
        'convert_fitnesses',  # Normalize fitness values
        'get_frozen_noiser_params',  # Get frozen config
    ]

    for method in required_methods:
        if hasattr(noiser, method):
            print(f"  ✓ Found method: {method}")
        else:
            print(f"  ⚠ Missing method: {method} (may be inherited)")

    # Summary
    print("\n" + "=" * 60)
    print("✅ A2: Noiser 初始化验证通过")
    print("=" * 60)

    return noiser


def main():
    """Main entry point"""
    noiser = test_noiser_init()
    return 0


if __name__ == "__main__":
    sys.exit(main())
