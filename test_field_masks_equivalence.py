#!/usr/bin/env python3
"""
Verify that get_field_masks_from_validation_matrix produces the same masks as get_field_masks_24.
"""

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax.numpy as jnp
from es_lobs5.training.es_trainer import get_field_masks_24, get_field_masks_from_validation_matrix

def test_equivalence():
    """Test that old and new implementations produce equivalent masks."""
    token_mode = 24
    vocab_size = 2112

    print(f"Testing field_masks equivalence for token_mode={token_mode}, vocab_size={vocab_size}")

    # Old implementation
    print("Creating masks with get_field_masks_24...")
    old_masks = get_field_masks_24(vocab_size=vocab_size)
    print(f"Old masks shape: {old_masks.shape}")

    # New implementation
    print("Creating masks with get_field_masks_from_validation_matrix...")
    new_masks = get_field_masks_from_validation_matrix(token_mode=token_mode, vocab_size=vocab_size)
    print(f"New masks shape: {new_masks.shape}")

    # Compare
    if jnp.allclose(old_masks, new_masks, atol=1e-6):
        print("\n✅ SUCCESS: Masks are equivalent!")
        return True
    else:
        print("\n❌ FAILURE: Masks differ!")
        diff_positions = jnp.where(jnp.abs(old_masks - new_masks) > 1e-6)
        print(f"Differences at {len(diff_positions[0])} positions")

        # Show first few differences
        for i in range(min(5, len(diff_positions[0]))):
            pos = diff_positions[0][i]
            tok = diff_positions[1][i]
            print(f"  Position {pos}, Token {tok}: old={old_masks[pos, tok]}, new={new_masks[pos, tok]}")

        return False

if __name__ == "__main__":
    success = test_equivalence()
    sys.exit(0 if success else 1)
