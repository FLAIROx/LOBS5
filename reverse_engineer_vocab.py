#!/usr/bin/env python3
"""
Reverse engineer the vocabulary that this checkpoint was trained with.
Based on observed token distribution: 344, 459, 463, 780, 890, etc.
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

print("="*80)
print("Reverse Engineering Checkpoint Vocabulary")
print("="*80)

# Observed tokens for size positions
observed_tokens = [344, 459, 463, 780, 890, 425, 430, 453, 526, 701, 464, 782]

print(f"\nObserved size tokens from model output: {observed_tokens}")
print(f"  Min: {min(observed_tokens)}")
print(f"  Max: {max(observed_tokens)}")
print(f"  Range: {max(observed_tokens) - min(observed_tokens)}")

# Hypothesis 1: These are event_type tokens (range 1-4)
# No - tokens are 344-890, way too large

# Hypothesis 2: These are time tokens (range 1000)
# Possible - but time uses 9 tokens, not in size position

# Hypothesis 3: Vocab layout is completely different
print("\n" + "="*80)
print("HYPOTHESIS: Checkpoint uses a different vocab layout")
print("="*80)

# Let's check what fields come BEFORE size in token sequence
# Token sequence for 24tok: [event(1), dir(1), sign(1), price(1), size_h(1), size_l(1), ...]
#                    indices:  [0],      [1],     [2],      [3],      [4],       [5]

# If tokens[4] and tokens[5] are in 300-900 range, they might be:
# - Part of a different field that got shifted
# - Or vocab fields are in different order

print("\nPossible explanations:")
print("1. Checkpoint was trained with 22tok vocab (different token layout)")
print("   - 22tok uses SINGLE token for size (range 0-10000)")
print("   - Token for size is at DIFFERENT position in sequence")
print("   - Vocab size can still be ~2112 if using different field ranges")
print("")
print("2. Checkpoint's vocab field ORDER is different")
print("   - Maybe size comes BEFORE price in the token sequence?")
print("   - Or other fields are inserted/removed")
print("")
print("3. Checkpoint uses a custom vocab variant")
print("   - Different token ranges for each field")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("Need to check the ACTUAL encoding.py used during checkpoint training!")
print("Find training logs or config to see:")
print("  - Exact Vocab() initialization code")
print("  - Field order in message encoding")
print("  - Token ranges for each field")
print("="*80)
