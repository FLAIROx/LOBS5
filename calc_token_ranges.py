#!/usr/bin/env python3
"""
Calculate exact token ranges for 22tok vs 24tok to understand the mismatch.
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from lob.encoding import Vocab

print("="*80)
print("Token Range Analysis: 22tok vs 24tok")
print("="*80)

for mode in [22, 24]:
    print(f"\n{'='*40}")
    print(f"Token Mode: {mode}")
    print(f"{'='*40}")

    vocab = Vocab(token_mode=mode)
    print(f"Total vocab size: {len(vocab)}\n")

    # Print each field's token range
    for field_name, (values, tokens) in vocab.ENCODING.items():
        print(f"{field_name:15s}: values {values.min():6d} to {values.max():6d}, "
              f"tokens {tokens.min():5d} to {tokens.max():5d} ({len(tokens):4d} tokens)")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

vocab_22 = Vocab(token_mode=22)
vocab_24 = Vocab(token_mode=24)

print("\nIf checkpoint was trained with 22tok but we decode with 24tok:")
print("  - Model outputs logits for 12012 tokens (22tok range)")
print("  - We sample and get tokens in 0-12000 range")
print("  - But we decode with 24tok mapping (only 2112 tokens)")
print("  - Tokens > 2112 → out of bounds or wrap around")
print("\nObserved tokens [344-890] fall in:")

if 'price' in vocab_22.ENCODING:
    _, price_tokens_22 = vocab_22.ENCODING['price']
    if price_tokens_22.min() <= 890 <= price_tokens_22.max():
        print(f"  ✓ 22tok price range: {price_tokens_22.min()}-{price_tokens_22.max()}")

if 'price' in vocab_24.ENCODING:
    _, price_tokens_24 = vocab_24.ENCODING['price']
    print(f"  24tok price range: {price_tokens_24.min()}-{price_tokens_24.max()}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("If model outputs tokens in 344-890 range, this matches:")
print("  → 22tok 'price' field token range!")
print("\nThis means:")
print("  1. Checkpoint d_output SHOULD be 12012 (22tok), not 2112")
print("  2. OR checkpoint was trained incorrectly with wrong d_output")
print("="*80)
