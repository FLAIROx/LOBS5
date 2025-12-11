#!/usr/bin/env python3
"""
Check if checkpoint vocab_size matches current Vocab size.
This could explain why qty is always 0.
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

print("="*80)
print("Vocab Size Mismatch Diagnostic")
print("="*80)

# Step 1: Load checkpoint config
print("\n[1] Loading checkpoint...")
from es_lobs5.utils.checkpoint_converter import load_checkpoint_for_es

ckpt_path = 'checkpoints/lobs5_d1024_l12_b16_bsz13x4_seed42_jid1684154_3zf0yp50'
lobs5_init, es_tree_key = load_checkpoint_for_es(ckpt_path)

print(f"    Checkpoint d_output (vocab_size): {lobs5_init.frozen_params.get('d_output', 'NOT FOUND')}")

# Step 2: Check current Vocab size
print("\n[2] Creating Vocab()...")
from lob.encoding import Vocab
vocab = Vocab()
print(f"    Vocab() size: {len(vocab)}")
print(f"    Vocab.counter: {vocab.counter}")

# Step 3: Check size_digit encoding
if 'size_digit' in vocab.ENCODING:
    size_digit_tokens = vocab.ENCODING['size_digit'][1]  # token IDs
    print(f"    size_digit token range: {size_digit_tokens.min()}-{size_digit_tokens.max()}")
else:
    print("    size_digit: NOT FOUND (old 22tok mode?)")

# Step 4: Check if size is encoded as single token
if 'size' in vocab.ENCODING:
    size_tokens = vocab.ENCODING['size'][1]
    print(f"    'size' field found (22tok mode): token range {size_tokens.min()}-{size_tokens.max()}")

print("\n" + "="*80)
print("DIAGNOSIS:")
if lobs5_init.frozen_params.get('d_output') == len(vocab):
    print("✓ Vocab sizes MATCH - this is NOT the problem")
else:
    print(f"✗ MISMATCH: checkpoint d_output={lobs5_init.frozen_params.get('d_output')} but Vocab size={len(vocab)}")
    print("  → Model will generate logits in wrong dimension range")
    print("  → This explains why qty is always 0!")
print("="*80)
