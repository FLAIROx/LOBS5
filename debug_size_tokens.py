#!/usr/bin/env python3
"""
Debug: Why are sampled tokens always decoding to size=0?
Check the actual token→value mapping for size_digit.
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
from lob.encoding import Vocab

print("="*80)
print("Size Token Mapping Diagnostic")
print("="*80)

vocab = Vocab()
encoder = vocab.ENCODING

print(f"\nVocab total size: {len(vocab)}")

# Check size_digit encoding
if 'size_digit' in encoder:
    values, tokens = encoder['size_digit']
    print(f"\n[size_digit] encoding:")
    print(f"  Number of values: {len(values)}")
    print(f"  Values range: {values.min()}-{values.max()}")
    print(f"  Token IDs range: {tokens.min()}-{tokens.max()}")

    # Show first 10 mappings
    print(f"\n  First 10 mappings (value → token):")
    for i in range(min(10, len(values))):
        print(f"    {values[i]:3d} → token {tokens[i]}")

    # Show mappings for value 0-9
    print(f"\n  Mappings for values 0-9:")
    for val in range(10):
        idx = jnp.where(values == val)[0]
        if len(idx) > 0:
            tok = tokens[idx[0]]
            print(f"    value {val} → token {tok}")
        else:
            print(f"    value {val} → NOT FOUND")

# Check if there's a legacy 'size' field (22tok mode)
if 'size' in encoder:
    values_legacy, tokens_legacy = encoder['size']
    print(f"\n[size] encoding (legacy 22tok):")
    print(f"  Number of values: {len(values_legacy)}")
    print(f"  Token range: {tokens_legacy.min()}-{tokens_legacy.max()}")

print("\n" + "="*80)
print("Now test actual token sampling and decoding:")
print("="*80)

# Load checkpoint and sample
print("\n[1] Loading checkpoint...")
from es_lobs5.utils.checkpoint_converter import load_checkpoint_for_es
lobs5_init, _ = load_checkpoint_for_es('checkpoints/lobs5_d1024_l12_b16_bsz13x4_seed42_jid1684154_3zf0yp50')

print("\n[2] Creating model forward function...")
from es_lobs5.models.lob_model import ES_PaddedLobPredModel
from es_lobs5.models.common import CommonParams

# Create dummy input
key = jax.random.PRNGKey(42)
hidden = ES_PaddedLobPredModel.initialize_carry(
    batch_size=1,
    ssm_size=256,
    n_message_layers=2,
    n_book_pre_layers=1,
    n_book_post_layers=1,
    n_fused_layers=12,
    d_model=1024,
    conj_sym=True,
)
msg_hist = jnp.zeros((24,), dtype=jnp.int32)
book_feat = jnp.zeros((503,), dtype=jnp.float32)

common_params = CommonParams(
    noiser=None,
    frozen_noiser_params=None,
    noiser_params=None,
    params=lobs5_init.params,
    es_tree_key=None,
    frozen_params=lobs5_init.frozen_params,
    iterinfo=None,
)

print("\n[3] Running forward pass...")
_, log_probs = ES_PaddedLobPredModel._forward_step(
    common_params, hidden, msg_hist, book_feat[None, :]
)

print(f"    log_probs shape: {log_probs.shape}")
print(f"    log_probs for last 24 positions: {log_probs[-24:].shape}")

# Sample tokens
print("\n[4] Sampling tokens...")
key, sample_key = jax.random.PRNGKey(42), jax.random.PRNGKey(43)
sampled_tokens = jax.random.categorical(sample_key, log_probs[-24:])

print(f"    Sampled tokens: {sampled_tokens}")
print(f"    Token for position 4 (size_high): {sampled_tokens[4]}")
print(f"    Token for position 5 (size_low): {sampled_tokens[5]}")

# Decode
print("\n[5] Decoding tokens...")
from lob import encoding
msg_decoded = encoding.decode_msg_24(sampled_tokens, encoder)

print(f"    Decoded size (index 5): {msg_decoded[5]}")
print(f"    Full decoded message:")
print(f"      event_type: {msg_decoded[1]}")
print(f"      direction: {msg_decoded[2]}")
print(f"      price: {msg_decoded[4]}")
print(f"      size: {msg_decoded[5]}")

# Check if tokens fall in valid range
print("\n[6] Checking token validity...")
if 'size_digit' in encoder:
    _, valid_tokens = encoder['size_digit']
    is_valid_high = sampled_tokens[4] in valid_tokens
    is_valid_low = sampled_tokens[5] in valid_tokens
    print(f"    size_high token {sampled_tokens[4]} valid: {is_valid_high}")
    print(f"    size_low token {sampled_tokens[5]} valid: {is_valid_low}")

print("\n" + "="*80)
