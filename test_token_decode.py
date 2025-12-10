#!/usr/bin/env python3
"""
Test script to diagnose why decoded quantity is always 0.

Checks:
1. Model's token output distribution
2. Token decoding logic (especially size field)
3. Whether validation is setting qty to 0
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys

# Add paths
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

print("=" * 80)
print("DIAGNOSTIC: Why is decoded quantity always 0?")
print("=" * 80)

# Step 1: Load encoder
print("\n[1] Loading encoder...")
from lob.encoding import Vocab
vocab = Vocab(token_mode=24)  # FIX: Use 24-token mode (base-100 size encoding)
encoder = vocab.ENCODING
print(f"    Vocab size: {vocab.counter}")
print(f"    Size encoder: {encoder.get('size_digit', 'NOT FOUND')}")

# Step 2: Test decoding with sample tokens
print("\n[2] Testing token decoding...")
import lob.encoding as encoding

# Create sample tokens (24 tokens for one message)
# Format: [event_type, direction, sign, price, size_high, size_low, time(9), price_ref(2), size_ref(2), time_ref(5)]
sample_tokens = jnp.array([
    3,    # event_type (new order = 1)
    5,    # direction (sell = 0, buy = 1)
    3,    # price sign
    10,   # price
    5,    # size_high (DEBUG: try non-zero)
    7,    # size_low (DEBUG: try non-zero)
    # time tokens (9 tokens)
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    # price_ref (2 tokens)
    3, 10,
    # size_ref (2 tokens)
    3, 3,
    # time_ref (5 tokens)
    3, 3, 3, 3, 3,
], dtype=jnp.int32)

print(f"    Sample tokens shape: {sample_tokens.shape}")
print(f"    size_high token: {sample_tokens[4]}")
print(f"    size_low token: {sample_tokens[5]}")

# Decode (use decode_msg_24 for 24-token mode)
msg_decoded = encoding.decode_msg_24(sample_tokens, encoder)
print(f"    Decoded message shape: {msg_decoded.shape}")
print(f"    Decoded fields:")
print(f"      event_type (index 1): {msg_decoded[1]}")
print(f"      direction (index 2): {msg_decoded[2]}")
print(f"      price (index 4): {msg_decoded[4]}")
print(f"      SIZE (index 5): {msg_decoded[5]}")  # <-- KEY!

# Step 3: Check size encoding/decoding
print("\n[3] Testing size encoding...")
from lob.encoding import encode, decode

# What does token 5 decode to for size_digit?
if 'size_digit' in encoder:
    size_encoder, size_decoder = encoder['size_digit']
    decoded_5 = decode(5, size_encoder, size_decoder)
    decoded_7 = decode(7, size_encoder, size_decoder)
    print(f"    Token 5 decodes to: {decoded_5}")
    print(f"    Token 7 decodes to: {decoded_7}")
    print(f"    Combined size (5 * 100 + 7): {decoded_5 * 100 + decoded_7}")
else:
    print("    ERROR: size_digit not in encoder!")

# Step 4: Sample from model's log_probs
print("\n[4] Loading model and sampling...")
try:
    from es_lobs5.utils import load_checkpoint_for_es
    from es_lobs5.models import ES_PaddedLobPredModel

    checkpoint_path = "checkpoints/lobs5_d1024_l12_b16_bsz13x4_seed42_jid1684154_3zf0yp50"
    print(f"    Loading checkpoint: {checkpoint_path}")

    lobs5_init, es_tree_key = load_checkpoint_for_es(checkpoint_path)
    print(f"    Checkpoint loaded")

    # Create dummy inputs
    key = jax.random.PRNGKey(42)
    msg_history = jnp.zeros((24,), dtype=jnp.int32)  # Last message tokens
    book_feat = jnp.zeros((503,), dtype=jnp.float32)  # Book features

    # Initialize carry
    fp = lobs5_init.frozen_params
    hidden = ES_PaddedLobPredModel.initialize_carry(
        batch_size=1,
        ssm_size=fp.get('ssm_size', 256),
        n_message_layers=fp.get('n_message_layers', 2),
        n_book_pre_layers=fp.get('n_book_pre_layers', 1),
        n_book_post_layers=fp.get('n_book_post_layers', 1),
        n_fused_layers=fp.get('n_fused_layers', 4),
        d_model=fp.get('d_model', 256),
        conj_sym=fp.get('conj_sym', True),
    )

    # Create CommonParams (no noise)
    from es_lobs5.models.common import CommonParams
    common_params = CommonParams(
        noiser=None,
        frozen_noiser_params=None,
        noiser_params=None,
        params=lobs5_init.params,
        es_tree_key=es_tree_key,
        frozen_params=lobs5_init.frozen_params,
        iterinfo=None,
    )

    print("    Running forward pass...")
    hidden, log_probs = ES_PaddedLobPredModel._forward_step(
        common_params, hidden, msg_history, book_feat[None, :]
    )

    print(f"    log_probs shape: {log_probs.shape}")

    # Sample tokens
    key, sample_key = jax.random.split(key)
    sampled_tokens = jax.random.categorical(sample_key, log_probs[-24:])

    print(f"    Sampled tokens shape: {sampled_tokens.shape}")
    print(f"    Sampled size_high token (index 4): {sampled_tokens[4]}")
    print(f"    Sampled size_low token (index 5): {sampled_tokens[5]}")

    # Decode sampled tokens
    decoded_msg = encoding.decode_msg_24(sampled_tokens, encoder)
    print(f"    Decoded SIZE from sampled tokens: {decoded_msg[5]}")

    # Sample multiple times to see distribution
    print("\n    Sampling 100 times to check size distribution...")
    sizes = []
    for i in range(100):
        key, sample_key = jax.random.split(key)
        tokens = jax.random.categorical(sample_key, log_probs[-24:])
        decoded = encoding.decode_msg_24(tokens, encoder)
        sizes.append(int(decoded[5]))

    sizes = np.array(sizes)
    print(f"    Size statistics:")
    print(f"      Mean: {sizes.mean():.2f}")
    print(f"      Std: {sizes.std():.2f}")
    print(f"      Min: {sizes.min()}")
    print(f"      Max: {sizes.max()}")
    print(f"      Zeros: {np.sum(sizes == 0)} / 100")
    print(f"      Non-zeros: {np.sum(sizes > 0)} / 100")

except Exception as e:
    print(f"    Error in model test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("If SIZE is always 0:")
print("  → Model likely trained on data where size=0 is common (cancels, etc.)")
print("  → ES needs to explore to find non-zero sizes")
print("  → Increase sigma or adjust ES strategy")
print("\nIf SIZE varies:")
print("  → Decoding works, check validation logic in get_sim_msg_es()")
print("=" * 80)
