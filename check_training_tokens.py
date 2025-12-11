#!/usr/bin/env python3
"""
Check if training data tokens are in correct ranges for 24tok vocab.
This will tell us if the data preprocessing was correct.
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import numpy as np
from lob.encoding import Vocab, encode_msg

print("="*80)
print("Training Data Token Distribution Analysis")
print("="*80)

# Load raw message data
print("\n[1] Loading preprocessed message data...")
data_file = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021/GOOG_2021-01-04_34200000_57600000_message_10_proc.npy'
messages = np.load(data_file)  # (N, 14) message fields
print(f"    Messages shape: {messages.shape}")
print(f"    Sample message[0]: {messages[0]}")

# Encode with 24tok vocab
print("\n[2] Encoding messages with token_mode=24...")
vocab_24 = Vocab(token_mode=24)
print(f"    Vocab size: {len(vocab_24)}")

import jax.numpy as jnp
msg_jnp = jnp.array(messages[:100])  # First 100 messages
tokens_24 = encode_msg(msg_jnp[0], vocab_24.ENCODING)
print(f"    Encoded tokens shape: {tokens_24.shape}")
print(f"    Position [4] (size_h): {tokens_24[4]}")
print(f"    Position [5] (size_l): {tokens_24[5]}")

# Check if size tokens are in valid range (1008-1107)
size_digit_values, size_digit_tokens = vocab_24.ENCODING['size_digit']
valid_size_tokens = size_digit_tokens[(size_digit_values >= 0) & (size_digit_values <= 99)]
print(f"\n[3] Valid size_digit token range: {valid_size_tokens.min()}-{valid_size_tokens.max()}")

is_valid_4 = tokens_24[4] in valid_size_tokens
is_valid_5 = tokens_24[5] in valid_size_tokens
print(f"    Position [4] token {tokens_24[4]} valid: {is_valid_4}")
print(f"    Position [5] token {tokens_24[5]} valid: {is_valid_5}")

# Check distribution across all messages
print("\n[4] Encoding all 100 messages and checking distribution...")
all_tokens = []
for i in range(100):
    tok = encode_msg(msg_jnp[i], vocab_24.ENCODING)
    all_tokens.append(np.array(tok))

all_tokens = np.array(all_tokens)  # (100, 24)
print(f"    Position [4] distribution: min={all_tokens[:, 4].min()}, max={all_tokens[:, 4].max()}")
print(f"    Position [5] distribution: min={all_tokens[:, 5].min()}, max={all_tokens[:, 5].max()}")

n_valid_4 = np.sum(np.isin(all_tokens[:, 4], valid_size_tokens))
n_valid_5 = np.sum(np.isin(all_tokens[:, 5], valid_size_tokens))
print(f"    Position [4] valid tokens: {n_valid_4}/100")
print(f"    Position [5] valid tokens: {n_valid_5}/100")

print("\n" + "="*80)
print("CONCLUSION:")
if n_valid_4 > 90 and n_valid_5 > 90:
    print("✓ Training data tokens are correctly in size_digit range")
    print("  → Problem is NOT in data preprocessing")
    print("  → Problem is in model training or inference")
else:
    print("✗ Training data tokens are NOT in expected range!")
    print("  → Data preprocessing used wrong Vocab")
print("="*80)
