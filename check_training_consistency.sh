#!/bin/bash
#
# Pre-training consistency check
# Run this BEFORE starting training to ensure all components are aligned
#

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

echo "="========================================================================
echo "LOBS5 Training Consistency Check"
echo "="========================================================================

python3 << 'PYEOF'
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from lob.encoding import Vocab, Message_Tokenizer, encode_msgs
import numpy as np
import jax.numpy as jnp

# Set token mode for Message_Tokenizer (critical!)
token_mode = 24
Message_Tokenizer.set_token_mode(token_mode)

print("\n[1] Checking Vocab configuration...")
vocab = Vocab(token_mode=token_mode)  # Should match what dataloader uses
print(f"    Token mode: {vocab.token_mode}")
print(f"    Vocab size: {len(vocab)}")
print(f"    Has 'size' field: {'size' in vocab.ENCODING}")
print(f"    Has 'size_digit' field: {'size_digit' in vocab.ENCODING}")

print("\n[2] Checking Message_Tokenizer...")
print(f"    MSG_LEN: {Message_Tokenizer.MSG_LEN}")
print(f"    TOK_LENS: {Message_Tokenizer.TOK_LENS}")
print(f"    Size field uses {Message_Tokenizer.TOK_LENS[3]} tokens")

print("\n[3] Testing encode_msgs with different token_modes...")
# Create dummy message
dummy_msg = jnp.array([1, 1, 0, 10000, 500, 50, 0, 0, 34200, 0, 0, 0, 0, 0])

try:
    tokens_22 = encode_msgs(dummy_msg[None, :], vocab.ENCODING, token_mode=22)
    print(f"    ✓ encode_msgs(token_mode=22) works, output shape: {tokens_22.shape}")
except Exception as e:
    print(f"    ✗ encode_msgs(token_mode=22) FAILED: {e}")

try:
    tokens_24 = encode_msgs(dummy_msg[None, :], vocab.ENCODING, token_mode=24)
    print(f"    ✓ encode_msgs(token_mode=24) works, output shape: {tokens_24.shape}")
    print(f"       Tokens[4:6] (size): {tokens_24[0, 4:6]}")
except Exception as e:
    print(f"    ✗ encode_msgs(token_mode=24) FAILED: {e}")

print("\n[4] Checking model d_output requirement...")
expected_d_output = len(vocab)
print(f"    For token_mode={vocab.token_mode}, model d_output should be: {expected_d_output}")

print("\n" + "="*80)
print("CONSISTENCY REQUIREMENTS:")
print("="*80)
print("✓ Dataloader: Vocab(token_mode=24)")
print("✓ Encoder: encode_msgs(..., token_mode=24)")
print("✓ Model: d_output=2112")
print("✓ Message_Tokenizer: MSG_LEN=24, TOK_LENS=[1,1,2,2,...]")
print("\n⚠️  ALL FOUR MUST USE token_mode=24!")
print("="*80)
PYEOF

echo ""
echo "Run this check before training to avoid corrupted checkpoints!"
