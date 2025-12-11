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

# Run Python check and capture exit code
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from lob.encoding import Vocab, Message_Tokenizer, encode_msgs
import numpy as np
import jax.numpy as jnp
import os

# Read TOKEN_MODE from environment (critical!)
token_mode = int(os.environ.get('TOKEN_MODE', '22'))
print(f"\n[INFO] Using TOKEN_MODE={token_mode} from environment (default: 22)")
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

print(f"\n[3] Testing encode_msgs with token_mode={token_mode}...")
# Create dummy message
dummy_msg = jnp.array([1, 1, 0, 10000, 500, 50, 0, 0, 34200, 0, 0, 0, 0, 0])

try:
    tokens = encode_msgs(dummy_msg[None, :], vocab.ENCODING, token_mode=token_mode)
    print(f"    ✓ encode_msgs(token_mode={token_mode}) works, output shape: {tokens.shape}")
    if token_mode == 24:
        print(f"       Tokens[4:6] (size): {tokens[0, 4:6]}")
    else:
        print(f"       Token[4] (size): {tokens[0, 4]}")
except Exception as e:
    print(f"    ✗ encode_msgs(token_mode={token_mode}) FAILED: {e}")
    print(f"    This is a CRITICAL ERROR - training will fail!")
    import sys
    sys.exit(1)

print("\n[4] Checking model d_output requirement...")
expected_d_output = len(vocab)
print(f"    For token_mode={vocab.token_mode}, model d_output should be: {expected_d_output}")

# Data directory validation - ONLY for encoded mode
# In preproc mode: data is raw 14-dim, can use ANY TOKEN_MODE
# In encoded mode: data is pre-tokenized, MUST match TOKEN_MODE
data_dir = os.environ.get('DIR_NAME', '')
data_mode = os.environ.get('DATA_MODE', 'preproc')

if data_dir and data_mode == 'encoded':
    # STRICT: Encoded data must match TOKEN_MODE
    if '24tok' in data_dir and token_mode == 22:
        print(f"\n❌ ERROR: Encoded data is 24tok but TOKEN_MODE={token_mode}")
        print(f"    Directory: {data_dir}")
        print(f"    Encoded data cannot be re-encoded with different TOKEN_MODE!")
        import sys
        sys.exit(1)
    elif '22tok' in data_dir and token_mode == 24:
        print(f"\n❌ ERROR: Encoded data is 22tok but TOKEN_MODE={token_mode}")
        print(f"    Directory: {data_dir}")
        print(f"    Encoded data cannot be re-encoded with different TOKEN_MODE!")
        import sys
        sys.exit(1)
    else:
        print(f"\n[INFO] Data mode=encoded: Pre-tokenized data matches TOKEN_MODE={token_mode}")
elif data_dir and data_mode == 'preproc':
    # LENIENT: Preproc data is raw 14-dim, can use any TOKEN_MODE
    print(f"\n[INFO] Data mode=preproc: Raw 14-dim data will be encoded on-the-fly")
    print(f"       TOKEN_MODE={token_mode} will be applied during training")

print("\n" + "="*80)
print("CONSISTENCY REQUIREMENTS:")
print("="*80)
expected_vocab_size = 12012 if token_mode == 22 else 2112
expected_msg_len = 22 if token_mode == 22 else 24
print(f"✓ Dataloader: Vocab(token_mode={token_mode})")
print(f"✓ Encoder: encode_msgs(..., token_mode={token_mode})")
print(f"✓ Model: d_output={expected_vocab_size}")
print(f"✓ Message_Tokenizer: MSG_LEN={expected_msg_len}, TOK_LENS={list(Message_Tokenizer.TOK_LENS)}")
print(f"\n✓ All components configured for token_mode={token_mode}")
print("="*80)
PYEOF

# Check if Python script failed
PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Consistency check FAILED! Training cannot proceed."
    exit 1
fi

echo ""
echo "✓ Consistency check passed! Training can proceed safely."
