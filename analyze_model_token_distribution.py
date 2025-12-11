#!/usr/bin/env python3
"""
Analyze LOBS5 model's token output distribution using its own inference code.
This isolates whether the problem is in LOBS5 inference itself.
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
import numpy as np
from lob.encoding import Vocab, Message_Tokenizer
import lob.train_helpers as train_helpers

print("="*80)
print("LOBS5 Model Token Distribution Analysis")
print("="*80)

# Step 1: Load checkpoint
print("\n[1] Loading checkpoint...")
ckpt_path = 'checkpoints/lobs5_d1024_l12_b16_bsz13x4_seed42_jid1684154_3zf0yp50'

from pathlib import Path
import orbax.checkpoint as ocp

checkpointer = ocp.PyTreeCheckpointer()
ckpt_dir = Path(ckpt_path)

# Find latest checkpoint
ckpt_subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()]
if not ckpt_subdirs:
    raise ValueError(f"No checkpoint subdirs found in {ckpt_path}")

latest_step = max(int(d.name) for d in ckpt_subdirs)
restore_dir = ckpt_dir / str(latest_step)

print(f"    Restoring from step {latest_step}")

# Load params
restored = checkpointer.restore(restore_dir)
params = restored['model']['params']
config_dict = dict(restored.get('config', {}))

print(f"    Config: d_model={config_dict.get('d_model')}, d_output={config_dict.get('d_output')}")

# Step 2: Create vocab
print("\n[2] Creating Vocab...")
vocab = Vocab(token_mode=24)  # Use tok24
encoder = vocab.ENCODING
print(f"    Vocab size: {len(vocab)}")
print(f"    MSG_LEN: {Message_Tokenizer.MSG_LEN}")

# Check size_digit token range
size_digit_vals, size_digit_toks = encoder['size_digit']
valid_size_mask = (size_digit_vals >= 0) & (size_digit_vals <= 99)
valid_size_tokens = size_digit_toks[valid_size_mask]
print(f"    Valid size_digit tokens: {valid_size_tokens.min()}-{valid_size_tokens.max()}")

# Step 3: Create model and test forward pass
print("\n[3] Testing model forward pass...")

from lob.lob_seq_model import PaddedLobPredModel

# Initialize model
model = PaddedLobPredModel(
    d_model=config_dict.get('d_model', 1024),
    d_output=len(vocab),
    d_book=config_dict.get('d_book', 503),
    n_layers=config_dict.get('n_layers', 12),
    ssm_size=config_dict.get('ssm_size', 256),
    blocks=config_dict.get('blocks', 16),
    activation=config_dict.get('activation', 'half_glu1'),
)

# Create initial hidden
hidden = model.initialize_carry(
    batch_size=1,
    ssm_size=config_dict.get('ssm_size', 256),
    n_message_layers=config_dict.get('n_message_layers', 2),
    n_book_pre_layers=config_dict.get('n_book_pre_layers', 1),
    n_book_post_layers=config_dict.get('n_book_post_layers', 1),
    n_fused_layers=config_dict.get('n_fused_layers', 12),
    d_model=config_dict.get('d_model', 1024),
    conj_sym=config_dict.get('conj_sym', True),
)

# Dummy input
msg_hist = jnp.zeros((12000,), dtype=jnp.int32)
book_feat = jnp.zeros((503,), dtype=jnp.float32)

print("    Running forward pass...")
_, logits = model.apply(
    {'params': params},
    hidden,
    msg_hist[None, :],
    book_feat[None, None, :],
    method='__call_rnn__'
)

print(f"    Logits shape: {logits.shape}")

# Step 4: Analyze distribution
print("\n[4] Token Distribution for Last 24 Positions")
print("="*80)

last_24 = logits[0, -24:, :]  # (24, vocab_size)

for pos in range(24):
    # Top tokens
    top_tokens = jnp.argsort(last_24[pos])[-10:][::-1]
    probs = jax.nn.softmax(last_24[pos])[top_tokens]

    # Field name
    cumsum_lens = np.cumsum([0] + list(Message_Tokenizer.TOK_LENS))
    field_idx = np.searchsorted(cumsum_lens, pos, side='right') - 1
    field_name = Message_Tokenizer.FIELDS[field_idx] if field_idx < len(Message_Tokenizer.FIELDS) else "unknown"

    print(f"\nPos {pos:2d} ({field_name:12s}): top_tokens={list(top_tokens[:5])}")

    if pos in [4, 5]:
        n_valid = jnp.sum((top_tokens >= 1008) & (top_tokens <= 1107))
        print(f"         ** SIZE POSITION ** Expected: 1008-1107, Valid: {n_valid}/10")
        if n_valid == 0:
            print(f"         ❌ WRONG! All top tokens outside expected range!")

print("\n" + "="*80)
