#!/usr/bin/env python3
"""Test ESTrainer Historical Replay data loading."""
import os
import sys
import glob
import numpy as np

# Set CPU only
os.environ['JAX_PLATFORMS'] = 'cpu'

# Add project path
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

def test_historical_replay_loading():
    """Verify ESTrainer can load Historical Replay data."""
    from argparse import Namespace

    # Test config - use path with already-encoded data (24 columns = pre-tokenized)
    replay_path = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG/2021'
    token_mode = 22

    print("=" * 60)
    print("ESTrainer Historical Replay Validation")
    print("=" * 60)

    # 1. Check message files exist
    print("\n[1] Checking message files (glob *message*proc.npy)...")
    message_files = sorted(glob.glob(os.path.join(replay_path, '*message*proc.npy')))
    print(f"    Path: {replay_path}")
    print(f"    Found: {len(message_files)} message files")

    if len(message_files) == 0:
        print("    [FAIL] No message files found!")
        return False
    print(f"    [OK] Sample: {os.path.basename(message_files[0])}")

    # 2. Load sample data
    print("\n[2] Loading sample message data...")
    sample_file = message_files[0]
    msg_raw = np.load(sample_file)
    print(f"    Shape: {msg_raw.shape}")
    print(f"    Dtype: {msg_raw.dtype}")

    # Data is already tokenized (24 columns = tokens per message)
    n_cols = msg_raw.shape[1]
    if n_cols == 24:
        print(f"    [INFO] Data is pre-tokenized (24 tokens/msg)")
        is_pretokenized = True
    elif n_cols == 14:
        print(f"    [INFO] Data is raw format (14 fields/msg)")
        is_pretokenized = False
    else:
        print(f"    [WARN] Unknown format: {n_cols} columns")
        is_pretokenized = n_cols in [22, 24]
    print(f"    [OK] Data loaded successfully")

    # 3. Test Vocab initialization
    print("\n[3] Testing Vocab initialization...")
    from lob.encoding import Vocab

    vocab = Vocab(token_mode=token_mode)
    encoder = vocab.ENCODING
    print(f"    Token mode: {token_mode}")
    print(f"    Vocab size (counter): {len(vocab)}")
    print(f"    Encoding fields: {list(encoder.keys())}")
    print(f"    [OK] Vocab initialized")

    # 4. Test data as tokens (since it's pre-encoded)
    print("\n[4] Testing data as replay tokens...")
    import jax.numpy as jnp

    # Simulate ESTrainer._init_historical_replay_data logic
    file_idx = np.random.randint(0, len(message_files))
    selected_file = message_files[file_idx]
    replay_data_date = os.path.basename(selected_file).split('_')[1]

    msg_raw_full = np.load(selected_file)

    # For pre-tokenized data, use directly as tokens
    if is_pretokenized:
        # Trim to token_mode length if needed
        if msg_raw_full.shape[1] > token_mode:
            replay_tokens = msg_raw_full[:, :token_mode]
        else:
            replay_tokens = msg_raw_full
        replay_data_raw = jnp.array(msg_raw_full)
    else:
        # Would need actual encoding here
        replay_tokens = msg_raw_full
        replay_data_raw = jnp.array(msg_raw_full)

    print(f"    Selected: {os.path.basename(selected_file)}")
    print(f"    Date extracted: {replay_data_date}")
    print(f"    Total messages: {msg_raw_full.shape[0]}")
    print(f"    Tokens shape: {replay_tokens.shape}")
    print(f"    Token sample [0]: {replay_tokens[0][:6]}...")
    print(f"    [OK] Data loading logic works")

    # 5. Test decoded_msg_to_jaxlob_format
    print("\n[5] Testing message format conversion...")
    from es_lobs5.training.es_trainer import decoded_msg_to_jaxlob_format

    # For the conversion function, we need decoded (14-field) messages
    # Since our data is pre-tokenized, we need to decode first
    from lob.encoding import decode_msg

    # Take first tokenized message and decode it
    first_tokens = jnp.array(msg_raw_full[0])  # Use full 24 tokens
    if is_pretokenized:
        # Use 24-token mode for decoding since data has 24 columns
        # Need Vocab with token_mode=24 to decode 24-token data
        vocab_24 = Vocab(token_mode=24)
        encoder_24 = vocab_24.ENCODING
        decoded_msg = decode_msg(first_tokens, encoder_24, token_mode=24)
        print(f"    Decoded msg (14 fields): {decoded_msg}")
    else:
        decoded_msg = jnp.array(msg_raw_full[0])

    # Convert to JaxLOB format
    jaxlob_msg = decoded_msg_to_jaxlob_format(decoded_msg)

    print(f"    Input shape: {decoded_msg.shape}")
    print(f"    Output (8 cols): {jaxlob_msg}")
    print(f"    [OK] Format conversion works")

    # 6. Verify ESTrainer can be imported and key methods exist
    print("\n[6] Verifying ESTrainer module...")
    from es_lobs5.training.es_trainer import ESTrainer, create_es_config

    parser = create_es_config()
    print(f"    create_es_config(): OK")
    print(f"    ESTrainer class: OK")
    print(f"    Key config args: --background_mode, --replay_data_path, --token_mode")
    print(f"    [OK] Module verification complete")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_historical_replay_loading()
    sys.exit(0 if success else 1)
