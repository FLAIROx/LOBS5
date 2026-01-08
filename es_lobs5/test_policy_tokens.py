"""
Debug script to inspect raw policy token values.

This checks if the size tokens are being generated correctly.
For 24-token mode, size is encoded as:
  - pos 4: size_digit_high (0-99, encoded as tokens 1008-1107)
  - pos 5: size_digit_low (0-99, encoded as tokens 1008-1107)
  - Size = high * 100 + low
"""

import os
import sys

LOBS5_ROOT = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5'
sys.path.insert(0, LOBS5_ROOT)
sys.path.insert(0, os.path.join(LOBS5_ROOT, 'HyperscaleES', 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

from lob.encoding import Vocab, decode_msgs


@dataclass
class ESConfig:
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    noiser: str = 'eggrollbs'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    group_size: int = 8
    n_perturbations: int = 8
    n_epochs: int = 1
    n_steps: int = 5  # Very short
    background_msgs_per_step: int = 5
    token_mode: int = 24
    background_mode: str = 'historical_replay'
    replay_data_path: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc'
    data_dir: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc'
    file_idx: int = 0
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    grad_clip: float = 1.0
    seed: int = 42
    n_warmup_msgs: int = 50
    temperature: float = 1.0


def main():
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 60)
    print(" Debug: Inspecting Raw Policy Token Values")
    print("=" * 60)

    config = ESConfig()
    print(f"[*] Initializing ESTrainer with token_mode={config.token_mode}...")
    trainer = ESTrainer(config)

    print(f"[*] Creating initial state...")
    initial_state, initial_msg_history = trainer._create_initial_sim_state()

    print(f"\n[*] Running 1 episode to collect policy tokens...")
    key = jax.random.PRNGKey(config.seed)
    fit, info = trainer.eval_single_thread(key, 0, 0, initial_state, initial_msg_history)

    policy_msgs = info.get('policy_msgs')
    if policy_msgs is None:
        print("ERROR: No policy_msgs in info dict!")
        return

    policy_msgs = np.array(policy_msgs)
    print(f"\n[*] policy_msgs shape: {policy_msgs.shape}")
    print(f"[*] policy_msgs dtype: {policy_msgs.dtype}")

    # For 24-token mode:
    # Token positions for size:
    #   pos 4 = size_digit_high (tokens 1008-1107 for values 0-99)
    #   pos 5 = size_digit_low (tokens 1008-1107 for values 0-99)
    # Token 1008 = 0, Token 1107 = 99
    SIZE_TOKEN_START = 1008  # First size_digit token

    print("\n" + "=" * 60)
    print(" RAW TOKEN VALUES (positions 4 and 5 are size)")
    print("=" * 60)

    for i, msg in enumerate(policy_msgs):
        print(f"\n  Msg {i}: {msg[:8]}...")  # First 8 tokens

        # Extract size tokens
        size_high_tok = int(msg[4])
        size_low_tok = int(msg[5])

        # Decode to values
        size_high = size_high_tok - SIZE_TOKEN_START
        size_low = size_low_tok - SIZE_TOKEN_START
        size = size_high * 100 + size_low

        print(f"    pos[4] = {size_high_tok} -> size_high = {size_high}")
        print(f"    pos[5] = {size_low_tok} -> size_low = {size_low}")
        print(f"    -> SIZE = {size}")

    # Also decode using Vocab
    print("\n" + "=" * 60)
    print(" DECODED USING decode_msgs (token_mode=24)")
    print("=" * 60)

    v = Vocab(token_mode=24)
    decoded = decode_msgs(policy_msgs, v.ENCODING, token_mode=24)
    decoded = np.array(decoded)

    print(f"\n[*] decoded shape: {decoded.shape}")
    print(f"[*] Column 5 (size) values:")
    for i, row in enumerate(decoded):
        print(f"    Msg {i}: size = {row[5]}")

    print(f"\n[*] Size statistics:")
    print(f"    mean: {decoded[:, 5].mean():.1f}")
    print(f"    median: {np.median(decoded[:, 5]):.1f}")
    print(f"    min: {decoded[:, 5].min()}")
    print(f"    max: {decoded[:, 5].max()}")


if __name__ == '__main__':
    main()
