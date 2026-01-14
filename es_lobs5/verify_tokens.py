"""
Verify Policy Token Distribution

Test whether baseline (thread_id=0) produces valid tokens
in the expected vocabulary ranges.
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

from lob.encoding import Vocab


@dataclass
class ESConfig:
    """Config for token verification test."""
    lobs5_checkpoint: str = '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/'
    noiser: str = 'eggrollbs'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    group_size: int = 8
    n_perturbations: int = 8
    n_epochs: int = 1
    n_steps: int = 10  # Just 10 steps for quick test
    background_msgs_per_step: int = 10
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
    n_warmup_msgs: int = 500


def print_vocab_ranges():
    """Print expected token ranges."""
    v = Vocab(token_mode=24)
    print("=" * 60)
    print(" Expected 24-Token Vocabulary Ranges")
    print("=" * 60)
    for name, (keys, vals) in v.ENCODING.items():
        non_special_vals = vals[vals >= 4]
        if len(non_special_vals) > 0:
            min_tok = int(non_special_vals.min())
            max_tok = int(non_special_vals.max())
            print(f"  {name:12s}: {min_tok:4d} - {max_tok:4d}")
    print("=" * 60)
    return v


def analyze_tokens(tokens: np.ndarray, name: str, expected_ranges: dict):
    """Analyze token distribution and check against expected ranges."""
    print(f"\n[{name}] Shape: {tokens.shape}")

    # Token position names for 24-token format
    pos_names = [
        'event', 'dir', 'p_sign', 'price', 'sz_hi', 'sz_lo',
        'dt_s', 'dt_ns0', 'dt_ns1', 'dt_ns2', 't_s0', 't_s1', 't_ns0', 't_ns1', 't_ns2',
        'p_ref_sign', 'p_ref', 'sz_ref_hi', 'sz_ref_lo',
        't_ref_s0', 't_ref_s1', 't_ref_ns0', 't_ref_ns1', 't_ref_ns2'
    ]

    # Expected ranges for key positions
    expected = {
        0: ('event_type', 1004, 1007),
        1: ('direction', 2110, 2111),
        2: ('sign', 2108, 2109),
        3: ('price', 1108, 2107),
        4: ('size_digit', 1008, 1107),
        5: ('size_digit', 1008, 1107),
    }

    print(f"  {'Pos':<6} {'Name':<10} {'Min':>6} {'Max':>6} {'Mean':>8} {'Expected':>15} {'OK?':<5}")
    print(f"  {'-'*60}")

    all_ok = True
    for pos in range(min(10, tokens.shape[1])):
        col = tokens[:, pos]
        min_v, max_v, mean_v = col.min(), col.max(), col.mean()

        if pos in expected:
            field, exp_min, exp_max = expected[pos]
            ok = (min_v >= exp_min and max_v <= exp_max)
            exp_str = f"{exp_min}-{exp_max}"
            ok_str = "✓" if ok else "✗"
            if not ok:
                all_ok = False
        else:
            exp_str = "-"
            ok_str = "-"

        name_str = pos_names[pos] if pos < len(pos_names) else f"tok{pos}"
        print(f"  [{pos:2d}]  {name_str:<10} {min_v:6d} {max_v:6d} {mean_v:8.1f} {exp_str:>15} {ok_str:<5}")

    return all_ok


def main():
    from es_lobs5.training.es_trainer import ESTrainer

    # Print vocab ranges
    v = print_vocab_ranges()

    # Initialize trainer
    print("\n[*] Initializing ESTrainer...")
    config = ESConfig()
    trainer = ESTrainer(config)

    # Create initial state
    print("[*] Creating initial state...")
    initial_state, initial_msg_history = trainer._create_initial_sim_state()

    print(f"[*] initial_msg_history shape: {initial_msg_history.shape}")
    print(f"[*] initial_msg_history sample (first 48 tokens):")
    print(f"    {list(initial_msg_history[:48])}")

    # Run baseline (thread_id=0, no noise with EggRollBS)
    print("\n[*] Running baseline episode (thread_id=0, EggRollBS = no noise)...")
    key = jax.random.PRNGKey(config.seed)
    fit, info = trainer.eval_single_thread(key, 0, 0, initial_state, initial_msg_history)

    print(f"[*] Fitness: {float(fit):.4f}")

    if 'policy_msgs' in info:
        policy_tokens = np.array(info['policy_msgs'])
        print(f"[*] Policy tokens shape: {policy_tokens.shape}")

        # Analyze tokens
        print("\n" + "=" * 60)
        print(" Token Analysis")
        print("=" * 60)

        baseline_ok = analyze_tokens(policy_tokens, "Baseline (thread_id=0)", {})

        # Also run noised (thread_id=2)
        print("\n[*] Running noised episode (thread_id=2)...")
        key2 = jax.random.fold_in(key, 2)
        fit2, info2 = trainer.eval_single_thread(key2, 2, 0, initial_state, initial_msg_history)

        if 'policy_msgs' in info2:
            noised_tokens = np.array(info2['policy_msgs'])
            noised_ok = analyze_tokens(noised_tokens, "Noised (thread_id=2)", {})

        # Summary
        print("\n" + "=" * 60)
        print(" Summary")
        print("=" * 60)
        print(f"  Baseline tokens in expected ranges: {'YES' if baseline_ok else 'NO'}")
        print(f"  Noised tokens in expected ranges:   {'YES' if noised_ok else 'NO'}")

        if not baseline_ok:
            print("\n  [!] PROBLEM: Baseline tokens are NOT in expected vocabulary ranges!")
            print("      This indicates the model is NOT outputting valid tokens.")
    else:
        print("[!] No policy_msgs in info!")


if __name__ == '__main__':
    main()
