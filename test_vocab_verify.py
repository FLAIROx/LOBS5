#!/usr/bin/env python3
"""
验证Policy输出的token范围
对比: 原始参数 vs Perturbed参数
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT')

import numpy as np
import jax
import jax.numpy as jnp

print("=" * 70)
print("Policy Token验证: 原始参数 vs Perturbed参数")
print("=" * 70)

# 1. Vocab结构
from lob.encoding import Vocab
v = Vocab(token_mode=24)
print(f"\n[Vocab] 总大小: {len(v)}")

# Token范围参考
print("\n[Token Ranges]")
for name, (keys, vals) in v.ENCODING.items():
    non_special = vals[vals >= 4]
    if len(non_special) > 0:
        print(f"  {name:12s}: {int(non_special.min()):4d}-{int(non_special.max()):4d}")

# 2. 初始化Trainer
print("\n" + "=" * 70)
print("[初始化ESTrainer]")
print("=" * 70)

DATA_DIR = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"

from es_lobs5.training.es_trainer import ESTrainer, Config

config = Config(
    data_dir=DATA_DIR,
    n_epochs=1,
    n_perturbations=16,
    n_steps=5,
    background_msgs_per_step=10,
    task_size=10,
    noiser='eggrollbs',
    group_size=8,
    use_wandb=False,
)

trainer = ESTrainer(config)
print("  Trainer初始化完成")

# 3. 获取两种CommonParams
print("\n" + "=" * 70)
print("[对比: 原始 vs Perturbed]")
print("=" * 70)

# 原始参数 (iterinfo=None)
policy_unnoised = trainer.create_world_common_params()  # iterinfo=None
print(f"  原始参数 iterinfo: {policy_unnoised.iterinfo}")

# Perturbed参数 (iterinfo=(epoch, thread_id))
policy_noised = trainer.create_policy_common_params(epoch=0, thread_id=0)
print(f"  Perturbed参数 iterinfo: {policy_noised.iterinfo}")

# 4. 运行simulate_episode收集两种输出
print("\n" + "=" * 70)
print("[运行模拟收集Policy Tokens]")
print("=" * 70)

key = jax.random.PRNGKey(42)
initial_sim_state = trainer.initial_sim_state
initial_msg_history = trainer.initial_msg_history

# 运行一次unnoised
print("\n[A] 原始参数 (iterinfo=None):")
key, subkey = jax.random.split(key)
result_unnoised = trainer.simulate_episode(
    subkey,
    world_common_params=policy_unnoised,
    policy_common_params=policy_unnoised,  # 使用unnoised作为policy
    initial_sim_state=initial_sim_state,
    initial_msg_history=initial_msg_history,
    thread_id=0,
)
fitness_unnoised, info_unnoised = result_unnoised
policy_msgs_unnoised = np.array(info_unnoised['policy_msgs'])
print(f"  Shape: {policy_msgs_unnoised.shape}")
print(f"  Token range: [{policy_msgs_unnoised.min()}, {policy_msgs_unnoised.max()}]")
print(f"  Fitness: {fitness_unnoised:.4f}")

# 运行一次noised
print("\n[B] Perturbed参数 (iterinfo=(0, 0)):")
key, subkey = jax.random.split(key)
result_noised = trainer.simulate_episode(
    subkey,
    world_common_params=policy_unnoised,  # world仍用unnoised
    policy_common_params=policy_noised,   # policy用noised
    initial_sim_state=initial_sim_state,
    initial_msg_history=initial_msg_history,
    thread_id=0,
)
fitness_noised, info_noised = result_noised
policy_msgs_noised = np.array(info_noised['policy_msgs'])
print(f"  Shape: {policy_msgs_noised.shape}")
print(f"  Token range: [{policy_msgs_noised.min()}, {policy_msgs_noised.max()}]")
print(f"  Fitness: {fitness_noised:.4f}")

# 5. 对比分析
print("\n" + "=" * 70)
print("[Token对比分析]")
print("=" * 70)

print("\n每个位置的token统计:")
position_names = [
    "event_type", "direction", "price_sign", "price",
    "size_high", "size_low",
    "delta_t_s", "delta_t_ns[0]", "delta_t_ns[1]", "delta_t_ns[2]",
    "time_s[0]", "time_s[1]", "time_ns[0]", "time_ns[1]", "time_ns[2]",
    "price_ref_sign", "price_ref",
    "size_ref_high", "size_ref_low",
    "time_s_ref[0]", "time_s_ref[1]", "time_ns_ref[0]", "time_ns_ref[1]", "time_ns_ref[2]"
]

print(f"{'Pos':<4} {'Field':<15} {'Unnoised':<20} {'Perturbed':<20} {'Same?'}")
print("-" * 70)
for i, name in enumerate(position_names):
    col_u = policy_msgs_unnoised[:, i]
    col_n = policy_msgs_noised[:, i]
    u_str = f"[{col_u.min():4d}, {col_u.max():4d}]"
    n_str = f"[{col_n.min():4d}, {col_n.max():4d}]"
    same = "Yes" if np.array_equal(col_u, col_n) else "No"
    print(f"[{i:2d}] {name:<15} {u_str:<20} {n_str:<20} {same}")

# 6. 检验token有效性
print("\n" + "=" * 70)
print("[Token有效性检查]")
print("=" * 70)

vocab_size = len(v)
for name, msgs in [("Unnoised", policy_msgs_unnoised), ("Perturbed", policy_msgs_noised)]:
    valid = (msgs >= 0) & (msgs < vocab_size)
    invalid_count = (~valid).sum()
    print(f"  {name}: {invalid_count} invalid tokens (out of {msgs.size})")
    if invalid_count > 0:
        invalid_vals = msgs[~valid]
        print(f"    Invalid values: {np.unique(invalid_vals)}")

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)
