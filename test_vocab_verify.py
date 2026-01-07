#!/usr/bin/env python3
"""
实测验证:
1. Policy输出的token是否在正确范围内
2. Decode后的订单字段是否合理
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT')

import numpy as np
import jax
import jax.numpy as jnp

print("=" * 70)
print("Policy Token & Decode 验证")
print("=" * 70)

# ============================================================
# 1. Vocab结构参考
# ============================================================
from lob.encoding import Vocab, decode_msgs
v = Vocab(token_mode=24)

print(f"\n[1] 24-Token Vocab (总大小: {len(v)})")
print("-" * 50)
TOKEN_RANGES = {}
for name, (keys, vals) in v.ENCODING.items():
    non_special = vals[vals >= 4]
    if len(non_special) > 0:
        min_t, max_t = int(non_special.min()), int(non_special.max())
        TOKEN_RANGES[name] = (min_t, max_t)
        print(f"  {name:12s}: {min_t:4d} - {max_t:4d}")

# 每个位置对应的field和范围
POSITION_INFO = [
    (0, "event_type", "event_type"),
    (1, "direction", "direction"),
    (2, "price_sign", "sign"),
    (3, "price", "price"),
    (4, "size_high", "size_digit"),
    (5, "size_low", "size_digit"),
    (6, "delta_t_s", "time"),
    (7, "delta_t_ns[0]", "time"),
    (8, "delta_t_ns[1]", "time"),
    (9, "delta_t_ns[2]", "time"),
    (10, "time_s[0]", "time"),
    (11, "time_s[1]", "time"),
    (12, "time_ns[0]", "time"),
    (13, "time_ns[1]", "time"),
    (14, "time_ns[2]", "time"),
    (15, "price_ref_sign", "sign"),
    (16, "price_ref", "price"),
    (17, "size_ref_high", "size_digit"),
    (18, "size_ref_low", "size_digit"),
    (19, "time_s_ref[0]", "time"),
    (20, "time_s_ref[1]", "time"),
    (21, "time_ns_ref[0]", "time"),
    (22, "time_ns_ref[1]", "time"),
    (23, "time_ns_ref[2]", "time"),
]

# ============================================================
# 2. 运行ES训练收集Policy Token
# ============================================================
print("\n" + "=" * 70)
print("[2] 运行ES训练收集Policy Tokens")
print("=" * 70)

DATA_DIR = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
from es_lobs5.training.es_trainer import ESTrainer
from argparse import Namespace

# Use argparse namespace as config with ALL required attributes
config = Namespace(
    data_dir=DATA_DIR,
    lobs5_checkpoint='/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/',
    n_epochs=1,
    n_perturbations=8,
    n_steps=10,
    background_msgs_per_step=10,
    task_size=50,
    noiser='eggrollbs',
    group_size=8,
    use_wandb=False,
    background_mode='historical_replay',
    seed=42,
    tick_size=100,
    # ES/LoRA parameters required by _init_noiser()
    sigma=0.02,
    lr=0.001,
    lora_rank=8,
    # Token mode (24-token format)
    token_mode=24,
    # Replay data path (same as data_dir for historical_replay mode)
    replay_data_path=DATA_DIR,
    file_idx=0,  # Use first file consistently
)

print("  初始化Trainer...")
trainer = ESTrainer(config)

print("  创建初始状态...")
initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
print(f"  initial_msg_history shape: {initial_msg_history.shape}")

print("  运行1个epoch...")
key = jax.random.PRNGKey(42)
results = trainer.train_epoch(key, epoch=0, initial_sim_state=initial_sim_state, initial_msg_history=initial_msg_history)

# 获取policy_msgs
policy_msgs = np.array(results['info']['policy_msgs'])
print(f"  收集到 policy_msgs shape: {policy_msgs.shape}")
# shape: (n_perturbations, n_steps, msg_len) 或 (n_steps, msg_len)

# Flatten if needed
if policy_msgs.ndim == 3:
    n_pert, n_steps, msg_len = policy_msgs.shape
    policy_msgs_flat = policy_msgs.reshape(-1, msg_len)
else:
    policy_msgs_flat = policy_msgs

print(f"  Flattened shape: {policy_msgs_flat.shape}")
n_orders = policy_msgs_flat.shape[0]

# ============================================================
# 3. 检查Token范围
# ============================================================
print("\n" + "=" * 70)
print("[3] Token范围检查")
print("=" * 70)

print(f"\n{'Pos':<4} {'Field':<15} {'Expected Range':<15} {'Actual Range':<15} {'Valid?'}")
print("-" * 70)

all_valid = True
for pos, name, field_type in POSITION_INFO:
    expected_min, expected_max = TOKEN_RANGES[field_type]
    col = policy_msgs_flat[:, pos]
    actual_min, actual_max = int(col.min()), int(col.max())

    # 检查是否在范围内（包括special tokens 0-3）
    in_range = (col >= 0) & (col <= expected_max)
    valid = in_range.all()
    status = "✓" if valid else "✗ INVALID"

    if not valid:
        all_valid = False
        invalid_vals = col[~in_range]
        status += f" ({len(invalid_vals)} bad)"

    print(f"[{pos:2d}] {name:<15} [{expected_min:4d},{expected_max:4d}]     [{actual_min:4d},{actual_max:4d}]     {status}")

print(f"\n总结: {'ALL VALID ✓' if all_valid else 'SOME INVALID ✗'}")

# ============================================================
# 4. Decode并检查字段合理性
# ============================================================
print("\n" + "=" * 70)
print("[4] Decode后字段合理性检查")
print("=" * 70)

# Decode
decoded = decode_msgs(jnp.array(policy_msgs_flat), v.ENCODING, token_mode=24)
decoded = np.array(decoded)
print(f"  Decoded shape: {decoded.shape}")

# 字段索引 (from decode_msg_24):
# [order_id, event_type, direction, price_abs, price, size,
#  delta_t_s, delta_t_ns, time_s, time_ns,
#  price_ref, size_ref, time_s_ref, time_ns_ref]
FIELD_IDX = {
    'order_id': 0,
    'event_type': 1,
    'direction': 2,
    'price_abs': 3,
    'price': 4,
    'size': 5,
    'delta_t_s': 6,
    'delta_t_ns': 7,
    'time_s': 8,
    'time_ns': 9,
    'price_ref': 10,
    'size_ref': 11,
    'time_s_ref': 12,
    'time_ns_ref': 13,
}

print(f"\n{'Field':<15} {'Min':<10} {'Max':<10} {'Mean':<12} {'Valid?'}")
print("-" * 60)

# 定义合理范围
VALID_RANGES = {
    'event_type': (1, 4),      # 1=new, 2=cancel, 3=delete, 4=execute
    'direction': (0, 1),       # 0=sell, 1=buy
    'price': (-999, 999),      # 相对mid price
    'size': (0, 9999),         # base-100 encoding max = 99*100+99 = 9999
    'delta_t_s': (0, 999),     # 秒
    'price_ref': (-999, 999),
    'size_ref': (0, 9999),
}

for field, (valid_min, valid_max) in VALID_RANGES.items():
    idx = FIELD_IDX[field]
    col = decoded[:, idx]

    # 过滤掉NA值 (-9999)
    valid_col = col[col != -9999]
    if len(valid_col) == 0:
        print(f"{field:<15} {'N/A':<10} {'N/A':<10} {'N/A':<12} (all NA)")
        continue

    min_v, max_v = int(valid_col.min()), int(valid_col.max())
    mean_v = float(valid_col.mean())

    in_range = (valid_col >= valid_min) & (valid_col <= valid_max)
    n_invalid = (~in_range).sum()
    status = "✓" if n_invalid == 0 else f"✗ {n_invalid} out of range"

    print(f"{field:<15} {min_v:<10} {max_v:<10} {mean_v:<12.2f} {status}")

# ============================================================
# 5. 显示几个样本
# ============================================================
print("\n" + "=" * 70)
print("[5] 样本订单 (前5个)")
print("=" * 70)

print(f"\n{'#':<3} {'Type':<6} {'Dir':<4} {'Price':<8} {'Size':<8} {'ΔT(s)':<8}")
print("-" * 50)
for i in range(min(5, n_orders)):
    evt = int(decoded[i, FIELD_IDX['event_type']])
    dir_ = int(decoded[i, FIELD_IDX['direction']])
    price = int(decoded[i, FIELD_IDX['price']])
    size = int(decoded[i, FIELD_IDX['size']])
    dt = int(decoded[i, FIELD_IDX['delta_t_s']])

    evt_str = {1:'NEW', 2:'CXL', 3:'DEL', 4:'EXE'}.get(evt, f'?{evt}')
    dir_str = 'BUY' if dir_ == 1 else 'SELL'

    print(f"{i:<3} {evt_str:<6} {dir_str:<4} {price:<8} {size:<8} {dt:<8}")

# ============================================================
# 6. 统计分布
# ============================================================
print("\n" + "=" * 70)
print("[6] 字段分布统计")
print("=" * 70)

# Event Type分布
evt_types = decoded[:, FIELD_IDX['event_type']]
evt_types = evt_types[evt_types != -9999]
print("\nEvent Type分布:")
for t, name in [(1, 'NEW'), (2, 'CANCEL'), (3, 'DELETE'), (4, 'EXECUTE')]:
    count = (evt_types == t).sum()
    pct = 100 * count / len(evt_types) if len(evt_types) > 0 else 0
    print(f"  {name}: {count} ({pct:.1f}%)")

# Direction分布
dirs = decoded[:, FIELD_IDX['direction']]
dirs = dirs[dirs != -9999]
print("\nDirection分布:")
sell_pct = 100 * (dirs == 0).sum() / len(dirs) if len(dirs) > 0 else 0
buy_pct = 100 * (dirs == 1).sum() / len(dirs) if len(dirs) > 0 else 0
print(f"  SELL: {(dirs == 0).sum()} ({sell_pct:.1f}%)")
print(f"  BUY:  {(dirs == 1).sum()} ({buy_pct:.1f}%)")

# Price分布
prices = decoded[:, FIELD_IDX['price']]
prices = prices[prices != -9999]
print("\nPrice分布 (相对mid):")
print(f"  Mean: {prices.mean():.2f}")
print(f"  Std:  {prices.std():.2f}")
print(f"  Range: [{prices.min()}, {prices.max()}]")

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)
