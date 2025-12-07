#!/usr/bin/env python3
"""
分析 LOBS5 输入数据的分布统计
用于确定合适的 normalization 策略

分析内容：
1. OrderBook 原始数据 (preproc)
2. OrderBook 转换后数据 (transformed)
3. Message Token 数据
"""

import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from preproc import transform_L2_state_numpy


def analyze_orderbook_raw(data_dir: str, num_files: int = 10):
    """分析原始 OrderBook 数据"""
    print("=" * 70)
    print("1. OrderBook RAW Data (before transform)")
    print("=" * 70)

    files = sorted([f for f in os.listdir(data_dir) if 'orderbook' in f])[:num_files]

    all_ch0 = []  # delta_mid
    all_ch1 = []  # time_s
    all_ch2 = []  # time_ns
    all_prices = []  # L2 prices
    all_volumes = []  # L2 volumes

    for f in files:
        book = np.load(os.path.join(data_dir, f))
        all_ch0.append(book[:, 0])
        all_ch1.append(book[:, 1])
        all_ch2.append(book[:, 2])
        # L2 data: [price, vol, price, vol, ...]
        all_prices.append(book[:, 3::2].flatten())
        all_volumes.append(book[:, 4::2].flatten())

    ch0 = np.concatenate(all_ch0)
    ch1 = np.concatenate(all_ch1)
    ch2 = np.concatenate(all_ch2)
    prices = np.concatenate(all_prices)
    volumes = np.concatenate(all_volumes)

    print(f"\nAnalyzed {len(files)} files, {len(ch0):,} rows total")

    print(f"\n[Ch0] delta_mid_price:")
    print(f"  Range:       [{ch0.min():.2f}, {ch0.max():.2f}]")
    print(f"  Mean/Std:    {ch0.mean():.4f} / {ch0.std():.4f}")
    print(f"  Percentiles: 1%={np.percentile(ch0, 1):.2f}, 50%={np.percentile(ch0, 50):.2f}, 99%={np.percentile(ch0, 99):.2f}")

    print(f"\n[Ch1] time_s (raw seconds):")
    print(f"  Range:       [{ch1.min():.0f}, {ch1.max():.0f}]")
    print(f"  Trading day: 09:30 (34200s) to 16:00 (57600s)")

    print(f"\n[Ch2] time_ns (raw nanoseconds):")
    print(f"  Range:       [{ch2.min():.0f}, {ch2.max():.0f}]")

    print(f"\n[L2 Prices] (raw cents):")
    print(f"  Range:       [{prices.min():.0f}, {prices.max():.0f}]")
    print(f"  Mean/Std:    {prices.mean():.2f} / {prices.std():.2f}")

    print(f"\n[L2 Volumes] (raw shares):")
    print(f"  Range:       [{volumes.min():.0f}, {volumes.max():.0f}]")
    print(f"  Mean/Std:    {volumes.mean():.2f} / {volumes.std():.2f}")
    print(f"  Percentiles: 1%={np.percentile(volumes, 1):.0f}, 50%={np.percentile(volumes, 50):.0f}, 99%={np.percentile(volumes, 99):.0f}")

    return {
        'ch0': {'min': ch0.min(), 'max': ch0.max(), 'mean': ch0.mean(), 'std': ch0.std()},
        'ch1': {'min': ch1.min(), 'max': ch1.max()},
        'ch2': {'min': ch2.min(), 'max': ch2.max()},
        'volumes': {'min': volumes.min(), 'max': volumes.max(), 'mean': volumes.mean(), 'std': volumes.std()}
    }


def analyze_orderbook_transformed(data_dir: str, num_files: int = 5, book_depth: int = 500):
    """分析转换后的 OrderBook 数据"""
    print("\n" + "=" * 70)
    print("2. OrderBook TRANSFORMED Data (after transform_L2_state_numpy)")
    print("=" * 70)

    files = sorted([f for f in os.listdir(data_dir) if 'orderbook' in f])[:num_files]

    all_ch0 = []
    all_ch1 = []
    all_ch2 = []
    all_vol = []

    for f in files:
        print(f"  Processing {f}...", end=" ", flush=True)
        book_raw = np.load(os.path.join(data_dir, f))
        book_transformed = transform_L2_state_numpy(book_raw, book_depth, 100)
        print(f"shape: {book_raw.shape} -> {book_transformed.shape}")

        all_ch0.append(book_transformed[:, 0])
        all_ch1.append(book_transformed[:, 1])
        all_ch2.append(book_transformed[:, 2])
        all_vol.append(book_transformed[:, 3:].flatten())

    ch0 = np.concatenate(all_ch0)
    ch1 = np.concatenate(all_ch1)
    ch2 = np.concatenate(all_ch2)
    vol = np.concatenate(all_vol)

    print(f"\n[Ch0] delta_mid_price (NOT normalized!):")
    print(f"  Range:       [{ch0.min():.4f}, {ch0.max():.4f}]")
    print(f"  Mean/Std:    {ch0.mean():.6f} / {ch0.std():.4f}")
    print(f"  Percentiles: 1%={np.percentile(ch0, 1):.2f}, 50%={np.percentile(ch0, 50):.2f}, 99%={np.percentile(ch0, 99):.2f}")
    print(f"  >>> PROBLEM: Range ~100x larger than other channels!")

    print(f"\n[Ch1] time_s (normalized to [0,1]):")
    print(f"  Range:       [{ch1.min():.6f}, {ch1.max():.6f}]")
    print(f"  Mean/Std:    {ch1.mean():.6f} / {ch1.std():.6f}")

    print(f"\n[Ch2] time_ns (normalized to [0,1]):")
    print(f"  Range:       [{ch2.min():.6f}, {ch2.max():.6f}]")
    print(f"  Mean/Std:    {ch2.mean():.6f} / {ch2.std():.6f}")

    print(f"\n[Vol] volume image (ch3-{book_depth+2}):")
    print(f"  Range:       [{vol.min():.6f}, {vol.max():.6f}]")
    print(f"  Mean/Std:    {vol.mean():.8f} / {vol.std():.6f}")
    nonzero = vol[vol != 0]
    if len(nonzero) > 0:
        print(f"  Non-zero:    {len(nonzero):,} ({100*len(nonzero)/len(vol):.2f}%)")
        print(f"  Non-zero range: [{nonzero.min():.6f}, {nonzero.max():.6f}]")
        print(f"  Non-zero mean:  {nonzero.mean():.6f}")

    # 计算各通道的量级比
    print(f"\n[Channel Scale Comparison]:")
    ch0_scale = max(abs(ch0.min()), abs(ch0.max()))
    ch1_scale = max(abs(ch1.min()), abs(ch1.max()))
    ch2_scale = max(abs(ch2.min()), abs(ch2.max()))
    vol_scale = max(abs(vol.min()), abs(vol.max()))

    print(f"  Ch0 (delta_mid): scale = {ch0_scale:.2f}")
    print(f"  Ch1 (time_s):    scale = {ch1_scale:.2f}")
    print(f"  Ch2 (time_ns):   scale = {ch2_scale:.2f}")
    print(f"  Vol:             scale = {vol_scale:.2f}")
    print(f"  >>> Ch0 is {ch0_scale/vol_scale:.1f}x larger than Vol!")

    return {
        'ch0': {'min': ch0.min(), 'max': ch0.max(), 'mean': ch0.mean(), 'std': ch0.std(),
                'p1': np.percentile(ch0, 1), 'p99': np.percentile(ch0, 99)},
        'ch1': {'min': ch1.min(), 'max': ch1.max()},
        'ch2': {'min': ch2.min(), 'max': ch2.max()},
        'vol': {'min': vol.min(), 'max': vol.max(), 'mean': vol.mean(), 'std': vol.std()}
    }


def analyze_message_tokens(data_dir: str, num_files: int = 10):
    """分析 Message Token 数据"""
    print("\n" + "=" * 70)
    print("3. Message Token Data")
    print("=" * 70)

    files = sorted([f for f in os.listdir(data_dir) if 'message' in f])[:num_files]

    all_msgs = []
    for f in files:
        msg = np.load(os.path.join(data_dir, f))
        all_msgs.append(msg)

    msgs = np.concatenate(all_msgs)
    print(f"\nAnalyzed {len(files)} files, {len(msgs):,} messages")
    print(f"Shape: {msgs.shape}, dtype: {msgs.dtype}")

    # 假设 24-token 格式
    if msgs.shape[1] >= 14:
        print(f"\n[Message Fields] (assuming 14+ columns):")
        field_names = ['order_id', 'event_type', 'direction', 'price_abs', 'price_rel',
                       'size', 'delta_t_s', 'delta_t_ns', 'time_s', 'time_ns',
                       'price_ref', 'size_ref', 'time_s_ref', 'time_ns_ref']

        for i, name in enumerate(field_names[:min(14, msgs.shape[1])]):
            col = msgs[:, i]
            print(f"  [{i}] {name:15s}: range=[{col.min():12.0f}, {col.max():12.0f}], "
                  f"unique={len(np.unique(col)):6d}")


def suggest_normalization(stats: dict):
    """基于统计数据建议 normalization 策略"""
    print("\n" + "=" * 70)
    print("4. Suggested Normalization Strategy")
    print("=" * 70)

    ch0_stats = stats.get('ch0', {})

    print(f"\n[Ch0 delta_mid_price]:")
    print(f"  Current range: [{ch0_stats.get('min', 'N/A'):.2f}, {ch0_stats.get('max', 'N/A'):.2f}]")
    print(f"  99th percentile: [{ch0_stats.get('p1', 'N/A'):.2f}, {ch0_stats.get('p99', 'N/A'):.2f}]")

    # 建议使用 99th percentile 来确定归一化范围
    p99_scale = max(abs(ch0_stats.get('p1', 100)), abs(ch0_stats.get('p99', 100)))
    suggested_divisor = np.ceil(p99_scale / 10) * 10  # Round up to nearest 10

    print(f"\n  Suggested options:")
    print(f"    Option A: Divide by {suggested_divisor:.0f} (based on 99th percentile)")
    print(f"              Result range: ~[{ch0_stats.get('p1', -100)/suggested_divisor:.2f}, {ch0_stats.get('p99', 100)/suggested_divisor:.2f}]")
    print(f"    Option B: Divide by 100 (fixed, simple)")
    print(f"              Result range: ~[-1, 1]")
    print(f"    Option C: Standard normalization (x - mean) / std")
    print(f"              Mean={ch0_stats.get('mean', 0):.4f}, Std={ch0_stats.get('std', 1):.4f}")


def main():
    # 数据目录
    data_dirs = [
        '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2016',
        '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2017',
        '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2018',
    ]

    # 找到第一个存在的目录
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d):
            data_dir = d
            break

    if data_dir is None:
        print("Error: No data directory found!")
        return

    print(f"Data directory: {data_dir}")
    print(f"Files: {len([f for f in os.listdir(data_dir) if f.endswith('.npy')])}")

    # 1. 分析原始数据
    raw_stats = analyze_orderbook_raw(data_dir, num_files=10)

    # 2. 分析转换后数据
    transformed_stats = analyze_orderbook_transformed(data_dir, num_files=3, book_depth=500)

    # 3. 分析 Message 数据
    analyze_message_tokens(data_dir, num_files=5)

    # 4. 建议 normalization 策略
    suggest_normalization(transformed_stats)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
