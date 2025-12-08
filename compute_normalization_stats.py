#!/usr/bin/env python3
"""
使用 online/streaming 方式计算 normalization 统计量
直接读取已经 encoded/transformed 的数据，避免重复计算
"""

import numpy as np
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class OnlineQuantileEstimator:
    """在线计算分位数 (使用 reservoir sampling)"""
    def __init__(self, quantiles=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]):
        self.quantiles = quantiles
        self.values = []
        self.max_samples = 1000000  # 最多保留 100万个样本

    def update(self, data):
        """更新样本 (reservoir sampling)"""
        data_flat = data.flatten()
        if len(self.values) < self.max_samples:
            add_n = min(len(data_flat), self.max_samples - len(self.values))
            self.values.extend(data_flat[:add_n].tolist())
        else:
            # 随机替换
            n_new = min(len(data_flat), 10000)
            indices = np.random.choice(len(self.values), size=n_new, replace=False)
            new_samples = np.random.choice(data_flat, size=n_new, replace=False)
            for idx, val in zip(indices, new_samples):
                self.values[idx] = float(val)

    def get_quantiles(self):
        """计算分位数"""
        arr = np.array(self.values)
        return {q: float(np.percentile(arr, q * 100)) for q in self.quantiles}


class OnlineStats:
    """在线计算均值、方差、min、max (批量更新优化版)"""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, data):
        """批量更新 (优化版 Welford)"""
        data_flat = data.flatten()
        n_new = len(data_flat)

        # 更新 min/max
        self.min_val = min(self.min_val, float(data_flat.min()))
        self.max_val = max(self.max_val, float(data_flat.max()))

        # 批量 Welford 更新
        new_mean = float(data_flat.mean())
        new_var = float(data_flat.var())

        if self.n == 0:
            self.n = n_new
            self.mean = new_mean
            self.M2 = new_var * n_new
        else:
            # 合并两个统计量
            n_old = self.n
            n_total = n_old + n_new

            # 合并均值
            delta = new_mean - self.mean
            self.mean = (n_old * self.mean + n_new * new_mean) / n_total

            # 合并方差 (parallel variance formula)
            self.M2 = self.M2 + new_var * n_new + delta**2 * n_old * n_new / n_total
            self.n = n_total

    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)


def process_single_file(args):
    """处理单个文件 (用于多进程)"""
    filepath, data_dir = args
    book = np.load(os.path.join(data_dir, filepath))

    if book.shape[1] != 503:
        return None, None, None, None

    ch0 = book[:, 0]
    vol = book[:, 3:].flatten()
    vol_nonzero = vol[vol != 0]

    # 返回原始数据供主进程聚合
    return ch0, vol_nonzero, filepath, book.shape


def compute_streaming_stats(data_dirs: list, max_files_per_dir: int = None, n_workers: int = None):
    """直接从 encoded 数据计算统计量 (多年份，多进程加速)"""
    print("=" * 80)
    print("Computing Statistics from Encoded Data (All Years, Parallel!)")
    print("=" * 80)

    # 收集所有年份的文件
    all_file_paths = []
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"  Skipping (not found): {data_dir}")
            continue
        year_files = sorted([f for f in os.listdir(data_dir) if 'orderbook' in f and f.endswith('.npy')])
        if max_files_per_dir:
            year_files = year_files[:max_files_per_dir]
        all_file_paths.extend([(f, data_dir) for f in year_files])
        print(f"  {data_dir}: {len(year_files)} files")

    if n_workers is None:
        n_workers = min(cpu_count(), 32)  # 最多用32核

    print(f"\nTotal files: {len(all_file_paths)}")
    print(f"Using: {n_workers} worker processes")
    print(f"Expected shape: (N, 503) for transformed data\n")

    # 初始化统计器
    ch0_stats = OnlineStats()
    ch0_quantile = OnlineQuantileEstimator()
    vol_stats = OnlineStats()
    vol_quantile = OnlineQuantileEstimator()

    # 多进程处理
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, all_file_paths),
            total=len(all_file_paths),
            desc="Processing",
            unit="file"
        ))

    # 聚合结果
    print(f"\nAggregating results...")
    for ch0, vol_nonzero, filename, shape in tqdm(results, desc="Aggregating"):
        if ch0 is None:
            print(f"  Skipped: {filename} (shape={shape})")
            continue

        ch0_stats.update(ch0)
        ch0_quantile.update(ch0)

        if vol_nonzero is not None and len(vol_nonzero) > 0:
            vol_stats.update(vol_nonzero)
            vol_quantile.update(vol_nonzero)

    print(f"\nDone! Processed {len([r for r in results if r[0] is not None])} files")

    return ch0_stats, ch0_quantile, vol_stats, vol_quantile


def print_statistics(name, stats, quantile_est):
    """打印统计结果"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    print(f"\nBasic Statistics:")
    print(f"  Count:  {stats.n:,}")
    print(f"  Mean:   {stats.mean:.6f}")
    print(f"  Std:    {stats.std:.6f}")
    print(f"  Min:    {stats.min_val:.6f}")
    print(f"  Max:    {stats.max_val:.6f}")

    print(f"\nDetailed Percentiles:")
    quantiles = quantile_est.get_quantiles()
    for q in sorted(quantiles.keys()):
        val = quantiles[q]
        print(f"  {q*100:6.1f}%: {val:12.6f}")

    return quantiles


def suggest_normalization(ch0_quantiles, vol_quantiles):
    """基于 quantile 建议归一化参数"""
    print(f"\n{'='*80}")
    print("NORMALIZATION RECOMMENDATIONS")
    print(f"{'='*80}")

    print("\n[Ch0] delta_mid_price:")
    print("  Current: NOT normalized")
    print("  Goal: normalize to [-1, 1] range\n")

    for q_val, q_name in [(0.99, '99th'), (0.995, '99.5th'), (0.999, '99.9th')]:
        q_low_key = (1 - q_val) / 2
        q_high_key = (1 + q_val) / 2
        # Find closest quantile key (handle floating point precision)
        q_low = min(ch0_quantiles.items(), key=lambda x: abs(x[0] - q_low_key))[1]
        q_high = min(ch0_quantiles.items(), key=lambda x: abs(x[0] - q_high_key))[1]
        divisor = max(abs(q_low), abs(q_high))

        print(f"  {q_name} percentile (clips {(1-q_val)*100:.2f}% outliers):")
        print(f"    Data range:  [{q_low:.2f}, {q_high:.2f}]")
        print(f"    Divisor:     {divisor:.1f}")
        print(f"    Normalized:  [{q_low/divisor:.3f}, {q_high/divisor:.3f}]")
        print()

    print("\n[Vol] volume image:")
    print("  Current: divide by 1000")
    print("  Goal: normalize to [-1, 1] range\n")

    for q_val, q_name in [(0.99, '99th'), (0.995, '99.5th'), (0.999, '99.9th')]:
        q_low_key = (1 - q_val) / 2
        q_high_key = (1 + q_val) / 2
        # Find closest quantile key (handle floating point precision)
        q_low = min(vol_quantiles.items(), key=lambda x: abs(x[0] - q_low_key))[1]
        q_high = min(vol_quantiles.items(), key=lambda x: abs(x[0] - q_high_key))[1]
        divisor = max(abs(q_low), abs(q_high))

        print(f"  {q_name} percentile (clips {(1-q_val)*100:.2f}% outliers, non-zero only):")
        print(f"    Data range:  [{q_low:.2f}, {q_high:.2f}]")
        print(f"    Divisor:     {divisor:.1f}")
        print(f"    Normalized:  [{q_low/divisor:.3f}, {q_high/divisor:.3f}]")
        print()

    # Final recommendation (99th percentile)
    print(f"\n{'='*80}")
    print("RECOMMENDED: 99th percentile normalization")
    print(f"{'='*80}")

    # Use closest keys for final recommendation
    ch0_q005 = min(ch0_quantiles.items(), key=lambda x: abs(x[0] - 0.005))[1]
    ch0_q995 = min(ch0_quantiles.items(), key=lambda x: abs(x[0] - 0.995))[1]
    ch0_divisor = max(abs(ch0_q005), abs(ch0_q995))

    vol_q005 = min(vol_quantiles.items(), key=lambda x: abs(x[0] - 0.005))[1]
    vol_q995 = min(vol_quantiles.items(), key=lambda x: abs(x[0] - 0.995))[1]
    vol_divisor = max(abs(vol_q005), abs(vol_q995))

    print(f"\nCode changes needed in preproc.py:")
    print(f"\n  File: preproc.py")
    print(f"  Function: transform_L2_state_numpy()")
    print(f"\n  # Add after line 183 (after time normalization):")
    print(f"  delta_p_mid_and_time[0] = delta_p_mid_and_time[0] / {ch0_divisor:.1f}")
    print(f"\n  # Change line 190 (volume normalization):")
    print(f"  # OLD: mybook.astype(np.float32) / 1000")
    print(f"  # NEW: mybook.astype(np.float32) / {vol_divisor:.1f}")
    print()


def main():
    # Use encoded data from ALL years (2016-2021)
    base_dir = '/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOG'

    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} not found!")
        return

    # Find all year directories
    year_dirs = []
    for year in range(2016, 2022):  # 2016-2021
        year_dir = os.path.join(base_dir, str(year))
        if os.path.exists(year_dir):
            year_dirs.append(year_dir)

    if not year_dirs:
        print("No year directories found!")
        return

    print(f"Base directory: {base_dir}")
    print(f"Years found: {[os.path.basename(d) for d in year_dirs]}\n")

    # 1. 每个年份单独分析
    year_results = {}
    for year_dir in year_dirs:
        year = os.path.basename(year_dir)
        print(f"\n{'='*80}")
        print(f"Analyzing Year: {year}")
        print(f"{'='*80}")

        ch0_stats, ch0_q, vol_stats, vol_q = compute_streaming_stats(
            [year_dir],
            max_files_per_dir=None,
            n_workers=None
        )

        year_results[year] = {
            'ch0_stats': ch0_stats,
            'ch0_quantiles': ch0_q.get_quantiles(),
            'vol_stats': vol_stats,
            'vol_quantiles': vol_q.get_quantiles(),
        }

        # 打印该年份的简要统计
        print(f"\n  {year} Summary:")
        print(f"    Ch0 99th: [{year_results[year]['ch0_quantiles'][0.005]:.2f}, {year_results[year]['ch0_quantiles'][0.995]:.2f}]")
        print(f"    Vol 99th: [{year_results[year]['vol_quantiles'][0.005]:.2f}, {year_results[year]['vol_quantiles'][0.995]:.2f}]")

    # 2. 所有年份合并统计
    print(f"\n\n{'='*80}")
    print(f"Analyzing ALL YEARS Combined (2016-2021)")
    print(f"{'='*80}")

    ch0_all, ch0_q_all, vol_all, vol_q_all = compute_streaming_stats(
        year_dirs,
        max_files_per_dir=None,
        n_workers=None
    )

    # Print detailed results
    ch0_q = print_statistics("[Ch0] delta_mid_price (NOT normalized) - ALL YEARS", ch0_all, ch0_q_all)
    vol_q = print_statistics("[Vol] volume image (after /1000) - ALL YEARS", vol_all, vol_q_all)

    # Suggest normalization based on ALL years
    suggest_normalization(ch0_q, vol_q)

    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"Output saved to: logs/data_analysis_<jobid>.out")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
