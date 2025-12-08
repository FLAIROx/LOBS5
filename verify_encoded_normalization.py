#!/usr/bin/env python3
"""
验证 encoded 数据的归一化是否正确
"""
import numpy as np
import glob
import sys
from pathlib import Path

def verify_directory(data_dir):
    """验证目录中的 encoded 数据"""
    print("="*70)
    print(f"检查目录: {data_dir}")
    print("="*70)

    # 查找 orderbook 文件
    book_files = sorted(glob.glob(str(data_dir) + '/**/*book*.npy', recursive=True))

    if not book_files:
        print(f"❌ 未找到 orderbook 文件")
        return False

    print(f"找到 {len(book_files)} 个 orderbook 文件")

    # 检查前5个文件
    print(f"\n检查前 5 个文件:")
    all_normalized = True

    for i, file_path in enumerate(book_files[:5]):
        book = np.load(file_path, mmap_mode='r')

        time_sec_min = book[:, 1].min()
        time_sec_max = book[:, 1].max()
        time_nano_min = book[:, 2].min()
        time_nano_max = book[:, 2].max()

        is_normalized = (time_sec_max <= 1.0) and (time_nano_max <= 1.0)

        fname = Path(file_path).name
        print(f"\n  [{i+1}] {fname}")
        print(f"      time_sec:  [{time_sec_min:.6f}, {time_sec_max:.6f}]")
        print(f"      time_nano: [{time_nano_min:.6f}, {time_nano_max:.6f}]")
        print(f"      状态: {'✅ 已归一化' if is_normalized else '❌ 未归一化'}")

        if not is_normalized:
            all_normalized = False

    print("\n" + "="*70)
    if all_normalized:
        print("✅ 所有检查的文件都已正确归一化")
    else:
        print("❌ 部分文件未归一化")
    print("="*70)

    return all_normalized

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # 默认检查新的 _dec 目录
        data_dir = Path("/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/GOOG")

    if not data_dir.exists():
        print(f"❌ 目录不存在: {data_dir}")
        print("\n用法:")
        print(f"  python {sys.argv[0]} <data_directory>")
        print("\n示例:")
        print(f"  python {sys.argv[0]} /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/GOOG")
        sys.exit(1)

    success = verify_directory(data_dir)
    sys.exit(0 if success else 1)
