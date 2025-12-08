#!/bin/bash
# 编码 JAN 2023 测试数据 (使用 numpy 版本转换，已验证正确)
# 用法: bash encode_JAN2023_dec.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# ✅ 强制使用 CPU，避免 GPU OOM
export JAX_PLATFORM_NAME=cpu

SCRIPT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/pre_encode_data.py"

echo "======================================================================="
echo "Encoding JAN 2023 Test Data with Fixed Normalization"
echo "======================================================================="
echo "Input:  /lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc/"
echo "Output: /lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_encoded_dec/"
echo "Suffix: _proc.npy → _encoded.npy"
echo "======================================================================="

# 清理旧输出（如果存在）
echo "Cleaning old output (if exists)..."
rm -rf /lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_encoded_dec/

# 创建输出目录
mkdir -p /lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_encoded_dec/

python "$SCRIPT" \
    --input_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc/ \
    --output_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_encoded_dec/ \
    --num_workers=18

echo ""
echo "======================================================================="
echo "✓ JAN 2023 Test Data Encoding DONE"
echo "======================================================================="
echo ""
echo "验证编码结果:"
python3 << 'EOF'
import numpy as np
import glob

files = sorted(glob.glob('/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_encoded_dec/*book*_encoded.npy'))
if files:
    book = np.load(files[0], mmap_mode='r')
    print(f"  文件数: {len(files)}")
    print(f"  样本文件: {files[0].split('/')[-1]}")
    print(f"  Shape: {book.shape}")
    print(f"  time_sec  范围: [{book[:, 1].min():.6f}, {book[:, 1].max():.6f}]")
    print(f"  time_nano 范围: [{book[:, 2].min():.6f}, {book[:, 2].max():.6f}]")
    if book[:, 1].max() <= 1.0 and book[:, 2].max() <= 1.0:
        print("  ✅ 归一化成功!")
    else:
        print("  ❌ 归一化失败")
else:
    print("  ⚠️  未找到编码文件")
EOF
