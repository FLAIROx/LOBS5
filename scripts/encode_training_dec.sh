#!/bin/bash
# 编码训练数据 (使用修复后的归一化，输出到 _dec 目录)
# 用法: bash encode_training_dec.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# ✅ 强制使用 CPU，避免 GPU OOM (JAX 会在导入时初始化 GPU)
export JAX_PLATFORM_NAME=cpu

SCRIPT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/pre_encode_data.py"

echo "======================================================================="
echo "Encoding Training Data with Fixed Normalization"
echo "======================================================================="
echo "Input:  /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/"
echo "Output: /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/"
echo "Suffix: _proc.npy → _encoded.npy"
echo "======================================================================="

python "$SCRIPT" \
    --input_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/ \
    --output_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/ \
    --num_workers=8

echo ""
echo "======================================================================="
echo "✓ Training Data Encoding DONE"
echo "======================================================================="
echo ""
echo "Verify normalization:"
echo "  python3 -c \""
echo "  import numpy as np, glob"
echo "  files = sorted(glob.glob('/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/GOOG/**/*book*.npy', recursive=True))"
echo "  book = np.load(files[0], mmap_mode='r')"
echo "  print(f'time_sec range: [{book[:, 1].min():.4f}, {book[:, 1].max():.4f}]')"
echo "  print(f'time_nano range: [{book[:, 2].min():.6f}, {book[:, 2].max():.6f}]')"
echo "  print('✅ Normalized!' if book[:, 1].max() <= 1.0 else '❌ Not normalized')"
echo "  \""
