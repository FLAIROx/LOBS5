#!/bin/bash
# 编码 GOOG 2018 数据 (使用 numpy 版本转换，已验证正确)
# 用法: bash encode_GOOG_2018_dec.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# ✅ 强制使用 CPU，避免 GPU OOM
export JAX_PLATFORM_NAME=cpu

SCRIPT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/pre_encode_data.py"

echo "======================================================================="
echo "Encoding GOOG 2018 with Fixed Normalization"
echo "======================================================================="
echo "Input:  /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2018/"
echo "Output: /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/GOOG/2018/"
echo "Suffix: _proc.npy → _encoded.npy"
echo "======================================================================="

# 清理旧输出（如果存在）
echo "Cleaning old output (if exists)..."
rm -rf /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/GOOG/2018/

# 创建输出目录
mkdir -p /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/GOOG/2018/

python "$SCRIPT" \
    --input_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2018/ \
    --output_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded_dec/GOOG/2018/ \
    --num_workers=50

echo ""
echo "======================================================================="
echo "✓ GOOG 2018 Encoding DONE"
echo "======================================================================="
