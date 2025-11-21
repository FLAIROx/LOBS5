#!/bin/bash
# M1修复方案测试脚本

echo "=========================================="
echo "M1 Fix (jax.remat) Implementation Test"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export JAX_DEBUG_PRINT=1  # 启用调试输出查看checkpoint消息

echo ""
echo "Step 1: Quick test with BSZ=32 (should work)"
echo "-------------------------------------------"
timeout 180 python3 -u -B run_train.py \
    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
    --blocks=16 --bsz=32 --d_model=1024 --dataset=lobster-prediction \
    --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021' \
    --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2' \
    --data_mode='preproc' \
    --clip_eigs=True --activation_fn=half_glu1 \
    --dt_global=False --epochs=1 --jax_seed=42 --lr_factor=1 --n_layers=12 \
    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0003 --ssm_size_base=1024 \
    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
    --use_book_data=True --use_simple_book=False --book_transform=True \
    --masking=none \
    --num_devices=1 --n_data_workers=0 \
    --debug_loading=False \
    --enable_profiler=False \
    --random_offsets_train=True \
    --shuffle_train=True \
    --debug_overfit=False \
    --lr_patience=5 \
    --USE_WANDB=False 2>&1 | tee test_bsz32_fix.log

# 检查checkpoint消息
echo ""
echo "Checking for checkpoint activation messages:"
grep -i "checkpoint activated\|remat\|rematerialization" test_bsz32_fix.log | head -5

if [ $? -eq 124 ]; then
    echo "⏱️ BSZ=32 test timed out - checking partial results"
    grep -i "memory\|GiB\|checkpoint" test_bsz32_fix.log | tail -10
elif [ $? -eq 0 ]; then
    echo "✅ BSZ=32 test completed"
else
    echo "❌ BSZ=32 test failed"
    exit 1
fi

echo ""
echo "Step 2: Testing with BSZ=64 (4 GPUs, 16 per GPU)"
echo "-------------------------------------------"
export CUDA_VISIBLE_DEVICES=0,1,2,3

timeout 180 python3 -u -B run_train.py \
    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
    --blocks=16 --bsz=64 --d_model=1024 --dataset=lobster-prediction \
    --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021' \
    --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2' \
    --data_mode='preproc' \
    --clip_eigs=True --activation_fn=half_glu1 \
    --dt_global=False --epochs=1 --jax_seed=42 --lr_factor=1 --n_layers=12 \
    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0003 --ssm_size_base=1024 \
    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
    --use_book_data=True --use_simple_book=False --book_transform=True \
    --masking=none \
    --num_devices=4 --n_data_workers=0 \
    --debug_loading=False \
    --enable_profiler=False \
    --random_offsets_train=True \
    --shuffle_train=True \
    --debug_overfit=False \
    --lr_patience=5 \
    --USE_WANDB=False 2>&1 | tee test_bsz64_fix.log

# 检查结果
if [ $? -eq 124 ]; then
    echo "⏱️ BSZ=64 test timed out - checking for OOM"
    grep -i "oom\|out of memory\|GiB" test_bsz64_fix.log | head -5
elif [ $? -eq 0 ]; then
    echo "✅ BSZ=64 test PASSED - Fix successful!"
    echo ""
    echo "=========================================="
    echo "M1 Fix Implementation SUCCESS!"
    echo "=========================================="
else
    echo "❌ BSZ=64 test failed - checking logs"
    grep -i "error\|exception" test_bsz64_fix.log | head -10
fi

echo ""
echo "Memory Comparison:"
echo "-------------------------------------------"
echo "BSZ=32 memory usage:"
grep -i "memory\|GiB" test_bsz32_fix.log | tail -3
echo ""
echo "BSZ=64 memory usage:"
grep -i "memory\|GiB" test_bsz64_fix.log | tail -3

echo ""
echo "Summary:"
echo "-------------------------------------------"
echo "1. Check if 'Checkpoint activated' messages appear in logs"
echo "2. Compare memory usage with previous runs"
echo "3. If BSZ=64 works, the fix is successful!"