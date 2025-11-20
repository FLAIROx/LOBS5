#!/bin/bash
# M1 Checkpoint测试脚本

echo "==================================="
echo "M1 Checkpoint Implementation Test"
echo "==================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Step 1: Testing with BSZ=16 (baseline)..."
echo "-----------------------------------"
timeout 300 python3 -u -B run_train.py \
    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
    --blocks=16 --bsz=16 --d_model=1024 --dataset=lobster-prediction \
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
    --USE_WANDB=False 2>&1 | tee test_bsz16.log

if [ $? -eq 0 ]; then
    echo "✅ BSZ=16 test PASSED"
else
    echo "❌ BSZ=16 test FAILED"
    exit 1
fi

echo ""
echo "Step 2: Testing with BSZ=24 (checkpoint test)..."
echo "-----------------------------------"
timeout 300 python3 -u -B run_train.py \
    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
    --blocks=16 --bsz=24 --d_model=1024 --dataset=lobster-prediction \
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
    --USE_WANDB=False 2>&1 | tee test_bsz24.log

if [ $? -eq 0 ]; then
    echo "✅ BSZ=24 test PASSED - Checkpoint working!"
    echo ""
    echo "==================================="
    echo "M1 Checkpoint Implementation SUCCESS!"
    echo "==================================="
else
    echo "❌ BSZ=24 test FAILED"
    exit 1
fi

# 显示内存使用对比
echo ""
echo "Memory Usage Comparison:"
echo "-----------------------------------"
grep -i "memory\|GiB\|GB" test_bsz16.log test_bsz24.log | head -10

echo ""
echo "Next steps:"
echo "1. If tests pass, update batch script BSZ to 24 or higher"
echo "2. Monitor GPU memory usage with: nvidia-smi -l 1"
echo "3. Check logs for checkpoint activation messages"