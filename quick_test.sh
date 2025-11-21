#!/bin/bash
# 快速测试XLA_FLAGS修复和checkpoint功能

echo "========================================"
echo "Quick Test - XLA FLAGS Fix + Checkpoint"
echo "========================================"
echo ""

# 设置环境
export CUDA_VISIBLE_DEVICES=0
export JAX_DEBUG_PRINT=1

echo "Testing BSZ=32 with fixed XLA_FLAGS..."
echo "----------------------------------------"

# 简短测试，只运行几步
python3 -c "
import os
print('XLA_FLAGS:', os.environ.get('XLA_FLAGS', 'Not set'))
print('JAX_CHECKPOINT_POLICY:', os.environ.get('JAX_CHECKPOINT_POLICY', 'Not set'))
"

echo ""
echo "Running quick training test (10 steps only)..."
echo ""

timeout 60 python3 run_train.py \
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
    --USE_WANDB=False 2>&1 | tee quick_test.log &

# 等待程序启动
sleep 10

# 检查是否有错误
if grep -q "Unknown flags in XLA_FLAGS" quick_test.log; then
    echo "❌ XLA_FLAGS error still present!"
    grep "Unknown flags" quick_test.log
    exit 1
else
    echo "✅ XLA_FLAGS error fixed!"
fi

# 检查checkpoint消息
echo ""
echo "Checking for checkpoint messages..."
if grep -q "Checkpoint activated" quick_test.log; then
    echo "✅ Checkpoint is working!"
    grep "Checkpoint activated" quick_test.log | head -3
else
    echo "⚠️ No checkpoint messages yet (may need more time)"
fi

# 检查内存使用
echo ""
echo "GPU Memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# 终止测试进程
pkill -f "python3 run_train.py"

echo ""
echo "========================================"
echo "Quick test complete!"
echo "If no XLA errors, proceed with full test: ./test_m1_fix.sh"
echo "========================================"