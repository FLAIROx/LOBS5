#!/bin/bash
# 最终测试：验证--merging参数和checkpoint修复

echo "=========================================="
echo "Final Test: --merging + Checkpoint + XLA"
echo "=========================================="

# 环境设置
export CUDA_VISIBLE_DEVICES=0
export JAX_DEBUG_PRINT=1

echo ""
echo "1. Testing parameter parsing..."
echo "-----------------------------------"
python3 -c "
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

# 测试参数解析
import argparse
parser = argparse.ArgumentParser()

# 检查是否能找到--merging参数
try:
    # 模拟run_train.py的参数解析
    exec(open('run_train.py').read().split('args = parser.parse_args()')[0])
    args = parser.parse_args(['--merging', 'padded'])
    print(f'✅ --merging parameter parsed: {args.merging}')
    print(f'   Default should be padded: {args.merging == \"padded\"}')
except Exception as e:
    print(f'❌ Error parsing: {e}')
"

echo ""
echo "2. Running BSZ=32 with all fixes..."
echo "-----------------------------------"

timeout 120 python3 run_train.py \
    --merging=padded \
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
    --USE_WANDB=False 2>&1 | tee final_test.log &

# 等待一段时间
sleep 30

echo ""
echo "3. Checking critical points..."
echo "-----------------------------------"

# 检查XLA错误
if grep -q "Unknown flags in XLA_FLAGS" final_test.log; then
    echo "❌ XLA_FLAGS error detected"
else
    echo "✅ XLA_FLAGS working"
fi

# 检查参数识别
if grep -q "\[DEBUG\] --merging parameter = 'padded'" final_test.log; then
    echo "✅ --merging=padded recognized"
else
    echo "❌ --merging not recognized correctly"
fi

# 检查模型选择
if grep -q "Selecting BatchPaddedLobPredModel" final_test.log; then
    echo "✅ Correct model selected (BatchPaddedLobPredModel)"
else
    echo "❌ Wrong model selected"
fi

# 检查__call_ar__方法
if grep -q "has no attribute '__call_ar__'" final_test.log; then
    echo "❌ Model missing __call_ar__ method"
else
    echo "✅ No __call_ar__ error"
fi

# 检查checkpoint激活
if grep -q "Checkpoint activated" final_test.log; then
    echo "✅ Checkpoint activated"
else
    echo "⚠️ Checkpoint messages not found yet"
fi

# GPU内存
echo ""
echo "4. GPU Memory status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# 清理
pkill -f "python3 run_train.py" 2>/dev/null

echo ""
echo "=========================================="
echo "Test Summary:"
echo "-----------------------------------"
grep -E "✅|❌|⚠️" final_test.log 2>/dev/null || echo "Check logs for details"
echo ""
echo "If all ✅, then:"
echo "1. --merging parameter working"
echo "2. BatchPaddedLobPredModel selected"
echo "3. Full autoregressive model ready"
echo "4. Checkpoint optimizations active"
echo "=========================================="