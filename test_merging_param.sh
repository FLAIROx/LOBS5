#!/bin/bash
# 测试--merging参数是否正确识别

echo "========================================"
echo "Testing --merging Parameter Recognition"
echo "========================================"

# 设置环境
export CUDA_VISIBLE_DEVICES=0
export JAX_DEBUG_PRINT=1

echo ""
echo "TEST 1: Verify --merging=padded is recognized"
echo "----------------------------------------"

# 只运行到模型初始化，不进行训练
timeout 30 python3 run_train.py \
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
    --USE_WANDB=False 2>&1 | tee test_merging_padded.log

echo ""
echo "Checking debug output for 'padded' mode:"
echo "----------------------------------------"
grep -E "\[DEBUG\]|BatchPaddedLobPredModel|BatchFullLobPredModel|merging" test_merging_padded.log | head -20

echo ""
echo "TEST 2: Check if model has __call_ar__ method"
echo "----------------------------------------"

# Check for the error about missing __call_ar__
if grep -q "__call_ar__" test_merging_padded.log; then
    if grep -q "has no attribute '__call_ar__'" test_merging_padded.log; then
        echo "❌ ERROR: Model still missing __call_ar__ method!"
        echo "   Wrong model is being used!"
    else
        echo "✅ Model has __call_ar__ method"
    fi
else
    echo "⚠️ Test may not have reached model initialization"
fi

echo ""
echo "TEST 3: Verify correct model selection"
echo "----------------------------------------"

if grep -q "Selecting BatchPaddedLobPredModel" test_merging_padded.log; then
    echo "✅ CORRECT: BatchPaddedLobPredModel is selected!"
elif grep -q "Selecting BatchFullLobPredModel" test_merging_padded.log; then
    echo "❌ WRONG: BatchFullLobPredModel is selected instead!"
else
    echo "⚠️ Model selection debug message not found"
fi

# 清理测试进程
pkill -f "python3 run_train.py" 2>/dev/null

echo ""
echo "========================================"
echo "Summary:"
echo "----------------------------------------"
echo "1. Check if --merging parameter is recognized"
echo "2. Check if BatchPaddedLobPredModel is selected"
echo "3. Check if __call_ar__ method is available"
echo ""
echo "If all tests pass, your full autoregressive model will work correctly!"
echo "========================================"