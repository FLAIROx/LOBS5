#!/bin/bash
# 测试 LOBS5 从 pmap 迁移到 jax.jit + shardings 的脚本

echo "=========================================="
echo "LOBS5 Sharding Migration Test Script"
echo "=========================================="
echo ""

# 测试配置
TEST_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
TOKEN_MODE=22

echo "[Test 1] 单设备测试（验证基本功能）"
echo "----------------------------------------"
echo "目的：验证 jit + shardings 在单设备上正常工作"
echo ""

export CUDA_VISIBLE_DEVICES=0

python LOBS5/run_train.py \
    --num_devices=1 \
    --bsz=32 \
    --curtail_epochs=5 \
    --epochs=1 \
    --dir_name="$TEST_DIR" \
    --token_mode="$TOKEN_MODE" \
    --ssm_size_base=256 \
    --blocks=4 \
    --n_layers=4 \
    --USE_WANDB=False \
    --debug_loading=False \
    --enable_profiler=False

echo ""
echo "[Test 1] Complete!"
echo "检查点："
echo "  - 是否成功创建 mesh？"
echo "  - 是否成功 JIT 编译 train_step？"
echo "  - 训练是否正常运行？"
echo "  - Loss 是否下降？"
echo ""

echo "=========================================="
echo "[Test 2] 双设备测试（验证数据并行）"
echo "----------------------------------------"
echo "目的：验证多设备数据并行正常工作"
echo ""

export CUDA_VISIBLE_DEVICES=0,1

python LOBS5/run_train.py \
    --num_devices=2 \
    --bsz=64 \
    --curtail_epochs=5 \
    --epochs=1 \
    --dir_name="$TEST_DIR" \
    --token_mode="$TOKEN_MODE" \
    --ssm_size_base=256 \
    --blocks=4 \
    --n_layers=4 \
    --USE_WANDB=False \
    --debug_loading=False \
    --enable_profiler=False

echo ""
echo "[Test 2] Complete!"
echo "检查点："
echo "  - 两个设备是否都被使用？"
echo "  - 吞吐量是否接近 2x 单设备？"
echo "  - Loss 曲线是否合理？"
echo ""

echo "=========================================="
echo "[Test 3] 性能对比测试"
echo "----------------------------------------"
echo "目的：对比 pmap 和 jit+shardings 的性能"
echo ""
echo "注意：需要手动切换回 pmap 版本进行对比"
echo "      - 注释掉 train_helpers.py 中的新 train_step"
echo "      - 取消注释 pmap 版本的 train_step"
echo "      - 运行相同配置，对比吞吐量和内存"
echo ""

echo "=========================================="
echo "测试完成！"
echo ""
echo "预期结果："
echo "  ✓ 功能正确：Loss 下降，准确率提升"
echo "  ✓ 性能相当：吞吐量 ±10% 以内"
echo "  ✓ 内存优化：内存占用相似或更低（因为 donate_argnums）"
echo "=========================================="
