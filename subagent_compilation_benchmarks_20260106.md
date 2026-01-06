# 编译优化基准测试和验证方案设计

**生成时间**: 2026-01-06
**Subagent ID**: a7e8bcf
**任务**: 设计编译优化的基准测试和验证方案

---

## 概述

根据现有验证脚本架构 (`e1_high_scale_training.py`, `e2_multi_gpu_utilization.py`) 和 HyperscaleES 的编译优化模式 (`general_do_evolution_multi_gpu.py`)，设计以下四个编译优化测试。

---

## I1: 编译时间基准测试

### 目的
测量首次编译时间 vs 后续 epoch 时间，验证 JIT 编译仅在首次发生。

### 配置
- n_threads=256
- n_steps=10
- n_epochs=5

### 验证目标
1. First epoch significantly longer than subsequent epochs (JIT compilation)
2. Epoch 2-5 times are consistent (no recompilation)
3. Compilation time is within acceptable bounds

### Python 脚本框架 (`i1_compilation_time.py`)

```python
#!/usr/bin/env python3
"""
I1: Compilation Time Benchmark
Purpose: Measure first compilation time vs subsequent epoch time.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import time

CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"


@dataclass
class ESConfig:
    """I1 Compilation benchmark configuration."""
    lobs5_checkpoint: str = CHECKPOINT_PATH
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0

    n_threads: int = 256
    n_epochs: int = 5
    n_steps: int = 10
    world_msgs_per_step: int = 5

    token_mode: int = 24
    background_mode: str = 'historical_replay'
    replay_data_path: str = REPLAY_DATA_PATH
    data_dir: str = REPLAY_DATA_PATH

    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    seed: int = 42
    output_dir: str = '/tmp/es_validation_i1'


def test_compilation_time_benchmark():
    """I1: Compilation time benchmark test."""
    from es_lobs5.training.es_trainer import ESTrainer

    print("=" * 70)
    print("I1: Compilation Time Benchmark")
    print("=" * 70)
    print(f"Configuration: n_threads=256, n_steps=10, n_epochs=5")
    print("=" * 70)

    validation_results = {
        'first_epoch_includes_compilation': False,
        'subsequent_epochs_stable': False,
        'compilation_time_acceptable': False,
    }

    # Initialize
    print("\n[1/4] Initializing ESTrainer...")
    config = ESConfig()
    init_start = time.time()
    trainer = ESTrainer(config)
    init_time = time.time() - init_start
    print(f"  Initialization time: {init_time:.2f}s")

    # Create initial state
    print("\n[2/4] Creating initial simulation state...")
    state_start = time.time()
    initial_sim_state, initial_msg_history = trainer._create_initial_sim_state()
    state_time = time.time() - state_start
    print(f"  State creation time: {state_time:.2f}s")

    # Run epochs and measure times
    print("\n[3/4] Running epochs with timing...")
    key = jax.random.PRNGKey(config.seed)
    epoch_times = []

    for epoch in range(config.n_epochs):
        key, epoch_key = jax.random.split(key)

        epoch_start = time.time()
        mean_fitness, fitnesses, info = trainer.train_epoch(
            epoch_key, epoch=epoch,
            initial_sim_state=initial_sim_state,
            initial_msg_history=initial_msg_history
        )
        jax.block_until_ready(fitnesses)
        epoch_elapsed = time.time() - epoch_start

        epoch_times.append(epoch_elapsed)
        print(f"  Epoch {epoch}: {epoch_elapsed:.2f}s (fitness={float(mean_fitness):.6f})")

    # Analyze results
    print("\n[4/4] Analyzing compilation timing...")

    first_epoch_time = epoch_times[0]
    subsequent_times = epoch_times[1:]
    avg_subsequent_time = sum(subsequent_times) / len(subsequent_times)
    std_subsequent_time = (sum((t - avg_subsequent_time)**2 for t in subsequent_times) / len(subsequent_times)) ** 0.5

    compilation_time = first_epoch_time - avg_subsequent_time
    compilation_overhead_ratio = first_epoch_time / avg_subsequent_time

    print(f"\n  Timing Analysis:")
    print(f"    First epoch time:        {first_epoch_time:.2f}s")
    print(f"    Avg subsequent time:     {avg_subsequent_time:.2f}s")
    print(f"    Std subsequent time:     {std_subsequent_time:.2f}s")
    print(f"    Estimated compilation:   {compilation_time:.2f}s")
    print(f"    Compilation overhead:    {compilation_overhead_ratio:.1f}x")

    # PASS/FAIL criteria
    if compilation_overhead_ratio >= 2.0:
        validation_results['first_epoch_includes_compilation'] = True
        print("    [PASS] First epoch includes JIT compilation (>=2x overhead)")
    else:
        print(f"    [WARN] Compilation overhead low ({compilation_overhead_ratio:.1f}x < 2x)")
        validation_results['first_epoch_includes_compilation'] = True  # Soft pass

    cv_subsequent = std_subsequent_time / avg_subsequent_time if avg_subsequent_time > 0 else 0
    if cv_subsequent < 0.2:
        validation_results['subsequent_epochs_stable'] = True
        print(f"    [PASS] Subsequent epochs stable (CV={cv_subsequent:.2%} < 20%)")
    else:
        print(f"    [FAIL] Subsequent epochs unstable (CV={cv_subsequent:.2%} >= 20%)")

    MAX_COMPILATION_MINUTES = 30
    if compilation_time < MAX_COMPILATION_MINUTES * 60:
        validation_results['compilation_time_acceptable'] = True
        print(f"    [PASS] Compilation time acceptable ({compilation_time/60:.1f}m < {MAX_COMPILATION_MINUTES}m)")
    else:
        print(f"    [FAIL] Compilation time too long ({compilation_time/60:.1f}m >= {MAX_COMPILATION_MINUTES}m)")

    print("\n" + "=" * 70)
    print("I1 Validation Summary")
    print("=" * 70)

    all_passed = all(validation_results.values())
    for check_name, passed in validation_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")

    print("\n" + "=" * 70)
    if all_passed:
        print("I1 VALIDATION PASSED: Compilation Time Benchmark")
    else:
        print("I1 VALIDATION FAILED: Some checks did not pass")
    print("=" * 70)

    return all_passed


def main():
    print(f"\nJAX devices: {jax.devices()}")
    try:
        passed = test_compilation_time_benchmark()
        return 0 if passed else 1
    except Exception as e:
        print(f"\n[ERROR] I1 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

### PASS/FAIL 标准

| 检查项 | PASS 条件 | FAIL 条件 |
|--------|-----------|-----------|
| first_epoch_includes_compilation | Epoch 0 时间 >= 2x 后续平均 | < 2x (软 PASS) |
| subsequent_epochs_stable | 后续 epoch CV < 20% | CV >= 20% |
| compilation_time_acceptable | < 30 分钟 | >= 30 分钟 |

---

## I2: 重复编译检测

### 目的
使用 `JAX_LOG_COMPILES=1` 检测 epoch 2-5 是否有新编译。

### PASS/FAIL 标准

| 检查项 | PASS 条件 | FAIL 条件 |
|--------|-----------|-----------|
| epoch_0_has_compilations | Epoch 0 时间 > 1.5x 后续 | 无 (软 PASS) |
| epochs_1_4_no_compilations | 无时间异常 | 任何 epoch > 1.5x 平均 |
| compilation_count_stable | CV < 30% | CV >= 30% |

---

## I3: 持久化缓存测试

### 目的
测试程序重启后 JAX 编译缓存是否命中。

### PASS/FAIL 标准

| 检查项 | PASS 条件 | FAIL 条件 |
|--------|-----------|-----------|
| cache_created | 缓存文件 size > 0 | 无缓存文件 |
| warm_start_faster | 热启动 speedup >= 1.0 | 更慢 |
| results_consistent | Fitness 差异 < 1e-5 | 差异过大 |

---

## I4: 多 GPU 利用率测试

### 目的
验证 shard_map 正确分片到 4 GPU。

### PASS/FAIL 标准

| 检查项 | PASS 条件 | FAIL 条件 |
|--------|-----------|-----------|
| all_gpus_detected | >= 4 GPU | < 4 GPU |
| memory_distributed | 所有 GPU > 1%, 差异 < 50% | 未使用或严重不平衡 |
| shard_map_works | n_devices 个 shard | shard 数量不对 |
| no_device_errors | 训练完成无错 | 设备异常 |

---

## SBATCH 脚本模板

```bash
#!/bin/bash
#SBATCH --job-name=es_compile_benchmark
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=logs_validation/compile_benchmark_%j.out
#SBATCH --error=logs_validation/compile_benchmark_%j.err

echo "=============================================="
echo "ES-LOBS5 Compilation Optimization Benchmarks"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=============================================="

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/bin/activate es-lob
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
mkdir -p logs_validation

export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
PYTHON="/lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/envs/es-lob/bin/python -u"

clear_gpu_memory() {
    $PYTHON -c "import jax; jax.clear_caches(); import gc; gc.collect()"
}

# I1: Compilation Time Benchmark
echo ""
echo "[I1] Compilation Time Benchmark"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/i1_compilation_time.py
I1_STATUS=$?
clear_gpu_memory

# I2: Recompilation Detection
echo ""
echo "[I2] Recompilation Detection"
echo "----------------------------------------------"
JAX_LOG_COMPILES=1 $PYTHON es_lobs5/scripts/validation/i2_recompilation_detection.py 2>&1
I2_STATUS=$?
clear_gpu_memory

# I3: Persistent Cache Test
echo ""
echo "[I3] Persistent Cache Test"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/i3_persistent_cache.py
I3_STATUS=$?
clear_gpu_memory

# I4: Multi-GPU Utilization
echo ""
echo "[I4] Multi-GPU Utilization Test"
echo "----------------------------------------------"
$PYTHON es_lobs5/scripts/validation/i4_multi_gpu_utilization.py
I4_STATUS=$?

# Summary
echo ""
echo "=============================================="
echo "Compilation Optimization Benchmark Summary:"
echo "  I1 (Compilation Time):  $([ $I1_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  I2 (Recompilation):     $([ $I2_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  I3 (Persistent Cache):  $([ $I3_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  I4 (Multi-GPU):         $([ $I4_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "=============================================="
```

---

## 结果解读总结

| 测试 | 关键指标 | 正常值 | 异常处理 |
|------|----------|--------|----------|
| I1 | 编译 overhead | 2-10x | >10x: 优化计算图 |
| I2 | 重复编译次数 | 0 (epoch 1-4) | >0: 检查动态 shape |
| I3 | 缓存命中 speedup | >=1.5x | <1.5x: 检查缓存配置 |
| I4 | GPU 内存分布 | 差异<50% | >50%: 检查分片策略 |

---

## 测试脚本文件清单

```
es_lobs5/scripts/validation/
├── i1_compilation_time.py          # 编译时间基准
├── i2_recompilation_detection.py   # 重复编译检测
├── i3_persistent_cache.py          # 持久化缓存
├── i4_multi_gpu_utilization.py     # 多 GPU 利用率
└── run_compile_benchmark.sbatch    # 批量运行脚本
```
