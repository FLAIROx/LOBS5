# ssm_stable 分支内存管理分析 (2026-02-18)

## 关键结论

ssm_stable **没有** del/重建 jit 函数的操作。它的 epoch 间内存管理完全依赖：
1. `gc.collect()` + `jax.clear_caches()` (每 epoch 末尾)
2. **`TF_GPU_ALLOCATOR=cuda_malloc_async`** (batch 脚本里) ← 这是关键！

## ssm_stable Epoch 末尾代码 (lob/train.py)

```python
# After each epoch
gc.collect()
# jax.clear_backends()  # 被注释掉
jax.clear_caches()
# jax.profiler.stop_trace()
if count > args.early_stop_patience:
    break
```

**没有** `del jit_train_step`，**没有** 重建 JIT 函数。

## ssm_stable JIT 函数生命周期

- 创建：epoch loop 之前，只创建一次
- 销毁：从不（整个训练复用）
- 唯一重建：Prodigy 切换时（一次性），与内存无关

## ssm_stable train_full_autoreg.batch 关键配置

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_GPU_ALLOCATOR=cuda_malloc_async  # added on Jan 4, 2026 — 关键！
```

## TF_GPU_ALLOCATOR=cuda_malloc_async 的作用

与默认的 BFC (Best-Fit with Coalescing) allocator 相比：
- BFC: 预分配大池，内部切分管理，容易碎片化
- cuda_malloc_async: 使用 CUDA 异步内存分配器，系统级内存管理，自动合并释放的块
- 效果：消除 train_step 和 eval_step 交替运行导致的内存碎片问题

## B1 迁移对比

B1 在 cf1cc67 commit 删除了部分 XLA flags，可能同时删掉了 TF_GPU_ALLOCATOR。
需要检查原始 B1 batch 脚本是否有这个设置，或者直接参照 ssm_stable 加回来。

## create_jit_train_step 关键参数

```python
jit_train_step = jax.jit(
    train_step,
    in_shardings=in_shardings,
    out_shardings=out_shardings,
    static_argnums=(5, 6, 7, 8),  # batchnorm, ignore_times, use_tbptt, n_tbptt_chunks
    donate_argnums=(0,),           # donate state for memory reuse
)
```

## 参考来源

- `git show ssm_stable:lob/train.py` (lines 855-857)
- `git show ssm_stable:train_full_autoreg.batch`
- `git show ssm_stable:lob/train_helpers.py` (lines 1776-1884)
