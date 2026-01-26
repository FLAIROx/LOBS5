"""
LOBMAX Performance Optimization Analysis
=========================================

Current Status: MFU ~1.8% (Very Low)

Root Causes Identified:
-----------------------

1. **Missing Gradient Synchronization**
   - Code removed pmean but didn't add proper cross-device gradient aggregation
   - In train_helpers.py lines 1266-1275: pmean removed but no replacement
   - Without explicit sync, gradients may not be properly averaged across devices

2. **Suboptimal Sharding Strategy**
   - Using P(None) for all params = fully replicated (no tensor parallelism)
   - Data parallel only, no FSDP or tensor parallelism
   - For 125M model on 4 GPUs, this should be fine but larger models will suffer

3. **No shard_map Usage**
   - Current code uses jax.jit with NamedSharding
   - shard_map offers more explicit control and better performance for some patterns

4. **DataLoader Bottleneck**
   - n_data_workers=12 may be too many
   - persistent_workers=True causes JAX CUDA init in workers (harmless but noisy)
   - prefetch_factor=6 may not be optimal

5. **Potential JIT Compilation Issues**
   - Large static_argnums may cause excessive recompilation
   - Debug buffer aliasing check runs every time

6. **XLA Optimization Flags Missing**
   - No XLA fusion flags
   - No NCCL optimization flags

Recommended Optimizations (Priority Order):
===========================================

PRIORITY 1: Fix Gradient Synchronization
-----------------------------------------
Add explicit gradient averaging in train_step after computing gradients:

```python
# After: grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
# Add:
grads = jax.lax.pmean(grads, axis_name='data')  # Need axis_name from mesh
loss = jax.lax.pmean(loss, axis_name='data')
```

Or use jax.experimental.multihost_utils for multi-node:
```python
grads = jax.tree_map(lambda x: jax.lax.psum(x, 'data') / jax.device_count(), grads)
```

PRIORITY 2: Add XLA Optimization Flags
--------------------------------------
Add to train_lobmax.batch:

```bash
export XLA_FLAGS="${XLA_FLAGS} \
  --xla_gpu_enable_cudnn_fmha=true \
  --xla_gpu_enable_triton_gemm=false \
  --xla_gpu_graph_level=0 \
  --xla_gpu_enable_async_collectives=true \
  --xla_gpu_enable_latency_hiding_scheduler=true \
  --xla_gpu_enable_highest_priority_async_stream=true"
```

PRIORITY 3: Reduce DataLoader Overhead
--------------------------------------
Reduce n_data_workers to 4 (1 per GPU)
Set CUDA_VISIBLE_DEVICES=-1 for workers properly

PRIORITY 4: Use shard_map for Train Step
----------------------------------------
Convert train_step to use shard_map for better sharding control:

```python
from jax.experimental.shard_map import shard_map

train_step_sharded = jax.jit(shard_map(
    train_step_inner,
    mesh=mesh,
    in_specs=(P(), P('data'), P('data'), P('data')),  # state replicated, data sharded
    out_specs=P(),  # output replicated
    check_rep=False,
))
```

PRIORITY 5: Increase Batch Size
-------------------------------
Current: 4 per GPU = 16 global
Recommended: 8-16 per GPU = 32-64 global
Larger batches improve MFU by better utilizing matrix cores

PRIORITY 6: Profile with JAX Profiler
-------------------------------------
Enable profiling to identify actual bottlenecks:
```python
with jax.profiler.trace("/tmp/jax-trace"):
    for step in range(10):
        state = train_step(state, ...)
```
"""

# Quick fix script for MFU optimization
import os
import sys

def apply_quick_fixes():
    """Apply quick fixes to improve MFU."""
    
    # 1. Set XLA flags
    xla_flags = [
        "--xla_gpu_enable_cudnn_fmha=true",
        "--xla_gpu_enable_async_collectives=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
    ]
    
    existing_flags = os.environ.get("XLA_FLAGS", "")
    new_flags = existing_flags + " " + " ".join(xla_flags)
    os.environ["XLA_FLAGS"] = new_flags.strip()
    
    print(f"[Optimization] XLA_FLAGS set: {os.environ['XLA_FLAGS']}")
    
    # 2. Set NCCL environment
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand if available
    
    print("[Optimization] NCCL environment configured")
    
    return True


if __name__ == "__main__":
    apply_quick_fixes()
    print("\nOptimizations applied. Key recommendations:")
    print("1. Add gradient sync (jax.lax.pmean) to train_step")
    print("2. Increase batch size to 8-16 per GPU")
    print("3. Reduce n_data_workers to 4")
    print("4. Consider using shard_map for better performance")
