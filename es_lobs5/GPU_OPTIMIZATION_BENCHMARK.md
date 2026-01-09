# ES Training GPU Optimization Benchmark Results

## Overview

This document tracks performance benchmarks for each optimization applied to the ES training forward pass.

**Benchmark Configuration:**
- N_EPOCHS: 10
- N_PERTURBATIONS: 128
- N_STEPS: 100
- N_WARMUP_MSGS: 500
- BACKGROUND_MSGS_PER_STEP: 10
- GPUs: 4
- FILE_IDX: 0 (fixed for reproducibility)

---

## ⚠️ IMPORTANT: Testing Condition Clarification

**Previous tests had confounded variables:**

| Test | freeze_nonlora | Parameters Trained | Result |
|------|----------------|-------------------|--------|
| Original Baseline (31d667d) | **False** | 360M (all) | 87s/epoch |
| Opt 1 Test (4e30831) | **True** | LORA only (~1M) | 6s/epoch |

The 93% improvement was **NOT** from batched loading, but from `freeze_nonlora=True` (only training LORA parameters).

**All future tests must use `freeze_nonlora=True` for valid comparison.**

---

## Baseline (Pre-optimization, freeze_nonlora=False) - DEPRECATED

**Date:** 2026-01-09
**Commit:** 31d667d
**Job ID:** 1865747
**⚠️ Status:** DEPRECATED - Uses freeze_nonlora=False (360M params)

| Metric | Value |
|--------|-------|
| Total Time (10 epochs) | 1696.5 seconds |
| Per-Epoch Time (avg) | 169.64 seconds |
| First Epoch (incl. XLA compile) | 573 seconds |
| Avg Epoch (epochs 2-10) | ~90 seconds |

---

## NEW Baseline (freeze_nonlora=True)

**Date:** 2026-01-09
**Commit:** 6937fe8 (revert batched loading)
**Job ID:** 1865796
**Status:** Running...

| Metric | Value |
|--------|-------|
| Total Time (10 epochs) | TBD |
| Per-Epoch Time (avg) | TBD |
| First Epoch (incl. XLA compile) | TBD |
| Avg Epoch (epochs 2-10) | TBD |

**Notes:**
- `freeze_nonlora=True` (LORA-only training)
- This is the correct baseline for optimization comparison

---

## Optimization 1: Historical Replay Batched Loading

**Status:** ❌ REJECTED
**Job ID:** 1865815
**Expected Impact:** 5-10%
**Actual Impact:** **No improvement (possibly negative)**

**Rationale:**
- Convert dynamic indexing `replay_tokens[replay_ptr]` to static indexing `batch[i]`
- XLA can better optimize static index patterns

**Test Results:**
| Metric | Baseline | Batched Loading | Change |
|--------|----------|-----------------|--------|
| Total Time | 596.3s | 561.8s | -5.8% (noise) |
| XLA Compile | 252s | **284s** | **+12.7% worse** |
| Training Loop | 304s | **334s** | **+10% worse** |

**Conclusion:**
- XLA already optimizes the original dynamic indexing well
- Pre-batching adds overhead (closure capture, dynamic_slice)
- **Optimization rejected** - no real benefit

---

## Optimization 2: Static Shape Optimization

**Status:** Pending
**Expected Impact:** 3-5%

**Changes:**
```python
# Before: Dynamic concatenate
msg_hist = jnp.concatenate([msg_hist[msg_len:], new_tokens])

# After: Static roll + update
msg_hist = jnp.roll(msg_hist, -msg_len, axis=0)
msg_hist = msg_hist.at[-msg_len:].set(new_tokens)
```

| Metric | Baseline | After | Change |
|--------|----------|-------|--------|
| Total Time | TBD | TBD | TBD |
| Per-Epoch Time | TBD | TBD | TBD |

---

## Optimization 3: XLA Environment Flags

**Status:** Pending
**Expected Impact:** Variable

**Changes:**
```bash
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=true"
```

| Metric | Baseline | After | Change |
|--------|----------|-------|--------|
| Total Time | TBD | TBD | TBD |
| Per-Epoch Time | TBD | TBD | TBD |

---

## Summary

| Optimization | Per-Epoch Change | Cumulative | Valid Test? |
|--------------|------------------|------------|-------------|
| OLD Baseline (freeze=False) | 87s | - | N/A |
| NEW Baseline (freeze=True) | TBD | - | ✅ |
| 1. Batched Loading | TBD | TBD | Pending |
| 2. Static Shape | TBD | TBD | Pending |
| 3. XLA Flags | TBD | TBD | Pending |
| **Final** | TBD | TBD | - |
