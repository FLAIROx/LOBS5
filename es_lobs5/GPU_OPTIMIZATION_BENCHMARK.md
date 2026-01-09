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

## Baseline (Pre-optimization)

**Date:** 2026-01-09
**Commit:** 31d667d (docs: add GPU optimization code changes reference)
**Job ID:** 1865747

| Metric | Value |
|--------|-------|
| Total Time (10 epochs) | **1696.5 seconds** |
| Per-Epoch Time (avg) | **169.64 seconds** |
| First Epoch (incl. XLA compile) | **573 seconds** |
| Avg Epoch (epochs 2-10) | **~90 seconds** |

**Detailed epoch times:**
- Epoch 0: 573s (XLA compilation)
- Epoch 1: 87s
- Epoch 2: 87s
- Epoch 3: 87s
- Epoch 4: 86s
- Epoch 5: 87s
- Epoch 6: 87s
- Epoch 7: 87s
- Epoch 8: 87s
- Epoch 9: 87s

**Notes:**
- XLA compilation cache enabled (first run triggers compilation)
- Standard float32 precision
- 360M parameter model (360,436,845 params)
- 4 GPUs (shard_map + vmap)

---

## Optimization 1: Historical Replay Batched Loading

**Status:** COMPLETED
**Job ID:** 1865778
**Commit:** 4e30831

| Metric | Baseline | After | Change |
|--------|----------|-------|--------|
| Total Time (10 epochs) | 1696.5s | **803.9s** | **-52.6%** |
| Per-Epoch Time (avg) | 169.64s | 80.39s | -52.6% |
| First Epoch (XLA compile) | 573s | 375s | **-34.6%** |
| Avg Epoch (2-10) | 87s | **~6s** | **-93.1%** |

**Changes:**
- Pre-slice all background messages before `jax.lax.scan` using `jax.lax.dynamic_slice`
- Replace dynamic indexing with batched array access
- XLA generates more efficient code with static slice shapes

**Analysis:**
- The optimization significantly reduces per-epoch time from 87s to ~6s (93% improvement!)
- XLA compilation is also faster (375s vs 573s) due to simpler compiled graph
- The batched approach eliminates dynamic indexing overhead inside the scan loop

---

## Optimization 2: BF16 Mixed Precision

**Status:** Pending
**Expected Impact:** 5-10%

| Metric | Baseline | After | Change |
|--------|----------|-------|--------|
| Total Time | - | - | - |
| Per-Epoch Time | - | - | - |

**Changes:**
- Enable bfloat16 for ES model weights
- Keep decoder/log_softmax in float32

---

## Optimization 3: Static Shape Optimization

**Status:** Pending
**Expected Impact:** 5-10%

| Metric | Baseline | After | Change |
|--------|----------|-------|--------|
| Total Time | - | - | - |
| Per-Epoch Time | - | - | - |

**Changes:**
- Replace dynamic concatenate with `jax.lax.dynamic_update_slice`
- Ensure static shapes for XLA fusion

---

## Optimization 4: XLA Environment Flags

**Status:** Pending
**Expected Impact:** Variable

| Metric | Baseline | After | Change |
|--------|----------|-------|--------|
| Total Time | - | - | - |
| Per-Epoch Time | - | - | - |

**Changes:**
- Add `XLA_FLAGS="--xla_gpu_enable_triton_gemm=true"`

---

## Summary

| Optimization | Per-Epoch Change | Cumulative |
|--------------|------------------|------------|
| Baseline | - | - |
| 1. Batched Loading | - | - |
| 2. BF16 | - | - |
| 3. Static Shape | - | - |
| 4. XLA Flags | - | - |
| **Final** | - | - |
