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
**Commit:** (current HEAD before optimization)
**Job ID:** TBD

| Metric | Value |
|--------|-------|
| Total Time (10 epochs) | TBD |
| Per-Epoch Time | TBD |
| First Epoch (incl. compile) | TBD |
| Avg Epoch (excluding first) | TBD |

**Notes:**
- XLA compilation cache enabled
- Standard float32 precision

---

## Optimization 1: Historical Replay Batched Loading

**Status:** Pending
**Expected Impact:** 10-15%

| Metric | Baseline | After | Change |
|--------|----------|-------|--------|
| Total Time | - | - | - |
| Per-Epoch Time | - | - | - |

**Changes:**
- Pre-slice all background messages before `jax.lax.scan`
- Replace dynamic indexing with batched access

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
