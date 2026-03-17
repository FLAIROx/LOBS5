# C10: NCCL_BUFFSIZE Systematic Sweep

## Goal
Find optimal NCCL_BUFFSIZE for 360M model on GH200 + Slingshot.
Current value (2MB) was optimized for 75M model — never tested beyond 4MB.

## Context
- Future: msg_seq_len 500→4000 (12K→96K tokens) → AllReduce payload grows
- GH200 has 85.5 GB VRAM — memory is NOT the constraint
- Real constraint: Slingshot NIC injection bandwidth (~25 GB/s)
- Per ring step injection = channels × BUFFSIZE (currently 40ch × 2MB = 80MB)

## Historical Data (75M model, from C5/C5a/C5a1)

| BUFFSIZE | Scale | s/step | Efficiency | Job |
|----------|-------|--------|------------|-----|
| 1 MB | 32N | 1.00 | 55.0% | 2426358 |
| 2 MB | 16N | 0.66 | 83.3% | 2426303 |
| 2 MB | 32N | 0.94 | 58.5% | 2424846 |
| 4 MB | 16N | 1.58 | 34.8% | 2424845 |
| 8MB+ | - | - | - | Never tested |

## Phase 1: Explore Upward (4 jobs @ 16N)

| Job | BUFFSIZE | Bytes | Purpose |
|-----|----------|-------|---------|
| C10-A | 2 MB | 2097152 | Control (current) |
| C10-B | 8 MB | 8388608 | First untested value |
| C10-C | 16 MB | 16777216 | Aggressive |
| C10-D | 32 MB | 33554432 | Upper bound test |

Config: 16N, 360M, CURTAIL_EPOCHS=300, PER_GPU_BSZ=2, --time=00:30:00

## Phase 2: Adaptive (depends on Phase 1)

| Phase 1 Result | Phase 2 Direction |
|----------------|-------------------|
| Larger is better | Test 64MB, 128MB, 256MB |
| Peak at 8MB | Test 4MB, 6MB, 10MB, 12MB |
| 2MB still best | Test 1.5MB, 3MB, 4MB (confirm) |
| All similar | BUFFSIZE insensitive for 360M |

## Measurement Method
- CURTAIL_EPOCHS=300 → exactly 301 steps per epoch
- Extract step 250 and step 300 elapsed from tqdm
- per_step_ms = (elapsed_300 - elapsed_250) / 50 * 1000
- Must wait 200+ steps for XLA autotune steady state
