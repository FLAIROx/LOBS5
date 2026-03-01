# Auto-Resume Job Chain (360M, shard-map branch)

## Context
- Script: `train_full_autoreg.batch` on `shard-map` branch
- Auto-resume triggers on non-zero srun exit (exit=1 watchdog kill, exit=143 SIGTERM/wall time)
- BSZ adjustment commit: `42154dca` (2026-02-27) — scales resume step when node count changes
- Only job 2503382+ used the new BSZ-adjusted code; earlier jobs used old `step + 1`
- All 2^x node jobs resumed at same node count → old code = new code (no impact)
- Model: 360M (d=2048, L=24, B=32, ssm=2048) | W&B project: `lobs5-360M-G30`
- Data: 8 tickers × 4yr (2022-2025) | steps_per_epoch: 26496 (256N) / 13248 (512N)

## Job Chain

| Job ID | Start (UTC) | End (UTC) | Nodes | Exit | Code | Resume From | Last Ckpt Step | W&B Run | Next Job |
|---|---|---|---|---|---|---|---|---|---|
| 2486590 | ~2026-02-26 01:54 | ~05:30 | ? | ? | old | (predecessor) | 1429 | zryalmcb | → 2488850 |
| **2488850** | 2026-02-26 05:41 | 2026-02-27 02:19 | 512N | 1 | old | j2486590 step 1429 | **9283** | hgzw5c99 | → 2495772 (depth 2) |
| 2495772 | — | — | ? | — | — | (NO LOGS — auto-cancelled or never ran) | — | — | — |
| 2495903 | 2026-02-27 03:04 | 2026-02-27 03:10 | 512N | 143 | old | j2488850 step 9283 → batch_idx=7854 | — | — | — (script bug) |
| 2495977 | 2026-02-27 03:53 | 2026-02-27 03:59 | 512N | 143 | old | j2488850 step 9283 → batch_idx=7854 | — | — | → 2495998 (depth 1) |
| 2495998 | — | — | ? | — | — | (NO LOGS — auto-cancelled or never ran) | — | — | — |
| **2496000** | 2026-02-27 04:06 | 2026-02-27 10:30 | 512N | 1 | old | j2488850 step 9283 → batch_idx=7854 | **10990** | e6i8kq38 | → 2496963 (depth 1) |
| 2496963 | — | — | ? | — | — | (NO LOGS — auto-cancelled or never ran) | — | — | — |
| 2498772 | 2026-02-27 13:10 | 2026-02-27 13:16 | 300N | 143 | old | j2496000 step 10990 → batch_idx=16316 (manual) | — | — | → 2498844 (depth 1) |
| 2498844 | 2026-02-27 13:16 | 2026-02-27 14:25 | 300N | cancelled | old | j2496000 → batch_idx=9561 (auto, NOT BSZ-adjusted) | — | — | — (priority chain: 2498846-48) |
| 2500301 | 2026-02-27 14:31 | 2026-02-27 17:52 | 320N | cancelled | — | j2496000 → batch_idx=15297 (manual) | — | — | — |
| **2503382** | 2026-02-27 18:23 | 2026-02-27 19:30 | 256N | 143 | new (BSZ-adj) | batch_idx=19122 | **22237** | 7ozaawgb | → 2504227 (depth 1) |
| **2504227** | 2026-02-27 19:41 | 2026-02-27 ~21:43 | 256N | 143 | new (BSZ-adj) | j2503382 step 22237 → batch_idx=19379 | **23258** | c4sz78k1 | → 2504777 (depth 2) |
| 2504777 | — | — | ? | — | — | (NO LOGS — auto-cancelled or never ran) | — | — | — |
| **2504793** | 2026-02-27 21:58 | ~22:30+ | 256N | 1 | new (BSZ-adj) | j2504227 step 23258 → batch_idx=20400 | **(none)** | ubz5isc7 | — (no ckpt, cannot resume) |
| **2507247** | 2026-02-28 19:38 | ? | 256N | ? | new (BSZ-adj) | fallback→j2503382, batch_idx=19379 | **(none)** | i887yu5e | → 2509331 (auto) |
| **2509490** | 2026-02-28 21:09 | ? | 256N | ? | new (BSZ-adj) | fallback→j2503382, batch_idx=19379 | **(none)** | hyea0zxy | → 2509713 (auto) |
| 2509763 | 2026-02-28 21:58 | ? | 256N | ? | new (BSZ-adj) | j2503382(?) batch_idx=19379(?) | **(none)** | — | — |
| **2510097** | 2026-02-28 22:31 | ? | 256N | ? | new (BSZ-adj) | fallback→j2504227, batch_idx=20400 | **(none)** | — | → 2510240 (auto) |
| 2510240 | 2026-02-28 22:40 | ? | 256N | ? | — | j2504227(?) batch_idx=20400(?) | **(none)** | — | — (cancelled?) |
| 2510306 | 2026-02-28 22:48 | ? | 256N | ? | — | j2504227(?) batch_idx=20400(?) | **(none)** | — | — (cancelled?) |

## Checkpoint Summary

| Checkpoint Dir | Steps Saved | Last Step | step_in_epoch | Loss | Size |
|---|---|---|---|---|---|
| `j2486590_zryalmcb_2486590/` | 6 | 1429 | 1244 | 1.310 | — |
| `j2488850_hgzw5c99_2488850/` | 17 | **9283** | 7853 | **0.647** | 66 GB |
| `j2496000_e6i8kq38_2496000/` | 10 | **10990** | 9560 | 0.664 | 39 GB |
| `j2503382_7ozaawgb_2503382/` | 1 | **22237** | 19378 | 0.780 | 3.8 GB |
| `j2504227_c4sz78k1_2504227/` | 3 | **23258** | 20399 | 0.887 | 12 GB |
| `j2504793_ubz5isc7_2504793/` | 0 (metadata only) | — | — | — | 12 KB |
| `j2507247_i887yu5e_2507247/` | 0 (metadata only) | — | — | — | — |
| `j2509490_hyea0zxy_2509490/` | 0 (metadata only) | — | — | — | — |

Best checkpoint: `j2488850/9283` (loss **0.647**, epoch 0, 59% complete at 512N scale)

## Chain Dependency Graph

```
j2486590 (step 1429, loss 1.310)
    │
    ▼
j2488850 ★ (512N, 19.6h, step 0→9283, loss 0.647)
    ├──auto──→ 2495772 (NO LOGS)
    ├── j2495903 (512N, 6min, exit 143, script bug: RESUME_FROM_STEP parse error)
    ├── j2495977 (512N, 6min, exit 143) ──auto──→ 2495998 (NO LOGS)
    │
    ▼
j2496000 ★ (512N, 6.4h, step 7854→10990, loss 0.664)
    ├──auto──→ 2496963 (NO LOGS)
    ├── j2498772 (300N, 6min, exit 143) ──auto──→ j2498844 (300N, CANCELLED 14:25)
    ├── j2500301 (320N, CANCELLED 17:52, no training)
    │
    ▼
j2503382 ★ (256N, 1.1h, step 19122→22237, loss 0.780)
    │
    ▼
j2504227 ★ (256N, ~2h, step 19379→23258, loss 0.887)
    ├──auto──→ 2504777 (NO LOGS)
    │
    ▼
j2504793 (256N, step 20400→20654, WATCHDOG TIMEOUT 900s, NO CKPT)
    │
    ▼ ─── Feb 28 (all failed, no new checkpoints) ───
j2507247 (fallback→j2503382) → j2509331
j2509490 (fallback→j2503382) → j2509713
j2509763 (no output)
j2510097 (fallback→j2504227) → j2510240
j2510240, j2510306 (cancelled?)
```

★ = produced new checkpoints

## Notes
- **2504277 does NOT exist**: the user-provided ID was a typo. Actual auto-resume from 2504227 went to **2504777** (confirmed by `lobs5_2504227.out`)
- Jobs 2495772, 2495998, 2496963, 2504777: no logs found anywhere — likely auto-cancelled (afternotok dependency not met or manual scancel)
- Job 2495903: `.err` shows `RESUME_FROM_STEP=7854: command not found` — shell parsing bug in auto-resume export statement
- Job 2498772: started with batch_idx=16316 (manually set), but its auto-resume computed 9561 using OLD code (512N→300N not adjusted)
- Job 2498844: auto-resumed from 2498772 with wrong batch_idx=9561 (should be ~16344 for 512N→300N). Created priority chain (2498846-2498848). Cancelled at 14:25
- Job 2500301: manually submitted 320N, cancelled at 17:52
- **nid011290 repeated failures**: appeared as first error node in both 2503382 (.err: task 234, exit 134 SIGABRT) and 2504227 (.err: task 234, exit 1). Should be added to `--exclude` list
- Job 2504793: reached step 20654/26496 (78%) at ~4.67 s/step, then FATAL watchdog timeout (900s, NCCL deadlock). No step checkpoints saved
- **Loss regression**: loss went from 0.647 (step 9283, 512N) → 0.664 (step 10990) → 0.780 (step 22237) → 0.887 (step 23258). The 512N→256N transition and/or batch_idx mismatch may have disrupted training
- Feb 28 jobs (2507247-2510306): all failed to produce new checkpoints, repeatedly falling back to j2503382 or j2504227 checkpoints
- BSZ adjustment code written to disk between 2026-02-27 ~13:16 and 18:23 UTC (uncommitted local edit)
- Exit 1 = watchdog kill, Exit 143 = SIGTERM (SLURM wall time or scancel)
- 300N/320N jobs are non-2^x → not in experiment scope

## W&B Runs

| Job ID | W&B Run ID | URL |
|---|---|---|
| 2486590 | zryalmcb | https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/zryalmcb |
| 2488850 | hgzw5c99 | https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/hgzw5c99 |
| 2496000 | e6i8kq38 | https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/e6i8kq38 |
| 2503382 | 7ozaawgb | https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/7ozaawgb |
| 2504227 | c4sz78k1 | https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/c4sz78k1 |
| 2504793 | ubz5isc7 | https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/ubz5isc7 |
| 2507247 | i887yu5e | https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/i887yu5e |
| 2509490 | hyea0zxy | https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/hyea0zxy |
