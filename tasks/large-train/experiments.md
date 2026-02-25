# Large-Train 32N Experiment Tracker

All 32-node (128 GPU) experiments recorded here. Non-KTL (IGNORE_TIMES=True) and KTL (IGNORE_TIMES=False) distinguished by column.

## G0-baselines-sweeplr (75M, LR Sweep, FP32)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| G0-baselines-sweeplr | 2439639 | 75M | True | None | 32 | 12 | 1e-3 | 1536 | 0 | E0 (FAILED exit 126, ~5min) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/k4slk85y | ~5min | 4eb4cf56-83e5-42ef-a308-e48c5ebcf9b8 | BSZ=12 first attempt, crash during init |
| G0-baselines-sweeplr | 2439643 | 75M | True | None | 32 | 12 | 3e-3 | 1536 | 0 | E0 (FAILED exit 126, ~5min) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/tni0pl9k | ~5min | 95539306-4f94-4edc-92da-437c09911636 | BSZ=12 lr=3e-3, crash during init |
| G0-baselines-sweeplr | 2439698 | 75M | True | None | 32 | 10 | 3e-3 | 1280 | 0 | TIMEOUT 30min (no training log) | -- | -- | -- | -- | -- | ~30min | 95539306-4f94-4edc-92da-437c09911636 | BSZ=10 short test, timeout before training started |
| G0-baselines-sweeplr | 2439364 | 75M | True | None | 32 | 12 | 3e-3 | 1536 | 16 | CANCELLED by user (~3h) | 72.16% | 69.55% | 1.432 | 1.567 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/xbvxb3ap | ~3h | 4eb4cf56-83e5-42ef-a308-e48c5ebcf9b8 | lr=3e-3 BSZ=12: Epoch 4 gradient explosion (183x), cancelled |
| G0-baselines-sweeplr | 2439703 | 75M | True | None | 32 | 10 | 5e-3 | 1280 | 6 | FAILED exit 126 (~1.5h) | 69.15% | 67.29% | 1.628 | 1.718 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/qqz17itg | ~1.5h | 4eb4cf56-83e5-42ef-a308-e48c5ebcf9b8 | lr=5e-3: Epoch 2 gradient explosion (910x), never recovered |
| G0-baselines-sweeplr | 2439704 | 75M | True | None | 32 | 10 | 1e-3 | 1280 | 5 | E5 batch 58 NCCL deadlock (~1.3h) | 72.50% | 69.70% | 1.412 | 1.553 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/r0ily8fd | ~1.3h | 4eb4cf56-83e5-42ef-a308-e48c5ebcf9b8 | lr=1e-3 Round 1: stable, NCCL deadlock killed it |
| G0-baselines-sweeplr | 2439874 | 75M | True | None | 32 | 10 | 3e-3 | 1280 | 0 | CANCELLED (~50min, no training log) | -- | -- | -- | -- | -- | ~50min | 95539306-4f94-4edc-92da-437c09911636 | lr=3e-3 BSZ=10, cancelled before training output |
| G0-baselines-sweeplr | 2440129 | 75M | True | None | 32 | 12 | 1e-3 | 1536 | 0 | FAILED ~12min (forgot BSZ=10) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/5wai54fo | ~12min | 42b74341 | Forgot BSZ=10, used default 12, OOM |
| G0-baselines-sweeplr | 2440130 | 75M | True | None | 32 | 12 | 5e-4 | 1536 | 0 | FAILED ~14min (forgot BSZ=10) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/4m61ugrr | ~14min | 42b74341 | Forgot BSZ=10, used default 12, OOM |
| G0-baselines-sweeplr | 2440156 | 75M | True | None | 32 | 10 | 1e-3 | 1280 | 20 (5+15 resume) | E20 batch 485 NCCL deadlock (~3.2h) | 75.33% | 72.28% | 1.250 | 1.398 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/qh0gzhae | ~3.2h | 42b74341 | **lr=1e-3 Round 2 resume step 2690. Best@E13. NCCL killed@E20** |
| G0-baselines-sweeplr | 2440157 | 75M | True | None | 32 | 10 | 5e-4 | 1280 | 14 | E14 batch 498 NCCL deadlock (~3h) | 74.01% | 70.98% | 1.324 | 1.475 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/o2loo87q | ~3h | 42b74341 | lr=5e-4 baseline fresh. Best@E14. NCCL killed@E14 |
| G0-baselines-sweeplr | 2458350 | 75M | True | None | 32 | 10 | 1e-3 | 1280 | 0 | CANCELLED (PENDING) | -- | -- | -- | -- | -- | 0 | this-session | Resume from step 6994, cancelled |
| G0-baselines-sweeplr | 2458351 | 75M | True | None | 32 | 10 | 5e-4 | 1280 | 0 | CANCELLED (PENDING) | -- | -- | -- | -- | -- | 0 | this-session | Resume from step 7532, cancelled |
| G0-baselines-sweeplr | 2458353 | 75M | True | None | 32 | 10 | 1e-3 | 1280 | **40** (E14→E40) | **COMPLETED 40ep** (~4h05m) | **76.61%** | **73.45%** | **1.177** | **1.323** | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/cgdexweb | ~4h | 8227a325 | **lr=1e-3 R3: NCCL 4-layer fix. Best@E40 (still improving!). First stable 40ep@32N** |
| G0-baselines-sweeplr | 2458354 | 75M | True | None | 32 | 10 | 5e-4 | 1280 | **40** (E15→E40) | **COMPLETED 40ep** (~3h58m) | **75.63%** | **72.46%** | **1.229** | **1.381** | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/e8d5fogg | ~4h | 8227a325 | **lr=5e-4 R2: NCCL 4-layer fix. Best@E40 (still improving!). First stable 40ep@32N** |

## G3-ignore-time (75M, IGNORE_TIMES=True, Speed + Production)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| main (pre-fix) | 2439132 | 75M | True | None | 32 | 12 | 1e-3 | 1536 | 11 | E12 train start, NCCL deadlock (~2h20m) | 73.92% | 70.87% | 1.332 | 1.482 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/i0w9srol | ~2h20m | -- | main branch pre-fix lr=1e-3. Best@E9. NCCL abort during E12 |
| G3 (pre-fix) | 2439006 | 75M | True | None | 32 | 12 | 3e-3 | 1536 | 1 | E2 test eval, CANCELLED by user (~35min) | 68.28% | 66.70% | 1.695 | 1.763 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/wgdbthqf | ~35min | -- | Pre-fix lr=3e-3. Cancelled during E2 test eval |
| G3 (pre-fix) | 2439012 | 75M | True | None | 32 | 12 | 5e-3 | 1536 | 0 | E1 test batch 1, NCCL watchdog timeout (~32min) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/7z4bpbo0 | ~32min | -- | Pre-fix lr=5e-3. E1 train+val OK, NCCL killed during test eval |
| G3-ignore-time | 2439536 | 75M | True | 300 | 32 | 12 | 3e-3 | 1536 | 3 (curtail) | Completed CURTAIL speed test | 69.41% | 67.59% | 1.611 | 1.704 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/kglo01io | ~8min | 1e0f0db4 | Speed benchmark: 1.31 s/step stable (step 200-300). Best@E2 |
| G3-ignore-time | 2439868 | 75M | True | None | 32 | 10 | 3e-3 | 1280 | 4 | E5 train, CANCELLED by user (~53min) | 68.82% | 67.09% | 1.663 | 1.745 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/y76ftu6h | ~53min | 1e0f0db4 | Full 40ep attempt. Best stuck@E1 (loss worsened E2-4). Cancelled |

## G2-22-24-tok (24tok encoding, vocab=2112, ~55M params)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| G2-24tok | 2440076 | 55M | True | None | 32 | 10 | 5e-4 | 1280 | 31 | FAILED exit 126 (~5h52m) | **78.87%** | **75.89%** | **1.058** | **1.190** | https://wandb.ai/kang-oxford/lobs5-75M-G2/runs/pmk5zbad | ~5h52m | 5c4fd67d | **24tok 32N contiguous. Best@E31. Gradient explosion E24 recovered. Still improving when crashed** |
| G2-24tok | 2440112 | 55M | True | None | 32 | 10 | 5e-4 | 1280 | **40** | **COMPLETED 40ep** (~7h06m) | **76.88%** | **73.95%** | **1.173** | **1.308** | https://wandb.ai/kang-oxford/lobs5-75M-G2/runs/i2n8qufp | ~7h06m | 5c4fd67d | **24tok 32N non-contiguous. Completed 40ep. Best@E40. Converged ~E8 then slow improvement** |
| G2-bsz-sweep | 2439389 | 55M | True | 300 | 1 | 12 | 3e-3 | 48 | 1 | E2 (CURTAIL) | 69.16% | 68.02% | 1.659 | -- | https://wandb.ai/kang-oxford/lobs5-G2-24tok-bsz-sweep/runs/ojgnb5w1 | -- | 097a1028 | 1N BSZ=12 OK. 1 epoch speed test |
| G2-bsz-sweep | 2439390 | 55M | True | 300 | 1 | 10 | 3e-3 | 40 | 2 | E2 loss diverged to 49.87 | 68.38% | 67.73% | 1.686 | -- | https://wandb.ai/kang-oxford/lobs5-G2-24tok-bsz-sweep/runs/x6pdaef1 | -- | 097a1028 | 1N BSZ=10 diverged E2 |
| G2-loss-compare | 2439711 | 55M | True | 300 | 1 | 12 | 3e-3 | 48 | 1 | E2 (CURTAIL) | 65.63% | 65.94% | 1.932 | -- | https://wandb.ai/kang-oxford/lobs5-G2-loss-compare/runs/zsva3l3j | -- | 5c4fd67d | 24tok vs 22tok loss compare (1 epoch) |
| A3-22tok | 2207381 | 75M | False | None | 1 | 8 | 5e-5 | 32 | 7 | TIMEOUT 24h | 71.27% | 39.42% | 1.299 | 6.453 | https://wandb.ai/kang-oxford/lobs5-A3-22vs24tok/runs/85ipv727 | ~24h | eb59c171 | 22tok baseline. Test loss diverged (OOD: GOOG 2022→JAN2023) |
| A3-24tok | 2207382 | 55M | False | None | 1 | 8 | 5e-5 | 32 | 7 | TIMEOUT ~26h | 74.63% | 36.00% | 1.158 | 6.981 | https://wandb.ai/kang-oxford/lobs5-A3-22vs24tok/runs/wx0h3ih5 | ~26h | eb59c171 | 24tok. Better val loss but worse OOD test divergence |

## G4-rsmnorm_nobias (75M, RMSNorm + Dense no-bias)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| G4 (2N test) | 2439376 | 75M | True | 300 | 2 | 12 | 3e-3 | 96 | 1 | E2 start, OOM 81.42 GiB during eval (~11min) | 67.16% | 65.81% | 1.775 | 1.817 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/dwilk5jl | ~11min | 1e0f0db4 | 2N speed test. 0.760 s/step. OOM at eval stage |
| G4 (2N test) | 2439494 | 75M | True | 300 | 2 | 12 | 3e-3 | 96 | 1 | E2 start, OOM 72.05 GiB during eval (~11min) | 67.15% | 65.84% | 1.779 | 1.820 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/mn3v73mh | ~11min | 1e0f0db4 | 2N MEM_FRACTION=0.80 retry. Still OOM at eval |
| G4-rsmnorm_nobias | 2439869 | 75M | True | 300 | 32 | 10 | 3e-3 | 1280 | 3 | E4 step 252 TIME LIMIT (~30min) | 70.88% | 68.40% | 1.518 | 1.640 | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/lq82mbzg | ~30min | 1e0f0db4 | 32N RMSNorm+NoBias, non-contiguous, ~1.20 s/step |
| G4-rsmnorm_nobias | 2440098 | 75M | True | 300 | 32 | 10 | 3e-3 | 1280 | 3 | E4 start, TIMEOUT 30min | 70.47% | 68.15% | 1.565 | 1.690 | https://wandb.ai/kang-oxford/lobs5-75M-G4/runs/y7h9lfhb | ~30min | 1e0f0db4 | 32N resubmit w/ correct WANDB_PROJECT. Best@E3 |

## G5-swiglu-mlp (75M baseline + 188M SwiGLU)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description | Who |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|-----|
| G5 (baseline) | 2439393 | 75M | True | 300 | 32 | 12 | 3e-3 | 1536 | 3 | TIMEOUT 30min | 70.49% | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/aoaf5zgw | ~30min | -- | Baseline half_glu1 for SwiGLU comparison; 1.30 s/step | -- |
| G5-swiglu | 2439378 | 188M | True | -- | 32 | 12 | 3e-3 | 1536 | 0 | OOM 100.31 GiB | -- | -- | -- | -- | -- | ~0min | -- | SwiGLU BSZ=12 instant OOM | -- |
| G5-swiglu | 2439495 | 188M | True | 300 | 32 | 8 | 3e-3 | 1024 | -- | eval overflow bug (math.exp) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/ciyg64tv | -- | -- | SwiGLU lecun_normal, eval PPL crash; 2.27 s/step | -- |
| G5-swiglu | 2439977 | 188M | True | 300 | 32 | 8 | 3e-3 | 1024 | 0 | E0 Train Loss = 35M (diverged) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/e5ceo23u | -- | -- | SwiGLU lecun_normal init, lr=3e-3, numerical explosion | -- |
| G5-swiglu | 2440041 | 188M | True | 300 | 32 | 8 | 5e-4 | 1024 | ~1 | E1 loss=900 (diverged) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/wp8nkn1h | -- | -- | SwiGLU lecun_normal init, warmup masked instability | -- |
| G5-swiglu | 2440136 | 188M | True | 300 | 32 | 8 | 1e-4 | 1024 | 2 | TIMEOUT 30min | 63.78% | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/1inj3iyr | ~30min | -- | **SwiGLU init fix (normal 0.02), stable; 2.19 s/step** | -- |
| G5-swiglu-bottleneck | 2466356 | TBD | True | 300 | 2 | 10 | 5e-4 | 80 | -- | PENDING | -- | -- | -- | -- | -- | -- | -- | SwiGLU 4:1 compression bottleneck (d_ff=256), no 2/3 factor, mlp_ratio=0.25 | Jonathan |

## G6-auto-lr (75M, Prodigy optimizer, 32N)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| G6-auto-lr | 2439976 | 75M | True | None | 32 | 10 | Prodigy | 1280 | 9 | NCCL deadlock E9 batch 64 | 71.15% | 68.49% | 1.524 | -- | https://wandb.ai/kang-oxford/lobs5-75M-G6/runs/ywg1j08o | -- | 1d5e0688 | Prodigy estim_lr=1.65e-3; still improving when killed |

## E1-mixed-precision (75M, BF16 Sandwich, 32N)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| E1-BF16 | 2439698 | 75M | True | 50 | 32 | 10 | 3e-3 | 1280 | 40 | Completed | -- | 69.39% | 1.433 | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/7tjl2w85 | -- | -- | BF16 CURTAIL=50, matches FP32 precision (Val Loss diff <0.1%) |
| E1-BF16 | 2439874 | 75M | True | None | 32 | 10 | 3e-3 | 1280 | 3 | E3 Train Loss exploded to 105.77 | -- | 69.16% | 1.463 | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/po31e4ku | -- | -- | BF16 full train LR=3e-3: diverged at epoch 3 |
| E1-BF16 | 2440028 | 75M | True | None | 32 | 10 | 1e-3 | 1280 | ~8 | E8 Train Loss diverged to 2.82 | -- | 69.94% | 1.397 | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/zjiruhie | -- | -- | BF16 full train LR=1e-3: diverged at epoch ~8 |
| E1-BF16 | 2440140 | 75M | True | None | 32 | 10 | 5e-4 | 1280 | -- | RUNNING | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/8g5s9vvh | -- | -- | BF16 full train LR=5e-4: stability TBD |

## G1-scale-up (360M, 32N)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| G1-scale-up | 2441019 | 360M | True | None | 32 | 2 | 5e-4 | 256 | 0 | E0 batch 64 NCCL deadlock (900s) | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-360M-G1/runs/5fl9ub72 | ~6min | 21ca5d32 | 360M (d=2048, n_layers=24) first 32N attempt, NCCL killed |

## NCCL Infrastructure Tests (32N)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| shard_autotuning | 2426449 | 75M | True | None | 32 | 7 | 3e-3 | 896 | 14 | E14 step 356 NCCL deadlock (6h) | -- | -- | -- | -- | -- | ~6h | -- | Production BSZ=7 run, 1.02 s/step. NCCL killed@E14 |
| NCCL-deadlock | 2438407 | 75M | True | -- | 32 | -- | -- | -- | 0 | E0 batch 1 NCCL deadlock | -- | -- | -- | -- | -- | -- | -- | NCCL deadlock E0 B1 (NCCL 2.28.9 ARM CAS bug) |
| D3-save-barrier | 2438653 | 75M | True | 50 | 32 | 12 | 3e-3 | 1536 | -- | (barrier verification test) | -- | -- | -- | -- | -- | -- | -- | Pre-eval barrier + 1200s timeout verification |
| NCCL-fix | 2447130 | 75M | True | -- | 32 | 10 | -- | 1280 | 4 | TIMEOUT 30min (no deadlock) | -- | -- | -- | -- | -- | ~30min | -- | CXI hang fix + NCCL 2.29.3. Stable 0.91 s/step. 4 epoch OK |
| NCCL-^LL128 | 2447647 | 75M | True | -- | 32 | 10 | -- | 1280 | 2+ | TIMEOUT 30min | -- | -- | -- | -- | -- | ~30min | -- | NCCL_PROTO=^LL128 test. 1.12 s/step (23% slower). LL128 is beneficial |
| NCCL-BlueConnect | 2440967 | 75M | True | -- | 32 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | BlueConnect test. 3.55 s/step = 3x regression vs 1.18 baseline |

## B2-scaling-benchmark (75M, 128GPU speed test)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| B2-2D-mesh | 2421932 | 75M | -- | 400 | 32 | 7 | -- | 896 | 0 (speed) | CURTAIL speed test | -- | -- | -- | -- | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/kdeylyfr | -- | -- | 128GPU 2D Mesh: 1.745 s/step, 513.5 samp/s, 23% eff. numpy eigh fix |

## G8-local-steps (75M, Local Steps K=10, 32N)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | LOCAL_STEPS_K | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|---------------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| G8-local-steps | 2458093 | 75M | True | 50 | 2 | 12 | 5e-4 | 96 | 10 | 2 | E3/B0 (OOM 80.46 GiB) | 59.14% | 60.11% | 2.208 | 2.167 | https://wandb.ai/kang-oxford/lobs5-75M-G8/runs/0ozcipoo | ~8min | 821d92ed | 2N test, BSZ=12 OOM at E3 (known BSZ=12 mem issue) |
| G8-local-steps | 2458144 | 75M | True | None | 32 | 10 | 5e-4 | 1280 | 10 | **38** | E39/S214 (SLURM 24h timeout) | **75.54%** | **72.38%** | **1.235** | **1.386** | https://wandb.ai/kang-oxford/lobs5-75M-G8/runs/xex7oro3 | ~6h | 821d92ed | **Local Steps K=10, 40ep, contiguous nid010998-011029. Best@E38. ~2pp below std hierarchical (76.61%)** |

## KTL: Keep-Time-Large (IGNORE_TIMES=False)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|
| KTL (keep-time) | 2458440 | 75M | **False** | None | 32 | 10 | 1e-3 | 1280 | 37 | CANCELLED by user E37 step 421 (~6h) | **80.28%** | **77.82%** | **1.017** | **1.140** | https://wandb.ai/kang-oxford/lobs5-75M-B1/runs/ew3af26l | ~6h | -- | **Keep-time experiment. Best@E37. Still improving when cancelled** |

## G7-muon-optimizer (75M, Muon Newton-Schulz, 32N)

| Task | Job ID | Model Size | IGNORE_TIMES | Curtail | Nodes | Micro BSZ | LR | Global BSZ | Completed Epochs | Stopped At (Epoch/Step) | Best Val Acc | Best Test Acc | Best Val Loss | Best Test Loss | W&B | Time | Session ID | Description | Who |
|------|--------|-----------|-------------|---------|-------|-----------|-----|-----------|-----------------|------------------------|-------------|--------------|--------------|---------------|-----|------|-----------|-------------|-----|
| G7-muon-optimizer | 2467214 | 75M | -- | 300 | 2 | 10 | 1e-3 | 80 | 0 | FAILED (conda PATH, python not found) | -- | -- | -- | -- | -- | ~1min | 6124a772 | 2N speed test — conda.sh CONDA_EXE pointed to kangli.s5e home, inaccessible on nid010936-010937 | Jonathan |
| G7-muon-optimizer | 2472218 | 75M | -- | 300 | 2 | 10 | 1e-3 | 80 | -- | -- | -- | -- | -- | -- | -- | -- | 94a426c2 | 2N speed test — conda PATH fix applied, OPT_CONFIG=muon, MUON_LR_FACTOR=1 (default) | Jonathan |

---

## Notes

- **Job 2439698**: Appears in both G0 (as FP32 timeout) and E1 (as BF16 CURTAIL=50). The E1 entry with W&B `7tjl2w85` is the accurate record.
- **Job 2439874**: Appears in both G0 (as cancelled) and E1 (as BF16 diverged E3). The E1 entry with W&B `po31e4ku` is the accurate record.
- **Job 2439364**: Shared between G0 (FP32 LR sweep) and E1 (FP32 32N production baseline). W&B `xbvxb3ap`.
- **NCCL root cause**: ARM CAS weak-failure bug in NCCL 2.28.9 + CXI eager buffer exhaustion. Fixed by NCCL 2.29.3 (base env) + CXI flags.
