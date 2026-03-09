# J2 Muon Optimizer — Phase 1 & 2 Sweep Results

## Experiment Setup

- **Proxy model**: d_model=512, n_layers=24, blocks=8, ssm_size=512 (~25M params)
- **Target model**: d_model=2048, n_layers=24, blocks=32, ssm_size=2048 (~360M params)
- **Data**: GOOG single ticker, 1 node (4 GPU), BSZ=8/gpu, global BSZ=32
- **Training**: CURTAIL_EPOCHS=300 (301 steps/epoch), 5 epochs (1505 total steps)
- **Transfer assumptions**: LR is width-invariant (Muon core property), WD scales as ~1/width (arxiv 2512.05620)

## Phase 1: LR Sweep (WD=0.01 fixed)

| Config | muon_lr | Val Loss | Val Acc | All Orders Loss | All Orders Acc | Last Order Loss | Last Order Acc | Job ID | W&B |
|--------|---------|----------|---------|-----------------|----------------|-----------------|----------------|--------|-----|
| **Muon** | **0.02** | **1.237** | **73.86%** | **0.932** | **79.76%** | **0.915** | **80.11%** | 2686751 | [odoyc6mx](https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/odoyc6mx) |
| Muon | 0.04 | 1.263 | 73.62% | 0.951 | 79.61% | 0.934 | 79.91% | 2686754 | [sf88xhp3](https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/sf88xhp3) |
| Muon | 0.01 | 1.275 | 73.38% | 0.961 | 79.35% | 0.943 | 79.72% | 2686753 | [yw665lm8](https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/yw665lm8) |
| AdamW | — | 1.521 | 71.68% | 1.159 | 77.85% | 1.139 | 78.24% | 2686752 | [864kle0t](https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/864kle0t) |

**Finding**: Muon LR=0.02 (paper default) wins. All Muon variants beat AdamW by ~20% lower loss and +2pp accuracy.

## Phase 2: WD Sweep (muon_lr=0.02 fixed)

| WD | Val Loss | Val Acc | All Orders Loss | All Orders Acc | Last Order Loss | Last Order Acc | Job ID | W&B |
|----|----------|---------|-----------------|----------------|-----------------|----------------|--------|-----|
| 0.005 | 1.237 | 73.83% | 0.931 | 79.76% | 0.914 | 80.06% | 2686988 | [zts0zuh2](https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/zts0zuh2) |
| 0.01 | 1.237 | 73.86% | 0.932 | 79.76% | 0.915 | 80.11% | 2686751 | [odoyc6mx](https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/odoyc6mx) |
| **0.02** | **1.231** | **73.93%** | **0.927** | **79.80%** | **0.911** | **80.11%** | 2686989 | [dzhawbha](https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/dzhawbha) |

**Finding**: WD=0.02 is marginally best but the curve is very flat (0.5% relative difference across 4x range). WD is not sensitive at this scale.

## Transferred HPs for 360M Target

| Parameter | Proxy (d=512) | 360M (d=2048) | Transfer Rule |
|-----------|---------------|---------------|---------------|
| Muon kernel LR | 0.02 | 0.02 | Width invariant |
| Weight decay | 0.02 | 0.005 | 1/width scaling |
| SSM LR (Adam) | 5e-4 | 5e-4 | Unchanged |
| AdamW LR (rest) | 5e-4 | 5e-4 | Unchanged |
