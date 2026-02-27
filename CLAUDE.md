# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LOBS5** — A token-level autoregressive generative model of limit order book (LOB) message flow using the S5 (Simplified Structured State Space) architecture in JAX/Flax. This experiment worktree (`exp/H1-scaling-law`) trains 5 model sizes (10M–120M params) on 8 tickers × 4 years to study neural scaling laws for LOB generation.

This is a **git worktree** of the main repo at `AlphaTrade/LOBS5/`.

## Architecture

```
LOBSTER .npy data
  │  (PyTorch DataLoader — CPU only, multi-ticker support)
  ▼
Tokenized messages (24 tok/msg, vocab=2112, base-100 encoding)
  + Order book volume images (500 depth levels)
  │
  ▼
┌─ PaddedLobPredModel (lob/lob_seq_model.py) ─────────────────┐
│  Message encoder: Embed → n_message_layers × SequenceLayer   │
│  Book encoder:    Dense → pre-layers → project → post-layers │
│  Padded fusion:   concat at message boundary                 │
│  Fused encoder:   n_layers × SequenceLayer(S5)               │
│  Decoder:         Dense → log_softmax (vocab_size)           │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
Next-token cross-entropy loss → AdamW (optax) + cosine anneal
  → Orbax checkpoints + W&B logging
```

Each **SequenceLayer** (`s5/layers.py`): PreNorm → S5SSM (parallel scan via `jax.lax.associative_scan`) → half_glu1 activation → skip connection.

**S5SSM** (`s5/ssm.py`): HiPPO-LegS init, diagonal complex state matrix, ZOH discretization. Supports both parallel scan (training) and RNN mode (inference).

### Key source files

| File | Role |
|------|------|
| `run_train.py` | Entry point — CLI args, JAX distributed init, calls `lob.train.train()` |
| `lob/train.py` | Training loop — epochs, mini-epochs, validation, checkpointing, W&B |
| `lob/train_helpers.py` | JIT-compiled `train_step`/`eval_step`, optimizer, LR schedule, StepWatchdog |
| `lob/init_train.py` | Model/TrainState init, Orbax checkpoint load/save |
| `lob/sharding_utils.py` | JAX mesh creation (1D flat / 2D hierarchical), data & param sharding |
| `lob/dataloading.py` | Dataset factory, multi-ticker support, distributed sampler |
| `lob/lobster_dataloader.py` | `LOBSTER_Dataset` (PyTorch Dataset), file caching, masking |
| `lob/encoding.py` | `Message_Tokenizer` — LOB messages ↔ token sequences |
| `lob/lob_seq_model.py` | `PaddedLobPredModel` — Flax module (production model) |
| `s5/ssm.py` | S5 state space model core |
| `s5/layers.py` | `SequenceLayer` — single S5 layer with norm/activation/skip |
| `lob/inference.py` | Autoregressive generation with error correction |

## Common Commands

### Training (all runs via SLURM — never run on login node)

```bash
# Single model benchmark (30min, CURTAIL=300 steps)
D_MODEL=1024 N_LAYERS=12 BLOCKS=16 SSM_SIZE_BASE=1024 PER_GPU_BSZ=10 \
  CURTAIL_EPOCHS=300 sbatch --contiguous --nodes=64 --time=00:30:00 train_full_autoreg.batch

# Full training (15% of 1 epoch)
D_MODEL=1024 N_LAYERS=12 BLOCKS=16 SSM_SIZE_BASE=1024 PER_GPU_BSZ=10 \
  CURTAIL_EPOCHS=3179 NO_VALIDATION=1 sbatch --contiguous --nodes=64 --time=24:00:00 train_full_autoreg.batch

# Batch submit all 5 scaling law models
./submit_scaling_law.sh benchmark     # 5 sequential benchmarks
./submit_scaling_law.sh train         # 5 sequential 15%-epoch training runs
./submit_scaling_law.sh epoch         # 4 models × full epoch (120M skipped)
./submit_scaling_law.sh single 55M benchmark  # single model

# Dry run (preview commands)
DRY_RUN=1 ./submit_scaling_law.sh train

# Override node count
NODES=32 ./submit_scaling_law.sh benchmark
```

### Evaluation

```bash
# Post-training eval (single node)
RESTORE=/path/to/checkpoint sbatch eval_post_training.batch
```

### Tests (Docker-based, for local dev)

```bash
make test        # pytest ./tests/ inside Docker
make build       # build Docker image
```

## Scaling Law Model Configurations

| Label | d_model | n_layers | blocks | ssm_size | BSZ/GPU | Global BSZ (64N) |
|-------|---------|----------|--------|----------|---------|------------------|
| 10M   | 512     | 6        | 8      | 512      | 20      | 5,120            |
| 22M   | 768     | 6        | 12     | 768      | 14      | 3,584            |
| 55M   | 1024    | 12       | 16     | 1024     | 10      | 2,560            |
| 85M   | 1280    | 12       | 20     | 1280     | 6       | 1,536            |
| 120M  | 1536    | 12       | 24     | 1536     | 4       | 1,024            |

Default model (when no env overrides): **360M** (d_model=2048, n_layers=24, blocks=32, ssm_size=2048, BSZ=2/GPU).

## SLURM Job Structure

```
sbatch train_full_autoreg.batch
  └→ srun (1 task/node) → node_wrapper.sh (per-node)
       └→ conda activate, CUDA, NCCL/XLA flags → python run_train.py
```

- `train_full_autoreg.batch`: env vars, model config, data paths, srun launch, auto-resume logic
- `node_wrapper.sh`: conda env, CUDA 12.6, custom NCCL 2.29.3, AWS OFI NCCL 1.18.0, XLA flags
- `run_train.py`: argparse, `jax.distributed.initialize()`, calls `lob.train.train()`

Environment variables override model config (e.g., `D_MODEL`, `N_LAYERS`, `BLOCKS`, `SSM_SIZE_BASE`, `PER_GPU_BSZ`, `CURTAIL_EPOCHS`, `NO_VALIDATION`, `HIERARCHICAL`).

Multi-node (≥2 nodes) automatically enables hierarchical 2D mesh AllReduce.

## Key Design Decisions

- **PyTorch for data loading only** — DataLoader runs on CPU; all GPU computation is JAX/Flax
- **24-token encoding** (base-100, vocab=2112) — replaced 22-token encoding (base-10000, vocab=12012)
- **Hierarchical AllReduce** — 2D mesh splits into NVLink (intra-node) + Slingshot (inter-node); without it, 32N+ is ~2x slower
- **Custom NCCL 2.29.3** — source-built with GCC 12.3 for ARM CAS fix, loaded via LD_PRELOAD
- **Cosine annealing with warmup** — √κ LR scaling for AdamW when batch size changes
- **Mini-epochs** — split 1 data epoch into K sub-epochs for frequent eval checkpoints
- **Auto-resume** — on crash, batch script resubmits with `RESTORE_PATH` (up to 3 retries)

## Known Issue: Mid-Epoch Resume

**BUG**: When resuming from a mid-epoch checkpoint, `resume_from_step` is NOT auto-set from `state.step`. The dataloader restarts from batch 0, causing data duplication.

- **What works**: model weights, optimizer state, LR schedule (all keyed by `state.step`)
- **What breaks**: dataloader position — it replays already-seen batches
- **Root cause**: `lob/train.py:339` reads `args.resume_from_step` (always None) instead of computing `state.step % steps_per_epoch`
- **Fix needed**: auto-compute `resume_from_step = int(state.step) % steps_per_epoch` when restoring mid-epoch checkpoints. The `DistributedSampler(seed=42)` produces deterministic order per epoch, so skipping to the correct batch index is safe.
- **Curtail uses `batch_idx` not `state.step`**: so without fix, resume runs ALL curtail steps again (31k steps instead of remaining 8k), with wrong LR schedule (state.step keeps incrementing past curtail target).

## Logs and Checkpoints

- SLURM logs: `logs_lobs5/lobs5_{JOBID}.out`
- Per-node logs: `logs_lobs5/training_{JOBID}_node{N}.log`
- Checkpoints: `checkpoints/j{JOBID}_{WANDB_ID}_{JOBID}/{step}/`
- W&B project: `lobs5-scaling-law`
- Data split info: `logs_lobs5/data_split_j{JOBID}.json`

## HPC Environment

- **Platform**: ARM (Grace Hopper), 4× GPU/node, NV6 (6× NVLink bonded per GPU pair)
- **Conda**: `/projects/s5e/quant/miniforge3` (base env) — JAX 0.9.0.1, Python 3.12.11
- **Partition**: `workq`
- **Login node**: no GPU computation allowed; everything via `sbatch`
