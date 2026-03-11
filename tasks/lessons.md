# J2 Muon Optimizer Experiment — Lessons Learned

## 2026-03-11: Manual cross-node resume requires RESUME_FROM_STEP

**Context**: Resuming 8N checkpoint on 32N via manual `sbatch` with `RESTORE_PATH=...`.

**What went wrong**: Forgot to set `RESUME_FROM_STEP=15115`. The auto-resume code
in `train_full_autoreg.batch` calculates this automatically on crash/timeout resubmit,
but manual `sbatch` requires it explicitly.

**Impact**: Model re-sees first 7% of epoch data. LR schedule was correctly remapped
by `remap_train_state_step()`, so optimizer state is fine. Minor data overlap.

**Fix for next time**: When manually resuming with a different node count:
```bash
# 1. Read checkpoint metadata for step_in_epoch
# 2. Compute: RESUME_FROM_STEP = step_in_epoch * old_gBSZ / new_gBSZ
# 3. Pass both:
RESTORE_PATH=<ckpt_dir> RESUME_FROM_STEP=<remapped_step> sbatch --nodes=N ...
```

## 2026-03-11: node_wrapper.sh redirects stdout — check per-node logs

**Context**: 32N job appeared "hung" with zero Python output for 35 min.

**What went wrong**: `node_wrapper.sh` line 16 does `exec > training_<jobid>_node0.log 2>&1`,
redirecting ALL Python/training output to per-node log files. The sbatch stdout
(`lobs5_<jobid>.out`) only gets the bash header from `train_full_autoreg.batch`.

**Impact**: Killed a perfectly healthy job (j2730490, step 436, 2.93 s/it) because we
were reading the wrong log file.

**Fix**: Always check `logs_lobs5/training_<jobid>_node0.log` for training output,
not `logs_lobs5/lobs5_<jobid>.out`.
