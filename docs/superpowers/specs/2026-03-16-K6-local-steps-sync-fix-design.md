# K6: Fix Local Steps Cross-Node Sync — Design Spec

## Problem

Local Steps mode (`local_steps_k > 0`) is supposed to skip cross-node
AllReduce for K-1 out of K steps, reducing Slingshot traffic by ~90%.
However, JAX `lax.cond` executes **both branches** unconditionally in XLA's
static computation graph, so `pmean(params, 'nodes')` runs every step.
Zero communication savings.

### History

- **Feb 27 (9050a8eb)**: First fix — two complete train step functions
  (`local_fn` + `sync_fn`), switched at Python level.
- **Feb 28 (779fcf75)**: **Reverted** — sync_fn caused GPU OOM (~57 GiB)
  at 256N. Root cause: "separate JIT functions prevent XLA from reusing
  memory across forward/backward/allreduce phases."
- **Mar 16 (51c3ef3f)**: New fix — minimal sync_fn that ONLY does
  `pmean(params, 'nodes')`, no forward/backward. Committed to shard-map.

## Design: Two Approaches (A/B Test)

### K6a: Minimal sync_fn (low risk)

**Architecture:**
```
Every step:  local_step_fn → forward + backward + pmean('gpus') + apply_grads
Every K:     sync_params_fn → ONLY pmean(params, 'nodes')
```

- `sync_params_fn` is a tiny separate `shard_map` + `jax.jit` function
- Called from Python loop AFTER `local_step_fn`, not instead of it
- Memory: only params allreduce buffer (no forward/backward activations)
- Add `donate_argnums=(0,)` to sync_params_fn to eliminate duplicate params
- MEM_FRACTION: keep current defaults (0.80 for 32+N)

**Code changes** (from shard-map commit 51c3ef3f):
1. `_create_hierarchical_train_step`: remove `pmean+lax.cond`, return dict
   `{step_fn, sync_fn, K, mode}` when `local_steps_k > 0`
2. `train_epoch`: detect dict, Python-level `if counter >= K: sync()`
3. `sync_params_fn = jax.jit(shard_map(pmean('nodes')), donate_argnums=(0,))`

### K6b: Dual full functions + lower MEM_FRACTION (high risk, control)

**Architecture:**
```
Steps 1~K-1: local_fn  → forward + backward + pmean('gpus') + apply_grads
Step K:      sync_fn   → forward + backward + pmean('gpus') + apply_grads
                          + pmean(params, 'nodes')
```

- Reproduces Feb 27 approach (9050a8eb) with mitigations:
  - Lower MEM_FRACTION: 0.70 for 32+N (from 0.80)
  - `donate_argnums=(0,)` on both functions
  - Shared `_forward_backward()` helper to maximize XLA cache reuse
- Two full XLA compilations (~60-120s each)
- Risk: XLA greedy allocation may still OOM at scale

**Code changes** (restore from 9050a8eb + mitigations):
1. `_create_local_steps_train_step`: two shard_map functions
2. Shared `_forward_backward()` closure
3. MEM_FRACTION tiers: 32+N → 0.70, 8-16N → 0.75, else 0.85
4. `train_epoch`: tuple detection `(local_fn, sync_fn, K)`

## Validation Plan

Both worktrees tested identically:

| Phase | Config | Purpose |
|-------|--------|---------|
| 1. Syntax | `python -c "import ast; ..."` | Parse check |
| 2. Functional | 2N, CURTAIL_EPOCHS=50, BSZ=10 | Verify training works |
| 3. Scale | 16N or 32N, CURTAIL_EPOCHS=300 | Verify no OOM at scale |

Success = no OOM + cross-node comm only at step K (verify via NCCL debug logs or profiler).

## Worktree Layout

| Worktree | Branch | Base |
|----------|--------|------|
| `experiments/exp_K6a_local_sync_minimal` | `exp/K6a-local-sync-minimal` | shard-map@51c3ef3f |
| `experiments/exp_K6b_local_sync_dual` | `exp/K6b-local-sync-dual` | shard-map@51c3ef3f |
