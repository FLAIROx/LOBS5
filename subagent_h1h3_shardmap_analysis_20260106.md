# Multi-GPU shard_map Implementation Design (H1-H3)

**Subagent ID**: aa2d5f2
**Date**: 2026-01-06
**Task**: Design shard_map multi-GPU implementation for es_trainer.py

---

## 1. Current vmap Usage

### Primary Location: Line 895
```python
fitnesses, infos = jax.vmap(eval_fn)(keys, thread_ids)
```

### Data Flow:
```
Inputs:
  - keys: (n_threads, 2) - PRNGKeys for each thread
  - thread_ids: (n_threads,) - Thread indices for ES perturbation
  - epoch: scalar
  - initial_sim_state: LobState (broadcast)
  - initial_msg_history: (context_len,) (broadcast)

Outputs:
  - fitnesses: (n_threads,)
  - infos: Dict[str, (n_threads,)]
```

---

## 2. Proposed Mesh Configuration (H3)

```python
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

n_devices = len(jax.devices())  # Expected: 4 GPUs
mesh = jax.make_mesh((n_devices,), ('data',))
```

---

## 3. PartitionSpecs

### in_specs:
```python
in_specs=(
    P('data'),    # keys: sharded
    P('data'),    # thread_ids: sharded
    P(),          # epoch: replicated
    P(),          # sim_state: replicated
    P(),          # msg_history: replicated
)
```

### out_specs:
```python
out_specs=(
    P('data'),    # fitnesses: sharded
    P('data'),    # infos: sharded
)
```

---

## 4. Code Changes

### A. Add Imports (line ~48):
```python
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
```

### B. Add Mesh in `__init__` (line ~377):
```python
self.n_devices = len(jax.devices())
self.mesh = jax.make_mesh((self.n_devices,), ('data',))
```

### C. Add Helper Functions:
```python
def _shard_to_mesh(self, x):
    return jax.device_put(x, NamedSharding(self.mesh, P('data')))
```

### D. Modify `train_epoch` (line 895):

Replace:
```python
fitnesses, infos = jax.vmap(eval_fn)(keys, thread_ids)
```

With:
```python
keys_sharded = self._shard_to_mesh(keys)
thread_ids_sharded = self._shard_to_mesh(thread_ids)

sharded_eval = shard_map(
    jax.vmap(_eval_thread, in_axes=(0, 0, None, None, None)),
    mesh=self.mesh,
    in_specs=(P('data'), P('data'), P(), P(), P()),
    out_specs=(P('data'), P('data')),
)

fitnesses, infos = sharded_eval(
    keys_sharded, thread_ids_sharded,
    epoch, initial_sim_state, initial_msg_history
)
```

---

## 5. Potential Issues

1. **n_threads % n_devices != 0**: Add validation check
2. **Self references in JIT**: Already handled by G5/G2
3. **LobState sharding**: Use P() for broadcast
4. **Info dict sharding**: JAX handles pytree automatically

---

## 6. Implementation Steps

1. **H3**: Add `jax.make_mesh()` in `__init__`
2. **H1**: Replace `jax.vmap(eval_fn)` with `shard_map(jax.vmap(...))`
3. **H2**: Verify with I4 multi-GPU test
