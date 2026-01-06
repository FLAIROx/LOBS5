# G1 AOT Compilation Design Analysis

**Subagent ID**: a881b1b
**Date**: 2026-01-06
**Task**: Design AOT compilation for eval_batch in es_trainer.py

---

## 1. Current Implementation Analysis

### Key Locations:
- **train_epoch**: Lines 873-920
- **eval_single_thread**: Lines 856-871
- **simulate_episode**: Lines 554-854
- **create_*_common_params**: Lines 527-552

### Current Problem (Lines 887-895):
```python
eval_fn = partial(
    self.eval_single_thread,
    epoch=epoch,
    initial_sim_state=initial_sim_state,
    initial_msg_history=initial_msg_history,
)
fitnesses, infos = jax.vmap(eval_fn)(keys, thread_ids)
```

**Issues:**
1. `eval_single_thread` captures `self`
2. No explicit `.lower().compile()`
3. Epoch-dependent params trigger tracing

---

## 2. Parameters Classification

### Static (captured in closure):
- `self.noiser_cls`
- `self.frozen_noiser_params`
- `self.es_tree_key`
- `self.lobs5_init.frozen_params`
- `self.config`
- `self.sim.process_order_array`
- `self.encoder`
- `self.replay_tokens`, `self.replay_data_raw`

### Dynamic (function args):
- `noiser_params` - updated each epoch
- `params` - model weights, updated
- `key` - different each epoch
- `epoch` (int)
- `thread_ids`
- `initial_sim_state`
- `initial_msg_history`

---

## 3. Proposed Code Changes

### A. Add `_build_eval_thread` (after line 552):

```python
def _build_eval_thread(self):
    """Build pure eval function for AOT compilation."""
    # Capture static references
    noiser_cls = self.noiser_cls
    frozen_noiser_params = self.frozen_noiser_params
    es_tree_key = self.es_tree_key
    frozen_params = self.lobs5_init.frozen_params
    config = self.config
    CommonParams = _get_common_params()

    def eval_thread(noiser_params, params, key, thread_id, epoch,
                    initial_sim_state, initial_msg_history):
        print("Compiling eval_thread")  # Tracing indicator

        iterinfo = (jnp.int32(epoch), jnp.int32(thread_id))
        policy_common_params = CommonParams(
            noiser=noiser_cls,
            frozen_noiser_params=frozen_noiser_params,
            noiser_params=noiser_params,
            params=params,
            es_tree_key=es_tree_key,
            frozen_params=frozen_params,
            iterinfo=iterinfo,
        )
        # ... inline simulate_episode logic
        return fitness, info

    return eval_thread
```

### B. Add `_compile_eval_batch` (after `_build_eval_thread`):

```python
def _compile_eval_batch(self):
    """AOT compile using .lower().compile() pattern."""
    from jax import ShapeDtypeStruct

    n_threads = self.config.n_threads
    _eval_thread = self._build_eval_thread()

    # Create ShapeDtypeStruct examples
    keys_example = ShapeDtypeStruct((n_threads, 2), jnp.uint32)
    thread_ids_example = ShapeDtypeStruct((n_threads,), jnp.int32)
    epoch_example = ShapeDtypeStruct((), jnp.int32)

    initial_sim_state, initial_msg_history = self._create_initial_sim_state()
    sim_state_example = jax.tree.map(
        lambda x: ShapeDtypeStruct(x.shape, x.dtype), initial_sim_state
    )

    # AOT compile
    eval_batch = jax.jit(
        jax.vmap(_eval_thread, in_axes=(None, None, 0, 0, None, None, None))
    ).lower(
        self.noiser_params, self.lobs5_init.params,
        keys_example, thread_ids_example, epoch_example,
        sim_state_example, initial_msg_history,
    ).compile()

    print(eval_batch.memory_analysis())
    return eval_batch
```

### C. Modify `__init__` (after line 376):
```python
print("[INIT] AOT compiling eval_batch...")
self._compiled_eval_batch = self._compile_eval_batch()
```

### D. Modify `train_epoch` (replace lines 887-895):
```python
fitnesses, infos = self._compiled_eval_batch(
    self.noiser_params, self.lobs5_init.params,
    keys, thread_ids, jnp.int32(epoch),
    initial_sim_state, initial_msg_history,
)
```

---

## 4. Potential Issues

| Issue | Problem | Solution |
|-------|---------|----------|
| LobState complexity | Pytree structure | Use `jax.tree.map` for ShapeDtypeStruct |
| simulate_episode self refs | Already addressed | G5/G2 extraction completed |
| Background mode branching | Conditional logic | Captured in closure at compile time |
| Initial state variability | Shape changes = recompile | Ensure fixed shapes |

---

## 5. Implementation Risk

| Component | Risk | Notes |
|-----------|------|-------|
| _build_eval_thread | Medium | Need to inline simulate_episode |
| _compile_eval_batch | Low | Standard pattern |
| train_epoch modification | Low | Simple replacement |
| LobState ShapeDtypeStruct | Medium | Pytree complexity |

---

## 6. Expected Outcome

After G1 implementation:
- First epoch: ~150s (one-time AOT compile)
- Subsequent epochs: ~10-20s (no recompilation)
- Program restart: ~10s (cache hit from G4)
