# GPU Optimization Code Changes

本文档包含所有优化的详细代码修改。每个优化独立应用和测试。

---

## Optimization 1: Historical Replay 批量加载

**文件**: `es_lobs5/training/es_trainer.py`
**位置**: `step_fn` 函数内部 (约 line 1500)

### 修改前 (当前代码)

```python
# Select background generation function
n_warmup_cfg = getattr(config, 'n_warmup_msgs', 500)
if config.background_mode == 'historical_replay':
    step_fn_background = historical_replay_step
    replay_ptr_init = jnp.int32(n_warmup_cfg + step_idx * config.background_msgs_per_step)
else:
    step_fn_background = world_model_step
    replay_ptr_init = jnp.int32(0)

# Generate background messages
# H2: Apply pvary to initial carry values when inside shard_map
background_scan_init = (
    maybe_pvary(key_world),
    maybe_pvary(msg_history),
    maybe_pvary_tree(hiddens_world),
    maybe_pvary_tree(sim_state),
    maybe_pvary(book_feat),
    maybe_pvary(world_oid_offset),
    maybe_pvary(replay_ptr_init),
)
(key_world, msg_history, hiddens_world, sim_state, book_feat,
 world_oid_offset, _), _ = jax.lax.scan(
    step_fn_background,
    background_scan_init,
    jnp.arange(config.background_msgs_per_step),
    length=config.background_msgs_per_step,
)
```

### 修改后 (优化代码)

```python
# Select background generation function
n_warmup_cfg = getattr(config, 'n_warmup_msgs', 500)
if config.background_mode == 'historical_replay':
    step_fn_background = historical_replay_step_batched
    replay_ptr_init = jnp.int32(n_warmup_cfg + step_idx * config.background_msgs_per_step)

    # PRE-BATCH: Slice all background messages before scan (optimization)
    # This eliminates dynamic indexing inside the scan loop
    bg_start = replay_ptr_init
    bg_end = bg_start + config.background_msgs_per_step
    # Use jax.lax.dynamic_slice for JAX-traceable slicing
    bg_tokens_batch = jax.lax.dynamic_slice(
        replay_tokens,
        (bg_start, 0),
        (config.background_msgs_per_step, msg_len)
    )
    bg_raw_batch = jax.lax.dynamic_slice(
        replay_data_raw,
        (bg_start, 0),
        (config.background_msgs_per_step, replay_data_raw.shape[1])
    )
else:
    step_fn_background = world_model_step
    replay_ptr_init = jnp.int32(0)
    bg_tokens_batch = None
    bg_raw_batch = None

# Generate background messages
# H2: Apply pvary to initial carry values when inside shard_map
background_scan_init = (
    maybe_pvary(key_world),
    maybe_pvary(msg_history),
    maybe_pvary_tree(hiddens_world),
    maybe_pvary_tree(sim_state),
    maybe_pvary(book_feat),
    maybe_pvary(world_oid_offset),
    maybe_pvary(replay_ptr_init),
)
(key_world, msg_history, hiddens_world, sim_state, book_feat,
 world_oid_offset, _), _ = jax.lax.scan(
    step_fn_background,
    background_scan_init,
    jnp.arange(config.background_msgs_per_step),
    length=config.background_msgs_per_step,
)
```

### 新增函数 `historical_replay_step_batched`

```python
def historical_replay_step_batched(wcarry, bg_msg_idx):
    """Load pre-batched messages from historical data (optimized).

    Uses pre-sliced bg_tokens_batch and bg_raw_batch instead of
    dynamic indexing into replay_tokens/replay_data_raw.
    """
    key, msg_hist, hidden, sim_st, book_f, oid_offset, replay_ptr = wcarry

    # Use pre-batched data (captured from closure)
    replayed_msg_tokens = bg_tokens_batch[bg_msg_idx]
    replayed_msg_raw = bg_raw_batch[bg_msg_idx]

    sim_msg = msg_to_jnp(replayed_msg_raw)
    bg_order_id = WORLD_ORDER_ID_START + oid_offset
    sim_msg = sim_msg.at[4].set(bg_order_id)
    sim_msg = sim_msg.at[5].set(HISTORICAL_TRADER_ID)

    sim_st = process_order_array(sim_st, sim_msg)
    book_f = transform_L2_state_wrapper(jaxlob_cfg, sim_st, price_levels=book_depth, tick_size=config.tick_size, in_shard_map=in_shard_map)
    msg_hist = jnp.concatenate([msg_hist[msg_len:], replayed_msg_tokens])

    oid_offset = oid_offset + 1
    # replay_ptr not used in batched version, but kept for API compatibility
    new_replay_ptr = replay_ptr + 1

    return (key, msg_hist, hidden, sim_st, book_f, oid_offset, new_replay_ptr), replayed_msg_tokens
```

---

## Optimization 2: BF16 混合精度

**文件**: `es_lobs5/scripts/es_training.sh`

### 修改 (添加环境变量)

```bash
# 在 Environment Setup 部分添加:
export USE_BF16=1
```

**注意**: Flax model 已有 BF16 支持，只需设置环境变量启用。

---

## Optimization 3: 静态形状优化

**文件**: `es_lobs5/training/es_trainer.py`
**位置**: `sample_policy_token_flax` 内部和 `historical_replay_step`

### 修改点 1: msg_hist 更新

```python
# 修改前:
msg_hist = jnp.concatenate([msg_hist[msg_len:], replayed_msg_tokens])

# 修改后:
msg_hist = jnp.roll(msg_hist, -msg_len, axis=0)
msg_hist = msg_hist.at[-msg_len:].set(replayed_msg_tokens)
```

### 修改点 2: token 生成中的 msg_hist 更新

```python
# 修改前 (在 sample_policy_token_flax):
msg_hist_p = jnp.concatenate([msg_hist_p[1:], next_token_p])

# 修改后:
msg_hist_p = jnp.roll(msg_hist_p, -1, axis=0)
msg_hist_p = msg_hist_p.at[-1].set(next_token_p[0])
```

---

## Optimization 4: XLA 环境变量

**文件**: `es_lobs5/scripts/es_training.sh` 和 `benchmark_es_training.sh`

### 修改 (添加 XLA flags)

```bash
# 在 Environment Setup 部分添加:
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=true --xla_gpu_triton_gemm_any=true"
```

---

## Optimization 5: Donate Arguments

**文件**: `es_lobs5/training/es_trainer.py`
**位置**: `_compile_eval_batch` 方法

### 修改

```python
# 修改前 (line ~1164):
compiled_eval = jax.jit(sharded_eval)

# 修改后:
compiled_eval = jax.jit(
    sharded_eval,
    donate_argnums=(2, 3)  # Donate keys and thread_ids
)
```

---

## 应用顺序

1. **Optimization 1**: Historical Replay 批量加载 → 测试
2. **Optimization 3**: 静态形状优化 → 测试
3. **Optimization 4**: XLA 环境变量 → 测试
4. **Optimization 2**: BF16 混合精度 → 测试
5. **Optimization 5**: Donate Arguments → 测试

每个优化应用后:
1. 提交 commit
2. 运行 benchmark 脚本
3. 记录结果到 GPU_OPTIMIZATION_BENCHMARK.md
4. 与基准对比
