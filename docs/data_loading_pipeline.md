# JAX/GPU Data Loading Pipeline (Multi-Node)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         JAX/GPU Data Loading Pipeline (Multi-Node)                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐     ┌─────────────┐     ┌────────────────┐     ┌─────────────┐     ┌─────────────┐
  │    DISK     │ ──▶ │  DataLoader │ ──▶ │  Pinned RAM    │ ──▶ │   VRAM      │ ──▶ │   TRAIN     │
  │  .npy/.ar   │     │ Grain/Torch │     │  (per node)    │     │  (per node) │     │  (global)   │
  └─────────────┘     └─────────────┘     └────────────────┘     └─────────────┘     └─────────────┘
        │                   │                    │                    │                   │
        ▼                   ▼                    ▼                    ▼                   ▼
   .npy/.arrayrecord   ShardOptions          pin_memory           device_put         jit train_step
   (mmap/random)      (node sharding)       (page-locked)      (CUDA async DMA)    (global sharding)


═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    MULTI-NODE GPU ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                         Global Mesh                                              │
  │   ┌───────────────────────────────┐   ┌───────────────────────────────┐                         │
  │   │        Node 0 (GPU node)      │   │        Node 1 (GPU node)      │         ...             │
  │   │  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │                         │
  │   │  │  DataLoader             │  │   │  │  DataLoader             │  │                         │
  │   │  │  shard_idx=0, count=N   │  │   │  │  shard_idx=1, count=N   │  │                         │
  │   │  │  batch = global/N       │  │   │  │  batch = global/N       │  │                         │
  │   │  └───────────┬─────────────┘  │   │  └───────────┬─────────────┘  │                         │
  │   │              │                │   │              │                │                         │
  │   │              ▼                │   │              ▼                │                         │
  │   │  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │                         │
  │   │  │  Pinned Memory (CPU)    │  │   │  │  Pinned Memory (CPU)    │  │                         │
  │   │  │  cudaHostAlloc()        │  │   │  │  cudaHostAlloc()        │  │                         │
  │   │  └───────────┬─────────────┘  │   │  └───────────┬─────────────┘  │                         │
  │   │              │ CUDA DMA       │   │              │ CUDA DMA       │                         │
  │   │       ┌──────┴──────┐         │   │       ┌──────┴──────┐         │                         │
  │   │       ▼      ▼      ▼         │   │       ▼      ▼      ▼         │                         │
  │   │    [GPU0] [GPU1] [GPU2]...    │   │    [GPU4] [GPU5] [GPU6]...    │                         │
  │   └───────────────────────────────┘   └───────────────────────────────┘                         │
  │                                                                                                  │
  │   jax.make_array_from_single_device_arrays() ───▶ Global jax.Array (logically unified)          │
  │                           or                                                                     │
  │   jax.pmap() with axis_name ───▶ Replicated across all GPUs                                     │
  └─────────────────────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    DATA TRANSFER FLOW (GPU)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
  │  np.ndarray    │     │  Pinned Mem    │     │  jax.Array     │     │  jax.Array     │
  │  (local batch) │ ──▶ │  (page-locked) │ ──▶ │  (per GPU)     │ ──▶ │  (global)      │
  │  CPU RAM       │     │  non-swappable │     │  local GPUs    │     │  all GPUs      │
  └────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘
                              cudaMallocHost      CUDA DMA async        make_array_from_
                              np.copyto()         device_put()          single_device_arrays()

  Key code:
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  # 1. Pin memory for faster DMA (if using custom loader)                                         │
  │  pinned_array = jax.device_put(array, jax.devices('cpu')[0])  # or use cuda.pinned_memory()      │
  │                                                                                                   │
  │  # 2. Split and transfer to local GPUs                                                           │
  │  local_device_arrays = np.split(array, len(jax.local_devices()), axis=0)                         │
  │  local_device_buffers = jax.device_put(local_device_arrays, jax.local_devices())                 │
  │                                                                                                   │
  │  # 3. Form global array (multi-node)                                                             │
  │  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]                        │
  │  return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)   │
  └──────────────────────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    JAX/GPU vs PyTorch DataLoader
═══════════════════════════════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────┬─────────────────────────────────┐
  │      PyTorch DataLoader         │      JAX/GPU DataLoader          │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │  num_workers=N                  │  grain_worker_count=N           │
  │  (subprocess parallel load)     │  (or Python multiprocessing)    │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │  pin_memory=True                │  pin_memory still useful!       │
  │  (page-locked for CUDA DMA)     │  jax.device_put from pinned     │
  │                                  │  memory is faster               │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │  prefetch_factor=M              │  prefetch_buffer_size=M         │
  │  (prefetch M batches)           │  (in ReadOptions)               │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │  DistributedSampler             │  ShardOptions(idx, count)       │
  │  (manual config)                │  (Grain native support)         │
  │                                  │  or manual data[idx::count]     │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │  .to(device, non_blocking=True) │  jax.device_put(array, device)  │
  │  (async H2D transfer)           │  (auto async CUDA DMA)          │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │  DataParallel / DDP             │  jax.pmap / jax.jit + sharding  │
  │  (needs NCCL init)              │  (needs jax.distributed.init)   │
  └─────────────────────────────────┴─────────────────────────────────┘
```

## GPU-specific Notes

1. **pin_memory still important**: GPU's CUDA DMA transfers faster from pinned memory
2. **NCCL communication**: JAX multi-node GPU needs `jax.distributed.initialize()`
3. **NVLink/InfiniBand**: Cross-node communication uses high-speed interconnect

## References

- MaxText: `maxtext/src/MaxText/multihost_dataloading.py`
- MaxDiffusion: `maxdiffusion/src/maxdiffusion/multihost_dataloading.py`
- Google Grain: https://google-grain.readthedocs.io/en/stable/tutorials/data_loader_tutorial.html
