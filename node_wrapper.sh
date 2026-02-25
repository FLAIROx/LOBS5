#!/bin/bash
# ========== Node Wrapper ==========
# Called by srun with --export=ALL: all env vars from batch script are available.
# 1 process per node, all GPUs visible.

# TMPDIR: must be set BEFORE any Python import (wandb creates tempdir at import time)
export TMPDIR=/tmp

# Force wandb online mode (directory-level "offline" setting overrides USE_WANDB=True)
export WANDB_MODE=online

echo "========================================"
echo "[Wrapper] Running on node: $(hostname)"
echo "[Wrapper] SLURM_NODEID: ${SLURM_NODEID:-N/A}"
echo "[Wrapper] SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "[Wrapper] SLURM_LOCALID: ${SLURM_LOCALID:-N/A}"
echo "[Wrapper] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# Activate conda env directly (conda.sh has hardcoded broken paths to kangli's home)
CONDA_ENV=${CONDA_ENV:-base}
if [ "$CONDA_ENV" = "base" ]; then
  export CONDA_PREFIX=/projects/s5e/quant/miniforge3
else
  export CONDA_PREFIX=/projects/s5e/quant/miniforge3/envs/$CONDA_ENV
fi
export PATH=$CONDA_PREFIX/bin:$PATH
echo "[Wrapper] Conda env: $CONDA_ENV ($CONDA_PREFIX)"
echo "[Wrapper] Python: $(which python) ($(python --version 2>&1))"

# Load CUDA module
module load cuda/12.6

# Set LD_LIBRARY_PATH
# NCCL override: use lobmax NCCL 2.29.2 (fixes ARM CAS weak failure hang, fixed in 2.29.x)
# The lob env has NCCL 2.28.9 which has the ARM CAS bug on GH200 ARM platform.
# CAVEAT: previous override pointed to ~/miniforge3/lib/... (base env) which DID NOT EXIST,
#         so all 12+ failed jobs loaded 2.28.9 despite the "override". Fixed 2026-02-23 (E1).
NCCL_LIB_OVERRIDE=/projects/s5e/quant/miniforge3/envs/lobmax/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$NCCL_LIB_OVERRIDE:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# NCCL OFI plugin for cross-node communication via Slingshot/libfabric
export LD_LIBRARY_PATH=/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo/lib:/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH

# Verify NCCL version (must be 2.29.x, NOT 2.28.x)
echo "[NCCL] Override lib path: $NCCL_LIB_OVERRIDE"
echo "[NCCL] Library: $(ls -la $NCCL_LIB_OVERRIDE/libnccl.so.2 2>/dev/null || echo NOT_FOUND)"
strings $NCCL_LIB_OVERRIDE/libnccl.so.2 2>/dev/null | grep "^NCCL version" | head -1 || echo "[NCCL] WARNING: Could not determine NCCL version"

# JAX environment
export XLA_PYTHON_CLIENT_PREALLOCATE=true
# 2D mesh (hierarchical) creates two NCCL communicators (gpus+nodes), more buffer needed
# 32+ nodes: 0.80 (XLA HLO planner is greedy — 0.85 just makes it allocate MORE, not leave headroom)
#   Job 2439132: MEM=0.80 → XLA requested 73.5 GiB (96.7% of 76 GB) → OOM epoch 11
#   Job 2439639: MEM=0.85 → XLA requested 78.7 GiB (96.4% of 81.6 GB) → OOM step 0
#   Fix: reduce PER_GPU_BSZ (12→10), not increase MEM_FRACTION
# 8-16N: 0.85 (moderate overhead)
# Override with MEM_FRACTION env var for BSZ tuning (e.g. MEM_FRACTION=0.85)
if [ -z "${MEM_FRACTION}" ]; then
  if [ "${NNODES}" -ge 32 ] && [ "${HIERARCHICAL}" = "True" ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
  elif [ "${NNODES}" -ge 8 ] && [ "${HIERARCHICAL}" = "True" ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  else
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
  fi
else
  export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}
fi
export JAX_PLATFORMS="cuda"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# CAVEAT — XLA FLAGS FOR MULTI-HOST AUTOTUNER
# autotune_level=0 is FORBIDDEN — XLA AutoTune (kernel fusion) is why we use JAX
#
# JAX 0.9.0 had multi-host autotuner crash: autotuner.cc:260 DEVICE_TYPE_INVALID
# Root cause: non-deterministic iteration + unsorted sharding across hosts
# Fixed in JAX 0.9.0.1 via XLA#36579 + XLA#36755
# CAVEAT: DO NOT re-enable these flags unless downgrading below JAX 0.9.0.1
# CAVEAT: ssm-stable, HyperscaleES, MaxText all use XLA defaults (no flags)
# Rollback to JAX 0.9.0: pip install jax==0.9.0 jaxlib==0.9.0 jax-cuda12-pjrt==0.9.0 jax-cuda12-plugin==0.9.0
# export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false --xla_gpu_shard_autotuning=false}"
# Force all CUDA modules to load at startup (vs LAZY default which loads on first use).
# Why EAGER: in multi-node training, lazy loading causes non-deterministic load timing
# across nodes → NCCL collective timeouts and XLA autotuner device-binding races.
# Trade-off: slightly slower startup, but eliminates mid-training CUDA load stalls.
# XLA AllReduce fusion for multi-node scaling (ref: MaxText GPU config)
# Without this, XLA creates 101 independent AllReduce ops (one per param tensor).
# 16-rank ring on 4N has 91ms/op launch latency → 101*91ms = 9.2s per step.
# With 128MB threshold, XLA combines into 2-3 large AllReduces → ~3*91ms = 0.27s.
# Profiler evidence: jobs 2381959(2N) vs 2381960(4N), 101 ops/step both configs.
# Latency hiding scheduler overlaps remaining AllReduce with compute.
export XLA_FLAGS="${XLA_FLAGS} \
  --xla_gpu_all_reduce_combine_threshold_bytes=134217728 \
  --xla_gpu_enable_latency_hiding_scheduler=true \
  --xla_gpu_enable_highest_priority_async_stream=true \
  --xla_gpu_nccl_terminate_on_error=true \
  --xla_gpu_nccl_termination_timeout_seconds=600 \
  --xla_gpu_first_collective_call_terminate_timeout_seconds=600"
# NOTE (G11): xla_gpu_first_collective_call_terminate_timeout_seconds is the correct flag
# for thunk init rendezvous timeout. xla_gpu_executable_terminate_timeout_seconds does NOT
# exist in JAX 0.9.0.1 (FATAL: Unknown flag, Job 2476156).
# CAVEAT: do NOT use --xla_gpu_all_reduce_blueconnect_num_devices_per_host=4 with shard_map
# BlueConnect decomposes AllReduce into RS+AR+AG, but shard_map already does 2-level decomposition.
# Result: 3x slowdown (3.55 s/step vs baseline 1.18 s/step). Verified job 2440967.

# 8+ nodes with shard_map: disable shard_autotuning
# The old CAVEAT (job 2421931: JIT >30min at 16N) was for 1D DDP without shard_map.
# With shard_map 2D mesh, shard_autotuning adds overhead: 16N 1.58→? s/step (testing).
# 32N with autotuning off: 0.94 s/step (58.4% eff); 16N with autotuning on: 1.58 s/step (34.8% eff).
if [ "${NNODES}" -ge 8 ] && [ "${HIERARCHICAL}" = "True" ]; then
  export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_shard_autotuning=false"
  echo "[XLA] ${NNODES}N hierarchical: shard_autotuning disabled"
fi

export CUDA_MODULE_LOADING=EAGER
echo "[XLA] XLA_FLAGS=${XLA_FLAGS}"

# NCCL config
# NCCL_TIMEOUT=3600 did NOT prevent a 6h hang at 32N (job 2426449, epoch 14 step 356).
# Multiple timeout mechanisms for broader NCCL version compatibility:
export NCCL_TIMEOUT=600                     # 10 min (NCCL 2.19+, seconds)
export NCCL_BLOCKING_WAIT=0                 # Non-blocking wait mode (NCCL 2.x)
export NCCL_ASYNC_ERROR_HANDLING=1           # Async error handling (NCCL 2.13+)
export NCCL_LAUNCH_ORDER_IMPLICIT=1         # Implicit ordering for multi-communicator ops (NCCL 2.26+)

# NCCL_BUFFSIZE=2MB — CRITICAL for multi-node shard_map performance
# Experimentally verified (C5a, 2026-02-21):
#   16N + BUFF=default(4MB): 1.58 s/step (34.8% eff) — Jobs 2424845, 2425406, 2426301
#   16N + BUFF=2MB:          0.66 s/step (83.3% eff) — Job 2426303  → 2.4x faster!
#   32N + BUFF=2MB:          0.94 s/step (58.5% eff) — Job 2424846
# With 40 NCCL channels (default on GH200), 4MB per channel saturates Slingshot
# injection bandwidth. 2MB reduces per-chunk size, improving pipeline efficiency.
# This is the sole cause of the previous 16N<32N efficiency anomaly.
if [ "${NNODES}" -ge 2 ]; then
  export NCCL_BUFFSIZE=2097152
  echo "[NCCL] Multi-node: NCCL_BUFFSIZE=2MB (verified 2.4x speedup at 16N)"
fi

# Isambard GH200 Slingshot best practices (ref: docs.isambard.ac.uk, NCCL Issue #1272)
# Experimentally: no measurable impact alone (1.60 vs 1.58 s/step), but recommended.
export NCCL_MIN_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=4
# CAVEAT — DO NOT re-enable NCCL_P2P_DISABLE=1
# GH200 nodes HAVE NV6 (6x NVLink bonded, 478 GB/s) between all 4 GPUs.
# P2P_DISABLE=1 forces intra-node comm to SHM (CPU memcpy, ~50 GB/s),
# causing 25x slowdown on 4N+ (11.2 s/step vs 0.46 s/step).
# 2N is unaffected because each NCCL comm has localRanks=1 (no intra-node comm).
# Verified: nvidia-smi topo -m shows NV6 between all GPU pairs.
#export NCCL_P2P_DISABLE=1

# NCCL collective algorithm — both TREE and RING are slow on 4N Slingshot
# NCCL_ALGO=TREE: crash, AllGather does not support TREE — job 2382073
# NCCL_ALGO=allreduce:tree: 21.83 s/step, TREE 2.2x SLOWER than RING — job 2382095
# NCCL_ALGO default RING: 9.78 s/step — job 2381259
# Root cause is XLA AllReduce combine threshold (see XLA_FLAGS above), not NCCL algo.
#export NCCL_ALGO="allreduce:tree"

# Slingshot/CXI optimization — PERFORMANCE tuning
# TESTED job 2382150: 13.36 s/step — 37% WORSE than baseline (9.78).
# Root cause of regression: NCCL_CROSS_NIC=1 + NCCL_NET_GDR_LEVEL=PHB, NOT the CXI hang-prevention vars.
# These two are harmful on our Slingshot topology. Do not re-enable without single-variable benchmarking.
#export NCCL_CROSS_NIC=1                   # CAVEAT: tested harmful (job 2382150)
#export NCCL_NET_GDR_LEVEL=PHB             # CAVEAT: tested harmful (job 2382150)
#export NCCL_PROTO=^LL128                  # CAVEAT: tested harmful — 23% regression (1.12 vs 0.91 s/step, job 2447647 vs 2447130)
#export FI_CXI_DEFAULT_CQ_SIZE=131072      # TODO: test separately
#export FI_CXI_DEFAULT_TX_SIZE=16384       # TODO: test separately
#export FI_CXI_RX_MATCH_MODE=software      # TODO: test separately (may affect small message perf)

# Slingshot CXI resilience — prevent NCCL deadlocks at 8+ nodes
# These are RESILIENCE tuning — no normal-path performance impact, only fault recovery.
# Root cause analysis (2026-02-23, HLO profiling + deep research):
#   - eval_step has ZERO NCCL collectives (HLO verified, Job 2440089)
#   - train_step has 336 AllReduce/step (169 intra-node + 167 inter-node)
#   - 32N × 336 × 549 steps/epoch × 6 epochs = ~1.1M collectives before hang
#   - NCCL 2.29.3 fixes ARM CAS weak failure (exactly our GH200 ARM platform)
#   - CXI eager message race condition is CSCS+Isambard+ALCF consensus workaround
if [ "${SLURM_NNODES:-1}" -ge 8 ]; then
  # --- Existing resilience ---
  export FI_CXI_RDZV_RETRIES=100           # default=5, survive transient Slingshot fabric errors
  export FI_CXI_OFLOW_BUF_SIZE=8388608     # 8MB overflow buffer (prevent CXI ENOMEM under bursty traffic)
  export FI_CXI_REQ_BUF_SIZE=8388608       # 8MB request buffer (reduce flow control stalls)

  # --- NEW: CXI hang prevention (CSCS + Isambard + ALCF consensus) ---
  # Disable eager messages to prevent CXI race condition under high concurrency.
  # CXI eager path buffer management has race condition at 128 GPU bursty traffic.
  # Setting all three to 0 forces rendezvous-only path for all message sizes.
  export FI_CXI_RDZV_GET_MIN=0             # Disable eager GET minimum size
  export FI_CXI_RDZV_THRESHOLD=0           # Force all messages through rendezvous
  export FI_CXI_RDZV_EAGER_SIZE=0          # No eager data in rendezvous
  export FI_CXI_RDZV_PROTO=alt_read        # Alternate read protocol (ALCF verified up to 540 nodes)

  # --- NEW: Host register deadlock prevention (Isambard docs) ---
  # Multi-process per GPU → page locking competition → deadlock
  export FI_CXI_DISABLE_HOST_REGISTER=1    # Prevent host buffer GPU registration deadlock
  export FI_MR_CACHE_MONITOR=userfaultfd   # Memory registration cache monitor

  # --- NEW: Prevent GPU-aware MPI + NCCL collision (CSCS docs) ---
  export MPICH_GPU_SUPPORT_ENABLED=0       # "easily leads to deadlocks" - CSCS

  echo "[CXI] ${SLURM_NNODES}N: Full CXI resilience (RDZV_RETRIES=100, eager=off, alt_read, host_reg=off)"
fi

# Multi-node JAX distributed info
echo "[Wrapper] SLURM_PROCID=${SLURM_PROCID:-0} (process rank)"
echo "[Wrapper] SLURM_NNODES=${SLURM_NNODES:-1} (total nodes)"
echo "[Wrapper] JAX_COORDINATOR_ADDRESS=${JAX_COORDINATOR_ADDRESS:-none}"
# NCCL debug for cross-node verification (set to WARN after verified)
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# CUDA config
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$HOME/.nv/ComputeCache" || true

echo "[Wrapper] Available GPUs:"
nvidia-smi --list-gpus | head -4

# Run training
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR:$PYTHONPATH"

# ============================================
# IGNORE_TIMES default: False (all tokens in loss)
# ============================================
# IGNORE_TIMES=False means ALL tokens (event + time) enter loss computation.
# IGNORE_TIMES=True skips time tokens in loss (only event tokens contribute).
#
# Changed to False (2026-02-24) based on KTL (Keep-Time-Large) experiment evidence:
#
#   | Config                    | Val Acc  | Test Acc | Val Loss | Job ID  | W&B       |
#   |---------------------------|----------|----------|----------|---------|-----------|
#   | IGNORE_TIMES=False (KTL)  | 80.28%   | 77.82%  | 1.017    | 2458440 | ew3af26l  |
#   | IGNORE_TIMES=True  (G0)   | 76.61%   | 73.45%  | 1.177    | 2458353 | cgdexweb  |
#   | Delta                     | +3.67pp  | +4.37pp | -0.160   |         |           |
#
# Both runs: 75M model, 32N (128 GPU), BSZ=10, lr=1e-3, 40 epochs.
# KTL was still improving when cancelled at E37 — ceiling likely higher.
# Conclusion: time prediction provides causal signal that improves event prediction.
# Override: IGNORE_TIMES=True sbatch ... (to revert to old behavior)
# ============================================

python -u -B run_train.py \
    --USE_WANDB=True \
    --wandb_project="${WANDB_PROJECT:-lobs5-360M-G30}" \
    --wandb_entity=kang-oxford \
    --C_init=trunc_standard_normal \
    --prenorm=True \
    --batchnorm=False \
    --bidirectional=False \
    --dataset=lobster-prediction \
    --merging=padded \
    --dir_name="$DATA_DIR" \
    ${TEST_DIR:+--test_dir_name="$TEST_DIR"} \
    --clip_eigs=True \
    --activation_fn=half_glu1 \
    --dt_global=False \
    --epochs="${EPOCHS:-1}" \
    --jax_seed=42 \
    --opt_config=standard \
    --p_dropout=0.0 \
    --warmup_end="$WARMUP_END" \
    --weight_decay=0.05 \
    --msg_seq_len=500 \
    --use_book_data=True \
    --use_simple_book=False \
    --book_transform=True \
    --masking=none \
    --num_devices="$GPUS_PER_NODE" \
    --n_data_workers=12 \
    --debug_loading=False \
    --enable_profiler=False \
    --random_offsets_train=True \
    --shuffle_train=True \
    --debug_overfit=False \
    --ignore_times="${IGNORE_TIMES:-False}" \
    --lr_patience=4 \
    --d_model="$D_MODEL" \
    --n_layers="$N_LAYERS" \
    --blocks="$BLOCKS" \
    --ssm_size_base="$SSM_SIZE_BASE" \
    --ssm_lr_base="$SSM_LR_BASE" \
    --lr_factor="$LR_FACTOR" \
    --micro_bsz="$PER_GPU_BSZ" \
    ${CURTAIL_EPOCHS:+--curtail_epochs=$CURTAIL_EPOCHS} \
    --mini_epochs=$MINI_EPOCHS \
    ${RESTORE_PATH:+--restore=$RESTORE_PATH} \
    ${RESTORE_STEP:+--restore_step=$RESTORE_STEP} \
    ${RESUME_FROM_STEP:+--resume_from_step=$RESUME_FROM_STEP} \
    ${HIERARCHICAL:+--hierarchical=$HIERARCHICAL} \
    ${LOCAL_STEPS_K:+--local_steps_k=$LOCAL_STEPS_K} \
    ${TICKERS:+--tickers=$TICKERS} \
    ${DATA_ROOT:+--data_root="$DATA_ROOT"} \
    ${TRAIN_DATE_RANGE:+--train_date_range=$TRAIN_DATE_RANGE} \
    ${TEST_DATE_RANGE:+--test_date_range=$TEST_DATE_RANGE} \
    --checkpoint_every_n_steps="$CHECKPOINT_EVERY" \
    --max_job_hours="$MAX_JOB_HOURS"
