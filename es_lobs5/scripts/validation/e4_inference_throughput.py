#!/usr/bin/env python3
"""
E4: Inference Throughput Benchmark for ES-LOBS5

Purpose: Measure inference throughput and latency for the ES-compatible LOBS5 model.

Tests:
1. Single forward pass latency (batch_size=1)
2. Batch throughput for batch_size=1,8,32,64 (tokens/sec)
3. Autoregressive inference (1000 steps)
4. JIT compilation time vs subsequent calls
5. Peak memory during inference

Usage:
    python e4_inference_throughput.py
    # Or via sbatch
"""

import sys
import os
import time
import glob
from functools import partial
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import jax
import jax.numpy as jnp

# Configuration
CHECKPOINT_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
REPLAY_DATA_PATH = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"

# Test configurations
BATCH_SIZES = [1, 8, 32, 64]
AUTOREGRESSIVE_STEPS = 1000
N_WARMUP_RUNS = 3
N_BENCHMARK_RUNS = 10
MSG_SEQ_LEN = 500
BOOK_DEPTH = 500
TOKEN_MODE = 24
VOCAB_SIZE = 2112  # for token_mode=24


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")


def format_time(seconds: float) -> str:
    """Format time with appropriate unit."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def format_throughput(tokens_per_sec: float) -> str:
    """Format throughput with appropriate unit."""
    if tokens_per_sec >= 1e6:
        return f"{tokens_per_sec / 1e6:.2f} M tokens/s"
    elif tokens_per_sec >= 1e3:
        return f"{tokens_per_sec / 1e3:.2f} K tokens/s"
    else:
        return f"{tokens_per_sec:.2f} tokens/s"


def get_memory_usage() -> Dict[str, float]:
    """Get current JAX memory usage in GB."""
    try:
        devices = jax.local_devices()
        memory_info = {}
        for dev in devices:
            if dev.platform == 'gpu':
                stats = dev.memory_stats()
                if stats:
                    memory_info[f'{dev.platform}:{dev.id}'] = {
                        'used_gb': stats.get('bytes_in_use', 0) / (1024**3),
                        'peak_gb': stats.get('peak_bytes_in_use', 0) / (1024**3),
                        'limit_gb': stats.get('bytes_limit', 0) / (1024**3),
                    }
        return memory_info
    except Exception as e:
        return {'error': str(e)}


def load_model_and_data():
    """Load model from checkpoint and prepare test data."""
    print_header("Loading Model and Data")

    # Import ES components
    from es_lobs5.adapters.checkpoint_adapter import load_checkpoint_for_es
    from es_lobs5.models import ES_PaddedLobPredModel
    from es_lobs5.models.common import CommonParams

    # Load checkpoint
    print(f"[1/3] Loading checkpoint from: {CHECKPOINT_PATH}")
    es_init, es_tree_key = load_checkpoint_for_es(CHECKPOINT_PATH)
    print(f"      Checkpoint loaded successfully")
    print(f"      d_model: {es_init.frozen_params.get('d_model')}")
    print(f"      d_output: {es_init.frozen_params.get('d_output')}")

    # Load test data
    print(f"\n[2/3] Loading test data from: {REPLAY_DATA_PATH}")
    msg_files = sorted(glob.glob(f"{REPLAY_DATA_PATH}/*_message_*.npy"))
    book_files = sorted(glob.glob(f"{REPLAY_DATA_PATH}/*_orderbook_*.npy"))

    if not msg_files or not book_files:
        raise FileNotFoundError(f"No data files found in {REPLAY_DATA_PATH}")

    # Load first message file
    raw_msg = np.load(msg_files[0])
    print(f"      Message data shape: {raw_msg.shape}")

    # Load corresponding orderbook file
    raw_book = np.load(book_files[0])
    print(f"      Orderbook data shape: {raw_book.shape}")

    # Prepare test inputs
    print(f"\n[3/3] Preparing test inputs...")

    # Extract config from frozen_params
    d_book = es_init.frozen_params.get('d_book', 3 + BOOK_DEPTH)
    msg_seq_len = es_init.frozen_params.get('msg_seq_len', MSG_SEQ_LEN)

    # Prepare message tokens (should be int32 indices)
    # The raw data should already be tokenized
    if raw_msg.dtype in [np.int32, np.int64, np.uint32, np.uint64]:
        # Already tokenized
        msg_tokens = raw_msg[:msg_seq_len].astype(np.int32)
    else:
        # Need to convert to tokens (assume already encoded, just cast)
        msg_tokens = raw_msg[:msg_seq_len].astype(np.int32)

    # Flatten if needed (should be 1D sequence of tokens)
    if len(msg_tokens.shape) > 1:
        msg_tokens = msg_tokens.flatten()[:msg_seq_len]

    # Prepare book features (should be float32)
    # Book data typically has shape (n_events, features) where features includes depth levels
    if len(raw_book.shape) == 2:
        # Take first msg_seq_len rows and first d_book columns
        book_features = raw_book[:msg_seq_len, :d_book].astype(np.float32)
    else:
        # Reshape if needed
        book_features = raw_book[:msg_seq_len * d_book].reshape(msg_seq_len, d_book).astype(np.float32)

    print(f"      Prepared message tokens: {msg_tokens.shape} (dtype: {msg_tokens.dtype})")
    print(f"      Prepared book features: {book_features.shape} (dtype: {book_features.dtype})")

    # Convert to JAX arrays
    x_m = jnp.array(msg_tokens)
    x_b = jnp.array(book_features)

    return es_init, es_tree_key, x_m, x_b


def create_dummy_noiser():
    """Create a dummy noiser that returns parameters unchanged.

    Matches HyperscaleES noiser interface (base_noiser.py):
    - do_mm: Matrix multiply (x @ param.T for MM which stores as (out, in))
    - do_Tmm: Transposed multiply (x @ param for TMM which stores as (in, out))
    - do_emb: Embedding lookup
    - get_noisy_standard: Standard parameter access
    """
    class DummyNoiser:
        """Dummy noiser for inference (no perturbation)."""

        def get_noisy_standard(self, frozen_noiser_params, noiser_params, params, es_tree_key, iterinfo):
            return params

        def do_mm(self, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
            # MM stores weight as (out_dim, in_dim), so x @ param.T
            return x @ param.T

        def do_Tmm(self, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
            # TMM stores weight as (in_dim, out_dim), so x @ param
            return x @ param

        def do_emb(self, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
            return param[x]

    return DummyNoiser()


def create_common_params(es_init, es_tree_key, noiser=None):
    """Create CommonParams for inference."""
    from es_lobs5.models.common import CommonParams

    if noiser is None:
        noiser = create_dummy_noiser()

    # Create a dummy iterinfo
    class DummyIterInfo:
        def __init__(self):
            self.gen = 0
            self.per_eval = 0

    return CommonParams(
        noiser=noiser,
        frozen_noiser_params={},
        noiser_params={},
        params=es_init.params,
        frozen_params=es_init.frozen_params,
        es_tree_key=es_tree_key,
        iterinfo=DummyIterInfo(),
    )


def benchmark_single_forward(common_params, x_m, x_b, n_warmup=N_WARMUP_RUNS, n_runs=N_BENCHMARK_RUNS):
    """Benchmark single forward pass latency."""
    from es_lobs5.models import ES_PaddedLobPredModel

    print_subheader("Test 1: Single Forward Pass Latency (batch_size=1)")

    # Define forward function
    @jax.jit
    def forward_fn(params, x_m, x_b):
        cp = common_params._replace(params=params)
        return ES_PaddedLobPredModel._forward(cp, x_m, x_b)

    # Measure JIT compilation time
    print("  Measuring JIT compilation time...")
    compile_start = time.perf_counter()
    output = forward_fn(common_params.params, x_m, x_b)
    output.block_until_ready()
    compile_time = time.perf_counter() - compile_start
    print(f"  JIT compilation + first run: {format_time(compile_time)}")

    # Warmup
    print(f"  Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        output = forward_fn(common_params.params, x_m, x_b)
        output.block_until_ready()

    # Benchmark
    print(f"  Benchmarking ({n_runs} runs)...")
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        output = forward_fn(common_params.params, x_m, x_b)
        output.block_until_ready()
        latencies.append(time.perf_counter() - start)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    print(f"\n  Results:")
    print(f"    Average latency: {format_time(avg_latency)} +/- {format_time(std_latency)}")
    print(f"    Min latency:     {format_time(min_latency)}")
    print(f"    Max latency:     {format_time(max_latency)}")

    return {
        'compile_time': compile_time,
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
    }


def benchmark_batch_throughput(common_params, x_m, x_b, batch_sizes=BATCH_SIZES,
                                n_warmup=N_WARMUP_RUNS, n_runs=N_BENCHMARK_RUNS):
    """Benchmark throughput at different batch sizes."""
    from es_lobs5.models import ES_PaddedLobPredModel

    print_subheader("Test 2: Batch Throughput (tokens/sec)")

    seq_len = x_m.shape[0]
    results = {}

    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")

        # Create batched inputs
        x_m_batch = jnp.tile(x_m[jnp.newaxis, :], (batch_size, 1))
        x_b_batch = jnp.tile(x_b[jnp.newaxis, :, :], (batch_size, 1, 1))

        # Define batched forward function using vmap
        @jax.jit
        def batched_forward(params, x_m_batch, x_b_batch):
            def single_forward(x_m, x_b):
                cp = common_params._replace(params=params)
                return ES_PaddedLobPredModel._forward(cp, x_m, x_b)
            return jax.vmap(single_forward)(x_m_batch, x_b_batch)

        # First run (compile + execute)
        compile_start = time.perf_counter()
        output = batched_forward(common_params.params, x_m_batch, x_b_batch)
        output.block_until_ready()
        compile_time = time.perf_counter() - compile_start
        print(f"    JIT compile: {format_time(compile_time)}")

        # Warmup
        for _ in range(n_warmup):
            output = batched_forward(common_params.params, x_m_batch, x_b_batch)
            output.block_until_ready()

        # Benchmark
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            output = batched_forward(common_params.params, x_m_batch, x_b_batch)
            output.block_until_ready()
            latencies.append(time.perf_counter() - start)

        avg_latency = np.mean(latencies)
        total_tokens = batch_size * seq_len
        throughput = total_tokens / avg_latency

        print(f"    Avg latency: {format_time(avg_latency)}")
        print(f"    Throughput:  {format_throughput(throughput)}")

        results[batch_size] = {
            'compile_time': compile_time,
            'avg_latency': avg_latency,
            'throughput': throughput,
            'tokens_per_batch': total_tokens,
        }

    return results


def benchmark_autoregressive(common_params, x_m, x_b, n_steps=AUTOREGRESSIVE_STEPS,
                              n_warmup=N_WARMUP_RUNS, n_runs=N_BENCHMARK_RUNS):
    """Benchmark autoregressive inference."""
    from es_lobs5.models import ES_PaddedLobPredModel

    print_subheader(f"Test 3: Autoregressive Inference ({n_steps} steps)")

    # For autoregressive inference, we simulate step-by-step generation
    # Here we use _forward_ar which returns per-token predictions

    @jax.jit
    def forward_ar(params, x_m, x_b):
        cp = common_params._replace(params=params)
        return ES_PaddedLobPredModel._forward_ar(cp, x_m, x_b)

    # Compile and warmup
    print("  Compiling autoregressive forward...")
    compile_start = time.perf_counter()
    output = forward_ar(common_params.params, x_m, x_b)
    output.block_until_ready()
    compile_time = time.perf_counter() - compile_start
    print(f"  JIT compile: {format_time(compile_time)}")

    # Warmup
    for _ in range(n_warmup):
        output = forward_ar(common_params.params, x_m, x_b)
        output.block_until_ready()

    # Simulate autoregressive generation
    # In real AR inference, we would call the model repeatedly
    # Here we simulate by calling forward_ar multiple times
    print(f"  Simulating {n_steps} autoregressive steps...")

    # For accurate simulation, we'll call the model n_steps times
    # Each call processes the full sequence but returns per-token outputs
    ar_latencies = []
    for run in range(n_runs):
        step_times = []
        for step in range(n_steps):
            start = time.perf_counter()
            # In real AR, we'd update x_m with generated token
            # Here we just call forward with same input
            output = forward_ar(common_params.params, x_m, x_b)
            output.block_until_ready()
            step_times.append(time.perf_counter() - start)
        ar_latencies.append(np.mean(step_times))

    avg_step_latency = np.mean(ar_latencies)
    tokens_per_sec = 1.0 / avg_step_latency  # One token generated per step

    print(f"\n  Results:")
    print(f"    Average step latency: {format_time(avg_step_latency)}")
    print(f"    Steps per second:     {tokens_per_sec:.2f}")
    print(f"    Total {n_steps} steps: {format_time(avg_step_latency * n_steps)}")

    return {
        'compile_time': compile_time,
        'avg_step_latency': avg_step_latency,
        'tokens_per_sec': tokens_per_sec,
        'total_time': avg_step_latency * n_steps,
    }


def benchmark_jit_vs_eager(common_params, x_m, x_b, n_runs=5):
    """Compare JIT compiled vs subsequent call times."""
    from es_lobs5.models import ES_PaddedLobPredModel

    print_subheader("Test 4: JIT Compilation vs Subsequent Calls")

    @jax.jit
    def forward_jit(params, x_m, x_b):
        cp = common_params._replace(params=params)
        return ES_PaddedLobPredModel._forward(cp, x_m, x_b)

    # First call (includes compilation)
    print("  First call (JIT compilation)...")
    first_start = time.perf_counter()
    output = forward_jit(common_params.params, x_m, x_b)
    output.block_until_ready()
    first_call = time.perf_counter() - first_start

    # Subsequent calls
    print("  Subsequent calls (cached)...")
    subsequent_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        output = forward_jit(common_params.params, x_m, x_b)
        output.block_until_ready()
        subsequent_times.append(time.perf_counter() - start)

    avg_subsequent = np.mean(subsequent_times)
    speedup = first_call / avg_subsequent

    print(f"\n  Results:")
    print(f"    First call (w/ compile): {format_time(first_call)}")
    print(f"    Avg subsequent call:     {format_time(avg_subsequent)}")
    print(f"    Speedup ratio:           {speedup:.1f}x")

    return {
        'first_call': first_call,
        'avg_subsequent': avg_subsequent,
        'speedup': speedup,
    }


def measure_peak_memory(common_params, x_m, x_b, batch_sizes=BATCH_SIZES):
    """Measure peak memory usage during inference."""
    from es_lobs5.models import ES_PaddedLobPredModel

    print_subheader("Test 5: Peak Memory During Inference")

    results = {}

    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")

        # Create batched inputs
        x_m_batch = jnp.tile(x_m[jnp.newaxis, :], (batch_size, 1))
        x_b_batch = jnp.tile(x_b[jnp.newaxis, :, :], (batch_size, 1, 1))

        @jax.jit
        def batched_forward(params, x_m_batch, x_b_batch):
            def single_forward(x_m, x_b):
                cp = common_params._replace(params=params)
                return ES_PaddedLobPredModel._forward(cp, x_m, x_b)
            return jax.vmap(single_forward)(x_m_batch, x_b_batch)

        # Clear any cached memory
        try:
            jax.clear_caches()
        except:
            pass

        # Run forward pass
        output = batched_forward(common_params.params, x_m_batch, x_b_batch)
        output.block_until_ready()

        # Get memory stats
        mem_info = get_memory_usage()

        if 'error' not in mem_info:
            for dev, stats in mem_info.items():
                print(f"    {dev}:")
                print(f"      Used:  {stats['used_gb']:.2f} GB")
                print(f"      Peak:  {stats['peak_gb']:.2f} GB")
                print(f"      Limit: {stats['limit_gb']:.2f} GB")
        else:
            print(f"    Memory stats not available: {mem_info.get('error', 'unknown')}")

        results[batch_size] = mem_info

    return results


def print_summary(results: Dict[str, Any]):
    """Print formatted summary table."""
    print_header("Summary Results")

    print("\n--- Single Forward Pass ---")
    if 'single_forward' in results:
        sf = results['single_forward']
        print(f"  Compile time:   {format_time(sf['compile_time'])}")
        print(f"  Avg latency:    {format_time(sf['avg_latency'])}")

    print("\n--- Batch Throughput ---")
    print(f"  {'Batch Size':>12} | {'Latency':>12} | {'Throughput':>20}")
    print("  " + "-" * 50)
    if 'batch_throughput' in results:
        for bs, data in sorted(results['batch_throughput'].items()):
            print(f"  {bs:>12} | {format_time(data['avg_latency']):>12} | {format_throughput(data['throughput']):>20}")

    print("\n--- Autoregressive Inference ---")
    if 'autoregressive' in results:
        ar = results['autoregressive']
        print(f"  Step latency:   {format_time(ar['avg_step_latency'])}")
        print(f"  Steps/sec:      {ar['tokens_per_sec']:.2f}")

    print("\n--- JIT Compilation ---")
    if 'jit_comparison' in results:
        jit = results['jit_comparison']
        print(f"  First call:     {format_time(jit['first_call'])}")
        print(f"  Subsequent:     {format_time(jit['avg_subsequent'])}")
        print(f"  Speedup:        {jit['speedup']:.1f}x")

    print("\n--- Peak Memory (batch_size=64) ---")
    if 'peak_memory' in results and 64 in results['peak_memory']:
        mem = results['peak_memory'][64]
        if 'error' not in mem:
            for dev, stats in mem.items():
                print(f"  {dev}: {stats['peak_gb']:.2f} GB peak / {stats['limit_gb']:.2f} GB limit")
        else:
            print("  Memory stats not available")

    print("\n" + "=" * 70)


def main():
    """Main benchmark entry point."""
    print_header("E4: Inference Throughput Benchmark for ES-LOBS5")

    print(f"\nConfiguration:")
    print(f"  Checkpoint:    {CHECKPOINT_PATH}")
    print(f"  Data path:     {REPLAY_DATA_PATH}")
    print(f"  Batch sizes:   {BATCH_SIZES}")
    print(f"  AR steps:      {AUTOREGRESSIVE_STEPS}")
    print(f"  Warmup runs:   {N_WARMUP_RUNS}")
    print(f"  Benchmark runs: {N_BENCHMARK_RUNS}")

    # Check JAX devices
    print(f"\nJAX devices: {jax.devices()}")

    # Load model and data
    try:
        es_init, es_tree_key, x_m, x_b = load_model_and_data()
    except Exception as e:
        print(f"Error loading model/data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create common params for inference
    common_params = create_common_params(es_init, es_tree_key)

    # Store all results
    results = {}

    # Run benchmarks
    try:
        # Test 1: Single forward pass
        results['single_forward'] = benchmark_single_forward(common_params, x_m, x_b)

        # Test 2: Batch throughput
        results['batch_throughput'] = benchmark_batch_throughput(common_params, x_m, x_b)

        # Test 3: Autoregressive inference (reduced steps for faster testing)
        results['autoregressive'] = benchmark_autoregressive(
            common_params, x_m, x_b, n_steps=min(100, AUTOREGRESSIVE_STEPS)
        )

        # Test 4: JIT vs subsequent
        results['jit_comparison'] = benchmark_jit_vs_eager(common_params, x_m, x_b)

        # Test 5: Peak memory
        results['peak_memory'] = measure_peak_memory(common_params, x_m, x_b)

    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print_summary(results)

    print("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
