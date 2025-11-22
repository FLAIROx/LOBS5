"""
Component Ablation Test for Compilation Memory Analysis

Tests different components incrementally to find which one triggers the 67.19 GB compilation memory.

Test environment: L=6000, BSZ=128 (known to OOM with 67.19 GB)

Usage:
  python test_ablation.py --msg_seq_len 250 --bsz 128
"""
import os
# Don't force CPU - let it use GPU to match actual training conditions
# os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import sys
import re
import time
import argparse
from functools import partial

# Import optax
import optax

# We'll parse args and reuse run_train.py's initialization
# Import after setting up environment
def setup_environment():
    """Setup same environment as run_train.py"""
    os.environ["JAX_CHECKPOINT_POLICY"] = "nothing_saveable"
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_deterministic_ops=true "
        "--xla_force_host_platform_device_count=1"  # Use 1 device for testing
    )
setup_environment()

# Capture compilation memory warnings
class CompilationMemoryMonitor:
    def __init__(self):
        self.compile_memory_gb = None
        self.original_stderr = sys.stderr

    def __enter__(self):
        import io
        self.stderr_capture = io.StringIO()
        sys.stderr = self.stderr_capture
        return self

    def __exit__(self, *args):
        stderr_output = self.stderr_capture.getvalue()
        sys.stderr = self.original_stderr

        # Parse "Can't reduce memory use below XX.XXGiB"
        match = re.search(r"Can't reduce memory use below (\d+\.\d+)GiB", stderr_output)
        if match:
            self.compile_memory_gb = float(match.group(1))

        # Print captured output
        if stderr_output:
            print(stderr_output, file=self.original_stderr)


def create_test_data(bsz=128, seq_len=6000):
    """Create dummy test data"""
    key = jax.random.PRNGKey(42)

    # Messages: (BSZ, L)
    messages = jax.random.randint(key, (bsz, seq_len), 0, 2112)

    # Books: (BSZ, L, book_dim)
    books = jax.random.normal(key, (bsz, seq_len // 24, 503))

    # Labels: (BSZ, L)
    labels = jax.random.randint(key, (bsz, seq_len), 0, 2112)

    # Integration timesteps
    msg_times = jax.random.normal(key, (bsz, seq_len))
    book_times = jax.random.normal(key, (bsz, seq_len // 24))

    return messages, books, labels, msg_times, book_times


def test_a_forward_only(model, params, inputs):
    """Test A: Forward pass only, return before decoder"""
    print("\n" + "="*70)
    print("Test A: Forward Only (no Decoder)")
    print("="*70)

    messages, books, msg_times, book_times = inputs

    @jax.jit
    def forward_only(params, messages, books, msg_times, book_times):
        # Call model but stop before decoder
        # This requires modifying the model or using a partial forward pass
        # For now, we'll call the full model but not compute loss

        # Simplified: just return the last hidden state before decoder
        # In actual model: x = fused_s5(x)
        # We'll simulate this by calling the model and returning intermediate

        # Since we can't easily intercept, we'll just do forward pass
        # and return the output without computing anything after
        output = model.apply(
            {"params": params},
            messages, books, msg_times, book_times,
            method='__call_ar__'
        )
        # Return without any further processing
        return output

    with CompilationMemoryMonitor() as monitor:
        start = time.time()
        result = forward_only(params, messages, books, msg_times, book_times)
        result.block_until_ready()
        compile_time = time.time() - start

    print(f"  Compilation time: {compile_time:.1f}s")
    if monitor.compile_memory_gb:
        print(f"  Compilation memory: ⚠️ {monitor.compile_memory_gb:.2f} GB")
        status = "❌ OOM" if monitor.compile_memory_gb > 65 else "⚠️ High memory"
    else:
        print(f"  Compilation memory: ✅ < 67 GB (no warning)")
        status = "✅ Success"
    print(f"  Status: {status}")

    return monitor.compile_memory_gb, status


def test_b_with_decoder(model, params, inputs):
    """Test B: Forward + Decoder (no log_softmax)"""
    print("\n" + "="*70)
    print("Test B: Forward + Decoder (no log_softmax)")
    print("="*70)

    messages, books, msg_times, book_times = inputs

    @jax.jit
    def forward_with_decoder(params, messages, books, msg_times, book_times):
        # Full forward pass including decoder
        logits = model.apply(
            {"params": params},
            messages, books, msg_times, book_times,
            method='__call_ar__'
        )
        # Return logits directly (model already includes log_softmax, but we test it)
        return logits

    with CompilationMemoryMonitor() as monitor:
        start = time.time()
        result = forward_with_decoder(params, messages, books, msg_times, book_times)
        result.block_until_ready()
        compile_time = time.time() - start

    print(f"  Compilation time: {compile_time:.1f}s")
    if monitor.compile_memory_gb:
        print(f"  Compilation memory: ⚠️ {monitor.compile_memory_gb:.2f} GB")
        status = "❌ OOM" if monitor.compile_memory_gb > 65 else "⚠️ High memory"
    else:
        print(f"  Compilation memory: ✅ < 67 GB (no warning)")
        status = "✅ Success"
    print(f"  Status: {status}")

    return monitor.compile_memory_gb, status


def test_c_with_logsoftmax(model, params, inputs):
    """Test C: Forward + Decoder + log_softmax"""
    print("\n" + "="*70)
    print("Test C: Forward + Decoder + log_softmax")
    print("="*70)

    messages, books, msg_times, book_times = inputs

    @jax.jit
    def forward_with_logsoftmax(params, messages, books, msg_times, book_times):
        # Model already includes log_softmax in __call_ar__
        logits = model.apply(
            {"params": params},
            messages, books, msg_times, book_times,
            method='__call_ar__'
        )
        # Logits are already log-probabilities
        return logits

    with CompilationMemoryMonitor() as monitor:
        start = time.time()
        result = forward_with_logsoftmax(params, messages, books, msg_times, book_times)
        result.block_until_ready()
        compile_time = time.time() - start

    print(f"  Compilation time: {compile_time:.1f}s")
    if monitor.compile_memory_gb:
        print(f"  Compilation memory: ⚠️ {monitor.compile_memory_gb:.2f} GB")
        status = "❌ OOM" if monitor.compile_memory_gb > 65 else "⚠️ High memory"
    else:
        print(f"  Compilation memory: ✅ < 67 GB (no warning)")
        status = "✅ Success"
    print(f"  Status: {status}")

    return monitor.compile_memory_gb, status


def test_d_with_loss(model, params, inputs, labels):
    """Test D: Forward + Decoder + Loss (no Gradient)"""
    print("\n" + "="*70)
    print("Test D: Forward + Decoder + Loss (no Gradient)")
    print("="*70)

    messages, books, msg_times, book_times = inputs

    @jax.jit
    def forward_with_loss(params, messages, books, msg_times, book_times, labels):
        # Forward pass
        logits = model.apply(
            {"params": params},
            messages, books, msg_times, book_times,
            method='__call_ar__'
        )

        # Compute cross-entropy loss
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(ce)

        return loss, ce

    with CompilationMemoryMonitor() as monitor:
        start = time.time()
        result = forward_with_loss(params, messages, books, msg_times, book_times, labels)
        jax.tree_map(lambda x: x.block_until_ready(), result)
        compile_time = time.time() - start

    print(f"  Compilation time: {compile_time:.1f}s")
    if monitor.compile_memory_gb:
        print(f"  Compilation memory: ⚠️ {monitor.compile_memory_gb:.2f} GB")
        status = "❌ OOM" if monitor.compile_memory_gb > 65 else "⚠️ High memory"
    else:
        print(f"  Compilation memory: ✅ < 67 GB (no warning)")
        status = "✅ Success"
    print(f"  Status: {status}")

    return monitor.compile_memory_gb, status


def test_e_with_gradient(model, params, inputs, labels):
    """Test E: Complete (Forward + Decoder + Loss + Gradient)"""
    print("\n" + "="*70)
    print("Test E: Complete (Forward + Loss + Gradient)")
    print("="*70)

    messages, books, msg_times, book_times = inputs

    @jax.jit
    def forward_with_gradient(params, messages, books, msg_times, book_times, labels):
        def loss_fn(p):
            # Forward pass
            logits = model.apply(
                {"params": p},
                messages, books, msg_times, book_times,
                method='__call_ar__'
            )

            # Loss
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            loss = jnp.mean(ce)

            return loss, (logits, ce)

        # Compute gradients
        (loss, (logits, ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        return loss, grads

    with CompilationMemoryMonitor() as monitor:
        start = time.time()
        result = forward_with_gradient(params, messages, books, msg_times, book_times, labels)
        jax.tree_map(lambda x: x.block_until_ready(), result)
        compile_time = time.time() - start

    print(f"  Compilation time: {compile_time:.1f}s")
    if monitor.compile_memory_gb:
        print(f"  Compilation memory: ⚠️ {monitor.compile_memory_gb:.2f} GB")
        status = "❌ OOM" if monitor.compile_memory_gb > 65 else "⚠️ High memory"
    else:
        print(f"  Compilation memory: ✅ < 67 GB (no warning)")
        status = "✅ Success"
    print(f"  Status: {status}")

    return monitor.compile_memory_gb, status


def main():
    """Run all ablation tests"""
    print("\n" + "="*70)
    print("Component Ablation Test")
    print("Finding the source of 67.19 GB compilation memory")
    print("="*70)
    print(f"\nTest Configuration:")
    print(f"  Sequence length: 6000")
    print(f"  Batch size: 128")
    print(f"  Expected: This configuration should trigger 67.19 GB compilation memory")
    print(f"  Goal: Find which component causes it")

    # Initialize model and data
    print("\n[*] Initializing model and data...")

    # Parse args (simplified)
    class Args:
        # Model config (matching your setup)
        C_init = 'trunc_standard_normal'
        prenorm = True
        batchnorm = False
        bidirectional = False
        blocks = 16
        d_model = 1024
        dataset = 'lobster-prediction'
        merging = 'padded'
        ssm_size_base = 1024
        n_layers = 12
        conj_sym = True
        clip_eigs = True
        activation_fn = 'half_glu1'
        dt_global = False
        p_dropout = 0.0
        discretization = 'zoh'
        mode = 'none'
        dt_min = 0.001
        dt_max = 0.1

        # For this test
        msg_seq_len = 250  # 250 * 24 = 6000
        use_book_data = True
        use_simple_book = False
        book_transform = True
        book_depth = 500

        n_message_layers = 2
        n_book_pre_layers = 1
        n_book_post_layers = 1

        # Training config
        bsz = 128
        jax_seed = 42

    args = Args()

    # Create dummy model and params
    from lob.lob_seq_model import BatchPaddedLobPredModel

    model = BatchPaddedLobPredModel(
        ssm_size=args.ssm_size_base,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_message_layers=args.n_message_layers,
        n_book_pre_layers=args.n_book_pre_layers,
        n_book_post_layers=args.n_book_post_layers,
        padded=True,
        activation=args.activation_fn,
        dropout=args.p_dropout,
        training=True,
        prenorm=args.prenorm,
        batchnorm=args.batchnorm,
        bn_momentum=0.95,
        step_rescale=1.0,
        blocks=args.blocks,
        C_init=args.C_init,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional,
        discretization=args.discretization,
        mode=args.mode,
        dt_global=args.dt_global,
        msg_seq_len=args.msg_seq_len,
        use_simple_book=args.use_simple_book,
    )

    # Initialize params
    key = jax.random.PRNGKey(args.jax_seed)

    # Dummy inputs for initialization
    dummy_msg = jnp.ones((args.bsz, 6000), dtype=jnp.int32)
    dummy_book = jnp.ones((args.bsz, 250, 503), dtype=jnp.float32)
    dummy_msg_times = jnp.ones((args.bsz, 6000), dtype=jnp.float32)
    dummy_book_times = jnp.ones((args.bsz, 250), dtype=jnp.float32)

    variables = model.init(
        key,
        dummy_msg, dummy_book,
        dummy_msg_times, dummy_book_times
    )
    params = variables['params']

    print(f"  Model initialized")
    print(f"  Total parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")

    # Create test data
    messages, books, labels, msg_times, book_times = create_test_data(bsz=args.bsz, seq_len=6000)
    inputs = (messages, books, msg_times, book_times)

    # Run tests
    results = {}

    # Test A: Forward only
    mem_a, status_a = test_a_forward_only(model, params, inputs)
    results['A'] = (mem_a, status_a)

    # Test B: Forward + Decoder
    mem_b, status_b = test_b_with_decoder(model, params, inputs)
    results['B'] = (mem_b, status_b)

    # Test C: Forward + Decoder + log_softmax
    mem_c, status_c = test_c_with_logsoftmax(model, params, inputs)
    results['C'] = (mem_c, status_c)

    # Test D: Forward + Decoder + Loss
    mem_d, status_d = test_d_with_loss(model, params, inputs, labels)
    results['D'] = (mem_d, status_d)

    # Test E: Complete with Gradient (only if D succeeded)
    if "Success" in status_d or "High" in status_d:
        mem_e, status_e = test_e_with_gradient(model, params, inputs, labels)
        results['E'] = (mem_e, status_e)
    else:
        print("\n" + "="*70)
        print("Test E: Skipped (Test D failed)")
        print("="*70)
        results['E'] = (None, "Skipped")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Test':<6} {'Component':<35} {'Compile Mem':<15} {'Status':<15}")
    print("-" * 70)

    components = {
        'A': 'Forward only',
        'B': 'Forward + Decoder',
        'C': 'Forward + Decoder + log_softmax',
        'D': 'Forward + Decoder + Loss',
        'E': 'Forward + Decoder + Loss + Gradient'
    }

    for test_id, component in components.items():
        mem, status = results[test_id]
        mem_str = f"{mem:.2f} GB" if mem else "< 67 GB"
        print(f"{test_id:<6} {component:<35} {mem_str:<15} {status:<15}")

    # Find the culprit
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    for i, (test_id, (mem, status)) in enumerate(results.items()):
        if mem and mem > 65:
            print(f"\n🔴 Component '{components[test_id]}' triggers 67.19 GB compilation memory!")
            print(f"   This is the bottleneck that needs optimization.")
            break
    else:
        print("\n❓ No single component triggered 67.19 GB.")
        print("   The issue might be cumulative or related to graph size.")

    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
