"""
GPU Memory Profiling Utilities for LOBS5 Training
Based on: https://github.com/jax-ml/jax/issues/8096
"""

import os
import psutil
import jax
import jax.numpy as jnp


def print_memory_usage(step_name=""):
    """
    Print GPU and system memory usage
    Based on: https://github.com/jax-ml/jax/issues/8096
    """
    process = psutil.Process(os.getpid())
    cpu_mem_gb = process.memory_info().rss / 1024 ** 3

    print(f"\n{'='*60}")
    if step_name:
        print(f"Memory Usage @ {step_name}")
    else:
        print(f"Memory Usage")
    print(f"{'='*60}")
    print(f"CPU Memory: {cpu_mem_gb:.2f} GB")

    # Debug: Check what devices are available
    devices = jax.local_devices()
    print(f"JAX devices: {len(devices)} found")

    if len(devices) == 0:
        print("WARNING: No JAX devices found! jax.local_devices() returned empty list")
        print("This may indicate GPU is disabled via environment variables")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', 'not set')}")
    else:
        # JAX device memory
        for device in devices:
            try:
                stats = device.memory_stats()
                if stats:
                    used_mb = stats['bytes_in_use'] / 1024**2
                    limit_gb = stats['bytes_limit'] / 1024**3
                    used_gb = stats['bytes_in_use'] / 1024**3
                    used_pct = (used_gb / limit_gb * 100) if limit_gb > 0 else 0
                    print(f"Device {device.id}: {used_gb:.2f} GB / {limit_gb:.2f} GB ({used_pct:.1f}%)")
                else:
                    print(f"Device {device.id}: memory_stats() returned None/empty")
            except Exception as e:
                print(f"Device {device.id}: Error - {e}")

    print(f"{'='*60}\n")


def log_memory(step_name="", verbose=True, device_id=0):
    """
    Log CPU and GPU memory usage

    Args:
        step_name: Identifier for this measurement point
        verbose: If True, print to console; if False, only return values
        device_id: Which GPU to report (default 0, since all GPUs have similar usage in pmap)

    Returns:
        dict with cpu_gb and gpu_stats
    """
    cpu_gb = get_process_memory_gb()
    gpu_stats = get_device_memory_stats(device_id)

    if verbose and gpu_stats:
        print(f"\n{'='*60}")
        print(f"Memory @ {step_name}")
        print(f"{'='*60}")
        print(f"CPU: {cpu_gb:.2f} GB")
        print(f"GPU {device_id}: {gpu_stats['used_gb']:.2f} / {gpu_stats['limit_gb']:.2f} GB ({gpu_stats['used_pct']:.1f}%)")
        print(f"{'='*60}\n")

    return {
        'step': step_name,
        'cpu_gb': cpu_gb,
        'gpu_stats': gpu_stats
    }


def log_all_devices(step_name=""):
    """Log memory usage for all GPU devices"""
    cpu_gb = get_process_memory_gb()
    device_stats = get_all_devices_memory()

    print(f"\n{'='*60}")
    print(f"All Devices Memory @ {step_name}")
    print(f"{'='*60}")
    print(f"CPU: {cpu_gb:.2f} GB")

    if len(device_stats) > 0:
        for device_id, stats in device_stats:
            print(f"GPU {device_id}: {stats['used_gb']:.2f} / {stats['limit_gb']:.2f} GB ({stats['used_pct']:.1f}%)")
    else:
        print("[WARNING] JAX device.memory_stats() not available, trying nvidia-smi fallback...")

        # Try nvidia-smi as fallback
        nvidia_stats = get_gpu_memory_nvidia_smi()
        if nvidia_stats:
            print("[INFO] Using nvidia-smi for GPU memory monitoring")
            for gpu_info in nvidia_stats:
                print(f"GPU {gpu_info['device_id']}: {gpu_info['used_gb']:.2f} / {gpu_info['limit_gb']:.2f} GB ({gpu_info['used_pct']:.1f}%)")
            device_stats = nvidia_stats  # Update return value
        else:
            print("[ERROR] Both device.memory_stats() and nvidia-smi failed")
            print("GPU memory monitoring not available on this platform")

    print(f"{'='*60}\n")

    return {
        'step': step_name,
        'cpu_gb': cpu_gb,
        'devices': device_stats
    }


def log_tensor_memory(tensor, name="tensor"):
    """
    Log memory footprint of a JAX tensor

    Args:
        tensor: JAX array
        name: Name for logging

    Returns:
        size in MB
    """
    size_bytes = tensor.size * tensor.itemsize
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)

    if size_gb > 1.0:
        print(f"Tensor '{name}': shape={tensor.shape}, size={size_gb:.2f} GB")
    else:
        print(f"Tensor '{name}': shape={tensor.shape}, size={size_mb:.2f} MB")

    return size_mb


# JAX-compatible debug print version for use inside jitted functions
def jax_debug_log_memory(step_name, debug_enabled=False):
    """
    JAX debug print version - can be called inside jitted functions
    Note: This will print during trace/compilation, not during execution

    Args:
        step_name: Identifier for this measurement point
        debug_enabled: If False, debug print is excluded from XLA compilation
    """
    if debug_enabled:  # Python-level conditional - evaluated at trace-time
        cpu_gb = get_process_memory_gb()
        gpu_stats = get_device_memory_stats(0)

        if gpu_stats:
            jax.debug.print(
                "Memory @ {}: CPU={:.2f}GB, GPU={:.2f}/{:.2f}GB ({:.1f}%)",
                step_name, cpu_gb, gpu_stats['used_gb'], gpu_stats['limit_gb'], gpu_stats['used_pct']
            )


class MemoryTracker:
    """Context manager for tracking memory usage across a code block"""

    def __init__(self, name="", device_id=0):
        self.name = name
        self.device_id = device_id
        self.start_stats = None
        self.end_stats = None

    def __enter__(self):
        self.start_stats = log_memory(f"{self.name} [START]", verbose=False, device_id=self.device_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_stats = log_memory(f"{self.name} [END]", verbose=False, device_id=self.device_id)

        # Calculate delta
        if self.start_stats['gpu_stats'] and self.end_stats['gpu_stats']:
            start_gb = self.start_stats['gpu_stats']['used_gb']
            end_gb = self.end_stats['gpu_stats']['used_gb']
            delta_gb = end_gb - start_gb

            print(f"\n{'='*60}")
            print(f"Memory Delta for '{self.name}'")
            print(f"{'='*60}")
            print(f"Start: {start_gb:.2f} GB")
            print(f"End:   {end_gb:.2f} GB")
            print(f"Delta: {delta_gb:+.2f} GB")
            print(f"{'='*60}\n")


def analyze_compilation_memory(train_step_fn, dummy_inputs, step_name="train_step"):
    """
    Analyze memory usage using jax.lower().compile().memory_analysis()

    Args:
        train_step_fn: The jitted function to analyze
        dummy_inputs: Tuple of dummy inputs matching the function signature
        step_name: Name for logging

    Returns:
        Memory analysis dictionary from JAX compilation
    """
    print(f"\n{'='*60}")
    print(f"JAX Compilation Memory Analysis for {step_name}")
    print(f"{'='*60}")

    def bytes_to_gb(bytes_val):
        """Convert bytes to GB"""
        return bytes_val / (1024**3)

    try:
        # Lower and compile the function
        lowered = jax.jit(train_step_fn).lower(*dummy_inputs)
        compiled = lowered.compile()

        # Get memory analysis
        mem_analysis = compiled.memory_analysis()

        print(f"\n[jax.lower().compile()] Compilation Memory Analysis:\n")

        # Device Memory (GPU)
        print("Device Memory (GPU):")
        if hasattr(mem_analysis, 'generated_code_size_in_bytes'):
            code_gb = bytes_to_gb(mem_analysis.generated_code_size_in_bytes)
            print(f"  Generated Code:     {code_gb:>8.3f} GB  ({mem_analysis.generated_code_size_in_bytes:,} bytes)")

        if hasattr(mem_analysis, 'argument_size_in_bytes'):
            arg_gb = bytes_to_gb(mem_analysis.argument_size_in_bytes)
            print(f"  Arguments:          {arg_gb:>8.3f} GB  ({mem_analysis.argument_size_in_bytes:,} bytes)")

        if hasattr(mem_analysis, 'output_size_in_bytes'):
            out_gb = bytes_to_gb(mem_analysis.output_size_in_bytes)
            print(f"  Outputs:            {out_gb:>8.3f} GB  ({mem_analysis.output_size_in_bytes:,} bytes)")

        if hasattr(mem_analysis, 'temp_size_in_bytes'):
            temp_gb = bytes_to_gb(mem_analysis.temp_size_in_bytes)
            print(f"  Temp/Scratch:       {temp_gb:>8.3f} GB  ({mem_analysis.temp_size_in_bytes:,} bytes)  ← Peak memory during execution")

        if hasattr(mem_analysis, 'alias_size_in_bytes'):
            alias_gb = bytes_to_gb(mem_analysis.alias_size_in_bytes)
            print(f"  Aliases:            {alias_gb:>8.3f} GB  ({mem_analysis.alias_size_in_bytes:,} bytes)")

        # Host Memory (CPU)
        print("\nHost Memory (CPU):")
        if hasattr(mem_analysis, 'host_generated_code_size_in_bytes'):
            host_code_gb = bytes_to_gb(mem_analysis.host_generated_code_size_in_bytes)
            print(f"  Generated Code:     {host_code_gb:>8.3f} GB  ({mem_analysis.host_generated_code_size_in_bytes:,} bytes)")

        if hasattr(mem_analysis, 'host_argument_size_in_bytes'):
            host_arg_gb = bytes_to_gb(mem_analysis.host_argument_size_in_bytes)
            print(f"  Arguments:          {host_arg_gb:>8.3f} GB  ({mem_analysis.host_argument_size_in_bytes:,} bytes)")

        if hasattr(mem_analysis, 'host_output_size_in_bytes'):
            host_out_gb = bytes_to_gb(mem_analysis.host_output_size_in_bytes)
            print(f"  Outputs:            {host_out_gb:>8.3f} GB  ({mem_analysis.host_output_size_in_bytes:,} bytes)")

        if hasattr(mem_analysis, 'host_temp_size_in_bytes'):
            host_temp_gb = bytes_to_gb(mem_analysis.host_temp_size_in_bytes)
            print(f"  Temp/Scratch:       {host_temp_gb:>8.3f} GB  ({mem_analysis.host_temp_size_in_bytes:,} bytes)")

        if hasattr(mem_analysis, 'host_alias_size_in_bytes'):
            host_alias_gb = bytes_to_gb(mem_analysis.host_alias_size_in_bytes)
            print(f"  Aliases:            {host_alias_gb:>8.3f} GB  ({mem_analysis.host_alias_size_in_bytes:,} bytes)")

        # Calculate total peak GPU memory
        print("\nSummary:")
        total_peak_gpu = 0
        if hasattr(mem_analysis, 'temp_size_in_bytes'):
            total_peak_gpu += mem_analysis.temp_size_in_bytes
        if hasattr(mem_analysis, 'output_size_in_bytes'):
            total_peak_gpu += mem_analysis.output_size_in_bytes
        if hasattr(mem_analysis, 'generated_code_size_in_bytes'):
            total_peak_gpu += mem_analysis.generated_code_size_in_bytes

        total_peak_gb = bytes_to_gb(total_peak_gpu)
        print(f"  Total Peak GPU Memory: ~{total_peak_gb:.2f} GB (temp + output + code)")

        print(f"\n[jax.lower().compile()] Raw Object: {mem_analysis}")

    except Exception as e:
        print(f"\n[jax.lower().compile()] Error during compilation memory analysis: {e}")
        mem_analysis = None

    print(f"{'='*60}\n")
    return mem_analysis


def detailed_memory_breakdown(state, batch_size_per_gpu, seq_len, vocab_size, d_model, n_layers):
    """
    Calculate and display detailed memory breakdown for training

    Args:
        state: TrainState object
        batch_size_per_gpu: Batch size per GPU device
        seq_len: Sequence length
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of layers
    """
    import jax.tree_util as tree

    print("\n" + "="*60)
    print("Detailed GPU Memory Breakdown (per GPU)")
    print("="*60)

    # 1. Model parameters
    param_leaves = tree.tree_leaves(state.params)
    param_bytes = sum(x.size * x.itemsize for x in param_leaves if hasattr(x, 'size'))
    param_mb = param_bytes / 1024**2
    param_gb = param_bytes / 1024**3
    print(f"1. Model Parameters: {param_mb:.2f} MB ({param_gb:.3f} GB)")

    # 2. Optimizer state (AdamW has momentum + variance = 2x params)
    opt_bytes = param_bytes * 2  # AdamW: m and v
    opt_mb = opt_bytes / 1024**2
    print(f"2. Optimizer State:  {opt_mb:.2f} MB ({opt_bytes/1024**3:.3f} GB)")

    # 3. Logits
    logits_bytes = batch_size_per_gpu * seq_len * vocab_size * 4
    logits_mb = logits_bytes / 1024**2
    print(f"3. Logits:           {logits_mb:.2f} MB ({logits_bytes/1024**3:.3f} GB)")

    # 4. Book input
    book_bytes = batch_size_per_gpu * 500 * 503 * 4  # 500 book states, 503 features
    book_mb = book_bytes / 1024**2
    print(f"4. Book Input:       {book_mb:.2f} MB ({book_bytes/1024**3:.3f} GB)")

    # 5. Estimated activations (per layer × n_layers)
    activation_bytes = batch_size_per_gpu * seq_len * d_model * n_layers * 4
    activation_mb = activation_bytes / 1024**2
    print(f"5. Activations (~):  {activation_mb:.2f} MB ({activation_bytes/1024**3:.3f} GB)")

    # 6. Estimated gradients (similar to activations)
    gradient_bytes = activation_bytes
    gradient_mb = gradient_bytes / 1024**2
    print(f"6. Gradients (~):    {gradient_mb:.2f} MB ({gradient_bytes/1024**3:.3f} GB)")

    # 7. Total theoretical
    total_bytes = param_bytes + opt_bytes + logits_bytes + book_bytes + activation_bytes + gradient_bytes
    total_gb = total_bytes / 1024**3
    print(f"---")
    print(f"Theoretical Total:   {total_gb:.2f} GB")

    # Get actual GPU memory usage
    gpu_stats = jax.local_devices()[0].memory_stats()
    if gpu_stats:
        actual_gb = gpu_stats['bytes_in_use'] / 1024**3
        print(f"Actual GPU Usage:    {actual_gb:.2f} GB (from device.memory_stats())")
        print(f"Difference:          {actual_gb - total_gb:+.2f} GB")
        print(f"  (includes JAX overhead, buffers, fragmentation)")

    print("="*60 + "\n")

    return {
        'params_mb': param_mb,
        'optimizer_mb': opt_mb,
        'logits_mb': logits_mb,
        'book_mb': book_mb,
        'activations_mb': activation_mb,
        'gradients_mb': gradient_mb,
        'total_gb': total_gb
    }
