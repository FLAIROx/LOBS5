#!/usr/bin/env python
"""
Flash Attention Runtime Check Script for LOBMAX

This script verifies that Flash Attention is properly configured and working.
Run this before training to ensure Flash Attention is enabled.
"""

import os
import sys

def check_flash_attention():
    """
    Check Flash Attention availability and configuration.
    Returns True if Flash Attention is working, False otherwise.
    """
    print("=" * 60)
    print("LOBMAX Flash Attention Runtime Check")
    print("=" * 60)
    
    # Step 1: Check environment
    print("\n[1] Environment Check:")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    jax_platforms = os.environ.get("JAX_PLATFORMS", "not set")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    print(f"    CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"    JAX_PLATFORMS: {jax_platforms}")
    print(f"    XLA_FLAGS: {xla_flags if xla_flags else 'not set'}")
    
    # Step 2: Check JAX and CUDA
    print("\n[2] JAX Device Check:")
    try:
        import jax
        import jax.numpy as jnp
        
        devices = jax.devices()
        print(f"    JAX version: {jax.__version__}")
        print(f"    Available devices: {len(devices)}")
        for i, d in enumerate(devices):
            print(f"      [{i}] {d.platform}: {d}")
        
        # Check if GPU is available
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if not gpu_devices:
            print("    [!] WARNING: No GPU devices found!")
            return False
            
        print(f"    GPU devices: {len(gpu_devices)}")
    except Exception as e:
        print(f"    [!] ERROR: JAX initialization failed: {e}")
        return False
    
    # Step 3: Check cuDNN
    print("\n[3] cuDNN Check:")
    try:
        # Check cuDNN version via cudnn library
        cudnn_lib = os.environ.get("LD_LIBRARY_PATH", "")
        if "cudnn" in cudnn_lib.lower():
            print(f"    cuDNN library path found in LD_LIBRARY_PATH")
        
        # Try to get cuDNN version from JAX's internal mechanism
        from jax._src.cudnn.fused_attention_stablehlo import check_cudnn_version
        print("    cuDNN Flash Attention module: Available")
    except ImportError:
        print("    cuDNN Flash Attention module: Not available (older JAX version)")
    except Exception as e:
        print(f"    cuDNN info: {e}")
    
    # Step 4: Check XLA Flags for Flash Attention
    print("\n[4] XLA Flash Attention Config:")
    cudnn_fmha_enabled = "--xla_gpu_enable_cudnn_fmha=true" in xla_flags
    if cudnn_fmha_enabled:
        print("    xla_gpu_enable_cudnn_fmha: ENABLED")
    else:
        print("    xla_gpu_enable_cudnn_fmha: Not explicitly set (may use default)")
    
    # Step 5: Test Flash Attention with a small kernel
    print("\n[5] Flash Attention Kernel Test:")
    try:
        # Create a small test to verify Flash Attention works
        key = jax.random.PRNGKey(0)
        
        # Test dimensions (must be divisible by 128 for Flash Attention)
        B, T, H, D = 2, 128, 8, 64  # batch, seq_len, heads, head_dim
        
        q = jax.random.normal(key, (B, T, H, D), dtype=jnp.bfloat16)
        k = jax.random.normal(key, (B, T, H, D), dtype=jnp.bfloat16)
        v = jax.random.normal(key, (B, T, H, D), dtype=jnp.bfloat16)
        
        # Test 1: Try explicit cuDNN implementation
        try:
            from jax.nn import dot_product_attention
            
            # cuDNN Flash Attention
            result_cudnn = dot_product_attention(q, k, v, implementation="cudnn")
            print(f"    cuDNN Flash Attention: WORKING")
            print(f"    Output shape: {result_cudnn.shape}")
            print(f"    Output dtype: {result_cudnn.dtype}")
            cudnn_works = True
        except Exception as e:
            print(f"    cuDNN Flash Attention: FAILED - {e}")
            cudnn_works = False
        
        # Test 2: Try XLA (fallback) implementation
        try:
            result_xla = dot_product_attention(q, k, v, implementation="xla")
            print(f"    XLA Attention (fallback): WORKING")
            xla_works = True
        except Exception as e:
            print(f"    XLA Attention: FAILED - {e}")
            xla_works = False
        
        # Test 3: Check which implementation is actually used
        print("\n[6] Active Implementation Check:")
        if cudnn_works:
            print("    [✓] cuDNN Flash Attention is AVAILABLE and WORKING")
            print("    Expected kernel: cuDNN Flash Attention (fused)")
        elif xla_works:
            print("    [!] cuDNN not available, using XLA (slower)")
            print("    Expected kernel: XLA Attention")
        else:
            print("    [X] No attention implementation is working!")
            return False
            
    except Exception as e:
        print(f"    [!] Flash Attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Memory bandwidth hint
    print("\n[7] Memory Configuration:")
    mem_fraction = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "not set")
    preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "not set")
    print(f"    XLA_PYTHON_CLIENT_MEM_FRACTION: {mem_fraction}")
    print(f"    XLA_PYTHON_CLIENT_PREALLOCATE: {preallocate}")
    
    # Final summary
    print("\n" + "=" * 60)
    if cudnn_works:
        print("RESULT: Flash Attention is ENABLED and WORKING")
        print("The training will use cuDNN Flash Attention for optimal performance.")
    else:
        print("RESULT: Flash Attention is NOT AVAILABLE")
        print("The training will use XLA attention (slower).")
        print("\nTo enable Flash Attention:")
        print("  1. Ensure cuDNN 8.9+ is installed")
        print("  2. Set XLA_FLAGS='--xla_gpu_enable_cudnn_fmha=true'")
        print("  3. Use JAX 0.4.20+ with CUDA 12+")
    print("=" * 60)
    
    return cudnn_works


def print_lobmax_attention_config(attention_kernel: str):
    """Print the attention configuration for LOBMAX."""
    print(f"\n[LOBMAX] Configured attention_kernel: {attention_kernel}")
    
    if attention_kernel == "flash":
        print("[LOBMAX] Using Flash Attention (cuDNN/XLA auto-select)")
    elif attention_kernel == "cudnn_flash_te":
        print("[LOBMAX] Using cuDNN Flash Attention with Transformer Engine")
    elif attention_kernel == "dot_product":
        print("[LOBMAX] Using standard dot product attention (memory intensive)")
    else:
        print(f"[LOBMAX] Unknown attention kernel: {attention_kernel}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_kernel", type=str, default="flash",
                       help="Attention kernel type (flash, cudnn_flash_te, dot_product)")
    args = parser.parse_args()
    
    # Run the check
    is_working = check_flash_attention()
    print_lobmax_attention_config(args.attention_kernel)
    
    # Exit with appropriate code
    sys.exit(0 if is_working else 1)
