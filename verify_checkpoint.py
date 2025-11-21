#!/usr/bin/env python3
"""
Quick verification script to check if checkpoint is working
"""

import jax
import jax.numpy as jnp
from functools import partial
import sys

# Test remat functionality
print("Testing JAX remat functionality...")
print("=" * 50)

# Check JAX environment settings
import os
print(f"JAX_DISABLE_JIT: {os.environ.get('JAX_DISABLE_JIT', 'Not set (JIT enabled)')}")
print(f"JAX_CHECKPOINT_POLICY: {os.environ.get('JAX_CHECKPOINT_POLICY', 'Not set')}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")
print()

# Simple test function that uses memory
def expensive_function(x):
    """Simulates an expensive operation"""
    for _ in range(10):
        x = jnp.sin(x) * jnp.cos(x) + x
    return x

# Test without remat
print("1. Testing WITHOUT remat:")
test_input = jnp.ones((1000, 1000))

try:
    # Compile without remat
    regular_fn = jax.jit(expensive_function)
    lowered = regular_fn.lower(test_input)
    compiled = lowered.compile()
    print(f"   Compilation successful")

    # Try to get memory info (may not be available)
    try:
        cost = compiled.cost_analysis()
        if cost:
            print(f"   Estimated memory: {cost}")
    except:
        pass

except Exception as e:
    print(f"   Error: {e}")

print()

# Test with remat
print("2. Testing WITH remat:")
try:
    # Compile with remat
    remat_fn = jax.jit(jax.remat(expensive_function,
                                  policy=jax.checkpoint_policies.nothing_saveable))
    lowered = remat_fn.lower(test_input)
    compiled = lowered.compile()
    print(f"   Compilation successful with remat")

    # Try to get memory info
    try:
        cost = compiled.cost_analysis()
        if cost:
            print(f"   Estimated memory with remat: {cost}")
    except:
        pass

except Exception as e:
    print(f"   Error: {e}")

print()
print("3. Testing associative_scan with remat:")

# Test associative scan specifically
def binary_op(a, b):
    return a + b

test_seq = jnp.ones((500, 100))  # 500 sequence length

try:
    # Without remat
    regular_scan = partial(jax.lax.associative_scan, binary_op)
    result1 = regular_scan((test_seq, test_seq))
    print(f"   Regular scan: Success")
except Exception as e:
    print(f"   Regular scan failed: {e}")

try:
    # With remat (as implemented in ssm.py)
    remat_scan = jax.remat(
        partial(jax.lax.associative_scan, binary_op),
        policy=jax.checkpoint_policies.nothing_saveable
    )
    result2 = remat_scan((test_seq, test_seq))
    print(f"   Remat scan: Success")
except Exception as e:
    print(f"   Remat scan failed: {e}")

print()
print("=" * 50)
print("Checkpoint verification complete!")
print()
print("Next steps:")
print("1. Run: ./test_m1_fix.sh")
print("2. Monitor GPU memory: watch -n 1 'nvidia-smi | grep python'")
print("3. Check logs for 'Checkpoint activated' messages")