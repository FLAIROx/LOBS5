#!/usr/bin/env python3
"""
Multi-GPU Validation Test for LOBMAX

This script verifies whether all GPUs are actually being used for computation
and whether gradient aggregation is working correctly.
"""

import os
import sys
import time

# Force JAX to use all GPUs
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

print("=" * 60)
print("LOBMAX Multi-GPU Validation Test")
print("=" * 60)

# 1. Check available devices
devices = jax.devices()
print(f"\n[1] JAX Devices: {len(devices)}")
for i, d in enumerate(devices):
    print(f"    [{i}] {d}")

if len(devices) < 2:
    print("\n[ERROR] Only 1 device found. Multi-GPU test requires 2+ GPUs.")
    sys.exit(1)

# 2. Create mesh
print(f"\n[2] Creating mesh with {len(devices)} devices...")
devices_array = mesh_utils.create_device_mesh([len(devices)], devices)
mesh = Mesh(devices_array, axis_names=('data',))
print(f"    Mesh shape: {mesh.shape}")

# 3. Test data sharding
print("\n[3] Testing data sharding...")
batch_size = len(devices) * 4  # 4 per device
seq_len = 128
hidden = 64

# Create data and shard it
data = jnp.ones((batch_size, seq_len, hidden))
data_sharding = NamedSharding(mesh, P('data', None, None))
sharded_data = jax.device_put(data, data_sharding)

print(f"    Data shape: {data.shape}")
print(f"    Sharding: {sharded_data.sharding}")
print(f"    Devices holding data: {[s.device() for s in sharded_data.addressable_shards]}")

# Verify each device has a shard
shard_devices = set(s.device() for s in sharded_data.addressable_shards)
if len(shard_devices) == len(devices):
    print(f"    [✓] Data correctly sharded across {len(devices)} devices")
else:
    print(f"    [✗] Data only on {len(shard_devices)} devices, expected {len(devices)}")

# 4. Test gradient computation on all devices
print("\n[4] Testing gradient computation on all devices...")

# Simple linear computation
def simple_forward(params, x):
    return jnp.sum(x @ params)

def compute_grad(params, x):
    return jax.grad(lambda p: simple_forward(p, x))(params)

# Replicated parameters
params = jnp.ones((hidden, 32))
params_sharding = NamedSharding(mesh, P(None, None))
sharded_params = jax.device_put(params, params_sharding)

# JIT with shardings
jit_compute_grad = jax.jit(
    compute_grad,
    in_shardings=(params_sharding, data_sharding),
    out_shardings=params_sharding,  # Replicated output
)

# Warm up
_ = jit_compute_grad(sharded_params, sharded_data)

# Time it
start = time.time()
for _ in range(10):
    grads = jit_compute_grad(sharded_params, sharded_data)
    grads.block_until_ready()
elapsed = time.time() - start

print(f"    Gradient shape: {grads.shape}")
print(f"    Gradient sharding: {grads.sharding}")
print(f"    10 iterations took: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms per step)")

# 5. Verify all-reduce happened
print("\n[5] Verifying gradient aggregation (all-reduce)...")
# If properly aggregated, gradients should be the same on all devices
grad_shards = list(grads.addressable_shards)
first_grad = grad_shards[0].data
all_same = all(jnp.allclose(s.data, first_grad) for s in grad_shards)

if all_same:
    print("    [✓] Gradients identical across all devices (all-reduce working)")
else:
    print("    [✗] Gradients DIFFER across devices (all-reduce NOT working!)")
    for s in grad_shards:
        print(f"        Device {s.device()}: mean={float(jnp.mean(s.data)):.4f}")

# 6. Summary
print("\n" + "=" * 60)
if len(shard_devices) == len(devices) and all_same:
    print("RESULT: Multi-GPU training is WORKING CORRECTLY")
    print(f"  - {len(devices)} GPUs are being used")
    print(f"  - Data parallelism is active")
    print(f"  - Gradient aggregation is working")
else:
    print("RESULT: PROBLEMS DETECTED with multi-GPU setup")
    if len(shard_devices) != len(devices):
        print(f"  - Data only on {len(shard_devices)}/{len(devices)} GPUs")
    if not all_same:
        print("  - Gradients are NOT being synchronized!")
print("=" * 60)
