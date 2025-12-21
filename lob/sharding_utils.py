"""
Sharding utilities for LOBS5 training.
Provides helpers for creating mesh and shardings for jax.jit + shardings migration.

This module replaces the implicit parallelism of jax.pmap with explicit
mesh-based sharding, following the MaxText approach.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from typing import Any, Tuple, Optional


# Global mesh storage
_GLOBAL_MESH = None


def create_simple_mesh(num_devices: int) -> Mesh:
    """
    Create a simple data-parallel mesh.

    Why this approach:
    - pmap implicitly parallelizes over the first axis
    - We explicitly create a mesh with only a 'data' axis to replicate the same behavior
    - This is the minimal change approach for migrating from pmap

    Args:
        num_devices: Number of devices

    Returns:
        Mesh: A mesh with only the 'data' axis
    """
    devices = jax.devices()[:num_devices]
    # Arrange devices in a 1D array with axis name 'data'
    device_array = jnp.array(devices).reshape(-1)
    mesh = Mesh(device_array, axis_names=('data',))
    print(f"[Sharding] Created mesh with {num_devices} devices along 'data' axis")
    print(f"[Sharding] Mesh shape: {mesh.shape}")
    return mesh


def initialize_mesh(num_devices: int) -> Mesh:
    """
    Initialize the global mesh.

    Why needed:
    - jax.jit + shardings requires mesh to be defined outside functions
    - train_step will be jit-compiled and needs access to mesh
    - We initialize once at training start, then use globally

    Args:
        num_devices: Number of devices

    Returns:
        Mesh: Created mesh
    """
    global _GLOBAL_MESH
    _GLOBAL_MESH = create_simple_mesh(num_devices)
    print(f"[Mesh] Initialized global mesh with {num_devices} devices")
    return _GLOBAL_MESH


def get_global_mesh() -> Mesh:
    """
    Get the global mesh.

    Returns:
        Mesh: Global mesh

    Raises:
        RuntimeError: If mesh not initialized
    """
    global _GLOBAL_MESH
    if _GLOBAL_MESH is None:
        raise RuntimeError("Mesh not initialized. Call initialize_mesh() first.")
    return _GLOBAL_MESH


def create_data_sharding(mesh: Mesh) -> NamedSharding:
    """
    Create sharding for data.

    Why this approach:
    - The first dimension (batch) should be sharded along the 'data' axis
    - This replicates pmap's behavior: each device processes part of the batch

    Args:
        mesh: JAX Mesh

    Returns:
        NamedSharding: Sharding for input data
    """
    # P('data', None, ...) means:
    # - First dimension (batch) is sharded along 'data' axis
    # - Other dimensions are not sharded (each device has full copy)
    return NamedSharding(mesh, P('data', None))


def create_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """
    Create fully replicated sharding.

    Why this approach:
    - Model parameters and optimizer state should be replicated across all devices
    - This replicates jax_utils.replicate() behavior

    Args:
        mesh: JAX Mesh

    Returns:
        NamedSharding: Fully replicated sharding
    """
    # P(None) means all dimensions are not sharded (replicated across all devices)
    return NamedSharding(mesh, P(None))


def tree_replicate_to_devices(pytree: Any, sharding: NamedSharding) -> Any:
    """
    Replicate a pytree to all devices.

    This is a replacement for jax_utils.replicate(), using sharding instead of pmap.

    Why this approach:
    - jax_utils.replicate() is designed for pmap, adds device dimension
    - We use jax.device_put with sharding to achieve the same effect
    - Key difference: sharding version doesn't change array shape, only distribution

    Args:
        pytree: Pytree to replicate
        sharding: Replication sharding

    Returns:
        Pytree replicated to all devices
    """
    return jax.device_put(pytree, sharding)


def tree_unreplicate(pytree: Any) -> Any:
    """
    Extract a single copy from a replicated pytree.

    This is a replacement for jax_utils.unreplicate().

    Why this approach:
    - pmap returns arrays with device count in first dimension
    - jit + shardings replicated arrays need different extraction

    Args:
        pytree: Replicated pytree

    Returns:
        Single copy
    """
    # For replicated arrays, values are identical on all devices
    # We just need to get the value (JAX handles this automatically)
    return jax.tree_map(lambda x: x if not isinstance(x, jax.Array) else x, pytree)


def create_state_shardings(state: Any, mesh: Mesh) -> Any:
    """
    Create shardings for train state.

    Why this approach:
    - train_step needs to know input/output sharding
    - In phase 1, we replicate all state (params, opt_state)
    - This replicates pmap behavior but with explicit sharding

    Args:
        state: TrainState
        mesh: JAX Mesh

    Returns:
        Sharding pytree with same structure as state
    """
    replicated = create_replicated_sharding(mesh)
    # Replicate entire state structure, all leaf nodes use replicated sharding
    return jax.tree_map(lambda _: replicated, state)


def get_data_shardings_for_batch(
        mesh: Mesh,
        has_book_data: bool = True,
    ) -> Tuple[Any, Any, Any]:
    """
    Create shardings for batch data.

    Why this approach:
    - LOBS5 batch has multiple components (inputs, labels, integration_times)
    - Each component needs appropriate sharding
    - inputs may be tuple (messages, books)

    Args:
        mesh: JAX Mesh
        has_book_data: Whether book data is present

    Returns:
        (inputs_sharding, labels_sharding, integration_times_sharding)
    """
    # 2D data sharding: P('data', None)
    # - First dimension (batch) sharded along 'data' axis
    # - Second dimension not sharded
    data_sharding_2d = create_data_sharding(mesh)

    # 1D data sharding: P('data')
    # - Only dimension sharded along 'data' axis
    data_sharding_1d = NamedSharding(mesh, P('data'))

    # inputs sharding
    if has_book_data:
        # inputs is tuple: (messages, books)
        # Both sharded along batch dimension
        inputs_sharding = (data_sharding_2d, data_sharding_2d)
    else:
        # inputs is single array (messages,)
        inputs_sharding = (data_sharding_2d,)

    # labels sharding: 1D array, sharded along data axis
    labels_sharding = data_sharding_1d

    # integration_times sharding
    if has_book_data:
        # Two timestep arrays
        integration_times_sharding = (data_sharding_2d, data_sharding_2d)
    else:
        # One timestep array
        integration_times_sharding = (data_sharding_2d,)

    return inputs_sharding, labels_sharding, integration_times_sharding
