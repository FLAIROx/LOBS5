"""
Sharding utilities for LOBS5 training.
Provides helpers for creating mesh and shardings for jax.jit + shardings migration.

This module replaces the implicit parallelism of jax.pmap with explicit
mesh-based sharding, following the MaxText approach.

Extracted from ssm_stable branch — only data-parallel functions (no FSDP).
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

    pmap implicitly parallelizes over the first axis.
    We explicitly create a mesh with only a 'data' axis to replicate the same behavior.

    In multi-node mode, each process creates LOCAL mesh with its own devices.
    Gradient sync across nodes would require psum across processes (not yet implemented).
    """
    from jax.experimental import mesh_utils

    local_devs = jax.local_devices()
    devices = local_devs[:num_devices]

    if jax.process_count() > 1:
        print(f"[Sharding] Multi-node mode: Process {jax.process_index()}/{jax.process_count()}")
        print(f"[Sharding] Using {len(devices)} LOCAL devices (gradient sync via psum)")
    else:
        print(f"[Sharding] Single-node mode: Using {len(devices)} local devices")

    actual_num_devices = len(devices)
    devices_array = mesh_utils.create_device_mesh(
        [actual_num_devices],
        devices,
    )

    mesh = Mesh(devices_array, axis_names=('data',))
    print(f"[Sharding] Created mesh with {actual_num_devices} devices along 'data' axis")
    return mesh


def initialize_mesh(num_devices: int) -> Mesh:
    """Initialize the global mesh. Call once at training start."""
    global _GLOBAL_MESH
    _GLOBAL_MESH = create_simple_mesh(num_devices)
    return _GLOBAL_MESH


def get_global_mesh() -> Mesh:
    """Get the global mesh."""
    global _GLOBAL_MESH
    if _GLOBAL_MESH is None:
        raise RuntimeError("Mesh not initialized. Call initialize_mesh() first.")
    return _GLOBAL_MESH


def create_data_sharding(mesh: Mesh) -> NamedSharding:
    """
    Create sharding for data: batch dimension sharded along 'data' axis.
    P('data', None) = first dim sharded, rest replicated.
    """
    return NamedSharding(mesh, P('data', None))


def create_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Create fully replicated sharding (for model parameters)."""
    return NamedSharding(mesh, P(None))


def tree_replicate_to_devices(pytree: Any, sharding: NamedSharding) -> Any:
    """Replicate a pytree to all devices. Replacement for jax_utils.replicate()."""
    return jax.device_put(pytree, sharding)


def tree_unreplicate(pytree: Any) -> Any:
    """
    Extract a single copy from a replicated pytree.
    Replacement for jax_utils.unreplicate().
    For jit+shardings, replicated arrays are already normal arrays — no-op.
    """
    return pytree


def create_state_shardings(state: Any, mesh: Mesh) -> Any:
    """
    Create shardings for train state. All parameters replicated.
    Scalars (rank 0) use P(), arrays use P(None).
    """
    def get_sharding_for_leaf(leaf):
        if isinstance(leaf, jax.Array):
            if leaf.ndim == 0:
                return NamedSharding(mesh, P())
            else:
                return NamedSharding(mesh, P(None))
        else:
            return NamedSharding(mesh, P())

    return jax.tree_util.tree_map(get_sharding_for_leaf, state)


def get_data_shardings_for_batch(
        mesh: Mesh,
        has_book_data: bool = True,
    ) -> Tuple[Any, Any, Any]:
    """
    Create shardings for batch data components.
    Returns (inputs_sharding, labels_sharding, integration_times_sharding).
    """
    data_sharding_2d = create_data_sharding(mesh)
    data_sharding_1d = NamedSharding(mesh, P('data'))

    if has_book_data:
        inputs_sharding = (data_sharding_2d, data_sharding_2d)
    else:
        inputs_sharding = (data_sharding_2d,)

    labels_sharding = data_sharding_1d

    if has_book_data:
        integration_times_sharding = (data_sharding_2d, data_sharding_2d)
    else:
        integration_times_sharding = (data_sharding_2d,)

    return inputs_sharding, labels_sharding, integration_times_sharding
