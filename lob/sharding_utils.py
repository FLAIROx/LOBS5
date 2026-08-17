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


def create_simple_mesh(num_devices: int, hierarchical: bool = False) -> Mesh:
    """
    Create a data-parallel mesh.

    hierarchical=False (default): 1D mesh ('data',) — flat AllReduce
    hierarchical=True (multi-node): 2D mesh ('nodes', 'gpus') — enables
      shard_map with per-axis pmean for hierarchical AllReduce:
      pmean('gpus') via NVLink within node, pmean('nodes') via Slingshot.

    Single-node: always 1D mesh (hierarchical ignored).
    """
    from jax.experimental import mesh_utils
    import numpy as np

    if jax.process_count() > 1:
        # Global mesh: use ALL devices across all nodes
        devices = jax.devices()
        num_nodes = jax.process_count()
        gpus_per_node = len(jax.local_devices())
        print(f"[Sharding] Multi-node mode: Process {jax.process_index()}/{num_nodes}")
        print(f"[Sharding] Global mesh with {len(devices)} devices across {num_nodes} processes")

        if hierarchical:
            # 2D mesh: (nodes, gpus) for shard_map hierarchical AllReduce
            devices_2d = np.array(devices).reshape(num_nodes, gpus_per_node)
            mesh = Mesh(devices_2d, axis_names=('nodes', 'gpus'))
            print(f"[Sharding] 2D hierarchical mesh: ({num_nodes} nodes, {gpus_per_node} gpus)")
            return mesh
    else:
        # Single-node: use local devices
        local_devs = jax.local_devices()
        devices = local_devs[:num_devices]
        print(f"[Sharding] Single-node mode: Using {len(devices)} local devices")
        if hierarchical:
            devices_2d = np.array(devices).reshape(1, len(devices))
            mesh = Mesh(devices_2d, axis_names=("nodes", "gpus"))
            print(f"[Sharding] 2D hierarchical mesh: (1 nodes, {len(devices)} gpus)")
            return mesh

    devices_array = np.array(devices).reshape(-1)
    mesh = Mesh(devices_array, axis_names=('data',))
    print(f"[Sharding] Created 1D mesh with {len(devices)} devices along 'data' axis")
    return mesh


def initialize_mesh(num_devices: int, hierarchical: bool = False) -> Mesh:
    """Initialize the global mesh. Call once at training start."""
    global _GLOBAL_MESH
    _GLOBAL_MESH = create_simple_mesh(num_devices, hierarchical=hierarchical)
    return _GLOBAL_MESH


def get_global_mesh() -> Mesh:
    """Get the global mesh."""
    global _GLOBAL_MESH
    if _GLOBAL_MESH is None:
        raise RuntimeError("Mesh not initialized. Call initialize_mesh() first.")
    return _GLOBAL_MESH


def _get_batch_axis(mesh: Mesh):
    """Return the batch PartitionSpec axis based on mesh dimensionality.

    1D mesh ('data',): returns 'data'
    2D mesh ('nodes', 'gpus'): returns ('nodes', 'gpus') — shards across both
    """
    if len(mesh.axis_names) == 2 and 'nodes' in mesh.axis_names:
        return ('nodes', 'gpus')
    return 'data'


def create_data_sharding(mesh: Mesh) -> NamedSharding:
    """
    Create sharding for data: batch dimension sharded along all mesh axes.
    1D: P('data', None), 2D: P(('nodes','gpus'), None)
    """
    batch_axis = _get_batch_axis(mesh)
    return NamedSharding(mesh, P(batch_axis, None))


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
    Auto-detects 1D vs 2D mesh for correct PartitionSpecs.
    """
    batch_axis = _get_batch_axis(mesh)
    data_sharding_2d = create_data_sharding(mesh)
    data_sharding_1d = NamedSharding(mesh, P(batch_axis))

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
