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
        num_devices: Number of devices (local devices for this process)

    Returns:
        Mesh: A mesh with only the 'data' axis

    Note on multi-node:
    - In multi-node mode, jax.devices() returns ALL global devices (e.g., 40 GPUs)
    - But we need LOCAL devices for this process (e.g., 4 GPUs)
    - jax.local_devices() returns only the devices assigned to this process
    """
    from jax.experimental import mesh_utils

    # For multi-node JAX distributed training:
    # - Each process creates LOCAL mesh with its own devices
    # - Gradient sync across nodes is handled by jax.lax.psum in train_step
    # - DO NOT use global mesh with all devices (causes OOM in device_put)
    local_devs = jax.local_devices()
    devices = local_devs[:num_devices]

    if jax.process_count() > 1:
        print(f"[Sharding] Multi-node mode: Process {jax.process_index()}/{jax.process_count()}")
        print(f"[Sharding] Using {len(devices)} LOCAL devices (gradient sync via psum)")
    else:
        print(f"[Sharding] Single-node mode: Using {len(devices)} local devices")

    # Use mesh_utils.create_device_mesh (MaxText approach)
    # This properly handles device topology and returns a numpy array of devices
    actual_num_devices = len(devices)
    devices_array = mesh_utils.create_device_mesh(
        [actual_num_devices],  # 1D mesh shape - use actual device count
        devices,
    )

    mesh = Mesh(devices_array, axis_names=('data',))
    print(f"[Sharding] Created mesh with {actual_num_devices} devices along 'data' axis")
    print(f"[Sharding] Mesh shape: {mesh.shape}")
    return mesh


def create_fsdp_mesh(num_devices: int, fsdp_parallelism: int = None) -> Mesh:
    """
    Create a mesh with FSDP (Fully Sharded Data Parallel) support.

    This enables training large models that don't fit in single GPU memory by
    sharding parameters across multiple GPUs.

    Args:
        num_devices: Total number of devices
        fsdp_parallelism: Number of devices for FSDP sharding. If None, uses all devices.
                         data_parallelism = num_devices // fsdp_parallelism

    Returns:
        Mesh: A 2D mesh with ('data', 'fsdp') axes

    Example:
        4 GPUs with fsdp_parallelism=4:
          - data_parallelism=1, fsdp_parallelism=4
          - Parameters split across 4 GPUs (FSDP only)

        4 GPUs with fsdp_parallelism=2:
          - data_parallelism=2, fsdp_parallelism=2
          - 2 data-parallel groups, each with 2-way FSDP
    """
    from jax.experimental import mesh_utils

    local_devs = jax.local_devices()
    devices = local_devs[:num_devices]
    actual_num_devices = len(devices)

    if fsdp_parallelism is None:
        fsdp_parallelism = actual_num_devices

    data_parallelism = actual_num_devices // fsdp_parallelism
    if data_parallelism * fsdp_parallelism != actual_num_devices:
        raise ValueError(
            f"num_devices ({actual_num_devices}) must be divisible by "
            f"fsdp_parallelism ({fsdp_parallelism})"
        )

    # Create 2D device mesh: (data_parallelism, fsdp_parallelism)
    devices_array = mesh_utils.create_device_mesh(
        [data_parallelism, fsdp_parallelism],
        devices,
    )

    mesh = Mesh(devices_array, axis_names=('data', 'fsdp'))
    print(f"[Sharding] Created FSDP mesh: data={data_parallelism}, fsdp={fsdp_parallelism}")
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


def create_fsdp_param_sharding(mesh: Mesh, param_shape: Tuple[int, ...]) -> NamedSharding:
    """
    Create FSDP sharding for a parameter tensor.

    Strategy: Shard the largest dimension along the 'fsdp' axis.
    This maximizes memory efficiency while minimizing communication overhead.

    Args:
        mesh: JAX Mesh with 'fsdp' axis
        param_shape: Shape of the parameter tensor

    Returns:
        NamedSharding for FSDP
    """
    if 'fsdp' not in mesh.axis_names:
        # Fallback to replicated if no FSDP axis
        return NamedSharding(mesh, P(None))

    ndim = len(param_shape)
    if ndim == 0:
        # Scalar: replicate
        return NamedSharding(mesh, P())

    # Find the largest dimension to shard
    largest_dim_idx = max(range(ndim), key=lambda i: param_shape[i])

    # Create PartitionSpec with 'fsdp' on the largest dimension
    spec_list = [None] * ndim
    spec_list[largest_dim_idx] = 'fsdp'
    return NamedSharding(mesh, P(*spec_list))


def create_fsdp_param_shardings_for_pytree(params: Any, mesh: Mesh) -> Any:
    """
    Create FSDP shardings for an entire parameter pytree.

    Args:
        params: Parameter pytree (can be abstract ShapeDtypeStruct)
        mesh: JAX Mesh with 'fsdp' axis

    Returns:
        Pytree of NamedSharding with same structure as params
    """
    def get_sharding(leaf):
        if hasattr(leaf, 'shape'):
            return create_fsdp_param_sharding(mesh, leaf.shape)
        else:
            return NamedSharding(mesh, P())

    return jax.tree_util.tree_map(get_sharding, params)


def sharded_init(
    init_fn,
    mesh: Mesh,
    abstract_params: Any = None,
) -> Any:
    """
    Initialize parameters with FSDP sharding to avoid OOM on large models.

    This function:
    1. Uses jax.eval_shape to get abstract parameter shapes (if not provided)
    2. Creates FSDP shardings for each parameter
    3. Uses jax.jit with output shardings to initialize directly on devices

    Args:
        init_fn: Function that returns initialized parameters (e.g., model.init)
        mesh: JAX Mesh with 'fsdp' axis
        abstract_params: Optional pre-computed abstract params from jax.eval_shape

    Returns:
        Sharded parameters distributed across FSDP devices

    Example:
        ```python
        mesh = create_fsdp_mesh(4, fsdp_parallelism=4)

        def init_fn():
            return model.init(key, x_m=dummy_msg, x_b=dummy_book)

        # Initialize with FSDP - each GPU only allocates 1/4 of parameters
        variables = sharded_init(init_fn, mesh)
        ```
    """
    # Get abstract parameter shapes if not provided
    if abstract_params is None:
        print("[FSDP Init] Computing abstract parameter shapes...")
        abstract_params = jax.eval_shape(init_fn)

    # Create FSDP shardings for all parameters
    print("[FSDP Init] Creating parameter shardings...")
    param_shardings = jax.tree_util.tree_map(
        lambda leaf: create_fsdp_param_sharding(mesh, leaf.shape) if hasattr(leaf, 'shape') else NamedSharding(mesh, P()),
        abstract_params
    )

    # JIT-compile init_fn with output shardings
    # This makes JAX initialize parameters directly on the target devices
    # instead of initializing on one device and then transferring
    print("[FSDP Init] Initializing with sharded output...")

    @jax.jit
    def sharded_init_fn():
        return init_fn()

    # Use jax.jit with out_shardings to control output placement
    sharded_init_jit = jax.jit(init_fn, out_shardings=param_shardings)

    with mesh:
        params = sharded_init_jit()

    print("[FSDP Init] Initialization complete")
    return params


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
    return jax.tree_util.tree_map(lambda x: x if not isinstance(x, jax.Array) else x, pytree)


def create_state_shardings(state: Any, mesh: Mesh) -> Any:
    """
    Create shardings for train state.

    Why this approach:
    - train_step needs to know input/output sharding
    - In phase 1, we replicate all state (params, opt_state)
    - This replicates pmap behavior but with explicit sharding
    - Special handling: scalars (rank 0) need P(), arrays need P(None)

    Args:
        state: TrainState
        mesh: JAX Mesh

    Returns:
        Sharding pytree with same structure as state
    """
    def get_sharding_for_leaf(leaf):
        """
        Get appropriate sharding for a leaf node.

        - Scalars (rank 0): Use P() for full replication
        - Arrays (rank > 0): Use P(None) for full replication
        """
        if isinstance(leaf, jax.Array):
            # Check if this is a scalar (rank 0)
            if leaf.ndim == 0:
                # Scalar: use P() (empty PartitionSpec)
                return NamedSharding(mesh, P())
            else:
                # Array: use P(None) for replication
                return NamedSharding(mesh, P(None))
        else:
            # Non-array leaf (e.g., static values, None): use scalar sharding
            return NamedSharding(mesh, P())

    return jax.tree_util.tree_map(get_sharding_for_leaf, state)


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
