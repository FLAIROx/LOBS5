# ES Training modules for LOBS5
"""
ES training implementations using JaxLOB environment.

Main components:
- ESTrainer: Core ES training loop with JaxLOB environment
- Fitness functions: PnL-based fitness computation

Note: ESTrainer is lazy-loaded to avoid import errors from optional dependencies.
"""

from .fitness import (
    compute_pnl_fitness,
    compute_execution_fitness,
    compute_advantage_fitness,
    compute_normalized_pnl_fitness,
    compute_cross_entropy_fitness,
)

# Lazy import for trainer to avoid optax/gymnax dependency issues
_ESTrainer = None
_create_es_config = None
_es_train = None


def get_trainer():
    """Lazy load ESTrainer and related functions."""
    global _ESTrainer, _create_es_config, _es_train
    if _ESTrainer is None:
        from .es_trainer import ESTrainer, create_es_config, es_train
        _ESTrainer = ESTrainer
        _create_es_config = create_es_config
        _es_train = es_train
    return _ESTrainer, _create_es_config, _es_train


def __getattr__(name):
    """Lazy attribute access for ESTrainer, create_es_config, es_train."""
    if name == 'ESTrainer':
        ESTrainer, _, _ = get_trainer()
        return ESTrainer
    elif name == 'create_es_config':
        _, create_es_config, _ = get_trainer()
        return create_es_config
    elif name == 'es_train':
        _, _, es_train = get_trainer()
        return es_train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main trainer (lazy-loaded)
    'ESTrainer',
    'create_es_config',
    'es_train',
    # Fitness functions
    'compute_pnl_fitness',
    'compute_execution_fitness',
    'compute_advantage_fitness',
    'compute_normalized_pnl_fitness',
    'compute_cross_entropy_fitness',
]
