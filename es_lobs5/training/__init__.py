# ES training infrastructure for LOBS5
"""
ES training with JaxLOB for execution optimization.

Pure ES with step-by-step JaxLOB simulation:
- World Model (frozen) generates background market order flow
- Policy (ES trained) observes and generates trading actions
- Fitness = total_revenue (PnL)
"""

# PnL-based fitness
from .fitness import (
    compute_pnl_fitness,
    compute_execution_fitness,
    compute_advantage_fitness,
)

# ES training with JaxLOB
from .es_jaxlob_train import es_jaxlob_train, ESJaxLOBTrainer, create_es_jaxlob_config
