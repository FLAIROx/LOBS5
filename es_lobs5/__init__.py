# ES-LOBS5: Evolution Strategies for LOB Prediction with S5 Models
"""
This module provides ES (Evolution Strategies) training infrastructure
for LOBS5 S5-based sequence models.

Main components:
- models/: ES-compatible model implementations (S5SSM, SequenceLayer, etc.)
- training/: ES training loop and fitness evaluation
- scripts/: Training launch scripts

Usage:
    from es_lobs5.models import ES_PaddedLobPredModel
    from es_lobs5.training import es_train
"""

# Lazy imports to avoid import-time dependencies
# Import explicitly when needed:
# from es_lobs5 import models
# from es_lobs5 import training
