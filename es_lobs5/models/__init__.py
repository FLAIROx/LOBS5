# ES-compatible model implementations for LOBS5
"""
ES-compatible versions of S5 models that work with HyperscaleES noiser framework.

Key classes:
- ES_S5SSM: ES-compatible S5 State Space Model
- ES_SequenceLayer: ES-compatible sequence layer (SSM + norm + activation)
- ES_StackedEncoder: ES-compatible stacked encoder
- ES_PaddedLobPredModel: ES-compatible LOB prediction model
"""

from .common import (
    PARAM, MM_PARAM, EMB_PARAM, EXCLUDED,
    ES_Parameter, ES_LayerNorm, ES_Linear, ES_MM, ES_TMM,
)
from .s5_ssm import ES_S5SSM
from .sequence_layer import ES_SequenceLayer
from .stacked_encoder import ES_StackedEncoder
from .lob_model import ES_PaddedLobPredModel, ES_LobBookModel
