# ES-compatible model implementations for LOBS5
"""
ES-compatible versions of S5 models that work with HyperscaleES noiser framework.

Key classes:
- ES_S5SSM: ES-compatible S5 State Space Model (from adapters/ssm_adapter.py)
- ES_SequenceLayer: ES-compatible sequence layer (SSM + norm + activation)
- ES_StackedEncoder: ES-compatible stacked encoder
- ES_PaddedLobPredModel: ES-compatible LOB prediction model
"""

from .common import (
    PARAM, MM_PARAM, EMB_PARAM, EXCLUDED,
    ES_Parameter, ES_LayerNorm, ES_Linear, ES_MM, ES_TMM,
    Model, CommonInit, CommonParams,
    merge_inits, merge_frozen, call_submodule,
)
from .s5_layer import ES_SequenceLayer
from .encoder import ES_StackedEncoder
from .lob_model import ES_PaddedLobPredModel, ES_LobBookModel
