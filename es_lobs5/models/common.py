"""
Common ES building blocks for S5 models.

Imports most components from HyperscaleES, only adds S5-specific ones:
- ES_LayerNorm (not in HyperscaleES)
- ES_Parameter with es_type support (for EXCLUDED params)
"""

import jax
import jax.numpy as jnp

# Import from centralized import_utils to avoid duplication
from ..utils.import_utils import get_base_model, get_common, get_hyperscalees_path

_base_model = get_base_model()
_common = get_common()
_hyperscalees_path = get_hyperscalees_path()

# Extract what we need from loaded modules
Model = _base_model.Model
CommonInit = _base_model.CommonInit
CommonParams = _base_model.CommonParams

PARAM = _common.PARAM
MM_PARAM = _common.MM_PARAM
EMB_PARAM = _common.EMB_PARAM
EXCLUDED = _common.EXCLUDED
merge_inits = _common.merge_inits
merge_frozen = _common.merge_frozen
call_submodule = _common.call_submodule
simple_es_tree_key = _common.simple_es_tree_key
recursive_scan_split = _common.recursive_scan_split
layer_norm = _common.layer_norm
ACTIVATIONS = _common.ACTIVATIONS
Parameter = _common.Parameter
MM = _common.MM
TMM = _common.TMM
Linear = _common.Linear
Embedding = _common.Embedding

# Re-export HyperscaleES components with ES_ prefix for consistency
ES_MM = MM
ES_TMM = TMM
ES_Embedding = Embedding

__all__ = [
    'Model', 'CommonInit', 'CommonParams',
    'PARAM', 'MM_PARAM', 'EMB_PARAM', 'EXCLUDED',
    'merge_inits', 'merge_frozen', 'call_submodule',
    'simple_es_tree_key', 'recursive_scan_split',
    'ES_Parameter', 'ES_MM', 'ES_TMM', 'ES_Linear', 'ES_LayerNorm',
    'ACTIVATIONS',
]

# Add gelu to ACTIVATIONS if not present
if 'gelu' not in ACTIVATIONS:
    ACTIVATIONS['gelu'] = jax.nn.gelu


# =============================================================================
# S5-specific components (not in HyperscaleES)
# =============================================================================

class ES_Parameter(Model):
    """
    ES-compatible parameter with es_type support.

    Extends HyperscaleES Parameter to support es_type argument
    for EXCLUDED params (e.g., LayerNorm weights).
    """

    @classmethod
    def rand_init(cls, key, shape=None, scale=1.0, raw_value=None,
                  dtype=jnp.float32, es_type=PARAM) -> CommonInit:
        if raw_value is not None:
            params = jnp.asarray(raw_value).astype(dtype)
        else:
            params = (jax.random.normal(key, shape) * scale).astype(dtype)

        return CommonInit(
            frozen_params=None,
            params=params,
            scan_map=(),
            es_map=es_type  # Support EXCLUDED, not just PARAM
        )

    @classmethod
    def _forward(cls, common_params: CommonParams):
        return common_params.noiser.get_noisy_standard(
            common_params.frozen_noiser_params,
            common_params.noiser_params,
            common_params.params,
            common_params.es_tree_key,
            common_params.iterinfo
        )


class ES_Linear(Model):
    """
    ES-compatible linear layer using ES_Parameter for bias.
    """

    @classmethod
    def rand_init(cls, key, in_dim: int, out_dim: int,
                  use_bias: bool = True, dtype=jnp.float32,
                  scale=None) -> CommonInit:
        key_w, key_b = jax.random.split(key)

        if use_bias:
            return merge_inits(
                weight=MM.rand_init(key_w, in_dim, out_dim, dtype),
                bias=ES_Parameter.rand_init(
                    key_b, raw_value=jnp.zeros(out_dim, dtype=dtype), dtype=dtype
                )
            )
        else:
            return merge_inits(
                weight=MM.rand_init(key_w, in_dim, out_dim, dtype),
            )

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        out = call_submodule(MM, 'weight', common_params, x)
        if 'bias' in common_params.params:
            out = out + call_submodule(ES_Parameter, 'bias', common_params)
        return out


class ES_LayerNorm(Model):
    """
    ES-compatible Layer Normalization.

    Not in HyperscaleES, needed for S5 models.
    weight/bias are EXCLUDED by default for stability.
    """

    @classmethod
    def rand_init(cls, key, dim: int, dtype=jnp.float32,
                  use_bias: bool = True, es_type=EXCLUDED) -> CommonInit:
        key_w, key_b = jax.random.split(key)

        if use_bias:
            return merge_inits(
                weight=ES_Parameter.rand_init(
                    key_w, raw_value=jnp.ones(dim, dtype=dtype),
                    dtype=dtype, es_type=es_type
                ),
                bias=ES_Parameter.rand_init(
                    key_b, raw_value=jnp.zeros(dim, dtype=dtype),
                    dtype=dtype, es_type=es_type
                )
            )
        else:
            return merge_inits(
                weight=ES_Parameter.rand_init(
                    key_w, raw_value=jnp.ones(dim, dtype=dtype),
                    dtype=dtype, es_type=es_type
                )
            )

    @classmethod
    def _forward(cls, common_params: CommonParams, x, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + eps)

        weight = call_submodule(ES_Parameter, 'weight', common_params)
        out = x_norm * weight

        if 'bias' in common_params.params:
            bias = call_submodule(ES_Parameter, 'bias', common_params)
            out = out + bias

        return out
