"""
Training Mode Configuration System for LOBS5 Evolution Strategy Training.

This module provides a declarative configuration matrix that maps MODE names
to parameter classification behaviors. It replaces the scattered boolean flags
(USE_LORA, FREEZE_NONLORA, LORA_V2, FREEZE_SSM) with a single source of truth.

Mode Overview:
    LORA       - LoRA on out2 only, freeze everything else
    LORA_V1.5  - LoRA on all projections, freeze SSM, train norms
    LORA_V2    - LoRA on all projections + train SSM/norms
    FULL       - Full parameter training (no LoRA)

Research References:
    [1] Hu et al. "LoRA" ICLR 2022: https://arxiv.org/abs/2106.09685
    [2] Galim et al. "PEFT of SSMs" ICML 2025: https://arxiv.org/abs/2410.09016
"""

from enum import IntEnum
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass


class ESMapType(IntEnum):
    """ES Map Classification Constants (must match HyperscaleES/common.py)."""
    PARAM = 0       # Full noise (if freeze_nonlora=False)
    MM_PARAM = 1    # LoRA noise (low-rank adaptation)
    EMB_PARAM = 2   # Embedding (not implemented)
    EXCLUDED = 3    # No noise, frozen parameter


class ParamBehavior(IntEnum):
    """How parameters should be treated during training."""
    FULL = 0        # Full gradient updates
    LORA = 1        # Low-rank adaptation
    FROZEN = 2      # No updates (excluded from training)


@dataclass
class ModeConfig:
    """Configuration for a training mode."""
    name: str
    description: str

    # Parameter classification rules
    projections: ParamBehavior     # out2, input_proj, proj
    ssm_params: ParamBehavior      # B, C, D, log_step
    norm_bias: ParamBehavior       # LayerNorm weights/biases

    # Internal flags (derived from classification rules)
    freeze_nonlora: bool           # Passed to noiser.init_noiser

    # Which projection patterns get LoRA treatment
    lora_patterns: tuple           # Patterns to convert to MM_PARAM


# Pattern definitions
LORA_V1_PATTERNS = ('out2/weight',)  # Default LoRA: only out2
LORA_V2_PATTERNS = ('out2/weight', 'input_proj/weight', 'proj/weight')  # Expanded LoRA
SSM_PATTERNS = ('ssm/B', 'ssm/C', 'ssm/D', 'ssm/log_step')  # SSM parameters


# =============================================================================
# TRAINING MODES CONFIGURATION MATRIX
# =============================================================================
# This is the single source of truth for all mode behaviors.
# Adding a new mode is as simple as adding a new entry here.

TRAINING_MODES: Dict[str, ModeConfig] = {
    "LORA": ModeConfig(
        name="LORA",
        description="LoRA on out2 only, freeze all other parameters",
        projections=ParamBehavior.LORA,      # Only out2 gets LoRA
        ssm_params=ParamBehavior.FROZEN,     # B, C, D, log_step frozen
        norm_bias=ParamBehavior.FROZEN,      # LayerNorm frozen
        freeze_nonlora=True,                 # Freeze non-LoRA params
        lora_patterns=LORA_V1_PATTERNS,
    ),

    "LORA_V1.5": ModeConfig(
        name="LORA_V1.5",
        description="LoRA on all projections, freeze SSM, train norms [Recommended]",
        projections=ParamBehavior.LORA,      # All projections get LoRA
        ssm_params=ParamBehavior.FROZEN,     # B, C, D, log_step frozen [2]
        norm_bias=ParamBehavior.FULL,        # LayerNorm trainable [1]
        freeze_nonlora=False,                # Allow non-LoRA param updates
        lora_patterns=LORA_V2_PATTERNS,
    ),

    "LORA_V2": ModeConfig(
        name="LORA_V2",
        description="LoRA on all projections + train SSM/norms",
        projections=ParamBehavior.LORA,      # All projections get LoRA
        ssm_params=ParamBehavior.FULL,       # B, C, D, log_step trainable
        norm_bias=ParamBehavior.FULL,        # LayerNorm trainable
        freeze_nonlora=False,                # Allow non-LoRA param updates
        lora_patterns=LORA_V2_PATTERNS,
    ),

    "FULL": ModeConfig(
        name="FULL",
        description="Full parameter training (no LoRA)",
        projections=ParamBehavior.FULL,      # Full updates
        ssm_params=ParamBehavior.FULL,       # Full updates
        norm_bias=ParamBehavior.FULL,        # Full updates
        freeze_nonlora=False,                # Not applicable
        lora_patterns=(),                    # No LoRA
    ),
}

# Legacy mode aliases for backwards compatibility
TRAINING_MODES["LORA+SSM"] = ModeConfig(
    name="LORA+SSM",
    description="LoRA on out2 + train SSM params (legacy)",
    projections=ParamBehavior.LORA,
    ssm_params=ParamBehavior.FULL,
    norm_bias=ParamBehavior.FULL,
    freeze_nonlora=False,
    lora_patterns=LORA_V1_PATTERNS,
)


def get_mode_config(mode: str) -> ModeConfig:
    """
    Get the configuration for a training mode.

    Args:
        mode: Mode name (LORA, LORA_V1.5, LORA_V2, FULL)

    Returns:
        ModeConfig with all classification rules

    Raises:
        ValueError: If mode is not recognized
    """
    if mode not in TRAINING_MODES:
        valid_modes = list(TRAINING_MODES.keys())
        raise ValueError(f"Unknown training mode: '{mode}'. Valid modes: {valid_modes}")
    return TRAINING_MODES[mode]


def is_lora_mode(mode: str) -> bool:
    """Check if a mode uses LoRA adaptation."""
    config = get_mode_config(mode)
    return config.projections == ParamBehavior.LORA


def get_param_classifier(mode: str) -> Callable[[str, Any], int]:
    """
    Get a parameter classifier function for the given mode.

    The classifier takes a parameter path and value, and returns
    the appropriate ESMapType value.

    Args:
        mode: Training mode name

    Returns:
        Classifier function: (path: str, param: jnp.ndarray) -> ESMapType
    """
    config = get_mode_config(mode)

    def matches_pattern(path: str, patterns: tuple) -> bool:
        """Check if path ends with any of the patterns."""
        return any(path.endswith(p) for p in patterns)

    def classifier(path: str, param: Any) -> int:
        """
        Classify a parameter based on its path and the mode configuration.

        Args:
            path: Parameter path (e.g., 'layers/0/ssm/B')
            param: The parameter array

        Returns:
            ESMapType value (PARAM=0, MM_PARAM=1, EXCLUDED=3)
        """
        # FULL mode: everything is PARAM (full training)
        if mode == "FULL":
            return ESMapType.PARAM

        # Check if this is a LoRA target projection
        if matches_pattern(path, config.lora_patterns):
            # Only 2D weights can use LoRA
            if hasattr(param, 'ndim') and param.ndim == 2:
                return ESMapType.MM_PARAM
            return ESMapType.PARAM

        # Check if this is an SSM parameter
        if matches_pattern(path, SSM_PATTERNS):
            if config.ssm_params == ParamBehavior.FROZEN:
                return ESMapType.EXCLUDED
            return ESMapType.PARAM

        # Default behavior for other params
        if config.freeze_nonlora:
            return ESMapType.EXCLUDED
        return ESMapType.PARAM

    return classifier


def apply_es_map_classification(es_map: dict, params: dict,
                                 classifier: Callable, path: str = "") -> dict:
    """
    Apply a classifier to remap es_map values based on mode configuration.

    Args:
        es_map: Current es_map tree
        params: Parameters tree (for shape checking)
        classifier: Function from get_param_classifier()
        path: Current path prefix (used in recursion)

    Returns:
        New es_map with updated classifications
    """
    if isinstance(es_map, dict):
        return {
            k: apply_es_map_classification(
                es_map[k], params[k], classifier,
                f"{path}/{k}" if path else k
            )
            for k in es_map
        }
    else:
        # Leaf node - apply classifier
        return classifier(path, params)


def print_mode_config(mode: str) -> None:
    """Print a nicely formatted mode configuration table."""
    config = get_mode_config(mode)

    # Behavior to string mapping
    behavior_str = {
        ParamBehavior.FULL: "FULL",
        ParamBehavior.LORA: "LoRA",
        ParamBehavior.FROZEN: "Frozen",
    }

    print(f"\n[MODE] ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print(f"[MODE] ║                                   TRAINING MODE: {config.name:^12}                                        ║")
    print(f"[MODE] ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"[MODE] ║  {config.description:<102} ║")
    print(f"[MODE] ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    print(f"[MODE]")
    print(f"[MODE] ┌───────────────────────────┬─────────────────┬───────────────────────────────────────────────────────────┐")
    print(f"[MODE] │       Parameter Type      │    Treatment    │                       Description                         │")
    print(f"[MODE] ├───────────────────────────┼─────────────────┼───────────────────────────────────────────────────────────┤")
    print(f"[MODE] │ Projections               │ {behavior_str[config.projections]:^15} │ out2, input_proj, proj weight matrices                    │")
    print(f"[MODE] │ SSM Parameters            │ {behavior_str[config.ssm_params]:^15} │ B, C, D, log_step (discretization params)                 │")
    print(f"[MODE] │ Norm/Bias                 │ {behavior_str[config.norm_bias]:^15} │ LayerNorm weights/biases                                  │")
    print(f"[MODE] └───────────────────────────┴─────────────────┴───────────────────────────────────────────────────────────┘")
    print(f"[MODE]")

    if config.lora_patterns:
        print(f"[MODE] LoRA patterns: {', '.join(config.lora_patterns)}")
    else:
        print(f"[MODE] LoRA patterns: None (full parameter training)")

    print(f"[MODE] freeze_nonlora: {config.freeze_nonlora}")
    print(f"[MODE]")


def get_all_modes() -> list:
    """Return list of all available mode names."""
    return list(TRAINING_MODES.keys())


# Convenience exports
__all__ = [
    'ESMapType',
    'ParamBehavior',
    'ModeConfig',
    'TRAINING_MODES',
    'SSM_PATTERNS',
    'LORA_V1_PATTERNS',
    'LORA_V2_PATTERNS',
    'get_mode_config',
    'is_lora_mode',
    'get_param_classifier',
    'apply_es_map_classification',
    'print_mode_config',
    'get_all_modes',
]
