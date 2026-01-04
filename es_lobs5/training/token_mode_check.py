"""
Token Mode Validation Utilities.

This module provides utilities for validating token_mode consistency between
checkpoints and CLI arguments. Token mode determines the vocabulary size used
for encoding limit order book data:

    token_mode=22: vocab_size = 12012 (base-10000 size encoding)
    token_mode=24: vocab_size = 2112  (base-100 size encoding)

The d_output (decoder output dimension) in a checkpoint directly corresponds
to the vocabulary size, allowing us to infer the token_mode from it.
"""

import os
from typing import Optional, Any, Tuple


# Token mode constants
TOKEN_MODE_22 = 22  # Base-10000 encoding
TOKEN_MODE_24 = 24  # Base-100 encoding

# Vocabulary sizes for each token mode
VOCAB_SIZE_TOKEN_MODE_22 = 12012  # For token_mode=22
VOCAB_SIZE_TOKEN_MODE_24 = 2112   # For token_mode=24

# Threshold for distinguishing token modes based on d_output
D_OUTPUT_THRESHOLD = 10000


def infer_token_mode_from_d_output(d_output: int) -> int:
    """
    Infer token_mode from the decoder output dimension (d_output).

    The vocabulary size directly maps to token_mode:
        - token_mode=22: vocab_size = 12012 (base-10000 size encoding)
        - token_mode=24: vocab_size = 2112  (base-100 size encoding)

    Since vocab_size > 10000 only for token_mode=22, we use this threshold
    to determine the token_mode.

    Args:
        d_output: The decoder output dimension (vocabulary size) from
                  checkpoint config or model parameters.

    Returns:
        int: Inferred token_mode (22 or 24).

    Examples:
        >>> infer_token_mode_from_d_output(12012)
        22
        >>> infer_token_mode_from_d_output(2112)
        24
        >>> infer_token_mode_from_d_output(15000)  # Any value > 10000
        22
        >>> infer_token_mode_from_d_output(5000)   # Any value <= 10000
        24
    """
    if d_output > D_OUTPUT_THRESHOLD:
        return TOKEN_MODE_22
    else:
        return TOKEN_MODE_24


def get_expected_vocab_size(token_mode: int) -> int:
    """
    Get the expected vocabulary size for a given token_mode.

    Args:
        token_mode: The token mode (22 or 24).

    Returns:
        int: Expected vocabulary size for the given token mode.

    Raises:
        ValueError: If token_mode is not 22 or 24.

    Examples:
        >>> get_expected_vocab_size(22)
        12012
        >>> get_expected_vocab_size(24)
        2112
    """
    if token_mode == TOKEN_MODE_22:
        return VOCAB_SIZE_TOKEN_MODE_22
    elif token_mode == TOKEN_MODE_24:
        return VOCAB_SIZE_TOKEN_MODE_24
    else:
        raise ValueError(
            f"Invalid token_mode={token_mode}. "
            f"Supported values: {TOKEN_MODE_22} (base-10000), {TOKEN_MODE_24} (base-100)"
        )


def load_checkpoint_config(checkpoint_path: str) -> dict:
    """
    Load configuration from an Orbax checkpoint.

    This function loads the metadata from a checkpoint directory and extracts
    the model configuration.

    Args:
        checkpoint_path: Path to the Orbax checkpoint directory
                         (e.g., 'checkpoints/lobs5_d3072_xxx/')

    Returns:
        dict: Configuration dictionary from checkpoint metadata.

    Raises:
        ValueError: If no checkpoint is found at the given path.
        ImportError: If orbax.checkpoint is not installed.
    """
    import orbax.checkpoint as ocp

    # Open checkpoint manager
    mgr = ocp.CheckpointManager(
        os.path.abspath(checkpoint_path),
        item_names=('state', 'metadata')
    )

    # Get latest step
    latest = mgr.latest_step()
    if latest is None:
        raise ValueError(f"No checkpoint found in {checkpoint_path}")

    # Restore metadata only
    restored = mgr.restore(
        latest,
        args=ocp.args.Composite(
            metadata=ocp.args.JsonRestore(),
        )
    )

    # Extract config from metadata
    metadata = restored.get('metadata', {})
    config = metadata.get('config', metadata)

    return config


def validate_token_mode(checkpoint_path: str, cli_token_mode: int) -> Tuple[int, int]:
    """
    Validate that a checkpoint's token_mode matches the CLI argument.

    This function loads the checkpoint configuration, infers the token_mode
    from d_output, and validates it against the CLI-specified token_mode.

    Args:
        checkpoint_path: Path to the Orbax checkpoint directory.
        cli_token_mode: Token mode specified via CLI argument.

    Returns:
        Tuple[int, int]: (checkpoint_token_mode, checkpoint_d_output)
                         Returns the inferred values for logging/debugging.

    Raises:
        ValueError: If the checkpoint's token_mode doesn't match the CLI argument.
                    The error message includes detailed information about the
                    mismatch and instructions for resolution.

    Examples:
        >>> # Assume checkpoint has d_output=12012 (token_mode=22)
        >>> validate_token_mode('/path/to/checkpoint', 22)
        (22, 12012)
        >>> validate_token_mode('/path/to/checkpoint', 24)  # Mismatch!
        Traceback (most recent call last):
            ...
        ValueError: Token mode mismatch! ...
    """
    # Load checkpoint config
    config = load_checkpoint_config(checkpoint_path)

    # Get d_output from config
    # First try explicit d_output, then vocab_size, then infer from token_mode
    checkpoint_token_mode = config.get('token_mode')

    if checkpoint_token_mode is not None:
        # Checkpoint has explicit token_mode stored
        default_vocab_size = get_expected_vocab_size(checkpoint_token_mode)
    else:
        # No explicit token_mode, will infer from d_output
        default_vocab_size = VOCAB_SIZE_TOKEN_MODE_22  # Default assumption

    d_output = config.get('d_output', config.get('vocab_size', default_vocab_size))

    # Infer token_mode from d_output
    inferred_token_mode = infer_token_mode_from_d_output(d_output)

    # If checkpoint had explicit token_mode, verify consistency
    if checkpoint_token_mode is not None and checkpoint_token_mode != inferred_token_mode:
        print(f"[WARNING] Checkpoint metadata token_mode={checkpoint_token_mode} "
              f"doesn't match inferred token_mode={inferred_token_mode} from d_output={d_output}")

    # Use inferred token_mode for validation (d_output is ground truth)
    checkpoint_token_mode = inferred_token_mode

    # Validate against CLI argument
    if checkpoint_token_mode != cli_token_mode:
        expected_vocab_size = get_expected_vocab_size(cli_token_mode)
        raise ValueError(
            f"Token mode mismatch!\n"
            f"\n"
            f"  Checkpoint: {checkpoint_path}\n"
            f"    - d_output (vocab_size): {d_output}\n"
            f"    - Inferred token_mode: {checkpoint_token_mode}\n"
            f"\n"
            f"  CLI argument:\n"
            f"    - token_mode: {cli_token_mode}\n"
            f"    - Expected vocab_size: {expected_vocab_size}\n"
            f"\n"
            f"  Resolution:\n"
            f"    Either change CLI argument to --token_mode={checkpoint_token_mode}\n"
            f"    or use a checkpoint trained with token_mode={cli_token_mode}.\n"
            f"\n"
            f"  Token mode reference:\n"
            f"    - token_mode=22: vocab_size=12012 (base-10000 size encoding)\n"
            f"    - token_mode=24: vocab_size=2112  (base-100 size encoding)"
        )

    return checkpoint_token_mode, d_output


def check_token_mode_consistency(
    args: Any,
    checkpoint_path: Optional[str] = None
) -> Optional[Tuple[int, int]]:
    """
    Unified entry point for token_mode consistency checking.

    This function provides a convenient way to validate token_mode consistency.
    It handles the case where no checkpoint is provided (returns None) and
    extracts the checkpoint path from args if not provided explicitly.

    Args:
        args: Namespace or object with CLI arguments. Must have 'token_mode'
              attribute. May optionally have 'checkpoint' or 'checkpoint_path'
              attribute.
        checkpoint_path: Optional explicit checkpoint path. If None, attempts
                         to extract from args.checkpoint or args.checkpoint_path.

    Returns:
        Optional[Tuple[int, int]]: If checkpoint is provided and valid,
                                   returns (checkpoint_token_mode, d_output).
                                   Returns None if no checkpoint is specified.

    Raises:
        ValueError: If token_mode mismatch is detected.
        AttributeError: If args doesn't have 'token_mode' attribute.

    Examples:
        >>> import argparse
        >>> args = argparse.Namespace(token_mode=22, checkpoint='/path/to/ckpt')
        >>> check_token_mode_consistency(args)
        (22, 12012)

        >>> args = argparse.Namespace(token_mode=22, checkpoint=None)
        >>> check_token_mode_consistency(args)
        None
    """
    # Get CLI token_mode
    if not hasattr(args, 'token_mode'):
        raise AttributeError(
            "args must have 'token_mode' attribute. "
            "Ensure your argument parser includes --token_mode."
        )

    cli_token_mode = args.token_mode

    # Determine checkpoint path
    if checkpoint_path is None:
        # Try to extract from args
        checkpoint_path = getattr(args, 'checkpoint', None)
        if checkpoint_path is None:
            checkpoint_path = getattr(args, 'checkpoint_path', None)

    # If no checkpoint, nothing to validate
    if checkpoint_path is None or not checkpoint_path:
        return None

    # Validate token_mode
    return validate_token_mode(checkpoint_path, cli_token_mode)


def print_token_mode_info(token_mode: int, d_output: Optional[int] = None) -> None:
    """
    Print human-readable token mode information.

    Args:
        token_mode: The token mode (22 or 24).
        d_output: Optional actual d_output value from checkpoint.

    Examples:
        >>> print_token_mode_info(22)
        [TOKEN_MODE] token_mode=22 (base-10000 size encoding)
        [TOKEN_MODE] Expected vocab_size: 12012

        >>> print_token_mode_info(24, 2112)
        [TOKEN_MODE] token_mode=24 (base-100 size encoding)
        [TOKEN_MODE] Expected vocab_size: 2112
        [TOKEN_MODE] Actual d_output: 2112
    """
    encoding_desc = "base-10000" if token_mode == TOKEN_MODE_22 else "base-100"
    expected_vocab = get_expected_vocab_size(token_mode)

    print(f"[TOKEN_MODE] token_mode={token_mode} ({encoding_desc} size encoding)")
    print(f"[TOKEN_MODE] Expected vocab_size: {expected_vocab}")

    if d_output is not None:
        print(f"[TOKEN_MODE] Actual d_output: {d_output}")
        if d_output != expected_vocab:
            print(f"[TOKEN_MODE] WARNING: d_output differs from expected vocab_size!")


# =============================================================================
# Self-test / CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Token mode validation utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a checkpoint
  python token_mode_check.py --checkpoint /path/to/ckpt --token_mode 22

  # Just print info about a token mode
  python token_mode_check.py --token_mode 24 --info
        """
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint directory')
    parser.add_argument('--token_mode', type=int, required=True,
                        help='Token mode (22 or 24)')
    parser.add_argument('--info', action='store_true',
                        help='Just print token mode info, no validation')

    args = parser.parse_args()

    if args.info:
        print_token_mode_info(args.token_mode)
    elif args.checkpoint:
        try:
            ckpt_token_mode, d_output = validate_token_mode(
                args.checkpoint, args.token_mode
            )
            print(f"[SUCCESS] Token mode validation passed!")
            print_token_mode_info(ckpt_token_mode, d_output)
        except ValueError as e:
            print(f"[FAILED] {e}")
            exit(1)
    else:
        print("No checkpoint specified. Use --checkpoint or --info.")
        print_token_mode_info(args.token_mode)
