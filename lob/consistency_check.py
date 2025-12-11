"""
Consistency checker for LOBS5 training pipeline.
Ensures data, dataloader, encoder, and model configs are aligned.
"""
import jax.numpy as jnp
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def check_vocab_consistency(
    vocab_instance,
    model_d_output: int,
    data_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Check consistency between Vocab, Model, and Data.

    Args:
        vocab_instance: Vocab instance from dataloader
        model_d_output: Model's output dimension
        data_path: Optional path to training data for additional checks

    Returns:
        (is_consistent, error_message)
    """
    errors = []

    # Check 1: Vocab size matches model d_output
    vocab_size = len(vocab_instance)
    if vocab_size != model_d_output:
        errors.append(
            f"❌ VOCAB-MODEL MISMATCH!\n"
            f"   Vocab size: {vocab_size}\n"
            f"   Model d_output: {model_d_output}\n"
            f"   → Model cannot output {vocab_size - model_d_output} tokens!\n"
            f"   → Training will fail or produce corrupted weights!"
        )

    # Check 2: Vocab token_mode is explicit
    if not hasattr(vocab_instance, 'token_mode'):
        errors.append(
            f"⚠️  Vocab missing token_mode attribute!\n"
            f"   This means it's using an old version without mode support.\n"
            f"   Recommendation: Update to Vocab(token_mode=22 or 24)"
        )
    else:
        token_mode = vocab_instance.token_mode
        expected_vocab_size = 12012 if token_mode == 22 else 2112

        if vocab_size != expected_vocab_size:
            errors.append(
                f"❌ VOCAB SIZE INCONSISTENT WITH TOKEN_MODE!\n"
                f"   token_mode: {token_mode}\n"
                f"   Vocab size: {vocab_size}\n"
                f"   Expected size: {expected_vocab_size}\n"
                f"   → Vocab may be incorrectly initialized!"
            )

        # Check if model d_output matches token_mode
        if model_d_output != expected_vocab_size:
            errors.append(
                f"❌ MODEL D_OUTPUT INCONSISTENT WITH TOKEN_MODE!\n"
                f"   token_mode: {token_mode}\n"
                f"   Model d_output: {model_d_output}\n"
                f"   Expected d_output: {expected_vocab_size}\n"
                f"   → Serious training bug! Model cannot handle all tokens!"
            )

    # Check 3: Size field encoding
    if hasattr(vocab_instance, 'ENCODING'):
        encoding = vocab_instance.ENCODING

        # Check if using 22tok (single 'size') or 24tok ('size_digit')
        has_size = 'size' in encoding
        has_size_digit = 'size_digit' in encoding

        if has_size and has_size_digit:
            errors.append(
                f"❌ VOCAB HAS BOTH 'size' AND 'size_digit'!\n"
                f"   This should not happen. Vocab is corrupted."
            )
        elif has_size:
            # 22tok mode
            if hasattr(vocab_instance, 'token_mode') and vocab_instance.token_mode != 22:
                errors.append(
                    f"❌ INCONSISTENT SIZE ENCODING!\n"
                    f"   Has 'size' field (22tok style)\n"
                    f"   But token_mode={vocab_instance.token_mode}\n"
                    f"   → Vocab initialization is broken!"
                )
        elif has_size_digit:
            # 24tok mode
            if hasattr(vocab_instance, 'token_mode') and vocab_instance.token_mode != 24:
                errors.append(
                    f"❌ INCONSISTENT SIZE ENCODING!\n"
                    f"   Has 'size_digit' field (24tok style)\n"
                    f"   But token_mode={vocab_instance.token_mode}\n"
                    f"   → Vocab initialization is broken!"
                )
        else:
            errors.append(
                f"❌ VOCAB MISSING SIZE ENCODING!\n"
                f"   Neither 'size' nor 'size_digit' found in encoding.\n"
                f"   → Vocab is broken!"
            )

    # Check 4: Message length
    from lob.encoding import Message_Tokenizer
    msg_len = Message_Tokenizer.MSG_LEN
    expected_msg_len = 22 if (has_size if 'has_size' in locals() else False) else 24

    if msg_len != expected_msg_len:
        errors.append(
            f"⚠️  MSG_LEN might be inconsistent\n"
            f"   MSG_LEN: {msg_len}\n"
            f"   Expected for current vocab: {expected_msg_len}\n"
            f"   → May cause encoding/decoding errors"
        )

    # Format output
    if errors:
        error_msg = "\n" + "="*80 + "\n"
        error_msg += "🚨 CONSISTENCY CHECK FAILED!\n"
        error_msg += "="*80 + "\n"
        error_msg += "\n".join(errors)
        error_msg += "\n" + "="*80 + "\n"
        error_msg += "⚠️  DO NOT PROCEED WITH TRAINING!\n"
        error_msg += "Fix these issues first or training will produce corrupted checkpoints.\n"
        error_msg += "="*80
        return False, error_msg
    else:
        success_msg = "\n" + "="*80 + "\n"
        success_msg += "✅ CONSISTENCY CHECK PASSED!\n"
        success_msg += "="*80 + "\n"
        success_msg += f"Vocab size: {vocab_size}\n"
        success_msg += f"Model d_output: {model_d_output}\n"
        if hasattr(vocab_instance, 'token_mode'):
            success_msg += f"Token mode: {vocab_instance.token_mode}\n"
        success_msg += f"MSG_LEN: {msg_len}\n"
        success_msg += "="*80
        return True, success_msg


def print_consistency_check(vocab, model_d_output, data_path=None):
    """Print consistency check results."""
    is_consistent, message = check_vocab_consistency(vocab, model_d_output, data_path)
    print(message)
    if not is_consistent:
        raise ValueError("Consistency check failed! See errors above.")
    return is_consistent
