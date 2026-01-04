"""
Test: Token mode validation and error handling
File: es_lobs5/tests/test_token_mode.py
"""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
from es_lobs5.models.lob_model import ES_PaddedLobPredModel
from es_lobs5.training.token_mode_check import infer_token_mode_from_d_output


def test_token_mode_22():
    """Test: 22-token mode uses vocab size 12012."""
    key = jax.random.PRNGKey(42)

    model_init = ES_PaddedLobPredModel.rand_init(
        key,
        d_output=12012,  # 22-token vocab
        d_model=32,
        d_book=103,
        n_message_layers=1,
        n_fused_layers=1,
        ssm_size=32,
        blocks=2,
    )

    # Verify embedding size matches vocab
    embedding_shape = model_init.params['message_encoder']['embedding'].shape
    assert embedding_shape[0] == 12012, f"Wrong embedding vocab: {embedding_shape[0]}"

    # Verify decoder output dim
    decoder_weight_shape = model_init.params['decoder']['weight'].shape
    assert decoder_weight_shape[0] == 12012, f"Wrong decoder output: {decoder_weight_shape[0]}"

    print("[PASS] 22-token mode: vocab size 12012")
    return True


def test_token_mode_24():
    """Test: 24-token mode uses vocab size 2112."""
    key = jax.random.PRNGKey(42)

    model_init = ES_PaddedLobPredModel.rand_init(
        key,
        d_output=2112,  # 24-token vocab
        d_model=32,
        d_book=103,
        n_message_layers=1,
        n_fused_layers=1,
        ssm_size=32,
        blocks=2,
    )

    embedding_shape = model_init.params['message_encoder']['embedding'].shape
    assert embedding_shape[0] == 2112, f"Wrong embedding vocab: {embedding_shape[0]}"

    decoder_weight_shape = model_init.params['decoder']['weight'].shape
    assert decoder_weight_shape[0] == 2112, f"Wrong decoder output: {decoder_weight_shape[0]}"

    print("[PASS] 24-token mode: vocab size 2112")
    return True


def test_infer_token_mode():
    """Test: Token mode inference from d_output."""
    # 22-token mode: vocab > 10000
    assert infer_token_mode_from_d_output(12012) == 22
    assert infer_token_mode_from_d_output(11000) == 22

    # 24-token mode: vocab <= 10000
    assert infer_token_mode_from_d_output(2112) == 24
    assert infer_token_mode_from_d_output(5000) == 24

    print("[PASS] Token mode inference works correctly")
    return True


def test_vocab_size_consistency():
    """Test: All model components use consistent vocab size."""
    key = jax.random.PRNGKey(456)

    for d_output in [12012, 2112]:
        model_init = ES_PaddedLobPredModel.rand_init(
            key,
            d_output=d_output,
            d_model=32,
            d_book=103,
            n_message_layers=1,
            n_fused_layers=1,
            ssm_size=32,
            blocks=2,
        )

        # Check embedding
        emb_vocab = model_init.params['message_encoder']['embedding'].shape[0]

        # Check decoder
        dec_output = model_init.params['decoder']['weight'].shape[0]

        # Check frozen_params
        fp_d_output = model_init.frozen_params['d_output']

        assert emb_vocab == d_output, f"Embedding vocab mismatch: {emb_vocab} vs {d_output}"
        assert dec_output == d_output, f"Decoder output mismatch: {dec_output} vs {d_output}"
        assert fp_d_output == d_output, f"Frozen d_output mismatch: {fp_d_output} vs {d_output}"

        print(f"[PASS] d_output={d_output}: all components consistent")

    return True


if __name__ == "__main__":
    test_token_mode_22()
    test_token_mode_24()
    test_infer_token_mode()
    test_vocab_size_consistency()
    print("All token mode tests passed!")
