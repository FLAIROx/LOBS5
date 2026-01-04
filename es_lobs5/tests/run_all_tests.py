"""
Master test runner for all ES/Eggroll modules
File: es_lobs5/tests/run_all_tests.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')


def run_all_tests():
    print("=" * 60)
    print("Running ES/Eggroll Module Tests")
    print("=" * 60)

    results = []

    # Test 1: SSM Adapter
    print(f"\n{'='*60}")
    print("Testing: SSM Adapter")
    print("="*60)
    try:
        from es_lobs5.tests import test_s5_ssm
        test_s5_ssm.test_ssm_forward_consistency()
        test_s5_ssm.test_ssm_with_noise()
        results.append(("SSM Adapter", "PASS"))
    except Exception as e:
        print(f"[FAIL] SSM Adapter: {e}")
        results.append(("SSM Adapter", f"FAIL: {e}"))

    # Test 2: Checkpoint Converter
    print(f"\n{'='*60}")
    print("Testing: Checkpoint Converter")
    print("="*60)
    try:
        from es_lobs5.tests import test_checkpoint_converter
        test_checkpoint_converter.test_dense_transpose()
        test_checkpoint_converter.test_layernorm_conversion()
        test_checkpoint_converter.test_ssm_conversion_unchanged()
        test_checkpoint_converter.test_full_model_conversion()
        results.append(("Checkpoint Converter", "PASS"))
    except Exception as e:
        print(f"[FAIL] Checkpoint Converter: {e}")
        results.append(("Checkpoint Converter", f"FAIL: {e}"))

    # Test 3: Sequence Layer
    print(f"\n{'='*60}")
    print("Testing: Sequence Layer")
    print("="*60)
    try:
        from es_lobs5.tests import test_sequence_layer
        test_sequence_layer.test_layer_forward_shape()
        test_sequence_layer.test_residual_connection()
        test_sequence_layer.test_glu_gates()
        results.append(("Sequence Layer", "PASS"))
    except Exception as e:
        print(f"[FAIL] Sequence Layer: {e}")
        results.append(("Sequence Layer", f"FAIL: {e}"))

    # Test 4: LOB Model
    print(f"\n{'='*60}")
    print("Testing: LOB Model")
    print("="*60)
    try:
        from es_lobs5.tests import test_lob_model
        test_lob_model.test_model_forward_pass()
        test_lob_model.test_model_modes()
        test_lob_model.test_autoregressive_forward()
        results.append(("LOB Model", "PASS"))
    except Exception as e:
        print(f"[FAIL] LOB Model: {e}")
        results.append(("LOB Model", f"FAIL: {e}"))

    # Test 5: Token Mode
    print(f"\n{'='*60}")
    print("Testing: Token Mode")
    print("="*60)
    try:
        from es_lobs5.tests import test_token_mode
        test_token_mode.test_token_mode_22()
        test_token_mode.test_token_mode_24()
        test_token_mode.test_infer_token_mode()
        test_token_mode.test_vocab_size_consistency()
        results.append(("Token Mode", "PASS"))
    except Exception as e:
        print(f"[FAIL] Token Mode: {e}")
        results.append(("Token Mode", f"FAIL: {e}"))

    # Test 6: Eggroll Noiser
    print(f"\n{'='*60}")
    print("Testing: Eggroll Noiser")
    print("="*60)
    try:
        from es_lobs5.tests import test_eggroll_noiser
        test_eggroll_noiser.test_fixed_eggroll_key_shape()
        test_eggroll_noiser.test_noiser_perturbation()
        test_eggroll_noiser.test_excluded_params_not_perturbed()
        test_eggroll_noiser.test_different_iterinfo_different_noise()
        test_eggroll_noiser.test_antithetic_sampling()
        results.append(("Eggroll Noiser", "PASS"))
    except Exception as e:
        print(f"[FAIL] Eggroll Noiser: {e}")
        results.append(("Eggroll Noiser", f"FAIL: {e}"))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")

    all_passed = all(s == "PASS" for _, s in results)
    passed_count = sum(1 for _, s in results if s == "PASS")
    total_count = len(results)

    print(f"\n{'='*60}")
    print(f"Results: {passed_count}/{total_count} tests passed")
    print(f"{'All tests PASSED!' if all_passed else 'Some tests FAILED!'}")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
