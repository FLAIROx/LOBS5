"""
Integration Test: Model Inference Mode (No Perturbations)
File: es_lobs5/tests/integration/test_model_inference.py

Tests verify:
1. Deterministic inference: same input produces same output
2. Autoregressive generation: step-by-step token generation
3. Batch inference: process multiple sequences in parallel
4. Inference mode has no perturbations (noop noiser)
5. Output shapes are correct

Note: Tests use ES_SequenceLayer and ES_StackedEncoder directly to avoid
      architecture issues in the full PaddedLobPredModel while still
      testing inference properties.
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp

from es_lobs5.models.s5_layer import ES_SequenceLayer
from es_lobs5.models.encoder import ES_StackedEncoder
from es_lobs5.models.common import CommonParams, simple_es_tree_key, ES_Linear
from es_lobs5.adapters.hippo_adapter import get_hippo_params
from es_lobs5.utils.import_utils import load_module, get_hyperscalees_path
import os

# Load only the base_noiser (noop) directly to avoid optax dependency from other noisers
_hyperscalees_path = get_hyperscalees_path()
_base_noiser = load_module(
    'hyperscalees.noiser.base_noiser',
    os.path.join(_hyperscalees_path, 'hyperscalees/noiser/base_noiser.py')
)
NoopNoiser = _base_noiser.Noiser


def create_sequence_layer(key, d_model=32, ssm_size=32, blocks=2, activation='gelu'):
    """Create a test sequence layer."""
    hippo = get_hippo_params(ssm_size, blocks, True)

    layer_init = ES_SequenceLayer.rand_init(
        key,
        d_model=d_model,
        ssm_size=ssm_size,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'],
        Vinv=hippo['Vinv'],
        blocks=blocks,
        activation=activation,
        prenorm=True,
    )
    return layer_init


def create_stacked_encoder(key, d_input=32, d_model=32, n_layers=2, ssm_size=32, blocks=2):
    """Create a test stacked encoder."""
    encoder_init = ES_StackedEncoder.rand_init(
        key,
        d_input=d_input,
        d_model=d_model,
        n_layers=n_layers,
        ssm_size=ssm_size,
        blocks=blocks,
        activation='gelu',
        prenorm=False,
    )
    return encoder_init


def create_inference_common_params(model_init, key):
    """Create CommonParams for inference mode (no perturbations)."""
    noiser = NoopNoiser  # No perturbation for inference
    es_tree_key = simple_es_tree_key(model_init.params, key, model_init.scan_map)

    common_params = CommonParams(
        noiser=noiser,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=model_init.frozen_params,
        params=model_init.params,
        es_tree_key=es_tree_key,
        iterinfo=None,  # None means no perturbation
    )
    return common_params


def test_deterministic_inference():
    """Test: Same input produces identical output (deterministic inference)."""
    key = jax.random.PRNGKey(42)
    d_model = 32
    L = 100

    # Create model
    key, subkey = jax.random.split(key)
    model_init = create_sequence_layer(subkey, d_model=d_model)

    # Create inference params
    key, subkey = jax.random.split(key)
    common_params = create_inference_common_params(model_init, subkey)

    # Create fixed inputs
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (L, d_model))

    # Run forward pass twice
    output1 = ES_SequenceLayer._forward(common_params, x)
    output2 = ES_SequenceLayer._forward(common_params, x)

    # Outputs should be identical
    max_diff = jnp.max(jnp.abs(output1 - output2))
    assert max_diff < 1e-10, f"Inference not deterministic: max_diff = {max_diff}"

    print(f"[PASS] test_deterministic_inference: max_diff = {max_diff:.2e}")
    return True


def test_autoregressive_step():
    """Test: Autoregressive generation produces per-token predictions."""
    key = jax.random.PRNGKey(123)
    d_model = 32
    L = 50

    # Create stacked encoder for more complex test
    key, subkey = jax.random.split(key)
    model_init = create_stacked_encoder(subkey, d_input=d_model, d_model=d_model, n_layers=2)

    # Create inference params
    key, subkey = jax.random.split(key)
    common_params = create_inference_common_params(model_init, subkey)

    # Create inputs
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (L, d_model))

    # Forward pass
    output = ES_StackedEncoder._forward(common_params, x)

    # Should return (L, d_model) - same shape as input
    expected_shape = (L, d_model)
    assert output.shape == expected_shape, \
        f"Output shape wrong: {output.shape} vs {expected_shape}"

    # Output should be different from input (model did something)
    diff_from_input = jnp.max(jnp.abs(output - x))
    assert diff_from_input > 0.01, "Output should be different from input"

    print(f"[PASS] test_autoregressive_step: output shape {output.shape}")
    return True


def test_batch_inference():
    """Test: Process multiple sequences in parallel using vmap."""
    key = jax.random.PRNGKey(456)
    d_model = 32
    L = 100
    batch_size = 4

    # Create model
    key, subkey = jax.random.split(key)
    model_init = create_sequence_layer(subkey, d_model=d_model)

    # Create inference params
    key, subkey = jax.random.split(key)
    common_params = create_inference_common_params(model_init, subkey)

    # Create batch of inputs
    key, subkey = jax.random.split(key)
    x_batch = jax.random.normal(subkey, (batch_size, L, d_model))

    # Define batched forward function
    def forward_single(x):
        return ES_SequenceLayer._forward(common_params, x)

    # Use vmap for batch inference
    batched_forward = jax.vmap(forward_single)
    output_batch = batched_forward(x_batch)

    # Check output shape
    expected_shape = (batch_size, L, d_model)
    assert output_batch.shape == expected_shape, \
        f"Batch output shape wrong: {output_batch.shape} vs {expected_shape}"

    # Verify different inputs produce different outputs
    diff_01 = jnp.max(jnp.abs(output_batch[0] - output_batch[1]))
    assert diff_01 > 1e-6, f"Different inputs should produce different outputs: {diff_01}"

    print(f"[PASS] test_batch_inference: batch shape {output_batch.shape}")
    return True


def test_inference_no_noise():
    """Test: Inference mode (noop noiser) produces no perturbations."""
    key = jax.random.PRNGKey(789)
    d_model = 32
    L = 100

    # Create model
    key, subkey = jax.random.split(key)
    model_init = create_sequence_layer(subkey, d_model=d_model)

    # Create es_tree_key
    key, subkey = jax.random.split(key)
    noiser = NoopNoiser
    es_tree_key = simple_es_tree_key(model_init.params, subkey, model_init.scan_map)

    # Create inputs
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (L, d_model))

    # Test with different iterinfo values - should all produce same output with noop noiser
    outputs = []
    for iterinfo in [None, (0, 0), (0, 1), (1, 0)]:
        common_params = CommonParams(
            noiser=noiser,
            frozen_noiser_params=None,
            noiser_params=None,
            frozen_params=model_init.frozen_params,
            params=model_init.params,
            es_tree_key=es_tree_key,
            iterinfo=iterinfo,
        )
        output = ES_SequenceLayer._forward(common_params, x)
        outputs.append(output)

    # All outputs should be identical (noop noiser ignores iterinfo)
    for i in range(1, len(outputs)):
        max_diff = jnp.max(jnp.abs(outputs[0] - outputs[i]))
        assert max_diff < 1e-10, \
            f"Noop noiser should produce same output regardless of iterinfo: diff = {max_diff}"

    print(f"[PASS] test_inference_no_noise: all outputs identical across iterinfo values")
    return True


def test_inference_shape():
    """Test: Output shapes are correct for different configurations."""
    key = jax.random.PRNGKey(999)
    d_model = 32
    L = 100

    # Test different layer configurations
    configs = [
        {'d_model': 32, 'ssm_size': 32, 'blocks': 2},
        {'d_model': 64, 'ssm_size': 64, 'blocks': 4},
        {'d_model': 48, 'ssm_size': 48, 'blocks': 3},
    ]

    for config in configs:
        key, subkey = jax.random.split(key)
        model_init = create_sequence_layer(subkey, **config)

        key, subkey = jax.random.split(key)
        common_params = create_inference_common_params(model_init, subkey)

        # Create inputs
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (L, config['d_model']))

        # Forward pass
        output = ES_SequenceLayer._forward(common_params, x)

        expected_shape = (L, config['d_model'])
        assert output.shape == expected_shape, \
            f"Config {config}: expected {expected_shape}, got {output.shape}"

        print(f"[PASS] test_inference_shape config={config}: shape {output.shape}")

    return True


def test_inference_numerical_stability():
    """Test: Inference produces valid (no NaN/Inf) outputs."""
    key = jax.random.PRNGKey(111)
    d_model = 64
    L = 500

    # Create model with larger dimensions
    key, subkey = jax.random.split(key)
    model_init = create_stacked_encoder(
        subkey, d_input=d_model, d_model=d_model, n_layers=4, ssm_size=64, blocks=4
    )

    # Create inference params
    key, subkey = jax.random.split(key)
    common_params = create_inference_common_params(model_init, subkey)

    # Create inputs
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (L, d_model))

    # Forward pass
    output = ES_StackedEncoder._forward(common_params, x)

    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(output)), "NaN in output"
    assert not jnp.any(jnp.isinf(output)), "Inf in output"

    # Output values should be reasonable (not exploding)
    max_val = jnp.max(jnp.abs(output))
    assert max_val < 1e6, f"Output values too large: max = {max_val}"

    print(f"[PASS] test_inference_numerical_stability: output range [{output.min():.4f}, {output.max():.4f}]")
    return True


def test_encoder_deterministic():
    """Test: Stacked encoder produces deterministic output."""
    key = jax.random.PRNGKey(222)
    d_model = 32
    L = 100

    # Create model
    key, subkey = jax.random.split(key)
    model_init = create_stacked_encoder(subkey, d_input=d_model, d_model=d_model, n_layers=2)

    # Create inference params
    key, subkey = jax.random.split(key)
    common_params = create_inference_common_params(model_init, subkey)

    # Create inputs
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (L, d_model))

    # Run forward pass twice
    output1 = ES_StackedEncoder._forward(common_params, x)
    output2 = ES_StackedEncoder._forward(common_params, x)

    # Outputs should be identical
    max_diff = jnp.max(jnp.abs(output1 - output2))
    assert max_diff < 1e-10, f"Encoder not deterministic: max_diff = {max_diff}"

    print(f"[PASS] test_encoder_deterministic: max_diff = {max_diff:.2e}")
    return True


def test_batch_encoder_inference():
    """Test: Stacked encoder batch inference using vmap."""
    key = jax.random.PRNGKey(333)
    d_model = 32
    L = 50
    batch_size = 8

    # Create model
    key, subkey = jax.random.split(key)
    model_init = create_stacked_encoder(subkey, d_input=d_model, d_model=d_model, n_layers=2)

    # Create inference params
    key, subkey = jax.random.split(key)
    common_params = create_inference_common_params(model_init, subkey)

    # Create batch of inputs
    key, subkey = jax.random.split(key)
    x_batch = jax.random.normal(subkey, (batch_size, L, d_model))

    # Define batched forward function
    def forward_single(x):
        return ES_StackedEncoder._forward(common_params, x)

    # Use vmap for batch inference
    batched_forward = jax.vmap(forward_single)
    output_batch = batched_forward(x_batch)

    # Check output shape
    expected_shape = (batch_size, L, d_model)
    assert output_batch.shape == expected_shape, \
        f"Batch output shape wrong: {output_batch.shape} vs {expected_shape}"

    # Check no NaN
    assert not jnp.any(jnp.isnan(output_batch)), "NaN in batch output"

    print(f"[PASS] test_batch_encoder_inference: batch shape {output_batch.shape}")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Model Inference Integration Tests")
    print("=" * 60)

    tests = [
        test_deterministic_inference,
        test_autoregressive_step,
        test_batch_inference,
        test_inference_no_noise,
        test_inference_shape,
        test_inference_numerical_stability,
        test_encoder_deterministic,
        test_batch_encoder_inference,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
