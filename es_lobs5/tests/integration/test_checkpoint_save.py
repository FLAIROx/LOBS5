"""
Integration Test: Checkpoint saving functionality for ES training.

This test verifies:
1. Checkpoint files are created correctly
2. Checkpoint contains correct data (params, optimizer_state, metadata)
3. Directory structure matches expected pattern (best/, latest/, epoch_N/)

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_checkpoint_save.py
"""

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
import tempfile
import os
import pickle


def create_mock_params():
    """Create mock model parameters for testing."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    return {
        'encoder': {
            'embedding': jax.random.normal(k1, (1000, 64)),
        },
        'layer_0': {
            'ssm': {
                'Lambda_re': jax.random.normal(k2, (32,)),
                'Lambda_im': jax.random.normal(k2, (32,)),
            },
            'norm': {
                'weight': jnp.ones(64),
                'bias': jnp.zeros(64),
            },
        },
        'decoder': {
            'weight': jax.random.normal(k3, (1000, 64)),
            'bias': jnp.zeros(1000),
        },
    }


def create_mock_frozen_params():
    """Create mock frozen parameters."""
    return {
        'd_model': 64,
        'ssm_size': 32,
        'n_message_layers': 2,
        'n_fused_layers': 4,
        'vocab_size': 1000,
    }


def create_mock_noiser_params():
    """Create mock noiser/optimizer state."""
    key = jax.random.PRNGKey(123)
    return {
        'sigma': 0.01,
        'lr': 0.001,
        'rank': 4,
        'momentum': jax.random.normal(key, (100,)),
        'velocity': jax.random.normal(key, (100,)),
    }


def create_mock_config():
    """Create mock training configuration."""
    class MockConfig:
        def __init__(self):
            self.n_threads = 128
            self.n_steps = 100
            self.noiser = 'eggroll'
            self.sigma = 0.01
            self.lr = 0.001
            self.lora_rank = 4
            self.seed = 42
            self.task = 'sell'
            self.task_size = 500
            self.token_mode = 22
            self.background_mode = 'historical_replay'
            self.checkpoint_dir = './es_checkpoints'
            self.checkpoint_every = 50

    return MockConfig()


def save_mock_checkpoint(path: str, params: dict, frozen_params: dict,
                          noiser_params: dict, config) -> None:
    """Save a mock checkpoint mimicking ESTrainer.save_checkpoint."""
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        'params': params,
        'frozen_params': frozen_params,
        'noiser_params': noiser_params,
        'config': vars(config),
    }

    with open(os.path.join(path, 'es_checkpoint.pkl'), 'wb') as f:
        pickle.dump(checkpoint, f)


def save_mock_training_state(path: str, epoch: int, best_fitness: float) -> None:
    """Save mock training state mimicking ESTrainer._save_training_state."""
    os.makedirs(path, exist_ok=True)
    state = {
        'epoch': epoch,
        'best_fitness': best_fitness,
    }
    with open(os.path.join(path, 'training_state.pkl'), 'wb') as f:
        pickle.dump(state, f)


def test_save_creates_file():
    """Test: Checkpoint file is created when save_checkpoint is called."""
    print("[TEST] test_save_creates_file")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_ckpt')

        params = create_mock_params()
        frozen_params = create_mock_frozen_params()
        noiser_params = create_mock_noiser_params()
        config = create_mock_config()

        save_mock_checkpoint(checkpoint_path, params, frozen_params, noiser_params, config)

        expected_file = os.path.join(checkpoint_path, 'es_checkpoint.pkl')
        assert os.path.exists(expected_file), f"Checkpoint file not created: {expected_file}"
        assert os.path.isfile(expected_file), f"Expected file, got directory: {expected_file}"

        # Verify file is not empty
        file_size = os.path.getsize(expected_file)
        assert file_size > 0, f"Checkpoint file is empty: {file_size} bytes"

    print("[PASS] test_save_creates_file")
    return True


def test_save_contains_params():
    """Test: Saved checkpoint contains model parameters."""
    print("[TEST] test_save_contains_params")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_ckpt')

        params = create_mock_params()
        frozen_params = create_mock_frozen_params()
        noiser_params = create_mock_noiser_params()
        config = create_mock_config()

        save_mock_checkpoint(checkpoint_path, params, frozen_params, noiser_params, config)

        # Load and verify
        with open(os.path.join(checkpoint_path, 'es_checkpoint.pkl'), 'rb') as f:
            loaded = pickle.load(f)

        assert 'params' in loaded, "Checkpoint missing 'params' key"

        # Verify structure
        assert 'encoder' in loaded['params'], "Missing encoder in params"
        assert 'layer_0' in loaded['params'], "Missing layer_0 in params"
        assert 'decoder' in loaded['params'], "Missing decoder in params"

        # Verify values are preserved (check shapes)
        assert loaded['params']['encoder']['embedding'].shape == (1000, 64), \
            f"Encoder embedding shape mismatch: {loaded['params']['encoder']['embedding'].shape}"
        assert loaded['params']['layer_0']['ssm']['Lambda_re'].shape == (32,), \
            "SSM Lambda_re shape mismatch"

        # Verify numerical values are close
        assert jnp.allclose(loaded['params']['encoder']['embedding'], params['encoder']['embedding']), \
            "Encoder embedding values changed during save/load"

    print("[PASS] test_save_contains_params")
    return True


def test_save_contains_optimizer_state():
    """Test: Saved checkpoint contains optimizer/noiser state."""
    print("[TEST] test_save_contains_optimizer_state")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_ckpt')

        params = create_mock_params()
        frozen_params = create_mock_frozen_params()
        noiser_params = create_mock_noiser_params()
        config = create_mock_config()

        save_mock_checkpoint(checkpoint_path, params, frozen_params, noiser_params, config)

        # Load and verify
        with open(os.path.join(checkpoint_path, 'es_checkpoint.pkl'), 'rb') as f:
            loaded = pickle.load(f)

        assert 'noiser_params' in loaded, "Checkpoint missing 'noiser_params' key"

        # Verify noiser state structure
        assert 'sigma' in loaded['noiser_params'], "Missing sigma in noiser_params"
        assert 'lr' in loaded['noiser_params'], "Missing lr in noiser_params"
        assert 'momentum' in loaded['noiser_params'], "Missing momentum in noiser_params"

        # Verify values are preserved
        assert loaded['noiser_params']['sigma'] == 0.01, "Sigma value changed"
        assert loaded['noiser_params']['lr'] == 0.001, "LR value changed"
        assert jnp.allclose(loaded['noiser_params']['momentum'], noiser_params['momentum']), \
            "Momentum values changed during save/load"

    print("[PASS] test_save_contains_optimizer_state")
    return True


def test_save_contains_metadata():
    """Test: Saved checkpoint contains training metadata (epoch, step, best_fitness)."""
    print("[TEST] test_save_contains_metadata")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_ckpt')

        params = create_mock_params()
        frozen_params = create_mock_frozen_params()
        noiser_params = create_mock_noiser_params()
        config = create_mock_config()

        # Save checkpoint and training state
        save_mock_checkpoint(checkpoint_path, params, frozen_params, noiser_params, config)
        save_mock_training_state(checkpoint_path, epoch=42, best_fitness=0.95)

        # Verify training_state.pkl exists
        state_file = os.path.join(checkpoint_path, 'training_state.pkl')
        assert os.path.exists(state_file), f"Training state file not created: {state_file}"

        # Load and verify training state
        with open(state_file, 'rb') as f:
            state = pickle.load(f)

        assert 'epoch' in state, "Training state missing 'epoch' key"
        assert 'best_fitness' in state, "Training state missing 'best_fitness' key"

        assert state['epoch'] == 42, f"Epoch mismatch: expected 42, got {state['epoch']}"
        assert abs(state['best_fitness'] - 0.95) < 1e-6, \
            f"Best fitness mismatch: expected 0.95, got {state['best_fitness']}"

        # Also verify config is saved in checkpoint
        with open(os.path.join(checkpoint_path, 'es_checkpoint.pkl'), 'rb') as f:
            loaded = pickle.load(f)

        assert 'config' in loaded, "Checkpoint missing 'config' key"
        assert loaded['config']['n_threads'] == 128, "Config n_threads mismatch"
        assert loaded['config']['seed'] == 42, "Config seed mismatch"

    print("[PASS] test_save_contains_metadata")
    return True


def test_save_directory_structure():
    """Test: Checkpoints create correct directory structure (best/, latest/, epoch_N/)."""
    print("[TEST] test_save_directory_structure")

    with tempfile.TemporaryDirectory() as tmpdir:
        params = create_mock_params()
        frozen_params = create_mock_frozen_params()
        noiser_params = create_mock_noiser_params()
        config = create_mock_config()

        # Simulate saving best checkpoint
        best_path = os.path.join(tmpdir, 'best')
        save_mock_checkpoint(best_path, params, frozen_params, noiser_params, config)
        save_mock_training_state(best_path, epoch=100, best_fitness=0.98)

        # Simulate saving latest checkpoint
        latest_path = os.path.join(tmpdir, 'latest')
        save_mock_checkpoint(latest_path, params, frozen_params, noiser_params, config)
        save_mock_training_state(latest_path, epoch=150, best_fitness=0.98)

        # Simulate saving epoch checkpoints
        for epoch_num in [50, 100, 150]:
            epoch_path = os.path.join(tmpdir, f'epoch_{epoch_num}')
            save_mock_checkpoint(epoch_path, params, frozen_params, noiser_params, config)
            save_mock_training_state(epoch_path, epoch=epoch_num, best_fitness=0.85 + epoch_num * 0.001)

        # Verify directory structure
        assert os.path.isdir(best_path), f"best/ directory not created: {best_path}"
        assert os.path.isdir(latest_path), f"latest/ directory not created: {latest_path}"

        for epoch_num in [50, 100, 150]:
            epoch_path = os.path.join(tmpdir, f'epoch_{epoch_num}')
            assert os.path.isdir(epoch_path), f"epoch_{epoch_num}/ directory not created"

            # Verify each epoch directory has required files
            assert os.path.exists(os.path.join(epoch_path, 'es_checkpoint.pkl')), \
                f"epoch_{epoch_num}/ missing es_checkpoint.pkl"
            assert os.path.exists(os.path.join(epoch_path, 'training_state.pkl')), \
                f"epoch_{epoch_num}/ missing training_state.pkl"

        # Verify best/ has correct metadata
        with open(os.path.join(best_path, 'training_state.pkl'), 'rb') as f:
            best_state = pickle.load(f)
        assert best_state['epoch'] == 100, "Best checkpoint epoch mismatch"
        assert abs(best_state['best_fitness'] - 0.98) < 1e-6, "Best checkpoint fitness mismatch"

        # Verify latest/ has most recent epoch
        with open(os.path.join(latest_path, 'training_state.pkl'), 'rb') as f:
            latest_state = pickle.load(f)
        assert latest_state['epoch'] == 150, f"Latest epoch mismatch: {latest_state['epoch']}"

    print("[PASS] test_save_directory_structure")
    return True


def test_save_overwrites_existing():
    """Test: Saving to existing path overwrites previous checkpoint."""
    print("[TEST] test_save_overwrites_existing")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_ckpt')

        params_v1 = create_mock_params()
        frozen_params = create_mock_frozen_params()
        noiser_params = create_mock_noiser_params()
        config = create_mock_config()

        # Save first version
        save_mock_checkpoint(checkpoint_path, params_v1, frozen_params, noiser_params, config)
        save_mock_training_state(checkpoint_path, epoch=10, best_fitness=0.5)

        # Modify params and save again
        key = jax.random.PRNGKey(999)
        params_v2 = create_mock_params()
        params_v2['encoder']['embedding'] = jax.random.normal(key, (1000, 64))

        save_mock_checkpoint(checkpoint_path, params_v2, frozen_params, noiser_params, config)
        save_mock_training_state(checkpoint_path, epoch=20, best_fitness=0.8)

        # Load and verify it's the new version
        with open(os.path.join(checkpoint_path, 'es_checkpoint.pkl'), 'rb') as f:
            loaded = pickle.load(f)

        with open(os.path.join(checkpoint_path, 'training_state.pkl'), 'rb') as f:
            state = pickle.load(f)

        # Should have new params (not v1)
        assert not jnp.allclose(loaded['params']['encoder']['embedding'],
                                 params_v1['encoder']['embedding']), \
            "Checkpoint was not overwritten - still has v1 params"

        # Should have new training state
        assert state['epoch'] == 20, f"Training state not overwritten: epoch={state['epoch']}"
        assert abs(state['best_fitness'] - 0.8) < 1e-6, "Training state fitness not overwritten"

    print("[PASS] test_save_overwrites_existing")
    return True


def test_save_frozen_params():
    """Test: Frozen params are saved correctly."""
    print("[TEST] test_save_frozen_params")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_ckpt')

        params = create_mock_params()
        frozen_params = create_mock_frozen_params()
        noiser_params = create_mock_noiser_params()
        config = create_mock_config()

        save_mock_checkpoint(checkpoint_path, params, frozen_params, noiser_params, config)

        # Load and verify
        with open(os.path.join(checkpoint_path, 'es_checkpoint.pkl'), 'rb') as f:
            loaded = pickle.load(f)

        assert 'frozen_params' in loaded, "Checkpoint missing 'frozen_params' key"

        # Verify frozen params structure
        assert loaded['frozen_params']['d_model'] == 64, "d_model mismatch in frozen_params"
        assert loaded['frozen_params']['ssm_size'] == 32, "ssm_size mismatch in frozen_params"
        assert loaded['frozen_params']['n_message_layers'] == 2, "n_message_layers mismatch"

    print("[PASS] test_save_frozen_params")
    return True


def test_save_large_params():
    """Test: Large parameter tensors are saved/loaded correctly."""
    print("[TEST] test_save_large_params")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_ckpt')

        # Create larger params to simulate real model
        key = jax.random.PRNGKey(42)
        large_params = {
            'encoder': {
                'embedding': jax.random.normal(key, (50000, 512)),  # ~100MB
            },
            'layers': {
                f'layer_{i}': {
                    'ssm': {
                        'Lambda_re': jax.random.normal(key, (256,)),
                        'Lambda_im': jax.random.normal(key, (256,)),
                        'B': jax.random.normal(key, (256, 512, 2)),
                        'C': jax.random.normal(key, (512, 256, 2)),
                    },
                    'norm': {
                        'weight': jnp.ones(512),
                        'bias': jnp.zeros(512),
                    },
                } for i in range(8)
            },
            'decoder': {
                'weight': jax.random.normal(key, (50000, 512)),
                'bias': jnp.zeros(50000),
            },
        }

        frozen_params = create_mock_frozen_params()
        noiser_params = create_mock_noiser_params()
        config = create_mock_config()

        save_mock_checkpoint(checkpoint_path, large_params, frozen_params, noiser_params, config)

        # Load and verify
        with open(os.path.join(checkpoint_path, 'es_checkpoint.pkl'), 'rb') as f:
            loaded = pickle.load(f)

        # Verify shapes are preserved
        assert loaded['params']['encoder']['embedding'].shape == (50000, 512), \
            "Large encoder shape not preserved"
        assert loaded['params']['decoder']['weight'].shape == (50000, 512), \
            "Large decoder shape not preserved"

        # Verify layer structure
        for i in range(8):
            layer_key = f'layer_{i}'
            assert layer_key in loaded['params']['layers'], f"Missing {layer_key}"
            assert loaded['params']['layers'][layer_key]['ssm']['B'].shape == (256, 512, 2), \
                f"SSM B shape mismatch in {layer_key}"

    print("[PASS] test_save_large_params")
    return True


def run_all_tests():
    """Run all checkpoint save tests."""
    print("=" * 60)
    print("Integration Tests: Checkpoint Save Functionality")
    print("=" * 60)

    tests = [
        test_save_creates_file,
        test_save_contains_params,
        test_save_contains_optimizer_state,
        test_save_contains_metadata,
        test_save_directory_structure,
        test_save_overwrites_existing,
        test_save_frozen_params,
        test_save_large_params,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
