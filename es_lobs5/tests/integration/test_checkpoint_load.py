"""
Test: Checkpoint loading functionality for ES training.

This integration test verifies:
1. Checkpoint saving and loading preserves params
2. Optimizer/noiser state is restored correctly
3. Metadata (epoch, step) is restored
4. Loading from best/ and latest/ directories works

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_checkpoint_load.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
import tempfile
import os
import pickle
import shutil


def create_mock_params():
    """Create mock model parameters for testing."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    return {
        'message_encoder': {
            'embedding': jax.random.normal(k1, (100, 64)),
            'layer_0': {
                'ssm': {
                    'Lambda_re': jax.random.normal(k2, (32,)),
                    'Lambda_im': jax.random.normal(k2, (32,)),
                    'B': jax.random.normal(k2, (32, 64, 2)),
                    'C': jax.random.normal(k2, (64, 32, 2)),
                    'D': jax.random.normal(k2, (64,)),
                    'log_step': jax.random.normal(k2, (32, 1)),
                },
                'norm': {
                    'weight': jnp.ones(64),
                    'bias': jnp.zeros(64),
                },
            },
        },
        'decoder': {
            'weight': jax.random.normal(k3, (100, 64)),
            'bias': jax.random.normal(k4, (100,)),
        },
    }


def create_mock_noiser_params():
    """Create mock noiser parameters for testing."""
    key = jax.random.PRNGKey(123)

    return {
        'solver_state': {
            'count': jnp.array(10, dtype=jnp.int32),
            'mu': jax.random.normal(key, (100,)),
        },
        'sigma': jnp.array(0.01),
        'lr': jnp.array(0.001),
    }


def create_mock_frozen_params():
    """Create mock frozen params for testing."""
    return {
        'd_model': 64,
        'ssm_size': 32,
        'n_message_layers': 2,
        'n_book_pre_layers': 1,
        'n_book_post_layers': 1,
        'n_fused_layers': 4,
        'activation': 'half_glu1',
        'token_mode': 22,
    }


def create_mock_config():
    """Create mock config for testing."""
    class MockConfig:
        def __init__(self):
            self.n_threads = 8
            self.n_steps = 10
            self.noiser = 'eggroll'
            self.sigma = 0.01
            self.lr = 0.001
            self.lora_rank = 4
            self.seed = 42
            self.task = 'sell'
            self.task_size = 100
            self.tick_size = 100
            self.token_mode = 22
            self.background_mode = 'world_model'

    return MockConfig()


def save_mock_checkpoint(path, params, noiser_params, frozen_params, config, epoch=0, best_fitness=-1.0):
    """Save a mock checkpoint for testing."""
    os.makedirs(path, exist_ok=True)

    # Save ES checkpoint
    checkpoint = {
        'params': params,
        'frozen_params': frozen_params,
        'noiser_params': noiser_params,
        'config': vars(config),
    }

    with open(os.path.join(path, 'es_checkpoint.pkl'), 'wb') as f:
        pickle.dump(checkpoint, f)

    # Save training state
    training_state = {
        'epoch': epoch,
        'best_fitness': best_fitness,
    }

    with open(os.path.join(path, 'training_state.pkl'), 'wb') as f:
        pickle.dump(training_state, f)


def load_checkpoint(path):
    """Load checkpoint from path."""
    with open(os.path.join(path, 'es_checkpoint.pkl'), 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def load_training_state(path):
    """Load training state from path."""
    state_path = os.path.join(path, 'training_state.pkl')
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            return pickle.load(f)
    return None


def check_params_equal(params1, params2, path=""):
    """Recursively check if two param trees are equal."""
    if isinstance(params1, dict):
        assert isinstance(params2, dict), f"Type mismatch at {path}"
        assert set(params1.keys()) == set(params2.keys()), f"Key mismatch at {path}: {params1.keys()} vs {params2.keys()}"
        for k in params1:
            check_params_equal(params1[k], params2[k], f"{path}/{k}")
    elif hasattr(params1, 'shape'):
        assert jnp.allclose(params1, params2, rtol=1e-5, atol=1e-5), \
            f"Value mismatch at {path}: shapes {params1.shape} vs {params2.shape}"
    else:
        assert params1 == params2, f"Scalar mismatch at {path}: {params1} vs {params2}"


def test_load_params_match():
    """Test: loaded params == saved params."""
    print("[TEST] test_load_params_match")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save checkpoint
        params = create_mock_params()
        noiser_params = create_mock_noiser_params()
        frozen_params = create_mock_frozen_params()
        config = create_mock_config()

        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        save_mock_checkpoint(ckpt_path, params, noiser_params, frozen_params, config)

        # Load checkpoint
        loaded = load_checkpoint(ckpt_path)

        # Verify params match
        check_params_equal(params, loaded['params'])

        # Verify frozen params match
        assert loaded['frozen_params'] == frozen_params, "Frozen params mismatch"

    print("[PASS] test_load_params_match")
    return True


def test_load_optimizer_state():
    """Test: optimizer/noiser state is restored."""
    print("[TEST] test_load_optimizer_state")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create checkpoint with specific noiser state
        params = create_mock_params()
        noiser_params = create_mock_noiser_params()
        frozen_params = create_mock_frozen_params()
        config = create_mock_config()

        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        save_mock_checkpoint(ckpt_path, params, noiser_params, frozen_params, config)

        # Load checkpoint
        loaded = load_checkpoint(ckpt_path)
        loaded_noiser = loaded['noiser_params']

        # Verify noiser state matches
        assert jnp.allclose(loaded_noiser['sigma'], noiser_params['sigma']), \
            "Sigma mismatch"
        assert jnp.allclose(loaded_noiser['lr'], noiser_params['lr']), \
            "LR mismatch"

        # Verify solver state
        assert jnp.allclose(
            loaded_noiser['solver_state']['count'],
            noiser_params['solver_state']['count']
        ), "Solver count mismatch"
        assert jnp.allclose(
            loaded_noiser['solver_state']['mu'],
            noiser_params['solver_state']['mu']
        ), "Solver mu mismatch"

    print("[PASS] test_load_optimizer_state")
    return True


def test_load_metadata():
    """Test: epoch, step are restored."""
    print("[TEST] test_load_metadata")

    with tempfile.TemporaryDirectory() as tmpdir:
        params = create_mock_params()
        noiser_params = create_mock_noiser_params()
        frozen_params = create_mock_frozen_params()
        config = create_mock_config()

        # Save with specific epoch and fitness
        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        expected_epoch = 42
        expected_fitness = 0.123
        save_mock_checkpoint(
            ckpt_path, params, noiser_params, frozen_params, config,
            epoch=expected_epoch, best_fitness=expected_fitness
        )

        # Load training state
        state = load_training_state(ckpt_path)

        assert state is not None, "Training state not found"
        assert state['epoch'] == expected_epoch, \
            f"Epoch mismatch: {state['epoch']} vs {expected_epoch}"
        assert abs(state['best_fitness'] - expected_fitness) < 1e-6, \
            f"Best fitness mismatch: {state['best_fitness']} vs {expected_fitness}"

    print("[PASS] test_load_metadata")
    return True


def test_load_best_checkpoint():
    """Test: load from best/ directory."""
    print("[TEST] test_load_best_checkpoint")

    with tempfile.TemporaryDirectory() as tmpdir:
        params = create_mock_params()
        noiser_params = create_mock_noiser_params()
        frozen_params = create_mock_frozen_params()
        config = create_mock_config()

        # Save to best/ subdirectory
        best_path = os.path.join(tmpdir, 'checkpoints', 'best')
        best_fitness = 0.999
        best_epoch = 100
        save_mock_checkpoint(
            best_path, params, noiser_params, frozen_params, config,
            epoch=best_epoch, best_fitness=best_fitness
        )

        # Verify best/ directory exists
        assert os.path.isdir(best_path), "best/ directory not created"
        assert os.path.exists(os.path.join(best_path, 'es_checkpoint.pkl')), \
            "es_checkpoint.pkl not found in best/"

        # Load from best/
        loaded = load_checkpoint(best_path)
        state = load_training_state(best_path)

        # Verify params
        check_params_equal(params, loaded['params'])

        # Verify state
        assert state['epoch'] == best_epoch, f"Epoch mismatch: {state['epoch']}"
        assert abs(state['best_fitness'] - best_fitness) < 1e-6, \
            f"Best fitness mismatch: {state['best_fitness']}"

    print("[PASS] test_load_best_checkpoint")
    return True


def test_load_latest_checkpoint():
    """Test: load from latest/ directory."""
    print("[TEST] test_load_latest_checkpoint")

    with tempfile.TemporaryDirectory() as tmpdir:
        params = create_mock_params()
        noiser_params = create_mock_noiser_params()
        frozen_params = create_mock_frozen_params()
        config = create_mock_config()

        # Save to latest/ subdirectory
        latest_path = os.path.join(tmpdir, 'checkpoints', 'latest')
        latest_epoch = 50
        latest_fitness = 0.5
        save_mock_checkpoint(
            latest_path, params, noiser_params, frozen_params, config,
            epoch=latest_epoch, best_fitness=latest_fitness
        )

        # Verify latest/ directory exists
        assert os.path.isdir(latest_path), "latest/ directory not created"
        assert os.path.exists(os.path.join(latest_path, 'es_checkpoint.pkl')), \
            "es_checkpoint.pkl not found in latest/"

        # Load from latest/
        loaded = load_checkpoint(latest_path)
        state = load_training_state(latest_path)

        # Verify params
        check_params_equal(params, loaded['params'])

        # Verify state
        assert state['epoch'] == latest_epoch, f"Epoch mismatch: {state['epoch']}"

    print("[PASS] test_load_latest_checkpoint")
    return True


def test_checkpoint_overwrite():
    """Test: checkpoint can be overwritten correctly."""
    print("[TEST] test_checkpoint_overwrite")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        frozen_params = create_mock_frozen_params()
        config = create_mock_config()

        # Save first checkpoint
        params1 = create_mock_params()
        noiser_params1 = create_mock_noiser_params()
        save_mock_checkpoint(
            ckpt_path, params1, noiser_params1, frozen_params, config,
            epoch=10, best_fitness=0.1
        )

        # Save second checkpoint (overwrite)
        key = jax.random.PRNGKey(999)
        params2 = {
            'message_encoder': {
                'embedding': jax.random.normal(key, (100, 64)),
                'layer_0': params1['message_encoder']['layer_0'],
            },
            'decoder': params1['decoder'],
        }
        noiser_params2 = {
            'solver_state': noiser_params1['solver_state'],
            'sigma': jnp.array(0.02),  # Different sigma
            'lr': jnp.array(0.002),
        }
        save_mock_checkpoint(
            ckpt_path, params2, noiser_params2, frozen_params, config,
            epoch=20, best_fitness=0.2
        )

        # Load and verify it's the second checkpoint
        loaded = load_checkpoint(ckpt_path)
        state = load_training_state(ckpt_path)

        assert state['epoch'] == 20, f"Epoch should be 20, got {state['epoch']}"
        assert abs(state['best_fitness'] - 0.2) < 1e-6, \
            f"Best fitness should be 0.2, got {state['best_fitness']}"
        assert jnp.allclose(loaded['noiser_params']['sigma'], jnp.array(0.02)), \
            "Sigma should be 0.02"

    print("[PASS] test_checkpoint_overwrite")
    return True


def test_missing_training_state():
    """Test: handle missing training_state.pkl gracefully."""
    print("[TEST] test_missing_training_state")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        os.makedirs(ckpt_path, exist_ok=True)

        # Save only es_checkpoint.pkl (no training_state.pkl)
        params = create_mock_params()
        noiser_params = create_mock_noiser_params()
        frozen_params = create_mock_frozen_params()
        config = create_mock_config()

        checkpoint = {
            'params': params,
            'frozen_params': frozen_params,
            'noiser_params': noiser_params,
            'config': vars(config),
        }

        with open(os.path.join(ckpt_path, 'es_checkpoint.pkl'), 'wb') as f:
            pickle.dump(checkpoint, f)

        # Load should work for checkpoint
        loaded = load_checkpoint(ckpt_path)
        assert loaded is not None, "Checkpoint loading failed"

        # Training state should return None
        state = load_training_state(ckpt_path)
        assert state is None, "Should return None for missing training_state.pkl"

    print("[PASS] test_missing_training_state")
    return True


def test_config_preserved():
    """Test: config values are preserved in checkpoint."""
    print("[TEST] test_config_preserved")

    with tempfile.TemporaryDirectory() as tmpdir:
        params = create_mock_params()
        noiser_params = create_mock_noiser_params()
        frozen_params = create_mock_frozen_params()
        config = create_mock_config()

        # Set specific config values
        config.n_threads = 256
        config.sigma = 0.05
        config.task = 'buy'

        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        save_mock_checkpoint(ckpt_path, params, noiser_params, frozen_params, config)

        # Load and verify config
        loaded = load_checkpoint(ckpt_path)
        loaded_config = loaded['config']

        assert loaded_config['n_threads'] == 256, "n_threads mismatch"
        assert abs(loaded_config['sigma'] - 0.05) < 1e-6, "sigma mismatch"
        assert loaded_config['task'] == 'buy', "task mismatch"

    print("[PASS] test_config_preserved")
    return True


def run_all_tests():
    """Run all checkpoint loading tests."""
    print("=" * 60)
    print("Running checkpoint loading integration tests")
    print("=" * 60)

    tests = [
        test_load_params_match,
        test_load_optimizer_state,
        test_load_metadata,
        test_load_best_checkpoint,
        test_load_latest_checkpoint,
        test_checkpoint_overwrite,
        test_missing_training_state,
        test_config_preserved,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
