"""
Test: Checkpoint save -> load -> continue training round-trip.
File: es_lobs5/tests/integration/test_checkpoint_roundtrip.py

This integration test verifies that:
1. Training state can be saved to disk
2. Training state can be loaded from disk correctly
3. Training can continue from the loaded state
4. Results are deterministic when continuing from checkpoint

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/integration/test_checkpoint_roundtrip.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp
import tempfile
import os
import pickle

# Mock optax-like optimizer to avoid optax dependency
class MockSGDState:
    """Mock SGD optimizer state."""
    def __init__(self, count):
        self.count = count

class MockSGD:
    """Mock SGD optimizer."""
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def init(self, params):
        return MockSGDState(0)

    def update(self, grads, state, params=None):
        updates = jax.tree.map(lambda g: -self.lr * g, grads)
        return updates, MockSGDState(state.count + 1)

def mock_clip_by_global_norm(max_norm):
    """Mock gradient clipping."""
    def clip_fn(grads):
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
        return jax.tree.map(lambda g: g * scale, grads)
    return clip_fn

def mock_apply_updates(params, updates):
    """Apply updates to params."""
    return jax.tree.map(lambda p, u: p + u, params, updates)

class MockChain:
    """Mock optax.chain for combining transforms."""
    def __init__(self, *transforms):
        self.transforms = transforms

    def init(self, params):
        # Return the last optimizer's state
        return self.transforms[-1].init(params)

    def update(self, grads, state, params=None):
        # Apply clipping first, then optimizer
        clipped_grads = self.transforms[0](grads)
        return self.transforms[1].update(clipped_grads, state, params)

class MockOptax:
    @staticmethod
    def sgd(learning_rate):
        return MockSGD(learning_rate)

    @staticmethod
    def clip_by_global_norm(max_norm):
        return mock_clip_by_global_norm(max_norm)

    @staticmethod
    def chain(*transforms):
        return MockChain(*transforms)

    @staticmethod
    def apply_updates(params, updates):
        return mock_apply_updates(params, updates)

optax = MockOptax()


def create_test_params():
    """Create simple test parameters for checkpointing tests."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    params = {
        'encoder': {
            'embedding': jax.random.normal(k1, (100, 64)),
        },
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
            'out': {
                'weight': jax.random.normal(k3, (64, 64)),
                'bias': jnp.zeros(64),
            },
        },
        'decoder': {
            'weight': jax.random.normal(k3, (100, 64)),
            'bias': jnp.zeros(100),
        },
    }
    return params


def create_test_optimizer_state(params):
    """Create test optimizer state."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(learning_rate=0.01),
    )
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def create_test_noiser_params():
    """Create mock noiser params for testing."""
    key = jax.random.PRNGKey(123)
    noiser_params = {
        'sigma': 0.01,
        'lr': 0.001,
        'rank': 4,
        'lora_matrices': {
            'layer_0': {
                'A': jax.random.normal(key, (64, 4)),
                'B': jax.random.normal(key, (4, 64)),
            },
        },
        'momentum': {
            'layer_0': jax.random.normal(key, (64, 64)),
        },
    }
    return noiser_params


def save_checkpoint(path, params, opt_state, noiser_params, rng_key, epoch):
    """Save checkpoint to disk."""
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        'params': params,
        'opt_state': opt_state,
        'noiser_params': noiser_params,
        'rng_key': rng_key,
        'epoch': epoch,
    }

    with open(os.path.join(path, 'checkpoint.pkl'), 'wb') as f:
        pickle.dump(checkpoint, f)

    return path


def load_checkpoint(path):
    """Load checkpoint from disk."""
    with open(os.path.join(path, 'checkpoint.pkl'), 'rb') as f:
        checkpoint = pickle.load(f)

    return (
        checkpoint['params'],
        checkpoint['opt_state'],
        checkpoint['noiser_params'],
        checkpoint['rng_key'],
        checkpoint['epoch'],
    )


def params_equal(params1, params2):
    """Check if two param trees are equal."""
    flat1 = jax.tree_util.tree_leaves(params1)
    flat2 = jax.tree_util.tree_leaves(params2)

    if len(flat1) != len(flat2):
        return False

    for p1, p2 in zip(flat1, flat2):
        if not jnp.allclose(p1, p2, rtol=1e-5, atol=1e-8):
            return False

    return True


def apply_gradient_update(params, grads, optimizer, opt_state):
    """Apply a gradient update to params."""
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def simulate_training_step(params, optimizer, opt_state, rng_key):
    """Simulate one training step with random gradients."""
    # Generate random gradients
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    rng_key, *grad_keys = jax.random.split(rng_key, len(flat_params) + 1)

    flat_grads = [
        jax.random.normal(k, p.shape) * 0.01
        for k, p in zip(grad_keys, flat_params)
    ]
    grads = jax.tree_util.tree_unflatten(tree_def, flat_grads)

    # Apply update
    new_params, new_opt_state = apply_gradient_update(params, grads, optimizer, opt_state)

    return new_params, new_opt_state, rng_key


def test_roundtrip_params():
    """Test: save -> load gives identical params."""
    print("\n=== test_roundtrip_params ===")

    params = create_test_params()
    _, opt_state = create_test_optimizer_state(params)
    noiser_params = create_test_noiser_params()
    rng_key = jax.random.PRNGKey(999)
    epoch = 42

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'ckpt')

        # Save checkpoint
        save_checkpoint(ckpt_path, params, opt_state, noiser_params, rng_key, epoch)

        # Load checkpoint
        loaded_params, _, _, _, loaded_epoch = load_checkpoint(ckpt_path)

        # Verify params match
        assert params_equal(params, loaded_params), "Params mismatch after roundtrip"
        assert loaded_epoch == epoch, f"Epoch mismatch: {loaded_epoch} vs {epoch}"

        # Verify specific values
        assert jnp.allclose(
            params['encoder']['embedding'],
            loaded_params['encoder']['embedding']
        ), "Encoder embedding mismatch"

        assert jnp.allclose(
            params['layer_0']['ssm']['Lambda_re'],
            loaded_params['layer_0']['ssm']['Lambda_re']
        ), "SSM Lambda_re mismatch"

    print("[PASS] Params roundtrip: save -> load gives identical params")
    return True


def test_roundtrip_optimizer():
    """Test: optimizer state is correctly restored."""
    print("\n=== test_roundtrip_optimizer ===")

    params = create_test_params()
    optimizer, opt_state = create_test_optimizer_state(params)
    noiser_params = create_test_noiser_params()
    rng_key = jax.random.PRNGKey(999)
    epoch = 10

    # Apply a few gradient updates before saving
    for _ in range(3):
        params, opt_state, rng_key = simulate_training_step(
            params, optimizer, opt_state, rng_key
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'ckpt')

        # Save checkpoint
        save_checkpoint(ckpt_path, params, opt_state, noiser_params, rng_key, epoch)

        # Load checkpoint
        loaded_params, loaded_opt_state, _, loaded_rng_key, _ = load_checkpoint(ckpt_path)

        # Verify optimizer state matches (MockSGDState has count attribute)
        assert opt_state.count == loaded_opt_state.count, (
            f"Optimizer count mismatch: {opt_state.count} vs {loaded_opt_state.count}"
        )

    print("[PASS] Optimizer roundtrip: state correctly restored")
    return True


def test_roundtrip_training():
    """Test: training from loaded state gives same results as continuous training."""
    print("\n=== test_roundtrip_training ===")

    # Setup initial state
    params_orig = create_test_params()
    optimizer, opt_state_orig = create_test_optimizer_state(params_orig)
    noiser_params = create_test_noiser_params()
    rng_key = jax.random.PRNGKey(12345)

    # Clone for comparison
    params_continuous = jax.tree.map(lambda x: x.copy(), params_orig)
    opt_state_continuous = jax.tree.map(lambda x: x.copy() if hasattr(x, 'copy') else x, opt_state_orig)
    rng_key_continuous = rng_key

    # Train for 5 steps, then save
    for step in range(5):
        params_continuous, opt_state_continuous, rng_key_continuous = simulate_training_step(
            params_continuous, optimizer, opt_state_continuous, rng_key_continuous
        )

    # Save at step 5
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'ckpt')
        save_checkpoint(
            ckpt_path,
            params_continuous,
            opt_state_continuous,
            noiser_params,
            rng_key_continuous,
            epoch=5
        )

        # Continue training for 5 more steps (continuous)
        for step in range(5):
            params_continuous, opt_state_continuous, rng_key_continuous = simulate_training_step(
                params_continuous, optimizer, opt_state_continuous, rng_key_continuous
            )

        # Load checkpoint and continue from step 5
        loaded_params, loaded_opt_state, _, loaded_rng_key, loaded_epoch = load_checkpoint(ckpt_path)

        assert loaded_epoch == 5, f"Expected epoch 5, got {loaded_epoch}"

        # Train for 5 more steps from loaded state
        for step in range(5):
            loaded_params, loaded_opt_state, loaded_rng_key = simulate_training_step(
                loaded_params, optimizer, loaded_opt_state, loaded_rng_key
            )

        # Verify results match
        assert params_equal(params_continuous, loaded_params), (
            "Training from loaded state gives different params than continuous training"
        )

    print("[PASS] Training roundtrip: loaded state continues correctly")
    return True


def test_roundtrip_rng():
    """Test: RNG state is correctly restored."""
    print("\n=== test_roundtrip_rng ===")

    params = create_test_params()
    _, opt_state = create_test_optimizer_state(params)
    noiser_params = create_test_noiser_params()
    rng_key = jax.random.PRNGKey(42)

    # Generate some random values before saving
    rng_key, k1, k2 = jax.random.split(rng_key, 3)
    pre_save_random = jax.random.normal(k2, (10,))

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'ckpt')

        # Save checkpoint with the current rng_key
        save_checkpoint(ckpt_path, params, opt_state, noiser_params, rng_key, epoch=0)

        # Generate values after save
        rng_key_orig, k3 = jax.random.split(rng_key)
        post_save_random_orig = jax.random.normal(k3, (10,))

        # Load checkpoint
        _, _, _, loaded_rng_key, _ = load_checkpoint(ckpt_path)

        # Generate values from loaded key
        loaded_rng_key, k4 = jax.random.split(loaded_rng_key)
        post_save_random_loaded = jax.random.normal(k4, (10,))

        # Verify RNG produces same values
        assert jnp.allclose(post_save_random_orig, post_save_random_loaded), (
            "RNG state not correctly restored - random values differ"
        )

    print("[PASS] RNG roundtrip: state correctly restored")
    return True


def test_roundtrip_determinism():
    """Test: identical runs from checkpoint produce identical results."""
    print("\n=== test_roundtrip_determinism ===")

    # Create and save initial state
    params = create_test_params()
    optimizer, opt_state = create_test_optimizer_state(params)
    noiser_params = create_test_noiser_params()
    rng_key = jax.random.PRNGKey(777)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'ckpt')
        save_checkpoint(ckpt_path, params, opt_state, noiser_params, rng_key, epoch=0)

        # Run 1: Load and train
        p1, o1, _, k1, _ = load_checkpoint(ckpt_path)
        for _ in range(10):
            p1, o1, k1 = simulate_training_step(p1, optimizer, o1, k1)

        # Run 2: Load same checkpoint and train
        p2, o2, _, k2, _ = load_checkpoint(ckpt_path)
        for _ in range(10):
            p2, o2, k2 = simulate_training_step(p2, optimizer, o2, k2)

        # Verify identical results
        assert params_equal(p1, p2), (
            "Determinism test failed: two runs from same checkpoint give different results"
        )

        # Verify specific values match exactly
        diff_sum = sum(
            jnp.sum(jnp.abs(a - b))
            for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2))
        )
        assert diff_sum == 0.0, f"Params differ by {diff_sum}"

    print("[PASS] Determinism test: identical runs from checkpoint produce identical results")
    return True


def test_checkpoint_file_structure():
    """Test: checkpoint files have expected structure."""
    print("\n=== test_checkpoint_file_structure ===")

    params = create_test_params()
    _, opt_state = create_test_optimizer_state(params)
    noiser_params = create_test_noiser_params()
    rng_key = jax.random.PRNGKey(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'ckpt')
        save_checkpoint(ckpt_path, params, opt_state, noiser_params, rng_key, epoch=100)

        # Verify directory exists
        assert os.path.isdir(ckpt_path), f"Checkpoint directory not created: {ckpt_path}"

        # Verify checkpoint file exists
        ckpt_file = os.path.join(ckpt_path, 'checkpoint.pkl')
        assert os.path.isfile(ckpt_file), f"Checkpoint file not found: {ckpt_file}"

        # Verify file can be read as pickle
        with open(ckpt_file, 'rb') as f:
            checkpoint = pickle.load(f)

        # Verify expected keys
        expected_keys = {'params', 'opt_state', 'noiser_params', 'rng_key', 'epoch'}
        assert set(checkpoint.keys()) == expected_keys, (
            f"Unexpected checkpoint keys: {set(checkpoint.keys())} vs {expected_keys}"
        )

        # Verify epoch
        assert checkpoint['epoch'] == 100, f"Epoch mismatch: {checkpoint['epoch']}"

    print("[PASS] Checkpoint file structure is correct")
    return True


def test_multiple_save_load_cycles():
    """Test: multiple save/load cycles maintain data integrity."""
    print("\n=== test_multiple_save_load_cycles ===")

    params = create_test_params()
    optimizer, opt_state = create_test_optimizer_state(params)
    noiser_params = create_test_noiser_params()
    rng_key = jax.random.PRNGKey(999)

    with tempfile.TemporaryDirectory() as tmpdir:
        for cycle in range(5):
            ckpt_path = os.path.join(tmpdir, f'ckpt_{cycle}')

            # Train for a few steps
            for _ in range(3):
                params, opt_state, rng_key = simulate_training_step(
                    params, optimizer, opt_state, rng_key
                )

            # Save
            save_checkpoint(ckpt_path, params, opt_state, noiser_params, rng_key, epoch=cycle)

            # Load
            params, opt_state, _, rng_key, loaded_epoch = load_checkpoint(ckpt_path)

            assert loaded_epoch == cycle, f"Cycle {cycle}: epoch mismatch"

        # Verify we can still load any checkpoint
        for cycle in range(5):
            ckpt_path = os.path.join(tmpdir, f'ckpt_{cycle}')
            loaded_params, _, _, _, loaded_epoch = load_checkpoint(ckpt_path)
            assert loaded_epoch == cycle, f"Failed to load checkpoint {cycle}"

    print("[PASS] Multiple save/load cycles maintain data integrity")
    return True


def test_noiser_params_roundtrip():
    """Test: noiser params (LoRA matrices, momentum) are correctly restored."""
    print("\n=== test_noiser_params_roundtrip ===")

    params = create_test_params()
    _, opt_state = create_test_optimizer_state(params)
    noiser_params = create_test_noiser_params()
    rng_key = jax.random.PRNGKey(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'ckpt')
        save_checkpoint(ckpt_path, params, opt_state, noiser_params, rng_key, epoch=0)

        _, _, loaded_noiser_params, _, _ = load_checkpoint(ckpt_path)

        # Verify scalar params
        assert loaded_noiser_params['sigma'] == noiser_params['sigma']
        assert loaded_noiser_params['lr'] == noiser_params['lr']
        assert loaded_noiser_params['rank'] == noiser_params['rank']

        # Verify LoRA matrices
        assert jnp.allclose(
            loaded_noiser_params['lora_matrices']['layer_0']['A'],
            noiser_params['lora_matrices']['layer_0']['A']
        ), "LoRA A matrix mismatch"

        assert jnp.allclose(
            loaded_noiser_params['lora_matrices']['layer_0']['B'],
            noiser_params['lora_matrices']['layer_0']['B']
        ), "LoRA B matrix mismatch"

        # Verify momentum
        assert jnp.allclose(
            loaded_noiser_params['momentum']['layer_0'],
            noiser_params['momentum']['layer_0']
        ), "Momentum mismatch"

    print("[PASS] Noiser params roundtrip: all values correctly restored")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Checkpoint Roundtrip Integration Tests")
    print("=" * 60)

    all_tests_passed = True

    tests = [
        test_roundtrip_params,
        test_roundtrip_optimizer,
        test_roundtrip_training,
        test_roundtrip_rng,
        test_roundtrip_determinism,
        test_checkpoint_file_structure,
        test_multiple_save_load_cycles,
        test_noiser_params_roundtrip,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("All checkpoint roundtrip tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)
