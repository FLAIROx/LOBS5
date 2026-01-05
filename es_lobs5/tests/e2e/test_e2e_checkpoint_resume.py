"""
Level 3 E2E Test: Checkpoint Save/Load/Resume Verification.

This test verifies that checkpoint save/load/resume works correctly for ES training:
1. Initialize model and train for 5 epochs
2. Save checkpoint to temporary directory
3. Load checkpoint into new model instance
4. Continue training for 5 more epochs
5. Verify:
   - Loaded params match saved params exactly
   - Training continues correctly from checkpoint
   - Epoch counter resumes correctly
   - Final results match continuous training

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/e2e/test_e2e_checkpoint_resume.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import os
import pickle


# =============================================================================
# Mock Components for Minimal Testing
# =============================================================================

class MockOptax:
    """Mock optax for testing without dependency."""

    @staticmethod
    def sgd(learning_rate):
        return MockSGD(learning_rate)

    @staticmethod
    def clip_by_global_norm(max_norm):
        def clip_fn(grads):
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
            return jax.tree.map(lambda g: g * scale, grads)
        return clip_fn

    @staticmethod
    def chain(*transforms):
        return MockChain(*transforms)

    @staticmethod
    def apply_updates(params, updates):
        return jax.tree.map(lambda p, u: p + u, params, updates)


class MockSGDState:
    def __init__(self, count):
        self.count = count


class MockSGD:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def init(self, params):
        return MockSGDState(0)

    def update(self, grads, state, params=None):
        updates = jax.tree.map(lambda g: -self.lr * g, grads)
        return updates, MockSGDState(state.count + 1)


class MockChain:
    def __init__(self, *transforms):
        self.transforms = transforms

    def init(self, params):
        return self.transforms[-1].init(params)

    def update(self, grads, state, params=None):
        clipped_grads = self.transforms[0](grads)
        return self.transforms[1].update(clipped_grads, state, params)


class MockConfig:
    """Minimal config for testing."""
    def __init__(self, seed=42):
        self.seed = seed
        self.n_threads = 4
        self.n_epochs = 5
        self.lr = 0.001
        self.sigma = 0.01
        self.lora_rank = 4
        self.checkpoint_dir = None
        self.checkpoint_every = 5


class MinimalESTrainer:
    """
    Minimal ES trainer for testing checkpoint functionality.

    Uses simple mock params and gradient updates without requiring
    actual model, JaxLOB, or data dependencies.
    """

    def __init__(self, config, key=None):
        self.config = config
        self.key = key if key is not None else jax.random.PRNGKey(config.seed)

        # Initialize params
        self.params = self._create_test_params()
        self.noiser_params = self._create_noiser_params()
        self.frozen_params = {
            'ssm_size': 32,
            'd_model': 64,
            'n_layers': 2,
        }

        # Optimizer
        optax = MockOptax()
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate=config.lr),
        )
        self.opt_state = self.optimizer.init(self.params)

        # Epoch counter
        self.current_epoch = 0
        self.training_history = []

    def _create_test_params(self):
        """Create minimal test parameters."""
        key = self.key
        k1, k2, k3, k4 = jax.random.split(key, 4)

        return {
            'encoder': {
                'embedding': jax.random.normal(k1, (100, 64)),
            },
            'ssm': {
                'Lambda_re': jax.random.normal(k2, (32,)),
                'Lambda_im': jax.random.normal(k2, (32,)),
                'B': jax.random.normal(k2, (32, 64, 2)),
                'C': jax.random.normal(k2, (64, 32, 2)),
            },
            'layer_0': {
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
                'weight': jax.random.normal(k4, (100, 64)),
                'bias': jnp.zeros(100),
            },
        }

    def _create_noiser_params(self):
        """Create mock noiser params."""
        key = jax.random.PRNGKey(123)
        return {
            'sigma': self.config.sigma,
            'lr': self.config.lr,
            'rank': self.config.lora_rank,
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

    def _compute_mock_fitness(self, params, key):
        """Compute mock fitness based on params."""
        # Simple fitness: sum of squared params
        flat_params = jax.tree.leaves(params)
        fitness = sum(jnp.sum(p**2) for p in flat_params)
        return -fitness / 1e6  # Negative because we want to minimize

    def _compute_mock_gradients(self, params, key):
        """Compute mock gradients for testing."""
        flat_params, tree_def = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(key, len(flat_params))

        flat_grads = [
            jax.random.normal(k, p.shape) * 0.01
            for k, p in zip(keys, flat_params)
        ]
        return jax.tree_util.tree_unflatten(tree_def, flat_grads)

    def train_epoch(self, epoch_key):
        """Run one training epoch."""
        # Compute mock fitness for each thread
        thread_keys = jax.random.split(epoch_key, self.config.n_threads)
        fitnesses = [self._compute_mock_fitness(self.params, k) for k in thread_keys]
        mean_fitness = np.mean(fitnesses)

        # Compute and apply gradients
        grad_key = jax.random.fold_in(epoch_key, 0)
        grads = self._compute_mock_gradients(self.params, grad_key)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = MockOptax.apply_updates(self.params, updates)

        # Update epoch counter
        self.current_epoch += 1
        self.training_history.append({
            'epoch': self.current_epoch,
            'fitness': float(mean_fitness),
        })

        return mean_fitness

    def train(self, n_epochs=None, start_epoch=0, use_saved_key=False):
        """Run training for n_epochs.

        Args:
            n_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resume)
            use_saved_key: If True, use self.key directly (for resume from checkpoint)
        """
        n_epochs = n_epochs or self.config.n_epochs

        if use_saved_key:
            # Use the key as-is from checkpoint (it was saved after training)
            key = self.key
        else:
            # Fresh training - use seed-derived key
            key = jax.random.PRNGKey(self.config.seed)
            # Fast-forward key to start_epoch
            for _ in range(start_epoch):
                key, _ = jax.random.split(key)

        self.current_epoch = start_epoch

        for epoch in range(n_epochs):
            key, epoch_key = jax.random.split(key)
            fitness = self.train_epoch(epoch_key)

        self.key = key
        return self.params

    def save_checkpoint(self, path):
        """Save checkpoint to disk."""
        os.makedirs(path, exist_ok=True)

        checkpoint = {
            'params': self.params,
            'frozen_params': self.frozen_params,
            'noiser_params': self.noiser_params,
            'opt_state': self.opt_state,
            'key': self.key,
            'current_epoch': self.current_epoch,
            'training_history': self.training_history,
            'config': vars(self.config),
        }

        with open(os.path.join(path, 'es_checkpoint.pkl'), 'wb') as f:
            pickle.dump(checkpoint, f)

        # Also save training state separately (for ESTrainer compatibility)
        training_state = {
            'epoch': self.current_epoch,
            'best_fitness': self.training_history[-1]['fitness'] if self.training_history else 0.0,
        }
        with open(os.path.join(path, 'training_state.pkl'), 'wb') as f:
            pickle.dump(training_state, f)

    def load_checkpoint(self, path):
        """Load checkpoint from disk."""
        with open(os.path.join(path, 'es_checkpoint.pkl'), 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint['params']
        self.frozen_params = checkpoint['frozen_params']
        self.noiser_params = checkpoint['noiser_params']
        self.opt_state = checkpoint['opt_state']
        self.key = checkpoint['key']
        self.current_epoch = checkpoint['current_epoch']
        self.training_history = checkpoint['training_history']

        return checkpoint


# =============================================================================
# Utility Functions
# =============================================================================

def params_equal(params1, params2, rtol=1e-5, atol=1e-8):
    """Check if two param trees are equal within tolerance."""
    flat1 = jax.tree_util.tree_leaves(params1)
    flat2 = jax.tree_util.tree_leaves(params2)

    if len(flat1) != len(flat2):
        return False

    for p1, p2 in zip(flat1, flat2):
        if not jnp.allclose(p1, p2, rtol=rtol, atol=atol):
            return False

    return True


def params_exactly_equal(params1, params2):
    """Check if two param trees are exactly equal (bitwise)."""
    flat1 = jax.tree_util.tree_leaves(params1)
    flat2 = jax.tree_util.tree_leaves(params2)

    if len(flat1) != len(flat2):
        return False

    for p1, p2 in zip(flat1, flat2):
        if not jnp.array_equal(p1, p2):
            return False

    return True


# =============================================================================
# Test Cases
# =============================================================================

def test_save_load_params():
    """Test: Params are identical after save/load."""
    print("\n=== test_save_load_params ===")

    config = MockConfig(seed=42)
    trainer = MinimalESTrainer(config)

    # Train for 5 epochs
    trainer.train(n_epochs=5)
    params_before_save = jax.tree.map(lambda x: x.copy(), trainer.params)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Save checkpoint
        trainer.save_checkpoint(ckpt_path)

        # Create new trainer and load
        new_trainer = MinimalESTrainer(config)
        new_trainer.load_checkpoint(ckpt_path)

        # Verify params match exactly
        assert params_exactly_equal(params_before_save, new_trainer.params), \
            "Params mismatch after save/load"

        # Verify specific values
        assert jnp.array_equal(
            params_before_save['encoder']['embedding'],
            new_trainer.params['encoder']['embedding']
        ), "Encoder embedding mismatch"

        assert jnp.array_equal(
            params_before_save['ssm']['Lambda_re'],
            new_trainer.params['ssm']['Lambda_re']
        ), "SSM Lambda_re mismatch"

        assert jnp.array_equal(
            params_before_save['decoder']['weight'],
            new_trainer.params['decoder']['weight']
        ), "Decoder weight mismatch"

    print("[PASS] Params are identical after save/load")
    return True


def test_resume_training():
    """Test: Training continues correctly from checkpoint."""
    print("\n=== test_resume_training ===")

    config = MockConfig(seed=42)

    # === Path A: Continuous training for 10 epochs ===
    trainer_continuous = MinimalESTrainer(config)
    trainer_continuous.train(n_epochs=10)
    final_params_continuous = jax.tree.map(lambda x: x.copy(), trainer_continuous.params)

    # === Path B: Train 5, save, load, train 5 more ===
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Train for 5 epochs and save
        trainer_split = MinimalESTrainer(config)
        trainer_split.train(n_epochs=5)
        trainer_split.save_checkpoint(ckpt_path)

        # Load into new trainer
        trainer_resumed = MinimalESTrainer(config)
        trainer_resumed.load_checkpoint(ckpt_path)

        # Continue training for 5 more epochs using saved key
        trainer_resumed.train(n_epochs=5, start_epoch=5, use_saved_key=True)
        final_params_resumed = trainer_resumed.params

    # Verify results match
    assert params_equal(final_params_continuous, final_params_resumed), \
        "Training from checkpoint gives different params than continuous training"

    # Verify training histories have same length
    assert len(trainer_continuous.training_history) == len(trainer_resumed.training_history), \
        f"Training history length mismatch: {len(trainer_continuous.training_history)} vs {len(trainer_resumed.training_history)}"

    print("[PASS] Training continues correctly from checkpoint")
    return True


def test_epoch_counter_preserved():
    """Test: Epoch resumes from correct number."""
    print("\n=== test_epoch_counter_preserved ===")

    config = MockConfig(seed=42)
    trainer = MinimalESTrainer(config)

    # Train for 5 epochs
    trainer.train(n_epochs=5)
    assert trainer.current_epoch == 5, f"Expected epoch 5, got {trainer.current_epoch}"

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Save checkpoint
        trainer.save_checkpoint(ckpt_path)

        # Load into new trainer
        new_trainer = MinimalESTrainer(config)
        new_trainer.load_checkpoint(ckpt_path)

        # Verify epoch counter
        assert new_trainer.current_epoch == 5, \
            f"Epoch counter not restored: expected 5, got {new_trainer.current_epoch}"

        # Verify training_state.pkl has correct epoch
        with open(os.path.join(ckpt_path, 'training_state.pkl'), 'rb') as f:
            training_state = pickle.load(f)

        assert training_state['epoch'] == 5, \
            f"training_state.pkl epoch mismatch: {training_state['epoch']}"

        # Continue training using saved key
        new_trainer.train(n_epochs=3, start_epoch=5, use_saved_key=True)
        assert new_trainer.current_epoch == 8, \
            f"Epoch counter after resume incorrect: expected 8, got {new_trainer.current_epoch}"

    print("[PASS] Epoch counter is preserved correctly")
    return True


def test_deterministic_resume():
    """Test: Same results with same seed from checkpoint."""
    print("\n=== test_deterministic_resume ===")

    config = MockConfig(seed=12345)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Create checkpoint at epoch 3
        trainer_init = MinimalESTrainer(config)
        trainer_init.train(n_epochs=3)
        trainer_init.save_checkpoint(ckpt_path)

        # Run 1: Load and train for 5 epochs
        trainer1 = MinimalESTrainer(config)
        trainer1.load_checkpoint(ckpt_path)
        trainer1.train(n_epochs=5, start_epoch=3, use_saved_key=True)
        params1 = jax.tree.map(lambda x: x.copy(), trainer1.params)

        # Run 2: Load same checkpoint and train for 5 epochs
        trainer2 = MinimalESTrainer(config)
        trainer2.load_checkpoint(ckpt_path)
        trainer2.train(n_epochs=5, start_epoch=3, use_saved_key=True)
        params2 = trainer2.params

        # Verify identical results
        assert params_exactly_equal(params1, params2), \
            "Determinism failed: two runs from same checkpoint give different results"

        # Verify specific tensors match exactly
        diff_sum = sum(
            float(jnp.sum(jnp.abs(a - b)))
            for a, b in zip(jax.tree_util.tree_leaves(params1), jax.tree_util.tree_leaves(params2))
        )
        assert diff_sum == 0.0, f"Params differ by {diff_sum}"

    print("[PASS] Deterministic resume produces identical results")
    return True


def test_noiser_params_preserved():
    """Test: Noiser params (LoRA, momentum) are correctly restored."""
    print("\n=== test_noiser_params_preserved ===")

    config = MockConfig(seed=42)
    trainer = MinimalESTrainer(config)
    trainer.train(n_epochs=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        trainer.save_checkpoint(ckpt_path)

        new_trainer = MinimalESTrainer(config)
        new_trainer.load_checkpoint(ckpt_path)

        # Verify scalar params
        assert new_trainer.noiser_params['sigma'] == trainer.noiser_params['sigma']
        assert new_trainer.noiser_params['lr'] == trainer.noiser_params['lr']
        assert new_trainer.noiser_params['rank'] == trainer.noiser_params['rank']

        # Verify LoRA matrices
        assert jnp.array_equal(
            new_trainer.noiser_params['lora_matrices']['layer_0']['A'],
            trainer.noiser_params['lora_matrices']['layer_0']['A']
        ), "LoRA A matrix mismatch"

        assert jnp.array_equal(
            new_trainer.noiser_params['lora_matrices']['layer_0']['B'],
            trainer.noiser_params['lora_matrices']['layer_0']['B']
        ), "LoRA B matrix mismatch"

        # Verify momentum
        assert jnp.array_equal(
            new_trainer.noiser_params['momentum']['layer_0'],
            trainer.noiser_params['momentum']['layer_0']
        ), "Momentum mismatch"

    print("[PASS] Noiser params are preserved correctly")
    return True


def test_optimizer_state_preserved():
    """Test: Optimizer state (step count) is correctly restored."""
    print("\n=== test_optimizer_state_preserved ===")

    config = MockConfig(seed=42)
    trainer = MinimalESTrainer(config)
    trainer.train(n_epochs=5)

    # Optimizer should have been updated 5 times
    original_count = trainer.opt_state.count
    assert original_count == 5, f"Expected optimizer count 5, got {original_count}"

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        trainer.save_checkpoint(ckpt_path)

        new_trainer = MinimalESTrainer(config)
        new_trainer.load_checkpoint(ckpt_path)

        # Verify optimizer state
        assert new_trainer.opt_state.count == original_count, \
            f"Optimizer count mismatch: {new_trainer.opt_state.count} vs {original_count}"

    print("[PASS] Optimizer state is preserved correctly")
    return True


def test_training_history_preserved():
    """Test: Training history is correctly restored."""
    print("\n=== test_training_history_preserved ===")

    config = MockConfig(seed=42)
    trainer = MinimalESTrainer(config)
    trainer.train(n_epochs=5)

    original_history = trainer.training_history.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        trainer.save_checkpoint(ckpt_path)

        new_trainer = MinimalESTrainer(config)
        new_trainer.load_checkpoint(ckpt_path)

        # Verify history length
        assert len(new_trainer.training_history) == len(original_history), \
            f"History length mismatch: {len(new_trainer.training_history)} vs {len(original_history)}"

        # Verify history contents
        for i, (orig, loaded) in enumerate(zip(original_history, new_trainer.training_history)):
            assert orig['epoch'] == loaded['epoch'], f"Epoch mismatch at index {i}"
            assert np.isclose(orig['fitness'], loaded['fitness']), f"Fitness mismatch at index {i}"

    print("[PASS] Training history is preserved correctly")
    return True


def test_checkpoint_file_structure():
    """Test: Checkpoint files have expected structure."""
    print("\n=== test_checkpoint_file_structure ===")

    config = MockConfig(seed=42)
    trainer = MinimalESTrainer(config)
    trainer.train(n_epochs=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')
        trainer.save_checkpoint(ckpt_path)

        # Verify directory exists
        assert os.path.isdir(ckpt_path), f"Checkpoint directory not created: {ckpt_path}"

        # Verify main checkpoint file exists
        main_ckpt = os.path.join(ckpt_path, 'es_checkpoint.pkl')
        assert os.path.isfile(main_ckpt), f"es_checkpoint.pkl not found: {main_ckpt}"

        # Verify training state file exists
        state_file = os.path.join(ckpt_path, 'training_state.pkl')
        assert os.path.isfile(state_file), f"training_state.pkl not found: {state_file}"

        # Verify main checkpoint contents
        with open(main_ckpt, 'rb') as f:
            checkpoint = pickle.load(f)

        expected_keys = {'params', 'frozen_params', 'noiser_params', 'opt_state',
                         'key', 'current_epoch', 'training_history', 'config'}
        assert set(checkpoint.keys()) == expected_keys, \
            f"Unexpected checkpoint keys: {set(checkpoint.keys())} vs {expected_keys}"

        # Verify training state contents
        with open(state_file, 'rb') as f:
            training_state = pickle.load(f)

        expected_state_keys = {'epoch', 'best_fitness'}
        assert set(training_state.keys()) == expected_state_keys, \
            f"Unexpected training_state keys: {set(training_state.keys())} vs {expected_state_keys}"

    print("[PASS] Checkpoint file structure is correct")
    return True


def test_multiple_save_load_cycles():
    """Test: Multiple save/load cycles maintain data integrity."""
    print("\n=== test_multiple_save_load_cycles ===")

    config = MockConfig(seed=42)
    trainer = MinimalESTrainer(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        for cycle in range(5):
            ckpt_path = os.path.join(tmpdir, f'ckpt_{cycle}')

            # Train for a few epochs (use_saved_key after first cycle)
            use_saved = cycle > 0
            trainer.train(n_epochs=2, start_epoch=trainer.current_epoch, use_saved_key=use_saved)

            # Save
            trainer.save_checkpoint(ckpt_path)

            # Load into new trainer
            new_trainer = MinimalESTrainer(config)
            new_trainer.load_checkpoint(ckpt_path)

            # Verify epoch
            expected_epoch = (cycle + 1) * 2
            assert new_trainer.current_epoch == expected_epoch, \
                f"Cycle {cycle}: epoch mismatch, expected {expected_epoch}, got {new_trainer.current_epoch}"

            # Continue with loaded trainer
            trainer = new_trainer

        # Verify we can still load any checkpoint
        for cycle in range(5):
            ckpt_path = os.path.join(tmpdir, f'ckpt_{cycle}')
            test_trainer = MinimalESTrainer(config)
            test_trainer.load_checkpoint(ckpt_path)
            expected_epoch = (cycle + 1) * 2
            assert test_trainer.current_epoch == expected_epoch, \
                f"Failed to load checkpoint {cycle}: epoch mismatch"

    print("[PASS] Multiple save/load cycles maintain data integrity")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E2E Test: Checkpoint Save/Load/Resume Verification (Level 3)")
    print("=" * 70)

    all_tests_passed = True

    tests = [
        test_save_load_params,
        test_resume_training,
        test_epoch_counter_preserved,
        test_deterministic_resume,
        test_noiser_params_preserved,
        test_optimizer_state_preserved,
        test_training_history_preserved,
        test_checkpoint_file_structure,
        test_multiple_save_load_cycles,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            all_tests_passed = False

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if all_tests_passed:
        print("All E2E checkpoint resume tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)
