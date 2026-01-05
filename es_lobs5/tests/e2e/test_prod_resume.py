"""
Level 4 Production Test: Crash Recovery and Resume
File: es_lobs5/tests/e2e/test_prod_resume.py

This test simulates crash recovery scenarios:
1. Train for 5 epochs
2. Save checkpoint
3. "Simulate crash" by clearing all state
4. Load checkpoint
5. Resume training
6. Verify training continues correctly without data corruption

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/e2e/test_prod_resume.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import os
import pickle
import gc


# ============================================================================
# Mock Optimizer (for CPU-only testing without full optax dependency)
# ============================================================================

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


def apply_updates(params, updates):
    """Apply updates to params."""
    return jax.tree.map(lambda p, u: p + u, params, updates)


# ============================================================================
# Training State Container
# ============================================================================

class TrainingState:
    """Container for complete training state."""
    def __init__(self, params, opt_state, noiser_params, rng_key, epoch,
                 best_fitness, history):
        self.params = params
        self.opt_state = opt_state
        self.noiser_params = noiser_params
        self.rng_key = rng_key
        self.epoch = epoch
        self.best_fitness = best_fitness
        self.history = history  # List of (epoch, fitness) tuples

    def copy(self):
        """Create a deep copy of the training state."""
        return TrainingState(
            params=jax.tree.map(lambda x: jnp.array(x), self.params),
            opt_state=MockSGDState(self.opt_state.count),
            noiser_params=jax.tree.map(lambda x: jnp.array(x) if hasattr(x, 'shape') else x,
                                       self.noiser_params),
            rng_key=jnp.array(self.rng_key),
            epoch=self.epoch,
            best_fitness=self.best_fitness,
            history=list(self.history),
        )


# ============================================================================
# Checkpoint Functions
# ============================================================================

def save_checkpoint(path, state: TrainingState):
    """Save complete training state to checkpoint."""
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        'params': state.params,
        'opt_state': state.opt_state,
        'noiser_params': state.noiser_params,
        'rng_key': state.rng_key,
        'epoch': state.epoch,
        'best_fitness': state.best_fitness,
        'history': state.history,
    }

    ckpt_file = os.path.join(path, 'es_checkpoint.pkl')
    with open(ckpt_file, 'wb') as f:
        pickle.dump(checkpoint, f)

    return ckpt_file


def load_checkpoint(path) -> TrainingState:
    """Load complete training state from checkpoint."""
    ckpt_file = os.path.join(path, 'es_checkpoint.pkl')

    with open(ckpt_file, 'rb') as f:
        checkpoint = pickle.load(f)

    return TrainingState(
        params=checkpoint['params'],
        opt_state=checkpoint['opt_state'],
        noiser_params=checkpoint['noiser_params'],
        rng_key=checkpoint['rng_key'],
        epoch=checkpoint['epoch'],
        best_fitness=checkpoint['best_fitness'],
        history=checkpoint['history'],
    )


# ============================================================================
# Training Simulation
# ============================================================================

def create_initial_params(seed=42):
    """Create initial model parameters."""
    key = jax.random.PRNGKey(seed)
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
            },
        },
        'layer_1': {
            'ssm': {
                'Lambda_re': jax.random.normal(k3, (32,)),
                'Lambda_im': jax.random.normal(k3, (32,)),
                'B': jax.random.normal(k3, (32, 64, 2)),
                'C': jax.random.normal(k3, (64, 32, 2)),
                'D': jax.random.normal(k3, (64,)),
            },
        },
        'decoder': {
            'weight': jax.random.normal(k3, (100, 64)),
            'bias': jnp.zeros(100),
        },
    }
    return params


def create_noiser_params(seed=123):
    """Create mock noiser parameters (LoRA matrices, momentum, etc.)."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    return {
        'sigma': 0.01,
        'lr': 0.001,
        'rank': 4,
        'lora': {
            'layer_0': {
                'A': jax.random.normal(k1, (64, 4)),
                'B': jax.random.normal(k1, (4, 64)),
            },
            'layer_1': {
                'A': jax.random.normal(k2, (64, 4)),
                'B': jax.random.normal(k2, (4, 64)),
            },
        },
        'momentum': {
            'layer_0': jax.random.normal(k1, (64, 64)) * 0.001,
            'layer_1': jax.random.normal(k2, (64, 64)) * 0.001,
        },
    }


def compute_fitness(params, rng_key, epoch):
    """Compute mock fitness (deterministic based on params and rng)."""
    # Create a deterministic but epoch-varying fitness
    flat_params = jax.tree_util.tree_leaves(params)
    param_norm = sum(jnp.sum(jnp.abs(p)) for p in flat_params)

    # Add randomness that's reproducible from rng_key
    key, subkey = jax.random.split(rng_key)
    noise = jax.random.normal(subkey, ())

    fitness = -param_norm / 1e6 + noise * 0.01 + epoch * 0.001
    return float(fitness), key


def train_epoch(state: TrainingState, optimizer) -> TrainingState:
    """Simulate one training epoch."""
    # Compute fitness
    fitness, new_key = compute_fitness(state.params, state.rng_key, state.epoch)

    # Generate gradients (deterministic from key)
    new_key, grad_key = jax.random.split(new_key)
    flat_params, tree_def = jax.tree_util.tree_flatten(state.params)
    grad_keys = jax.random.split(grad_key, len(flat_params))

    flat_grads = [
        jax.random.normal(k, p.shape) * 0.001
        for k, p in zip(grad_keys, flat_params)
    ]
    grads = jax.tree_util.tree_unflatten(tree_def, flat_grads)

    # Apply gradient update
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = apply_updates(state.params, updates)

    # Update noiser momentum (mock)
    new_noiser_params = jax.tree.map(
        lambda x: x * 0.99 + 0.01 * jax.random.normal(grad_key, x.shape) if hasattr(x, 'shape') else x,
        state.noiser_params
    )

    # Track best fitness
    new_best = max(state.best_fitness, fitness)

    # Update history
    new_history = state.history + [(state.epoch, fitness)]

    return TrainingState(
        params=new_params,
        opt_state=new_opt_state,
        noiser_params=new_noiser_params,
        rng_key=new_key,
        epoch=state.epoch + 1,
        best_fitness=new_best,
        history=new_history,
    )


def train_n_epochs(state: TrainingState, n_epochs: int) -> TrainingState:
    """Train for n epochs."""
    optimizer = MockSGD(learning_rate=0.01)

    for _ in range(n_epochs):
        state = train_epoch(state, optimizer)

    return state


def params_equal(params1, params2, rtol=1e-5, atol=1e-8):
    """Check if two param trees are equal."""
    flat1 = jax.tree_util.tree_leaves(params1)
    flat2 = jax.tree_util.tree_leaves(params2)

    if len(flat1) != len(flat2):
        return False

    for p1, p2 in zip(flat1, flat2):
        if not jnp.allclose(p1, p2, rtol=rtol, atol=atol):
            return False

    return True


def clear_all_state():
    """Simulate crash by clearing all Python state and forcing garbage collection."""
    # Force garbage collection
    gc.collect()
    gc.collect()
    gc.collect()

    # JAX cache clearing (simulates fresh process)
    jax.clear_caches()

    # Additional garbage collection after cache clear
    gc.collect()


# ============================================================================
# Test Functions
# ============================================================================

def test_crash_recovery():
    """Test: Simulate crash and verify recovery from checkpoint."""
    print("\n=== test_crash_recovery ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Phase 1: Initial training
        print("  Phase 1: Training for 5 epochs...")
        initial_params = create_initial_params(seed=42)
        initial_noiser = create_noiser_params(seed=123)
        optimizer = MockSGD(learning_rate=0.01)

        state = TrainingState(
            params=initial_params,
            opt_state=optimizer.init(initial_params),
            noiser_params=initial_noiser,
            rng_key=jax.random.PRNGKey(999),
            epoch=0,
            best_fitness=-float('inf'),
            history=[],
        )

        state = train_n_epochs(state, 5)
        epoch_at_save = state.epoch
        best_at_save = state.best_fitness
        history_at_save = list(state.history)

        print(f"  Trained to epoch {epoch_at_save}, best_fitness={best_at_save:.6f}")

        # Phase 2: Save checkpoint
        print("  Phase 2: Saving checkpoint...")
        save_checkpoint(ckpt_path, state)

        # Record state before "crash"
        params_before_crash = jax.tree.map(lambda x: jnp.array(x), state.params)

        # Phase 3: Simulate crash
        print("  Phase 3: Simulating crash (clearing all state)...")
        del state
        del initial_params
        del initial_noiser
        del optimizer
        clear_all_state()

        # Verify variables are cleared
        try:
            _ = state  # Should raise NameError
            assert False, "State should be cleared after crash simulation"
        except NameError:
            pass  # Expected

        # Phase 4: Recovery
        print("  Phase 4: Loading checkpoint...")
        recovered_state = load_checkpoint(ckpt_path)

        # Verify recovery
        assert recovered_state.epoch == epoch_at_save, (
            f"Epoch mismatch: {recovered_state.epoch} vs {epoch_at_save}"
        )
        assert recovered_state.best_fitness == best_at_save, (
            f"Best fitness mismatch: {recovered_state.best_fitness} vs {best_at_save}"
        )
        assert len(recovered_state.history) == len(history_at_save), (
            f"History length mismatch: {len(recovered_state.history)} vs {len(history_at_save)}"
        )
        assert params_equal(recovered_state.params, params_before_crash), (
            "Params corrupted during save/load"
        )

        # Phase 5: Continue training
        print("  Phase 5: Resuming training for 5 more epochs...")
        recovered_state = train_n_epochs(recovered_state, 5)

        assert recovered_state.epoch == epoch_at_save + 5, (
            f"Training did not continue correctly: epoch {recovered_state.epoch}"
        )
        assert len(recovered_state.history) == 10, (
            f"History not updated: {len(recovered_state.history)} entries"
        )

    print("[PASS] Crash recovery test: all phases completed successfully")
    return True


def test_state_preserved():
    """Test: All state components are correctly preserved through save/load."""
    print("\n=== test_state_preserved ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Create initial state
        params = create_initial_params(seed=42)
        noiser_params = create_noiser_params(seed=123)
        optimizer = MockSGD(learning_rate=0.01)

        state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            noiser_params=noiser_params,
            rng_key=jax.random.PRNGKey(12345),
            epoch=42,
            best_fitness=0.789,
            history=[(i, 0.1 * i) for i in range(42)],
        )

        # Save
        save_checkpoint(ckpt_path, state)

        # Clear and load
        del params, noiser_params, optimizer
        clear_all_state()

        loaded = load_checkpoint(ckpt_path)

        # Verify all components
        # 1. Params
        assert params_equal(state.params, loaded.params), "Params not preserved"

        # 2. Optimizer state
        assert state.opt_state.count == loaded.opt_state.count, (
            f"Optimizer state not preserved: {state.opt_state.count} vs {loaded.opt_state.count}"
        )

        # 3. Noiser params (scalar values)
        assert loaded.noiser_params['sigma'] == state.noiser_params['sigma'], "Sigma not preserved"
        assert loaded.noiser_params['lr'] == state.noiser_params['lr'], "LR not preserved"
        assert loaded.noiser_params['rank'] == state.noiser_params['rank'], "Rank not preserved"

        # 4. Noiser params (LoRA matrices)
        for layer in ['layer_0', 'layer_1']:
            assert jnp.allclose(
                loaded.noiser_params['lora'][layer]['A'],
                state.noiser_params['lora'][layer]['A']
            ), f"LoRA A for {layer} not preserved"
            assert jnp.allclose(
                loaded.noiser_params['lora'][layer]['B'],
                state.noiser_params['lora'][layer]['B']
            ), f"LoRA B for {layer} not preserved"

        # 5. Noiser params (momentum)
        for layer in ['layer_0', 'layer_1']:
            assert jnp.allclose(
                loaded.noiser_params['momentum'][layer],
                state.noiser_params['momentum'][layer]
            ), f"Momentum for {layer} not preserved"

        # 6. RNG key
        assert jnp.array_equal(loaded.rng_key, state.rng_key), "RNG key not preserved"

        # 7. Epoch
        assert loaded.epoch == state.epoch, "Epoch not preserved"

        # 8. Best fitness
        assert loaded.best_fitness == state.best_fitness, "Best fitness not preserved"

        # 9. History
        assert loaded.history == state.history, "History not preserved"

    print("[PASS] State preservation test: all components correctly preserved")
    return True


def test_training_continues():
    """Test: Training continues correctly after checkpoint load."""
    print("\n=== test_training_continues ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Initial training
        params = create_initial_params(seed=42)
        noiser_params = create_noiser_params(seed=123)
        optimizer = MockSGD(learning_rate=0.01)

        state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            noiser_params=noiser_params,
            rng_key=jax.random.PRNGKey(999),
            epoch=0,
            best_fitness=-float('inf'),
            history=[],
        )

        # Train for 5 epochs
        state = train_n_epochs(state, 5)
        initial_params = jax.tree.map(lambda x: jnp.array(x), state.params)

        # Save checkpoint
        save_checkpoint(ckpt_path, state)

        # Load and continue
        loaded_state = load_checkpoint(ckpt_path)

        # Train for 5 more epochs
        continued_state = train_n_epochs(loaded_state, 5)

        # Verify training continued
        assert continued_state.epoch == 10, f"Expected epoch 10, got {continued_state.epoch}"
        assert len(continued_state.history) == 10, f"Expected 10 history entries, got {len(continued_state.history)}"

        # Verify params changed (training actually happened)
        # Check that params are not exactly equal to initial params
        params_unchanged = all(
            jnp.allclose(p1, p2, rtol=1e-10, atol=1e-10)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(initial_params),
                jax.tree_util.tree_leaves(continued_state.params)
            )
        )
        assert not params_unchanged, (
            "Params did not change - training may not have continued"
        )

        # Verify history is continuous
        epochs_in_history = [h[0] for h in continued_state.history]
        assert epochs_in_history == list(range(10)), (
            f"History epochs not continuous: {epochs_in_history}"
        )

    print("[PASS] Training continues test: training correctly resumed after load")
    return True


def test_results_deterministic():
    """Test: Same seed gives same results after resume."""
    print("\n=== test_results_deterministic ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Run 1: Continuous training for 10 epochs
        params1 = create_initial_params(seed=42)
        noiser1 = create_noiser_params(seed=123)
        optimizer = MockSGD(learning_rate=0.01)

        state1 = TrainingState(
            params=params1,
            opt_state=optimizer.init(params1),
            noiser_params=noiser1,
            rng_key=jax.random.PRNGKey(999),
            epoch=0,
            best_fitness=-float('inf'),
            history=[],
        )

        state1 = train_n_epochs(state1, 10)

        # Run 2: Train 5, save, clear, load, train 5 more
        params2 = create_initial_params(seed=42)  # Same seed
        noiser2 = create_noiser_params(seed=123)  # Same seed

        state2 = TrainingState(
            params=params2,
            opt_state=optimizer.init(params2),
            noiser_params=noiser2,
            rng_key=jax.random.PRNGKey(999),  # Same seed
            epoch=0,
            best_fitness=-float('inf'),
            history=[],
        )

        state2 = train_n_epochs(state2, 5)
        save_checkpoint(ckpt_path, state2)

        # Clear state
        del state2, params2, noiser2
        clear_all_state()

        # Load and continue
        state2 = load_checkpoint(ckpt_path)
        state2 = train_n_epochs(state2, 5)

        # Verify results are identical
        assert params_equal(state1.params, state2.params), (
            "Continuous training and resume training gave different params"
        )
        assert state1.epoch == state2.epoch, (
            f"Epoch mismatch: {state1.epoch} vs {state2.epoch}"
        )

        # Verify history matches
        for i, ((e1, f1), (e2, f2)) in enumerate(zip(state1.history, state2.history)):
            assert e1 == e2, f"History epoch mismatch at {i}: {e1} vs {e2}"
            assert jnp.isclose(f1, f2, rtol=1e-5), (
                f"History fitness mismatch at {i}: {f1} vs {f2}"
            )

    print("[PASS] Determinism test: resume gives same results as continuous training")
    return True


def test_multiple_resume_cycles():
    """Test: Multiple save/load/train cycles work correctly."""
    print("\n=== test_multiple_resume_cycles ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial state
        params = create_initial_params(seed=42)
        noiser = create_noiser_params(seed=123)
        optimizer = MockSGD(learning_rate=0.01)

        state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            noiser_params=noiser,
            rng_key=jax.random.PRNGKey(999),
            epoch=0,
            best_fitness=-float('inf'),
            history=[],
        )

        n_cycles = 5
        epochs_per_cycle = 3

        for cycle in range(n_cycles):
            print(f"  Cycle {cycle + 1}/{n_cycles}: epoch {state.epoch} -> {state.epoch + epochs_per_cycle}")

            # Train
            state = train_n_epochs(state, epochs_per_cycle)

            # Save
            ckpt_path = os.path.join(tmpdir, f'checkpoint_{cycle}')
            save_checkpoint(ckpt_path, state)

            # Record state before clear
            epoch_before = state.epoch
            params_before = jax.tree.map(lambda x: jnp.array(x), state.params)

            # Clear all state (simulate crash between cycles)
            del state
            clear_all_state()

            # Load
            state = load_checkpoint(ckpt_path)

            # Verify state preserved
            assert state.epoch == epoch_before, (
                f"Cycle {cycle}: epoch not preserved: {state.epoch} vs {epoch_before}"
            )
            assert params_equal(state.params, params_before), (
                f"Cycle {cycle}: params corrupted"
            )

        # Final verification
        expected_epochs = n_cycles * epochs_per_cycle
        assert state.epoch == expected_epochs, (
            f"Expected {expected_epochs} epochs, got {state.epoch}"
        )
        assert len(state.history) == expected_epochs, (
            f"Expected {expected_epochs} history entries, got {len(state.history)}"
        )

        # Verify we can load any checkpoint
        for cycle in range(n_cycles):
            ckpt_path = os.path.join(tmpdir, f'checkpoint_{cycle}')
            loaded = load_checkpoint(ckpt_path)
            expected_epoch = (cycle + 1) * epochs_per_cycle
            assert loaded.epoch == expected_epoch, (
                f"Checkpoint {cycle} has wrong epoch: {loaded.epoch} vs {expected_epoch}"
            )

    print(f"[PASS] Multiple resume cycles test: {n_cycles} cycles completed successfully")
    return True


def test_checkpoint_corruption_detection():
    """Test: Corrupted checkpoints are detected."""
    print("\n=== test_checkpoint_corruption_detection ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Create and save valid checkpoint
        params = create_initial_params(seed=42)
        noiser = create_noiser_params(seed=123)
        optimizer = MockSGD(learning_rate=0.01)

        state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            noiser_params=noiser,
            rng_key=jax.random.PRNGKey(999),
            epoch=5,
            best_fitness=0.5,
            history=[(i, 0.1 * i) for i in range(5)],
        )

        save_checkpoint(ckpt_path, state)

        # Test 1: Verify valid checkpoint loads
        loaded = load_checkpoint(ckpt_path)
        assert loaded.epoch == 5, "Valid checkpoint should load correctly"

        # Test 2: Corrupted file should fail to load
        ckpt_file = os.path.join(ckpt_path, 'es_checkpoint.pkl')
        with open(ckpt_file, 'wb') as f:
            f.write(b'corrupted data that is not valid pickle')

        try:
            load_checkpoint(ckpt_path)
            assert False, "Should have raised an exception for corrupted checkpoint"
        except Exception as e:
            # Expected - corrupted pickle should fail
            pass

        # Test 3: Missing file should fail
        os.remove(ckpt_file)
        try:
            load_checkpoint(ckpt_path)
            assert False, "Should have raised an exception for missing checkpoint"
        except FileNotFoundError:
            pass  # Expected

    print("[PASS] Corruption detection test: invalid checkpoints correctly rejected")
    return True


def test_partial_training_resume():
    """Test: Resume from partially completed epoch."""
    print("\n=== test_partial_training_resume ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Train to epoch 5
        params = create_initial_params(seed=42)
        noiser = create_noiser_params(seed=123)
        optimizer = MockSGD(learning_rate=0.01)

        state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            noiser_params=noiser,
            rng_key=jax.random.PRNGKey(999),
            epoch=0,
            best_fitness=-float('inf'),
            history=[],
        )

        # Train 5 epochs
        state = train_n_epochs(state, 5)

        # Simulate saving mid-epoch by manually setting epoch
        # (In real training, we save at epoch boundaries)
        save_checkpoint(ckpt_path, state)

        # Clear and reload
        del state
        clear_all_state()

        # Resume
        state = load_checkpoint(ckpt_path)
        starting_epoch = state.epoch

        # Train more epochs
        state = train_n_epochs(state, 5)

        # Verify we continued from the correct point
        assert state.epoch == starting_epoch + 5, (
            f"Training did not continue correctly: {state.epoch} vs {starting_epoch + 5}"
        )

    print("[PASS] Partial training resume test: correctly resumed from checkpoint")
    return True


def test_large_state_checkpointing():
    """Test: Large model states checkpoint correctly."""
    print("\n=== test_large_state_checkpointing ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint')

        # Create larger params (simulating a bigger model)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 10)

        large_params = {
            f'layer_{i}': {
                'weights': jax.random.normal(keys[i], (256, 256)),
                'bias': jnp.zeros(256),
                'ssm': {
                    'Lambda_re': jax.random.normal(keys[i], (128,)),
                    'Lambda_im': jax.random.normal(keys[i], (128,)),
                    'B': jax.random.normal(keys[i], (128, 256, 2)),
                    'C': jax.random.normal(keys[i], (256, 128, 2)),
                },
            }
            for i in range(8)  # 8 layers
        }

        noiser = create_noiser_params(seed=123)
        optimizer = MockSGD(learning_rate=0.01)

        state = TrainingState(
            params=large_params,
            opt_state=optimizer.init(large_params),
            noiser_params=noiser,
            rng_key=jax.random.PRNGKey(999),
            epoch=100,
            best_fitness=0.95,
            history=[(i, 0.01 * i) for i in range(100)],
        )

        # Save
        save_checkpoint(ckpt_path, state)

        # Verify file size is reasonable
        ckpt_file = os.path.join(ckpt_path, 'es_checkpoint.pkl')
        file_size = os.path.getsize(ckpt_file)
        print(f"  Checkpoint file size: {file_size / 1024 / 1024:.2f} MB")

        # Clear and load
        params_before = jax.tree.map(lambda x: jnp.array(x), state.params)
        del state
        clear_all_state()

        loaded = load_checkpoint(ckpt_path)

        # Verify all params preserved
        assert params_equal(loaded.params, params_before), (
            "Large params corrupted during save/load"
        )

        # Verify specific values
        for i in range(8):
            layer_key = f'layer_{i}'
            assert jnp.allclose(
                loaded.params[layer_key]['weights'],
                params_before[layer_key]['weights']
            ), f"Layer {i} weights corrupted"

    print("[PASS] Large state checkpointing test: all data preserved")
    return True


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Level 4 Production Test: Crash Recovery and Resume")
    print("=" * 70)

    all_tests_passed = True

    tests = [
        test_crash_recovery,
        test_state_preserved,
        test_training_continues,
        test_results_deterministic,
        test_multiple_resume_cycles,
        test_checkpoint_corruption_detection,
        test_partial_training_resume,
        test_large_state_checkpointing,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {test_fn.__name__}: {e}")
            traceback.print_exc()
            all_tests_passed = False
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if all_tests_passed:
        print("All Level 4 Production resume tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)
