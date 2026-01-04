"""Test Eggroll LoRA decomposition."""
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/HyperscaleES/src')


# Copy the core LoRA functions from HyperscaleES/src/hyperscalees/noiser/eggroll.py
# to test independently without gymnax dependency

def get_lora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    """Generate LoRA decomposition A and B matrices.

    The update is computed as A @ B.T where:
    - A: (a, r) - scaled by sigma
    - B: (b, r)
    - param shape: (a, b)
    """
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    a, b = param.shape
    lora_params = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        (a + b, frozen_noiser_params["rank"]),
        dtype=param.dtype
    )
    B = lora_params[:b]  # b x r
    A = lora_params[b:]  # a x r

    # update is A @ B.T
    return A * sigma, B


class EggRoll:
    """Minimal EggRoll noiser for testing LoRA functionality."""

    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None,
                    group_size=0, freeze_nonlora=False, noise_reuse=0, rank=1, **kwargs):
        """Initialize noiser with frozen and mutable parameters."""
        if solver is None:
            solver = optax.sgd
        if solver_kwargs is None:
            solver_kwargs = {}
        true_solver = solver(lr, **solver_kwargs)
        opt_state = true_solver.init(params)

        return {
            "group_size": group_size,
            "freeze_nonlora": freeze_nonlora,
            "noise_reuse": noise_reuse,
            "solver": true_solver,
            "rank": rank
        }, {"sigma": sigma, "opt_state": opt_state}

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        """Matrix multiplication with LoRA perturbation: x @ (param + A @ B.T).T"""
        base_ans = x @ param.T
        if iterinfo is None:
            return base_ans
        A, B = get_lora_update_params(
            frozen_noiser_params,
            noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]),
            iterinfo, param, base_key
        )
        return base_ans + x @ B @ A.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        """Transposed matrix multiplication with LoRA perturbation."""
        base_ans = x @ param
        if iterinfo is None:
            return base_ans
        A, B = get_lora_update_params(
            frozen_noiser_params,
            noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]),
            iterinfo, param, base_key
        )
        return base_ans + x @ A @ B.T


def test_lora_rank():
    """Test: LoRA decomposition A: (r, d), B: (d, r)."""
    key = jax.random.PRNGKey(42)

    # Test parameters
    rank = 4
    a_dim = 64   # output dimension
    b_dim = 128  # input dimension
    sigma = 0.1

    # Create a 2D parameter matrix
    param = jax.random.normal(key, (a_dim, b_dim))

    # Setup frozen_noiser_params
    frozen_noiser_params = {
        "rank": rank,
        "noise_reuse": 0,
    }

    iterinfo = (0, 0)  # (epoch, thread_id)
    param_key = jax.random.PRNGKey(123)

    # Get LoRA decomposition
    A, B = get_lora_update_params(frozen_noiser_params, sigma, iterinfo, param, param_key)

    # Verify shapes:
    # A should have shape (a_dim, rank) - scaled by sigma
    # B should have shape (b_dim, rank)
    # The update is A @ B.T which gives (a_dim, b_dim)

    assert A.shape == (a_dim, rank), f"A shape should be ({a_dim}, {rank}), got {A.shape}"
    assert B.shape == (b_dim, rank), f"B shape should be ({b_dim}, {rank}), got {B.shape}"

    # Verify that A @ B.T has the correct shape
    update = A @ B.T
    assert update.shape == param.shape, f"Update shape should be {param.shape}, got {update.shape}"

    # Test with different ranks
    for test_rank in [1, 2, 8, 16]:
        frozen_noiser_params["rank"] = test_rank
        A_test, B_test = get_lora_update_params(frozen_noiser_params, sigma, iterinfo, param, param_key)

        assert A_test.shape == (a_dim, test_rank), f"A shape should be ({a_dim}, {test_rank}), got {A_test.shape}"
        assert B_test.shape == (b_dim, test_rank), f"B shape should be ({b_dim}, {test_rank}), got {B_test.shape}"

    print(f"[PASS] test_lora_rank: A shape {A.shape}, B shape {B.shape}")
    return True


def test_lora_reconstruction():
    """Test: A @ B approximately reconstructs the perturbation pattern."""
    key = jax.random.PRNGKey(42)

    # Test parameters
    rank = 16  # Higher rank for better approximation
    a_dim = 32
    b_dim = 64
    sigma = 0.1

    # Create a 2D parameter matrix
    param = jax.random.normal(key, (a_dim, b_dim))

    frozen_noiser_params = {
        "rank": rank,
        "noise_reuse": 0,
    }

    iterinfo = (0, 0)
    param_key = jax.random.PRNGKey(456)

    # Get LoRA decomposition
    A, B = get_lora_update_params(frozen_noiser_params, sigma, iterinfo, param, param_key)

    # Reconstruct the update
    update = A @ B.T

    # Verify properties:
    # 1. Update has same shape as param
    assert update.shape == param.shape, f"Update shape mismatch"

    # 2. Update is not zero (perturbation is applied)
    assert jnp.abs(update).max() > 1e-6, "Update should not be zero"

    # 3. Update has low rank structure (rank at most r)
    # This means the update can be exactly represented by A @ B.T
    U, S, Vt = jnp.linalg.svd(update, full_matrices=False)

    # Singular values beyond rank should be essentially zero
    # Use relative threshold based on largest singular value
    threshold = S[0] * 1e-5
    svd_rank = int(jnp.sum(S > threshold))
    assert svd_rank <= rank, f"Update should have rank at most {rank}, got {svd_rank}"

    # 4. Verify consistency: same key and iterinfo should give same decomposition
    A2, B2 = get_lora_update_params(frozen_noiser_params, sigma, iterinfo, param, param_key)
    update2 = A2 @ B2.T

    assert jnp.allclose(update, update2, atol=1e-6), "Same key should give same update"

    print(f"[PASS] test_lora_reconstruction: update shape {update.shape}, effective rank <= {rank}")
    return True


def test_lora_antithetic():
    """Test: thread_id 0 and 1 produce antithetic LoRA perturbations."""
    key = jax.random.PRNGKey(42)

    rank = 4
    a_dim = 32
    b_dim = 64
    sigma = 0.1

    param = jax.random.normal(key, (a_dim, b_dim))

    frozen_noiser_params = {
        "rank": rank,
        "noise_reuse": 0,
    }

    param_key = jax.random.PRNGKey(789)

    # Get LoRA decomposition for thread_id=0 (even)
    iterinfo_even = (0, 0)
    A_even, B_even = get_lora_update_params(frozen_noiser_params, sigma, iterinfo_even, param, param_key)
    update_even = A_even @ B_even.T

    # Get LoRA decomposition for thread_id=1 (odd)
    iterinfo_odd = (0, 1)
    A_odd, B_odd = get_lora_update_params(frozen_noiser_params, sigma, iterinfo_odd, param, param_key)
    update_odd = A_odd @ B_odd.T

    # Antithetic: updates should sum to zero (or close to it)
    sum_updates = update_even + update_odd
    max_sum = jnp.abs(sum_updates).max()

    assert max_sum < 1e-6, f"Antithetic updates should sum to zero, got max {max_sum}"

    print(f"[PASS] test_lora_antithetic: |update_even + update_odd| max = {max_sum:.2e}")
    return True


def test_lora_different_threads():
    """Test: Different thread pairs produce different LoRA perturbations."""
    key = jax.random.PRNGKey(42)

    rank = 4
    a_dim = 32
    b_dim = 64
    sigma = 0.1

    param = jax.random.normal(key, (a_dim, b_dim))

    frozen_noiser_params = {
        "rank": rank,
        "noise_reuse": 0,
    }

    param_key = jax.random.PRNGKey(321)

    # Thread pair 0-1 (true_thread_idx = 0)
    A_0, B_0 = get_lora_update_params(frozen_noiser_params, sigma, (0, 0), param, param_key)
    update_0 = A_0 @ B_0.T

    # Thread pair 2-3 (true_thread_idx = 1)
    A_2, B_2 = get_lora_update_params(frozen_noiser_params, sigma, (0, 2), param, param_key)
    update_2 = A_2 @ B_2.T

    # Different thread pairs should produce different updates
    diff = jnp.abs(update_0 - update_2).max()
    assert diff > 1e-6, f"Different thread pairs should produce different updates, diff = {diff}"

    print(f"[PASS] test_lora_different_threads: diff = {diff:.6f}")
    return True


def test_lora_noise_reuse():
    """Test: noise_reuse controls how noise is shared across epochs."""
    key = jax.random.PRNGKey(42)

    rank = 4
    a_dim = 32
    b_dim = 64
    sigma = 0.1

    param = jax.random.normal(key, (a_dim, b_dim))
    param_key = jax.random.PRNGKey(654)

    # With noise_reuse=2, epochs 0 and 1 should use same noise
    frozen_noiser_params = {
        "rank": rank,
        "noise_reuse": 2,
    }

    A_epoch0, B_epoch0 = get_lora_update_params(frozen_noiser_params, sigma, (0, 0), param, param_key)
    A_epoch1, B_epoch1 = get_lora_update_params(frozen_noiser_params, sigma, (1, 0), param, param_key)
    A_epoch2, B_epoch2 = get_lora_update_params(frozen_noiser_params, sigma, (2, 0), param, param_key)

    # Epochs 0 and 1 should have same noise (true_epoch = 0)
    assert jnp.allclose(A_epoch0, A_epoch1), "Epochs 0 and 1 should share noise with noise_reuse=2"
    assert jnp.allclose(B_epoch0, B_epoch1), "Epochs 0 and 1 should share noise with noise_reuse=2"

    # Epoch 2 should have different noise (true_epoch = 1)
    assert not jnp.allclose(A_epoch0, A_epoch2), "Epoch 2 should have different noise"

    print(f"[PASS] test_lora_noise_reuse: noise_reuse=2 correctly shares noise across epochs")
    return True


def test_lora_eggroll_integration():
    """Test: EggRoll noiser correctly uses LoRA for matrix multiplication."""
    key = jax.random.PRNGKey(42)

    rank = 4
    a_dim = 32
    b_dim = 64
    sigma = 0.1

    # Create a weight matrix (a_dim x b_dim)
    param = jax.random.normal(key, (a_dim, b_dim))

    # Initialize EggRoll noiser
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        {"weight": param},
        sigma=sigma,
        lr=0.001,
        rank=rank,
    )

    param_key = jax.random.PRNGKey(987)

    # Create input: (batch, b_dim) for x @ weight.T -> (batch, a_dim)
    batch_size = 8
    x = jax.random.normal(key, (batch_size, b_dim))

    # Forward pass without noise (iterinfo=None)
    out_clean = EggRoll.do_mm(frozen_noiser_params, noiser_params, param, param_key, None, x)
    expected_clean = x @ param.T
    assert jnp.allclose(out_clean, expected_clean, atol=1e-6), "Clean forward should match x @ param.T"

    # Forward pass with noise (iterinfo=(0, 0))
    iterinfo = (0, 0)
    out_noisy = EggRoll.do_mm(frozen_noiser_params, noiser_params, param, param_key, iterinfo, x)

    # Noisy output should differ from clean
    diff = jnp.abs(out_noisy - out_clean).max()
    assert diff > 1e-6, f"Noisy forward should differ from clean, diff = {diff}"

    # Verify the LoRA structure is applied correctly
    # out_noisy = x @ param.T + x @ B @ A.T
    A, B = get_lora_update_params(frozen_noiser_params, sigma / jnp.sqrt(rank), iterinfo, param, param_key)
    expected_noisy = x @ param.T + x @ B @ A.T
    assert jnp.allclose(out_noisy, expected_noisy, atol=1e-5), "Noisy forward should match x @ param.T + x @ B @ A.T"

    print(f"[PASS] test_lora_eggroll_integration: EggRoll correctly applies LoRA perturbation")
    return True


if __name__ == "__main__":
    test_lora_rank()
    test_lora_reconstruction()
    test_lora_antithetic()
    test_lora_different_threads()
    test_lora_noise_reuse()
    test_lora_eggroll_integration()
    print("All Eggroll LoRA tests passed!")
