"""
Level 4 Production Test: Metrics and Logging for ES Training

This test verifies production-level metrics collection and W&B compatibility:
1. Run training and collect metrics
2. Verify all expected metrics are computed:
   - mean_fitness, max_fitness, min_fitness, std_fitness
   - gradient_norm (per layer and total)
   - param_norm (parameter statistics)
   - learning_rate
   - epoch, step
3. Verify metrics format for W&B compatibility (JSON serializable)
4. Test metric aggregation over epochs

Run with: JAX_PLATFORMS=cpu python es_lobs5/tests/e2e/test_prod_metrics.py
"""
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')
import jax
import jax.numpy as jnp
import numpy as np
import json
from typing import Dict, List, Any, Optional


# ============================================================================
# MetricsCollector: Core class for tracking all training metrics
# ============================================================================

class MetricsCollector:
    """Collects and tracks all training metrics for production logging.

    This class manages:
    - Fitness metrics (mean, max, min, std)
    - Gradient metrics (per-layer norms, total norm)
    - Parameter metrics (norms, statistics)
    - Training progress (epoch, step, learning_rate)
    - Historical tracking over multiple epochs

    All metrics are stored as Python-native types for JSON serialization
    and W&B compatibility.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self.history: List[Dict[str, Any]] = []
        self.current_epoch: int = 0
        self.current_step: int = 0
        self._epoch_metrics: List[Dict[str, Any]] = []

    def reset(self):
        """Reset all tracked metrics."""
        self.history = []
        self.current_epoch = 0
        self.current_step = 0
        self._epoch_metrics = []

    def _to_python_scalar(self, value: Any) -> Any:
        """Convert JAX/numpy arrays to Python native types for JSON serialization.

        Args:
            value: Any value (JAX array, numpy array, or Python native)

        Returns:
            Python-native scalar (float, int) or the original value if already native
        """
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            if value.ndim == 0:
                # Scalar array
                return float(value) if value.dtype.kind == 'f' else int(value)
            else:
                # Return as list for non-scalar arrays
                return value.tolist()
        elif hasattr(value, 'item'):
            # JAX/numpy scalar types
            return value.item()
        elif isinstance(value, (float, int, str, bool, type(None))):
            return value
        elif isinstance(value, dict):
            return {k: self._to_python_scalar(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._to_python_scalar(v) for v in value]
        else:
            # Try to convert to float, fallback to str
            try:
                return float(value)
            except (TypeError, ValueError):
                return str(value)

    def compute_fitness_metrics(self, fitness_scores: jnp.ndarray) -> Dict[str, float]:
        """Compute fitness statistics from a batch of scores.

        Args:
            fitness_scores: Array of fitness values, shape (n_workers,)

        Returns:
            Dictionary with mean_fitness, max_fitness, min_fitness, std_fitness
        """
        return {
            'mean_fitness': self._to_python_scalar(jnp.mean(fitness_scores)),
            'max_fitness': self._to_python_scalar(jnp.max(fitness_scores)),
            'min_fitness': self._to_python_scalar(jnp.min(fitness_scores)),
            'std_fitness': self._to_python_scalar(jnp.std(fitness_scores)),
        }

    def compute_gradient_metrics(self, gradients: Dict) -> Dict[str, float]:
        """Compute gradient norms (per-layer and total).

        Args:
            gradients: Nested dictionary of gradient arrays

        Returns:
            Dictionary with gradient_norm (total) and gradient_norm/<layer_name>
        """
        metrics = {}
        total_sq_norm = 0.0

        def process_gradients(grads, prefix=""):
            nonlocal total_sq_norm
            if isinstance(grads, dict):
                for key, value in grads.items():
                    new_prefix = f"{prefix}/{key}" if prefix else key
                    process_gradients(value, new_prefix)
            elif isinstance(grads, (jnp.ndarray, np.ndarray)):
                layer_norm = float(jnp.sqrt(jnp.sum(grads ** 2)))
                metrics[f'gradient_norm/{prefix}'] = layer_norm
                total_sq_norm += layer_norm ** 2

        process_gradients(gradients)
        metrics['gradient_norm'] = float(np.sqrt(total_sq_norm))

        return metrics

    def compute_param_metrics(self, params: Dict) -> Dict[str, float]:
        """Compute parameter statistics (norms and statistics).

        Args:
            params: Nested dictionary of parameter arrays

        Returns:
            Dictionary with param_norm (total), param_mean, param_std,
            and per-layer norms
        """
        metrics = {}
        total_sq_norm = 0.0
        all_values = []

        def process_params(p, prefix=""):
            nonlocal total_sq_norm
            if isinstance(p, dict):
                for key, value in p.items():
                    new_prefix = f"{prefix}/{key}" if prefix else key
                    process_params(value, new_prefix)
            elif isinstance(p, (jnp.ndarray, np.ndarray)):
                layer_norm = float(jnp.sqrt(jnp.sum(p ** 2)))
                metrics[f'param_norm/{prefix}'] = layer_norm
                total_sq_norm += layer_norm ** 2
                all_values.append(p.flatten())

        process_params(params)
        metrics['param_norm'] = float(np.sqrt(total_sq_norm))

        # Compute overall statistics
        if all_values:
            all_params = jnp.concatenate(all_values)
            metrics['param_mean'] = self._to_python_scalar(jnp.mean(all_params))
            metrics['param_std'] = self._to_python_scalar(jnp.std(all_params))
            metrics['param_min'] = self._to_python_scalar(jnp.min(all_params))
            metrics['param_max'] = self._to_python_scalar(jnp.max(all_params))
            metrics['param_count'] = int(all_params.shape[0])

        return metrics

    def log_step(self,
                 epoch: int,
                 step: int,
                 fitness_scores: jnp.ndarray,
                 gradients: Optional[Dict] = None,
                 params: Optional[Dict] = None,
                 learning_rate: Optional[float] = None,
                 extra_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Log a single training step.

        Args:
            epoch: Current epoch number
            step: Current step within epoch
            fitness_scores: Array of fitness values
            gradients: Optional gradient dictionary
            params: Optional parameter dictionary
            learning_rate: Optional learning rate
            extra_metrics: Optional additional metrics

        Returns:
            Dictionary of all logged metrics (JSON-serializable)
        """
        self.current_epoch = epoch
        self.current_step = step

        # Core metrics
        metrics = {
            'epoch': int(epoch),
            'step': int(step),
        }

        # Fitness metrics
        metrics.update(self.compute_fitness_metrics(fitness_scores))

        # Gradient metrics
        if gradients is not None:
            metrics.update(self.compute_gradient_metrics(gradients))

        # Parameter metrics
        if params is not None:
            metrics.update(self.compute_param_metrics(params))

        # Learning rate
        if learning_rate is not None:
            metrics['learning_rate'] = self._to_python_scalar(learning_rate)

        # Extra metrics
        if extra_metrics is not None:
            for key, value in extra_metrics.items():
                metrics[key] = self._to_python_scalar(value)

        # Store in history
        self.history.append(metrics)
        self._epoch_metrics.append(metrics)

        return metrics

    def end_epoch(self) -> Dict[str, Any]:
        """Finalize and aggregate metrics for the current epoch.

        Returns:
            Dictionary of epoch-aggregated metrics
        """
        if not self._epoch_metrics:
            return {}

        # Aggregate metrics across steps in this epoch
        aggregated = {
            'epoch': self.current_epoch,
            'num_steps': len(self._epoch_metrics),
        }

        # Compute aggregated fitness metrics
        all_mean_fitness = [m['mean_fitness'] for m in self._epoch_metrics]
        aggregated['epoch_mean_fitness'] = float(np.mean(all_mean_fitness))
        aggregated['epoch_max_fitness'] = float(np.max([m['max_fitness'] for m in self._epoch_metrics]))
        aggregated['epoch_min_fitness'] = float(np.min([m['min_fitness'] for m in self._epoch_metrics]))

        # Aggregate gradient norms if available
        grad_norms = [m.get('gradient_norm', 0.0) for m in self._epoch_metrics]
        if any(g > 0 for g in grad_norms):
            aggregated['epoch_mean_gradient_norm'] = float(np.mean(grad_norms))
            aggregated['epoch_max_gradient_norm'] = float(np.max(grad_norms))

        # Reset epoch metrics for next epoch
        self._epoch_metrics = []

        return aggregated

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full metrics history.

        Returns:
            List of all logged metrics dictionaries
        """
        return self.history

    def to_json(self) -> str:
        """Serialize all history to JSON string.

        Returns:
            JSON string of all metrics
        """
        return json.dumps(self.history, indent=2)

    def get_metric_series(self, metric_name: str) -> List[float]:
        """Extract a single metric series from history.

        Args:
            metric_name: Name of the metric to extract

        Returns:
            List of metric values over time
        """
        return [m.get(metric_name) for m in self.history if metric_name in m]


# ============================================================================
# Mock Training Functions
# ============================================================================

def create_mock_params(key: jax.random.PRNGKey, layer_sizes: List = None) -> Dict:
    """Create mock model parameters."""
    if layer_sizes is None:
        layer_sizes = [(64, 128), (128, 64), (64, 32)]

    params = {}
    for i, (in_dim, out_dim) in enumerate(layer_sizes):
        key, k1, k2 = jax.random.split(key, 3)
        params[f'layer_{i}'] = {
            'weights': jax.random.normal(k1, (in_dim, out_dim)) * 0.1,
            'bias': jax.random.normal(k2, (out_dim,)) * 0.01,
        }
    return params


def create_mock_gradients(key: jax.random.PRNGKey, params: Dict) -> Dict:
    """Create mock gradients with same structure as params."""
    def gen_grad(k, p):
        return jax.random.normal(k, p.shape) * 0.01

    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(flat_params))
    flat_grads = [gen_grad(k, p) for k, p in zip(keys, flat_params)]
    return jax.tree_util.tree_unflatten(tree_def, flat_grads)


def simulate_training_step(key: jax.random.PRNGKey,
                           params: Dict,
                           n_workers: int = 16,
                           sigma: float = 0.1) -> tuple:
    """Simulate a single ES training step.

    Returns:
        (new_params, fitness_scores, gradients)
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Simulate fitness scores with some variance
    base_fitness = -1.0  # Negative MSE
    fitness_scores = base_fitness + sigma * jax.random.normal(k1, (n_workers,))

    # Generate gradients
    gradients = create_mock_gradients(k2, params)

    # Update params slightly
    lr = 0.01
    new_params = jax.tree.map(
        lambda p, g: p - lr * g,
        params,
        gradients
    )

    return new_params, fitness_scores, gradients


# ============================================================================
# Test Functions
# ============================================================================

def test_fitness_metrics():
    """Test: Fitness metrics (mean, max, min, std) are correctly computed."""
    print("test_fitness_metrics...")

    key = jax.random.PRNGKey(42)
    collector = MetricsCollector()

    # Create test fitness scores
    fitness_scores = jnp.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7])

    metrics = collector.compute_fitness_metrics(fitness_scores)

    # Verify all expected metrics are present
    required = ['mean_fitness', 'max_fitness', 'min_fitness', 'std_fitness']
    for key in required:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (float, int)), f"{key} should be Python scalar"

    # Verify values
    assert abs(metrics['mean_fitness'] - 0.45) < 0.01, f"mean_fitness wrong: {metrics['mean_fitness']}"
    assert abs(metrics['max_fitness'] - 0.8) < 0.01, f"max_fitness wrong: {metrics['max_fitness']}"
    assert abs(metrics['min_fitness'] - 0.1) < 0.01, f"min_fitness wrong: {metrics['min_fitness']}"
    assert metrics['std_fitness'] > 0, "std_fitness should be positive"

    # Verify JSON serializable
    json_str = json.dumps(metrics)
    assert json_str is not None, "Should be JSON serializable"

    print(f"  mean={metrics['mean_fitness']:.4f}, max={metrics['max_fitness']:.4f}")
    print(f"  min={metrics['min_fitness']:.4f}, std={metrics['std_fitness']:.4f}")
    print("[PASS] test_fitness_metrics")
    return True


def test_gradient_metrics():
    """Test: Gradient norms (per layer and total) are correctly computed."""
    print("test_gradient_metrics...")

    key = jax.random.PRNGKey(123)
    collector = MetricsCollector()

    # Create mock params and gradients
    params = create_mock_params(key)
    key, grad_key = jax.random.split(key)
    gradients = create_mock_gradients(grad_key, params)

    metrics = collector.compute_gradient_metrics(gradients)

    # Verify total gradient norm is present
    assert 'gradient_norm' in metrics, "Missing gradient_norm"
    assert isinstance(metrics['gradient_norm'], float), "gradient_norm should be float"
    assert metrics['gradient_norm'] > 0, "gradient_norm should be positive"

    # Verify per-layer norms are present
    for layer_name in params.keys():
        for param_name in params[layer_name].keys():
            key_name = f'gradient_norm/{layer_name}/{param_name}'
            assert key_name in metrics, f"Missing {key_name}"
            assert metrics[key_name] >= 0, f"{key_name} should be non-negative"

    # Verify total norm is consistent with per-layer norms
    # total^2 should equal sum of per-layer^2
    per_layer_norms = [v for k, v in metrics.items() if k.startswith('gradient_norm/')]
    computed_total = np.sqrt(sum(n**2 for n in per_layer_norms))
    assert abs(computed_total - metrics['gradient_norm']) < 1e-5, \
        f"Total norm inconsistent: {computed_total} vs {metrics['gradient_norm']}"

    # Verify JSON serializable
    json_str = json.dumps(metrics)
    assert json_str is not None

    print(f"  total gradient_norm={metrics['gradient_norm']:.6f}")
    print(f"  num per-layer norms={len(per_layer_norms)}")
    print("[PASS] test_gradient_metrics")
    return True


def test_param_metrics():
    """Test: Parameter norms and statistics are correctly computed."""
    print("test_param_metrics...")

    key = jax.random.PRNGKey(456)
    collector = MetricsCollector()

    # Create mock params
    params = create_mock_params(key)

    metrics = collector.compute_param_metrics(params)

    # Verify total param norm
    assert 'param_norm' in metrics, "Missing param_norm"
    assert metrics['param_norm'] > 0, "param_norm should be positive"

    # Verify statistics
    assert 'param_mean' in metrics, "Missing param_mean"
    assert 'param_std' in metrics, "Missing param_std"
    assert 'param_min' in metrics, "Missing param_min"
    assert 'param_max' in metrics, "Missing param_max"
    assert 'param_count' in metrics, "Missing param_count"

    # Verify per-layer norms
    for layer_name in params.keys():
        for param_name in params[layer_name].keys():
            key_name = f'param_norm/{layer_name}/{param_name}'
            assert key_name in metrics, f"Missing {key_name}"

    # Verify param count matches actual count
    actual_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    assert metrics['param_count'] == actual_count, \
        f"param_count mismatch: {metrics['param_count']} vs {actual_count}"

    # Verify JSON serializable
    json_str = json.dumps(metrics)
    assert json_str is not None

    print(f"  param_norm={metrics['param_norm']:.6f}")
    print(f"  param_mean={metrics['param_mean']:.6f}, std={metrics['param_std']:.6f}")
    print(f"  param_count={metrics['param_count']}")
    print("[PASS] test_param_metrics")
    return True


def test_epoch_metrics():
    """Test: Per-epoch aggregated metrics are correctly computed."""
    print("test_epoch_metrics...")

    key = jax.random.PRNGKey(789)
    collector = MetricsCollector()

    # Simulate multiple steps within an epoch
    n_steps = 5
    params = create_mock_params(key)

    for step in range(n_steps):
        key, step_key = jax.random.split(key)
        _, fitness_scores, gradients = simulate_training_step(step_key, params)

        collector.log_step(
            epoch=0,
            step=step,
            fitness_scores=fitness_scores,
            gradients=gradients,
            params=params,
            learning_rate=0.01,
        )

    # End epoch and get aggregated metrics
    epoch_metrics = collector.end_epoch()

    # Verify epoch metrics
    assert 'epoch' in epoch_metrics, "Missing epoch"
    assert 'num_steps' in epoch_metrics, "Missing num_steps"
    assert epoch_metrics['num_steps'] == n_steps, f"num_steps wrong: {epoch_metrics['num_steps']}"

    assert 'epoch_mean_fitness' in epoch_metrics, "Missing epoch_mean_fitness"
    assert 'epoch_max_fitness' in epoch_metrics, "Missing epoch_max_fitness"
    assert 'epoch_min_fitness' in epoch_metrics, "Missing epoch_min_fitness"
    assert 'epoch_mean_gradient_norm' in epoch_metrics, "Missing epoch_mean_gradient_norm"

    # Verify aggregation is consistent
    history = collector.get_history()
    mean_fitness_values = [m['mean_fitness'] for m in history]
    expected_mean = np.mean(mean_fitness_values)
    assert abs(epoch_metrics['epoch_mean_fitness'] - expected_mean) < 1e-6, \
        "epoch_mean_fitness aggregation incorrect"

    # Verify JSON serializable
    json_str = json.dumps(epoch_metrics)
    assert json_str is not None

    print(f"  num_steps={epoch_metrics['num_steps']}")
    print(f"  epoch_mean_fitness={epoch_metrics['epoch_mean_fitness']:.6f}")
    print(f"  epoch_max_fitness={epoch_metrics['epoch_max_fitness']:.6f}")
    print("[PASS] test_epoch_metrics")
    return True


def test_json_serializable():
    """Test: All metrics can be JSON serialized (W&B compatibility)."""
    print("test_json_serializable...")

    key = jax.random.PRNGKey(111)
    collector = MetricsCollector()

    # Log several steps with all metric types
    params = create_mock_params(key)

    for epoch in range(3):
        for step in range(4):
            key, step_key = jax.random.split(key)
            params, fitness_scores, gradients = simulate_training_step(step_key, params)

            collector.log_step(
                epoch=epoch,
                step=step,
                fitness_scores=fitness_scores,
                gradients=gradients,
                params=params,
                learning_rate=0.01 * (0.99 ** epoch),
                extra_metrics={
                    'pnl': jnp.array(100.5),  # JAX scalar
                    'trades': np.int32(42),    # numpy int
                    'completion': 0.95,        # Python float
                }
            )
        collector.end_epoch()

    # Test full JSON serialization
    try:
        json_str = collector.to_json()
        assert json_str is not None
        assert len(json_str) > 0

        # Verify it can be parsed back
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 12  # 3 epochs * 4 steps

        # Verify each entry is a dict with expected keys
        for entry in parsed:
            assert isinstance(entry, dict)
            assert 'epoch' in entry
            assert 'step' in entry
            assert 'mean_fitness' in entry

            # Verify all values are JSON-compatible types
            for key, value in entry.items():
                assert isinstance(value, (int, float, str, bool, list, dict, type(None))), \
                    f"Value for {key} is not JSON-compatible: {type(value)}"
    except json.JSONDecodeError as e:
        raise AssertionError(f"JSON serialization failed: {e}")

    print(f"  serialized {len(parsed)} metric entries")
    print(f"  JSON size: {len(json_str)} bytes")
    print("[PASS] test_json_serializable")
    return True


def test_metric_history():
    """Test: Metrics are correctly tracked over multiple epochs."""
    print("test_metric_history...")

    key = jax.random.PRNGKey(222)
    collector = MetricsCollector()

    # Simulate training over multiple epochs
    n_epochs = 5
    n_steps_per_epoch = 3
    params = create_mock_params(key)

    for epoch in range(n_epochs):
        for step in range(n_steps_per_epoch):
            key, step_key = jax.random.split(key)
            params, fitness_scores, gradients = simulate_training_step(step_key, params)

            collector.log_step(
                epoch=epoch,
                step=step,
                fitness_scores=fitness_scores,
                gradients=gradients,
                learning_rate=0.01,
            )
        collector.end_epoch()

    # Verify history length
    history = collector.get_history()
    expected_len = n_epochs * n_steps_per_epoch
    assert len(history) == expected_len, f"History length wrong: {len(history)} vs {expected_len}"

    # Verify epoch progression
    epochs = [m['epoch'] for m in history]
    for i, epoch in enumerate(epochs):
        expected_epoch = i // n_steps_per_epoch
        assert epoch == expected_epoch, f"Epoch at index {i} wrong: {epoch} vs {expected_epoch}"

    # Extract metric series
    mean_fitness_series = collector.get_metric_series('mean_fitness')
    assert len(mean_fitness_series) == expected_len, "mean_fitness series length wrong"
    assert all(isinstance(v, float) for v in mean_fitness_series), "Series should be floats"

    grad_norm_series = collector.get_metric_series('gradient_norm')
    assert len(grad_norm_series) == expected_len, "gradient_norm series length wrong"

    # Verify no missing values
    assert all(v is not None for v in mean_fitness_series), "No None values in series"
    assert all(v is not None for v in grad_norm_series), "No None values in series"

    print(f"  total entries: {len(history)}")
    print(f"  epochs: {n_epochs}, steps/epoch: {n_steps_per_epoch}")
    print(f"  mean_fitness range: [{min(mean_fitness_series):.4f}, {max(mean_fitness_series):.4f}]")
    print("[PASS] test_metric_history")
    return True


def test_wandb_format():
    """Test: Metrics are in correct format for W&B logging."""
    print("test_wandb_format...")

    key = jax.random.PRNGKey(333)
    collector = MetricsCollector()

    params = create_mock_params(key)
    key, step_key = jax.random.split(key)
    _, fitness_scores, gradients = simulate_training_step(step_key, params)

    metrics = collector.log_step(
        epoch=0,
        step=0,
        fitness_scores=fitness_scores,
        gradients=gradients,
        params=params,
        learning_rate=0.01,
    )

    # W&B expects flat dict with string keys and numeric/string values
    for key, value in metrics.items():
        # Keys should be strings
        assert isinstance(key, str), f"Key should be string: {key}"

        # Values should be numeric, string, or list
        assert isinstance(value, (int, float, str, list)), \
            f"Value for {key} has wrong type: {type(value)}"

        # Numeric values should be finite
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"Value for {key} is not finite: {value}"

    # Verify key naming convention (W&B uses / for hierarchical metrics)
    hierarchical_keys = [k for k in metrics.keys() if '/' in k]
    assert len(hierarchical_keys) > 0, "Should have hierarchical keys for W&B"

    # Verify essential metrics for monitoring
    essential = ['mean_fitness', 'max_fitness', 'gradient_norm', 'epoch', 'step']
    for key in essential:
        assert key in metrics, f"Missing essential metric: {key}"

    print(f"  total metrics: {len(metrics)}")
    print(f"  hierarchical metrics: {len(hierarchical_keys)}")
    print(f"  essential metrics present: {essential}")
    print("[PASS] test_wandb_format")
    return True


def test_learning_rate_tracking():
    """Test: Learning rate is correctly tracked over training."""
    print("test_learning_rate_tracking...")

    key = jax.random.PRNGKey(444)
    collector = MetricsCollector()

    params = create_mock_params(key)

    # Simulate training with decaying learning rate
    initial_lr = 0.1
    decay_rate = 0.9
    n_steps = 10

    for step in range(n_steps):
        key, step_key = jax.random.split(key)
        _, fitness_scores, _ = simulate_training_step(step_key, params)

        # Decaying learning rate
        current_lr = initial_lr * (decay_rate ** step)

        collector.log_step(
            epoch=0,
            step=step,
            fitness_scores=fitness_scores,
            learning_rate=current_lr,
        )

    # Verify learning rate series
    lr_series = collector.get_metric_series('learning_rate')
    assert len(lr_series) == n_steps, "LR series length wrong"

    # Verify decay pattern
    for i in range(1, len(lr_series)):
        assert lr_series[i] < lr_series[i-1], "LR should be decaying"

    # Verify first and last values
    assert abs(lr_series[0] - initial_lr) < 1e-6, "Initial LR wrong"
    expected_final = initial_lr * (decay_rate ** (n_steps - 1))
    assert abs(lr_series[-1] - expected_final) < 1e-6, "Final LR wrong"

    print(f"  initial_lr={lr_series[0]:.6f}")
    print(f"  final_lr={lr_series[-1]:.6f}")
    print(f"  decay_rate={decay_rate}")
    print("[PASS] test_learning_rate_tracking")
    return True


def test_full_training_simulation():
    """Test: Complete training simulation with all metrics."""
    print("test_full_training_simulation...")

    key = jax.random.PRNGKey(555)
    collector = MetricsCollector()

    # Training configuration
    n_epochs = 3
    n_steps_per_epoch = 5
    n_workers = 16
    initial_lr = 0.01

    params = create_mock_params(key)

    all_epoch_metrics = []

    for epoch in range(n_epochs):
        for step in range(n_steps_per_epoch):
            key, step_key = jax.random.split(key)
            params, fitness_scores, gradients = simulate_training_step(
                step_key, params, n_workers=n_workers
            )

            # Decaying learning rate
            lr = initial_lr * (0.99 ** (epoch * n_steps_per_epoch + step))

            metrics = collector.log_step(
                epoch=epoch,
                step=step,
                fitness_scores=fitness_scores,
                gradients=gradients,
                params=params,
                learning_rate=lr,
                extra_metrics={
                    'n_workers': n_workers,
                }
            )

        epoch_summary = collector.end_epoch()
        all_epoch_metrics.append(epoch_summary)

    # Verify full history
    history = collector.get_history()
    total_steps = n_epochs * n_steps_per_epoch
    assert len(history) == total_steps, f"History length: {len(history)} vs {total_steps}"

    # Verify all metrics are JSON serializable
    json_str = collector.to_json()
    parsed = json.loads(json_str)
    assert len(parsed) == total_steps

    # Verify epoch summaries
    assert len(all_epoch_metrics) == n_epochs
    for summary in all_epoch_metrics:
        assert 'epoch_mean_fitness' in summary
        assert 'epoch_mean_gradient_norm' in summary

    # Verify we can extract any metric series
    fitness_series = collector.get_metric_series('mean_fitness')
    grad_series = collector.get_metric_series('gradient_norm')
    lr_series = collector.get_metric_series('learning_rate')

    assert len(fitness_series) == total_steps
    assert len(grad_series) == total_steps
    assert len(lr_series) == total_steps

    print(f"  epochs={n_epochs}, steps/epoch={n_steps_per_epoch}")
    print(f"  total_steps={total_steps}")
    print(f"  final mean_fitness={fitness_series[-1]:.6f}")
    print(f"  final gradient_norm={grad_series[-1]:.6f}")
    print(f"  final lr={lr_series[-1]:.6f}")
    print("[PASS] test_full_training_simulation")
    return True


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    """Run all production metrics tests."""
    print("=" * 60)
    print("Level 4 Production Test: Metrics and Logging")
    print("=" * 60)

    tests = [
        test_fitness_metrics,
        test_gradient_metrics,
        test_param_metrics,
        test_epoch_metrics,
        test_json_serializable,
        test_metric_history,
        test_wandb_format,
        test_learning_rate_tracking,
        test_full_training_simulation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
