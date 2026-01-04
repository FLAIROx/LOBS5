"""Test Model base class interface."""
import sys
import inspect

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.models.common import (
    Model, CommonInit, CommonParams,
    ES_Parameter, ES_MM, ES_Linear, ES_LayerNorm
)


def test_model_has_rand_init():
    """Test: Model subclasses implement rand_init method."""
    # Verify base Model class has rand_init as classmethod
    assert hasattr(Model, 'rand_init'), "Model should have rand_init method"
    assert isinstance(inspect.getattr_static(Model, 'rand_init'), classmethod), (
        "rand_init should be a classmethod"
    )

    # Verify subclasses implement rand_init (not just inherit NotImplementedError)
    subclasses = [ES_Parameter, ES_MM, ES_Linear, ES_LayerNorm]

    for cls in subclasses:
        assert hasattr(cls, 'rand_init'), f"{cls.__name__} should have rand_init method"

        # Check that subclass actually overrides rand_init (not using base Model's)
        # Base Model.rand_init raises NotImplementedError
        cls_rand_init = getattr(cls, 'rand_init')
        base_rand_init = getattr(Model, 'rand_init')

        # They should be different methods (subclass overrides)
        assert cls_rand_init.__func__ is not base_rand_init.__func__, (
            f"{cls.__name__}.rand_init should override Model.rand_init"
        )

    print("[PASS] test_model_has_rand_init")
    return True


def test_model_has_forward():
    """Test: Model subclasses implement _forward method."""
    # Verify base Model class has _forward as classmethod
    assert hasattr(Model, '_forward'), "Model should have _forward method"
    assert isinstance(inspect.getattr_static(Model, '_forward'), classmethod), (
        "_forward should be a classmethod"
    )

    # Verify base Model class has forward wrapper as classmethod
    assert hasattr(Model, 'forward'), "Model should have forward method"
    assert isinstance(inspect.getattr_static(Model, 'forward'), classmethod), (
        "forward should be a classmethod"
    )

    # Verify subclasses implement _forward (not just inherit NotImplementedError)
    subclasses = [ES_Parameter, ES_MM, ES_Linear, ES_LayerNorm]

    for cls in subclasses:
        assert hasattr(cls, '_forward'), f"{cls.__name__} should have _forward method"

        # Check that subclass actually overrides _forward (not using base Model's)
        # Base Model._forward raises NotImplementedError
        cls_forward = getattr(cls, '_forward')
        base_forward = getattr(Model, '_forward')

        # They should be different methods (subclass overrides)
        assert cls_forward.__func__ is not base_forward.__func__, (
            f"{cls.__name__}._forward should override Model._forward"
        )

    print("[PASS] test_model_has_forward")
    return True


def test_model_returns_common_init():
    """Test: rand_init returns CommonInit."""
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(42)

    # Test ES_Parameter.rand_init returns CommonInit
    param_init = ES_Parameter.rand_init(key, shape=(10,), scale=0.1, dtype=jnp.float32)
    assert isinstance(param_init, CommonInit), (
        f"ES_Parameter.rand_init should return CommonInit, got {type(param_init)}"
    )
    assert param_init.params is not None, "CommonInit.params should not be None"
    assert param_init.scan_map == (), "ES_Parameter should have empty scan_map"

    # Test ES_MM.rand_init returns CommonInit
    key = jax.random.PRNGKey(43)
    mm_init = ES_MM.rand_init(key, 32, 64, jnp.float32)
    assert isinstance(mm_init, CommonInit), (
        f"ES_MM.rand_init should return CommonInit, got {type(mm_init)}"
    )
    assert mm_init.params is not None, "CommonInit.params should not be None"
    assert mm_init.params.shape == (64, 32), (
        f"ES_MM params shape should be (64, 32), got {mm_init.params.shape}"
    )

    # Test ES_Linear.rand_init returns CommonInit
    key = jax.random.PRNGKey(44)
    linear_init = ES_Linear.rand_init(key, 32, 64, use_bias=True, dtype=jnp.float32)
    assert isinstance(linear_init, CommonInit), (
        f"ES_Linear.rand_init should return CommonInit, got {type(linear_init)}"
    )
    assert linear_init.params is not None, "CommonInit.params should not be None"
    assert 'weight' in linear_init.params, "ES_Linear should have 'weight' in params"
    assert 'bias' in linear_init.params, "ES_Linear with use_bias=True should have 'bias'"

    # Test ES_LayerNorm.rand_init returns CommonInit
    key = jax.random.PRNGKey(45)
    ln_init = ES_LayerNorm.rand_init(key, 64, dtype=jnp.float32, use_bias=True)
    assert isinstance(ln_init, CommonInit), (
        f"ES_LayerNorm.rand_init should return CommonInit, got {type(ln_init)}"
    )
    assert ln_init.params is not None, "CommonInit.params should not be None"
    assert 'weight' in ln_init.params, "ES_LayerNorm should have 'weight' in params"
    assert 'bias' in ln_init.params, "ES_LayerNorm with use_bias=True should have 'bias'"

    # Verify CommonInit unpacking works correctly
    frozen_params, params, scan_map, es_map = param_init
    assert frozen_params is None or isinstance(frozen_params, (dict, type(None)))
    assert params is not None
    assert isinstance(scan_map, tuple)

    print("[PASS] test_model_returns_common_init")
    return True


if __name__ == "__main__":
    test_model_has_rand_init()
    test_model_has_forward()
    test_model_returns_common_init()
    print("All Model base class interface tests passed!")
