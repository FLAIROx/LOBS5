"""Test CommonInit and CommonParams structures."""
import sys

sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

from es_lobs5.models.common import CommonInit, CommonParams


def test_common_init_fields():
    """Test: CommonInit has frozen_params, params, scan_map, es_map fields."""
    # Verify CommonInit is a NamedTuple with expected fields
    expected_fields = ('frozen_params', 'params', 'scan_map', 'es_map')

    assert hasattr(CommonInit, '_fields'), "CommonInit should be a NamedTuple"
    assert CommonInit._fields == expected_fields, (
        f"Expected fields {expected_fields}, got {CommonInit._fields}"
    )

    # Test instantiation with positional args
    init = CommonInit(
        frozen_params={'layer1': 'frozen'},
        params={'layer1': 'trainable'},
        scan_map=(0, 1),
        es_map='PARAM'
    )

    assert init.frozen_params == {'layer1': 'frozen'}
    assert init.params == {'layer1': 'trainable'}
    assert init.scan_map == (0, 1)
    assert init.es_map == 'PARAM'

    # Test tuple unpacking
    fp, p, sm, em = init
    assert fp == init.frozen_params
    assert p == init.params
    assert sm == init.scan_map
    assert em == init.es_map

    print("[PASS] test_common_init_fields")
    return True


def test_common_params_fields():
    """Test: CommonParams has frozen_params, params, noiser related fields."""
    # Verify CommonParams is a NamedTuple with expected fields
    expected_fields = (
        'noiser', 'frozen_noiser_params', 'noiser_params',
        'frozen_params', 'params', 'es_tree_key', 'iterinfo'
    )

    assert hasattr(CommonParams, '_fields'), "CommonParams should be a NamedTuple"
    assert CommonParams._fields == expected_fields, (
        f"Expected fields {expected_fields}, got {CommonParams._fields}"
    )

    # Test instantiation with keyword args
    params = CommonParams(
        noiser='mock_noiser',
        frozen_noiser_params={'sigma': 0.1},
        noiser_params={'lr': 0.001},
        frozen_params={'layer1': 'frozen'},
        params={'layer1': 'trainable'},
        es_tree_key='mock_key',
        iterinfo=(0, 0)
    )

    assert params.noiser == 'mock_noiser'
    assert params.frozen_noiser_params == {'sigma': 0.1}
    assert params.noiser_params == {'lr': 0.001}
    assert params.frozen_params == {'layer1': 'frozen'}
    assert params.params == {'layer1': 'trainable'}
    assert params.es_tree_key == 'mock_key'
    assert params.iterinfo == (0, 0)

    # Test tuple unpacking (7 fields)
    n, fnp, np, fp, p, etk, ii = params
    assert n == params.noiser
    assert fnp == params.frozen_noiser_params
    assert np == params.noiser_params
    assert fp == params.frozen_params
    assert p == params.params
    assert etk == params.es_tree_key
    assert ii == params.iterinfo

    print("[PASS] test_common_params_fields")
    return True


if __name__ == "__main__":
    test_common_init_fields()
    test_common_params_fields()
    print("All CommonInit/CommonParams tests passed!")
