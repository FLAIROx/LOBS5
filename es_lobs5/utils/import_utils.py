"""
Shared import utilities to avoid gymnax dependency.

HyperscaleES's __init__.py imports rl.py which requires gymnax.
We use importlib to load specific modules directly without triggering the full package import.
"""

import importlib.util
import sys
import os

# Path to HyperscaleES source
_hyperscalees_path = os.path.join(os.path.dirname(__file__), '../../HyperscaleES/src')


def get_hyperscalees_path():
    """Return the path to HyperscaleES source directory."""
    return _hyperscalees_path


def load_module(module_name: str, file_path: str):
    """
    Load a Python module directly from file without triggering __init__.py.

    This allows importing specific modules from HyperscaleES without
    loading the full package (which would fail due to gymnax dependency).

    Args:
        module_name: Name to register the module under in sys.modules
        file_path: Absolute path to the .py file

    Returns:
        The loaded module
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_hyperscalees_path():
    """Ensure HyperscaleES source is in sys.path."""
    if _hyperscalees_path not in sys.path:
        sys.path.insert(0, _hyperscalees_path)


# Pre-loaded modules cache
_loaded_modules = {}


def get_base_model():
    """Load and cache hyperscalees.models.base_model."""
    if 'base_model' not in _loaded_modules:
        _loaded_modules['base_model'] = load_module(
            'hyperscalees.models.base_model',
            os.path.join(_hyperscalees_path, 'hyperscalees/models/base_model.py')
        )
    return _loaded_modules['base_model']


def get_common():
    """Load and cache hyperscalees.models.common."""
    if 'common' not in _loaded_modules:
        _loaded_modules['common'] = load_module(
            'hyperscalees.models.common',
            os.path.join(_hyperscalees_path, 'hyperscalees/models/common.py')
        )
    return _loaded_modules['common']


def get_noiser_modules():
    """Load and cache all noiser modules."""
    if 'noisers' not in _loaded_modules:
        ensure_hyperscalees_path()
        noisers = {
            'base_noiser': load_module(
                'hyperscalees.noiser.base_noiser',
                os.path.join(_hyperscalees_path, 'hyperscalees/noiser/base_noiser.py')
            ),
            'open_es': load_module(
                'hyperscalees.noiser.open_es',
                os.path.join(_hyperscalees_path, 'hyperscalees/noiser/open_es.py')
            ),
            'eggroll': load_module(
                'hyperscalees.noiser.eggroll',
                os.path.join(_hyperscalees_path, 'hyperscalees/noiser/eggroll.py')
            ),
            'eggroll_bs': load_module(
                'hyperscalees.noiser.eggroll_baseline_subtraction',
                os.path.join(_hyperscalees_path, 'hyperscalees/noiser/eggroll_baseline_subtraction.py')
            ),
            'sparse': load_module(
                'hyperscalees.noiser.sparse',
                os.path.join(_hyperscalees_path, 'hyperscalees/noiser/sparse.py')
            ),
        }
        _loaded_modules['noisers'] = noisers
    return _loaded_modules['noisers']


def get_all_noisers():
    """Return dict of all available noisers."""
    noisers = get_noiser_modules()
    return {
        "noop": noisers['base_noiser'].Noiser,
        "open_es": noisers['open_es'].OpenES,
        "eggroll": noisers['eggroll'].EggRoll,
        "eggrollbs": noisers['eggroll_bs'].EggRollBS,
        "reeggroll": noisers['eggroll'].EggRoll,
        "sparse": noisers['sparse'].Sparse,
    }
