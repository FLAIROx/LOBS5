"""
SSM Adapter: Wraps s5/ssm.py for ES training.

This adapter provides ES-compatible S5 SSM parameters and forward functions
by directly reusing the core SSM functions from s5/ssm.py.

Key features:
- No code duplication: imports apply_ssm, discretize_zoh, etc. from s5/ssm.py
- ES parameter classification (PARAM vs EXCLUDED for stability)
- Noiser injection for ES gradient estimation

Usage:
    from es_lobs5.adapters.ssm_adapter import S5SSMParams
    from es_lobs5.adapters.hippo_adapter import get_hippo_params

    # Get HiPPO initialization
    hippo = get_hippo_params(ssm_size=256, blocks=8)

    # Initialize SSM params
    init = S5SSMParams.rand_init(
        key=key, H=512, P=128,
        Lambda_re_init=hippo['Lambda_re_init'],
        Lambda_im_init=hippo['Lambda_im_init'],
        V=hippo['V'], Vinv=hippo['Vinv'],
    )

    # Forward pass
    output = S5SSMParams.forward(noiser, ..., x)
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal

# Import HyperscaleES base classes via import_utils (avoids gymnax dependency)
from ..utils.import_utils import get_base_model, get_common, load_module
import os

_base_model = get_base_model()
_common = get_common()

Model = _base_model.Model
CommonInit = _base_model.CommonInit
CommonParams = _base_model.CommonParams
PARAM = _common.PARAM
EXCLUDED = _common.EXCLUDED

# Import core SSM functions from s5/ssm.py via importlib (avoids flax dependency)
# s5/ssm.py has `from flax import linen as nn` at top level which fails without flax.
# However, the core functions (apply_ssm, discretize_zoh, etc.) don't use flax.
# We use exec to extract just the functions we need.
_s5_path = os.path.join(os.path.dirname(__file__), '../../s5')


def _load_ssm_functions():
    """Load SSM functions from s5/ssm.py without triggering flax import."""
    ssm_file = os.path.join(_s5_path, 'ssm.py')
    with open(ssm_file, 'r') as f:
        ssm_code = f.read()

    # Create a namespace with required dependencies
    namespace = {
        'jax': jax,
        'np': jnp,  # ssm.py uses `import jax.numpy as np`
        'partial': __import__('functools').partial,
    }

    # Execute the code (flax import will fail but we catch it)
    # We need to skip the flax import and nn class
    lines = ssm_code.split('\n')
    filtered_lines = []
    in_class = False
    class_indent = 0

    for line in lines:
        # Skip flax import
        if 'from flax import' in line or 'import flax' in line:
            continue
        # Skip lecun_normal, normal imports (we import from jax directly)
        if 'from jax.nn.initializers import' in line:
            continue
        # Skip ssm_init imports (we import separately)
        if 'from .ssm_init import' in line:
            continue
        # Skip class definitions (S5SSM uses flax)
        if line.strip().startswith('class ') and 'nn.Module' in line:
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            continue
        # Skip class body
        if in_class:
            if line.strip() == '' or (len(line) - len(line.lstrip()) > class_indent):
                continue
            elif line.strip().startswith('def ') or line.strip().startswith('class ') or \
                 (len(line) - len(line.lstrip()) <= class_indent and line.strip() != ''):
                in_class = False
            else:
                continue

        filtered_lines.append(line)

    filtered_code = '\n'.join(filtered_lines)

    # Execute filtered code
    exec(filtered_code, namespace)

    return namespace


# Load SSM functions
_ssm_namespace = _load_ssm_functions()

# Extract functions we need
discretize_bilinear = _ssm_namespace['discretize_bilinear']
discretize_zoh = _ssm_namespace['discretize_zoh']
binary_operator = _ssm_namespace['binary_operator']
binary_operator_reset = _ssm_namespace['binary_operator_reset']
apply_ssm = _ssm_namespace['apply_ssm']
apply_ssm_rnn = _ssm_namespace['apply_ssm_rnn']

# Import initialization helpers from s5/ssm_init.py (no flax dependency)
from s5.ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal

__all__ = [
    'S5SSMParams',
    # Re-export for convenience
    'apply_ssm',
    'apply_ssm_rnn',
    'discretize_zoh',
    'discretize_bilinear',
]


class S5SSMParams(Model):
    """
    ES-compatible S5 State Space Model.

    Parameters are classified for ES as follows:
    - Lambda_re, Lambda_im: EXCLUDED (stability critical - eigenvalues)
    - log_step: EXCLUDED (stability critical - timescale)
    - B: PARAM (safe to perturb - input projection)
    - C: PARAM (safe to perturb - output projection)
    - D: PARAM (simple feedthrough)

    Complex parameters (B, C) are stored as (..., 2) real tensors for (real, imag).
    """

    @classmethod
    def rand_init(
        cls,
        key,
        H: int,
        P: int,
        Lambda_re_init: jnp.ndarray,
        Lambda_im_init: jnp.ndarray,
        V: jnp.ndarray,
        Vinv: jnp.ndarray,
        C_init: str = 'trunc_standard_normal',
        discretization: str = 'zoh',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conj_sym: bool = True,
        clip_eigs: bool = True,
        bidirectional: bool = False,
        step_rescale: float = 1.0,
        dtype=jnp.float32,
    ) -> CommonInit:
        """
        Initialize S5 SSM parameters for ES training.

        Args:
            key: JAX random key
            H: Feature dimension (hidden size / model dim)
            P: State space dimension (after conj_sym: P = ssm_size // 2 if conj_sym else ssm_size)
            Lambda_re_init: Initial eigenvalue real parts from HiPPO (P,)
            Lambda_im_init: Initial eigenvalue imaginary parts from HiPPO (P,)
            V: Eigenvector matrix from HiPPO
            Vinv: Inverse eigenvector matrix from HiPPO
            C_init: C matrix initialization method ('trunc_standard_normal', 'lecun_normal', 'complex_normal')
            discretization: Discretization method ('zoh' or 'bilinear')
            dt_min, dt_max: Timescale range for log_step initialization
            conj_sym: Whether conjugate symmetry is enforced
            clip_eigs: Whether to clip eigenvalues to left-half plane (recommended for autoreg)
            bidirectional: Whether to use bidirectional processing
            step_rescale: Timescale scaling factor
            dtype: Parameter data type

        Returns:
            CommonInit with frozen_params, params, scan_map, es_map
        """
        keys = jax.random.split(key, 4)

        # With conj_sym, local_P is double because we expand during forward
        local_P = 2 * P if conj_sym else P

        # Initialize B (input matrix) using s5's init_VinvB
        # Shape: (local_P, H, 2) where last dim is (real, imag)
        B = init_VinvB(lecun_normal(), keys[0], (local_P, H), Vinv).astype(dtype)

        # Initialize C (output matrix) using s5's init_CV
        if bidirectional:
            C_shape = (H, 2 * local_P, 2)
        else:
            C_shape = (H, local_P, 2)

        if C_init == 'trunc_standard_normal':
            C = init_CV(trunc_standard_normal, keys[1], C_shape, V).astype(dtype)
        elif C_init == 'lecun_normal':
            C = init_CV(lecun_normal(), keys[1], C_shape, V).astype(dtype)
        elif C_init == 'complex_normal':
            # Direct complex normal - no V multiplication
            from jax.nn.initializers import normal
            C_init_fn = normal(stddev=0.5 ** 0.5)
            C = C_init_fn(keys[1], C_shape).astype(dtype)
        else:
            raise ValueError(f"Unknown C_init method: {C_init}")

        # Initialize D (feedthrough) - simple scalar per feature
        D = (jax.random.normal(keys[2], (H,)) * 1.0).astype(dtype)

        # Initialize log_step using s5's init_log_steps
        # Shape: (P, 1) where P is state dimension
        log_step = init_log_steps(keys[3], (P, dt_min, dt_max)).astype(dtype)

        # Build params dict
        params = {
            'Lambda_re': Lambda_re_init.astype(dtype),
            'Lambda_im': Lambda_im_init.astype(dtype),
            'B': B,
            'C': C,
            'D': D,
            'log_step': log_step,
        }

        # ES map: which params get which perturbation type
        # EXCLUDED: stability-critical params that should NOT be perturbed
        # PARAM: safe to perturb during ES
        es_map = {
            'Lambda_re': EXCLUDED,  # Eigenvalues - stability critical
            'Lambda_im': EXCLUDED,  # Eigenvalues - stability critical
            'B': PARAM,             # Input projection - safe to perturb
            'C': PARAM,             # Output projection - safe to perturb
            'D': PARAM,             # Feedthrough - safe to perturb
            'log_step': EXCLUDED,   # Timescale - stability critical
        }

        # Scan map: no vmap dimensions for these params (all are per-layer)
        scan_map = {k: () for k in params}

        # Frozen params: constants needed for forward pass (not learned)
        frozen_params = {
            'V': V,
            'Vinv': Vinv,
            'H': H,
            'P': P,
            'conj_sym': conj_sym,
            'clip_eigs': clip_eigs,
            'bidirectional': bidirectional,
            'discretization': discretization,
            'step_rescale': step_rescale,
        }

        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params: CommonParams, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute S5 SSM forward pass.

        Pipeline:
        1. Get (potentially noisy) parameters via noiser
        2. Reconstruct complex Lambda, B_tilde, C_tilde
        3. Discretize (ZOH or bilinear)
        4. Apply SSM via parallel scan (from s5/ssm.py)
        5. Add feedthrough D*x

        Args:
            common_params: CommonParams with noiser, params, iterinfo etc.
            x: Input sequence (L, H) float

        Returns:
            Output sequence (L, H) float
        """
        fp = common_params.frozen_params
        noiser = common_params.noiser

        # Helper to get (potentially noisy) parameters
        def get_param(name):
            """Apply noise to param if iterinfo is provided (ES training)."""
            param = common_params.params[name]
            es_key = common_params.es_tree_key[name]
            es_map_val = common_params.es_map[name]
            if common_params.iterinfo is None:
                # Inference mode - no noise
                return param
            return noiser.get_noisy_standard(
                common_params.frozen_noiser_params,
                common_params.noiser_params,
                param, es_key, es_map_val, common_params.iterinfo
            )

        # Get parameters (with potential ES noise)
        Lambda_re = get_param('Lambda_re')
        Lambda_im = get_param('Lambda_im')
        B = get_param('B')
        C = get_param('C')
        D = get_param('D')
        log_step = get_param('log_step')

        # Reconstruct complex eigenvalues
        Lambda = Lambda_re + 1j * Lambda_im
        if fp['clip_eigs']:
            # Enforce left-half plane: Re(Lambda) < 0 for stability
            Lambda = jnp.clip(Lambda.real, None, -1e-4) + 1j * Lambda.imag

        # Reconstruct complex B and C from (real, imag) storage
        # B: (local_P, H, 2) -> (local_P, H) complex
        # C: (H, local_P, 2) -> (H, local_P) complex
        B_tilde = B[..., 0] + 1j * B[..., 1]
        C_tilde = C[..., 0] + 1j * C[..., 1]

        # Compute discretization step from log_step
        step = fp['step_rescale'] * jnp.exp(log_step[:, 0])

        # Discretize continuous-time SSM to discrete-time
        if fp['discretization'] == 'zoh':
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif fp['discretization'] == 'bilinear':
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise ValueError(f"Unknown discretization: {fp['discretization']}")

        # Cast input to float32 for SSM computation (numerical stability)
        input_dtype = x.dtype
        x_fp32 = x.astype(jnp.float32)

        # Apply SSM using parallel scan from s5/ssm.py
        # This is the core computation: y = C * scan(Lambda_bar, B_bar * x)
        ys = apply_ssm(
            Lambda_bar, B_bar, C_tilde, x_fp32,
            fp['conj_sym'], fp['bidirectional']
        )

        # Add feedthrough: output = y + D * x
        # D is (H,), x is (L, H), so D * x broadcasts to (L, H)
        Du = D * x_fp32
        output = ys + Du

        return output.astype(input_dtype)

    @classmethod
    def _forward_rnn(
        cls,
        common_params: CommonParams,
        hidden: jnp.ndarray,
        x: jnp.ndarray,
        resets: jnp.ndarray = None
    ) -> tuple:
        """
        Compute S5 SSM forward pass in RNN mode (step-by-step with hidden state).

        Uses apply_ssm_rnn from s5/ssm.py for the core computation.

        Args:
            common_params: CommonParams with noiser, params, etc.
            hidden: Hidden state (1, P) complex64
            x: Input sequence (L, H) float
            resets: Optional reset signals (L,) bool - resets hidden state

        Returns:
            (hidden_out, output): new hidden state (1, P) and output (L, H)
        """
        fp = common_params.frozen_params
        noiser = common_params.noiser

        # Helper to get (potentially noisy) parameters
        def get_param(name):
            param = common_params.params[name]
            es_key = common_params.es_tree_key[name]
            es_map_val = common_params.es_map[name]
            if common_params.iterinfo is None:
                return param
            return noiser.get_noisy_standard(
                common_params.frozen_noiser_params,
                common_params.noiser_params,
                param, es_key, es_map_val, common_params.iterinfo
            )

        # Get parameters
        Lambda_re = get_param('Lambda_re')
        Lambda_im = get_param('Lambda_im')
        B = get_param('B')
        C = get_param('C')
        D = get_param('D')
        log_step = get_param('log_step')

        # Reconstruct complex
        Lambda = Lambda_re + 1j * Lambda_im
        if fp['clip_eigs']:
            Lambda = jnp.clip(Lambda.real, None, -1e-4) + 1j * Lambda.imag

        B_tilde = B[..., 0] + 1j * B[..., 1]
        C_tilde = C[..., 0] + 1j * C[..., 1]

        # Discretize
        step = fp['step_rescale'] * jnp.exp(log_step[:, 0])
        if fp['discretization'] == 'zoh':
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif fp['discretization'] == 'bilinear':
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise ValueError(f"Unknown discretization: {fp['discretization']}")

        # Cast input to float32
        input_dtype = x.dtype
        x_fp32 = x.astype(jnp.float32)

        # Squeeze hidden from (batch, 1, P) to (1, P) for apply_ssm_rnn
        # apply_ssm_rnn expects 2D hidden state
        if hidden.ndim == 3:
            hidden_2d = hidden.squeeze(axis=0)  # (1, P)
        else:
            hidden_2d = hidden

        # Apply SSM RNN mode
        hidden_out_2d, ys = apply_ssm_rnn(
            Lambda_bar, B_bar, C_tilde, hidden_2d, x_fp32, resets,
            fp['conj_sym'], fp['bidirectional']
        )

        # Unsqueeze hidden back to (batch, 1, P) for consistency
        if hidden.ndim == 3:
            hidden_out = hidden_out_2d[None, :, :]  # (1, 1, P)
        else:
            hidden_out = hidden_out_2d

        # Add feedthrough
        Du = D * x_fp32
        output = ys + Du

        return hidden_out, output.astype(input_dtype)

    @classmethod
    def initialize_carry(cls, frozen_params: dict, batch_size: int = 1) -> jnp.ndarray:
        """
        Initialize hidden state (carry) for RNN mode.

        Args:
            frozen_params: Frozen params from rand_init containing P, conj_sym
            batch_size: Batch size (default 1)

        Returns:
            Initial hidden state (batch_size, 1, P) complex64
        """
        P = frozen_params['P']
        conj_sym = frozen_params['conj_sym']

        # local_P is the actual state dimension used
        local_P = 2 * P if conj_sym else P

        # Hidden state is complex-valued
        return jnp.zeros((batch_size, 1, local_P), dtype=jnp.complex64)
