"""Equivalence test for the fused Pallas S5 kernel drop-in (`s5/pallas_ssm.py`).

Builds the REAL flax `S5SSM` module with the real HiPPO init, wraps it in an `nn.vmap` exactly
the way `lob/lob_seq_model.py` does, and checks that flipping `LOBS5_PALLAS_SSM` changes
neither the forward output nor ANY parameter gradient. Runs on CPU: off-TPU the kernel falls
back to Pallas' interpret mode, so this is a pure-logic check -- the Mosaic-compile check has
to happen on a real TPU.

    LOBS5/.venv/bin/python -m pytest tests/test_pallas_ssm.py -q
"""
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from flax import linen as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5 import pallas_ssm  # noqa: E402
from s5.ssm import init_S5SSM  # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO  # noqa: E402

H, P_FULL, L, BSZ = 16, 16, 48, 3          # P_FULL//2 complex modes under conj_sym
BLOCKS = 2


def _make_layer():
    """ Build the S5SSM exactly as lob/init_train.py does (conj_sym, clip_eigs, zoh).
        Returns:
            ssm_init_fn (callable): partial(S5SSM, ...) ready to instantiate
    """
    block_size = P_FULL // BLOCKS
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
    block_size = block_size // 2                     # conj_sym halves the state
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vinv = V.conj().T
    Lambda = (Lambda * jnp.ones((BLOCKS, block_size))).ravel()
    V = jax.scipy.linalg.block_diag(*([V] * BLOCKS))
    Vinv = jax.scipy.linalg.block_diag(*([Vinv] * BLOCKS))
    return init_S5SSM(H=H, P=Lambda.shape[0], Lambda_re_init=Lambda.real,
                      Lambda_im_init=Lambda.imag, V=V, Vinv=Vinv,
                      C_init="trunc_standard_normal", discretization="zoh",
                      dt_min=0.001, dt_max=0.1, conj_sym=True, clip_eigs=True,
                      bidirectional=False)


class _Batched(nn.Module):
    """`nn.vmap` over the batch, matching lob/lob_seq_model.py's BatchPaddedLobPredModel."""
    ssm: callable

    @nn.compact
    def __call__(self, x):                            # x (b, L, H)
        return nn.vmap(lambda mdl, xx: self.ssm(name="ssm")(xx),
                       variable_axes={"params": None}, split_rngs={"params": False},
                       in_axes=0, out_axes=0)(self, x)


def _fwd_and_grads(use_pallas, params, x, ct, model):
    """ Run the model forward and backward with the kernel on or off.

        grad_u matters as much as the parameter grads: in the stacked model every
        earlier layer sees only grad_u, so a kernel returning a zero/wrong grad_u
        would train the deepest layer correctly and silently starve everything below.
        Args:
            use_pallas (bool): select the kernel or the stock associative_scan
            params: flax parameter pytree
            x (float32): input batch                                    (b, L, H)
            ct (float32): output cotangent                              (b, L, H)
            model: the nn.vmap-wrapped module
        Returns:
            y (float32) (b, L, H), grads wrt params, grad wrt x (float32) (b, L, H)
    """
    os.environ["LOBS5_PALLAS_SSM"] = "1" if use_pallas else "0"
    assert pallas_ssm.use_pallas_ssm() is use_pallas
    loss = lambda p, xx: jnp.sum(model.apply(p, xx) * ct)
    y = model.apply(params, x)
    g_params, g_x = jax.grad(loss, argnums=(0, 1))(params, x)
    return y, g_params, g_x


@pytest.fixture(scope="module")
def _fixture():
    prev = {k: os.environ.get(k) for k in
            ("LOBS5_PALLAS_SSM", "LOBS5_PALLAS_CHUNK", "LOBS5_PALLAS_FUSED_BWD")}
    os.environ["LOBS5_PALLAS_SSM"] = "0"
    # Force MULTIPLE chunks: at the default cap of 240 an L=48 sequence is a single chunk, so
    # the cross-chunk carry (a VMEM scratch threaded through the grid -- the part most likely
    # to be wrong) would never run. L=48 / chunk=16 -> 3 chunks.
    os.environ["LOBS5_PALLAS_CHUNK"] = "16"
    assert pallas_ssm._pick_chunk(L, pallas_ssm.config().max_chunk) == 16 and L // 16 >= 3
    model = _Batched(ssm=_make_layer())
    x = jax.random.normal(jax.random.PRNGKey(0), (BSZ, L, H), jnp.float32)
    params = model.init(jax.random.PRNGKey(1), x)
    ct = jax.random.normal(jax.random.PRNGKey(2), (BSZ, L, H), jnp.float32)
    yield model, params, x, ct
    for k, v in prev.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _rel(a, b):
    """ Energy-relative difference. A flat allclose would let a small-magnitude
        gradient (e.g. grad_D) hide a real bias behind a large-magnitude one.
        Args:
            a: array under test
            b: reference array
        Returns:
            rel (float): RMS(a-b) / RMS(b)
    """
    a, b = onp.asarray(a), onp.asarray(b)
    den = onp.sqrt((onp.abs(b) ** 2).mean()) + 1e-12
    return float(onp.sqrt((onp.abs(a - b) ** 2).mean()) / den)


def test_forward_matches_associative_scan(_fixture):
    model, params, x, ct = _fixture
    y_ref, _, _ = _fwd_and_grads(False, params, x, ct, model)
    y_ker, _, _ = _fwd_and_grads(True, params, x, ct, model)
    assert y_ker.shape == y_ref.shape == (BSZ, L, H)
    assert _rel(y_ker, y_ref) < 1e-3, f"forward rel err {_rel(y_ker, y_ref)}"


def test_every_param_gradient_matches(_fixture):
    model, params, x, ct = _fixture
    _, g_ref, _ = _fwd_and_grads(False, params, x, ct, model)
    _, g_ker, _ = _fwd_and_grads(True, params, x, ct, model)
    flat_ref = jax.tree_util.tree_flatten_with_path(g_ref)[0]
    flat_ker = jax.tree_util.tree_flatten_with_path(g_ker)[0]
    assert [k for k, _ in flat_ref] == [k for k, _ in flat_ker]
    assert len(flat_ref) >= 5, f"expected >=5 SSM params, got {[k for k, _ in flat_ref]}"
    for (path, r), (_, k) in zip(flat_ref, flat_ker):
        name = jax.tree_util.keystr(path)
        assert _rel(k, r) < 1e-2, f"grad {name}: rel err {_rel(k, r)}"


def test_input_gradient_matches(_fixture):
    """grad_u -- what every layer BELOW an S5 layer receives. The param-gradient test above
    passes even if grad_u is identically zero, which in the stacked model would silently kill
    training for everything upstream."""
    model, params, x, ct = _fixture
    _, _, gx_ref = _fwd_and_grads(False, params, x, ct, model)
    _, _, gx_ker = _fwd_and_grads(True, params, x, ct, model)
    assert gx_ker.shape == gx_ref.shape == (BSZ, L, H)
    assert float(onp.abs(onp.asarray(gx_ref)).max()) > 0, "reference grad_u is degenerate"
    assert _rel(gx_ker, gx_ref) < 1e-2, f"grad_u rel err {_rel(gx_ker, gx_ref)}"


@pytest.mark.parametrize("fused_bwd", ["1", "0"])
def test_both_backward_backends_match(_fixture, fused_bwd):
    """--pallas_fused_bwd is user-facing, so BOTH settings must produce correct gradients --
    the fused reverse-grid kernel and the pure-XLA analytic adjoint fallback."""
    model, params, x, ct = _fixture
    prev = os.environ.get("LOBS5_PALLAS_FUSED_BWD")
    try:
        _, g_ref, gx_ref = _fwd_and_grads(False, params, x, ct, model)
        os.environ["LOBS5_PALLAS_FUSED_BWD"] = fused_bwd
        _, g_ker, gx_ker = _fwd_and_grads(True, params, x, ct, model)
        assert _rel(gx_ker, gx_ref) < 1e-2, f"grad_u rel err {_rel(gx_ker, gx_ref)}"
        for (path, r), (_, k) in zip(jax.tree_util.tree_flatten_with_path(g_ref)[0],
                                     jax.tree_util.tree_flatten_with_path(g_ker)[0]):
            assert _rel(k, r) < 1e-2, \
                f"fused_bwd={fused_bwd} grad {jax.tree_util.keystr(path)}: rel {_rel(k, r)}"
    finally:
        os.environ.pop("LOBS5_PALLAS_FUSED_BWD", None) if prev is None \
            else os.environ.__setitem__("LOBS5_PALLAS_FUSED_BWD", prev)


@pytest.mark.parametrize("var,bad", [("LOBS5_PALLAS_CHUNK", "4"),
                                     ("LOBS5_PALLAS_CHUNK", "notanint"),
                                     ("LOBS5_PALLAS_VMEM_MB", "0")])
def test_config_rejects_bad_values(var, bad):
    """A typo in a knob must fail loudly, not silently fall back to the default."""
    prev = os.environ.get(var)
    try:
        os.environ[var] = bad
        with pytest.raises(ValueError):
            pallas_ssm.config()
    finally:
        os.environ.pop(var, None) if prev is None else os.environ.__setitem__(var, prev)


def test_fused_backward_kernel_actually_ran(_fixture):
    """With LOBS5_PALLAS_FUSED_BWD on (the default) the BACKWARD must also be a pallas_call.
    The forward-jaxpr check below cannot see that -- it would pass just as happily with the
    pure-XLA adjoint, which is the slow path we deliberately turned off."""
    model, params, x, ct = _fixture
    os.environ["LOBS5_PALLAS_SSM"] = "1"
    grad_fn = jax.grad(lambda p: jnp.sum(model.apply(p, x) * ct))
    # Counting "pallas_call" in the printed jaxpr does NOT work here: custom_vjp_call prints
    # both its primal and its fwd jaxpr, so the forward kernel alone shows up twice. Count
    # actual invocations of the backward kernel builder instead.
    core = pallas_ssm._core
    prev_flag, prev_fn = os.environ.get("LOBS5_PALLAS_FUSED_BWD"), core._fused_backward
    calls = []

    def counting(*a, **kw):
        calls.append(1)
        return prev_fn(*a, **kw)

    try:
        core._fused_backward = counting
        os.environ["LOBS5_PALLAS_FUSED_BWD"] = "1"
        jax.block_until_ready(grad_fn(params))
        assert calls, "LOBS5_PALLAS_FUSED_BWD=1 did not invoke the fused backward kernel"
        calls.clear()
        os.environ["LOBS5_PALLAS_FUSED_BWD"] = "0"
        jax.block_until_ready(grad_fn(params))
        assert not calls, "the XLA-adjoint path invoked the fused backward kernel anyway"
    finally:
        core._fused_backward = prev_fn
        os.environ.pop("LOBS5_PALLAS_FUSED_BWD", None) if prev_flag is None \
            else os.environ.__setitem__("LOBS5_PALLAS_FUSED_BWD", prev_flag)


def test_pallas_path_actually_ran(_fixture):
    """Guard against a silently-disabled kernel making the equivalence tests vacuous."""
    model, params, x, _ = _fixture
    os.environ["LOBS5_PALLAS_SSM"] = "1"
    # pallas_call sits inside the custom_vjp_call sub-jaxpr, so search the printed form.
    text = str(jax.make_jaxpr(lambda p: model.apply(p, x))(params))
    assert "pallas_call" in text, "kernel path did not emit a pallas_call"
    os.environ["LOBS5_PALLAS_SSM"] = "0"
    assert "pallas_call" not in str(jax.make_jaxpr(lambda p: model.apply(p, x))(params))
