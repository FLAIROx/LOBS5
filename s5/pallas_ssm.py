"""Drop-in Pallas-TPU replacement for the body of `s5.ssm.S5SSM.__call__`.

Upstream computes, per sequence (no batch dim -- the whole model is `nn.vmap`ped over batch,
see `lob/lob_seq_model.py:BatchPaddedLobPredModel`):

    ys = apply_ssm(Lambda_bar, B_bar, C_tilde, u, conj_sym=True, bidirectional=False)
    y  = ys + D * u

via `jax.lax.associative_scan`, which materialises the (L,P) complex state sequences in HBM.
This module routes the same math through the fused Pallas kernel developed in
`kernels/isopod/experiments/s5ssm/` (one `pallas_call`, chunk grid, complex state carried in a
VMEM scratch). The kernel folds the `D * u` skip in, so it replaces the WHOLE `__call__` body,
not just `apply_ssm`.

Three things make the swap non-trivial, all handled here:

1. **Autodiff.** `jax.vjp` does not flow through the forward kernel (its cross-chunk carry
   lives in a VMEM scratch and Pallas has no AD rule for a stateful ref), so the kernel is
   wrapped in `jax.custom_vjp` with the exact analytic adjoint -- the same wiring as
   `experiments/s5ssm/s5ssm_grad.py`, reproduced here because that file relies on the
   experiment harness injecting `_core`'s names rather than importing them.

2. **Batch.** The kernel wants `u` as `(b, L, H)` and builds a `(b, C)` grid; upstream calls it
   under `nn.vmap` with `u` as `(L, H)`. Letting Pallas' generic batching rule add the batch
   dim would break the kernel: it keys its cross-chunk carry reset off `pl.program_id(1)`, and
   the generic rule shifts every program id by one. So `s5_pallas_apply` is a
   `jax.custom_batching.custom_vmap`: unbatched it calls the kernel with `b=1`, and under
   `vmap` its rule hands the REAL batch straight to the kernel. `grad(vmap(...))` -- the order
   flax produces (nn.vmap inside value_and_grad) -- resolves batching first, so the custom_vjp
   backward then sees the batched call it was written for.

3. **Chunk divisibility.** The kernel requires `L % chunk == 0` and Mosaic requires
   `chunk % 8 == 0`. `_pick_chunk` picks the largest admissible chunk <= LOBS5_PALLAS_CHUNK.

4. **Matmul precision.** The vendored kernel defaults to `Precision.HIGHEST`, which is wrong for
   a drop-in -- it makes the kernel MORE precise than the `associative_scan` it replaces, at ~6x
   the MXU passes. `_match_upstream_precision` pins it to what upstream actually uses instead.

Preconditions: conj_sym=True, bidirectional=False -- the S5 default and the only config LOBS5
trains. A bidirectional C_tilde is (H, 2P) and is rejected by the kernel's own shape assert.

Enable with `--use_pallas_ssm=True` on run_train.py (or `LOBS5_PALLAS_SSM=1`).

`pallas_s5_core.py` is a VERBATIM copy of `experiments/s5ssm/_core.py` in the kernels repo;
do not edit it here -- re-copy it instead (`tools/sync_pallas_kernel.py`).
"""
import collections
import functools
import os

import jax

from . import pallas_s5_core as _core

# ---------------------------------------------------------------- configuration
# Every knob is read from the environment AT CALL TIME, never at import time. run_train.py sets
# these from its --pallas_* flags after argparse, which happens well after `s5.pallas_ssm` may
# already have been imported (run_train.py imports lob.dataloading at module scope) -- an
# import-time read would silently ignore the flags.
_DEFAULTS = {
    # Largest chunk (in tokens) to try; the value actually used must also divide L and be a
    # multiple of 8. 240 divides both 12000 (msg_seq_len=500) and 1200, and sits well inside
    # the Lambda_bar**-t f32-overflow ceiling (~1650 at S5's HiPPO init). Bigger chunks are not
    # faster at production widths and the largest ones VMEM-OOM.
    "LOBS5_PALLAS_CHUNK": "240",
    "LOBS5_PALLAS_VMEM_MB": "64",
    # Backward implementation. The pure-XLA analytic adjoint is correct-by-construction but
    # runs two sequential length-L lax.scans and materialises three (b,L,P) complex arrays in
    # HBM -- exactly the traffic the fused forward exists to delete. At d_model=2048/P=1024/
    # L=12000/b=2 on v5e-1 it pins fwd+bwd at 41.7 ms *regardless of how fast the forward is*
    # (slower than stock's 29.0 ms end to end); the fused backward gives 10.17 ms (2.87x).
    # So: fused by default. `0` selects the XLA adjoint as a correctness fallback.
    "LOBS5_PALLAS_FUSED_BWD": "1",
}
_Cfg = collections.namedtuple("_Cfg", "max_chunk vmem_mb fused_bwd")


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _get(name: str) -> str:
    return os.environ.get(name, _DEFAULTS[name])


def config() -> _Cfg:
    """ Read the kernel configuration fresh from the environment.
        Validated here so a typo fails loudly at the first S5 layer
        instead of silently selecting a default.
        Returns:
            _Cfg(max_chunk (int), vmem_mb (int), fused_bwd (bool))
    """
    try:
        max_chunk, vmem_mb = int(_get("LOBS5_PALLAS_CHUNK")), int(_get("LOBS5_PALLAS_VMEM_MB"))
    except ValueError as e:
        raise ValueError(f"LOBS5_PALLAS_CHUNK / LOBS5_PALLAS_VMEM_MB must be integers: {e}")
    if max_chunk < 8:
        raise ValueError(f"LOBS5_PALLAS_CHUNK must be >= 8 (Mosaic block rule), got {max_chunk}")
    if vmem_mb < 1:
        raise ValueError(f"LOBS5_PALLAS_VMEM_MB must be >= 1, got {vmem_mb}")
    return _Cfg(max_chunk, vmem_mb, _truthy(_get("LOBS5_PALLAS_FUSED_BWD")))


def use_pallas_ssm() -> bool:
    """ Whether the fused Pallas S5 kernel should replace the associative_scan.
        Returns:
            enabled (bool): True if LOBS5_PALLAS_SSM is set truthy
    """
    return _truthy(os.environ.get("LOBS5_PALLAS_SSM", "0"))


def _pick_chunk(L: int, max_chunk: int) -> int:
    """ Choose the kernel's chunk length. Mosaic requires a block's last two dims
        to be divisible by (8,128) or equal the full array dim; the block here is
        (1, chunk, H) with H the full feature dim, so only chunk % 8 is constrained.
        Args:
            L (int): sequence length
            max_chunk (int): upper bound from LOBS5_PALLAS_CHUNK
        Returns:
            chunk (int): largest multiple of 8 <= max_chunk that divides L
        Raises:
            ValueError: if no such chunk exists (L not divisible by 8)
    """
    for c in range(min(max_chunk, L) // 8 * 8, 0, -8):
        if L % c == 0:
            return c
    raise ValueError(
        f"no admissible Pallas chunk for L={L} with LOBS5_PALLAS_CHUNK={max_chunk}: need a "
        f"multiple of 8 that divides L (L % 8 == {L % 8}). Pad the sequence, pick an L "
        f"divisible by 8, or raise LOBS5_PALLAS_CHUNK.")


# ---------------------------------------------------------------- custom_vjp wiring
# Mirrors experiments/s5ssm/s5ssm_grad.py. The whole config rides as non-differentiable leading
# args -- not just so the values are static, but so the SAME values provably reach both fwd and
# bwd. Reading `fused_bwd` independently in each would let an env change between the forward and
# the backward pair a states-emitting forward with the XLA adjoint (or worse, the reverse).
def _match_upstream_precision():
    """ Pin the kernel's matmul precision to exactly what upstream apply_ssm uses.

    Upstream writes bare `B_bar @ u` / `C_tilde @ x` with no `precision=` argument anywhere, and
    nothing in this repo sets `jax_default_matmul_precision` -- so its matmuls run at whatever
    JAX's global default is. `None` is that same "unspecified" value, so the kernel inherits the
    global knob exactly as their code does; hardcoding Precision.DEFAULT would instead pin the
    kernel and let it silently diverge from the rest of the model if anyone ever set the global.

    The vendored kernel defaults this to HIGHEST, which is ~1e-5-accurate but costs ~6x the MXU
    passes (f32 on the MXU is emulated as several bf16 passes) -- 5.6x on the forward at
    production widths, and it made the whole-model XLA compile at d_model=2048/L=12000
    impractical. It also made the kernel MORE precise than the baseline it replaces, which is
    not what a drop-in should do.

    Set here rather than in `pallas_s5_core.py` because that file is a verbatim vendored copy.
    The kernel body reads this module global at TRACE time and tracing happens synchronously
    inside the `_fused_*` calls below, so assigning immediately beforehand is what takes effect.
    """
    _core._MATMUL_PREC = None


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _s5_diff(interpret, chunk, vmem_mb, fused_bwd, Lambda_bar, B_bar, C_tilde, D, u):
    _match_upstream_precision()
    return _core._fused_forward(Lambda_bar, B_bar, C_tilde, D, u, interpret, chunk, vmem_mb)


def _s5_diff_fwd(interpret, chunk, vmem_mb, fused_bwd, Lambda_bar, B_bar, C_tilde, D, u):
    _match_upstream_precision()
    if fused_bwd:
        # The states-emitting forward is what makes the fused backward possible: it saves each
        # chunk's ENTRY state as a residual so the backward can recompute x on-chip.
        y, xent_r, xent_i = _core._fused_forward(Lambda_bar, B_bar, C_tilde, D, u, interpret,
                                                 chunk, vmem_mb, return_states=True)
    else:
        y = _core._fused_forward(Lambda_bar, B_bar, C_tilde, D, u, interpret, chunk, vmem_mb)
        xent_r = xent_i = None
    return y, (Lambda_bar, B_bar, C_tilde, D, u, xent_r, xent_i)


def _s5_diff_bwd(interpret, chunk, vmem_mb, fused_bwd, res, ybar):
    Lambda_bar, B_bar, C_tilde, D, u, xent_r, xent_i = res
    _match_upstream_precision()
    if fused_bwd:
        return _core._fused_backward(Lambda_bar, B_bar, C_tilde, D, u, ybar, xent_r, xent_i,
                                     interpret, chunk, vmem_mb)
    return _core._s5_bwd(Lambda_bar, B_bar, C_tilde, D, u, ybar)


_s5_diff.defvjp(_s5_diff_fwd, _s5_diff_bwd)


def _call(Lambda_bar, B_bar, C_tilde, D, u_bLH):
    """ Resolve the config once and dispatch the batched kernel call.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix   (P,)
            B_bar (complex64): discretized input matrix                 (P, H)
            C_tilde (complex64): output matrix                          (H, P)
            D (float32): feedthrough vector                             (H,)
            u_bLH (float32): batched input sequences                    (b, L, H)
        Returns:
            output sequences (float32)                                  (b, L, H)
    """
    cfg = config()
    return _s5_diff(jax.default_backend() != "tpu", _pick_chunk(u_bLH.shape[1], cfg.max_chunk),
                    cfg.vmem_mb, cfg.fused_bwd, Lambda_bar, B_bar, C_tilde, D, u_bLH)


# ---------------------------------------------------------------- vmap wiring
@jax.custom_batching.custom_vmap
def s5_pallas_apply(Lambda_bar, B_bar, C_tilde, D, u):
    """ Compute the LxH output of the S5 SSM given an LxH input sequence,
        using the fused Pallas TPU kernel instead of the parallel scan.
        Drop-in for the body of S5SSM.__call__ (the D*u skip is folded in).
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix   (P,)
            B_bar (complex64): discretized input matrix                 (P, H)
            C_tilde (complex64): output matrix                          (H, P)
            D (float32): feedthrough vector                             (H,)
            u (float32): input sequence                                 (L, H)
        Returns:
            output sequence (float32): 2 Re(C_tilde @ x) + D * u        (L, H)
    """
    return _call(Lambda_bar, B_bar, C_tilde, D, u[None])[0]


@s5_pallas_apply.def_vmap
def _s5_pallas_apply_vmap(axis_size, in_batched, Lambda_bar, B_bar, C_tilde, D, u):
    """ custom_vmap rule: hand the real (b, L, H) batch straight to the kernel,
        instead of letting Pallas' generic batching rule prepend a grid dim
        (which would shift every pl.program_id and break the carry reset).
        Args:
            axis_size (int): size of the mapped axis (unused)
            in_batched (tuple of bool): which arguments carry a batch dim
            Lambda_bar, B_bar, C_tilde, D: SSM parameters, must be unbatched
            u (float32): batched input sequences                        (b, L, H)
        Returns:
            output sequences (float32) (b, L, H), and True (result is batched)
        Raises:
            NotImplementedError: if the parameters are batched or u is not
    """
    l_b, b_b, c_b, d_b, u_b = in_batched
    if l_b or b_b or c_b or d_b:
        raise NotImplementedError(
            "the fused S5 kernel needs the SSM parameters shared across the batch "
            "(LOBS5's nn.vmap broadcasts them: variable_axes maps params to None)")
    if not u_b:
        raise NotImplementedError("expected the input sequence to be the batched argument")
    return _call(Lambda_bar, B_bar, C_tilde, D, u), True
