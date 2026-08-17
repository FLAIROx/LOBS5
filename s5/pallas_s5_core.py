# experiments/s5ssm/_core.py
"""Shared core for the s5ssm experiment group: the S5 diagonal SSM forward
(arXiv:2208.04933, code FLAIROx/LOBS5 s5/ssm.py `apply_ssm`), in two independent
formulations, plus shapes and inputs.

The op (conj_sym=True, bidirectional=False -- the S5 default; batched over sequences):

    Bu[l]  = B_bar @ u[l]                              # input->state, (P,) complex
    x[l]   = Lambda_bar (.) x[l-1] + Bu[l]             # diagonal complex recurrence, x[-1]=0
    y[l]   = 2 * Re( C_tilde @ x[l] ) + D (.) u[l]     # state->output (conj_sym x2) + skip

Lambda_bar (P,) / B_bar (P,H) / C_tilde (H,P) are complex64 (already discretized); u (b,L,H)
and D (H,) are f32; y is f32. This is a LINEAR TIME-INVARIANT diagonal SSM: Lambda_bar is the
same at every step (unlike Mamba-2's per-token, input-dependent decay), so the intra-chunk
decay is a pure power Lambda_bar**(i-j).

Two formulations:
  * `_s5_scan` -- the literal parallel-prefix recurrence via jax.lax.associative_scan, exactly
    as LOBS5 ships it. The trustworthy golden AND the baseline to beat. On TPU the scan is a
    log-depth tree of small complex ops (VPU/gather bound), which is what we want to replace.
  * `_s5_chunked` -- the chunked / "state-space-duality" reformulation. Splits the sequence
    into chunks of length T; each chunk's state is (input->state matmul) + (bounded in-chunk
    prefix sum over the decay-normalized inputs); chunks are stitched by a tiny per-chunk
    diagonal carry; the output is a (state->output matmul). The two projections land on the
    MXU and the scan collapses to a length-T cumsum + a length-C carry -- the blueprint the
    Pallas kernel fuses into one chunk-grid pass.

Both return y f32 (b,L,H). Everything is complex64 here (XLA handles complex natively); the
Pallas kernel later splits re/im and realises each complex matmul as 2 real matmuls -- note
`2*Re(C@x) = 2*(Re(C)@Re(x) - Im(C)@Im(x))` and `Bu = Re(B)@u + i*Im(B)@u` (u real) both need
only 2 real matmuls, not 4.

Shapes via S5_SHAPE env ('local' tiny CPU preset | 's5' realistic long-sequence regime),
auto-selected by backend like mamba2/mhc_lite: 's5' only on TPU.
"""
import functools as _functools
import math as _math
import os as _os

import numpy as _onp

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_S5_SHAPES = {
    # b=batch, L=seqlen, H=features (d_model), P=SSM state size (# complex modes), T=chunk len
    "local": dict(b=2, L=64, H=16, P=24, T=16),
    # Very-long-sequence S5 layer: d_model=256, state=256, L=16384 (max scan pressure -- the
    # associative_scan baseline dominates here, so the chunked/MXU win is largest). conj_sym
    # halves the true state; P is the # of complex modes actually scanned. T=128 default keeps
    # the in-chunk Lambda_bar**-t dynamic range safe (T bounds it, NOT L -- see _s5_chunked).
    "s5": dict(b=4, L=16384, H=256, P=256, T=128),
}

# S5 discretization defaults (LOBS5 init_log_steps / ssm_init).
_DT_MIN, _DT_MAX = 1e-3, 1e-1


def _s5_default_shape():
    try:
        is_tpu = jax.default_backend() == "tpu"
    except Exception:
        is_tpu = False
    want = _os.environ.get("S5_SHAPE")
    # 's5' (L=4096 golden associative_scan) would crawl on CPU -- never select it off-TPU,
    # so pytest stays fast and safe.
    if want in _S5_SHAPES and not (want == "s5" and not is_tpu):
        return want
    return "s5" if is_tpu else "local"


S5_SHAPE = _s5_default_shape()
_S5CFG = _S5_SHAPES[S5_SHAPE]


def _discretize_zoh(Lambda, B_tilde, Delta):
    """Zero-order-hold discretization (LOBS5 discretize_zoh): Lambda (P,), B_tilde (P,H),
    Delta (P,) -> Lambda_bar (P,), B_bar (P,H), all complex. |Lambda_bar| = exp(Re(Lambda)*Delta)."""
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1.0 / Lambda * (Lambda_bar - 1.0))[:, None] * B_tilde
    return Lambda_bar, B_bar


def make_inputs(seed=0, *, b=None, L=None, H=None, P=None):
    """(Lambda_bar, B_bar, C_tilde, D, u) at the current preset -- a realistic random S5
    parameterization, already discretized (the op boundary is `apply_ssm`, whose inputs are
    the discretized matrices). Continuous eigenvalues follow S5 init: Re<0 in a HiPPO-ish
    band, log-uniform step Delta in [dt_min, dt_max]; |Lambda_bar| lands in ~[0.74, 0.999]
    (long-memory modes near the unit circle -- exactly the regime that makes the scan the
    bottleneck and bounds the chunk length, see `_s5_chunked`). The keyword shape overrides
    let tests sample the same distribution at non-preset shapes (e.g. LOBS5's L=12000/H=32/
    P=16); the harness always calls make_inputs(seed) = the preset."""
    b = _S5CFG["b"] if b is None else b
    L = _S5CFG["L"] if L is None else L
    H = _S5CFG["H"] if H is None else H
    P = _S5CFG["P"] if P is None else P
    ks = jax.random.split(jax.random.key(seed), 7)
    c64 = jnp.complex64

    # Continuous diagonal eigenvalues: Lambda = -exp(unif[log .5, log 2]) + i*w.
    lam_re = -jnp.exp(jax.random.uniform(ks[0], (P,), jnp.float32,
                                         _math.log(0.5), _math.log(2.0)))
    lam_im = jax.random.uniform(ks[1], (P,), jnp.float32, 0.0, 20.0)
    Lambda = (lam_re + 1j * lam_im).astype(c64)
    Delta = jnp.exp(jax.random.uniform(ks[2], (P,), jnp.float32,
                                       _math.log(_DT_MIN), _math.log(_DT_MAX)))

    # B_tilde (P,H), C_tilde (H,P) complex, lecun-ish scaling; D (H,) real; u (b,L,H) real.
    B_tilde = ((jax.random.normal(ks[3], (P, H), jnp.float32)
                + 1j * jax.random.normal(ks[4], (P, H), jnp.float32))
               / _math.sqrt(H)).astype(c64)
    C_tilde = ((jax.random.normal(ks[5], (H, P), jnp.float32)
                + 1j * jax.random.normal(ks[6], (H, P), jnp.float32))
               / _math.sqrt(P)).astype(c64)
    D = jax.random.normal(jax.random.key(seed + 101), (H,), jnp.float32)
    u = jax.random.normal(jax.random.key(seed + 202), (b, L, H), jnp.float32)

    Lambda_bar, B_bar = _discretize_zoh(Lambda, B_tilde, Delta)
    return (Lambda_bar, B_bar, C_tilde, D, u)


# The in-chunk prefix normalizes by Lambda_bar**-t, whose magnitude grows like
# |Lambda_bar|**-(T-1) (and Lambda_bar**t grows the same way for |Lambda_bar|>1). Past f32
# range the kernel silently returns NaN/garbage while the associative_scan golden stays exact
# -- clip_eigs only bounds Re(Lambda) from ABOVE, so a legitimately fast-decaying trained mode
# can cross the cliff at the shipped T=128/256. Budget 120 bits (f32 max is ~2**128; the ~8
# spare bits absorb the |Bu| factors and the T-term prefix sums that overflow slightly before
# the ramp itself does).
_RAMP_MAX_BITS = 120.0


def _guard_ramp_range(Lambda_bar, T):
    """Raise ValueError if the (T,P) decay ramps for chunk length T can overflow f32 for this
    Lambda_bar. Host-side and best-effort: when Lambda_bar is a tracer (whole call under
    jax.jit) the values are unreadable and the guard silently skips -- eager callers and the
    test suite always get it."""
    try:
        mags = _onp.abs(_onp.asarray(Lambda_bar))
    except jax.errors.TracerArrayConversionError:
        return
    mmin, mmax = float(mags.min()), float(mags.max())
    if mmin == 0.0:
        raise ValueError("Lambda_bar contains a zero mode: Lambda_bar**-t is undefined")
    bits_per_step = _math.log2(max(mmax, 1.0 / mmin))
    if (T - 1) * bits_per_step > _RAMP_MAX_BITS:
        t_max = int(_RAMP_MAX_BITS / bits_per_step) + 1
        raise ValueError(
            f"chunk length {T} overflows f32 in the Lambda_bar**±t decay ramps: the extreme "
            f"mode (min|Lambda_bar|={mmin:.4g}, max={mmax:.4g}) spans 2**{(T - 1) * bits_per_step:.0f} "
            f"over one chunk (limit 2**{_RAMP_MAX_BITS:.0f}). Use chunk <= {t_max} "
            f"(the associative_scan golden handles such modes exactly; the chunked forms cannot).")


def _binary_op(q_i, q_j):
    """S5's associative-scan combiner for a diagonal linear recurrence (LOBS5 binary_operator):
    combine prefixes i (earlier) and j (later) -> (A_j*A_i, A_j*b_i + b_j)."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def _s5_scan(Lambda_bar, B_bar, C_tilde, D, u):
    """The golden: literal S5 `apply_ssm` (conj_sym) via associative_scan, batched over
    sequences. u (b,L,H) f32 -> y (b,L,H) f32."""
    b, L, _ = u.shape
    P = Lambda_bar.shape[0]
    uc = u.astype(Lambda_bar.dtype)                                  # (b,L,H) complex
    Bu = jnp.einsum("ph,blh->blp", B_bar, uc)                        # input->state (b,L,P)
    Lam = jnp.broadcast_to(Lambda_bar, (b, L, P))                    # LTI: same at every step
    _, xs = jax.lax.associative_scan(_binary_op, (Lam, Bu), axis=1)  # states (b,L,P)
    ys = 2.0 * jnp.einsum("hp,blp->blh", C_tilde, xs).real           # conj_sym state->output
    return (ys + D[None, None, :] * u).astype(u.dtype)


def _s5_chunked(Lambda_bar, B_bar, C_tilde, D, u, *, chunk):
    """Chunked reformulation of `apply_ssm` (the MXU-friendly form). Mathematically identical
    to `_s5_scan`; differs only in association order (bounded per-chunk prefix + per-chunk
    carry vs one global log-depth scan). u (b,L,H) f32 -> y (b,L,H) f32.

    Stability: the in-chunk prefix normalizes inputs by Lambda_bar**(-t), t in [0,T), which
    grows like |Lambda_bar|**-(T-1). For long-memory modes (|Lambda_bar| ~ 0.9-0.999) this is
    tiny even at T=256; T is bounded by the FASTEST-decaying mode, the same chunk-size/decay
    tradeoff SSD has. A log-space-stabilized variant is a follow-up A/B knob."""
    b, L, _ = u.shape
    P = Lambda_bar.shape[0]
    T = int(chunk)
    assert L % T == 0, "seq must be divisible by chunk"
    _guard_ramp_range(Lambda_bar, T)
    C = L // T
    cdt = Lambda_bar.dtype
    uc = u.astype(cdt)

    Bu = jnp.einsum("ph,blh->blp", B_bar, uc).reshape(b, C, T, P)    # input->state matmul

    t = jnp.arange(T)
    pow_pos = Lambda_bar[None, :] ** t[:, None]                     # (T,P) = Lambda_bar**t
    pow_neg = Lambda_bar[None, :] ** (-t[:, None])                  # (T,P) = Lambda_bar**-t

    # --- intra-chunk state: x_intra[c,t] = sum_{t'<=t} Lambda_bar**(t-t') Bu[c,t'] ---------
    # = Lambda_bar**t (.) cumsum_t( Lambda_bar**-t' (.) Bu ). Bounded-range in-chunk prefix.
    Bn = Bu * pow_neg[None, None]                                   # (b,C,T,P)
    x_intra = jnp.cumsum(Bn, axis=2) * pow_pos[None, None]          # (b,C,T,P)

    # --- cross-chunk carry: each chunk's end-state, stitched by a tiny diagonal recurrence --
    s_end = x_intra[:, :, -1, :]                                    # (b,C,P) chunk end-states
    dT = Lambda_bar ** T                                           # per-chunk decay (LTI)
    Lam_c = jnp.broadcast_to(dT, (b, C, P))
    _, full_end = jax.lax.associative_scan(_binary_op, (Lam_c, s_end), axis=1)  # (b,C,P)
    # state ENTERING chunk c = end-state of chunk c-1 (exclusive); chunk 0 enters at zero.
    s_carry = jnp.concatenate([jnp.zeros_like(full_end[:, :1]), full_end[:, :-1]], axis=1)

    # --- total state + output --------------------------------------------------------------
    # x[c,t] = Lambda_bar**(t+1) (.) s_carry[c] + x_intra[c,t]
    x = x_intra + (pow_pos * Lambda_bar[None, :])[None, None] * s_carry[:, :, None, :]
    xs = x.reshape(b, L, P)
    ys = 2.0 * jnp.einsum("hp,blp->blh", C_tilde, xs).real         # conj_sym state->output
    return (ys + D[None, None, :] * u).astype(u.dtype)


def _s5_cost(b, L, H, P, *, out_bytes=4):
    """(flops, bytes_accessed) of the chunked S5 forward -- for pl.CostEstimate and roofline
    math. Dominated by the two complex projections, each = 2 real matmuls (u real, output
    real part), so 4 real (b*L*H*P) MACs total * 2 flop/MAC = 8*b*L*H*P. The in-chunk cumsum,
    decay powers and per-chunk carry are O(b*L*P) -- negligible next to the matmuls."""
    flops = 8 * b * L * H * P
    bytes_accessed = (
        4 * b * L * H                      # u f32 in
        + out_bytes * b * L * H            # y out
        + 8 * (P + P * H + H * P)          # Lambda_bar, B_bar, C_tilde complex64
        + 4 * H                            # D f32
    )
    return flops, bytes_accessed


# --- backward (adjoint) ------------------------------------------------------------------
#
# The S5 forward is a diagonal LINEAR recurrence, so its backward is the classic result: the
# adjoint of a linear scan is ANOTHER linear diagonal scan, run in REVERSE time with the
# conjugated decay. Given cotangent ybar (b,L,H) on y:
#
#     lam[l] = 2 conj(C_tilde)^T ybar[l] + conj(Lambda_bar) (.) lam[l+1],   lam[L]=0   (reverse)
#
# and the parameter/input gradients are per-timestep contractions of lam and the forward
# states x (recomputed here by a plain lax.scan):
#
#     grad_u[l] = D (.) ybar[l] + Re( conj(B_bar)^T lam[l] )
#     grad_D    = sum_l ybar[l] (.) u[l]
#     grad_C    = 2 sum_l ybar[l] (x) x[l]           (H,P)   complex
#     grad_B    = sum_l conj(lam[l]) (x) u[l]        (P,H)   complex
#     grad_L    = sum_l conj(lam[l]) (.) x[l-1]      (P,)    complex,  x[-1]=0
#
# The conjugations on grad_C/grad_B/grad_L match JAX's complex-VJP convention (jax.vjp returns
# conj(Wirtinger dL/dz) for complex inputs); this whole function is checked bit-for-bit against
# jax.vjp(_s5_scan) in test_s5ssm_grad.py, and transitively against the real upstream apply_ssm
# via the gradient fixture. Pure JAX, no Pallas -- this is the GOLDEN the fused backward kernel
# must reproduce, and doubles as the (XLA) backward for the custom_vjp's first, kernel-free rung.
def _s5_bwd(Lambda_bar, B_bar, C_tilde, D, u, ybar):
    """Analytic S5 backward. Returns (grad_Lambda_bar, grad_B_bar, grad_C_tilde, grad_D,
    grad_u) in jax.vjp argument order, matching JAX's complex-grad convention exactly."""
    b, _, _ = u.shape
    P = Lambda_bar.shape[0]
    cdt = Lambda_bar.dtype
    uc = u.astype(cdt)

    # forward states x (b,L,P) -- recomputed (cheap; a fused kernel would recompute in-chunk).
    Bu = jnp.einsum("ph,blh->blp", B_bar, uc)
    def _fstep(x, Bu_l):
        x = Lambda_bar[None, :] * x + Bu_l
        return x, x
    _, xs = jax.lax.scan(_fstep, jnp.zeros((b, P), cdt), jnp.moveaxis(Bu, 1, 0))
    x = jnp.moveaxis(xs, 0, 1)                                      # (b,L,P)

    # adjoint reverse scan for lam (b,L,P).
    src = 2.0 * jnp.einsum("hp,blh->blp", jnp.conj(C_tilde), ybar.astype(cdt))
    Lconj = jnp.conj(Lambda_bar)
    def _bstep(lam_next, src_l):
        lam = src_l + Lconj[None, :] * lam_next
        return lam, lam
    _, lam_rev = jax.lax.scan(_bstep, jnp.zeros((b, P), cdt),
                              jnp.moveaxis(src, 1, 0), reverse=True)
    lam = jnp.moveaxis(lam_rev, 0, 1)                              # (b,L,P)

    x_prev = jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
    lam_c = jnp.conj(lam)
    grad_u = D[None, None, :] * ybar + jnp.einsum("ph,blp->blh", jnp.conj(B_bar), lam).real
    grad_D = jnp.einsum("blh,blh->h", ybar, u)
    grad_C = 2.0 * jnp.einsum("blh,blp->hp", ybar.astype(cdt), x)
    grad_B = jnp.einsum("blp,blh->ph", lam_c, uc)
    grad_L = jnp.einsum("blp,blp->p", lam_c, x_prev)
    return grad_L, grad_B, grad_C, grad_D.astype(u.dtype), grad_u.astype(u.dtype)


# ==================== fused Pallas kernels (forward + backward) ===========================
# Shared by the differentiable variant (s5ssm_grad.py) and the fwd+bwd bench variants; kept in
# _core so any sibling can use them (siblings don't import each other -- only _core is injected).
#
# Matmul precision knob. TPU default bf16-rounds f32 dot operands (~4e-3 grad error vs the
# f32-accurate associative_scan autodiff -- the mamba2 gotcha); HIGHEST -> ~1e-5 but costs
# ~2-3x on the matmul-heavy backward. Default HIGHEST (accurate gradients, matches the
# "correct vs ground truth" bar). NOTE: the forward-only bench kernel in s5ssm.py is a
# separate copy that deliberately keeps the TPU DEFAULT precision (like upstream apply_ssm,
# which pins nothing) -- so the two "identical" forwards differ on real TPU by ~bf16 rounding.
#
# S5_KERNEL_PREC=default overrides to the TPU default (bf16-rounded MXU passes) -- the SAME
# numerics class as upstream apply_ssm, which pins nothing. At narrow widths (H<=256) HIGHEST
# is free (kernel is HBM/VPU-bound, MXU idle); at LOBS5's production widths (H=1024/P=512+)
# the projection matmuls dominate and HIGHEST's extra MXU passes are the kernel's single
# biggest handicap vs the default-precision baseline (measured on the lobs5_75m_short bench).
# Interpret mode ignores precision entirely, so local goldens are unaffected either way.
_MATMUL_PREC = (jax.lax.Precision.DEFAULT
                if _os.environ.get("S5_KERNEL_PREC", "").lower() == "default"
                else jax.lax.Precision.HIGHEST)
def _s5_fwd_kernel(u_ref, Br_ref, Bi_ref, Cr_ref, Ci_ref, D_ref,
                   ppr_ref, ppi_ref, pnr_ref, pni_ref, pp1r_ref, pp1i_ref,
                   y_ref, st_r, st_i, *, T):
    """Fused S5 forward (same math as s5ssm.py's kernel; this copy pins HIGHEST matmul
    precision -- see the _MATMUL_PREC note). grid (b,C); (1,P) complex state carried
    in a VMEM scratch across chunks; complex as split re/im; in-chunk prefix as a tril matmul."""
    f32 = jnp.float32
    ci = pl.program_id(1)

    @pl.when(ci == 0)
    def _reset():
        st_r[...] = jnp.zeros_like(st_r)
        st_i[...] = jnp.zeros_like(st_i)

    u_c = u_ref[0]
    dot = lambda a, b, ca, cb: jax.lax.dot_general(
        a, b, (((ca,), (cb,)), ((), ())), preferred_element_type=f32,
        precision=_MATMUL_PREC)                # see _MATMUL_PREC note above

    Bur = dot(u_c, Br_ref[...], 1, 1)
    Bui = dot(u_c, Bi_ref[...], 1, 1)
    pnr, pni = pnr_ref[...], pni_ref[...]
    Bnr = Bur * pnr - Bui * pni
    Bni = Bur * pni + Bui * pnr
    ii = jax.lax.broadcasted_iota(jnp.int32, (T, T), 0)
    jj = jax.lax.broadcasted_iota(jnp.int32, (T, T), 1)
    tril = (ii >= jj).astype(f32)
    Bcr = dot(tril, Bnr, 1, 0)
    Bci = dot(tril, Bni, 1, 0)
    ppr, ppi = ppr_ref[...], ppi_ref[...]
    xir = Bcr * ppr - Bci * ppi
    xii = Bcr * ppi + Bci * ppr
    scr, sci = st_r[...], st_i[...]
    pp1r, pp1i = pp1r_ref[...], pp1i_ref[...]
    xr = xir + (pp1r * scr - pp1i * sci)
    xi = xii + (pp1r * sci + pp1i * scr)

    y = 2.0 * (dot(xr, Cr_ref[...], 1, 1) - dot(xi, Ci_ref[...], 1, 1))
    y_ref[0] = y + D_ref[...] * u_c
    st_r[...] = xr[T - 1:T, :]
    st_i[...] = xi[T - 1:T, :]


def _s5_fwd_kernel_states(u_ref, Br_ref, Bi_ref, Cr_ref, Ci_ref, D_ref,
                          ppr_ref, ppi_ref, pnr_ref, pni_ref, pp1r_ref, pp1i_ref,
                          y_ref, xer_ref, xei_ref, st_r, st_i, *, T):
    """Forward kernel that ALSO emits each chunk's ENTRY forward-state (the carry read at the top
    of the program) into xer/xei (b,C,P), so the backward can recompute the in-chunk forward
    states x on-chip -- x is never materialised to HBM."""
    f32 = jnp.float32
    ci = pl.program_id(1)

    @pl.when(ci == 0)
    def _reset():
        st_r[...] = jnp.zeros_like(st_r)
        st_i[...] = jnp.zeros_like(st_i)

    scr, sci = st_r[...], st_i[...]
    xer_ref[0, 0] = scr                                 # entry state = carry before this chunk
    xei_ref[0, 0] = sci                                 # block (1,1,1,P) -> [0,0] is (1,P)

    u_c = u_ref[0]
    dot = lambda a, b, ca, cb: jax.lax.dot_general(
        a, b, (((ca,), (cb,)), ((), ())), preferred_element_type=f32,
        precision=_MATMUL_PREC)                # see _MATMUL_PREC note above

    Bur = dot(u_c, Br_ref[...], 1, 1)
    Bui = dot(u_c, Bi_ref[...], 1, 1)
    pnr, pni = pnr_ref[...], pni_ref[...]
    Bnr = Bur * pnr - Bui * pni
    Bni = Bur * pni + Bui * pnr
    ii = jax.lax.broadcasted_iota(jnp.int32, (T, T), 0)
    jj = jax.lax.broadcasted_iota(jnp.int32, (T, T), 1)
    tril = (ii >= jj).astype(f32)
    Bcr = dot(tril, Bnr, 1, 0)
    Bci = dot(tril, Bni, 1, 0)
    ppr, ppi = ppr_ref[...], ppi_ref[...]
    xir = Bcr * ppr - Bci * ppi
    xii = Bcr * ppi + Bci * ppr
    pp1r, pp1i = pp1r_ref[...], pp1i_ref[...]
    xr = xir + (pp1r * scr - pp1i * sci)
    xi = xii + (pp1r * sci + pp1i * scr)

    y = 2.0 * (dot(xr, Cr_ref[...], 1, 1) - dot(xi, Ci_ref[...], 1, 1))
    y_ref[0] = y + D_ref[...] * u_c
    st_r[...] = xr[T - 1:T, :]
    st_i[...] = xi[T - 1:T, :]


def _decay_ramps(Lambda_bar, T):
    """LTI decay ramps precomputed complex on host, returned split as (T,P) f32:
    (Lam^t, Lam^-t, Lam^(t+1)) x (re, im). Guarded against f32 range overflow (fast-decaying
    modes at large T silently NaN the whole kernel otherwise -- see _guard_ramp_range)."""
    _guard_ramp_range(Lambda_bar, T)
    f32 = jnp.float32
    t = jnp.arange(T)
    ppos = Lambda_bar[None, :] ** t[:, None]
    pneg = Lambda_bar[None, :] ** (-t[:, None])
    pp1 = ppos * Lambda_bar[None, :]
    return [r.astype(f32) for r in
            (ppos.real, ppos.imag, pneg.real, pneg.imag, pp1.real, pp1.imag)]


def _decay_ramps_bwd(Lambda_bar, T):
    """The one extra backward ramp: cramp = conj(Lambda_bar)^(T-t), t=0..T-1, split (T,P) f32.
    (conj(L^t), conj(L^-t) are obtained in-kernel by negating the forward ramps' imag parts.)"""
    f32 = jnp.float32
    t = jnp.arange(T)
    cramp = jnp.conj(Lambda_bar)[None, :] ** (T - t[:, None])
    return cramp.real.astype(f32), cramp.imag.astype(f32)


def _fused_forward(Lambda_bar, B_bar, C_tilde, D, u, interpret, chunk, vmem_limit_mb,
                   return_states=False):
    """Forward pallas_call. return_states=True additionally emits per-chunk entry forward-states
    xent (b,C,P split re/im) for the backward -- used only on the custom_vjp fwd path."""
    b, L, H = u.shape
    P = Lambda_bar.shape[0]
    # conj_sym=True unidirectional shapes only: a bidirectional C_tilde is (H,2P) and would
    # otherwise be SILENTLY half-read by the resident (H,P) BlockSpec.
    assert B_bar.shape == (P, H) and C_tilde.shape == (H, P), (
        f"B_bar {B_bar.shape} / C_tilde {C_tilde.shape} do not match (P,H)/(H,P)=({P},{H})/"
        f"({H},{P}) -- bidirectional (H,2P) C_tilde is not supported")
    T = int(chunk) if chunk is not None else min(_S5CFG["T"], L)
    assert L % T == 0, "seq must be divisible by chunk"
    C = L // T
    f32 = jnp.float32

    ramps = _decay_ramps(Lambda_bar, T)
    Br, Bi = B_bar.real.astype(f32), B_bar.imag.astype(f32)
    Cr, Ci = C_tilde.real.astype(f32), C_tilde.imag.astype(f32)
    Df = D.astype(f32)
    uf = u.astype(f32)

    flops, bytes_accessed = _s5_cost(b, L, H, P)
    flops += 4 * b * L * T * P
    cost = pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed)

    resident = lambda *s: pl.BlockSpec(s, lambda bi, ci: tuple(0 for _ in s))
    tp = lambda: resident(T, P)
    in_specs = [
        pl.BlockSpec((1, T, H), lambda bi, ci: (bi, ci, 0)),
        resident(P, H), resident(P, H),
        resident(H, P), resident(H, P),
        resident(H),
        tp(), tp(), tp(), tp(), tp(), tp(),
    ]
    y_spec = pl.BlockSpec((1, T, H), lambda bi, ci: (bi, ci, 0))
    common = dict(
        grid=(b, C), in_specs=in_specs,
        scratch_shapes=[pltpu.VMEM((1, P), f32), pltpu.VMEM((1, P), f32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=int(vmem_limit_mb) * 1024 * 1024),
        cost_estimate=cost, interpret=interpret)
    args = (uf, Br, Bi, Cr, Ci, Df, *ramps)

    if not return_states:
        return pl.pallas_call(
            _functools.partial(_s5_fwd_kernel, T=T),
            out_shape=jax.ShapeDtypeStruct((b, L, H), f32),
            out_specs=y_spec, **common)(*args)

    # entry-states as (b,C,1,P): block (1,1,1,P) keeps the last two block dims (1,P) equal to
    # the array's -- the Mosaic "divisible by (8,128) or full-dim" rule ((1,1,P) fails it).
    xent_spec = pl.BlockSpec((1, 1, 1, P), lambda bi, ci: (bi, ci, 0, 0))
    return pl.pallas_call(
        _functools.partial(_s5_fwd_kernel_states, T=T),
        out_shape=(jax.ShapeDtypeStruct((b, L, H), f32),
                   jax.ShapeDtypeStruct((b, C, 1, P), f32),
                   jax.ShapeDtypeStruct((b, C, 1, P), f32)),
        out_specs=(y_spec, xent_spec, xent_spec), **common)(*args)


# The adjoint of the S5 forward is another diagonal linear scan run in REVERSE time with the
# conjugated decay. This kernel fuses it: grid (b,C) walked so each program handles chunk
# c = C-1-ci (reverse chunk order), carrying the adjoint state lam in a VMEM scratch. Per chunk
# it recomputes the forward states x on-chip (from the entry state emitted by the forward pass)
# and the adjoint states lam (reverse in-chunk prefix), then reduces both straight into the
# parameter-gradient accumulators -- so the (b,L,P) x and lam NEVER touch HBM, exactly the fusion
# win the forward has. grad_u is written per chunk; grad_{D,C,B,Lambda} accumulate into resident
# outputs (shared across all programs), zeroed at the very first program.
def _s5_bwd_kernel(u_ref, yb_ref, xer_ref, xei_ref, Br_ref, Bi_ref, Cr_ref, Ci_ref, D_ref,
                   ppr_ref, ppi_ref, pnr_ref, pni_ref, pp1r_ref, pp1i_ref, crr_ref, cri_ref,
                   gu_ref, gD_ref, gCr_ref, gCi_ref, gBr_ref, gBi_ref, gLr_ref, gLi_ref,
                   la_r, la_i, *, T):
    f32 = jnp.float32
    bi = pl.program_id(0)
    ci = pl.program_id(1)
    dot = lambda a, b, ca, cb: jax.lax.dot_general(
        a, b, (((ca,), (cb,)), ((), ())), preferred_element_type=f32,
        precision=_MATMUL_PREC)                # see _MATMUL_PREC note above

    @pl.when(ci == 0)
    def _reset_carry():                                  # rightmost chunk of each batch: lam_next=0
        la_r[...] = jnp.zeros_like(la_r)
        la_i[...] = jnp.zeros_like(la_i)

    @pl.when((bi == 0) & (ci == 0))
    def _zero_accum():                                   # init the shared grad accumulators once
        gD_ref[...] = jnp.zeros_like(gD_ref)
        gCr_ref[...] = jnp.zeros_like(gCr_ref)
        gCi_ref[...] = jnp.zeros_like(gCi_ref)
        gBr_ref[...] = jnp.zeros_like(gBr_ref)
        gBi_ref[...] = jnp.zeros_like(gBi_ref)
        gLr_ref[...] = jnp.zeros_like(gLr_ref)
        gLi_ref[...] = jnp.zeros_like(gLi_ref)

    u_c = u_ref[0]                                        # (T,H)
    yb = yb_ref[0]                                        # (T,H)
    xent_r = xer_ref[0, 0]                                # (1,P) entry forward-state (block 1,1,1,P)
    xent_i = xei_ref[0, 0]
    pr, pi = ppr_ref[...], ppi_ref[...]                  # L^t
    nr, ni = pnr_ref[...], pni_ref[...]                  # L^-t
    p1r, p1i = pp1r_ref[...], pp1i_ref[...]              # L^(t+1)

    # --- recompute forward states x on-chip (same prefix as the forward kernel) -------------
    Bur = dot(u_c, Br_ref[...], 1, 1)
    Bui = dot(u_c, Bi_ref[...], 1, 1)
    Bnr = Bur * nr - Bui * ni
    Bni = Bur * ni + Bui * nr
    ii = jax.lax.broadcasted_iota(jnp.int32, (T, T), 0)
    jj = jax.lax.broadcasted_iota(jnp.int32, (T, T), 1)
    tril = (ii >= jj).astype(f32)
    Bcr = dot(tril, Bnr, 1, 0)
    Bci = dot(tril, Bni, 1, 0)
    xir = Bcr * pr - Bci * pi
    xii = Bcr * pi + Bci * pr
    xr = xir + (p1r * xent_r - p1i * xent_i)             # (T,P)
    xi = xii + (p1r * xent_i + p1i * xent_r)
    xpr = jnp.concatenate([xent_r, xr[:T - 1]], axis=0)  # x_prev = [xent, x[:T-1]]  (T,P)
    xpi = jnp.concatenate([xent_i, xi[:T - 1]], axis=0)

    # --- adjoint states lam via the reverse in-chunk prefix (triu) --------------------------
    # src = 2 conj(C)^T ybar ; conj(pos)=(pr,-pi) ; conj(neg)=(nr,-ni)
    sr = 2.0 * dot(yb, Cr_ref[...], 1, 0)                # (T,P)
    si = -2.0 * dot(yb, Ci_ref[...], 1, 0)
    snr = sr * pr + si * pi                              # src (.) conj(pos)
    sni = -sr * pi + si * pr
    triu = (jj >= ii).astype(f32)
    rcr = dot(triu, snr, 1, 0)                           # reverse cumsum (T,P)
    rci = dot(triu, sni, 1, 0)
    lir = rcr * nr + rci * ni                            # (.) conj(neg)
    lii = -rcr * ni + rci * nr
    lan_r, lan_i = la_r[...], la_i[...]                  # lam_next (1,P), broadcast over T
    crr, cri = crr_ref[...], cri_ref[...]               # cramp = conj(L)^(T-t) (T,P)
    lam_r = lir + (crr * lan_r - cri * lan_i)           # (T,P)
    lam_i = lii + (crr * lan_i + cri * lan_r)
    la_r[...] = lam_r[0:1, :]                            # new lam_next -> chunk c-1
    la_i[...] = lam_i[0:1, :]

    # --- grads -----------------------------------------------------------------------------
    # grad_u[t] = D (.) ybar[t] + Re(conj(B)^T lam[t]) = D*yb + lam_r@Br + lam_i@Bi
    gu_ref[0] = D_ref[...] * yb + dot(lam_r, Br_ref[...], 1, 0) + dot(lam_i, Bi_ref[...], 1, 0)
    gD_ref[...] += jnp.sum(yb * u_c, axis=0, keepdims=True)
    gCr_ref[...] += (2.0 * dot(yb, xr, 0, 0)).astype(gCr_ref.dtype) # 2 sum_t ybar (x) x  -> (H,P)
    gCi_ref[...] += (2.0 * dot(yb, xi, 0, 0)).astype(gCi_ref.dtype)
    gBr_ref[...] += dot(lam_r, u_c, 0, 0).astype(gBr_ref.dtype)     # sum_t conj(lam) (x) u -> (P,H)
    gBi_ref[...] += (-dot(lam_i, u_c, 0, 0)).astype(gBi_ref.dtype)
    gLr_ref[...] += jnp.sum(lam_r * xpr + lam_i * xpi, axis=0, keepdims=True)  # (1,P)
    gLi_ref[...] += jnp.sum(lam_r * xpi - lam_i * xpr, axis=0, keepdims=True)


def _fused_backward(Lambda_bar, B_bar, C_tilde, D, u, ybar, xent_r, xent_i,
                    interpret, chunk, vmem_limit_mb):
    """Fused backward pallas_call. xent_r/xent_i (b,C,P) are the per-chunk entry forward-states
    from the forward pass. Returns grads (grad_Lambda_bar, grad_B_bar, grad_C_tilde, grad_D,
    grad_u), complex/real, matching jax.vjp's convention."""
    b, L, H = u.shape
    P = Lambda_bar.shape[0]
    assert B_bar.shape == (P, H) and C_tilde.shape == (H, P), (
        f"B_bar {B_bar.shape} / C_tilde {C_tilde.shape} do not match (P,H)/(H,P)=({P},{H})/"
        f"({H},{P}) -- bidirectional (H,2P) C_tilde is not supported")
    T = int(chunk) if chunk is not None else min(_S5CFG["T"], L)
    assert L % T == 0, "seq must be divisible by chunk"  # else the reverse grid silently
    C = L // T                                           # covers only C*T of L timesteps
    f32 = jnp.float32
    bf16 = jnp.bfloat16

    fr = _decay_ramps(Lambda_bar, T)                     # pos,neg,pp1 (re,im)
    cr_r, cr_i = _decay_ramps_bwd(Lambda_bar, T)
    Br, Bi = B_bar.real.astype(f32), B_bar.imag.astype(f32)
    Cr, Ci = C_tilde.real.astype(f32), C_tilde.imag.astype(f32)
    Df = D.astype(f32)
    uf, ybf = u.astype(f32), ybar.astype(f32)

    resident = lambda *s: pl.BlockSpec(s, lambda bi, ci: tuple(0 for _ in s))
    tp = lambda: resident(T, P)
    seq = lambda: pl.BlockSpec((1, T, H), lambda bi, ci: (bi, C - 1 - ci, 0))   # reverse chunk
    ent = lambda: pl.BlockSpec((1, 1, 1, P), lambda bi, ci: (bi, C - 1 - ci, 0, 0))

    out_specs = (
        seq(),                                           # grad_u (reverse chunk)
        resident(1, H),                                  # grad_D
        resident(H, P), resident(H, P),                  # grad_C re,im
        resident(P, H), resident(P, H),                  # grad_B re,im
        resident(1, P), resident(1, P),                  # grad_Lambda re,im
    )
    out_shape = (
        jax.ShapeDtypeStruct((b, L, H), f32),
        jax.ShapeDtypeStruct((1, H), f32),
        jax.ShapeDtypeStruct((H, P), bf16), jax.ShapeDtypeStruct((H, P), bf16),
        jax.ShapeDtypeStruct((P, H), bf16), jax.ShapeDtypeStruct((P, H), bf16),
        jax.ShapeDtypeStruct((1, P), f32), jax.ShapeDtypeStruct((1, P), f32),
    )
    gu, gD, gCr, gCi, gBr, gBi, gLr, gLi = pl.pallas_call(
        _functools.partial(_s5_bwd_kernel, T=T),
        out_shape=out_shape,
        grid=(b, C),
        in_specs=[
            seq(), seq(),                                # u, ybar (reverse chunk)
            ent(), ent(),                                # xent re, im (reverse chunk)
            resident(P, H), resident(P, H),              # B re, im
            resident(H, P), resident(H, P),              # C re, im
            resident(H),                                 # D
            tp(), tp(), tp(), tp(), tp(), tp(),          # pos, neg, pp1 (re,im)
            tp(), tp(),                                  # cramp (re,im)
        ],
        out_specs=out_specs,
        scratch_shapes=[pltpu.VMEM((1, P), f32), pltpu.VMEM((1, P), f32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=int(vmem_limit_mb) * 1024 * 1024),
        interpret=interpret,
    )(uf, ybf, xent_r, xent_i, Br, Bi, Cr, Ci, Df, *fr, cr_r, cr_i)

    grad_Lambda_bar = (gLr[0] + 1j * gLi[0]).astype(Lambda_bar.dtype)
    grad_B_bar = (gBr + 1j * gBi).astype(B_bar.dtype)
    grad_C_tilde = (gCr + 1j * gCi).astype(C_tilde.dtype)
    grad_D = gD[0].astype(D.dtype)
    grad_u = gu.astype(u.dtype)
    return grad_Lambda_bar, grad_B_bar, grad_C_tilde, grad_D, grad_u


# --- fwd+bwd bench helpers (used by the s5ssm_gradstep_* variants) ------------------------
def _grad_cotangent(u):
    """Deterministic output cotangent ybar (b,L,H) for the fwd+bwd bench, seeded off nothing but
    a fixed key so run and reference differentiate against the SAME cotangent."""
    return jax.random.normal(jax.random.key(1234), u.shape, u.dtype)


def _flat_grads(grads):
    """Flatten the 5 S5 gradients (complex params + real D/u) into one f32 vector, so a bench's
    run/reference return a single comparable array AND every grad is a live output (no DCE that
    would under-time the backward)."""
    gL, gB, gC, gD, gu = grads
    parts = [gL.real, gL.imag, gB.real, gB.imag, gC.real, gC.imag, gD, gu]
    return jnp.concatenate([p.astype(jnp.float32).reshape(-1) for p in parts])


def _grad_seg_compare(got, ref):
    """Per-gradient compare for the gradstep variants' flat vector (mhclite split_bwd lesson:
    a single allclose over concat(grads) lets a small-scale segment hide errors under the
    large segments' atol -- the segments here span ~3 decades). Splits the _flat_grads vector
    back into its 5 gradients (gL, gB, gC each re+im, gD, gu) at the preset shapes and bounds
    each segment's ENERGY-relative error at 1e-3 -- loose vs the measured truth (~1e-6
    interpret / ~4e-5 TPU at HIGHEST) but tight enough to catch the known Mosaic-only failure
    class (losing the HIGHEST matmul-precision pin reads ~4e-3). Returns (ok, worst_rel)."""
    b, L, H, P = (_S5CFG[k] for k in ("b", "L", "H", "P"))
    sizes = {"grad_Lambda": 2 * P, "grad_B": 2 * P * H, "grad_C": 2 * H * P,
             "grad_D": H, "grad_u": b * L * H}
    got = _onp.asarray(got, _onp.float32).reshape(-1)
    ref = _onp.asarray(ref, _onp.float32).reshape(-1)
    assert got.size == ref.size == sum(sizes.values()), \
        f"flat grad vector size {got.size} != expected {sum(sizes.values())} at the preset"
    worst, off = 0.0, 0
    for name, n in sizes.items():
        g, r = got[off:off + n], ref[off:off + n]
        off += n
        rel = float(_onp.sqrt(_onp.sum((g - r) ** 2) / (_onp.sum(r ** 2) + 1e-30)))
        worst = max(worst, rel)
    return worst < 1e-3, worst


def _s5_stock_grads(Lambda_bar, B_bar, C_tilde, D, u):
    """The no-kernel training-step baseline: jax.vjp through the associative_scan golden
    (_s5_scan) for the fixed bench cotangent. Returns the flat grad vector. Pinned to HIGHEST
    matmul precision to MATCH the kernel's precision -- otherwise the accurate (HIGHEST) kernel
    is compared against a bf16-rounded reference (~4e-3) and the on-hardware correctness check
    trips on near-zero grad elements. associative_scan is scan-bound (few matmuls), so HIGHEST
    barely changes its runtime."""
    ybar = _grad_cotangent(u)
    with jax.default_matmul_precision("highest"):
        _, vjp = jax.vjp(_s5_scan, Lambda_bar, B_bar, C_tilde, D, u)
        return _flat_grads(vjp(ybar))
