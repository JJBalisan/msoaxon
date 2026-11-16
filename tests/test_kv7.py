"""The M-type (Kv7) K current of both models (#85).

Off by default: without it neither model's state, keys or right-hand side
changes (the golden traces and the bitwise Rust tests pin the bits; here the
shapes and keys). On: the numpy and Rust right-hand sides agree to rtol
1e-12 (bit for bit on darwin-arm64, as tests/test_fast.py's), the analytic
Jacobian pattern holds every entry the right-hand side has, and native runs
agree with exact ones.
"""

import math
import platform
import sys
import warnings

import numpy as np
import pytest
from scipy import sparse

from msoaxon import _common, _fast, constants as C, multi, synaptic, two

BITWISE = sys.platform == "darwin" and platform.machine() == "arm64"
needs_ext = pytest.mark.skipif(
    not _fast.available(), reason="needs the fast extra (msoaxon_fast)"
)
# Goldwyn et al 2019's passive calibration, the starting point of Jing et al
# 2025's model, and their deprivation scenario on it
GOLDWYN = {"r1": 8.5, "tau_est": 0.34, "gKHT": 0.0}
DEPRIVED = {"klt_to_m": True, "axon_cap_scale": 1.5, "axon_leak_scale": 1.5}


def km_tau(v: float) -> float:
    """km.mod's 1 / (a + b) at 35 C, from its two rates as written."""
    x = v + 30.0
    a = 0.001 * x / (1 - math.exp(-x / 9))
    b = -0.001 * x / (1 - math.exp(x / 9))
    return 2.3 ** ((23 - 35) / 10) / (a + b)


# --- the gate ----------------------------------------------------------------


def test_the_gate_is_km_mod_and_jing_s_tau_at_their_rest():
    # Jing et al 2025: tau_u = 12 ms at their -58 mV rest; Kuba et al 2015:
    # V-half -30 mV, slope 9 mV
    assert C.tauu(-58.0) == pytest.approx(12.02, abs=0.01)
    assert C.uinf(-30.0) == 0.5
    assert C.uinf(-21.0) == pytest.approx(1 / (1 + math.exp(-1)))
    for v in (-90.0, -58.0, -45.0, -29.0, 0.0, 30.0):
        assert C.tauu(v) == pytest.approx(km_tau(v), rel=1e-12)


def test_tauu_is_continuous_through_its_removable_point():
    at = C.tauu(-30.0)
    assert math.isfinite(at)
    for d in (1e-9, 1e-6, 1e-3):
        assert C.tauu(-30.0 + d) == pytest.approx(at, rel=1e-6)
        assert C.tauu(-30.0 - d) == C.tauu(-30.0 + d)  # even in V + 30
    V = np.array([-30.0, -58.0, -1e4, 1e4])
    tau = C.tauu(V)
    assert np.all(np.isfinite(tau))
    # the array and the scalar branch agree bit for bit
    np.testing.assert_array_equal(tau, [C.tauu(float(v)) for v in V])


def test_the_rate_factor_divides_tauu():
    assert C.tauu(-58.0, q=2.0) == pytest.approx(C.tauu(-58.0) / 2)
    assert C.tauu(-58.0, q=1.0) == C.tauu(-58.0)


# --- off by default ----------------------------------------------------------


def test_multi_without_m_keeps_seven_blocks_and_its_key():
    mem = multi.membrane(-68.0)
    assert mem.g_m is None
    assert mem.blocks == 7
    assert multi.jac_sparsity(multi.LUMPED, mem) is multi.LUMPED.jac
    assert multi.initial_state(multi.LUMPED, mem, -68.0).size == 7 * multi.N
    assert multi.membrane(-68.0, ais_m_density=0.0).key == mem.key
    # unit AIS scales return the morphology itself
    assert multi.membrane(-68.0, ais_cap_scale=1.0).morph is multi.LUMPED


def test_two_without_m_keeps_eleven_states():
    P = two.get_params(-68.0, 3, 1, "active-full")
    assert (P.gM1, P.gM2, P.IM0_1, P.IM0_2, P.has_m) == (0, 0, 0, 0, False)
    assert two.initial_state(P).size == 11
    assert len(two._rhs(0.0, two.initial_state(P), P, "none", None)) == 11


def test_two_unit_scales_change_no_parameter():
    a = two.get_params(-58.0, 3, 1, "active-full", **GOLDWYN)
    b = two.get_params(
        -58.0,
        3,
        1,
        "active-full",
        axon_cap_scale=1.0,
        axon_leak_scale=1.0,
        gM1=0.0,
        **GOLDWYN,
    )
    assert vars(a) == vars(b)


# --- on: the models ----------------------------------------------------------


def m_membrane(**kw) -> multi.Membrane:
    """KLT swapped for M at the AIS, M at the nodes, AIS +50% C and leak."""
    return multi.membrane(
        -68.0,
        ais_klt_scale=0.0,
        ais_m_density=1.55,
        node_m_density=0.3,
        soma_m_density=0.01,
        ais_cap_scale=1.5,
        ais_leak_scale=1.5,
        **kw,
    )


def test_multi_with_m_has_an_eighth_block_and_its_own_keys():
    mem = m_membrane()
    n = multi.N
    assert mem.blocks == 8
    labels = np.array(multi.LUMPED.labels)
    np.testing.assert_array_equal(mem.g_m[labels == "AIS"], 1.55)
    np.testing.assert_array_equal(mem.g_m[labels == "node"], 0.3)
    assert mem.g_m[0] == 0.01
    np.testing.assert_array_equal(mem.g_m[labels == "internode"], 0.0)
    assert mem.key[-1] == ("kv7", 0.01, 1.55, 0.3)
    assert mem.morph.key == ("ais", "lumped", 1.5, 1.5)
    ais = labels == "AIS"
    np.testing.assert_array_equal(mem.morph.cap[ais], multi.CAP[ais] * 1.5)
    np.testing.assert_array_equal(mem.morph.cap[~ais], multi.CAP[~ais])
    np.testing.assert_array_equal(mem.morph.g_lk[ais], C.G_LK[ais] * 1.5)
    x0 = multi.initial_state(mem.morph, mem, -68.0)
    assert x0.size == 8 * n
    np.testing.assert_array_equal(x0[7 * n :], C.uinf(-68.0))
    assert multi.jac_sparsity(mem.morph, mem).shape == (8 * n, 8 * n)


def test_multi_m_rebalanced_rest_is_a_steady_state():
    mem = m_membrane(rebalance_rest=True)
    x0 = multi.initial_state(mem.morph, mem, -68.0)
    dx = multi._rhs(0.0, x0, "none", None, 1, (True,) * 6, mem, mem.morph)
    assert np.abs(dx).max() < 1e-9
    np.testing.assert_array_equal(x0[7 * multi.N :], C.uinf(-68.0))


def test_multi_m_rebalanced_rest_with_a_separate_axon_rest():
    # the root-found rest (_solve_rest) carries the M gate too
    mem = m_membrane(rebalance_rest=True, axon_rest=-70.0)
    assert mem.v_rest is not None
    x0 = multi.initial_state(mem.morph, mem, -68.0)
    dx = multi._rhs(0.0, x0, "none", None, 1, (True,) * 6, mem, mem.morph)
    assert np.abs(dx).max() < 1e-9


def test_multi_m_dv_is_rhs_voltage_block():
    mem = m_membrane(rebalance_rest=True)
    syn = synaptic.SynParams(t_end=20.0)
    s = _common.stimulus("EPSGpair", 5.0, 5.2, 60.0, 20.0, syn)
    x = states(mem.morph.n, 8, np.random.default_rng(1), 1)[1]
    args = ("EPSGpair", s, 1)
    full = multi._rhs(5.3, x, *args, (True,) * 6, mem, mem.morph)
    dv = multi._dv(5.3, x, *args, mem, mem.morph)
    np.testing.assert_array_equal(dv, full[: multi.N])


def test_multi_u_moves_with_w_only():
    mem = m_membrane()
    x = states(mem.morph.n, 8, np.random.default_rng(2), 1)[1]
    n = multi.N
    frozen = multi._rhs(
        1.0, x, "none", None, 1, multi.ACTIVE_GATES["active-H"], mem, mem.morph
    )
    assert np.all(frozen[7 * n :] == 0.0)
    moving = multi._rhs(1.0, x, "none", None, 1, (True,) * 6, mem, mem.morph)
    np.testing.assert_array_equal(
        moving[7 * n :], (C.uinf(x[:n]) - x[7 * n :]) / C.tauu(x[:n])
    )


def test_multi_jacobian_pattern_holds_every_m_entry():
    # central differences of _rhs: every entry that moves is in the pattern
    mem = m_membrane(rebalance_rest=True)
    morph = mem.morph
    n = morph.n
    pattern = sparse.csc_matrix(multi.jac_sparsity(morph, mem)).toarray() != 0
    x = states(n, 8, np.random.default_rng(4), 1)[1]
    active = (True,) * 6
    for j in [0, 1, 2, 4, 7 * n + 1, 7 * n + 2, 7 * n + 4, 4 * n + 2]:
        up, down = x.copy(), x.copy()
        up[j] += 1e-6
        down[j] -= 1e-6
        diff = multi._rhs(0.0, up, "none", None, 1, active, mem, morph) - (
            multi._rhs(0.0, down, "none", None, 1, active, mem, morph)
        )
        moved = np.flatnonzero(diff != 0)
        assert pattern[moved, j].all(), (j, moved[~pattern[moved, j]])


def test_multi_m_rejects_bad_settings():
    for kw in (
        {"ais_m_density": -1.0},
        {"node_m_density": math.nan},
        {"soma_m_density": math.inf},
        {"ais_cap_scale": 0.0},
        {"ais_leak_scale": -1.0},
        {"ais_cap_scale": math.nan},
    ):
        with pytest.raises(ValueError, match=next(iter(kw))):
            multi.membrane(-68.0, **kw)


def test_two_klt_to_m_swaps_the_axon_klt_only():
    P0 = two.get_params(-58.0, 3, 1, "active-full", **GOLDWYN)
    P = two.get_params(-58.0, 3, 1, "active-full", **GOLDWYN, **DEPRIVED)
    assert P.has_m
    assert (P.gKLT2, P.IKLT0_2, P.gM1) == (0.0, 0.0, 0.0)
    assert P.gM2 == P0.gKLT2
    assert P.gKLT1 == P0.gKLT1  # the soma keeps its KLT
    assert P.glk2 == P0.glk2 * 1.5
    assert P.cap2 == P0.cap2 * 1.5
    want = P.gM2 * C.uinf(-58.0) * (-58.0 - P.EK)
    assert want == P.IM0_2
    explicit = two.get_params(
        -58.0, 3, 1, "active-full", **GOLDWYN, **DEPRIVED, gM2=12.0
    )
    assert explicit.gM2 == 12.0


def test_two_m_needs_a_model_type_with_klt():
    for mt in ("passive", "active-H", "Active-sodium"):
        P = two.get_params(-68.0, 3, 1, mt, gM1=5.0, gM2=5.0, klt_to_m=True)
        assert not P.has_m, mt


def test_two_m_rest_is_a_steady_state_and_dv_matches():
    P = two.get_params(
        -58.0, 3, 1, "active-full", gM1=20.0, **GOLDWYN, **DEPRIVED
    )
    x0 = two.initial_state(P)
    assert x0.size == 13
    assert np.abs(two._rhs(0.0, x0, P, "none", None)).max() < 1e-12
    syn = synaptic.SynParams(t_end=20.0)
    s = _common.stimulus("EPSGpair", 5.0, 5.2, 40.0, 20.0, syn)
    x = x0 + np.linspace(-0.05, 0.05, 13)
    np.testing.assert_array_equal(
        two._dv(5.3, x, P, "EPSGpair", s),
        two._rhs(5.3, x, P, "EPSGpair", s)[:2],
    )


def test_two_m_rejects_bad_settings():
    for kw, match in (
        ({"gM1": -1.0}, "gM1"),
        ({"gM2": math.nan}, "gM2"),
        ({"axon_cap_scale": 0.0}, "axon_cap_scale"),
        ({"axon_leak_scale": math.inf}, "axon_leak_scale"),
        ({"klt_to_m": True, "klt_to_leak": "chord"}, "not both"),
    ):
        with pytest.raises(ValueError, match=match):
            two.get_params(-68.0, 3, 1, "active-full", **kw)


# --- on: numpy against Rust --------------------------------------------------


def states(n: int, blocks: int, rng, count: int) -> list[np.ndarray]:
    """Rest-like and random states: V in -90..40 mV, gates in 0..1.

    One random state puts compartment 2 at -30 mV exactly, where tauu's
    formula is 0 / 0.
    """
    g = blocks - 1
    out = [np.concatenate([np.full(n, -68.0), np.full(g * n, 0.3)])]
    for _ in range(count):
        x = np.concatenate([rng.uniform(-90, 40, n), rng.random(g * n)])
        x[2] = -30.0
        out.append(x)
    return out


def assert_same(got, want, blocks):
    """Bit for bit on darwin-arm64, else rtol 1e-12 per block of states.

    Each block (`blocks`, a list of slices) gets an absolute tolerance
    from its own scale, as in tests/test_fast.py, and at least 1e-15 /ms:
    at an exact rest a gate's derivative is 0 in numpy and a last bit of
    exp (1e-16 for KLT's w at -58 mV on x86-64 Linux) in Rust.
    """
    if BITWISE:
        np.testing.assert_array_equal(got, want)
        return
    for block in blocks:
        g, w = got[block], want[block]
        atol = max(1e-12 * float(np.max(np.abs(w))), 1e-15)
        np.testing.assert_allclose(g, w, rtol=1e-12, atol=atol)


MULTI_M = {
    "AIS swap, step": ({}, "step", 5.0, 50.0, 800.0, 1),
    "AIS swap, EPSG pair at node 3": ({}, "EPSGpair", 5.0, 5.2, 60.0, 3),
    "rebalanced, EPSG": ({"rebalance_rest": True}, "EPSG", 5.0, 15.0, 50.0, 1),
    "30 C": ({"temperature": 30.0}, "EPSGpair", 5.0, 5.2, 60.0, 1),
}


@needs_ext
@pytest.mark.parametrize("name", sorted(MULTI_M))
@pytest.mark.parametrize("model_type", ["active-full", "active-H"])
def test_rust_multi_rhs_matches_numpy_with_m(name, model_type):
    kw, stim, start, stop, amp, k = MULTI_M[name]
    mem = m_membrane(**kw)
    morph = mem.morph
    syn = synaptic.SynParams(t_end=20.0)
    s = _common.stimulus(stim, start, stop, amp, 20.0, syn)
    active = tuple(bool(g) for g in multi.ACTIVE_GATES[model_type])
    args = (stim, s, k, active, mem, morph, None)
    fast = _fast._build(*args)
    rng = np.random.default_rng(21)
    for t in (1.0, 5.1, 5.3, 12.0):
        for x in states(morph.n, 8, rng, 3):
            want = multi._rhs(t, x, *args)
            got = fast(t, x)
            assert got.shape == want.shape == (8 * morph.n,)
            n = morph.n
            assert_same(
                got, want, [slice(i * n, (i + 1) * n) for i in range(8)]
            )


@needs_ext
def test_rust_multi_rejects_a_seven_block_state_with_m():
    mem = m_membrane()
    s = _common.stimulus("none", 0, 0, 0, 20.0, synaptic.SynParams(t_end=20.0))
    fast = _fast._build("none", s, 1, (True,) * 6, mem, mem.morph, None)
    with pytest.raises(ValueError, match="expected 360"):
        fast(0.0, np.zeros(7 * multi.N))


TWO_M = {
    "deprived, step": (DEPRIVED, "step", 5.0, 50.0, 500.0, 1),
    "deprived, EPSG pair": (DEPRIVED, "EPSGpair", 5.0, 5.2, 40.0, 1),
    "both compartments, axon input": (
        {"gM1": 40.0, "gM2": 8.0},
        "EPSGpair",
        5.0,
        5.2,
        40.0,
        3,
    ),
}


@needs_ext
@pytest.mark.parametrize("name", sorted(TWO_M))
def test_rust_two_rhs_matches_numpy_with_m(name):
    kw, stim, start, stop, amp, k = TWO_M[name]
    P = two.get_params(-58.0, 3, k, "active-full", **GOLDWYN, **kw)
    assert P.has_m
    syn = synaptic.SynParams(t_end=60.0)
    s = _common.stimulus(stim, start, stop, amp, 60.0, syn)
    fast = _fast._build_two(stim, s, P)
    rng = np.random.default_rng(22)
    for t in (1.0, 5.1, 5.3, 30.0):
        xs = [two.initial_state(P)]
        for _ in range(4):
            x = np.concatenate([rng.uniform(-90, 40, 2), rng.random(11)])
            x[1] = -30.0 if len(xs) == 1 else x[1]
            xs.append(x)
        for x in xs:
            want = np.array(two._rhs(t, x, P, stim, s))
            got = fast(t, x)
            assert got.shape == want.shape == (13,)
            assert_same(got, want, [slice(0, 2), slice(2, 11), slice(11, 13)])


@needs_ext
@pytest.mark.filterwarnings("ignore::msoaxon.DriftWarning")
def test_native_matches_exact_with_m_in_both_models():
    # the golden traces' tolerance, on the soma and the compared node
    mem = m_membrane(rebalance_rest=True)
    runs = {}
    for mode in ("exact", "native"):
        t, y = multi.mso_axon(
            "EPSGpair", 5, 5.2, 70.0, 3, "active-full", 12, -68.0, 1,
            mem=mem, mode=mode,
        )  # fmt: skip
        runs[mode] = (t, y)
        assert y.shape[1] == 8 * multi.N
    for mode in ("exact", "native"):
        assert runs[mode][0][-1] == 12
    grid = np.linspace(0, 12, 400)
    for col in (0, 2):
        a = np.interp(grid, runs["exact"][0], runs["exact"][1][:, col])
        b = np.interp(grid, runs["native"][0], runs["native"][1][:, col])
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=5e-3)
    two_runs = {
        mode: two.two_cpt(
            "step",
            5,
            40,
            400.0,
            3,
            "active-full",
            50,
            -58.0,
            1,
            mode=mode,
            **GOLDWYN,
            **DEPRIVED,
        )  # fmt: skip
        for mode in ("exact", "native")
    }
    grid = np.linspace(0, 50, 1000)
    for col in (0, 1):
        a = np.interp(grid, two_runs["exact"][0], two_runs["exact"][1][:, col])
        b = np.interp(
            grid, two_runs["native"][0], two_runs["native"][1][:, col]
        )
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=5e-3)


@needs_ext
def test_self_check_is_untouched_by_the_m_gating_names():
    # the new gating names reach both Rust constructors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        differ, compared, worst = _fast.self_check(2)
    # (states that differ in any bit, states compared, worst difference):
    # how many differ depends on the platform's libm, so only the bound
    assert compared > 0 and 0 <= differ <= compared and worst < 1e-10
