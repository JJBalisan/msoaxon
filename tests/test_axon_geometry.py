"""multi.with_axon: the axon geometry as knobs (#81)."""

import warnings

import numpy as np
import pytest

import msoaxon
from msoaxon import _dispatch, multi

ARRAYS = (
    "sa",
    "cap",
    "g_na",
    "g_kht",
    "g_klt",
    "g_h",
    "g_lk",
    "parent",
    "child",
    "g_ax",
)


def test_defaults_match_lumped_bit_for_bit():
    m = multi.with_axon()
    assert m.n == multi.LUMPED.n
    assert m.chain and m.labels == multi.LUMPED.labels
    for name in ARRAYS:
        got, want = getattr(m, name), getattr(multi.LUMPED, name)
        assert got.dtype == want.dtype, name
        assert got.tobytes() == want.tobytes(), name
    assert m.jac is multi.LUMPED.jac


def test_unit_scales_and_held_ld_at_default_diameter_are_still_lumped():
    m = multi.with_axon(
        internode_scale=np.ones(21), node_scale=1.0, hold_ld=True
    )
    for name in ARRAYS:
        assert (
            getattr(m, name).tobytes() == getattr(multi.LUMPED, name).tobytes()
        )


def test_built_arrays_are_read_only_and_keys_distinct():
    m = multi.with_axon(diameter=1.0)
    assert not m.sa.flags.writeable
    assert m.key != multi.LUMPED.key
    assert multi.with_axon().key == multi.with_axon().key
    assert multi.with_axon(hold_ld=True).key != multi.with_axon().key


def test_internode_and_node_lengths_move_only_their_compartments():
    m = multi.with_axon(internode_length=50.0, node_length=2.0)
    inter, node = np.arange(3, 45, 2), np.arange(4, 45, 2)
    np.testing.assert_allclose(m.sa[inter], multi.LUMPED.sa[inter] / 2)
    np.testing.assert_allclose(m.sa[node], multi.LUMPED.sa[node] * 2)
    np.testing.assert_array_equal(m.sa[:3], multi.LUMPED.sa[:3])
    np.testing.assert_array_equal(m.g_na, multi.LUMPED.g_na)


def test_ais_length_splits_in_two_and_myelin_cap_sets_internodes():
    m = multi.with_axon(ais_length=30.0, myelin_cap=0.033)
    assert m.sa[2] == pytest.approx(np.pi * 15.0 * 0.66)
    inter = np.arange(3, 45, 2)
    base = multi.LUMPED.cap
    np.testing.assert_allclose(m.cap[inter], base[inter] * 3.3)
    np.testing.assert_array_equal(m.cap[np.arange(4, 45, 2)], base[4:45:2])


def test_hold_ld_scales_internode_length_with_diameter():
    held = multi.with_axon(diameter=1.32, hold_ld=True)
    free = multi.with_axon(diameter=1.32)
    inter = np.arange(3, 45, 2)
    # SA = pi L d: held is 2x longer and 2x wider, free only 2x wider
    np.testing.assert_allclose(held.sa[inter], multi.LUMPED.sa[inter] * 4)
    np.testing.assert_allclose(free.sa[inter], multi.LUMPED.sa[inter] * 2)


def test_distal_shortening_vector_is_per_pair():
    scale = np.linspace(1.0, 0.5, 21)
    m = multi.with_axon(internode_scale=scale)
    inter = np.arange(3, 45, 2)
    np.testing.assert_allclose(m.sa[inter], multi.LUMPED.sa[inter] * scale)
    np.testing.assert_array_equal(m.sa[4:45:2], multi.LUMPED.sa[4:45:2])


def test_ais_start_inserts_a_hillock_in_physical_order():
    m = multi.with_axon(ais_start=5.0)
    assert m.n == 46 and m.chain
    assert m.labels == ["soma", "hillock", *multi.LUMPED.labels[1:]]
    assert m.g_na[1] == 0.0 and m.g_kht[1] == 0.0
    for name in ("sa", "cap", "g_na", "g_kht", "g_klt", "g_h", "g_lk"):
        got, want = getattr(m, name), getattr(multi.LUMPED, name)
        assert got[0] == want[0], name
        np.testing.assert_array_equal(got[2:], want[1:], err_msg=name)
    np.testing.assert_array_equal(m.parent, np.arange(45))
    np.testing.assert_array_equal(m.child, np.arange(1, 46))


@pytest.mark.parametrize(
    "kw",
    [
        {"internode_length": 0.0},
        {"diameter": -1.0},
        {"node_length": 0.0},
        {"ais_length": 0.0},
        {"ais_start": -1.0},
        {"myelin_cap": -0.1},
        {"myelin_cap": 0.0},
        {"myelin_cap": np.inf},
        {"internode_length": np.inf},
        {"ais_start": np.inf},
        {"ais_start": np.nan},
        {"diameter": np.nan},
        {"internode_scale": np.inf},
        {"node_scale": np.nan},
        {"internode_scale": np.zeros(21)},
        {"node_scale": -1.0},
        {"internode_scale": np.ones(5)},
    ],
)
def test_bad_arguments_raise(kw):
    with pytest.raises(ValueError):
        multi.with_axon(**kw)


def _g_ax_by_hand(lengths, diams, edges):
    """Axial conductance [S] per edge from cylinders, no builder code."""
    out = []
    for a, b in edges:
        half = [
            (lengths[k] * 1e-4) / (np.pi * (diams[k] / 2) ** 2 * 1e-8)
            for k in (a, b)
        ]
        out.append((2 / multi.R_AXIAL) / (half[0] + half[1]))
    return np.array(out)


def test_g_ax_follows_the_geometry_chain():
    scale_i, scale_n = np.linspace(1, 0.5, 21), np.linspace(1, 3, 21)
    m = multi.with_axon(
        internode_length=80.0,
        node_length=2.0,
        diameter=0.8,
        ais_length=30.0,
        internode_scale=scale_i,
        node_scale=scale_n,
    )
    li, ln = 80.0 * scale_i, 2.0 * scale_n
    lengths = np.concatenate(
        [[multi.C.L[0], 15.0, 15.0], np.stack([li, ln]).T.ravel()]
    )
    # the soma and first AIS compartment are not cylinders: take the
    # effective diameter from their area, check the rest independently
    diams = np.full(45, 0.8)
    for k in (0, 1):
        diams[k] = m.sa[k] / (np.pi * lengths[k])
    want = _g_ax_by_hand(lengths, diams, [(k, k + 1) for k in range(44)])
    np.testing.assert_allclose(m.g_ax, want, rtol=1e-12)
    np.testing.assert_allclose(m.sa[3::2], np.pi * li * 0.8)
    np.testing.assert_allclose(m.sa[4::2], np.pi * ln * 0.8)
    np.testing.assert_allclose(m.sa[2], np.pi * 15.0 * 0.8)


def test_g_ax_with_hillock_edges():
    m = multi.with_axon(ais_start=5.0, ais_length=24.0, diameter=0.7)
    assert m.sa[1] == pytest.approx(np.pi * 1.64 * 5.0)
    lengths = np.concatenate(
        [[multi.C.L[0], 5.0, 12.0, 12.0], np.tile([100.0, 1.0], 21)]
    )
    diams = np.full(46, 0.7)
    diams[[0, 2]] = m.sa[[0, 2]] / (np.pi * lengths[[0, 2]])
    diams[1] = 1.64
    edges = [(k, k + 1) for k in range(45)]
    want = _g_ax_by_hand(lengths, diams, edges)
    np.testing.assert_allclose(m.g_ax, want, rtol=1e-12)


def test_ais_length_sets_ais_areas():
    short = multi.with_axon(ais_length=10.0)
    long = multi.with_axon(ais_length=40.0)
    assert short.sa[1] < multi.LUMPED.sa[1] < long.sa[1]
    assert short.sa[2] == pytest.approx(np.pi * 5.0 * 0.66)
    assert multi.with_axon(ais_start=3.0).sa[2] == multi.LUMPED.sa[1]


def test_node_diameter_and_hillock_channels():
    m = multi.with_axon(diameter=1.0, node_length=2.0, ais_start=5.0)
    np.testing.assert_allclose(m.sa[np.arange(5, 46, 2)], np.pi * 2.0 * 1.0)
    assert m.g_klt[1] == multi.C.G_KLT[1] > 0
    assert m.g_h[1] == multi.C.G_H[1] > 0
    assert m.g_lk[1] == multi.C.G_LK[1]
    assert m.cap[1] == multi.LUMPED.cap[1]


def test_membrane_dendrite_scale_rejects_hillock_morphology():
    m = multi.with_axon(ais_start=5.0)
    with pytest.raises(ValueError):
        multi.membrane(-68.0, morph=m, dendrite_klt_scale=3.0)
    mem = multi.membrane(-68.0, morph=m, ais_klt_scale=3.0)
    np.testing.assert_allclose(mem.g_klt[2:4], 3.0 * m.g_klt[2:4])
    assert mem.g_klt[1] == m.g_klt[1]  # the hillock is not the AIS
    assert mem.g_klt[0] == m.g_klt[0] and mem.g_klt[5] == m.g_klt[5]


def test_hold_ld_scales_length_with_diameter_only():
    m = multi.with_axon(internode_length=50.0, diameter=1.32, hold_ld=True)
    # L = 50 * 1.32 / 0.66 = 100 um, so L/d = 50 / 0.66, not 100 / 0.66
    assert m.sa[3] == pytest.approx(np.pi * 100.0 * 1.32)


@pytest.mark.parametrize("ais_start", [0.0, 4.0])
def test_a_spike_travels_through_non_default_geometry(ais_start):
    m = multi.with_axon(
        diameter=0.8,
        ais_length=24.0,
        ais_start=ais_start,
        internode_scale=np.linspace(1, 0.6, 21),
    )
    mem = multi.membrane(-68.0, morph=m)
    t, y = multi.mso_axon(
        "EPSGpair", 2, 2.2, 100.0, 9, "active-full", 10, -68.0, 1, mem=mem
    )
    assert y.shape == (len(t), 7 * m.n)
    assert np.isfinite(y).all()
    # the same node 4 and last node at either ais_start, whatever their number
    shift = int(ais_start > 0)
    assert m.labels[8 + shift] == "node" and m.labels[44 + shift] == "node"
    assert y[:, 8 + shift].max() > 0 and y[:, 44 + shift].max() > 0


@pytest.mark.parametrize("ais_start", [0.0, 4.0])
def test_non_default_geometry_runs_through_mso_axon(ais_start):
    m = multi.with_axon(
        diameter=0.8,
        ais_length=24.0,
        ais_start=ais_start,
        internode_scale=np.linspace(1, 0.6, 21),
    )
    mem = multi.membrane(-68.0, morph=m)
    assert mem.morph is m
    t, y = multi.mso_axon(
        "step", 2, 4, 0.0, 3, "active-full", 6, -68.0, 1, mem=mem
    )
    assert y.shape == (len(t), 7 * m.n)
    assert np.isfinite(y).all()


def test_the_last_node_of_a_hillock_chain_can_be_the_compared_compartment():
    mem = multi.membrane(-68.0, morph=multi.with_axon(ais_start=4.0))
    t, _ = multi.mso_axon(
        "step", 2, 4, 0.0, 46, "active-full", 3, -68.0, 1, mem=mem
    )
    assert t[-1] == 3
    with pytest.raises(ValueError, match="node must be"):
        multi.mso_axon(
            "step", 2, 4, 0.0, 47, "active-full", 3, -68.0, 1, mem=mem
        )
    with pytest.raises(ValueError, match="node must be"):
        multi.mso_axon("step", 2, 4, 0.0, 46, "active-full", 3, -68.0, 1)


def test_a_rebalanced_hillock_chain_holds_rest():
    m = multi.with_axon(ais_start=4.0)
    mem = multi.membrane(-68.0, morph=m, rebalance_rest=True)
    _, y = multi.mso_axon(
        "step", 2, 4, 0.0, 3, "active-full", 20, -68.0, 1, mem=mem
    )
    assert np.abs(y[:, 0] + 68.0).max() < 1e-3


def test_fast_mode_matches_exact_on_a_hillock_chain():
    pytest.importorskip("msoaxon_fast", reason="needs the fast extra")
    mem = multi.membrane(-68.0, morph=multi.with_axon(ais_start=4.0))
    args = ("EPSGpair", 2, 2.2, 100.0, 9, "active-full", 8, -68.0, 1)
    t0, y0 = msoaxon.mso_axon(*args, mem=mem, mode="exact")
    t1, y1 = msoaxon.mso_axon(*args, mem=mem, mode="fast")
    np.testing.assert_allclose(
        np.interp(t0, t1, y1[:, 10]), y0[:, 10], rtol=1e-2, atol=0.5
    )


def test_the_default_site_on_a_hillock_chain_warns():
    m = multi.with_axon(ais_start=4.0)
    args = ("step", 2, 4, 0.0)
    with pytest.warns(UserWarning, match="proximal AIS"):
        _dispatch.run_model("multi", *args, 3, 3, -68.0, 1, morph=m)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _dispatch.run_model("multi", *args, 4, 3, -68.0, 1, morph=m)
        _dispatch.run_model("multi", *args, 3, 3, -68.0, 1)
