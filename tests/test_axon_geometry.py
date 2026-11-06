"""multi.with_axon: the axon geometry as knobs (#81)."""

import numpy as np
import pytest

from msoaxon import multi

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


def test_ais_start_appends_a_hillock_and_keeps_indices():
    m = multi.with_axon(ais_start=5.0)
    assert m.n == 46 and not m.chain
    assert m.labels[:45] == multi.LUMPED.labels and m.labels[45] == "hillock"
    assert m.g_na[45] == 0.0 and m.g_kht[45] == 0.0
    for name in ("sa", "cap", "g_na", "g_klt", "g_lk"):
        np.testing.assert_array_equal(
            getattr(m, name)[:45], getattr(multi.LUMPED, name)
        )
    edges = set(zip(m.parent.tolist(), m.child.tolist(), strict=True))
    assert (0, 45) in edges and (45, 1) in edges and (0, 1) not in edges
    assert len(edges) == 45


@pytest.mark.parametrize(
    "kw",
    [
        {"internode_length": 0.0},
        {"diameter": -1.0},
        {"node_length": 0.0},
        {"ais_length": 0.0},
        {"ais_start": -1.0},
        {"myelin_cap": -0.1},
        {"internode_scale": np.zeros(21)},
        {"node_scale": -1.0},
        {"internode_scale": np.ones(5)},
    ],
)
def test_bad_arguments_raise(kw):
    with pytest.raises(ValueError):
        multi.with_axon(**kw)


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
