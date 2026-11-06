"""Stimulus branches, input validation and synaptic input.

Also the threshold search, plotting and the combine_all script.
"""

import json
import math
import pathlib
import runpy
import subprocess
import sys

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pytest

import msoaxon
from msoaxon import (
    _bisect,
    _dispatch,
    _parallel,
    _solve,
    coincidence,
    plotting,
    spiking,
    synaptic,
    threshold as threshold_mod,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
Q = 150 / 128  # EPSGpair search grid

# (stimType, start, stop, I) that make both models fire at node 3
FIRING = {
    "step": (5, 10, 8500),
    "ramp": (5, 5.5, 8500),
    "ramp2": (5, 5.5, 8500),
    "sine": (5, 7.5, 8500),
    "EPSG": (5, 10, 108),
    "EPSGpair": (5, 5, 100),
}


@pytest.mark.parametrize("model", [msoaxon.two_cpt, msoaxon.mso_axon])
@pytest.mark.parametrize("stim", sorted(FIRING))
def test_every_stimulus_fires(model, stim):
    start, stop, I = FIRING[stim]
    t, x = model(stim, start, stop, I, 3, "active-full", 20, -68, 1)
    axon = 1 if model is msoaxon.two_cpt else 2
    assert np.isfinite(x).all() and t[0] == 0 and t[-1] == 20
    assert (x[:, axon] - x[:, 0]).max() > 30


@pytest.mark.parametrize("model", [msoaxon.two_cpt, msoaxon.mso_axon])
@pytest.mark.parametrize("stim", ["Synaptic", "SynapticPair"])
def test_synaptic_inputs_depolarise(model, stim):
    syn = msoaxon.SynParams(t_end=20, random_in=1396, diff=1.0)
    _, x = model(stim, 5, 15, 0, 3, "active-full", 20, -68, 1, syn)
    assert np.isfinite(x).all()
    assert x[:, 0].max() > -67  # noise-driven EPSPs lift the soma above rest


def test_synaptic_pair_near_t_end_stays_finite():
    # t + diff runs past the conductance table; MATLAB's interp1q gave NaN here
    syn = msoaxon.SynParams(t_end=20, diff=3.0)
    for model in (msoaxon.two_cpt, msoaxon.mso_axon):
        _, x = model(
            "SynapticPair", 5, 20, 0, 3, "active-full", 20, -68, 1, syn
        )
        assert np.isfinite(x).all()


def test_synaptic_conductance_shape_and_seed():
    t, g = synaptic.synaptic(msoaxon.SynParams(t_end=20, random_in=7))
    assert len(t) == len(g) == 2001 and np.isclose(t[-1], 20)
    assert (g >= 0).all() and g.max() > 0
    np.testing.assert_array_equal(
        g, synaptic.synaptic(msoaxon.SynParams(t_end=20, random_in=7))[1]
    )
    assert not np.array_equal(
        g, synaptic.synaptic(msoaxon.SynParams(t_end=20, random_in=8))[1]
    )


def test_input_to_axon_compartment():
    # input_node != 1 routes the 2-CPT input into compartment 2
    _, x = msoaxon.two_cpt("EPSGpair", 5, 5, 100, 3, "active-full", 20, -68, 3)
    assert x[:, 1].max() > x[:, 0].max()


@pytest.mark.parametrize(
    "model,node,input_node",
    [
        (msoaxon.two_cpt, 1, 1),  # node 1 is the soma; would read coupling[-1]
        (msoaxon.two_cpt, 46, 1),
        (msoaxon.two_cpt, 3, 0),
        (msoaxon.mso_axon, 3, 0),  # would inject into compartment 45
        (msoaxon.mso_axon, 3, 46),
    ],
)
def test_out_of_range_nodes_are_rejected(model, node, input_node):
    with pytest.raises(ValueError, match="node"):
        model("EPSGpair", 5, 5, 100, node, "active-full", 20, -68, input_node)


@pytest.mark.parametrize("model", [msoaxon.two_cpt, msoaxon.mso_axon])
def test_unknown_names_are_rejected(model):
    with pytest.raises(ValueError, match="model type"):
        model("EPSGpair", 5, 5, 100, 3, "active-ful", 20, -68, 1)
    with pytest.raises(ValueError, match="stimType"):
        model("epsgpair", 5, 5, 100, 3, "active-full", 20, -68, 1)


def test_every_model_type_runs():
    for mt in (
        "passive",
        "active-KLT",
        "active-H",
        "active-KLT+H",
        "active-sodium",
        "active-KHT",
        "active-full",
    ):
        for model in (msoaxon.two_cpt, msoaxon.mso_axon):
            _, x = model("EPSG", 5, 10, 50, 3, mt, 10, -68, 1)
            assert np.isfinite(x).all(), (model.__name__, mt)


def test_spiking_reset_carries_across_columns_mult():
    # column 2 ends above threshold; with the carried-over reset, column 3's
    # first excursion is not marked until its difference drops to <= 0
    x = np.zeros((4, 315))
    x[:, 1] = [0, 50, 50, 50]
    x[:, 2] = [50, 50, 0, 50]
    s = spiking.spiking(x, 30, "Mult")
    np.testing.assert_array_equal(s[:, 1], [0, 1, 0, 0])
    np.testing.assert_array_equal(s[:, 2], [0, 0, 0, 1])


@pytest.mark.parametrize(
    ("model", "ncols"),
    [("two", 2), ("Two", 2), ("multi", 45), ("Mult", 45)],
)
def test_spiking_model_names(model, ncols):
    x = np.zeros((4, 315))
    assert spiking.spiking(x, 30, model).shape == (4, ncols)


@pytest.mark.parametrize("model", ["multi ", "mutli", "TWO", "", "45"])
def test_spiking_rejects_unknown_model_names(model):
    with pytest.raises(ValueError, match=repr(model)):
        spiking.spiking(np.zeros((4, 315)), 30, model)


# EPSGpair thresholds on the grid of the repo's EPSGpair_Thresholds_CPT*.jpg
# (making_threshold_graphs.py --only EPSGpair --jpg-grid), delays 0-0.6 ms.
# The halving search only returns multiples of Q, so its output is pinned
# exactly: the closest true threshold to any probe is 8.5e-4 relative away,
# against ~1e-14 drift across platforms and modes. The jpg column is the
# MATLAB value read off the plot, kept as history; 20 of 28 match and the
# other 8 are one output (2Q) away (PYTHON.md, "Validation").
# node, model, delay [ms], threshold / Q, jpg / Q, match
JPG_GRID = [
    (3, "multi", 0.0, 41, 41, True),
    (3, "multi", 0.1, 43, 43, True),
    (3, "multi", 0.2, 47, 47, True),
    (3, "multi", 0.3, 55, 55, True),
    (3, "multi", 0.4, 71, 71, True),
    (3, "multi", 0.5, 81, 81, True),
    (3, "multi", 0.6, 81, 83, False),  # the plateau, 94.9 vs 97.3
    (3, "two", 0.0, 41, 41, True),
    (3, "two", 0.1, 43, 41, False),
    (3, "two", 0.2, 47, 45, False),
    (3, "two", 0.3, 55, 55, True),
    (3, "two", 0.4, 75, 73, False),
    (3, "two", 0.5, 81, 81, True),
    (3, "two", 0.6, 81, 81, True),
    (5, "multi", 0.0, 39, 39, True),
    (5, "multi", 0.1, 39, 41, False),
    (5, "multi", 0.2, 43, 43, True),
    (5, "multi", 0.3, 51, 51, True),
    (5, "multi", 0.4, 65, 65, True),
    (5, "multi", 0.5, 77, 77, True),
    (5, "multi", 0.6, 77, 77, True),
    (5, "two", 0.0, 37, 37, True),
    (5, "two", 0.1, 39, 39, True),
    (5, "two", 0.2, 41, 43, False),  # one step low at three delays
    (5, "two", 0.3, 49, 51, False),
    (5, "two", 0.4, 67, 69, False),
    (5, "two", 0.5, 75, 75, True),
    (5, "two", 0.6, 75, 75, True),
]


def test_jpg_grid_table_records_the_matlab_comparison():
    assert len(JPG_GRID) == 28
    assert all(match == (q == jpg) for *_, q, jpg, match in JPG_GRID)
    assert sum(match for *_, match in JPG_GRID) == 20
    assert all(abs(q - jpg) in (0, 2) for *_, q, jpg, _ in JPG_GRID)


@pytest.mark.slow
@pytest.mark.parametrize("node", [3, 5])
def test_epsgpair_jpg_grid_thresholds(node):
    script = runpy.run_path(
        str(REPO / "scripts" / "making_threshold_graphs.py")
    )
    _, kw = script["epsgpair_grid"](jpg_grid=True)
    _, factor, max_I, *_ = script["SWEEPS"]["EPSGpair"]
    rows = [r for r in JPG_GRID if r[0] == node]
    n = len(rows) // 2  # the jpg grid's first 7 points
    multi, two = threshold_mod.binary_search(
        "EPSGpair", n, node, factor, max_I, workers=4, **kw
    )
    got = [
        (node, model, round(d, 1), v / Q)
        for model, values in (("multi", multi), ("two", two))
        for d, v in zip(
            (
                threshold_mod.sweep_x("EPSGpair", i, kw["epsg_pair_dt"])
                for i in range(1, n + 1)
            ),
            values,
            strict=True,
        )
    ]
    assert got == [r[:4] for r in rows]


def test_binary_search_rounding_and_ceiling():
    multi, two = threshold_mod.binary_search("step", 1, 3, 30, 15000, workers=1)
    assert all(v % 10 == 0 and 0 < v < 15000 for v in multi + two)
    # a stimulus that never fires reports the ceiling, as BinarySearch.m does
    multi, _ = threshold_mod.binary_search("EPSG", 1, 3, 30, 15000, workers=1)
    assert multi == [15000]


@pytest.fixture(scope="module")
def runs():
    syn = msoaxon.SynParams(t_end=20)
    args = ("EPSGpair", 5, 5, 100)
    t1, y = msoaxon.mso_axon(*args, 3, "active-full", 20, -68, 1, syn)
    t2, x = msoaxon.two_cpt(*args, 3, "active-full", 20, -68, 1, syn)
    t3, z = msoaxon.two_cpt(*args, 5, "active-full", 20, -68, 1, syn)
    graph = {
        "node": 3,
        "tEnd": 20,
        "stimType": "EPSGpair",
        "inputNode": 1,
        "start": 5,
        "stop": 5,
        "I": 100,
        "Syn": syn,
    }
    return graph, (t1, y, t2, x, t3, z)


@pytest.mark.parametrize(
    "kind",
    [
        "Contour",
        "ModelComparison",
        "ModelComparison2",
        "NodeComparison",
        "MultiCompartment",
        "Input",
        "Multigraph",
    ],
)
def test_graph_types_render(runs, kind):
    graph, data = runs
    fig = plotting.graphing({**graph, "Type": kind}, *data)
    assert fig.axes
    assert fig.get_size_inches()[0] == plotting.WIDTHS["double"]
    fig.canvas.draw()  # resolves tick labels, so every text has its size
    assert min(t.get_fontsize() for t in fig.findobj(mpl.text.Text)) >= 6
    mpl.pyplot.close(fig)


def test_two_compartment_graph_and_unknown_type(runs):
    graph, (_, _, t2, x, _, _) = runs
    mpl.pyplot.close(
        plotting.graphing({**graph, "Type": "TwoCompartment"}, t2, x)
    )
    with pytest.raises(ValueError):
        plotting.graphing({**graph, "Type": "Nope"}, t2, x)
    fig = plotting.threshold_figure([0, 1], [1, 2], [1, 2], "t", "x")
    assert fig.get_size_inches()[0] == plotting.WIDTHS["single"]
    mpl.pyplot.close(fig)


def test_style_file_sets_paper_defaults():
    plotting.use_style()
    rc = mpl.rcParams
    assert rc["savefig.dpi"] == 300
    assert rc["pdf.fonttype"] == 42  # text stays text in the PDF
    assert rc["figure.constrained_layout.use"]
    assert not rc["axes.spines.top"] and not rc["axes.spines.right"]
    assert min(rc["xtick.labelsize"], rc["legend.fontsize"]) >= 6


@pytest.mark.parametrize(
    "width, inches", [("single", 3.4), ("double", 7.0), (5, 5.0)]
)
def test_figure_widths(width, inches):
    fig = plotting.figure(width)
    assert tuple(fig.get_size_inches()) == (inches, 0.75 * inches)
    mpl.pyplot.close(fig)
    fig = plotting.figure(width, height=2.0)
    assert fig.get_size_inches()[1] == 2.0
    mpl.pyplot.close(fig)


@pytest.mark.parametrize("name", ["out", "out.png", "out.pdf"])
def test_save_writes_png_and_pdf(tmp_path, name):
    fig = plotting.figure("single")
    fig.subplots().plot([0, 1], [0, 1])
    paths = plotting.save(fig, tmp_path / name)
    assert paths == [tmp_path / "out.png", tmp_path / "out.pdf"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert not mpl.pyplot.fignum_exists(fig.number)  # save closes it
    assert mpl.image.imread(paths[0]).shape[1] == round(3.4 * 300)


def test_save_svg_on_request(tmp_path):
    paths = plotting.save(plotting.figure(), tmp_path / "f", svg=True)
    assert [p.suffix for p in paths] == [".png", ".pdf", ".svg"]
    assert all(p.exists() for p in paths)


def test_units_glyphs():
    assert plotting.units("Delay", "us") == "Delay (\N{MICRO SIGN}s)"
    assert (
        plotting.units("Rin", "MOhm") == "Rin (M\N{GREEK CAPITAL LETTER OMEGA})"
    )
    assert plotting.units("Threshold", "pA") == "Threshold (pA)"


def test_panel_label_and_threshold_ratio_panel():
    fig = plotting.figure()
    ax = fig.subplots()
    th = np.array([40.0, 44.0, 80.0])
    plotting.threshold_ratio_panel(
        ax,
        [0.0, 0.1, 0.2],
        [(th, {"label": "a"}), (2 * th, {"ls": "--"})],
        margins=(0.005, 0.03),
    )
    plotting.panel_label(ax, "E")
    curves, hlines = ax.lines[:2], ax.lines[2:]
    for line in curves:  # each curve over its own zero-delay threshold
        np.testing.assert_allclose(line.get_xdata(), [0, 100, 200])
        np.testing.assert_allclose(line.get_ydata(), th / 40)
    assert curves[1].get_linestyle() == "--"
    assert [h.get_ydata()[0] for h in hlines] == [1.005, 1.03]
    assert ax.get_ylim() == (0.98, 2.05)
    assert "E" in [t.get_text() for t in ax.texts]
    mpl.pyplot.close(fig)


def test_combine_all_script(tmp_path):
    out = tmp_path / "fig.png"
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "combine_all.py"),
            "--stim",
            "EPSGpair",
            "--I",
            "100",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg", "PATH": ""},
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert (
        "multi-compartment 1, two-compartment 1" in r.stdout
        and out.stat().st_size > 0
        and out.with_suffix(".pdf").stat().st_size > 0
    )
    # the JSON redraws it without simulating (#139)
    data = json.loads(out.with_suffix(".json").read_text())
    assert data["graph"]["stimType"] == "EPSGpair" and data["graph"]["I"] == 100
    assert len(data["y"][0]) == 45 and len(data["x"][0]) == 2
    copy = tmp_path / "replot" / "fig.json"
    copy.parent.mkdir()
    copy.write_text(out.with_suffix(".json").read_text())
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "combine_all.py"),
            "--replot",
            str(copy),
        ],
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg", "PATH": ""},
        check=False,
    )
    assert r.returncode == 0, r.stderr
    shapes = [
        mpl.image.imread(p).shape for p in (out, copy.with_suffix(".png"))
    ]
    assert shapes[0] == shapes[1]


@pytest.mark.parametrize("model", [msoaxon.two_cpt, msoaxon.mso_axon])
def test_pre_stimulus_cache_does_not_change_results(model, fresh_quiet_cache):
    t1, x1 = model("EPSGpair", 5, 5.3, 70, 3, "active-full", 20, -68, 1)
    assert _solve._QUIET_CACHE  # first run filled it
    t2, x2 = model(
        "EPSGpair", 5, 5.3, 90, 3, "active-full", 20, -68, 1
    )  # reuses it
    _solve._QUIET_CACHE.clear()
    t3, x3 = model(
        "EPSGpair", 5, 5.3, 90, 3, "active-full", 20, -68, 1
    )  # recomputes
    np.testing.assert_array_equal(t2, t3)
    np.testing.assert_array_equal(x2, x3)


@pytest.mark.parametrize(
    "model,axon", [(msoaxon.two_cpt, 1), (msoaxon.mso_axon, 2)]
)
def test_stop_on_spike_truncates_the_full_run(model, axon):
    args = ("EPSGpair", 5, 5.3, 70, 3, "active-full", 20, -68, 1)
    t, x = model(*args)
    te, xe = model(*args, stop_on_spike=10)
    assert te[-1] < 20 and np.isclose(xe[-1, axon] - xe[-1, 0], 10)
    # same accepted steps as the full run until the step where the spike is
    # found
    np.testing.assert_array_equal(te[:-1], t[: len(te) - 1])
    np.testing.assert_array_equal(xe[:-1], x[: len(te) - 1])
    # and the first sample above the factor in the full run comes right after
    first = np.argmax(x[:, axon] - x[:, 0] > 10)
    assert t[first - 1] <= te[-1] <= t[first]


def test_stop_on_spike_leaves_non_spiking_runs_alone():
    args = ("EPSGpair", 5, 5, 40, 3, "active-full", 20, -68, 1)
    t, x = msoaxon.mso_axon(*args)
    te, xe = msoaxon.mso_axon(*args, stop_on_spike=10)
    np.testing.assert_array_equal(t, te)
    np.testing.assert_array_equal(x, xe)


def test_model_modules_are_importable():
    # the re-exported functions used to shadow their own submodules
    from msoaxon import multi, two  # noqa: PLC0415 (the import is the test)

    assert multi.mso_axon is msoaxon.mso_axon and two.two_cpt is msoaxon.two_cpt


def test_parallel_runs_match_serial():

    serial = threshold_mod.binary_search(
        "EPSGpair", 2, 3, 10, 150, epsg_pair_dt=0.3, workers=1
    )
    assert (
        threshold_mod.binary_search(
            "EPSGpair", 2, 3, 10, 150, epsg_pair_dt=0.3, workers=2
        )
        == serial
    )
    curve = coincidence.threshold_curve("two", [0.0, 0.2], workers=1)
    with _parallel.process_pool(2) as pool:  # one pool reused across calls
        assert (
            threshold_mod.binary_search(
                "EPSGpair", 2, 3, 10, 150, epsg_pair_dt=0.3, executor=pool
            )
            == serial
        )
        np.testing.assert_array_equal(
            coincidence.threshold_curve("two", [0.0, 0.2], executor=pool), curve
        )


def test_smallest_firing():

    assert _bisect.smallest_firing(
        lambda x: x >= 3.0, 1.0, 100.0, 1e-6
    ) == pytest.approx(3.0, rel=1e-6)
    assert (
        _bisect.smallest_firing(lambda x: x >= 3.0, 1.0, 100.0, 1e-6) >= 3.0
    )  # returns a firing value
    assert _bisect.smallest_firing(lambda x: False, 1.0, 100.0, 1e-3) == np.inf
    # the ceiling bounds the doubling: 64 is tested, 128 exceeds 100, so 90 is
    # never reached
    assert (
        _bisect.smallest_firing(lambda x: x >= 90.0, 1.0, 100.0, 1e-3) == np.inf
    )


def test_run_model_rejects_multi_only_keywords_for_two():

    with pytest.raises(ValueError, match="mem only apply"):
        _dispatch.run_model(
            "two", "step", 5, 10, 100.0, 3, 10, -68.0, 1, mem=object()
        )
    with pytest.raises(ValueError, match='"multi" or "two"'):
        _dispatch.run_model("Two", "step", 5, 10, 100.0, 3, 10, -68.0, 1)


def _smooth_response(threshold, level=10.0):
    """A stand-in for an EPSG-pair search: the peak rises convexly and smoothly
    through `level` at `threshold`, like the models' axon - soma at 10 mV."""
    calls = []

    def respond(x):
        calls.append(x)
        peak = level * (x / threshold) ** 12
        return peak >= level, min(peak, level)

    return respond, calls


@pytest.mark.parametrize("threshold", [3.7, 47.72, 48.0, 190.0, 1500.0])
def test_smallest_crossing_keeps_bisections_guarantee(threshold):

    respond, calls = _smooth_response(threshold)
    hi = _bisect.smallest_crossing(respond, 50.0, 2000.0, 1e-4, 10.0)
    assert (
        threshold <= hi <= threshold * (1 + 1e-4)
    )  # spikes, and within rel_tol
    n_bisect = [0]

    def fires(x):
        n_bisect[0] += 1
        return x >= threshold

    _bisect.smallest_firing(fires, 50.0, 2000.0, 1e-4)
    assert len(calls) < n_bisect[0]  # and in fewer runs than bisection


def test_smallest_crossing_gives_up_past_the_ceiling():

    respond, _ = _smooth_response(5000.0)
    assert (
        _bisect.smallest_crossing(respond, 50.0, 2000.0, 1e-4, 10.0) == np.inf
    )


def test_crossing_works_in_either_direction():
    # window() searches delay, where the spiking side is the smaller value

    def respond(
        d,
    ):  # the peak falls smoothly through 10 as the delay passes 0.1
        peak = 10.0 * np.exp(-(d - 0.1) * 8)
        return peak >= 10.0, min(peak, 10.0)

    far, near = _bisect.crossing(
        respond, 1.0, 0.0, lambda n, y: 1e-4, 10.0, [(1.0, respond(1.0)[1])]
    )
    assert near <= 0.1 <= far and far - near <= 1e-4


def test_threshold_figure_draws_the_ceiling_apart():
    # the ramp sweep's first point returns the ceiling in both models (#136)
    x = [5.1, 5.2, 5.3]
    multi, two = [15000.0, 14130.0, 9990.0], [15000.0, 13710.0, 9700.0]
    fig = plotting.threshold_figure(
        x, multi, two, "Ramp", "x", ceiling=15000, resolution=10, unit="pA"
    )
    ax = fig.axes[0]
    curves = [line for line in ax.lines if line.get_label().endswith("ment")]
    for line in curves:  # every point marked, the ceiling point left out
        assert line.get_marker() == "o"
        assert np.isnan(line.get_ydata()[0])
        assert not np.isnan(line.get_ydata()[1:]).any()
    hollow = [line for line in ax.lines if line.get_markerfacecolor() == "none"]
    assert len(hollow) == 2
    for line in hollow:
        assert list(line.get_xdata()) == [5.1]
        assert list(line.get_ydata()) == [15000]
    labels = ax.get_legend_handles_labels()[1]
    assert "no threshold below 15000 pA" in labels
    assert "search resolution 10 pA" in ax.get_title()
    mpl.pyplot.close(fig)


def test_threshold_figure_without_ceiling_hits_has_no_ceiling_line():
    fig = plotting.threshold_figure(
        [1, 2], [48, 50], [48, 50], "EPSGpair", "x", ceiling=150
    )
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert labels == ["Multi-Compartment", "Two-Compartment"]
    mpl.pyplot.close(fig)


def test_ceiling_mask():
    np.testing.assert_array_equal(
        plotting.ceiling_mask([15000, 14990, 15000.0], 15000),
        [True, False, True],
    )


def test_colour_limits_have_a_minimum_span():
    # the quick coupling plane's windows: a 2 us spread around 200 us
    lo, hi = plotting.colour_limits([199.0, 200.5, 201.0])
    assert hi - lo == pytest.approx(20.05)  # 10% of the 200.5 median
    assert (lo + hi) / 2 == pytest.approx(200.0)
    # a spread wider than the minimum is kept as it is
    assert plotting.colour_limits([10.0, 50.0]) == (10.0, 50.0)
    assert plotting.colour_limits([0.0, 0.0]) == (-0.5, 0.5)


def test_wilson_interval():
    lo, hi = plotting.wilson_interval([0.0, 0.5, 1.0], 10)
    # 0 of 10: [0, 0.2775]; symmetric around 0.5; 10 of 10 mirrors 0 of 10
    np.testing.assert_allclose(lo, [0, 0.2366, 0.7225], atol=1e-4)
    np.testing.assert_allclose(hi, [0.2775, 0.7634, 1], atol=1e-4)


def test_width_strip_draws_the_widths_centred():
    fig = plotting.figure()
    ax = fig.subplots()
    plotting.width_strip(ax, [("a", 211.0, "C0"), ("b", 172.0, "C1")])
    assert [list(line.get_xdata()) for line in ax.lines] == [
        [-105.5, 105.5],
        [-86.0, 86.0],
    ]
    assert [t.get_text() for t in ax.texts] == [
        "a: 211 \N{MICRO SIGN}s",
        "b: 172 \N{MICRO SIGN}s",
    ]
    assert not ax.yaxis.get_visible()
    mpl.pyplot.close(fig)


def test_coincidence_strip_rows_are_the_three_percent_windows():
    script = runpy.run_path(str(REPO / "scripts" / "coincidence_window.py"))
    kinetics = list(script["KINETICS"])
    direct = {
        (c, k): {0.005: 1.0 + i, 0.03: 100.0 + i}
        for i, c in enumerate(script["CONFIGS"])
        for k in kinetics
    }
    rows = script["strip_rows"](direct, ["r", "g", "b"])
    assert rows == [
        ("45-compartment", 100.0, "r"),
        ("2-cpt, TwoCpt.m", 101.0, "g"),
        ("2-cpt, Goldwyn 2019", 102.0, "b"),
    ]


@pytest.mark.parametrize(
    "stim, jpg_grid, expected",
    [("ramp", False, 10), ("EPSGpair", False, 1), ("EPSGpair", True, None)],
)
def test_threshold_resolution(stim, jpg_grid, expected):
    script = runpy.run_path(
        str(REPO / "scripts" / "making_threshold_graphs.py")
    )
    assert script["resolution"](stim, jpg_grid) == expected


def test_model_colours_are_distinct_and_colour_blind_safe():
    colours = list(plotting.COLOURS.values())
    assert len(set(colours)) == len(colours)
    # the Okabe-Ito entries the scripts share (#138)
    assert plotting.COLOURS == {
        "multi": "#0072B2",
        "two": "#D55E00",
        "goldwyn": "#009E73",
    }
    assert plotting.GREY not in colours and plotting.SOMA not in colours
    assert plotting.model_style("two", 1) == {"color": "#D55E00", "ls": "--"}


def test_sequential_is_ordered_and_stable():
    lum = [
        0.299 * r + 0.587 * g + 0.114 * b
        for r, g, b, _ in plotting.sequential([1.5, 2.0, 3.0])
    ]
    assert lum == sorted(lum)  # dark to light along the parameter
    # a quick run's subset keeps the full run's colours
    full = plotting.sequential([1.5, 2.0, 3.0])
    assert plotting.sequential([1.5, 3.0], 1.5, 3.0) == [full[0], full[2]]


def test_glyphs():
    assert plotting.glyphs("FWHM 200 us; distal (180 um), 10 MOhm") == (
        "FWHM 200 \N{MICRO SIGN}s; distal (180 \N{MICRO SIGN}m), "
        "10 M\N{GREEK CAPITAL LETTER OMEGA}"
    )
    assert plotting.glyphs("bus stop") == "bus stop"


def test_graphing_colours_follow_the_models(runs):
    graph, data = runs
    fig = plotting.graphing({**graph, "Type": "Multigraph"}, *data)
    top, _bar, two3, two5, inp = fig.axes
    colours = [mpl.colors.to_hex(line.get_color()) for line in top.lines]
    assert colours[0] == "#000000"  # the soma
    assert len(set(colours)) == 9  # nine distinct, ordered node colours
    assert colours[1:] == [
        mpl.colors.to_hex(c)
        for c in plotting.sequential([3, 5, 9, 13, 21, 29, 37, 45])
    ]
    for ax in (two3, two5):
        assert [line.get_color() for line in ax.lines] == [
            plotting.SOMA,
            plotting.COLOURS["two"],
        ]
    assert inp.lines[0].get_color() == plotting.GREY
    mpl.pyplot.close(fig)
    fig = plotting.graphing({**graph, "Type": "NodeComparison"}, *data)
    for ax in fig.axes:
        assert [line.get_color() for line in ax.lines] == [
            plotting.COLOURS["multi"],
            plotting.COLOURS["two"],
        ]
    mpl.pyplot.close(fig)


def test_threshold_figure_colours_and_units():
    fig = plotting.threshold_figure(
        [1, 2], [48, 50], [48, 51], "EPSGpair", "x", unit="nS per EPSG"
    )
    ax = fig.axes[0]
    assert [line.get_color() for line in ax.lines] == [
        plotting.COLOURS["multi"],
        plotting.COLOURS["two"],
    ]
    assert ax.get_ylabel() == "Threshold (nS per EPSG)"
    mpl.pyplot.close(fig)


def test_resonance_arms_use_their_model_colour():
    script = runpy.run_path(str(REPO / "scripts" / "resonance.py"))
    assert set(script["ARM_STYLES"]) == set(script["ARMS"])
    for name, (model, _) in script["ARM_STYLES"].items():
        assert script["ARMS"][name][0] == model  # never another model's


def test_save_writes_the_replot_json(tmp_path):
    fig = plotting.figure("single")
    data = {"x": [1.0, 2.0], "nan": float("nan")}
    paths = plotting.save(fig, tmp_path / "f.png", data=data, args={"n": 1})
    assert paths[-1] == tmp_path / "f.json"
    d = plotting.load(tmp_path / "f.png")  # found beside the figure
    assert d["x"] == [1.0, 2.0] and math.isnan(d["nan"])
    assert d["meta"]["args"] == {"n": 1}
    assert d["meta"]["version"] == plotting.version() != ""
    assert d["meta"]["mode"] in ("exact", "fast", "native")
    # a replot (no data) rewrites the figure, not the JSON; .json is dropped
    paths = plotting.save(plotting.figure(), tmp_path / "f.json")
    assert paths == [tmp_path / "f.png", tmp_path / "f.pdf"]
    again = plotting.load(tmp_path / "f.json")
    assert math.isnan(again.pop("nan"))
    assert {k: v for k, v in d.items() if k != "nan"} == again


def test_version_falls_back_without_git(monkeypatch):
    plotting.version.cache_clear()
    monkeypatch.setattr(plotting.shutil, "which", lambda _: None)
    try:
        assert plotting.version() == f"msoaxon {msoaxon.__version__}"
    finally:
        plotting.version.cache_clear()


def test_compact_and_thin():
    assert plotting.compact([1.23456789, 1234567.89]) == [1.23457, 1234570.0]
    assert plotting.compact([[0.1, float("nan")]])[0][0] == 0.1
    assert plotting.thin(5000) == slice(None, None, 3)
    assert plotting.thin(100) == slice(None, None, 1)


def test_threshold_residual():
    r = plotting.threshold_residual(
        [15000, 10000, 200], [15000, 10500, 190], 15000
    )
    assert np.isnan(r[0])  # no threshold in either model: no residual
    np.testing.assert_allclose(r[1:], [5.0, -5.0])


def test_mark_minimum_marks_only_an_interior_minimum():
    fig = plotting.figure()
    ax = fig.subplots()
    assert plotting.mark_minimum(ax, [1, 2, 3, 4], [9, 4, 5, 6], "ms") == 1
    assert ax.lines[-1].get_xdata()[0] == 2
    assert ax.texts[-1].get_text() == "min 2 ms"
    assert plotting.mark_minimum(ax, [1, 2, 3], [5, 6, 7], "ms") is None
    # a ceiling point is no minimum, and neither is the end it leaves
    assert plotting.mark_minimum(ax, [1, 2, 3], [1, 2, 3], "ms", 1) is None
    mpl.pyplot.close(fig)


def _panel(title, unit, ceiling, thresholds, mark_min):
    return {
        "title": title,
        "xlabel": "x",
        "x_unit": "us" if unit == "nS per EPSG" else "ms",
        "x_scale": 1000 if unit == "nS per EPSG" else 1,
        "unit": unit,
        "ceiling": ceiling,
        "resolution": 10 if unit == "pA" else 1,
        "mark_min": mark_min,
        "x": [0.0, 0.04, 0.08],
        "thresholds": thresholds,
    }


def test_threshold_grid():
    panels = {
        "ramp": _panel(
            "Ramp",
            "pA",
            15000,
            {
                "3": {"multi": [15000, 9000, 9500], "two": [15000, 8800, 9600]},
                "5": {"multi": [12000, 6000, 7000], "two": [12000, 6100, 7100]},
            },
            True,
        ),
        "EPSGpair": _panel(
            "EPSG pair",
            "nS per EPSG",
            150,
            {
                "3": {"multi": [48, 50, 60], "two": [48, 51, 60]},
                "5": {"multi": [47, 49, 58], "two": [47, 49, 59]},
            },
            False,
        ),
    }
    fig = plotting.threshold_grid(panels, [3, 5])
    # per column: node 3, its residual, node 5, its residual
    ramp3, ramp3_res, ramp5, ramp5_res = fig.axes[:4]
    pair3 = fig.axes[4]
    assert ramp3.get_ylim() == ramp5.get_ylim()  # nodes share y limits
    assert ramp3.get_ylabel() == "Threshold (pA)"
    assert pair3.get_ylabel() == "Threshold (nS per EPSG)"
    assert ramp5_res.get_xlabel() == "x (ms)"
    assert fig.axes[7].get_xlabel() == "x (\N{MICRO SIGN}s)"
    # EPSG-pair delays drawn in us
    np.testing.assert_allclose(pair3.lines[0].get_xdata(), [0, 40, 80])
    # the two-compartment curve dashed, the 45's solid
    assert ramp3.lines[0].get_linestyle() == "-"
    assert ramp3.lines[2].get_linestyle() == "--"
    # residual strip: (two - multi) / multi in percent, none at the ceiling
    res = ramp3_res.lines[-1].get_ydata()
    assert np.isnan(res[0])
    np.testing.assert_allclose(res[1:], [-200 / 90, 100 / 95])
    # the ramp's interior minimum is marked for each model; EPSGpair is not
    assert [t.get_text() for t in ramp3.texts] == ["min 0.04 ms"]
    assert len([ln for ln in ramp3.lines if ln.get_marker() == "v"]) == 2
    assert not pair3.texts
    assert "search resolution 1 nS per EPSG, 10 pA" in fig.get_suptitle()
    mpl.pyplot.close(fig)


def test_threshold_figure_draws_two_dashed():
    fig = plotting.threshold_figure([1, 2], [48, 50], [48, 51], "t", "x")
    assert [line.get_linestyle() for line in fig.axes[0].lines] == ["-", "--"]
    mpl.pyplot.close(fig)


def test_sweeps_carry_units():
    script = runpy.run_path(
        str(REPO / "scripts" / "making_threshold_graphs.py")
    )
    sweeps = script["SWEEPS"]
    assert {s.unit for s in sweeps.values()} == {"pA", "nS per EPSG"}
    assert sweeps["EPSGpair"].x_unit == "us"  # as in the coincidence figures
    assert [k for k, s in sweeps.items() if s.mark_min] == ["ramp", "sine"]


def test_event_window():
    t = np.linspace(0, 20, 2001)
    V = np.full((len(t), 2), -68.0)
    V[t >= 5.3, 1] = -50.0  # node rises 18 mV at 5.3 ms
    assert plotting.event_window(t, V, 20) == pytest.approx((4.3, 9.3))
    # no trace rises: centred on the stimulus onset, within the run
    flat = np.full(len(t), -68.0)
    assert plotting.event_window(t, flat, 20, start=19) == (18.0, 20.0)


def test_graphing_zooms_to_the_event_and_shares_y(runs):
    graph, data = runs
    fig = plotting.graphing({**graph, "Type": "ModelComparison"}, *data)
    a1, a2 = fig.axes
    t0, t1 = a1.get_xlim()
    assert 3.5 < t0 < 5.5 and t1 - t0 == pytest.approx(5.0)  # not 0 to 20
    assert a1.get_ylim() == a2.get_ylim()  # the two models compare by eye
    assert a1.lines[0].get_zorder() > a1.lines[1].get_zorder()  # soma on top
    assert a2.get_title() == "2-compartment model, node 3"
    mpl.pyplot.close(fig)
    fig = plotting.graphing(
        {**graph, "Type": "ModelComparison"}, *data, window=None
    )
    assert fig.axes[0].get_xlim() == (0.0, 20.0)  # the whole run
    mpl.pyplot.close(fig)
    fig = plotting.graphing(
        {**graph, "Type": "ModelComparison"}, *data, window=(6, 8)
    )
    assert fig.axes[0].get_xlim() == (6.0, 8.0)
    mpl.pyplot.close(fig)


def test_contour_is_a_heat_map_and_multigraph_has_a_node_bar(runs):
    graph, data = runs
    fig = plotting.graphing({**graph, "Type": "Contour"}, *data)
    ax, bar = fig.axes
    mesh = ax.collections[0]
    assert mesh.get_array().shape == (45, len(data[0]))  # cpt x time
    assert bar.get_ylabel() == "Voltage (mV)"
    mpl.pyplot.close(fig)
    fig = plotting.graphing({**graph, "Type": "Multigraph"}, *data)
    top, bar, two3, two5, _ = fig.axes
    assert list(bar.get_yticks()) == [3, 5, 9, 13, 21, 29, 37, 45]
    assert top.get_ylim() == two3.get_ylim() == two5.get_ylim()
    assert [t.get_text() for t in top.get_legend().get_texts()] == ["Soma"]
    mpl.pyplot.close(fig)


def _script(name):
    return runpy.run_path(str(REPO / "scripts" / name))


def test_resonance_marks_the_json_numbers():
    script = _script("resonance.py")
    arm = {
        "subthreshold": {
            "resonant": True,
            "f_res": 373.2,
            "z_peak": 6.7,
            "freqs": [20.0, 373.2, 500.0],
            "z": [2.4, 6.7, 5.5],
        },
        "spike": {
            "resonant": True,
            "f_best": 475.0,
            "threshold": 3600.0,
            "freqs": [250.0, 475.0, 650.0],
            "thresholds": [5000.0, 3600.0, 4400.0],
        },
        "epsg": {"resonant": False, "f_best": 300.0, "dip": 0.99},
    }
    assert script["peak"]("subthreshold", arm["subthreshold"]) == (373.2, 6.7)
    assert script["peak"]("spike", arm["spike"]) == (475.0, 3600.0)
    assert script["peak"]("epsg", arm["epsg"]) is None
    assert script["legend_label"]("arm", arm) == (
        "arm: 373 (+131), 475 (+190), none Hz"
    )


def test_resonance_plot_marks_peaks(monkeypatch, tmp_path):
    script = runpy.run_path(
        str(REPO / "scripts" / "resonance.py"), run_name="resonance_test"
    )
    figs = []
    monkeypatch.setattr(
        plotting, "save", lambda fig, path, **kw: figs.append(fig)
    )
    curve = {
        "freqs": [250.0, 450.0, 650.0],
        "thresholds": [5000.0, 3600.0, 4400.0],
    }
    arm = {
        "subthreshold": {
            "resonant": True,
            "f_res": 350.0,
            "z_peak": 6.7,
            "freqs": [20.0, 350.0, 500.0],
            "z": [2.4, 6.7, 5.5],
        },
        "spike": {
            **curve,
            "resonant": True,
            "f_best": 450.0,
            "threshold": 3600.0,
        },
        "epsg": {
            **curve,
            "resonant": False,
            "f_best": 450.0,
            "threshold": 3600.0,
        },
    }
    script["plot"]({"45-compartment (msoAxon.m)": arm}, tmp_path / "r.png")
    sub, spike, epsg = figs[0].axes
    diamonds = [ln for ln in sub.lines if ln.get_marker() == "D"]
    assert [(ln.get_xdata()[0], ln.get_ydata()[0]) for ln in diamonds] == [
        (350.0, 6.7)
    ]
    assert not [ln for ln in epsg.lines if ln.get_marker() == "D"]
    # plain tick labels on the log axes, no 6 x 10^3
    spike.figure.canvas.draw()
    labels = [t.get_text() for t in spike.get_yticklabels(minor=True)]
    assert all("times" not in t and "^" not in t for t in labels)
    mpl.pyplot.close(figs[0])


def test_somatic_draw_aligns_and_brackets_the_amplitude():
    script = _script("somatic_spike.py")
    t = np.linspace(4.8, 8.0, 321)
    v = -68 + 40 * np.exp(-(((t - 5.5) / 0.1) ** 2))

    def case(amp):
        return {
            "rheobase_pA": 3800.0,
            "spikes": [
                {
                    "multiple": 1.5,
                    "amplitude": amp,
                    "t_inflection": 5.3,
                    "v_inflection": -50.0,
                    "trace": {"t_ms": list(t), "soma_mV": list(v)},
                }
            ],
        }

    names = list(script["CASES"])
    fig = script["draw"]({names[0]: case(22.0), names[1]: case(30.0)})
    a0, a1 = fig.axes
    assert a0.get_xlim() == a1.get_xlim() == (-1.0, 2.0)  # aligned, shared
    trace, bracket = a0.lines[0], a0.lines[1]
    np.testing.assert_allclose(trace.get_xdata()[0], 4.8 - 5.3)
    np.testing.assert_allclose(trace.get_ydata().max(), -28 + 50)
    assert list(bracket.get_ydata()) == [0, 22.0]  # the JSON's amplitude
    assert bracket.get_xdata()[0] == pytest.approx(0.2, abs=0.01)  # the peak
    # Scott et al 2005: a 17 +- 2 mV band, and lines for dendrotoxin
    band = a0.patches[0]
    assert (band.get_y(), band.get_y() + band.get_height()) == (15.0, 19.0)
    refs = [ln for ln in a1.lines if ln.get_linestyle() in ("--", ":")]
    assert sorted(ln.get_ydata()[0] for ln in refs) == [15.0, 37.0]
    mpl.pyplot.close(fig)


def test_dendrites_e_marks_the_half_windows():
    script = _script("dendrites.py")
    E = {
        "model EPSG (0.18 ms)": {"margin_0.005": 60.0, "margin_0.03": 166.0},
        "Myoga EPSG (0.3 ms)": {"margin_0.005": 70.0, "margin_0.03": 182.0},
    }
    assert [x for x, _ in script["window_marks"](E)] == [83.0, 91.0]
