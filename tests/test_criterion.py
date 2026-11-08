"""The spike criterion: 10 mV by default, or relative to the spike height.

Issue #21. Most tests stand a fake model in for _dispatch, so they check
which criterion reaches the runs without simulating anything.
"""

import numpy as np
import pytest

from msoaxon import (
    _bisect,
    _dispatch,
    _fast,
    coincidence,
    multi,
    somatic,
    synaptic,
    two,
)

GOLDWYN_KW = {k: v for k, v in two.GOLDWYN_2019.items() if k != "v0"}
TH = 40.0  # the fake model's threshold
HEIGHT = 30.0  # the fake spike height


@pytest.fixture(autouse=True)
def _empty_height_cache(monkeypatch):
    monkeypatch.setattr(coincidence, "_HEIGHTS", {})


@pytest.fixture
def factors(monkeypatch):
    """Fakes the runs; returns the criteria they were given, in order."""
    seen = []

    def response(model, stim, start, stop, amp, *args, factor, **kw):
        seen.append(factor)
        if amp >= TH and stop - start < 0.5:
            return True, factor
        return False, factor * min(amp / TH, 0.999) ** 6

    def spikes(model, stim, start, stop, amp, *args, factor, **kw):
        seen.append(factor)
        return amp >= TH

    def run_model(*args, stop_on_spike=None, **kw):
        seen.append(stop_on_spike)
        return np.array([0.0, 110.0]), np.zeros((2, 2))

    monkeypatch.setattr(_dispatch, "response", response)
    monkeypatch.setattr(_dispatch, "spikes", spikes)
    monkeypatch.setattr(_dispatch, "run_model", run_model)
    return seen


@pytest.fixture
def heights(monkeypatch):
    """Fakes spike_height(); returns the configurations it was asked for."""
    asked = []

    def spike_height(
        model,
        node=3,
        v0=-68.0,
        epsg_tau=synaptic.EPSG_TAU,
        model_kw=None,
        **site,
    ):
        asked.append(
            {"model": model, "node": node, "v0": v0, "model_kw": model_kw}
        )
        return HEIGHT

    monkeypatch.setattr(coincidence, "spike_height", spike_height)
    return asked


def _every_search(criterion=None):
    """Runs each search on the fake model, with `criterion` if given."""
    kw = {} if criterion is None else {"criterion": criterion}
    th = coincidence.threshold("two", 0.0, rel_tol=1e-3, **kw)
    coincidence.window("two", 0.03, threshold0=TH, delay_tol=0.01, **kw)
    coincidence.threshold_curve("two", [0.0, 0.1], workers=1, **kw)
    coincidence.probability_trials("two", [0.0], TH, 3, workers=1, **kw)
    somatic.rheobase("two", rel_tol=1e-2, **kw)
    somatic.spike_amplitude("two", 1.0, **kw)
    return th


def test_default_is_the_absolute_10_mv(factors, heights):
    assert coincidence.FACTOR == somatic.FACTOR == 10.0
    th = _every_search()
    assert th == pytest.approx(TH, rel=1e-3)
    assert factors and set(factors) == {10.0}
    assert not heights


def test_a_float_is_an_absolute_criterion(factors, heights):
    _every_search(12.5)
    assert set(factors) == {12.5}
    assert not heights


def test_relative_resolves_to_its_fraction_of_the_spike_height(
    factors, heights
):
    _every_search(coincidence.Relative(0.5))
    assert set(factors) == {0.5 * HEIGHT}


def test_curves_and_trials_resolve_once_not_per_task(factors, heights):
    half = coincidence.Relative(0.5)
    coincidence.threshold_curve(
        "two", [0.0, 0.1, 0.2], workers=1, criterion=half
    )
    assert len(heights) == 1
    coincidence.probability_trials(
        "two", [0.0, 0.1], TH, 4, workers=1, criterion=half
    )
    assert len(heights) == 2


def test_somatic_measures_height_on_its_own_membrane(factors, heights):
    mem = multi.membrane(-60.0)
    somatic.fires(
        "multi", 1.0, 3, -60.0, mem, criterion=coincidence.Relative(0.5)
    )
    (asked,) = heights
    assert (asked["model"], asked["node"], asked["v0"]) == ("multi", 3, -60.0)
    assert list(asked["model_kw"]) == ["mem"]
    assert asked["model_kw"]["mem"] is mem
    somatic.fires("two", 1.0, criterion=coincidence.Relative(0.5))
    assert heights[1]["model_kw"] == {}


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5])
def test_relative_needs_a_fraction_strictly_between_0_and_1(fraction):
    with pytest.raises(ValueError, match="fraction"):
        coincidence.Relative(fraction)


def test_goldwyn_at_half_its_height_is_about_10_mv(monkeypatch):
    runs = []
    run_model = _dispatch.run_model

    def counted(*args, **kw):
        runs.append(args)
        return run_model(*args, **kw)

    monkeypatch.setattr(_dispatch, "run_model", counted)
    height = coincidence.spike_height("two", v0=-58.0, model_kw=GOLDWYN_KW)
    assert 19.5 < height < 21.0  # report: 20.2 mV
    assert runs
    runs.clear()
    half = coincidence.resolve_criterion(
        coincidence.Relative(0.5), "two", v0=-58.0, model_kw=GOLDWYN_KW
    )
    assert half == pytest.approx(10.0, abs=0.5)
    assert half == 0.5 * height
    assert not runs  # cached


def test_height_key_follows_the_configuration_not_the_object():
    key = coincidence._height_key
    a = key("multi", 3, -68.0, (0.1, 0.18), {"mem": multi.membrane(-68.0)})
    b = key("multi", 3, -68.0, (0.1, 0.18), {"mem": multi.membrane(-68.0)})
    assert a == b
    others = [
        key("multi", 3, -60.0, (0.1, 0.18), {"mem": multi.membrane(-60.0)}),
        key("multi", 3, -68.0, (0.1, 0.18), {"morph": multi.with_dendrites()}),
        key("multi", 3, -68.0, (0.1, 0.3), {"mem": multi.membrane(-68.0)}),
        key("multi", 5, -68.0, (0.1, 0.18), {"mem": multi.membrane(-68.0)}),
        key("two", 3, -58.0, (0.1, 0.18), GOLDWYN_KW),
        key("multi", 3, -68.0, (0.1, 0.18), {}, stim="EPSGbilateral"),
    ]
    assert len({a, *others}) == len(others) + 1


@pytest.mark.parametrize("model", ["two", "multi"])
def test_height_key_follows_the_effective_mode(model, monkeypatch):
    # only effective() reads _ext here, so this runs without the extension too
    tau = (0.1, 0.18)
    monkeypatch.setattr(_fast, "_ext", object())
    monkeypatch.setenv(_fast.MODE_ENV, "exact")
    exact = coincidence._height_key(model, 3, -68.0, tau, {})
    assert exact == coincidence._height_key(
        model, 3, -68.0, tau, {"mode": "exact"}
    )
    monkeypatch.setenv(_fast.MODE_ENV, "fast")
    assert exact != coincidence._height_key(model, 3, -68.0, tau, {})
    # asked for fast with no extension: the runs are exact, so the key is too
    monkeypatch.setattr(_fast, "_ext", None)
    assert exact == coincidence._height_key(model, 3, -68.0, tau, {})


def test_height_key_rejects_settings_it_cannot_name():
    with pytest.raises(TypeError, match="weights"):
        coincidence._height_key("two", 3, -68.0, (0.1, 0.18), {"weights": []})


def test_height_run_is_one_unstopped_pair_at_twice_threshold(monkeypatch):
    seen = {}

    def run_model(model, stim, start, stop, amp, node, *args, **kw):
        seen.update(stim=stim, delay=stop - start, amp=amp, kw=kw)
        x = np.zeros((3, 5))
        x[:, 4] = [0.0, 25.0, 5.0]  # node 5, 1-indexed
        return np.array([0.0, 1.0, 20.0]), x

    monkeypatch.setattr(coincidence, "threshold", lambda *a, **k: TH)
    monkeypatch.setattr(_dispatch, "run_model", run_model)
    height = coincidence.spike_height(
        "multi", 5, stim="EPSGbilateral", input_node=2, input_node2=7
    )
    assert height == 25.0
    assert seen["stim"] == "EPSGbilateral" and seen["delay"] == 0.0
    assert seen["amp"] == coincidence.HEIGHT_AT * TH
    assert seen["kw"] == {"input_node2": 7}  # no stop_on_spike


def test_no_height_when_nothing_spikes(monkeypatch):
    monkeypatch.setattr(coincidence, "threshold", lambda *a, **k: np.inf)
    with pytest.raises(ValueError, match="no spike height"):
        coincidence.spike_height("two")
    assert not coincidence._HEIGHTS


def test_height_reports_a_cell_that_fires_without_input(monkeypatch):
    def fires_alone(*args, **kw):
        raise _bisect.FiresWithoutInputError("fires alone")

    monkeypatch.setattr(coincidence, "threshold", fires_alone)
    with pytest.raises(_bisect.FiresWithoutInputError, match="spike_height"):
        coincidence.resolve_criterion(coincidence.Relative(0.5), "two")
    assert not coincidence._HEIGHTS


@pytest.mark.slow
def test_relative_criterion_on_the_45_compartment_model():
    height = coincidence.spike_height("multi")
    assert 37.0 < height < 40.0  # report: 38.4 mV
    th10 = coincidence.threshold("multi", 0.0, rel_tol=1e-3)
    th50 = coincidence.threshold(
        "multi", 0.0, rel_tol=1e-3, criterion=coincidence.Relative(0.5)
    )
    assert 1.005 < th50 / th10 < 1.03  # report: +1.67%


# PYTHON.md "Spike criterion": what the searches of the criterion table cost,
# as simulations through _dispatch.run_model (the count
# scripts/benchmark_search.py reports), in exact mode, at the 10 mV default
# and at 50% of the spike height: (coincident threshold, 3% window, rheobase),
# and the spike height's own runs. The 45-compartment counts (9; 10, 14, 10
# and 12, 11, 12) are documented only: they take minutes here.
RUNS = {
    "two": {"height": 8, 10.0: (9, 11, 10), 0.5: (14, 12, 13)},
    "goldwyn": {"height": 12, 10.0: (12, 11, 8), 0.5: (12, 12, 8)},
}


@pytest.mark.slow
@pytest.mark.parametrize("config", sorted(RUNS))
def test_search_run_counts_at_both_criteria(monkeypatch, config):
    v0 = -68.0 if config == "two" else two.GOLDWYN_2019["v0"]
    kw = {"mode": "exact", **(GOLDWYN_KW if config == "goldwyn" else {})}
    counts = []
    run_model = _dispatch.run_model

    def counted(*args, **kwargs):
        counts[-1] += 1
        return run_model(*args, **kwargs)

    monkeypatch.setattr(_dispatch, "run_model", counted)

    def count(fn, *args, **kwargs):
        counts.append(0)
        value = fn(*args, **kwargs)
        return value, counts[-1]

    _, n_height = count(coincidence.spike_height, "two", v0=v0, model_kw=kw)
    assert abs(n_height - RUNS[config]["height"]) <= 1
    for level, crit in ((10.0, 10.0), (0.5, coincidence.Relative(0.5))):
        th, n_th = count(
            coincidence.threshold,
            "two",
            0.0,
            v0=v0,
            model_kw=kw,
            criterion=crit,
        )
        _, n_win = count(
            coincidence.window,
            "two",
            0.03,
            v0=v0,
            model_kw=kw,
            criterion=crit,
            threshold0=th,
        )
        _, n_rb = count(
            somatic.rheobase, "two", v0=v0, model_kw=kw, criterion=crit
        )
        expected = RUNS[config][level]
        # a last-bit difference between platforms can move one probe
        for n, e in zip((n_th, n_win, n_rb), expected, strict=True):
            assert abs(n - e) <= 1, (level, (n_th, n_win, n_rb), expected)
        assert abs(n_th + n_win + n_rb - sum(expected)) <= 2
