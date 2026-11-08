"""Inputs no cell or search could use raise a clear ValueError (issue #72).

Each case used to return a silent wrong answer (a window at a negative
margin, NaN conductances), or fail deep inside numpy or scipy with a message
that did not name the argument. The model runs are patched to fail the test
if they are reached, so a passing test also shows that the error arrives
before any simulation.
"""

import math
import warnings

import numpy as np
import pytest

from msoaxon import (
    _dispatch,
    _solve,
    coincidence,
    multi,
    spiking,
    synaptic,
    two,
)

BAD = [0.0, -1.0, float("nan")]


@pytest.fixture
def no_simulation(monkeypatch):
    """Fails the test if anything reaches a model run or the solver."""

    def reached(*args, **kw):
        raise AssertionError("a model run happened before the check")

    for name in ("run_model", "response", "spikes"):
        monkeypatch.setattr(_dispatch, name, reached)
    monkeypatch.setattr(_solve, "integrate", reached)


# --- coincidence: margins and the criterion ---


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_window_rejects_a_margin_that_is_not_positive(no_simulation, bad):
    with pytest.raises(ValueError, match=r"margin must be > 0"):
        coincidence.window("two", bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_window_with_events_rejects_a_margin_that_is_not_positive(
    no_simulation, bad
):
    with pytest.raises(ValueError, match=r"margin must be > 0"):
        coincidence.window("two", bad, dt_inh=-0.1)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_window_edges_rejects_a_margin_that_is_not_positive(no_simulation, bad):
    src = synaptic.PairSource()
    with pytest.raises(ValueError, match=r"margin must be > 0"):
        coincidence.window_edges("two", bad, events=src)


@pytest.mark.parametrize("bad", [-0.1, float("nan")])
def test_half_width_rejects_a_negative_margin(bad):
    with pytest.raises(ValueError, match=r"margin must be >= 0"):
        coincidence.half_width([0.0, 0.1, 0.2], [80.0, 90.0, 120.0], bad)


def test_half_width_still_takes_a_zero_margin():
    assert coincidence.half_width([0.0, 0.1], [80.0, 90.0], 0.0) == 0.0


@pytest.mark.parametrize("bad", BAD)
def test_resolve_criterion_rejects_a_float_that_is_not_positive(bad):
    with pytest.raises(ValueError, match=r"criterion must be > 0"):
        coincidence.resolve_criterion(bad, "two")


@pytest.mark.parametrize("bad", BAD)
def test_threshold_rejects_a_float_criterion_that_is_not_positive(
    no_simulation, bad
):
    with pytest.raises(ValueError, match=r"criterion must be > 0"):
        coincidence.threshold("two", 0.0, criterion=bad)


# --- with_dendrites: sizes ---


@pytest.mark.parametrize(
    "kw",
    [
        {"n_seg": 0},
        {"n_seg": -1},
        {"length": 0.0},
        {"length": -10.0},
        {"length": float("nan")},
        {"diameter": 0.0},
        {"diameter": -1.0},
    ],
)
def test_with_dendrites_rejects_sizes_that_are_not_positive(kw):
    name = next(iter(kw))
    with pytest.raises(ValueError, match=rf"{name} must be > 0"):
        multi.with_dendrites(**kw)


# --- the models: t_end, max_step, r1 ---

EPSG = ("EPSG", 5.0, 9.0, 60.0, 3, "active-full")


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("model", [multi.mso_axon, two.two_cpt])
def test_models_reject_a_t_end_that_is_not_positive(no_simulation, model, bad):
    with pytest.raises(ValueError, match=r"t_end must be > 0"):
        model(*EPSG, bad, -68.0, 1)


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("model", [multi.mso_axon, two.two_cpt])
def test_models_reject_a_max_step_that_is_not_positive(
    no_simulation, model, bad
):
    with pytest.raises(ValueError, match=r"max_step must be > 0"):
        model(*EPSG, 20.0, -68.0, 1, max_step=bad)


@pytest.mark.parametrize("bad", BAD)
def test_two_cpt_rejects_an_r1_that_is_not_positive(no_simulation, bad):
    with pytest.raises(ValueError, match=r"r1 must be > 0"):
        two.two_cpt(*EPSG, 20.0, -68.0, 1, r1=bad)


@pytest.mark.parametrize("bad", BAD)
def test_get_params_rejects_an_r1_that_is_not_positive(bad):
    with pytest.raises(ValueError, match=r"r1 must be > 0"):
        two.get_params(-68.0, 3, 1, "active-full", r1=bad)


# --- two_cpt: the calibration and the conductances must be finite (#170) ---
#
# Each used to run on past get_params and fail in the 1.12.0 rest check with
# numpy's LinAlgError ("Array must not contain infs or NaNs"), after
# RuntimeWarnings from two._rhs; none names the argument.

INF = [float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", BAD)
def test_two_cpt_rejects_a_tau_est_that_is_not_positive(no_simulation, bad):
    with pytest.raises(ValueError, match=r"tau_est must be > 0"):
        two.two_cpt(*EPSG, 6.0, -68.0, 1, mode="exact", tau_est=bad)


@pytest.mark.parametrize("bad", BAD)
def test_get_params_rejects_a_tau_est_that_is_not_positive(bad):
    with pytest.raises(ValueError, match=r"tau_est must be > 0"):
        two.get_params(-68.0, 3, 1, "active-full", tau_est=bad)


# the keyword, the value, and the message's bound: a positive infinity
# clears "> 0" and ">= 0", so it is reported as not finite; a negative
# one fails the bound first, as 0 and NaN do
NOT_FINITE = [
    ("v0", float("inf"), "finite"),
    ("v0", float("-inf"), "finite"),
    ("v0", float("nan"), "finite"),
    ("r1", float("inf"), "finite"),
    ("r1", float("-inf"), "> 0"),
    ("tau_est", float("inf"), "finite"),
    ("tau_est", float("-inf"), "> 0"),
    ("gC", float("inf"), "finite"),
    ("gC", float("-inf"), "> 0"),
    ("gC", float("nan"), "> 0"),
    ("area_ratio", float("inf"), "finite"),
    ("area_ratio", float("nan"), "> 0"),
    ("gNa2", float("inf"), "finite"),
    ("gNa2", float("-inf"), ">= 0"),
    ("gNa2", float("nan"), ">= 0"),
    ("gKHT", float("inf"), "finite"),
    ("gKHT", float("-inf"), ">= 0"),
    ("gKHT", float("nan"), ">= 0"),
]


@pytest.mark.parametrize(("name", "bad", "bound"), NOT_FINITE)
def test_get_params_rejects_a_value_that_is_not_finite(name, bad, bound):
    kw = {"v0": -68.0, name: bad}
    with pytest.raises(ValueError, match=rf"{name} must be {bound}"):
        two.get_params(kw.pop("v0"), 3, 1, "active-full", **kw)


@pytest.mark.parametrize(("name", "bad", "bound"), NOT_FINITE)
def test_two_cpt_rejects_a_value_that_is_not_finite(
    no_simulation, name, bad, bound
):
    kw = {"v0": -68.0, name: bad}
    with pytest.raises(ValueError, match=rf"{name} must be {bound}"):
        two.two_cpt(*EPSG, 6.0, kw.pop("v0"), 1, mode="exact", **kw)


@pytest.mark.parametrize(
    "kw",
    [{"tau_est": 0.0}, {"tau_est": float("nan")}, {"gNa2": float("inf")}],
    ids=["tau_est=0", "tau_est=nan", "gNa2=inf"],
)
def test_two_cpt_names_the_argument_before_any_arithmetic(no_simulation, kw):
    # issue #170's three cases: the ValueError, and no RuntimeWarning (a
    # divide by zero or an invalid value in two._rhs) ahead of it
    name = next(iter(kw))
    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError, match=rf"^{name} must be"),
    ):
        warnings.simplefilter("error")
        args = ("EPSG", 5, 10, 1.0, 3, "active-full", 6, -68.0, 1)
        two.two_cpt(*args, mode="exact", **kw)


def test_two_cpt_still_takes_goldwyns_calibration(no_simulation):
    # the checks accept every calibration in use: no_simulation fails the
    # test at the solver, so this gets exactly that far
    kw = dict(two.GOLDWYN_2019)
    with pytest.raises(AssertionError, match="a model run happened"):
        two.two_cpt(*EPSG, 6.0, kw.pop("v0"), 1, mode="exact", **kw)


# --- SynParams: EPSG kinetics ---


@pytest.mark.parametrize(
    "tau", [(0.1, 0.1), (-0.1, 0.18), (0.1, -0.18), (0.0, 0.18)]
)
def test_syn_params_rejects_equal_or_non_positive_taus(tau):
    with pytest.raises(ValueError, match=r"epsg_tau needs"):
        synaptic.SynParams(epsg_tau=tau)


def test_syn_params_takes_the_kinetics_in_use():
    for tau in (synaptic.EPSG_TAU, (0.1, 0.3), synaptic.IPSG_TAU):
        assert synaptic.SynParams(epsg_tau=tau).epsg_tau == tau


# --- matlab_round with digits > 0 ---


def test_matlab_round_is_exact_for_positive_digits():
    assert spiking.matlab_round(0.3, 1) == 0.3
    assert spiking.matlab_round(0.25, 1) == 0.3
    assert spiking.matlab_round(-0.25, 1) == -0.3
    assert spiking.matlab_round(1.2345, 2) == 1.23


def test_matlab_round_is_unchanged_for_the_digits_callers_use():
    """threshold.binary_search passes 0 and -1 only: same bits as before."""

    def old(x, digits):
        scale = 10.0**digits
        return math.copysign(math.floor(abs(x) * scale + 0.5), x) * (
            10.0**-digits
        )

    xs = np.random.default_rng(72).uniform(-5000.0, 5000.0, 2000)
    xs = np.concatenate([xs, np.arange(-200.0, 200.0, 0.5)])
    for digits in (0, -1):
        for x in xs:
            assert spiking.matlab_round(x, digits).hex() == old(x, digits).hex()
