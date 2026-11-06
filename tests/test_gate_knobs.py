"""membrane()'s Ih, Na activation, dendritic KHT and Q10 knobs (#82).

The defaults are today's literals, and they are pinned bit for bit: the
constants' gating functions against the expressions they had before the
knobs, and a whole run against the same run through the unchanged default
membrane. The golden traces (tests/test_golden.py) pin the rest.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from msoaxon import constants as C, multi

D = multi.with_dendrites()
V = np.linspace(-95.0, 45.0, 57)
N = multi.N

ALL = (True,) * 6
STEP = ("step", 5, 10, 8500, 3)


def _rhs(mem, x, morph=None):
    morph = morph or mem.morph
    return multi._rhs(0.0, x, "none", None, 1, ALL, mem, morph)


def _gate_state(n, v=-60.0):
    return np.concatenate([np.full(n, v), np.full(6 * n, 0.4)])


def _blocks(d, n):
    return d.reshape(7, n)


def test_default_gating_functions_are_the_old_literals_bit_for_bit():
    # the expressions as they were before the rate factor and the Na
    # activation / h midpoint arguments existed
    old = {
        "minf": 1.0 / (1.0 + np.exp((V + 46.0) / -11.0)),
        "ainf": 1.0 / (1.0 + np.exp(0.1 * (V + 80.4))),
        "taum": (0.141 + (-0.0826 / (1 + np.exp((-20.5 - V) / 10.8)))) / 3,
        "tauh": (4 + (-3.74 / (1 + np.exp((-40.6 - V) / 5.05)))) / 3,
        "taup": C._P_TEMP
        * (100 / (4 * np.exp((V + 60) / 32) + 5 * np.exp(-(V + 60) / 22)) + 5),
        "tauw": 21.5
        / (6 * np.exp((V + 60) / 7) + 24 * np.exp(-(V + 60) / 50.6))
        + 0.35,
        "tauz": 170 / (5 * np.exp((V + 60) / 10) + np.exp(-(V + 70) / 8))
        + 10.7,
        "taua": C._A_TEMP * (79 + 417 * np.exp(-((V + 61.5) ** 2) / 800)),
    }
    for name, want in old.items():
        np.testing.assert_array_equal(getattr(C, name)(V), want, err_msg=name)
        if name.startswith("tau"):
            np.testing.assert_array_equal(
                getattr(C, name)(V, 1.0), want, err_msg=name
            )


def test_defaults_build_the_default_membrane():
    m = multi.membrane(-68.0)
    assert m.morph is multi.LUMPED
    assert m.morph_key == "lumped"
    assert (m.na_act_vhalf, m.na_act_k) == (46.0, -11.0)
    assert (m.ih_vhalf, m.rate, m.ih_rate) == (80.4, 1.0, 1.0)
    d = multi.membrane(-68.0, morph=D)
    assert d.morph is D
    np.testing.assert_array_equal(d.morph.g_kht[N:], 0.0)


def test_default_run_is_bit_identical_to_explicit_defaults():
    # the knobs spelt out at their defaults, and 35 C at any Q10, change no bit
    args = (*STEP, "active-full", 8, -68.0, 3)
    t0, y0 = multi.mso_axon(*args, mode="exact")
    spelt = multi.membrane(
        -68.0,
        ih_density_scale=1.0,
        ih_vhalf=-80.4,
        ih_tau_scale=1.0,
        na_act_vhalf=-46.0,
        na_act_slope=11.0,
        dendrite_kht_scale=0.0,
        temperature=35.0,
        q10=7.5,
    )
    t1, y1 = multi.mso_axon(*args, mem=spelt, mode="exact")
    np.testing.assert_array_equal(t0, t1)
    np.testing.assert_array_equal(y0, y1)


def test_ih_density_scales_gh_everywhere_and_nothing_else():
    mem = multi.membrane(-68.0, ih_density_scale=2.5)
    np.testing.assert_array_equal(mem.morph.g_h, 2.5 * multi.LUMPED.g_h)
    np.testing.assert_array_equal(mem.morph.g_kht, multi.LUMPED.g_kht)
    assert mem.morph_key == mem.morph.key != multi.LUMPED.key
    d = multi.membrane(-68.0, morph=D, ih_density_scale=0.0)
    assert not d.morph.g_h.any()
    # the shared arrays are untouched
    assert multi.LUMPED.g_h[0] == C.G_H[0]
    assert D.g_h[0] == multi.with_dendrites().g_h[0]


def test_ih_density_moves_the_h_current_in_the_right_hand_side():
    x = _gate_state(N)
    base = multi.membrane(-68.0)
    big = multi.membrane(-68.0, ih_density_scale=3.0)
    # only Ih differs, and only through the voltage derivative
    d0, d1 = _blocks(_rhs(base, x), N), _blocks(_rhs(big, x), N)
    np.testing.assert_array_equal(d0[1:], d1[1:])
    ih = multi.LUMPED.g_h * 0.4 * (-60.0 - multi.V_H)
    want = d0[0] - 2.0 * ih / multi.LUMPED.cap
    np.testing.assert_allclose(d1[0], want, rtol=1e-12)


def test_ih_vhalf_moves_the_h_steady_state_only():
    mem = multi.membrane(-68.0, ih_vhalf=-70.0)
    assert mem.ih_vhalf == 70.0
    # a sits at its steady state (0.5 at the midpoint) when V = vhalf
    x = _gate_state(N, -70.0)
    x.reshape(7, N)[6] = 0.5
    assert np.all(_blocks(_rhs(mem, x), N)[6] == 0.0)
    assert np.all(_blocks(_rhs(multi.membrane(-68.0), x), N)[6] != 0.0)


def test_ih_tau_scale_scales_only_the_h_rate():
    x = _gate_state(N)
    d0 = _blocks(_rhs(multi.membrane(-68.0), x), N)
    d1 = _blocks(_rhs(multi.membrane(-68.0, ih_tau_scale=4.0), x), N)
    np.testing.assert_array_equal(d0[:6], d1[:6])
    np.testing.assert_allclose(d1[6], d0[6] / 4.0, rtol=1e-12)


def test_na_activation_knobs_move_minf_and_taum_stays():
    mem = multi.membrane(-68.0, na_act_vhalf=-52.0, na_act_slope=8.0)
    assert (mem.na_act_vhalf, mem.na_act_k) == (52.0, -8.0)
    x = _gate_state(N, -52.0)
    x.reshape(7, N)[1] = 0.5  # m at its steady state at the midpoint
    d = _blocks(_rhs(mem, x), N)
    assert np.all(d[1] == 0.0)
    # every other derivative is the default membrane's
    d0 = _blocks(_rhs(multi.membrane(-68.0), x), N)
    np.testing.assert_array_equal(d[2:], d0[2:])
    np.testing.assert_array_equal(d[0], d0[0])


def test_dendrite_kht_is_a_multiple_of_the_somatic_density():
    mem = multi.membrane(-68.0, morph=D, dendrite_kht_scale=0.5)
    np.testing.assert_array_equal(mem.morph.g_kht[N:], 0.5 * C.G_KHT[0])
    np.testing.assert_array_equal(mem.morph.g_kht[:N], C.G_KHT)
    assert D.g_kht[N:].max() == 0.0  # the shared morphology is untouched
    assert mem.morph_key == mem.morph.key != D.key
    with pytest.raises(ValueError, match="dendrite_kht_scale needs"):
        multi.membrane(-68.0, dendrite_kht_scale=0.5)


def test_dendrite_kht_changes_the_dendritic_current():
    mem = multi.membrane(-68.0, morph=D, dendrite_kht_scale=1.0)
    base = multi.membrane(-68.0, morph=D)
    x = _gate_state(D.n)
    d0, d1 = _blocks(_rhs(base, x), D.n), _blocks(_rhs(mem, x), D.n)
    assert np.all(d1[0, N:] != d0[0, N:])
    np.testing.assert_array_equal(d1[1:], d0[1:])


@pytest.mark.parametrize(
    "temperature, q10, rate",
    [
        (35.0, 3.0, 1.0),
        (25.0, 3.0, 1 / 3),
        (45.0, 2.0, 2.0),
        (22.0, 3.0, 3**-1.3),
    ],
)
def test_temperature_divides_every_time_constant(temperature, q10, rate):
    mem = multi.membrane(
        -68.0, temperature=temperature, q10=q10, q10_conductance=1.0
    )
    assert mem.rate == pytest.approx(rate, rel=1e-14)
    x = _gate_state(N)
    d0 = _blocks(_rhs(multi.membrane(-68.0), x), N)
    d1 = _blocks(_rhs(mem, x), N)
    # the voltage derivative is the default's; each gate relaxes `rate` times
    # as fast
    np.testing.assert_array_equal(d1[0], d0[0])
    np.testing.assert_allclose(d1[1:], rate * d0[1:], rtol=1e-12)


def test_temperature_and_ih_tau_compose():
    x = _gate_state(N)
    d0 = _blocks(_rhs(multi.membrane(-68.0), x), N)
    mem = multi.membrane(-68.0, temperature=25.0, ih_tau_scale=2.0)
    d1 = _blocks(_rhs(mem, x), N)
    np.testing.assert_allclose(d1[6], d0[6] / 3 / 2, rtol=1e-12)
    np.testing.assert_allclose(d1[1], d0[1] / 3, rtol=1e-12)


def test_temperature_changes_a_run_and_35_does_not():
    args = (*STEP, "active-full", 8, -68.0, 3)
    _, y35 = multi.mso_axon(
        *args, mem=multi.membrane(-68.0, temperature=35.0), mode="exact"
    )
    _, y25 = multi.mso_axon(
        *args, mem=multi.membrane(-68.0, temperature=25.0), mode="exact"
    )
    _, y0 = multi.mso_axon(*args, mode="exact")
    np.testing.assert_array_equal(y35, y0)
    assert y25.shape != y0.shape or not np.allclose(y25, y0, atol=1e-3)


@pytest.mark.parametrize(
    "knob, value",
    [
        ("ih_density_scale", -0.1),
        ("ih_density_scale", float("nan")),
        ("dendrite_kht_scale", -1.0),
        ("ih_tau_scale", 0.0),
        ("ih_tau_scale", float("inf")),
        ("na_act_slope", -11.0),
        ("na_act_slope", 0.0),
        ("q10", 0.0),
        ("q10", float("nan")),
        ("ih_vhalf", float("nan")),
        ("na_act_vhalf", float("inf")),
        ("temperature", float("nan")),
    ],
)
def test_bad_knobs_are_refused(knob, value):
    with pytest.raises(ValueError, match=knob):
        multi.membrane(-68.0, morph=D, **{knob: value})


def test_knobs_enter_the_cache_keys():
    base = multi.membrane(-68.0)
    for knob, value in (
        ("ih_density_scale", 2.0),
        ("ih_vhalf", -75.0),
        ("ih_tau_scale", 2.0),
        ("na_act_vhalf", -50.0),
        ("na_act_slope", 9.0),
        ("temperature", 30.0),
        ("q10", 2.0),
    ):
        kw = {knob: value}
        if knob == "q10":
            kw["temperature"] = 30.0  # q10 alone is inert at 35 C
        other = multi.membrane(-68.0, **kw)
        assert (other.key, other.morph_key) != (base.key, base.morph_key), knob
    assert (
        multi.membrane(-68.0, morph=D, dendrite_kht_scale=0.5).key
        != multi.membrane(-68.0, morph=D).key
    )


def test_a_mem_with_new_channels_refuses_the_plain_morphology():
    mem = multi.membrane(-68.0, ih_density_scale=2.0)
    with pytest.raises(ValueError):
        multi.mso_axon(
            *STEP, "active-full", 8, -68.0, 3, mem=mem, morph=multi.LUMPED
        )


@pytest.mark.parametrize("axon_rest", [None, -60.0])
def test_a_rebalanced_rest_follows_the_knobs(axon_rest):
    # the rest is rebuilt from the knobbed gates: a steady state, uniform
    # (axon_rest == v0) or root-found (it differs)
    knobs = {
        "ih_density_scale": 2.0,
        "ih_vhalf": -72.0,
        "ih_tau_scale": 0.5,
        "na_act_vhalf": -50.0,
        "na_act_slope": 9.0,
        "temperature": 28.0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", multi.UnstableRestWarning)
        mem = multi.membrane(
            -60.0, rebalance_rest=True, axon_rest=axon_rest, **knobs
        )
    x = multi.initial_state(mem.morph, mem, -60.0)
    assert np.abs(_rhs(mem, x)).max() < 1e-8
    # and it is not the default membrane's rest
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", multi.UnstableRestWarning)
        plain = multi.membrane(-60.0, rebalance_rest=True, axon_rest=axon_rest)
    assert not np.array_equal(mem.vlk, plain.vlk)


def test_feeding_a_knobbed_morph_back_keeps_its_dendritic_kht():
    m1 = multi.membrane(-68.0, morph=D, dendrite_kht_scale=0.5)
    again = multi.membrane(-68.0, morph=m1.morph, ih_density_scale=2.0)
    np.testing.assert_array_equal(again.morph.g_kht, m1.morph.g_kht)
    assert again.morph.g_kht[N:].max() > 0


@pytest.mark.parametrize("temperature", [1e5, -1e5])
def test_an_extreme_temperature_is_a_value_error(temperature):
    with pytest.raises(ValueError, match="temperature"):
        multi.membrane(-68.0, temperature=temperature)


@pytest.mark.parametrize(
    "knob,value",
    [("ih_vhalf", -70.0), ("na_act_vhalf", -48.0), ("temperature", 30.0)],
)
def test_klt_to_leak_morphology_key_follows_the_rest_knobs(knob, value):
    kw = {"rebalance_rest": True, "klt_to_leak": "chord"}
    a = multi.membrane(-60.0, **kw)
    b = multi.membrane(-60.0, **kw, **{knob: value})
    assert a.morph.key != b.morph.key


def test_temperature_scales_the_voltage_gated_densities_only():
    mem = multi.membrane(-68.0, temperature=45.0)
    L = multi.LUMPED
    np.testing.assert_array_equal(mem.g_na, L.g_na * 2.0)
    np.testing.assert_array_equal(mem.g_klt, L.g_klt * 2.0)
    np.testing.assert_array_equal(mem.morph.g_h, L.g_h * 2.0)
    np.testing.assert_array_equal(mem.morph.g_kht, L.g_kht * 2.0)
    np.testing.assert_array_equal(mem.morph.g_lk, L.g_lk)
    cool = multi.membrane(-68.0, temperature=25.0, q10_conductance=1.5)
    np.testing.assert_allclose(cool.g_na, L.g_na / 1.5, rtol=1e-15)


def test_35_c_ignores_the_conductance_q10():
    mem = multi.membrane(-68.0, temperature=35.0, q10_conductance=7.5)
    assert mem.morph is multi.LUMPED


def test_conductance_q10_enters_the_keys_and_the_description():
    a = multi.membrane(-68.0, temperature=30.0)
    b = multi.membrane(-68.0, temperature=30.0, q10_conductance=1.5)
    assert a.key != b.key and a.morph_key != b.morph_key


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_bad_conductance_q10_is_refused(value):
    with pytest.raises(ValueError):
        multi.membrane(-68.0, q10_conductance=value)


def test_dendrite_kht_needs_real_dendrites_not_just_extra_compartments():
    extra = dataclasses.replace(multi.LUMPED, n=46)
    with pytest.raises(ValueError, match="dendrite_kht_scale"):
        multi.membrane(-68.0, morph=extra, dendrite_kht_scale=1.0)


@pytest.mark.parametrize(
    "again", [{"temperature": 25.0}, {}, {"soma_klt_scale": 2.0}]
)
def test_a_temperatures_morphology_cannot_build_another_membrane(again):
    a = multi.membrane(-68.0, temperature=25.0)
    with pytest.raises(ValueError, match="temperature's densities"):
        multi.membrane(-68.0, morph=a.morph, **again)


def test_a_temperatures_morphology_cannot_run_without_its_membrane():
    a = multi.membrane(-68.0, temperature=25.0)
    with pytest.raises(ValueError, match="temperature's densities"):
        multi.mso_axon(
            "step", 2, 4, 0.0, 3, "active-full", 3, -68.0, 1, morph=a.morph
        )
    t, _ = multi.mso_axon(
        "step", 2, 4, 0.0, 3, "active-full", 3, -68.0, 1, mem=a, morph=a.morph
    )
    assert t[-1] == 3


def test_a_second_ih_scale_is_refused():
    a = multi.membrane(-68.0, ih_density_scale=2.0)
    with pytest.raises(ValueError, match="ih_density_scale"):
        multi.membrane(-68.0, morph=a.morph, ih_density_scale=2.0)


def test_the_error_for_a_plain_morphology_says_what_to_pass():
    a = multi.membrane(-68.0, morph=D, temperature=25.0)
    with pytest.raises(ValueError, match="morph=mem.morph"):
        multi.mso_axon(
            "step", 2, 4, 0.0, 3, "active-full", 3, -68.0, 1, mem=a, morph=D
        )
