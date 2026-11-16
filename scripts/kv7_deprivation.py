"""Kv7 / M-type K: Jing et al 2025's deprivation scenario, both models (#85).

    uv run scripts/kv7_deprivation.py --out-dir figures/kv7
    uv run scripts/kv7_deprivation.py --quick          # coarse, smoke tests

Jing, Xi, Fransazov & Goldwyn 2025 (J Comput Neurosci, doi:10.1007/s10827-025-
00902-9) model AIS plasticity after auditory deprivation in a two-compartment
soma/AIS model: the AIS's low-threshold K (KLT) is replaced by the slower
M-type K and the AIS's capacitance and leak grow by 50%. Their single-spike
threshold falls from about 1360 to 500 pA, and the cell turns from phasic
(one spike at step onset) to tonic (repetitive firing). Their full text could
not be read for this script (see constants.py on the M gate, an inference),
so this is the scenario applied to this repository's models, not a
reproduction of theirs:

- two-compartment: two_cpt(klt_to_m=True, axon_cap_scale=1.5,
  axon_leak_scale=1.5) on Goldwyn et al 2019's passive calibration
  (GOLDWYN_2019: v0 -58 mV, 8.5 MOhm, 0.34 ms), node 3, gM2 = the removed
  gKLT2; the soma keeps its KLT. Their text and Table 1 disagree on gKHT
  (150 and 225 nS appear in the table), and their KHT sits in the AIS while
  this model's is somatic, so gKHT is run at 0 (the preset), 150 and 225
  nS, at the soma. TwoCpt.m's own calibration (v0 -68 mV, 10 MOhm) is run
  as well. "deprived+soma" swaps the soma's KLT for M too: a sensitivity
  check, since this calibration puts almost all KLT in the soma (2968 nS
  against the axon's 29 nS at node 3), where the AIS swap cannot reach it.
- gM: Jing et al's value could not be read, so the M conductance is run
  at the removed KLT's maximal conductance (1x) and at a tenth of it.
- 45-compartment: membrane(ais_klt_scale=0, ais_m_density=1.55,
  ais_cap_scale=1.5, ais_leak_scale=1.5, rebalance_rest=True) against the
  rebalanced control, compared at node 3; "deprived+soma" swaps the soma's
  KLT too (soma_klt_scale=0, soma_m_density=1.55).
- the 45-compartment nodal / AIS Kv7 sweep: M added on top of KLT at the AIS
  and at the nodes (Battefeld et al 2014: Kv7.2/7.3 rises along the AIS and
  sits at nodes), rebalanced to -68 mV.

Threshold is somatic.rheobase: the smallest 100 ms somatic step that makes
the compared compartment rise 10 mV above the soma. Firing type: spikes
during the 100 ms step at 1.2, 2 and 3 times threshold; "phasic" if every one
fires once, "tonic" if the 3x step fires 3 or more times, else "mixed".
Nothing here changes a default; the tables are proposals for the PR.
"""

import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np

import msoaxon
from msoaxon import _dispatch, constants, multi, somatic, two

MULTS = (1.2, 2.0, 3.0)
NODE = 3
GKHT = (0.0, 150.0, 225.0)  # [nS]: the preset's, and the two of Jing's Table 1
DEPRIVED = {"klt_to_m": True, "axon_cap_scale": 1.5, "axon_leak_scale": 1.5}
# the sensitivity check: the soma's KLT swapped for M too
DEPRIVED_SOMA = {**DEPRIVED, "soma_klt_to_m": True}
AIS_M = float(constants.G_KLT[1])  # the AIS's KLT density, 1.55 nS/um^2
SWEEP = (0.0, 0.1, 0.4, 1.55)  # M densities [nS/um^2]
SWEEP_QUICK = (0.0, 0.4)
TARGET = {"control": 1360.0, "deprived": 500.0}  # Jing et al 2025 [pA]
# gM as a fraction of the gKLT it replaces: Jing et al's gM could not be read
M_FRACTIONS = (1.0, 0.1)


def spike_times(t: np.ndarray, soma: np.ndarray, axon: np.ndarray) -> list:
    """Returns the time of each spike, counted as spiking.count_spikes does.

    A spike is the axon above the soma by somatic.FACTOR; the counter
    re-arms once axon < soma.
    """
    out, armed = [], True
    for ti, u1, u2 in zip(t, soma, axon, strict=True):
        if u2 - u1 > somatic.FACTOR and armed:
            out.append(float(ti))
            armed = False
        if u2 < u1:
            armed = True
    return out


def firing_type(counts: list[int]) -> str:
    """Returns "phasic", "tonic" or "mixed" from the counts at MULTS."""
    if all(c == 1 for c in counts):
        return "phasic"
    if counts[-1] >= 3:
        return "tonic"
    return "mixed"


def measure(
    model: str, v0: float, kw: dict[str, Any], rel_tol: float
) -> dict[str, Any]:
    """Returns the threshold and the step responses of one configuration.

    Args:
        model: "multi" or "two".
        v0: The resting potential [mV].
        kw: The model keywords (mem for "multi"; two_cpt's for "two").
        rel_tol: The threshold's relative tolerance.

    Returns:
        threshold [pA], and per multiple of it the spike count and times.
    """
    th = somatic.rheobase(model, NODE, v0, rel_tol=rel_tol, model_kw=kw)
    row: dict[str, Any] = {"threshold_pA": th, "steps": []}
    if not math.isfinite(th):
        row["type"] = "no spike up to 1e6 pA"
        return row
    col = _dispatch.axon_column(model, NODE)
    for k in MULTS:
        t, x = _dispatch.run_model(
            model,
            "step",
            somatic.START,
            somatic.STOP,
            k * th,
            NODE,
            somatic.T_END,
            v0,
            1,
            **kw,
        )
        times = spike_times(t, x[:, 0], x[:, col])
        row["steps"].append(
            {"multiple": k, "count": len(times), "times_ms": times}
        )
    row["type"] = firing_type([s["count"] for s in row["steps"]])
    return row


def two_rows(rel_tol: float, mode: str | None) -> list[dict[str, Any]]:
    """The two-compartment scenario: control and deprived, per calibration."""
    calibrations = [
        (
            f"Goldwyn 2019, gKHT {g:g} nS",
            two.GOLDWYN_2019["v0"],
            {
                "r1": two.GOLDWYN_2019["r1"],
                "tau_est": two.GOLDWYN_2019["tau_est"],
                "gKHT": g,
            },
        )
        for g in GKHT
    ]
    calibrations.append(("TwoCpt.m (v0 -68, 10 MOhm)", -68.0, {}))
    rows = []
    for label, v0, base in calibrations:
        P0 = two.get_params(v0, NODE, 1, "active-full", **base)
        cases = [("control", {}, None)]
        for frac in M_FRACTIONS:
            cases.append(
                ("deprived", {**DEPRIVED, "gM2": frac * P0.gKLT2}, frac)
            )
            soma = {**DEPRIVED_SOMA, "gM1": frac * P0.gKLT1}
            cases.append(
                ("deprived+soma", {**soma, "gM2": frac * P0.gKLT2}, frac)
            )
        for state, extra, frac in cases:
            kw = {**base, **extra, "mode": mode}
            P = two.get_params(
                v0,
                NODE,
                1,
                "active-full",
                **{k: v for k, v in kw.items() if k != "mode"},
            )
            row = measure("two", v0, kw, rel_tol)
            row.update(
                model="two",
                config=label,
                state=state,
                m_fraction=frac,
                v0=v0,
                gKLT1_nS=P.gKLT1,
                gM1_nS=P.gM1,
                gKLT2_nS=P.gKLT2,
                gM2_nS=P.gM2,
                cap2_pF=P.cap2,
                glk2_nS=P.glk2,
                gKHT_nS=P.gKHT,
            )
            rows.append(row)
            print_row(row)
    return rows


def multi_rows(rel_tol: float, mode: str | None) -> list[dict[str, Any]]:
    """The 45-compartment scenario: rebalanced control and deprived."""
    cases = [("control", {}, None)]
    for frac in M_FRACTIONS:
        deprived = {
            "ais_klt_scale": 0.0,
            "ais_m_density": frac * AIS_M,
            "ais_cap_scale": 1.5,
            "ais_leak_scale": 1.5,
        }
        soma = {
            **deprived,
            "soma_klt_scale": 0.0,
            "soma_m_density": frac * float(constants.G_KLT[0]),
        }
        cases += [("deprived", deprived, frac), ("deprived+soma", soma, frac)]
    rows = []
    for state, mkw, frac in cases:
        mem = multi.membrane(-68.0, rebalance_rest=True, **mkw)
        row = measure("multi", -68.0, {"mem": mem, "mode": mode}, rel_tol)
        row.update(
            model="multi",
            config="LUMPED, rebalanced",
            state=state,
            m_fraction=frac,
        )
        rows.append(row)
        print_row(row)
    return rows


def sweep_rows(
    densities: tuple[float, ...], rel_tol: float, mode: str | None
) -> list[dict[str, Any]]:
    """The 45-compartment nodal x AIS Kv7 sweep, KLT kept."""
    rows = []
    for ais in densities:
        for node in densities:
            mem = multi.membrane(
                -68.0,
                rebalance_rest=True,
                ais_m_density=ais,
                node_m_density=node,
            )
            row = measure("multi", -68.0, {"mem": mem, "mode": mode}, rel_tol)
            row.update(
                model="multi",
                config=f"Kv7 AIS {ais:g}, nodes {node:g} nS/um^2",
                state="sweep",
                ais_m=ais,
                node_m=node,
            )
            rows.append(row)
            print_row(row)
    return rows


def print_row(row: dict[str, Any]) -> None:
    """Prints one configuration's line of the table."""
    counts = "/".join(str(s["count"]) for s in row["steps"]) or "-"
    frac = row.get("m_fraction")
    state = row["state"] + ("" if frac is None else f" gM {frac:g}x")
    print(
        f"{row['model']:5s} {row['config']:38s} {state:22s} "
        f"threshold {row['threshold_pA']:9.1f} pA  spikes {counts:8s} "
        f"{row['type']}",
        flush=True,
    )


def main() -> None:
    """Runs every configuration and saves the table as JSON."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", default="figures/kv7")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="1%% threshold tolerance and a 2 x 2 sweep (smoke tests)",
    )
    ap.add_argument(
        "--mode",
        choices=("exact", "fast", "native"),
        default=None,
        help="the right-hand side (default: MSOAXON_MODE, else native)",
    )
    a = ap.parse_args()
    rel_tol = 1e-2 if a.quick else 1e-4
    print(
        f"spike counts at {', '.join(f'{k:g}' for k in MULTS)} x threshold, "
        f"100 ms somatic steps; target (Jing et al 2025, two-compartment): "
        f"{TARGET['control']:g} -> {TARGET['deprived']:g} pA, phasic -> tonic"
    )
    rows = two_rows(rel_tol, a.mode) + multi_rows(rel_tol, a.mode)
    rows += sweep_rows(SWEEP_QUICK if a.quick else SWEEP, rel_tol, a.mode)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "msoaxon": msoaxon.__version__,
        "quick": a.quick,
        "rel_tol": rel_tol,
        "multiples": MULTS,
        "target": TARGET,
        "rows": rows,
    }
    (out / "kv7_deprivation.json").write_text(json.dumps(result, indent=1))
    print(f"\nsaved {out}/kv7_deprivation.json (mode={a.mode or 'default'})")


if __name__ == "__main__":
    main()
