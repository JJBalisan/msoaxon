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
reproduction of theirs. In particular, the 0.1x "deprived+soma" rows that
land near 500 pA are a tuning of the free gM, not a reproduction of Jing's
threshold.

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
  The second compartment is not an AIS like Jing's (their soma-AIS coupling
  is g_ax = 56.4 nS): it is TwoCpt.m's node-3 coupling compartment, with
  cap2 = 0.095 pF against the soma's 40 pF (1/422) and glk2 = 0.098 nS on
  Goldwyn's calibration (0.17 pF against 71 pF, 0.084 nS on TwoCpt.m's).
  Scaling it by 1.5 cannot move the threshold, so in this model the
  scenario's capacitance and leak part does nothing by construction, and
  "the AIS-only swap does little" is a statement about this model's
  compartment sizes, not about AIS plasticity.
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
fires once and none ends in depolarisation block, "tonic" if the 3x step
fires 3 or more times, "1 spike, block" if every step fires once and at
least one ends in block, else "mixed". A step is in block when the compared
compartment's Na inactivation h ends below BLOCK_H; each step records that
h, its minimum over the step and the soma's V at the step's end, since a
soma with little K can depolarise far past any spike (to +95 mV at 3 nA with
both compartments' KLT swapped for gM = 0 and no KHT) and still count as one
spike. Nothing here changes a default; the tables are proposals for the PR.
"""

import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np

import msoaxon
from msoaxon import _dispatch, constants, multi, somatic, spiking, two

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
# Na inactivation of the compared compartment below which a step's end is
# depolarisation block: the K-free soma runs reach h ~ 0, the 0.1x rows that
# fire once stay at 0.12-0.16 (#176's first review)
BLOCK_H = 0.05


def spike_times(t: np.ndarray, soma: np.ndarray, axon: np.ndarray) -> list:
    """Returns the time of each spike, by spiking.spike_indices's rule.

    A spike is the axon above the soma by somatic.FACTOR; the counter
    re-arms once axon < soma.
    """
    idx = spiking.spike_indices(soma, axon, somatic.FACTOR)
    return [float(t[i]) for i in idx]


def h_column(model: str, kw: dict[str, Any]) -> int:
    """Returns the state column of the compared compartment's Na h."""
    if model == "two":
        return 7  # h2, two.py's state order
    return 2 * kw["mem"].morph.n + NODE - 1  # multi's third block


def firing_type(counts: list[int], blocked: list[bool]) -> str:
    """Returns the row's firing type from its counts and blocks at MULTS."""
    if all(c == 1 for c in counts):
        return "1 spike, block" if any(blocked) else "phasic"
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
        threshold [pA], and per multiple of it the spike count and times,
        the soma's V at the step's end, the compared compartment's Na h at
        the end and its minimum over the step, and whether it is in block.
    """
    th = somatic.rheobase(model, NODE, v0, rel_tol=rel_tol, model_kw=kw)
    row: dict[str, Any] = {"threshold_pA": th, "steps": []}
    if not math.isfinite(th):
        row["type"] = "no spike up to 1e6 pA"
        return row
    col = _dispatch.axon_column(model, NODE)
    hcol = h_column(model, kw)
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
        on = (t >= somatic.START) & (t <= somatic.STOP)
        end = int(np.flatnonzero(on)[-1])  # the step's last sample
        h_end = float(x[end, hcol])
        row["steps"].append(
            {
                "multiple": k,
                "count": len(times),
                "times_ms": times,
                "end_v_soma_mV": float(x[end, 0]),
                "end_h": h_end,
                "min_h": float(x[on, hcol].min()),
                "block": h_end < BLOCK_H,
            }
        )
    row["type"] = firing_type(
        [s["count"] for s in row["steps"]],
        [s["block"] for s in row["steps"]],
    )
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
                cap1_pF=P.cap1,
                gC_nS=P.gC,
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
