"""Somatic spike size vs mature MSO cells (Scott et al 2005 protocol).

    uv run scripts/somatic_spike.py --out-dir figures/somatic

100 ms somatic steps at multiples of rheobase; amplitude from the inflection
point. Includes the dendrotoxin control: KLT removed at soma and AIS, with rest
rebalanced so the cell doesn't fire spontaneously.
"""

import argparse
import pathlib
import textwrap
from typing import Any

import numpy as np

from msoaxon import multi, somatic

# the somatic trace has a distinct inflection only from ~1.35x to ~4.5x rheobase
# (spike_amplitude's `distinct` flag)
MULTS = (1.5, 2.0, 3.0)
# Scott et al 2005: mature MSO, 17 +- 2 mV; with dendrotoxin, 15 -> 37 mV
SCOTT_MATURE = (15.0, 19.0)
SCOTT_DTX = (15.0, 37.0)
CASES = {
    "45-compartment (msoAxon.m)": ("multi", {}),
    "45-compartment, KLT blocked at soma + AIS (dendrotoxin)": (
        "multi",
        {"soma_klt_scale": 0, "ais_klt_scale": 0, "rebalance_rest": True},
    ),
    "2-compartment (TwoCpt.m)": ("two", {}),
}


def main() -> None:
    """Runs the somatic-spike protocol and saves the figure and JSON."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", default="figures/somatic")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="1%% rheobase tolerance, 1.5x and 3x only: a smoke test, "
        "not results",
    )
    ap.add_argument(
        "--replot",
        metavar="JSON",
        help="redraw the figure from a previous run's somatic_spike.json, "
        "beside it, without simulating",
    )
    a = ap.parse_args()
    # imported here rather than at the top, like the pool scripts, so importing
    # this module stays cheap
    from msoaxon import plotting

    plotting.agg()
    if a.replot:
        for path in plotting.save(draw(plotting.load(a.replot)), a.replot):
            print(f"saved {path}")
        return
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, (model, knobs) in CASES.items():
        mem = multi.membrane(-68.0, **knobs) if knobs else None
        rb = somatic.rheobase(model, mem=mem, rel_tol=1e-2 if a.quick else 1e-4)
        rows = []
        for k in (1.5, 3.0) if a.quick else MULTS:
            r = somatic.spike_amplitude(model, rb * k, mem=mem)
            if not r["distinct"]:
                raise SystemExit(
                    f"{name}: no distinct inflection at {k}x rheobase"
                )
            t, x = r.pop("trace")  # the run the amplitude was measured on
            keep = t >= 4.8  # the part the figure draws
            rows.append(
                {
                    "multiple": k,
                    **{
                        key: float(v)
                        for key, v in r.items()
                        if key != "distinct"
                    },
                    "trace": {
                        "t_ms": plotting.compact(t[keep]),
                        "soma_mV": plotting.compact(x[keep, 0]),
                    },
                }
            )
        results[name] = {"rheobase_pA": float(rb), "spikes": rows}
        print(
            f"{name}: rheobase {rb:.0f} pA; "
            + ", ".join(
                f"{r['multiple']}x {r['amplitude']:.1f} mV" for r in rows
            ),
            flush=True,
        )
    plotting.save(
        draw(results), out / "somatic_spike", data=results, args=vars(a)
    )
    print(f"saved {out}/somatic_spike.png, .pdf and .json")


def reference(name: str) -> tuple[float, float]:
    """Returns the Scott et al 2005 amplitudes a case is compared with.

    Args:
        name: A key of CASES.

    Returns:
        (low, high) [mV]: 17 +- 2 mV for mature MSO cells, or for the
        dendrotoxin case the amplitude before (15) and with it (37).
    """
    return SCOTT_DTX if "dendrotoxin" in name else SCOTT_MATURE


def draw(results: dict[str, Any]) -> Any:
    """Draws one panel per case from the results (or their JSON).

    Each trace is drawn from its inflection point (time and voltage), so
    the spikes align and the amplitude is the height above 0; a bracket at
    the peak shows the amplitude the legend gives, and the Scott et al 2005
    values are drawn as a band (mature cells) or as lines (dendrotoxin).

    Args:
        results: Per case: rheobase_pA and spikes, each spike with its
            multiple, amplitude, inflection point and trace.

    Returns:
        The figure.
    """
    from msoaxon import plotting

    cases = {k: v for k, v in results.items() if k != "meta"}
    fig = plotting.figure("double", height=3.0)
    axes = fig.subplots(1, len(cases), sharex=True, sharey=True)
    for ax, (name, case) in zip(axes, cases.items(), strict=True):
        lo, hi = reference(name)
        if "dendrotoxin" in name:
            ax.axhline(lo, color=plotting.GREY, ls=":", lw=0.8)
            ax.axhline(hi, color=plotting.GREY, ls="--", lw=0.8)
            ref = f"Scott 2005: {lo:g} mV, with dendrotoxin {hi:g} mV"
        else:
            ax.axhspan(lo, hi, color="0.88", zorder=0)
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            ref = f"Scott 2005, mature MSO: {mid:g} ± {half:g} mV"
        for r in case["spikes"]:
            # one sequential map over MULTS: the same multiple is the same
            # colour in every panel, and in a quick run
            (colour,) = plotting.sequential(
                [r["multiple"]], min(MULTS), max(MULTS)
            )
            t = np.asarray(r["trace"]["t_ms"]) - r["t_inflection"]
            v = np.asarray(r["trace"]["soma_mV"]) - r["v_inflection"]
            ax.plot(
                t,
                v,
                color=colour,
                label=f"{r['multiple']}x: {r['amplitude']:.1f} mV",
            )
            # the amplitude as a bracket at the peak, from the inflection
            t_pk = t[int(np.argmax(v))]
            ax.plot(
                [t_pk, t_pk],
                [0, r["amplitude"]],
                color=colour,
                lw=0.8,
                marker="_",
                ms=6,
            )
        ax.plot(0, 0, "k.", ms=5)  # the inflection, where every trace starts
        ax.set_title(
            f"{textwrap.fill(name, 30)}\nrheobase {case['rheobase_pA']:.0f} pA"
        )
        ax.set_xlabel("Time from the inflection (ms)")
        ax.set_xlim(-1.0, 2.0)
        # below the panel: the spike and its dot fill the panel's corners
        ax.legend(
            title=f"rheobase multiple: amplitude\n({ref})",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
        )
    axes[0].set_ylabel("Soma voltage from the inflection (mV)")
    fig.suptitle("Somatic spike, 100 ms step (Scott et al 2005 protocol)")
    return fig


if __name__ == "__main__":
    main()
