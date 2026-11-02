"""Somatic spike size vs mature MSO cells (Scott et al 2005 protocol).

    uv run scripts/somatic_spike.py --out-dir figures/somatic

100 ms somatic steps at multiples of rheobase; amplitude from the inflection
point. Includes the dendrotoxin control: KLT removed at soma and AIS, with rest
rebalanced so the cell doesn't fire spontaneously.
"""

import argparse
import json
import pathlib
import textwrap

from msoaxon import multi, somatic

# the somatic trace has a distinct inflection only from ~1.35x to ~4.5x rheobase
# (spike_amplitude's `distinct` flag)
MULTS = (1.5, 2.0, 3.0)
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
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # imported here rather than at the top, like the pool scripts, so importing
    # this module stays cheap
    from msoaxon import plotting

    plotting.agg()
    fig = plotting.figure("double", height=2.8)
    axes = fig.subplots(1, len(CASES), sharey=True)
    results = {}
    for ax, (name, (model, knobs)) in zip(axes, CASES.items(), strict=True):
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
            rows.append(
                {
                    "multiple": k,
                    **{
                        key: float(v)
                        for key, v in r.items()
                        if key != "distinct"
                    },
                }
            )
            keep = t >= 4.8
            ax.plot(t[keep], x[keep, 0], label=f"{k}x: {r['amplitude']:.1f} mV")
            ax.plot(r["t_inflection"], r["v_inflection"], "k.", ms=5)
        results[name] = {"rheobase_pA": float(rb), "spikes": rows}
        # wrapped: three long titles side by side on a 7 in figure
        ax.set_title(f"{textwrap.fill(name, 30)}\nrheobase {rb:.0f} pA")
        ax.set_xlabel("Time (ms)")
        ax.legend(title="amplitude from inflection (dot)")
        print(
            f"{name}: rheobase {rb:.0f} pA; "
            + ", ".join(
                f"{r['multiple']}x {r['amplitude']:.1f} mV" for r in rows
            ),
            flush=True,
        )
    axes[0].set_ylabel("Soma voltage (mV)")
    fig.suptitle(
        "Somatic spike, 100 ms step (Scott et al 2005 protocol). "
        "Mature MSO: 17 ± 2 mV; dendrotoxin: 15 → 37 mV"
    )
    plotting.save(fig, out / "somatic_spike")
    (out / "somatic_spike.json").write_text(json.dumps(results, indent=1))
    print(f"saved {out}/somatic_spike.png, .pdf and .json")


if __name__ == "__main__":
    main()
