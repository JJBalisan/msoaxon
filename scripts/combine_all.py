"""Run both models on one stimulus and plot them (port of Combine_all.m).

    uv run scripts/combine_all.py --stim EPSGpair --node 3 --out comparison.png
"""  # noqa: D208, E501  # fmt: skip
# (the docstring is argparse's --help text, so the formatter leaves it as is)

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import msoaxon
from msoaxon import plotting, spiking

# per-stimulus defaults from Combine_all.m: (stop, I)
DEFAULTS = {
    "step": (10.0, 8500.0),
    "ramp": (
        5.5,
        8500.0,
    ),  # Combine_all.m has stop == start (5), a zero-length ramp
    "ramp2": (5.5, 8500.0),
    "sine": (5 + 1000 / (2 * 200), 8500.0),
    "EPSG": (10.0, 108.0),
    "EPSGpair": (5.0, 40.0),  # stop is the second EPSG onset
    "Synaptic": (15.0, 0.0),
    "SynapticPair": (15.0, 0.0),
}


def syn_params(t_end: float) -> msoaxon.SynParams:
    """Returns the synaptic settings of Combine_all.m for a run length."""
    return msoaxon.SynParams(t_end=t_end, random_in=1396, diff=1.0, f=200.0)


def draw(d: dict[str, Any]) -> Any:
    """Draws the figure from its data (or its JSON).

    Args:
        d: The graph settings (without Syn, rebuilt from tEnd), the time
            window drawn, and the runs' times and states as drawn: t1, y
            (the 45 compartments' voltages), t2, x and t3, z (soma and
            axon).

    Returns:
        The figure.
    """
    graph = {**d["graph"], "Syn": syn_params(d["graph"]["tEnd"])}
    runs = (np.asarray(d[k]) for k in ("t1", "y", "t2", "x", "t3", "z"))
    return plotting.graphing(graph, *runs, window=d.get("window", "event"))


def main() -> None:
    """Parses the command line, runs both models and draws the figure."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stim", default="EPSGpair", choices=DEFAULTS)
    ap.add_argument(
        "--type",
        default="active-full",
        help="passive, active-KLT, active-H, active-KLT+H, active-sodium, "
        "active-KHT, active-full",
    )
    ap.add_argument(
        "--node",
        type=int,
        default=3,
        help="axon compartment to compare (1-indexed)",
    )
    ap.add_argument("--input-node", type=int, default=1)
    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--v0", type=float, default=-68.0)
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument(
        "--stop", type=float, help="override the per-stimulus default"
    )
    ap.add_argument("--I", type=float, help="override the per-stimulus default")
    ap.add_argument(
        "--factor",
        type=float,
        default=30.0,
        help="mV above soma counted as a spike",
    )
    ap.add_argument(
        "--graph",
        default="ModelComparison",
        help="ModelComparison, NodeComparison, MultiCompartment, Contour, "
        "Input, Multigraph, ModelComparison2",
    )
    ap.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("T0", "T1"),
        help="time axis [ms] (default: a few ms around the first trace to "
        "rise 10 mV, see plotting.event_window)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="draw the whole run, 0 to --t-end, as Combine_all.m did",
    )
    ap.add_argument(
        "--out",
        help="save the figure here (PNG and PDF, and the JSON --replot "
        "reads) instead of showing it",
    )
    ap.add_argument(
        "--replot",
        metavar="JSON",
        help="redraw the figure from a previous run's JSON, beside it, "
        "without simulating",
    )
    a = ap.parse_args()
    if a.replot:
        plotting.agg()
        for path in plotting.save(draw(plotting.load(a.replot)), a.replot):
            print(f"saved {path}")
        return

    stop, I = DEFAULTS[a.stim]
    stop = a.stop if a.stop is not None else stop
    I = a.I if a.I is not None else I
    syn = syn_params(a.t_end)

    args = (a.stim, a.start, stop, I)
    t1, y = msoaxon.mso_axon(
        *args, a.node, a.type, a.t_end, a.v0, a.input_node, syn
    )
    t2, x = msoaxon.two_cpt(
        *args, a.node, a.type, a.t_end, a.v0, a.input_node, syn
    )
    # Combine_all.m ran TwoCpt twice with identical arguments for the
    # node-3/node-5 panels; here the second run uses node 5 so those panels mean
    # what they say.
    t3, z = msoaxon.two_cpt(*args, 5, a.type, a.t_end, a.v0, a.input_node, syn)

    # Combine_all.m counted column 1 of the 2-CPT markers (soma minus soma,
    # always 0); column 2 is the axon compartment it meant.
    n_two = spiking.count_rising_edges(
        spiking.spiking(x, a.factor, "Two")[:, 1]
    )
    n_multi = spiking.count_rising_edges(
        spiking.spiking(y, a.factor, "Mult")[:, a.node - 1]
    )
    print(
        f"spikes at node {a.node}: multi-compartment {n_multi}, "
        f"two-compartment {n_two}"
    )

    # what the figure draws, thinned to what a 7 in figure shows and rounded
    # to 6 significant digits: the 45 compartments' voltages (Contour and
    # Multigraph), and the two-compartment soma and axon
    s1, s2, s3 = (plotting.thin(len(t)) for t in (t1, t2, t3))
    graph = {
        "Type": a.graph,
        "node": a.node,
        "tEnd": a.t_end,
        "stimType": a.stim,
        "inputNode": a.input_node,
        "start": a.start,
        "stop": stop,
        "I": I,
    }
    # resolved on the full runs, so a replot draws the same window
    window = plotting.graph_window(
        graph, t1, y, t2, x, None if a.full else (a.window or "event")
    )
    data = {
        "window": list(window),
        "graph": graph,
        "t1": plotting.compact(t1[s1]),
        "y": plotting.compact(y[s1, :45]),
        "t2": plotting.compact(t2[s2]),
        "x": plotting.compact(x[s2, :2]),
        "t3": plotting.compact(t3[s3]),
        "z": plotting.compact(z[s3, :2]),
    }
    fig = plotting.graphing(
        {**graph, "Syn": syn}, t1, y, t2, x, t3, z, window=window
    )
    if a.out:  # out.png writes out.png, out.pdf and out.json
        for path in plotting.save(fig, a.out, data=data, args=vars(a)):
            print(f"saved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
