"""Spiking-threshold curves for both models (port of Making_Threshold_graphs.m).

    uv run scripts/making_threshold_graphs.py --out-dir figures/thresholds   # the 2 x 4 composite
    uv run scripts/making_threshold_graphs.py --node 3 --per-stimulus --out-dir figures
    uv run scripts/making_threshold_graphs.py --only EPSGpair --jpg-grid   # the repo's jpgs

The main output is thresholds_overview.png (and .pdf, .json): one column per
stimulus, one row per --node (default 3 and 5), the rows of a column on the
same y axis, and under each panel the two-compartment model's difference from
the 45-compartment one in percent. --per-stimulus also writes the single
panels, <stim>_thresholds_cpt<node>.png.

Each binary-search point runs about 8 simulations per model for EPSGpair
(ceiling 150) and about 14 for ramp, ramp2 and sine (ceiling 15000), halving
from the ceiling to a step of 1. Points run in parallel across all CPUs unless
--workers says otherwise.
"""  # noqa: E501 (printed verbatim as --help)

import argparse
import pathlib
from concurrent import futures
from typing import Any, NamedTuple

import numpy as np

import msoaxon
from msoaxon import threshold


class Sweep(NamedTuple):
    """One threshold sweep: its search and how its figure labels it."""

    n_points: int
    factor: float  # the spike criterion, axon - soma [mV]
    max_I: float  # the search's start and ceiling  # noqa: N815
    title: str
    xlabel: str
    x_unit: str  # the unit x is drawn in
    x_scale: float  # x as drawn = threshold.sweep_x times this
    unit: str  # the thresholds' unit
    mark_min: bool  # mark each model's optimum (an interior minimum)


# x-values come from threshold.sweep_x (ms, Hz, ms); the EPSG-pair delay is
# drawn in us, as in the coincidence and dendrites figures
SWEEPS = {
    "ramp": Sweep(15, 30, 15000, "Ramp up, then off", "Ramp top", "ms", 1,
                  "pA", True),
    "sine": Sweep(12, 30, 15000, "Sine", "Frequency", "Hz", 1, "pA", True),
    "ramp2": Sweep(15, 30, 15000, "Ramp2", "Ramp stop", "ms", 1, "pA", False),
    "EPSGpair": Sweep(25, 10, 150, "EPSG pair", "EPSG delay", "us",
                      1000, "nS per EPSG", False),
}  # fmt: skip
NODES = (3, 5)


def epsgpair_grid(jpg_grid: bool) -> tuple[int, dict[str, float | bool]]:
    """Returns the EPSGpair sweep's grid.

    tests/test_coverage.py pins the jpg grid's thresholds through this.

    Args:
        jpg_grid: Use the grid the repo's EPSGpair_Thresholds*.jpg were
            made on (11 points 0.1 ms apart, unrounded) instead of
            BinarySearch.m's.

    Returns:
        (n_points, keywords), the keywords being binary_search's
        epsg_pair_dt and rounded.
    """
    if jpg_grid:
        return 11, {"epsg_pair_dt": 0.1, "rounded": False}
    return SWEEPS["EPSGpair"][0], {"epsg_pair_dt": 1 / 25, "rounded": True}


def resolution(stim: str, jpg_grid: bool) -> float | None:
    """Returns the rounding binary_search applies to a sweep's thresholds.

    Args:
        stim: The stimulus swept.
        jpg_grid: Whether EPSGpair runs on the jpg grid (unrounded).

    Returns:
        1 for EPSGpair, 10 otherwise (BinarySearch.m's rounding), or None
        when the sweep is not rounded.
    """
    if stim == "EPSGpair":
        return None if jpg_grid else 1
    return 10


def main() -> None:
    """Parses the command line and runs the threshold sweeps."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--node",
        type=int,
        action="append",
        help="compartment compared with the soma, one row of the composite "
        "each (repeatable; default 3 and 5)",
    )
    ap.add_argument(
        "--only", choices=SWEEPS, action="append", help="run just these sweeps"
    )
    ap.add_argument(
        "--jpg-grid",
        action="store_true",
        help="EPSGpair on the grid the repo's jpgs used: 11 points 0.1 ms "
        "apart, unrounded",
    )
    ap.add_argument(
        "--per-stimulus",
        action="store_true",
        help="also write one figure per stimulus and node",
    )
    ap.add_argument(
        "--out-dir", help="save PNGs and PDFs here instead of showing them"
    )
    ap.add_argument(
        "--workers",
        type=int,
        help="processes for sweep points (default: all CPUs)",
    )
    ap.add_argument(
        "--points", type=int, help="run only the first N points of each sweep"
    )
    ap.add_argument(
        "--replot",
        metavar="JSON",
        help="redraw the figure from a previous run's JSON, beside it, "
        "without simulating",
    )
    a = ap.parse_args()
    a.node = a.node or list(NODES)
    if a.replot:
        from msoaxon import plotting

        plotting.agg()
        for p in plotting.save(draw(plotting.load(a.replot)), a.replot):
            print(f"saved {p}")
    elif a.workers == 1:  # in-process, as binary_search(workers=1) does
        run(a, None)
    else:
        with msoaxon.process_pool(a.workers) as pool:  # shared by every sweep
            run(a, pool)


def draw(d: dict[str, Any]) -> Any:
    """Draws a figure from its data (or its JSON).

    Args:
        d: Either the composite's panels and nodes (see
            plotting.threshold_grid), or one sweep's x, multi, two, title,
            xlabel, ceiling, resolution and unit.

    Returns:
        The figure.
    """
    from msoaxon import plotting

    if "panels" in d:
        return plotting.threshold_grid(d["panels"], d["nodes"])
    return plotting.threshold_figure(
        d["x"],
        d["multi"],
        d["two"],
        d["title"],
        d["xlabel"],
        ceiling=d["ceiling"],
        resolution=d["resolution"],
        unit=d["unit"],
    )


def panel(stim: str, jpg_grid: bool, x: np.ndarray) -> dict[str, Any]:
    """Returns how the composite draws one stimulus (without thresholds).

    Args:
        stim: The stimulus swept.
        jpg_grid: Whether EPSGpair runs on the jpg grid.
        x: threshold.sweep_x per point.

    Returns:
        The panel entry of plotting.threshold_grid, its thresholds empty.
    """
    s = SWEEPS[stim]
    return {
        "title": s.title,
        "xlabel": s.xlabel,
        "x_unit": s.x_unit,
        "x_scale": s.x_scale,
        "unit": s.unit,
        "ceiling": s.max_I,
        "resolution": resolution(stim, jpg_grid),
        "mark_min": s.mark_min,
        "x": x.tolist(),
        "thresholds": {},
    }


def run(
    a: argparse.Namespace, pool: futures.ProcessPoolExecutor | None
) -> None:
    """Runs the threshold sweeps and draws or saves the figures.

    Args:
        a: The parsed command-line arguments.
        pool: The process pool every sweep shares, or None to run
            in-process.
    """
    # imported here rather than at the top: pool workers re-import this script,
    # and only the main process draws (msoaxon.plotting imports matplotlib)
    import matplotlib.pyplot as plt

    from msoaxon import plotting

    stims = a.only or list(SWEEPS)
    out = pathlib.Path(a.out_dir) if a.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    def sweep(
        stim: str, node: int
    ) -> tuple[np.ndarray, list[float], list[float]]:
        n, factor, max_I = SWEEPS[stim][:3]
        kw, dt = {}, 1 / 25
        if stim == "EPSGpair":
            n, kw = epsgpair_grid(a.jpg_grid)
            dt = kw["epsg_pair_dt"]
        n = min(n, a.points) if a.points else n
        x_values = np.array(
            [threshold.sweep_x(stim, i, dt) for i in range(1, n + 1)]
        )
        multi, two = threshold.binary_search(
            stim,
            n,
            node,
            factor,
            max_I,
            workers=a.workers,
            executor=pool,
            **kw,
        )
        return x_values, multi, two

    # binary_search maps its points onto the pool and waits for them, so called
    # one sweep at a time it would drain the pool between sweeps. From one
    # thread per sweep, every sweep's points are queued at once (submitting to
    # the pool is thread-safe) and the pool stays busy until the last point;
    # the figures are still drawn here, in order. Without a pool the sweeps run
    # in-process, one after another.
    combos = [(stim, node) for stim in stims for node in a.node]
    if pool is None:
        sweeps = {c: sweep(*c) for c in combos}
    else:
        with futures.ThreadPoolExecutor(len(combos)) as threads:
            jobs = {c: threads.submit(sweep, *c) for c in combos}
            sweeps = {c: job.result() for c, job in jobs.items()}

    panels = {}
    for stim, node in combos:
        x_values, multi, two = sweeps[stim, node]
        prefix = f"{stim} node {node}"
        print(f"{prefix}: multi={multi}\n{' ' * len(prefix)}  two  ={two}")
        if stim not in panels:
            panels[stim] = panel(stim, a.jpg_grid, x_values)
        panels[stim]["thresholds"][str(node)] = {
            "multi": list(multi),
            "two": list(two),
        }
        if not a.per_stimulus:
            continue
        s = SWEEPS[stim]
        data = {
            "stim": stim,
            "node": node,
            "x": (x_values * s.x_scale).tolist(),
            "multi": list(multi),
            "two": list(two),
            "title": f"{s.title}, node {node}",
            "xlabel": plotting.units(s.xlabel, s.x_unit),
            "ceiling": s.max_I,
            "resolution": resolution(stim, a.jpg_grid),
            "unit": s.unit,
        }
        fig = draw(data)
        if out:
            path = out / f"{stim}_thresholds_cpt{node}.png"
            for p in plotting.save(fig, path, data=data, args=vars(a)):
                print(f"saved {p}")
    data = {"nodes": a.node, "panels": panels}
    fig = draw(data)
    if out:
        path = out / "thresholds_overview.png"
        for p in plotting.save(fig, path, data=data, args=vars(a)):
            print(f"saved {p}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
