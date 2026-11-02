"""Figures (port of Graphing.m) and the style every script draws with.

`graph` is a dict with the same keys as the MATLAB struct: Type, node, tEnd,
stimType, inputNode, start, stop, I, Syn.

Every figure comes from `figure()` (3.4 or 7.0 in wide, in the style of
msoaxon.mplstyle) and is written by `save()` (PNG at 300 dpi plus PDF, and
with `data=` a JSON beside them that the script's --replot redraws from).
Scripts that open worker pools import this module, and call `agg()`, only
where they draw: pool workers re-import the running script, and only the
main process needs matplotlib.
"""

import functools
import importlib.metadata
import json
import pathlib
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from msoaxon import _common, _fast, two

STYLE = pathlib.Path(__file__).with_name("msoaxon.mplstyle")
# figure widths for a two-column paper [in]
WIDTHS = {"single": 3.4, "double": 7.0}
# units as the scripts spell them in ASCII, and the glyphs a figure shows
_GLYPHS = {"us": "\N{MICRO SIGN}s", "MOhm": "M\N{GREEK CAPITAL LETTER OMEGA}"}
_FORMATS = (".png", ".pdf", ".svg")
# One colour per model, colour-blind safe (Okabe-Ito), the same in every
# figure. A variant of a model (KLT replaced or blocked, rest rebalanced,
# dendrites added) keeps its model's colour and takes a line style from
# VARIANT_STYLES; it never borrows another model's colour.
COLOURS = {
    "multi": "#0072B2",  # 45-compartment (msoAxon.m)
    "two": "#D55E00",  # 2-compartment (TwoCpt.m)
    "goldwyn": "#009E73",  # 2-compartment recalibrated to Goldwyn 2019
}
VARIANT_STYLES = ("-", "--", ":", "-.")
GREY = "#666666"  # references and inputs, which are no model
SOMA = "k"  # the soma, against a model's axon compartments
# an ordered parameter: one sequential map, without viridis' palest yellow,
# which vanishes on white
SEQUENTIAL = "viridis"
_SEQUENTIAL_SPAN = (0.0, 0.85)


def use_style() -> None:
    """Applies msoaxon.mplstyle to matplotlib's rcParams."""
    plt.style.use(STYLE)


def agg() -> None:
    """Selects the non-interactive Agg backend and applies the style.

    The scripts that only write files call this where they draw, after
    importing this module there (not at the top of the script; see the
    module docstring).
    """
    mpl.use("Agg")
    use_style()


def figure(
    width: str | float = "double", height: float | None = None, **kw: Any
) -> matplotlib.figure.Figure:
    """Returns an empty figure at a paper width, in the shared style.

    Args:
        width: "single" (3.4 in, one column), "double" (7.0 in, the page)
            or a width in inches.
        height: The height [in]; default three quarters of the width.
        **kw: Passed to matplotlib.pyplot.figure.

    Returns:
        The figure, with constrained layout (from the style).
    """
    use_style()
    w = WIDTHS[width] if isinstance(width, str) else float(width)
    return plt.figure(figsize=(w, 0.75 * w if height is None else height), **kw)


@functools.lru_cache(maxsize=1)
def version() -> str:
    """Returns `git describe` of the checkout, else the package version.

    The scripts run from a checkout; an installed package, a missing git
    (the tests run scripts with PATH="") or a failing git falls back to
    the version msoaxon was installed as.
    """
    git = shutil.which("git")
    if git is not None:
        try:
            r = subprocess.run(  # noqa: S603 (fixed arguments, no shell)
                [git, "describe", "--tags", "--always", "--dirty"],
                cwd=pathlib.Path(__file__).parent,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            if r.stdout.strip():
                return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return f"msoaxon {importlib.metadata.version('msoaxon')}"


def compact(values: Any, digits: int = 6) -> Any:
    """Rounds an array to `digits` significant digits, as nested lists.

    For the arrays a figure draws (traces, waveforms) that its JSON keeps:
    more digits than this are not visible, and the files are shared by
    hand.

    Args:
        values: A number or an array-like of numbers.
        digits: Significant digits kept.

    Returns:
        The rounded values, as a float or (nested) list; nan stays nan.
    """
    a = np.asarray(values, dtype=float)
    flat = [float(f"{v:.{digits}g}") for v in a.ravel()]
    return np.array(flat).reshape(a.shape).tolist()


def thin(n: int, max_points: int = 2000) -> slice:
    """Returns the stride that keeps at most max_points of n samples.

    The solvers' adaptive steps cluster where the voltage moves, so a
    uniform stride in sample index keeps the spikes' share of points; at
    most about 2000 per trace is what a 7 in figure at 300 dpi shows.

    Args:
        n: The number of samples.
        max_points: The most to keep.

    Returns:
        A slice for the time axis.
    """
    return slice(None, None, max(1, -(-n // max_points)))


def meta(args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Returns what a figure's JSON records about the run that made it.

    Args:
        args: The script's parsed arguments (vars(namespace)).

    Returns:
        dict(script, args, version, mode): the script's file name, its
        arguments (paths as strings), version() and the model mode the
        run used.
    """
    args = {
        k: str(v) if isinstance(v, pathlib.PurePath) else v
        for k, v in (args or {}).items()
    }
    return {
        "script": pathlib.Path(sys.argv[0]).name,
        "args": args,
        "version": version(),
        "mode": _fast.effective(args.get("mode")),
    }


def save(
    fig: matplotlib.figure.Figure,
    path: str | pathlib.Path,
    svg: bool = False,
    data: dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
) -> list[pathlib.Path]:
    """Writes a figure as PNG and PDF (and SVG on request), then closes it.

    This is the only place a figure is written. With `data`, it also
    writes `<name>.json` beside the images: the data plus a "meta" entry
    (see meta()), which the script's --replot reads to redraw the figure
    without simulating.

    Args:
        fig: The figure.
        path: The file name without its extension; a .png, .pdf or .svg
            (or .json) suffix is dropped, so `out.png` writes out.png and
            out.pdf.
        svg: Also write an SVG.
        data: What the figure draws (and any results kept with it), as
            JSON-ready values; None writes no JSON (a replot).
        args: The script's arguments, recorded in the JSON's meta.

    Returns:
        The paths written, PNG first (the JSON last).
    """
    base = pathlib.Path(path)
    if base.suffix.lower() in (*_FORMATS, ".json"):
        base = base.with_suffix("")
    kinds = ("png", "pdf", "svg") if svg else ("png", "pdf")
    paths = [base.with_name(f"{base.name}.{kind}") for kind in kinds]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    if data is not None:
        paths.append(base.with_name(f"{base.name}.json"))
        paths[-1].write_text(json.dumps({**data, "meta": meta(args)}, indent=1))
    return paths


def load(path: str | pathlib.Path) -> dict[str, Any]:
    """Reads a figure's JSON (see save()) for a replot.

    Args:
        path: The JSON, or its figure (same name, .json beside it).

    Returns:
        The data, with its "meta" entry.
    """
    return json.loads(pathlib.Path(path).with_suffix(".json").read_text())


def units(label: str, unit: str) -> str:
    """Returns an axis label with its unit, e.g. "Delay (\N{MICRO SIGN}s)".

    Args:
        label: The quantity.
        unit: Its unit; "us" and "MOhm" become their glyphs, anything else
            (pA, nS, ms, Hz, mV) is used as it is.

    Returns:
        "label (unit)".
    """
    return f"{label} ({_GLYPHS.get(unit, unit)})"


def model_style(model: str, variant: int = 0) -> dict[str, str]:
    """Returns a model's colour and a line style for one of its variants.

    Args:
        model: A key of COLOURS ("multi", "two" or "goldwyn").
        variant: 0 for the model itself, 1 to 3 for its variants (dashed,
            dotted, dash-dotted).

    Returns:
        dict(color=..., ls=...), keywords for ax.plot.
    """
    return {"color": COLOURS[model], "ls": VARIANT_STYLES[variant]}


def sequential(
    values: np.ndarray | Sequence[float],
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[tuple[float, float, float, float]]:
    """Returns colours for an ordered parameter from one sequential map.

    The same value gets the same colour whenever vmin and vmax are the
    same, so a quick run (a subset of the values) matches a full one.

    Args:
        values: The parameter's values (rheobase multiple, node number,
            distance from the soma).
        vmin: The value drawn darkest (default: the smallest).
        vmax: The value drawn lightest (default: the largest).

    Returns:
        One RGBA colour per value.
    """
    v = np.asarray(values, dtype=float)
    lo = v.min() if vmin is None else vmin
    hi = v.max() if vmax is None else vmax
    frac = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    a, b = _SEQUENTIAL_SPAN
    cmap = matplotlib.colormaps[SEQUENTIAL]
    return [tuple(cmap(a + (b - a) * f)) for f in frac]


def glyphs(text: str) -> str:
    """Writes the units the scripts spell in ASCII with their glyphs.

    "us" and "um" after a number become "\N{MICRO SIGN}s" and
    "\N{MICRO SIGN}m", and "MOhm" becomes "M\N{GREEK CAPITAL LETTER OMEGA}",
    for legend and title text built from the scripts' ASCII names.

    Args:
        text: The text.

    Returns:
        The text with the glyphs.
    """
    text = re.sub(r"(?<=\d )u([sm])\b", "\N{MICRO SIGN}\\1", text)
    return text.replace("MOhm", _GLYPHS["MOhm"])


def plain_log_ticks(ax: matplotlib.axes.Axes) -> None:
    """Labels a log y axis with plain numbers (6000, not 6 x 10^3).

    Args:
        ax: A panel with a log y axis.
    """
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain)


def panel_label(ax: matplotlib.axes.Axes, letter: str) -> None:
    """Writes a bold panel letter just above an axes' top-left corner.

    Args:
        ax: The panel.
        letter: Its label, e.g. "A".
    """
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-32, 4),
        textcoords="offset points",
        fontweight="bold",
        fontsize=9,
        va="bottom",
    )


def threshold_ratio_panel(
    ax: matplotlib.axes.Axes,
    delays_ms: np.ndarray | Sequence[float],
    curves: Iterable[tuple[np.ndarray | Sequence[float], dict[str, Any]]],
    margins: Sequence[float] = (0.03,),
) -> None:
    """Draws EPSG-pair thresholds over the coincident one against delay.

    The panel coincidence_window.py (A) and dendrites.py (E) share: each
    curve divided by its zero-delay threshold, with a dotted line at
    1 + margin for each input margin.

    Args:
        ax: The panel.
        delays_ms: The delays between the EPSGs [ms]; the first is 0.
        curves: (thresholds at those delays, keywords for ax.plot) per
            curve.
        margins: The input margins to mark (0.03 is 3% above the
            coincident threshold).
    """
    x = np.asarray(delays_ms) * 1e3
    for curve, kw in curves:
        th = np.asarray(curve)
        ax.plot(x, th / th[0], **kw)
    for m in margins:
        ax.axhline(1 + m, color="0.5", lw=0.8, ls=":")
    ax.set(
        xlabel=units("Delay between EPSGs", "us"),
        ylabel="Threshold / coincident threshold",
        ylim=(0.98, 2.05),
    )


def _panel(
    ax: matplotlib.axes.Axes,
    t: np.ndarray,
    V: np.ndarray,
    title: str,
    labels: list[str],
    window: tuple[float, float],
    ylabel: str = "Voltage (mV)",
    colours: Sequence[Any] = (),
) -> None:
    lines = ax.plot(t, V)
    for line, colour in zip(lines, colours, strict=False):
        line.set_color(colour)
    if colours and colours[0] == SOMA and len(lines) > 1:
        # the soma thin and on top: the axon trace would hide it
        lines[0].set(zorder=3, lw=0.9)
    ax.set_title(title)
    if labels:
        ax.legend(labels)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*window)


# an event figure shows EVENT_SPAN around the first time a trace rises
# EVENT_RISE above where it started
EVENT_RISE = 10.0  # [mV]
EVENT_SPAN = (1.0, 4.0)  # before and after the crossing [ms]


def event_window(
    t: np.ndarray, V: np.ndarray, t_end: float, start: float = 0.0
) -> tuple[float, float]:
    """Returns the time window around a run's event.

    Args:
        t: Times [ms].
        V: Voltages, one column per trace [mV].
        t_end: The run's end [ms].
        start: Where the window centres when no trace rises EVENT_RISE
            (the stimulus onset).

    Returns:
        (t0, t1): EVENT_SPAN around the first crossing, within the run.
    """
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float).reshape(len(t), -1)
    hit = np.nonzero(((V - V[0]) >= EVENT_RISE).any(axis=1))[0]
    tc = float(t[hit[0]]) if hit.size else float(start)
    return max(0.0, tc - EVENT_SPAN[0]), min(float(t_end), tc + EVENT_SPAN[1])


def ceiling_mask(
    values: np.ndarray | Sequence[float], ceiling: float
) -> np.ndarray:
    """Marks the thresholds that are a search's ceiling, not a measurement.

    threshold.binary_search returns its ceiling (max_I) when nothing fires
    below it, as BinarySearch.m does: such a point means "no threshold
    found", not "threshold = ceiling".

    Args:
        values: The thresholds.
        ceiling: The search's ceiling.

    Returns:
        A boolean array, True where a value is at (or above) the ceiling.
    """
    return np.asarray(values, dtype=float) >= ceiling


def plot_thresholds(
    ax: matplotlib.axes.Axes,
    x: np.ndarray | Sequence[float],
    values: np.ndarray | Sequence[float],
    ceiling: float | None = None,
    **kw: Any,
) -> np.ndarray:
    """Draws one threshold curve, with ceiling points drawn apart.

    Every point gets a marker, so the search's rounding reads as a
    resolution rather than as structure. A point at the ceiling (see
    ceiling_mask) is a hollow marker on the ceiling and the line breaks
    there instead of joining it to its neighbours.

    Args:
        ax: The panel.
        x: What the sweep varies, per point.
        values: The thresholds.
        ceiling: The search's ceiling, or None to mark nothing.
        **kw: Passed to ax.plot for the curve (label, colour, style).

    Returns:
        The ceiling mask (all False when ceiling is None).
    """
    x, v = np.asarray(x, dtype=float), np.asarray(values, dtype=float)
    hit = (
        np.zeros(v.shape, dtype=bool)
        if ceiling is None
        else ceiling_mask(v, ceiling)
    )
    (line,) = ax.plot(x, np.where(hit, np.nan, v), marker="o", ms=3, **kw)
    if hit.any():
        ax.plot(
            x[hit],
            np.full(hit.sum(), float(ceiling)),
            ls="none",
            marker="o",
            ms=5,
            mfc="none",
            color=line.get_color(),  # not the next colour in the cycle
        )
    return hit


def colour_limits(
    values: np.ndarray | Sequence[float], min_span: float = 0.1
) -> tuple[float, float]:
    """Returns colour-scale limits that never stretch a tiny spread.

    An autoscaled colour bar maps any spread, however small, onto the
    whole colour map; here the limits span at least min_span times the
    median's magnitude, centred on the data.

    Args:
        values: The values the colours show.
        min_span: The narrowest span, as a fraction of |median|.

    Returns:
        (vmin, vmax).
    """
    v = np.asarray(values, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, min_span * abs(float(np.median(v)))) or 1.0
    mid = (lo + hi) / 2
    return mid - span / 2, mid + span / 2


def wilson_interval(
    p: np.ndarray | Sequence[float], n: int, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the Wilson score interval of binomial proportions.

    Args:
        p: Observed proportions (successes / n).
        n: The number of trials behind each.
        z: The normal quantile (1.96: a 95% interval).

    Returns:
        (lower, upper) bounds, one per proportion.
    """
    p = np.asarray(p, dtype=float)
    z2n = z * z / n
    centre = (p + z2n / 2) / (1 + z2n)
    half = z / (1 + z2n) * np.sqrt(p * (1 - p) / n + z2n / (4 * n))
    return centre - half, centre + half


def width_strip(
    ax: matplotlib.axes.Axes,
    rows: Sequence[tuple[str, float, Any]],
    unit: str = "us",
) -> None:
    """Draws widths centred on 0 as labelled bars, one per row.

    For a strip under a panel with a symmetric x axis (a coincidence
    window): the bars' heights carry no data, so the y axis is hidden.

    Args:
        ax: The strip.
        rows: (label, width, colour) per bar, drawn top to bottom.
        unit: The width's unit, written after each label.
    """
    for k, (label, w, colour) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([-w / 2, w / 2], [y, y], color=colour, lw=4, alpha=0.8)
        ax.annotate(
            f"{label}: {w:.0f} {_GLYPHS.get(unit, unit)}",
            xy=(w / 2, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6,
        )
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)


def input_trace(
    graph: dict[str, Any], t2: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Computes the applied current along the two-compartment solution.

    Graphing.m recomputes this with its own copy of the stimulus code
    (and indexes the 2-CPT voltage with the multi-compartment time
    index); here it reuses the model's own stimulus function evaluated
    on the 2-CPT run instead.

    Args:
        graph: The graph settings (see the module docstring).
        t2: Times of the two-compartment run [ms].
        x: The two-compartment run's state.

    Returns:
        The applied current [pA] at each time.
    """
    s = _common.stimulus(
        graph["stimType"],
        graph["start"],
        graph["stop"],
        graph["I"],
        graph["tEnd"],
        graph["Syn"],
    )
    return np.array(
        [
            two.applied_current(t, V1, graph["stimType"], s)
            for t, V1 in zip(t2, x[:, 0], strict=True)
        ]
    )


def graph_window(
    graph: dict[str, Any],
    t1: np.ndarray,
    y: np.ndarray,
    t2: np.ndarray | None,
    x: np.ndarray | None,
    window: str | Sequence[float] | None,
) -> tuple[float, float]:
    """Resolves graphing()'s window argument to (t0, t1) [ms].

    Args:
        graph: The graph settings (Type, tEnd and start are used).
        t1: Times of the multi-compartment run [ms].
        y: Its state.
        t2: Times of the two-compartment run [ms], or None.
        x: Its state, or None.
        window: "event", None (the whole run) or (t0, t1).

    Returns:
        (t0, t1) [ms].
    """
    kind, t_end = graph["Type"], graph["tEnd"]
    if window is None:
        return 0.0, float(t_end)
    if isinstance(window, str):
        if kind in ("ModelComparison2", "Input"):
            ref_t, ref_v = t2, x[:, :2]
        elif kind == "TwoCompartment":
            ref_t, ref_v = t1, y[:, :2]
        else:
            ref_t, ref_v = t1, y[:, :45]
        return event_window(ref_t, ref_v, t_end, graph.get("start", 0.0))
    return float(window[0]), float(window[1])


def _contour(
    fig: matplotlib.figure.Figure,
    t1: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
) -> None:
    """Draws the Contour graph: a compartment-by-time heat map."""
    ax = fig.add_subplot()
    mesh = ax.pcolormesh(
        t1,
        np.arange(1, 46),
        y[:, :45].T,
        cmap=SEQUENTIAL,
        shading="nearest",
        rasterized=True,
    )
    fig.colorbar(mesh, ax=ax, label="Voltage (mV)")
    ax.set(
        title="45-compartment model: voltage along the cell",
        xlabel="Time (ms)",
        ylabel="Compartment (1 = soma)",
        xlim=window,
    )


def _multigraph(
    fig: matplotlib.figure.Figure,
    graph: dict[str, Any],
    runs: tuple[np.ndarray, ...],
    window: tuple[float, float],
) -> None:
    """Draws the Multigraph graph (see graphing())."""
    t1, y, t2, x, t3, z = runs
    two_title = "2-compartment model, node {}"
    gs = fig.add_gridspec(2, 3)
    cols = [1, 3, 5, 9, 13, 21, 29, 37, 45]
    top = fig.add_subplot(gs[0, :])
    # the nodes are a spatial series: one sequential map, darkest
    # nearest the soma, which is black; a colour bar gives the order
    _panel(
        top,
        t1,
        y[:, [c - 1 for c in cols]],
        "45-compartment model, soma and axon nodes",
        [],
        window,
        colours=[SOMA, *sequential(cols[1:])],
    )
    top.legend(top.lines[:1], ["Soma"], loc="upper right")
    ramp = matplotlib.colors.ListedColormap(
        sequential(np.linspace(cols[1], cols[-1], 256))
    )
    bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(
            matplotlib.colors.Normalize(cols[1], cols[-1]), ramp
        ),
        ax=top,
        label="Axon node",
    )
    bar.set_ticks(cols[1:])
    b0 = fig.add_subplot(gs[1, 0], sharey=top)
    _panel(
        b0,
        t2,
        x[:, :2],
        two_title.format(3),
        ["Soma", "Node 3"],
        window,
        colours=(SOMA, COLOURS["two"]),
    )
    _panel(
        fig.add_subplot(gs[1, 1], sharey=top),
        t3,
        z[:, :2],
        two_title.format(5),
        ["Soma", "Node 5"],
        window,
        colours=(SOMA, COLOURS["two"]),
    )
    _panel(
        fig.add_subplot(gs[1, 2]),
        t2,
        input_trace(graph, t2, x),
        "Input current",
        [],
        window,
        ylabel="Current (pA)",
        colours=[GREY],
    )
    fig.suptitle("45- vs 2-compartment model with the input")


def graphing(
    graph: dict[str, Any],
    t1: np.ndarray,
    y: np.ndarray,
    t2: np.ndarray | None = None,
    x: np.ndarray | None = None,
    t3: np.ndarray | None = None,
    z: np.ndarray | None = None,
    window: str | Sequence[float] | None = "event",
) -> matplotlib.figure.Figure:
    """Draws graph["Type"] and returns the figure (Graphing.m).

    What each Type uses:

        Contour, MultiCompartment        t1, y
        TwoCompartment                   t1, y, where y is a
                                         *two-compartment* run
        ModelComparison, NodeComparison  t1, y, t2, x
        ModelComparison2                 t2, x, t3, z
        Input                            t2, x (the current is
                                         recomputed on it)
        Multigraph                       all six

    Panels that compare voltages share their y axis. Contour is a
    compartment-by-time heat map.

    Args:
        graph: The graph settings (see the module docstring).
        t1: Times of the multi-compartment run [ms].
        y: The multi-compartment run's state.
        t2: Times of the two-compartment run at graph["node"].
        x: That run's state.
        t3: Times of a second two-compartment run (node 5 in
            Combine_all.m).
        z: That run's state.
        window: The time axis: "event" (default) for event_window()
            around the first trace to rise, None for the whole run, or
            (t0, t1) [ms].

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: For an unknown graph Type.
    """
    node, kind = graph["node"], graph["Type"]
    if kind not in _GRAPH_TYPES:
        raise ValueError(f"unknown graph Type {kind!r}")
    window = graph_window(graph, t1, y, t2, x, window)
    fig = figure("double", height=5.5)
    two_title = "2-compartment model, node {}"

    if kind == "Contour":
        _contour(fig, t1, y, window)

    elif kind == "ModelComparison":
        a1, a2 = fig.subplots(2, 1, sharex=True, sharey=True)
        _panel(
            a1,
            t1,
            y[:, [0, node - 1]],
            "45-compartment model",
            ["Soma", f"Node {node}"],
            window,
            colours=(SOMA, COLOURS["multi"]),
        )
        _panel(
            a2,
            t2,
            x[:, :2],
            two_title.format(node),
            ["Soma", f"Node {node}"],
            window,
            colours=(SOMA, COLOURS["two"]),
        )
        fig.suptitle(f"45- vs 2-compartment model, soma and node {node}")

    elif kind == "ModelComparison2":
        a1, a2 = fig.subplots(2, 1, sharex=True, sharey=True)
        _panel(
            a1,
            t2,
            x[:, :2],
            two_title.format(3),
            ["Soma", "Node 3"],
            window,
            colours=(SOMA, COLOURS["two"]),
        )
        _panel(
            a2,
            t3,
            z[:, :2],
            two_title.format(5),
            ["Soma", "Node 5"],
            window,
            colours=(SOMA, COLOURS["two"]),
        )
        fig.suptitle("2-compartment model at node 3 and node 5")

    elif kind == "NodeComparison":
        a1, a2 = fig.subplots(2, 1, sharex=True, sharey=True)
        labels = ["45-compartment", "2-compartment"]
        a1.plot(t1, y[:, 0], color=COLOURS["multi"])
        _panel(
            a1, t2, x[:, 0], "Soma", labels, window, colours=[COLOURS["two"]]
        )
        a2.plot(t1, y[:, node - 1], color=COLOURS["multi"])
        _panel(
            a2,
            t2,
            x[:, 1],
            f"Node {node}",
            labels,
            window,
            colours=[COLOURS["two"]],
        )
        fig.suptitle(f"45- vs 2-compartment model, soma and node {node}")

    elif kind == "MultiCompartment":
        _panel(
            fig.add_subplot(),
            t1,
            y[:, [0, node - 1]],
            f"45-compartment model, node {node}",
            ["Soma", f"Node {node}"],
            window,
            colours=(SOMA, COLOURS["multi"]),
        )

    elif kind == "TwoCompartment":
        _panel(
            fig.add_subplot(),
            t1,
            y[:, :2],
            "2-compartment model",
            ["Soma", "Node"],
            window,
            colours=(SOMA, COLOURS["two"]),
        )

    elif kind == "Input":
        _panel(
            fig.add_subplot(),
            t2,
            input_trace(graph, t2, x),
            "Input current",
            [],
            window,
            ylabel="Current (pA)",
            colours=[GREY],
        )

    else:  # Multigraph
        _multigraph(fig, graph, (t1, y, t2, x, t3, z), window)

    return fig


_GRAPH_TYPES = (
    "Contour",
    "ModelComparison",
    "ModelComparison2",
    "NodeComparison",
    "MultiCompartment",
    "TwoCompartment",
    "Input",
    "Multigraph",
)


def threshold_figure(
    x_values: np.ndarray | Sequence[float],
    multi: np.ndarray | Sequence[float],
    two: np.ndarray | Sequence[float],
    title: str,
    xlabel: str,
    ceiling: float | None = None,
    resolution: float | None = None,
    unit: str = "",
) -> matplotlib.figure.Figure:
    """Draws one panel of Making_Threshold_graphs.m.

    Args:
        x_values: What the sweep varies, per point.
        multi: The multi-compartment thresholds.
        two: The two-compartment thresholds.
        title: The panel title.
        xlabel: The x-axis label.
        ceiling: The search's ceiling: points at it found no threshold
            and are drawn as hollow markers on a labelled dotted line
            (see plot_thresholds), or None.
        resolution: The search's rounding, noted under the title so
            steps of that size read as resolution, or None.
        unit: The thresholds' unit, for the ceiling and resolution notes.

    Returns:
        The matplotlib figure.
    """
    fig = figure("single", height=2.6)
    ax = fig.subplots()
    hit = plot_thresholds(
        ax,
        x_values,
        multi,
        ceiling,
        label="Multi-Compartment",
        color=COLOURS["multi"],
    )
    hit |= plot_thresholds(
        ax,
        x_values,
        two,
        ceiling,
        label="Two-Compartment",
        color=COLOURS["two"],
        ls="--",  # so the 45-compartment curve shows where they agree
    )
    if hit.any():
        ax.axhline(
            ceiling,
            color="0.5",
            ls=":",
            lw=0.8,
            label=f"no threshold below {ceiling:g} {unit}".rstrip(),
        )
    if resolution is not None:
        title = f"{title}\n(search resolution {resolution:g} {unit})"
    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=units("Threshold", unit) if unit else "Threshold",
    )
    ax.legend()
    return fig


def threshold_residual(
    multi: np.ndarray | Sequence[float],
    two: np.ndarray | Sequence[float],
    ceiling: float | None = None,
) -> np.ndarray:
    """Returns the two-compartment model's thresholds relative to the 45's.

    Args:
        multi: The 45-compartment thresholds.
        two: The two-compartment thresholds at the same points.
        ceiling: The search's ceiling: a point where either model found no
            threshold has no residual (nan).

    Returns:
        (two - multi) / multi in percent, per point.
    """
    m, t = np.asarray(multi, dtype=float), np.asarray(two, dtype=float)
    r = (t - m) / m * 100
    if ceiling is not None:
        r[ceiling_mask(m, ceiling) | ceiling_mask(t, ceiling)] = np.nan
    return r


def mark_minimum(
    ax: matplotlib.axes.Axes,
    x: np.ndarray | Sequence[float],
    values: np.ndarray | Sequence[float],
    unit: str,
    ceiling: float | None = None,
    label: bool = True,
    **kw: Any,
) -> int | None:
    """Marks a curve's interior minimum with a labelled marker.

    A minimum at either end of the sweep is no optimum (the sweep may just
    stop there), and ceiling points are no thresholds, so neither is
    marked.

    Args:
        ax: The panel.
        x: What the sweep varies (as drawn).
        values: The thresholds.
        unit: x's unit, for the label.
        ceiling: The search's ceiling, or None.
        label: Write "min <x> <unit>" above the marker.
        **kw: Passed to ax.plot for the marker (its colour).

    Returns:
        The index marked, or None.
    """
    x, v = np.asarray(x, dtype=float), np.asarray(values, dtype=float)
    if ceiling is not None:
        v = np.where(ceiling_mask(v, ceiling), np.nan, v)
    if len(v) < 3 or np.isnan(v).all():
        return None
    i = int(np.nanargmin(v))
    if i in (0, len(v) - 1):
        return None
    ax.plot(x[i], v[i], marker="v", ms=6, mec="k", ls="none", **kw)
    if label:
        _min_label(ax, x[i], v[i], f"min {x[i]:g} {_GLYPHS.get(unit, unit)}")
    return i


def _min_label(ax: matplotlib.axes.Axes, x: float, y: float, text: str) -> None:
    """Writes a minimum's label above it, in the curve's valley."""
    ax.annotate(
        text,
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6,
    )


def threshold_grid(
    panels: dict[str, dict[str, Any]], nodes: Sequence[int]
) -> matplotlib.figure.Figure:
    """Draws every threshold sweep for every node, with residual strips.

    One column per stimulus and one row per node; the nodes of a column
    share their y limits, so their curves compare by eye. Under each panel
    a strip shows the two-compartment model's difference from the
    45-compartment one, in percent (threshold_residual). Ceiling points
    are hollow markers (plot_thresholds) and, where a panel asks for it,
    each model's interior minimum is marked (mark_minimum).

    Args:
        panels: Per stimulus: title, xlabel, x_unit, x_scale (x as drawn
            is x times this), unit, ceiling, resolution, mark_min, x, and
            thresholds[str(node)] = dict(multi=..., two=...).
        nodes: The nodes, one row each, top to bottom.

    Returns:
        The figure.
    """
    stims = list(panels)
    fig = figure("double", height=0.6 + 2.3 * len(nodes))
    gs = fig.add_gridspec(
        2 * len(nodes), len(stims), height_ratios=[3, 1] * len(nodes)
    )
    for j, stim in enumerate(stims):
        p = panels[stim]
        x = np.asarray(p["x"], dtype=float) * p.get("x_scale", 1)
        first = first_res = res = None
        for i, node in enumerate(nodes):
            ax = fig.add_subplot(gs[2 * i, j], sharey=first)
            res = fig.add_subplot(gs[2 * i + 1, j], sharex=ax, sharey=first_res)
            first, first_res = first or ax, first_res or res
            t = p["thresholds"][str(node)]
            hit = plot_thresholds(
                ax,
                x,
                t["multi"],
                p["ceiling"],
                label="45-compartment",
                color=COLOURS["multi"],
            )
            hit |= plot_thresholds(
                ax,
                x,
                t["two"],
                p["ceiling"],
                label="2-compartment",
                color=COLOURS["two"],
                ls="--",
            )
            if hit.any():
                ax.axhline(
                    p["ceiling"],
                    color=GREY,
                    ls=":",
                    lw=0.8,
                    label="no threshold below the ceiling",
                )
            if p.get("mark_min"):
                at = {
                    model: mark_minimum(
                        ax,
                        x,
                        t[model],
                        p["x_unit"],
                        p["ceiling"],
                        label=False,
                        color=COLOURS[model],
                    )
                    for model in ("multi", "two")
                }
                if None not in at.values():  # one label for both markers
                    u = _GLYPHS.get(p["x_unit"], p["x_unit"])
                    im, it = at["multi"], at["two"]
                    text = (
                        f"min {x[im]:g} {u}"
                        if im == it
                        else f"min 45: {x[im]:g}, 2: {x[it]:g} {u}"
                    )
                    y = min(t["multi"][im], t["two"][it])
                    _min_label(ax, x[im], y, text)
            ax.set_title(f"{p['title']}, node {node}")
            ax.set_ylabel(units("Threshold", p["unit"]))
            ax.tick_params(labelbottom=False)
            res.axhline(0, color=GREY, lw=0.6)
            res.plot(
                x,
                threshold_residual(t["multi"], t["two"], p["ceiling"]),
                marker="o",
                ms=2,
                color=COLOURS["two"],
            )
            res.set_ylabel("2 vs 45\n(%)")
            if i < len(nodes) - 1:
                res.tick_params(labelbottom=False)
        res.set_xlabel(units(p["xlabel"], p["x_unit"]))
    handles, labels = fig.axes[0].get_legend_handles_labels()
    for ax in fig.axes:  # the ceiling line may be in any panel
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    notes = sorted(
        {
            f"{p['resolution']:g} {p['unit']}"
            for p in panels.values()
            if p.get("resolution") is not None
        }
    )
    fig.suptitle(
        "Spiking thresholds, 45- vs 2-compartment model"
        + (f" (search resolution {', '.join(notes)})" if notes else "")
    )
    return fig
