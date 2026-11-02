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
import matplotlib.figure
import matplotlib.pyplot as plt
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
    t_end: float,
    ylabel: str = "Voltage (mV)",
    colours: Sequence[Any] = (),
) -> None:
    for line, colour in zip(ax.plot(t, V), colours, strict=False):
        line.set_color(colour)
    ax.set_title(title)
    if labels:
        ax.legend(labels)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, t_end)


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


def graphing(
    graph: dict[str, Any],
    t1: np.ndarray,
    y: np.ndarray,
    t2: np.ndarray | None = None,
    x: np.ndarray | None = None,
    t3: np.ndarray | None = None,
    z: np.ndarray | None = None,
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

    Args:
        graph: The graph settings (see the module docstring).
        t1: Times of the multi-compartment run [ms].
        y: The multi-compartment run's state.
        t2: Times of the two-compartment run at graph["node"].
        x: That run's state.
        t3: Times of a second two-compartment run (node 5 in
            Combine_all.m).
        z: That run's state.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: For an unknown graph Type.
    """
    node, t_end, kind = graph["node"], graph["tEnd"], graph["Type"]
    fig = figure("double", height=5.5)

    if kind == "Contour":
        ax = fig.add_subplot(projection="3d")
        X, Y = np.meshgrid(np.arange(1, 46), t1)
        ax.plot_surface(X, Y, y[:, :45], cmap="viridis", linewidth=0)
        ax.set(
            title="Voltage Propagation",
            xlabel="Compartment Number (1-45)",
            ylabel="Time",
            zlabel="Voltage (mV)",
        )

    elif kind == "ModelComparison":
        a1, a2 = fig.subplots(2, 1)
        _panel(
            a1,
            t1,
            y[:, [0, node - 1]],
            "multi-compartment",
            ["Soma", f"node {node}"],
            t_end,
            colours=(SOMA, COLOURS["multi"]),
        )
        _panel(
            a2,
            t2,
            x[:, :2],
            "two-compartment",
            ["Soma", "node"],
            t_end,
            colours=(SOMA, COLOURS["two"]),
        )
        fig.suptitle("Comparison of 2 Compartment and Multi-Compartment Models")

    elif kind == "ModelComparison2":
        a1, a2 = fig.subplots(2, 1)
        _panel(
            a1,
            t2,
            x[:, :2],
            "Two-Compartment (Node 3)",
            ["Soma", "node"],
            t_end,
            colours=(SOMA, COLOURS["two"]),
        )
        _panel(
            a2,
            t3,
            z[:, :2],
            "Two-Compartment (Node 5)",
            ["Soma", "node"],
            t_end,
            colours=(SOMA, COLOURS["two"]),
        )
        fig.suptitle("Comparison of 2 Compartment Models")

    elif kind == "NodeComparison":
        a1, a2 = fig.subplots(2, 1)
        labels = ["Multi-Compartment", "2 Compartment"]
        a1.plot(t1, y[:, 0], color=COLOURS["multi"])
        _panel(a1, t2, x[:, 0], "Soma", labels, t_end, colours=[COLOURS["two"]])
        a2.plot(t1, y[:, node - 1], color=COLOURS["multi"])
        _panel(
            a2,
            t2,
            x[:, 1],
            f"node {node}",
            labels,
            t_end,
            colours=[COLOURS["two"]],
        )
        fig.suptitle(f"Comparison of Soma and node {node}")

    elif kind == "MultiCompartment":
        _panel(
            fig.add_subplot(),
            t1,
            y[:, [0, node - 1]],
            f"Multi-compartment model, node: {node}",
            ["Soma", f"node {node}"],
            t_end,
            colours=(SOMA, COLOURS["multi"]),
        )

    elif kind == "TwoCompartment":
        _panel(
            fig.add_subplot(),
            t1,
            y[:, :2],
            "two-compartment",
            ["Soma", "node"],
            t_end,
            colours=(SOMA, COLOURS["two"]),
        )

    elif kind == "Input":
        _panel(
            fig.add_subplot(),
            t2,
            input_trace(graph, t2, x),
            "Input",
            [],
            t_end,
            ylabel="Current (pA)",
            colours=[GREY],
        )

    elif kind == "Multigraph":
        gs = fig.add_gridspec(2, 3)
        cols = [1, 3, 5, 9, 13, 21, 29, 37, 45]
        top = fig.add_subplot(gs[0, :])
        # the nodes are a spatial series: one sequential map, darkest
        # nearest the soma, which is black
        _panel(
            top,
            t1,
            y[:, [c - 1 for c in cols]],
            "multi-compartment",
            ["Soma"] + [f"node {c}" for c in cols[1:]],
            t_end,
            colours=[SOMA, *sequential(cols[1:])],
        )
        top.legend(
            ["Soma"] + [f"node {c}" for c in cols[1:]],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        _panel(
            fig.add_subplot(gs[1, 0]),
            t2,
            x[:, :2],
            "Two-Compartment (Node 3)",
            ["Soma", "node"],
            t_end,
            colours=(SOMA, COLOURS["two"]),
        )
        _panel(
            fig.add_subplot(gs[1, 1]),
            t3,
            z[:, :2],
            "Two-Compartment (Node 5)",
            ["Soma", "node"],
            t_end,
            colours=(SOMA, COLOURS["two"]),
        )
        _panel(
            fig.add_subplot(gs[1, 2]),
            t2,
            input_trace(graph, t2, x),
            "Input",
            [],
            t_end,
            ylabel="Current (pA)",
            colours=[GREY],
        )
        fig.suptitle(
            "Comparison of 2 Compartment and Multi-Compartment Models "
            "with Input"
        )

    else:
        plt.close(fig)
        raise ValueError(f"unknown graph Type {kind!r}")

    return fig


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
