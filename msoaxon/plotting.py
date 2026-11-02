"""Figures (port of Graphing.m) and the style every script draws with.

`graph` is a dict with the same keys as the MATLAB struct: Type, node, tEnd,
stimType, inputNode, start, stop, I, Syn.

Every figure comes from `figure()` (3.4 or 7.0 in wide, in the style of
msoaxon.mplstyle) and is written by `save()` (PNG at 300 dpi plus PDF).
Scripts that open worker pools import this module, and call `agg()`, only
where they draw: pool workers re-import the running script, and only the
main process needs matplotlib.
"""

import pathlib
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from msoaxon import _common, two

STYLE = pathlib.Path(__file__).with_name("msoaxon.mplstyle")
# figure widths for a two-column paper [in]
WIDTHS = {"single": 3.4, "double": 7.0}
# units as the scripts spell them in ASCII, and the glyphs a figure shows
_GLYPHS = {"us": "\N{MICRO SIGN}s", "MOhm": "M\N{GREEK CAPITAL LETTER OMEGA}"}
_FORMATS = (".png", ".pdf", ".svg")


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


def save(
    fig: matplotlib.figure.Figure, path: str | pathlib.Path, svg: bool = False
) -> list[pathlib.Path]:
    """Writes a figure as PNG and PDF (and SVG on request), then closes it.

    Args:
        fig: The figure.
        path: The file name without its extension; a .png, .pdf or .svg
            suffix is dropped, so `out.png` writes out.png and out.pdf.
        svg: Also write an SVG.

    Returns:
        The paths written, PNG first.
    """
    base = pathlib.Path(path)
    if base.suffix.lower() in _FORMATS:
        base = base.with_suffix("")
    kinds = ("png", "pdf", "svg") if svg else ("png", "pdf")
    paths = [base.with_name(f"{base.name}.{kind}") for kind in kinds]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


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
) -> None:
    ax.plot(t, V)
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
        )
        _panel(a2, t2, x[:, :2], "two-compartment", ["Soma", "node"], t_end)
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
        )
        _panel(
            a2,
            t3,
            z[:, :2],
            "Two-Compartment (Node 5)",
            ["Soma", "node"],
            t_end,
        )
        fig.suptitle("Comparison of 2 Compartment Models")

    elif kind == "NodeComparison":
        a1, a2 = fig.subplots(2, 1)
        labels = ["Multi-Compartment", "2 Compartment"]
        a1.plot(t1, y[:, 0])
        _panel(a1, t2, x[:, 0], "Soma", labels, t_end)
        a2.plot(t1, y[:, node - 1])
        _panel(a2, t2, x[:, 1], f"node {node}", labels, t_end)
        fig.suptitle(f"Comparison of Soma and node {node}")

    elif kind == "MultiCompartment":
        _panel(
            fig.add_subplot(),
            t1,
            y[:, [0, node - 1]],
            f"Multi-compartment model, node: {node}",
            ["Soma", f"node {node}"],
            t_end,
        )

    elif kind == "TwoCompartment":
        _panel(
            fig.add_subplot(),
            t1,
            y[:, :2],
            "two-compartment",
            ["Soma", "node"],
            t_end,
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
        )

    elif kind == "Multigraph":
        gs = fig.add_gridspec(2, 3)
        cols = [1, 3, 5, 9, 13, 21, 29, 37, 45]
        _panel(
            fig.add_subplot(gs[0, :]),
            t1,
            y[:, [c - 1 for c in cols]],
            "multi-compartment",
            ["Soma"] + [f"node {c}" for c in cols[1:]],
            t_end,
        )
        _panel(
            fig.add_subplot(gs[1, 0]),
            t2,
            x[:, :2],
            "Two-Compartment (Node 3)",
            ["Soma", "node"],
            t_end,
        )
        _panel(
            fig.add_subplot(gs[1, 1]),
            t3,
            z[:, :2],
            "Two-Compartment (Node 5)",
            ["Soma", "node"],
            t_end,
        )
        _panel(
            fig.add_subplot(gs[1, 2]),
            t2,
            input_trace(graph, t2, x),
            "Input",
            [],
            t_end,
            ylabel="Current (pA)",
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
        ax, x_values, multi, ceiling, label="Multi-Compartment"
    )
    hit |= plot_thresholds(ax, x_values, two, ceiling, label="Two-Compartment")
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
    ax.set(title=title, xlabel=xlabel, ylabel="Threshold")
    ax.legend()
    return fig
