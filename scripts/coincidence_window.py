"""Coincidence-detection windows vs Myoga et al 2014, with and without recalibration.

    uv run scripts/coincidence_window.py --out-dir figures/coincidence

Runs three models (45-compartment; two-compartment as in TwoCpt.m; two-compartment
recalibrated to Goldwyn et al 2019's passive targets) with the model's EPSG
(decay 0.18 ms) and Myoga's (0.3 ms), finds each spike-probability half-width
directly (coincidence.window: fixed input, bisection over delay), and checks it
against noisy trials. Threshold curves are drawn on a 50 us grid.
"""  # noqa: E501 (printed verbatim as --help)

import argparse
import pathlib
from concurrent import futures
from typing import Any

import numpy as np

import msoaxon
from msoaxon import coincidence, synaptic, two

MYOGA_US = (
    221  # AP-probability half-width without inhibition (Myoga et al 2014)
)
MARGINS = (
    0.005,
    0.03,
)  # inputs this far above coincident threshold (Myoga: "200 pS (~3%)")
KINETICS = {
    "model EPSG (decay 0.18 ms)": synaptic.EPSG_TAU,
    "Myoga EPSG (decay 0.3 ms)": (0.1, 0.3),
}
# the noisy-trial check's (amplitude CV, onset jitter [ms]) per EPSG: small
# noise keeps the peak probability near 100%, as in Myoga's protocol;
# probability_trials' defaults (3%, 15 us) keep it well below (PYTHON.md
# "Coincidence window")
NOISE = {"small": (0.01, 0.005), "default": (0.03, 0.015)}
_goldwyn = {k: v for k, v in two.GOLDWYN_2019.items() if k != "v0"}
CONFIGS = {
    "45-compartment": ("multi", -68.0, {}),
    "2-cpt, TwoCpt.m (10 MOhm, 0.71 ms, -68 mV)": ("two", -68.0, {}),
    "2-cpt, Goldwyn 2019 (8.5 MOhm, 0.34 ms, -58 mV)": (
        "two",
        two.GOLDWYN_2019["v0"],
        _goldwyn,
    ),
}


def noisy_half_width(delays: np.ndarray, prob: np.ndarray) -> float:
    """Returns the full width at half the peak probability [us].

    Args:
        delays: The delays of the trials, from 0 up [ms].
        prob: The spike probability at each delay.

    Returns:
        Twice the delay where the probability falls to half its peak,
        linearly interpolated between the two delays around it; nan when
        it never falls to half within the delays, or never rises.
    """
    p_half = prob.max() / 2
    below = np.nonzero(prob < p_half)[0]
    if len(below) == 0 or below[0] == 0:
        return float("nan")
    j = below[0]
    return float(
        2
        * np.interp(p_half, [prob[j], prob[j - 1]], [delays[j], delays[j - 1]])
        * 1e3
    )


def strip_rows(
    direct: dict[tuple[str, str], dict[float, float]], colours: list[str]
) -> list[tuple[str, float, str]]:
    """Returns the bars of the window strip: each model's 3% window.

    Args:
        direct: window()'s half-widths [us] by (model, kinetics) and margin.
        colours: One colour per model, in CONFIGS order.

    Returns:
        (short model name, width [us], colour) per model, model EPSG.
    """
    kname = next(iter(KINETICS))
    return [
        (cname.split(" (")[0], direct[cname, kname][0.03], c)
        for cname, c in zip(CONFIGS, colours, strict=True)
    ]


def draw(summary: dict[str, Any]) -> Any:
    """Draws the threshold curves and the probability panel from the JSON.

    Args:
        summary: What run() writes to coincidence_window.json.

    Returns:
        The figure.
    """
    from msoaxon import plotting

    delays = np.array(summary["curve_delays_ms"])
    curves = {
        (c, k): np.array(summary["curves"][f"{c} | {k}"])
        for c in CONFIGS
        for k in KINETICS
    }
    direct = {
        (r["model"], r["epsg"]): {m: r[f"margin_{m}"] for m in MARGINS}
        for r in summary["half_widths_us"]
    }
    noisy = summary["noisy_check"]
    mc_delays, prob = (
        np.array(noisy["delays_ms"]),
        np.array(noisy["probability"]),
    )
    mc_width, trials = noisy["half_width_us"], noisy["n_trials"]
    # the default-noise check is absent from JSONs written before it existed
    louder = summary.get("noisy_check_default_noise")

    fig = plotting.figure("double", height=4.4)
    gs = fig.add_gridspec(2, 2, height_ratios=(4, 1))
    a1 = fig.add_subplot(gs[:, 0])
    a2 = fig.add_subplot(gs[0, 1])
    strip = fig.add_subplot(gs[1, 1], sharex=a2)
    colours = [plotting.COLOURS[k] for k in ("multi", "two", "goldwyn")]
    plotting.threshold_ratio_panel(
        a1,
        delays,
        [
            (
                curves[(cname, kname)],
                {
                    "color": c,
                    "ls": ls,
                    "label": f"{cname.split(' (')[0]}, {kname.split(' (')[0]}",
                },
            )
            for c, cname in zip(colours, CONFIGS, strict=True)
            for kname, ls in zip(KINETICS, ("-", "--"), strict=True)
        ],
        MARGINS,
    )
    a1.set(
        xlim=(0, 600),
        title="Threshold rises with delay\n(dotted: 0.5% and 3% input margins)",
    )
    # below the panel: six entries do not fit beside the curves at 7 in
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    plotting.panel_label(a1, "A")

    sym = np.concatenate([-mc_delays[:0:-1], mc_delays]) * 1e3
    p_sym = np.concatenate([prob[:0:-1], prob])
    lo, hi = plotting.wilson_interval(p_sym, trials)
    a2.errorbar(
        sym,
        p_sym,
        yerr=[p_sym - lo, hi - p_sym],
        fmt="o-",
        color=colours[1],
        capsize=2,
        label=plotting.glyphs(
            f"2-cpt TwoCpt.m, noisy trials (FWHM {mc_width:.0f} us;\n"
            f"{trials} trials per delay, 95% Wilson interval)"
        ),
    )
    if louder is not None:
        p_loud = np.array(louder["probability"])
        p_loud = np.concatenate([p_loud[:0:-1], p_loud])
        lo, hi = plotting.wilson_interval(p_loud, louder["n_trials"])
        a2.errorbar(
            sym,
            p_loud,
            yerr=[p_loud - lo, hi - p_loud],
            fmt="s--",
            color=colours[1],
            markerfacecolor="none",
            capsize=2,
            label=plotting.glyphs(
                f"same, {louder['amp_cv']:.0%} amplitude CV and "
                f"{louder['jitter_ms'] * 1e3:.0f} us jitter "
                f"(FWHM {louder['half_width_us']:.0f} us)"
            ),
        )
    for ax in (a2, strip):
        ax.axvspan(
            -MYOGA_US / 2,
            MYOGA_US / 2,
            color="0.85",
            zorder=0,
            label=plotting.glyphs(f"Myoga 2014 in vitro: {MYOGA_US} us"),
        )
    a2.set(
        xlim=(-320, 320),
        ylim=(-0.03, 1.05),
        ylabel="Spike probability",
        title="Coincidence window at half maximum (model EPSG)",
    )
    a2.tick_params(labelbottom=False)
    # window()'s widths in their own strip: a bar's height carries no data
    plotting.width_strip(strip, strip_rows(direct, colours))
    strip.set_xlabel(plotting.units("Delay between EPSGs", "us"))
    strip.set_title("window(), 3% margin", fontsize=7, loc="left")
    # below the strip, as panel A's is below A
    strip.legend(
        *a2.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.9),
    )
    plotting.panel_label(a2, "B")
    return fig


def main() -> None:
    """Parses the command line and runs the coincidence-window comparison."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", default="figures/coincidence")
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument(
        "--trials",
        type=int,
        default=200,
        help="noisy trials per delay for the check",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="coarse grids and 1%% threshold tolerance: a smoke test, "
        "not results",
    )
    ap.add_argument(
        "--replot",
        metavar="JSON",
        help="redraw the figure from a previous run's JSON, beside it, "
        "without simulating",
    )
    a = ap.parse_args()
    if a.replot:
        from msoaxon import plotting

        plotting.agg()
        for path in plotting.save(draw(plotting.load(a.replot)), a.replot):
            print(f"saved {path}")
        return
    # one pool for every parallel call below
    with msoaxon.process_pool() as pool:
        run(a, pool)


def run(a: argparse.Namespace, pool: futures.ProcessPoolExecutor) -> None:
    """Runs the coincidence-window comparison and saves the figure and JSON.

    Args:
        a: The parsed command-line arguments.
        pool: The process pool every parallel call shares.
    """
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tol = 1e-2 if a.quick else 1e-4  # threshold tolerance
    delay_tol = 1e-3 if a.quick else 1e-4  # window boundary tolerance [ms]
    combos = [(c, k) for c in CONFIGS for k in KINETICS]

    # The widths come from window() and the curves are only drawn, so each
    # combination needs the delay-0 threshold exactly once: it goes to the pool
    # first, and then feeds both the windows (threshold0=) and the curve. The
    # windows are submitted before the curve points so that their chain (delay
    # 0, then the bisection over delay) starts early; the pool is never left
    # waiting on one combination at a time.
    def th_job(cname: str, kname: str, delay: float) -> futures.Future:
        model, v0, kw = CONFIGS[cname]
        return pool.submit(
            coincidence.threshold,
            model,
            float(delay),
            a.node,
            v0,
            KINETICS[kname],
            kw,
            tol,
        )

    th0_jobs = {(c, k): th_job(c, k, 0.0) for c, k in combos}
    by_job = {j: ck for ck, j in th0_jobs.items()}
    win_jobs = {}
    for th0_job in futures.as_completed(by_job):
        cname, kname = by_job[th0_job]
        model, v0, kw = CONFIGS[cname]
        th0 = th0_job.result()
        for m in MARGINS:
            win_jobs[cname, kname, m] = pool.submit(
                coincidence.window,
                model,
                m,
                node=a.node,
                v0=v0,
                epsg_tau=KINETICS[kname],
                model_kw=kw,
                rel_tol=tol,
                delay_tol=delay_tol,
                threshold0=th0,
            )
    delays = np.round(np.arange(0, 1.0001, 0.1 if a.quick else 0.05), 4)
    curve_jobs = {
        (cname, kname): [th_job(cname, kname, d) for d in delays[1:]]
        for cname, kname in combos
    }
    curves, rows, direct = {}, [], {}
    for cname, kname in combos:
        th = curves[cname, kname] = np.array(
            [th0_jobs[cname, kname].result()]
            + [j.result() for j in curve_jobs[cname, kname]]
        )
        res = [win_jobs[cname, kname, m].result() for m in MARGINS]
        th0, widths = res[0][1], [w * 1e3 for w, _ in res]
        direct[cname, kname] = dict(zip(MARGINS, widths, strict=True))
        rows.append((cname, kname, th0, widths))
        print(
            f"{cname} | {kname}: threshold(0) {th0:.2f}, "
            f"plateau/zero-delay {th[-1] / th0:.2f}, "
            + ", ".join(
                f"half-width @{m:.1%} {w:.1f} us"
                for m, w in zip(MARGINS, widths, strict=True)
            ),
            flush=True,
        )

    # check window() against real noisy trials (2-cpt TwoCpt.m, model EPSG, 3%
    # margin), at two noise levels: small noise, so probability peaks near
    # 100% as in Myoga's protocol, and probability_trials' default, which
    # keeps the peak well below 100% (PYTHON.md "Coincidence window")
    cname, kname = list(CONFIGS)[1], list(KINETICS)[0]
    model, v0, kw = CONFIGS[cname]
    amp = (1 + 0.03) * rows[combos.index((cname, kname))][2]
    mc_delays = np.round(np.arange(0, 0.3001, 0.1 if a.quick else 0.02), 4)
    direct_us = direct[cname, kname][0.03]
    noisy_checks = {}
    for key, (amp_cv, jitter) in NOISE.items():
        prob = coincidence.probability_trials(
            model,
            mc_delays,
            amp,
            n_trials=a.trials,
            amp_cv=amp_cv,
            jitter=jitter,
            node=a.node,
            v0=v0,
            epsg_tau=KINETICS[kname],
            model_kw=kw,
            executor=pool,
        )
        mc_width = noisy_half_width(mc_delays, prob)
        print(
            f"noisy-trial check ({cname}, {kname}, 3%, {amp_cv:.0%} amplitude "
            f"CV, {jitter * 1e3:.0f} us jitter): "
            f"peak probability {prob.max():.2f}, "
            f"half-width {mc_width:.0f} us vs window() {direct_us:.0f} us",
            flush=True,
        )
        noisy_checks[key] = {
            "half_width_us": mc_width,
            "window_us": direct_us,
            "peak_probability": float(prob.max()),
            "amp_cv": amp_cv,
            "jitter_ms": jitter,
            "delays_ms": mc_delays.tolist(),
            "probability": prob.tolist(),
            "n_trials": a.trials,
        }

    summary = {
        "curve_delays_ms": delays.tolist(),
        "margins": MARGINS,
        "myoga_half_width_us": MYOGA_US,
        "half_width_method": (
            "coincidence.window: fixed input, bisection over delay"
        ),
        "curves": {f"{c} | {k}": v.tolist() for (c, k), v in curves.items()},
        "half_widths_us": [
            {
                "model": c,
                "epsg": k,
                "threshold0": t0,
                **{f"margin_{m}": w for m, w in zip(MARGINS, ws, strict=True)},
            }
            for c, k, t0, ws in rows
        ],
        "noisy_check": noisy_checks["small"],
        "noisy_check_default_noise": noisy_checks["default"],
    }
    # imported here rather than at the top: pool workers re-import this script,
    # and only the main process draws
    from msoaxon import plotting

    plotting.agg()
    plotting.save(
        draw(summary), out / "coincidence_window", data=summary, args=vars(a)
    )
    print(f"saved {out}/coincidence_window.png, .pdf and .json")


if __name__ == "__main__":
    main()
