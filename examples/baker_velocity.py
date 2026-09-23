#!/usr/bin/env python3
"""LOS velocity by hour of day, ice against the bedrock that scores it.

    python examples/baker_velocity.py --scenes 20170803_full 20170827 20180808

``baker_population.py`` writes, per campaign, the median LOS displacement of
the RGI ice and of the held-out bedrock, on the corrected series: reference,
per-epoch drift, per-epoch turbulence screen, then the temporal path delay of
:mod:`gpri_tools.pathdelay`.  This differences those series over a
``--window`` of hours -- at a two-minute cadence the epoch-to-epoch
difference is noise -- and stacks the result by hour of day
(:func:`gpri_tools.diurnal.hour_composite`), which is the same clock
``baker_composite.py`` puts the displacement anomaly on.

Two panels, one clock.  Above, the ice; below, the held-out bedrock, which
does not move and so carries the noise floor the ice has to beat.  Beside
each campaign's composite the script prints its peak-to-peak and the hour it
peaks, and the peak-to-peak a 24 h harmonic fitted to the same series would
imply, which is the sinusoid-assuming answer to the same question.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baker_aps import SCENES                                             # noqa: E402
from baker_population import population_path                             # noqa: E402
from gpri_tools.diurnal import DIURNAL, MIN_CYCLES, hour_composite, m_per_yr  # noqa: E402

PALETTE = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
           "tab:brown", "tab:pink")


def velocity(hours, y, window):
    """Centred difference of ``y`` (mm) over ``window`` hours, in m/yr."""
    step = float(np.median(np.diff(hours)))
    k = max(1, int(round(window / step / 2)))
    v = np.full(y.size, np.nan)
    v[k:-k] = (y[2 * k:] - y[:-2 * k]) / (hours[2 * k:] - hours[:-2 * k])
    return m_per_yr(v * 24.0, "mm")


def harmonic_swing(t, y):
    """Peak-to-peak velocity of a 24 h harmonic fitted to ``y``, m/yr."""
    w = 2 * np.pi / DIURNAL
    ok = np.isfinite(y)
    G = np.column_stack([np.ones(ok.sum()), t[ok], np.cos(w * t[ok]),
                         np.sin(w * t[ok])])
    x, *_ = np.linalg.lstsq(G, y[ok], rcond=None)
    return m_per_yr(2 * np.hypot(x[2], x[3]) * w, "mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="campaigns to draw; the default is every one whose "
                         "population series spans a diurnal cycle")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--window", type=float, default=2.0,
                    help="hours the displacement is differenced over")
    ap.add_argument("--utc-offset", type=float, default=-7.0,
                    help="local time minus UTC, hours; local night 00-06 is "
                         "shaded")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    names = args.scenes or sorted(SCENES)
    if not args.scenes:
        # the same campaign under two names (`X` and `X_full`) would draw
        # twice; the longer copy is the one the analysis uses
        names = [n for n in names if f"{n}_full" not in names]
    rows, drawn = [], []
    for name in names:
        npz = population_path(Path(SCENES.get(name, name)), args.antenna,
                              args.decimate)
        if not npz.exists():
            print(f"{name}: no population cache -- run baker_population.py")
            continue
        c = dict(np.load(npz, allow_pickle=False))
        hours, origin = c["hours"], float(c["origin"])
        t = hours / 24.0
        if t[-1] < DIURNAL * MIN_CYCLES:
            print(f"{name}: spans {hours[-1]:.1f} h ({t[-1]:.2f} cycles); an "
                  f"hour-of-day composite needs {MIN_CYCLES:g} -- skipped")
            continue
        hod = np.mod(origin + hours, 24.0)
        entry = {"name": name, "span": hours[-1]}
        for pop in ("ice", "rock"):
            y = c[f"{pop}_series"]
            v = velocity(hours, y, args.window)
            comp, count = hour_composite(hod, v)
            ok = np.isfinite(comp)
            entry[pop] = {
                "comp": comp, "count": count,
                "pp": float(np.nanmax(comp) - np.nanmin(comp)) if ok.any() else np.nan,
                "peak": float(np.arange(24)[ok][np.nanargmax(comp[ok])] + 0.5)
                        if ok.any() else np.nan,
                "swing": float(harmonic_swing(t, y)),
                "median": float(np.nanmedian(v)),
            }
        rows.append(entry)
        drawn.append(name)

    if not rows:
        sys.exit("no campaign had a population series spanning a cycle")

    print(f"\nLOS velocity by hour of day, {args.window:g} h differences, "
          f"{args.antenna} antenna")
    print(f"{'campaign':16s} {'span':>7s} {'population':11s} "
          f"{'peak-to-peak':>13s} {'peak':>7s} {'median':>9s} {'harmonic':>10s}")
    for e in rows:
        for pop in ("ice", "rock"):
            d = e[pop]
            span = f"{e['span']:6.1f}h" if pop == "ice" else " " * 7
            print(f"{e['name'] if pop == 'ice' else '':16s} {span} "
                  f"{pop:11s} {d['pp']:10.1f} m/yr {d['peak']:5.1f} h "
                  f"{d['median']:+7.1f} m/yr {d['swing']:7.1f} m/yr")

    hours24 = np.arange(24) + 0.5
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for i, e in enumerate(rows):
        colour = PALETTE[i % len(PALETTE)]
        for ax, pop, lw in ((axes[0], "ice", 2.0), (axes[1], "rock", 1.4)):
            ax.plot(hours24, e[pop]["comp"], color=colour, lw=lw,
                    drawstyle="steps-mid",
                    label=e["name"] if pop == "ice" else None)
    # local night 00-06 on the UTC clock
    night = np.mod(np.array([0.0, 6.0]) - args.utc_offset, 24.0)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.grid(axis="x", color="0.85", lw=0.5)
        if night[0] < night[1]:
            ax.axvspan(night[0], night[1], color="0.93", zorder=0)
        else:
            ax.axvspan(night[0], 24, color="0.93", zorder=0)
            ax.axvspan(0, night[1], color="0.93", zorder=0)
    axes[0].set_ylabel("Ice velocity (m/yr)")
    axes[1].set_ylabel("Rock velocity (m/yr)")
    axes[1].set_xlabel("UTC (hr)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.antenna == "upper" else f"_{args.antenna}"
    out = args.outdir / f"31_velocity{tag}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out} ({len(drawn)} campaigns: {', '.join(drawn)})")


if __name__ == "__main__":
    main()
