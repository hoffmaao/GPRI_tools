#!/usr/bin/env python3
"""Displacement and LOS velocity against the temperature that drove them.

    python examples/baker_weather_plots.py --scenes 20170803_full 20180808 20190719

Three panels per campaign, all on the radar's own clock:

1. the ice anomaly and the air temperature 0.8 km away, as time series;
2. **LOS velocity** against temperature, colour-coded by hour of day;
3. **displacement anomaly** against temperature, the same way.

The scatter panels are the point.  A response with no memory — refractivity
tracking the state of the air — plots as a line: the same temperature gives the
same reading whether the air is warming or cooling.  A response with a delay
plots as a **loop**, because the afternoon and the small hours pass through the
same temperature with the glacier in two different states, and the loop's width
is the lag.  Which way the loop is traversed says which quantity leads.

Held-out bedrock is drawn behind the ice in grey at the same scale.  It does
not move, so whatever shape it has is the measurement's own.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES                                        # noqa: E402
from baker_lag import clock_origin, velocity_anomaly                # noqa: E402
from baker_population import population_path                       # noqa: E402

MM_PER_HR_TO_M_PER_YR = 24.0 * 365.25 / 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+",
                    default=["20170803_full", "20180808", "20190719"])
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--window", type=float, default=2.0,
                    help="hours over which the velocity anomaly is differenced")
    ap.add_argument("--station", default="1011")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    work = Path(os.environ.get("GPRI_WORK_ROOT", "work"))

    for name in args.scenes:
        scene = Path(SCENES.get(name, name))
        pop = population_path(scene, args.antenna, args.decimate)
        metf = work / "met" / f"met_{name}.npz"
        if not pop.exists() or not metf.exists():
            print(f"{name}: needs baker_population.py and baker_met.py first")
            continue
        z, m = np.load(pop), np.load(metf, allow_pickle=True)
        hours = z["hours"]
        temp = m[f"snotel_{args.station}_TOBS_at_epochs"]
        if not np.isfinite(temp).any():
            print(f"{name}: no station temperature at these epochs")
            continue
        origin = clock_origin(z, m)
        utc = np.mod(hours + origin, 24.0)

        ice, rock = z["ice"], z["rock"]
        v_ice = velocity_anomaly(hours, ice, args.window) * MM_PER_HR_TO_M_PER_YR
        v_rock = velocity_anomaly(hours, rock, args.window) * MM_PER_HR_TO_M_PER_YR

        fig = plt.figure(figsize=(13, 8.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.32,
                              wspace=0.22)

        # ---- 1. the two series ------------------------------------------
        ax = fig.add_subplot(gs[0, :])
        ax.plot(hours, ice, "k", lw=1.2, label="ice anomaly")
        ax.plot(hours, rock, color="0.6", lw=1.0, label="held-out bedrock")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Ice anomaly (mm)")
        ax.set_xlabel("Elapsed time (hr)")
        ax2 = ax.twinx()
        ax2.plot(hours, temp, color="tab:red", lw=1.2, alpha=0.8)
        ax2.set_ylabel("Air temperature (°C)", color="tab:red")
        ax2.tick_params(axis="y", colors="tab:red")
        for night in np.arange(-24, hours[-1] + 24, 24.0):
            lo, hi = night + (3.5 - origin), night + (13.0 - origin)
            ax.axvspan(max(lo, 0), min(hi, hours[-1]), color="0.93", zorder=0)
        ax.set_xlim(0, hours[-1])
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{name}: the ice anomaly and the air 0.8 km away "
                     f"(+ toward radar; shading: local night)", fontsize=10)

        # ---- 2 and 3. against temperature, coloured by hour --------------
        for k, (y, y_rock, label) in enumerate(
                ((v_ice, v_rock, "LOS velocity (m/yr)"),
                 (ice, rock, "Ice anomaly (mm)"))):
            ax = fig.add_subplot(gs[1, k])
            ok = np.isfinite(temp) & np.isfinite(y)
            ax.scatter(temp[ok], y_rock[ok], s=3, color="0.8", alpha=0.5,
                       label="held-out bedrock")
            sc = ax.scatter(temp[ok], y[ok], c=utc[ok], s=4, cmap="twilight",
                            vmin=0, vmax=24)
            # the mean path through the day: where the loop shows
            bins = np.arange(0, 25, 2.0)
            idx = np.digitize(utc[ok], bins) - 1
            bx = [np.nanmean(temp[ok][idx == i]) for i in range(len(bins) - 1)]
            by = [np.nanmean(y[ok][idx == i]) for i in range(len(bins) - 1)]
            ax.plot(bx + bx[:1], by + by[:1], "k-", lw=1.0, alpha=0.7, zorder=3)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("Air temperature (°C)")
            ax.set_ylabel(label)
            ax.set_title("a line means no lag, a loop means a lag"
                         if k == 0 else "the same, for displacement",
                         fontsize=9)
            if k == 1:
                cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
                cb.set_label("UTC (hr)")
                cb.set_ticks([0, 6, 12, 18, 24])
            else:
                ax.legend(loc="upper left", fontsize=8)

        args.outdir.mkdir(parents=True, exist_ok=True)
        out = args.outdir / f"22_weather_{name}.png"
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
        r = np.corrcoef(temp[np.isfinite(temp) & np.isfinite(ice)],
                        ice[np.isfinite(temp) & np.isfinite(ice)])[0, 1]
        print(f"{name}: r(ice, temperature) {r:+.2f}; wrote {out}")


if __name__ == "__main__":
    main()
