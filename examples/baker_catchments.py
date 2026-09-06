#!/usr/bin/env python3
"""The glaciers one by one: each catchment's LOS velocity beside the brightness.

    python examples/baker_catchments.py --scene 20190719

One figure per campaign, two panels on one UTC clock.  Above, the mean LOS
velocity of the coherent ice in each named RGI glacier the radar sees —
Coleman, Roosevelt, Mazama and whatever else clears ``--min-pixels`` — in
m/yr, positive toward the radar: the corrected displacement (the validated
recipe, bedrock reference + per-epoch drift + per-epoch turbulence screen)
averaged over the catchment and differenced over a ``--window`` of hours,
because at a two-minute cadence the epoch-to-epoch difference is noise.
Below, the mean backscatter over the whole glacier outline, in dB, as
``baker_melt.py`` cached it.  The panels carry no titles and nothing but
two-word axis labels; the local night (00-06) is shaded.

The catchment means are cached in ``work/<scene>/catchments_u_dec16.npz``;
the figure is rebuilt from the cache and ``melt_u_dec16.npz``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, integrate, load                       # noqa: E402
from baker_brightness import shade_local_nights, utc_epochs          # noqa: E402
from baker_lag import velocity_anomaly                               # noqa: E402
from baker_melt import load_melt                                     # noqa: E402
from baker_north_side import decimated_par                           # noqa: E402

from gpri_tools.aps import epoch_screen_correction, turbulence_screen    # noqa: E402
from gpri_tools.diurnal import m_per_yr                                  # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry          # noqa: E402
from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask  # noqa: E402
from gpri_tools.heading import scene_heading                              # noqa: E402
from gpri_tools.timeseries import los_displacement                        # noqa: E402


def catchments_path(scene: Path, antenna: str, dec: int) -> Path:
    root = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
    return root / scene.name / f"catchments_{antenna[0].lower()}_dec{dec}.npz"


def short_name(name) -> str:
    """'Coleman Glacier WA' -> 'Coleman'; an unnamed outline gives ''."""
    if not isinstance(name, str):
        return ""
    words = [w for w in name.split() if w not in ("Glacier", "Glaciers", "WA")]
    return " ".join(words)


def compute(scene, args):
    """The corrected displacement averaged over each named glacier's ice."""
    stack, net, phase, cc, r, az, n = load(scene, args.decimate, 0, antenna=args.antenna)
    lam = stack.wavelength
    mean_cc = cc.mean(axis=0)
    del cc
    geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                         heading=scene_heading(scene, default=BAKERBEND1_HEADING))
    la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1], cols=[0, geom.shape[1] - 1])
    bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"), bbox=bbox)
    stable, _ = stable_ground_mask(mean_cc, geom, gdf, threshold=args.stable_coherence)
    coherent = mean_cc >= args.ice_coherence

    # the named glaciers with enough coherent ice to average
    masks, names, ids = [], [], []
    named = gdf[gdf["Name"].notna()] if "Name" in gdf.columns else gdf[:0]
    for _, row in named.sort_values("Area", ascending=False).iterrows():
        nm = short_name(row["Name"])
        if not nm:
            continue
        m = coherent & glacier_mask(geom, gdf[gdf["RGIId"] == row["RGIId"]])
        if m.sum() < args.min_pixels:
            continue
        masks.append(m); names.append(nm); ids.append(row["RGIId"])
    if not masks:
        sys.exit("no named glacier has enough coherent ice in view")
    print("catchments: " + ", ".join(f"{nm} ({m.sum():,} px)" for nm, m in zip(names, masks)))

    # the validated recipe, every bedrock pixel feeding it (a product, not a test)
    d, times = integrate(los_displacement(phase, lam), net, n)
    del phase
    d, _ = epoch_screen_correction(d, stable, r, model="linear", weights=mean_cc)
    t0 = time.time()
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], stable, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        d[k] -= scr
    print(f"drift + turbulence corrections in {time.time() - t0:.0f} s")

    disp = np.stack([np.nanmean(d[:, m], axis=1) for m in masks], axis=1) * 1000  # mm
    n_px = np.array([m.sum() for m in masks])
    return {"disp": disp.astype(np.float32), "names": np.array(names), "ids": np.array(ids),
            "n_pixels": n_px, "epoch_hours": np.asarray(times, float) * 24.0,
            "epoch0": np.datetime64(net.epochs[0]).astype("datetime64[s]"),
            "window": args.window, "utc_offset": args.utc_offset}


def figure(c, melt, name, args):
    t = utc_epochs(c)
    fig, (ax_v, ax_b) = plt.subplots(2, 1, figsize=(7.68, 5.6), dpi=110, sharex=True)
    for k, nm in enumerate(c["names"]):
        v = m_per_yr(velocity_anomaly(c["epoch_hours"], c["disp"][:, k], args.window) * 24,
                     "mm")
        ax_v.plot(t, v, lw=1.0, label=nm)
    ax_v.axhline(0, color="0.5", lw=0.6)
    ax_v.set_ylabel("LOS velocity (m/yr)")
    ax_v.legend(loc="upper right", fontsize=8, frameon=False, ncol=min(len(c["names"]), 4))

    if melt is not None and "db_ice" in melt:
        tb = utc_epochs(melt)
        ax_b.plot(tb, melt["db_ice"], color="k", lw=1.0)
    else:
        ax_b.text(0.5, 0.5, f"run baker_melt.py --scene {name} --recompute",
                  transform=ax_b.transAxes, ha="center", va="center", fontsize=9)
    ax_b.set_ylabel("Backscatter (dB)")
    ax_b.set_xlabel("Time (UTC)")
    for ax in (ax_v, ax_b):
        shade_local_nights(ax, t[0], t[-1], args.utc_offset)
        ax.grid(alpha=0.3)
    ax_b.set_xlim(t[0], t[-1])
    loc = mdates.AutoDateLocator()
    ax_b.xaxis.set_major_locator(loc)
    ax_b.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    out = args.outdir / f"27_catchments_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5,
                    help="mean coherence an ice pixel needs to be averaged")
    ap.add_argument("--min-pixels", type=int, default=200,
                    help="coherent ice pixels a glacier needs to be a line")
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0),
                    help="turbulence screen kernel (azimuth, range) px")
    ap.add_argument("--window", type=float, default=2.0,
                    help="hours the velocity is differenced over")
    ap.add_argument("--utc-offset", type=float, default=-7.0,
                    help="local clock minus UTC, for the night shading")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    name = args.scene
    scene = Path(SCENES.get(name, name))
    cache = catchments_path(scene, args.antenna, args.decimate)
    if cache.exists() and not args.recompute:
        c = dict(np.load(cache, allow_pickle=False))
        print(f"loaded {cache}")
    else:
        c = compute(scene, args)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **c)
        print(f"cached {cache}")
    melt = load_melt(scene, args.antenna, args.decimate)
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure(c, melt, name, args)


if __name__ == "__main__":
    main()
