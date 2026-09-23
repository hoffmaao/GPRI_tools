#!/usr/bin/env python3
"""How long the surface stays recognisable: coherence against temporal baseline.

    python examples/baker_decorrelation.py --scene 20170913 --looks 3 15

Everywhere else in this analysis coherence is a weight.  Here it is the
measurement.  The long-baseline pair caches — the ones built for the
multi-baseline path delay, lags 1 to 360 or 720 epochs, two minutes to
twelve or twenty-four hours — carry a coherence per pair per pixel, so the
decorrelation curve of every pixel is already on disk.

:func:`gpri_tools.decorrelation.coherence_by_baseline` collapses the stack to
one coherence per baseline class per pixel, and
:func:`~gpri_tools.decorrelation.decorrelation_time` reads off the baseline
at which each pixel's coherence has fallen to ``1/e`` of its two-minute
value, interpolated in log baseline and assuming no model.  That map is the
figure's left panel.  The right panel is the curve itself, per mask, with the
:func:`~gpri_tools.decorrelation.exponential_decorrelation` fit of
``(g0 - g_inf) exp(-t / tau) + g_inf`` drawn over it — the aggregate view,
where a single exponential is worth fitting.

Two things the numbers are not.  A coherence estimated from ``L`` looks is
biased high, so the long-baseline end is a ceiling rather than a
measurement; the bias depends only on ``L``, so it moves every class the
same way and the shape the estimators read survives it.  And ``tau`` is a
baseline at which a ratio is crossed, not a physical timescale of anything
in particular — what it is good for is comparing pixels, masks and days
measured the same way.
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
import matplotlib.pyplot as plt                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baker_aps import SCENES, load                                       # noqa: E402
from baker_movie import Resampler                                        # noqa: E402
from baker_delay_movie import decimated_geom                             # noqa: E402
from gpri_tools.decorrelation import (coherence_by_baseline,             # noqa: E402
                                      decorrelation_time,
                                      exponential_decorrelation)
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.glaciers import (glacier_mask, load_outlines,            # noqa: E402
                                 stable_ground_mask)
from gpri_tools.heading import scene_heading                             # noqa: E402

LONG_LAGS = (1, 2, 3, 30, 60, 90, 180, 360)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170913")
    ap.add_argument("--lags", type=int, nargs="+", default=list(LONG_LAGS),
                    help="the pair cache to read; these are the lags the "
                         "long-baseline caches were built at")
    ap.add_argument("--looks", type=int, nargs=2, default=(3, 15))
    ap.add_argument("--decimate", type=int, default=1)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--stable-coherence", type=float, default=0.6)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--level", type=float, default=0.5,
                    help="absolute coherence the crossing time is read at; "
                         "comparable between pixels and campaigns")
    ap.add_argument("--fraction", type=float, default=None,
                    help="read the crossing at this fraction of each pixel's "
                         "own shortest-baseline coherence instead of --level")
    ap.add_argument("--gamma-at", type=float, default=1.0,
                    help="baseline (hours) the mapped coherence is read at; "
                         "the nearest class is used")
    ap.add_argument("--spacing", type=float, default=40.0)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    stack, net, phase, cc, r, az, n = load(
        scene, args.decimate, 0, antenna=args.antenna,
        lags=tuple(int(x) for x in args.lags), looks=tuple(args.looks))
    del phase
    pairs = np.asarray(net.pairs[:n], int)
    times = np.asarray(net.times, float)
    mean_cc = cc[(pairs[:, 1] - pairs[:, 0]) <= 3].mean(axis=0)

    geom = decimated_geom(stack, args.decimate, heading)
    la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1],
                           cols=[0, geom.shape[1] - 1])
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                        bbox=(lo.min() - .02, la.min() - .02,
                              lo.max() + .02, la.max() + .02))
    stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                   threshold=args.stable_coherence)
    ice = (mean_cc >= args.ice_coherence) & glacier_mask(geom, gdf)
    print(f"{scene.name}: {n:,} pairs, {len(times)} epochs, grid "
          f"{cc.shape[1]} x {cc.shape[2]}; ice {ice.sum():,} px, "
          f"bedrock {stable.sum():,} px")

    t0 = time.time()
    baselines, gamma, counts = coherence_by_baseline(cc, pairs, times,
                                                     by="lag", min_pairs=20)
    del cc
    hours = baselines * 24.0
    print(f"{len(baselines)} baseline classes in {time.time() - t0:.0f} s: "
          + ", ".join(f"{h * 60:.0f} min ({c} pairs)" if h < 1
                      else f"{h:.1f} h ({c} pairs)"
                      for h, c in zip(hours, counts)))

    level = None if args.fraction is not None else args.level
    tau = decorrelation_time(hours, gamma, level=level,
                             fraction=args.fraction or 1.0 / np.e)   # hours
    target = (f"coherence {args.level:.2f}" if level is not None
              else f"{args.fraction:.2f} of their short-baseline coherence")
    at = int(np.argmin(np.abs(hours - args.gamma_at)))
    print(f"\n{'population':18s} {'pixels':>8s} " +
          " ".join(f"{h * 60:5.0f}m" if h < 1 else f"{h:5.1f}h" for h in hours))
    curves = {}
    for label, m in (("RGI ice", ice), ("bedrock", stable)):
        curve = np.array([np.nanmedian(g[m]) for g in gamma])
        curves[label] = curve
        print(f"{label:18s} {m.sum():8,d} " +
              " ".join(f"{v:6.3f}" for v in curve))
    print()
    for label, m in (("RGI ice", ice), ("bedrock", stable)):
        fit = exponential_decorrelation(hours, curves[label])
        finite = np.isfinite(tau[m])
        p16, p50, p84 = (np.nanpercentile(tau[m], [16, 50, 84])
                         if finite.any() else (np.nan,) * 3)
        print(f"{label:18s} crossing {p50:6.2f} h (p16-p84 {p16:5.2f}-{p84:6.2f}), "
              f"{100 * finite.mean():4.1f} % of pixels reach {target}; "
              f"fitted tau {fit.tau:6.2f} h, g0 {fit.g0:.3f}, "
              f"floor {fit.g_inf:.3f}, rms {fit.rms:.4f}")

    # ---- figure ------------------------------------------------------------
    res = Resampler(geom, args.spacing)
    show = ice | stable
    fig = plt.figure(figsize=(15.5, 5.4), dpi=140)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0])
    ax_g = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    im_g = ax_g.imshow(res(np.where(show, gamma[at], np.nan)), origin="lower",
                       cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    label_h = (f"{hours[at] * 60:.0f} min" if hours[at] < 1
               else f"{hours[at]:.0f} hr")
    fig.colorbar(im_g, ax=ax_g, label=f"Coherence ({label_h})", fraction=0.046)
    tau_seen = tau[show][np.isfinite(tau[show])]
    lim = float(np.nanpercentile(tau_seen, 98)) if tau_seen.size else 1.0
    im = ax_m.imshow(res(np.where(show, tau, np.nan)), origin="lower",
                     cmap="cividis", vmin=0, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax_m,
                 label=(f"Time to {args.level:.1f} (hr)" if level is not None
                        else "Crossing time (hr)"), fraction=0.046)
    seen = np.isfinite(res(np.where(show, 0.0, np.nan)))
    if seen.any():
        rr, cc_ = np.where(seen)
        for ax in (ax_g, ax_m):
            ax.set_xlim(max(cc_.min() - 4, 0), min(cc_.max() + 4, seen.shape[1] - 1))
            ax.set_ylim(max(rr.min() - 4, 0), min(rr.max() + 4, seen.shape[0] - 1))
    for ax in (ax_g, ax_m):
        ax.set_xticks([]); ax.set_yticks([])

    fine = np.logspace(np.log10(hours.min() / 2), np.log10(hours.max() * 1.5), 200)
    for label, colour in (("RGI ice", "tab:blue"), ("bedrock", "tab:red")):
        curve = curves[label]
        ax_c.plot(hours, curve, "o", ms=5, color=colour, label=label)
        fit = exponential_decorrelation(hours, curve)
        ax_c.plot(fine, fit(fine), "-", lw=1.4, color=colour, alpha=0.8)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Baseline (hr)")
    ax_c.set_ylabel("Coherence")
    ax_c.set_ylim(0, 1)
    ax_c.grid(alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    la_, lr = args.looks
    out = args.outdir / f"33_decorrelation_{scene.name}_lk{la_}x{lr}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")

    root = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
    npz = root / f"decorrelation_{args.antenna[0].lower()}_lk{la_}x{lr}.npz"
    np.savez(npz, baseline_hours=hours, counts=counts,
             gamma_ice=curves["RGI ice"], gamma_rock=curves["bedrock"],
             tau=tau.astype(np.float32), level=np.nan if level is None else level,
             gamma_at=gamma[at].astype(np.float32), gamma_at_hours=hours[at],
             ice=ice, stable=stable)
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
