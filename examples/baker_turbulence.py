#!/usr/bin/env python3
"""What the turbulence screen removes, measured instead of assumed.

    python examples/baker_turbulence.py --scene 20170803_full --rgi

The ladder takes a linear range screen off every epoch and then a normalised
convolution at ``sigma = (5, 25)`` pixels.  That second number is a claim
about the distance over which the atmosphere stays coherent, and nothing in
this repository has ever measured it.  This does.

For each epoch the script forms the field the turbulence screen is about to
see — the epoch's displacement after the reference and the linear range
screen, on bedrock, where nothing moves — and computes its structure
function ``D(r) = <[phi(x+r) - phi(x)]^2>``
(:func:`gpri_tools.turbulence.structure_function`) over pixel pairs binned by
ground separation, then fits ``D(r) ~ r^alpha``
(:func:`~gpri_tools.turbulence.power_law`).

Three readings come out of that, and the figure has one panel each: the
curves themselves, coloured by hour of day; the exponent against UTC hour,
which says whether the field at these separations behaves like turbulence
(2/3 to 5/3) or like noise (0); and the amplitude at 1 km against UTC hour,
which is the size of what the screen has to remove.  The structure function
needs no mean, so the arbitrary constant and ramp a bedrock-fitted screen
carries do not enter it.

The separations are ground distances from the radar geometry, not pixels, so
the answer can be read against the ``sigma`` the ladder uses only after that
sigma is converted the same way — which the script prints.
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
from baker_aps import SCENES, integrate, load, split_mask                # noqa: E402
from baker_brightness import shade_local_nights                          # noqa: E402
from baker_movie import decimated_geom                                   # noqa: E402
from gpri_tools.aps import epoch_screen_correction                       # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.glaciers import (glacier_mask, load_outlines,            # noqa: E402
                                 stable_ground_mask)
from gpri_tools.heading import scene_heading                             # noqa: E402
from gpri_tools.timeseries import los_displacement                       # noqa: E402
from gpri_tools.turbulence import power_law, structure_function          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803_full")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--rgi", action="store_true",
                    help="tie the bedrock mask to the glacier outlines")
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0),
                    help="the ladder's screen width, in pixels; printed in "
                         "metres for comparison with the separations")
    ap.add_argument("--field", default="pair", choices=("pair", "cumulative"),
                    help="'pair' measures the epoch-to-epoch increment, which "
                         "is the atmosphere's own change over one cadence; "
                         "'cumulative' measures the integrated series, whose "
                         "structure function grows with time from the "
                         "reference epoch because the series does")
    ap.add_argument("--stride", type=int, default=5,
                    help="use every Nth epoch")
    ap.add_argument("--bins", type=float, nargs="+", default=None,
                    help="separation bin edges in metres; log-spaced over the "
                         "mask by default")
    ap.add_argument("--max-pairs", type=int, default=60_000,
                    help="pixel pairs sampled per separation bin per epoch")
    ap.add_argument("--r0", type=float, default=1000.0,
                    help="separation the amplitude is quoted at, metres")
    ap.add_argument("--utc-offset", type=float, default=-7.0)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    stack, net, phase, cc, r, az, n = load(scene, args.decimate, 0,
                                           antenna=args.antenna)
    mean_cc = cc.mean(axis=0)
    del cc
    geom = decimated_geom(stack, args.decimate, heading)
    stable = mean_cc >= args.stable_coherence
    if args.rgi:
        la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1],
                               cols=[0, geom.shape[1] - 1])
        gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                            bbox=(lo.min() - .02, la.min() - .02,
                                  lo.max() + .02, la.max() + .02))
        stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                       threshold=args.stable_coherence)
    fit_m, held_m = split_mask(stable)

    # ground coordinates of every pixel, so separations are metres on the
    # ground rather than pixels: horizontal distance along each beam, laid
    # out on that beam's true bearing
    gr = np.asarray(geom.ground_range(), float)               # per range sample
    bearing = np.deg2rad(np.asarray(geom.bearings(), float))  # per azimuth line
    east = gr[None, :] * np.sin(bearing)[:, None]
    north = gr[None, :] * np.cos(bearing)[:, None]

    d, times = integrate(los_displacement(phase, stack.wavelength), net, n)
    del phase
    d, _ = epoch_screen_correction(d, fit_m, r, model="linear", weights=mean_cc)
    d = d * 1000.0                                            # mm
    t = np.asarray(times, float)
    if args.field == "pair":
        # the change over one cadence: what the atmosphere did between two
        # acquisitions, with everything static differenced away
        cadence = float(np.median(np.diff(t)) * 24 * 60)
        d = np.diff(d, axis=0)
        t = t[1:]
        print(f"measuring the {cadence:.1f} min increment")
    keep = np.arange(0 if args.field == "pair" else 1, d.shape[0],
                     max(1, args.stride))
    origin = (net.epochs[0].hour + net.epochs[0].minute / 60.0
              + net.epochs[0].second / 3600.0)
    hod = np.mod(origin + t * 24.0, 24.0)

    px, py = east[stable], north[stable]
    extent = float(np.hypot(px.max() - px.min(), py.max() - py.min()))
    if args.bins is None:
        bins = np.logspace(np.log10(max(extent / 200.0, 20.0)),
                           np.log10(extent / 2.0), 9)
    else:
        bins = np.asarray(args.bins, float)
    # the ladder's sigma in metres, at the median range of the bedrock it is
    # fitted on: an azimuth pixel is an arc, a range pixel is a step in range
    r_med = float(np.median(gr[np.any(stable, axis=0)])) if stable.any() else np.nan
    d_bearing = float(np.median(np.abs(np.diff(bearing))))
    d_range = float(np.median(np.abs(np.diff(gr))))
    sig_m = (args.sigma[0] * r_med * d_bearing, args.sigma[1] * d_range)
    print(f"{scene.name}: {stable.sum():,} bedrock px over {extent / 1000:.1f} km; "
          f"{len(keep)} of {d.shape[0]} epochs (stride {args.stride}); "
          f"separations {bins[0]:.0f}-{bins[-1]:.0f} m; the ladder's sigma is "
          f"{args.sigma[0]:g} x {args.sigma[1]:g} px "
          f"= {sig_m[0]:.0f} x {sig_m[1]:.0f} m")

    g = np.random.default_rng(0)
    t0 = time.time()
    curves, alpha, amp, rms, used = [], [], [], [], []
    for k in keep:
        centres, D, counts = structure_function(d[k][stable], px, py, bins,
                                                max_pairs=args.max_pairs, rng=g)
        try:
            fit = power_law(centres, D, r0=args.r0)
        except ValueError:
            continue                      # nothing positive to fit at this epoch
        curves.append((centres, D))
        alpha.append(fit.exponent); amp.append(fit.amplitude); rms.append(fit.rms)
        used.append(k)
    if not curves:
        sys.exit("no epoch had a structure function to fit")
    alpha = np.array(alpha); amp = np.array(amp); rms = np.array(rms)
    used = np.array(used, int)
    print(f"{len(curves)} structure functions in {time.time() - t0:.0f} s "
          f"({len(keep) - len(curves)} epochs had nothing to fit)")
    hours = hod[used]
    print(f"\nexponent      median {np.nanmedian(alpha):+.2f}, "
          f"p16-p84 {np.nanpercentile(alpha, 16):+.2f} to {np.nanpercentile(alpha, 84):+.2f}")
    print(f"D at {args.r0:.0f} m   median {np.nanmedian(amp):7.2f} mm^2 "
          f"(rms difference {np.sqrt(np.nanmedian(amp)):.2f} mm), "
          f"p16-p84 {np.nanpercentile(amp, 16):.2f} to {np.nanpercentile(amp, 84):.2f}")
    print(f"log10 residual of the power law: median {np.nanmedian(rms):.3f}")
    night = (hours >= np.mod(-args.utc_offset, 24)) & \
            (hours < np.mod(6 - args.utc_offset, 24))
    if night.any() and (~night).any():
        print(f"local night   exponent {np.nanmedian(alpha[night]):+.2f}, "
              f"D {np.nanmedian(amp[night]):7.2f} mm^2 ({night.sum()} epochs)")
        print(f"local day     exponent {np.nanmedian(alpha[~night]):+.2f}, "
              f"D {np.nanmedian(amp[~night]):7.2f} mm^2 ({(~night).sum()} epochs)")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=140)
    cmap = plt.get_cmap("twilight")
    for (centres, D), h in zip(curves, hours):
        axes[0].plot(centres, D, "-", lw=0.7, alpha=0.5, color=cmap(h / 24.0))
    med = np.nanmedian(np.stack([D for _, D in curves]), axis=0)
    axes[0].plot(np.nanmedian(np.stack([c for c, _ in curves]), axis=0), med,
                 "k-", lw=2.2)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Separation (m)")
    axes[0].set_ylabel("Structure function (mm²)")
    axes[0].grid(alpha=0.3, which="both")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 24))
    fig.colorbar(sm, ax=axes[0], label="UTC (hr)", fraction=0.046)

    for ax, y, label in ((axes[1], alpha, "Exponent"),
                         (axes[2], amp, f"D at {args.r0:.0f} m (mm²)")):
        ax.plot(hours, y, ".", ms=4, color="tab:blue")
        ax.set_xlabel("UTC (hr)")
        ax.set_ylabel(label)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        ax.grid(alpha=0.3)
    axes[1].axhspan(2 / 3, 5 / 3, color="0.85", zorder=0)
    axes[2].set_yscale("log")
    for ax in axes[1:]:
        lo, hi = np.mod(-args.utc_offset, 24), np.mod(6 - args.utc_offset, 24)
        if lo < hi:
            ax.axvspan(lo, hi, color="0.93", zorder=0)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    day = scene.name + ("" if args.antenna == "upper" else f"_{args.antenna}")
    out = (args.outdir / f"34_turbulence_{day}"
           f"{'' if args.field == 'pair' else '_cumulative'}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")

    root = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
    npz = (root / f"turbulence_{args.antenna[0].lower()}_dec{args.decimate}"
           f"{'' if args.field == 'pair' else '_cumulative'}.npz")
    np.savez(npz, hours=hours, exponent=alpha, amplitude=amp, rms=rms,
             bins=bins, r0=args.r0,
             separations=np.stack([c for c, _ in curves]),
             structure=np.stack([D for _, D in curves]))
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
