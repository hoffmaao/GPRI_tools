#!/usr/bin/env python3
"""How fast the rate changes from pixel to pixel: the velocity gradient map.

    python examples/baker_strain.py --scene 20170827

Every product so far reports a rate per pixel or a mean per catchment.  The
spatial *derivative* of that rate is a different measurement: where the ice
is speeding up or slowing down along its own flow, which is what a strain
rate is made of.  This fits a local plane to the per-pixel line-of-sight rate
over a moving neighbourhood in ground metres and maps the gradient.

Two honest limits, both of which the script prints rather than hides.  One
look direction gives one component of a three-dimensional velocity field, so
what is mapped is the gradient of the *line-of-sight* rate, not a strain rate
tensor; turning it into one needs a flow direction and a depth assumption
that this data cannot supply.  And a gradient differentiates noise: held-out
bedrock, which does not move and whose rate should be flat, is carried
through the identical calculation and its gradient is the noise floor the
ice's has to clear.

The per-pixel rates come from `baker_harmonics.py`'s cache where one exists
(the secular term of its 24 h fit) and are computed from the corrected
series otherwise.
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
from baker_movie import Resampler, decimated_geom                        # noqa: E402
from gpri_tools.aps import epoch_screen_correction, turbulence_screen    # noqa: E402
from gpri_tools.diurnal import m_per_yr                                  # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.glaciers import (glacier_mask, load_outlines,            # noqa: E402
                                 stable_ground_mask)
from gpri_tools.heading import scene_heading                             # noqa: E402
from gpri_tools.pathdelay import (displacement_delay_field,              # noqa: E402
                                  pair_variance_from_coherence)
from gpri_tools.timeseries import los_displacement                       # noqa: E402


def local_gradient(values, east, north, mask, radius, min_points=8):
    """Plane fitted to ``values`` in a ground-metre neighbourhood, per pixel.

    Returns ``(d/d east, d/d north)`` in the values' units per metre, NaN
    where the neighbourhood held fewer than ``min_points`` usable pixels.
    The fit is a plane rather than a difference because the pixels are
    anisotropic and irregularly masked, and a plane is the smallest model
    that gives a gradient without assuming which axis to difference along.
    """
    from scipy.spatial import cKDTree
    idx = np.flatnonzero(mask.ravel() & np.isfinite(values.ravel()))
    if idx.size < min_points:
        raise ValueError("not enough usable pixels to fit gradients")
    pts = np.column_stack([east.ravel()[idx], north.ravel()[idx]])
    vals = values.ravel()[idx]
    tree = cKDTree(pts)
    neighbours = tree.query_ball_point(pts, r=float(radius))
    gx = np.full(values.size, np.nan)
    gy = np.full(values.size, np.nan)
    for k, nb in enumerate(neighbours):
        if len(nb) < min_points:
            continue
        p = pts[nb] - pts[k]
        A = np.column_stack([np.ones(len(nb)), p[:, 0], p[:, 1]])
        coef, *_ = np.linalg.lstsq(A, vals[nb], rcond=None)
        gx[idx[k]] = coef[1]
        gy[idx[k]] = coef[2]
    return gx.reshape(values.shape), gy.reshape(values.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170827")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--radius", type=float, default=200.0,
                    help="neighbourhood the plane is fitted over, ground metres")
    ap.add_argument("--min-points", type=int, default=12)
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--rgi", action="store_true")
    ap.add_argument("--no-path-delay", dest="path_delay", action="store_false")
    ap.add_argument("--protect-period", type=float, default=1.0)
    ap.add_argument("--max-response", type=float, default=0.01)
    ap.add_argument("--spacing", type=float, default=40.0)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)

    cached = (Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
              / f"harmonics_u_dec{args.decimate}.npz")
    stack, net, phase, cc, r, az, n = load(scene, args.decimate, 0,
                                           antenna="upper")
    mean_cc = cc.mean(axis=0)
    geom = decimated_geom(stack, args.decimate, heading)
    usable = mean_cc >= args.ice_coherence
    stable = mean_cc >= args.stable_coherence
    if args.rgi:
        la_, lo_ = geom.geodetic(rows=[0, geom.shape[0] - 1],
                                 cols=[0, geom.shape[1] - 1])
        gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                            bbox=(lo_.min() - .02, la_.min() - .02,
                                  lo_.max() + .02, la_.max() + .02))
        stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                       threshold=args.stable_coherence)
        ice = usable & glacier_mask(geom, gdf)
    else:
        ice = usable & ~stable
    fit_m, held_m = split_mask(stable)

    if cached.exists():
        c = dict(np.load(cached, allow_pickle=False))
        rate = np.asarray(c["rate"], float)
        print(f"{scene.name}: per-pixel rates from {cached.name}")
        del cc, phase
    else:
        pair_var = (pair_variance_from_coherence(cc[:n], ice | fit_m)
                    if args.path_delay else None)
        del cc
        d, times = integrate(los_displacement(phase, stack.wavelength), net, n)
        del phase
        d, _ = epoch_screen_correction(d, fit_m, r, model="linear", weights=mean_cc)
        for k in range(d.shape[0]):
            scr, _ = turbulence_screen(d[k], fit_m, sigma=tuple(args.sigma),
                                       weights=mean_cc, wrapped=False)
            d[k] -= scr
        if args.path_delay:
            field, plam = displacement_delay_field(
                d, np.asarray(net.pairs[:n], int), np.asarray(times, float),
                ice | fit_m, weights=mean_cc, sigma=tuple(args.sigma),
                protect_period=args.protect_period,
                max_response=args.max_response, pair_variance=pair_var)
            d -= field.astype(d.dtype)
            del field
        t = np.asarray(times, float)
        flat = d.reshape(d.shape[0], -1) * 1000.0
        A = np.column_stack([np.ones(t.size), t])
        coef, *_ = np.linalg.lstsq(A, flat, rcond=None)
        rate = m_per_yr(coef[1].reshape(d.shape[1:]), "mm")
        del d, flat
        print(f"{scene.name}: per-pixel rates fitted from the corrected series")

    gr = np.asarray(geom.ground_range(), float)
    bearing = np.deg2rad(np.asarray(geom.bearings(), float))
    east = gr[None, :] * np.sin(bearing)[:, None]
    north = gr[None, :] * np.cos(bearing)[:, None]
    stack.close()

    print(f"ice {ice.sum():,} px, held-out bedrock {held_m.sum():,} px; "
          f"fitting planes over {args.radius:.0f} m neighbourhoods")
    t0 = time.time()
    gx, gy = local_gradient(rate, east, north, ice | held_m, args.radius,
                            args.min_points)
    mag = np.hypot(gx, gy) * 1000.0                     # m/yr per km
    direction = np.degrees(np.arctan2(gx, gy)) % 360.0  # bearing of increase
    print(f"gradients in {time.time() - t0:.0f} s")

    print(f"\n{'population':18s} {'pixels':>8s} {'rate (m/yr)':>22s} "
          f"{'|gradient| (m/yr per km)':>28s}")
    out_rows = {}
    for label, m in (("RGI ice", ice), ("held-out bedrock", held_m)):
        mm = m & np.isfinite(mag)
        if not mm.any():
            continue
        p16, p50, p84 = np.nanpercentile(mag[mm], [16, 50, 84])
        out_rows[label] = (np.nanmedian(rate[mm]), p50, p16, p84)
        print(f"{label:18s} {mm.sum():8,d} {np.nanmedian(rate[mm]):+10.2f} "
              f"(p16-p84 {np.nanpercentile(rate[mm], 16):+6.1f} to "
              f"{np.nanpercentile(rate[mm], 84):+6.1f}) {p50:14.2f} "
              f"(p16-p84 {p16:5.2f} to {p84:6.2f})")
    if len(out_rows) == 2:
        ratio = out_rows["RGI ice"][1] / max(out_rows["held-out bedrock"][1], 1e-9)
        print(f"\nthe ice's gradient is {ratio:.1f} times the bedrock's, and "
              f"bedrock does not move: that ratio is what the map is worth")

    # ---- figure ------------------------------------------------------------
    res = Resampler(geom, args.spacing)
    show = ice | stable
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=140)
    rlim = float(np.nanpercentile(np.abs(rate[show]), 98)) or 1.0
    im0 = axes[0].imshow(res(np.where(show, rate, np.nan)), origin="lower",
                         cmap="RdBu_r", vmin=-rlim, vmax=rlim,
                         interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], label="LOS rate (m/yr)", fraction=0.046)
    glim = float(np.nanpercentile(mag[np.isfinite(mag)], 98)) or 1.0
    im1 = axes[1].imshow(res(np.where(np.isfinite(mag), mag, np.nan)),
                         origin="lower", cmap="magma", vmin=0, vmax=glim,
                         interpolation="nearest")
    fig.colorbar(im1, ax=axes[1], label="Gradient (m/yr per km)", fraction=0.046)
    seen = np.isfinite(res(np.where(show, 0.0, np.nan)))
    if seen.any():
        rr, cc_ = np.where(seen)
        for ax in axes[:2]:
            ax.set_xlim(max(cc_.min() - 4, 0), min(cc_.max() + 4, seen.shape[1] - 1))
            ax.set_ylim(max(rr.min() - 4, 0), min(rr.max() + 4, seen.shape[0] - 1))
            ax.set_xticks([]); ax.set_yticks([])
    bins = np.linspace(0, glim, 40)
    for label, m, colour in (("ice", ice, "tab:blue"),
                             ("held-out rock", held_m, "tab:red")):
        mm = m & np.isfinite(mag)
        if mm.any():
            axes[2].hist(mag[mm], bins=bins, histtype="step", density=True,
                         color=colour, label=label, lw=1.6)
    axes[2].set_xlabel("Gradient (m/yr per km)")
    axes[2].set_ylabel("Density")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"39_strain_{scene.name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")

    root = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
    npz = root / f"strain_u_dec{args.decimate}.npz"
    np.savez(npz, rate=rate.astype(np.float32), gx=gx.astype(np.float32),
             gy=gy.astype(np.float32), magnitude=mag.astype(np.float32),
             direction=direction.astype(np.float32), ice=ice, held=held_m,
             radius=args.radius)
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
