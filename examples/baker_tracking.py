#!/usr/bin/env python3
"""Speckle tracking: displacement from the amplitude, beside the phase's answer.

    python examples/baker_tracking.py --scene 20170803_full --hours 12

The phase measures displacement to a fraction of a millimetre and needs
coherence to do it.  Cross-correlating the amplitude pattern of two
acquisitions measures the same displacement with no ambiguity and no
coherence requirement, at a resolution set by the cell size rather than the
wavelength.  :func:`gpri_tools.tracking.patch_offsets` does the correlation;
this script runs it between two epochs of one campaign and puts the answer
beside the phase's.

The comparison is the point.  Over ``--hours`` of a Mount Baker record the
phase reports millimetres of line-of-sight motion, and a range sample here
is tens of centimetres, so tracking *should* read zero on both ice and rock:
the test is whether it does, and how tightly.  That number is the noise
floor of tracking on this instrument, and it says directly how fast a
glacier would have to move before tracking could see it at all — which is
the question worth answering before anyone reaches for tracking on the
faster ice the phase cannot follow.

Range offsets convert to line-of-sight displacement through the range
sample spacing; azimuth offsets convert to cross-range metres through the
arc a line subtends at that range.  Held-out bedrock is carried through as
the control, and the printed table gives both populations.
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
from baker_aps import SCENES, integrate, load, open_stack, split_mask    # noqa: E402
from baker_movie import decimated_geom                                   # noqa: E402
from gpri_tools.aps import epoch_screen_correction, turbulence_screen    # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.glaciers import (glacier_mask, load_outlines,            # noqa: E402
                                 stable_ground_mask)
from gpri_tools.heading import scene_heading                             # noqa: E402
from gpri_tools.timeseries import los_displacement                       # noqa: E402
from gpri_tools.tracking import patch_offsets                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803_full")
    ap.add_argument("--hours", type=float, default=12.0,
                    help="separation between the two epochs tracked")
    ap.add_argument("--patch", type=int, nargs=2, default=(48, 384),
                    help="patch size in (multilooked) cells, azimuth x range; "
                         "a patch has to hold enough terrain texture to match, "
                         "and at this instrument's sampling that means a large "
                         "one")
    ap.add_argument("--step", type=int, nargs=2, default=(24, 192),
                    help="patch spacing; half the patch by default overlaps")
    ap.add_argument("--search", type=int, nargs=2, default=(2, 4))
    ap.add_argument("--looks", type=int, nargs=2, default=(1, 1),
                    help="intensity multilooking before the correlation: "
                         "speckle is what the patches would otherwise match, "
                         "and it is exactly what does not repeat. Offsets are "
                         "converted back to full-resolution samples")
    ap.add_argument("--min-correlation", type=float, default=0.3)
    ap.add_argument("--decimate", type=int, default=16,
                    help="decimation of the phase product compared against")
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--rgi", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")

    # ---- the two SLCs ------------------------------------------------------
    stack = open_stack(scene, "upper", lags=(1,), looks=(1, 1))
    times = np.asarray(stack.network.times, float)
    span = (times[-1] - times[0]) * 24.0
    want = min(args.hours, span)
    j = int(np.argmin(np.abs((times - times[0]) * 24.0 - want)))
    if j == 0:
        sys.exit(f"the record spans {span:.1f} h; --hours must fit inside it")
    dt_h = float((times[j] - times[0]) * 24.0)
    print(f"{scene.name}: tracking epoch 0 against epoch {j}, {dt_h:.2f} h apart "
          f"(record {span:.1f} h)")
    a = stack.read_slc(0)
    b = stack.read_slc(j)
    par = stack.par
    wavelength = stack.wavelength
    stack.close()

    def look(z, looks):
        la_, lr_ = int(looks[0]), int(looks[1])
        p_ = np.abs(z) ** 2
        if (la_, lr_) == (1, 1):
            return p_
        na_ = p_.shape[0] // la_ * la_
        nr_ = p_.shape[1] // lr_ * lr_
        return p_[:na_, :nr_].reshape(na_ // la_, la_, nr_ // lr_, lr_).mean(axis=(1, 3))

    la_l, lr_l = int(args.looks[0]), int(args.looks[1])
    t0 = time.time()
    off = patch_offsets(look(a, args.looks), look(b, args.looks),
                        patch=tuple(args.patch), step=tuple(args.step),
                        search=tuple(args.search))
    print(f"{off}\n  in {time.time() - t0:.0f} s at {la_l} x {lr_l} looks")
    del a, b
    # back to full-resolution samples and lines
    off.azimuth *= la_l
    off.range *= lr_l
    off.rows = off.rows * la_l
    off.cols = off.cols * lr_l

    # ---- what the offsets are worth in metres ------------------------------
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    geom = decimated_geom(stack, 1, heading)
    slant = np.asarray(geom.slant_range(), float)
    d_range = float(np.median(np.diff(slant)))                # m per sample
    bearing = np.deg2rad(np.asarray(geom.bearings(), float))
    d_bearing = float(np.median(np.abs(np.diff(bearing))))    # rad per line
    r_patch = slant[np.clip(off.cols, 0, slant.size - 1)][None, :]
    los_mm = off.range * d_range * 1000.0                     # mm, + away
    cross_m = off.azimuth * d_bearing * r_patch               # m along the arc
    print(f"  a range sample is {d_range * 100:.1f} cm; an azimuth line is "
          f"{d_bearing * 1e3:.3f} mrad, {d_bearing * float(np.median(slant)):.2f} m "
          f"at the median range")

    # ---- the masks and the phase's answer over the same interval -----------
    stack2, net, phase, cc, r, az, n = load(scene, args.decimate, 0,
                                            antenna="upper")
    mean_cc = cc.mean(axis=0)
    del cc
    geom_d = decimated_geom(stack2, args.decimate, heading)
    usable = mean_cc >= args.ice_coherence
    stable = mean_cc >= args.stable_coherence
    if args.rgi:
        la_, lo_ = geom_d.geodetic(rows=[0, geom_d.shape[0] - 1],
                                   cols=[0, geom_d.shape[1] - 1])
        gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                            bbox=(lo_.min() - .02, la_.min() - .02,
                                  lo_.max() + .02, la_.max() + .02))
        stable, _ = stable_ground_mask(mean_cc, geom_d, gdf,
                                       threshold=args.stable_coherence)
        ice = usable & glacier_mask(geom_d, gdf)
    else:
        ice = usable & ~stable
    fit_m, held_m = split_mask(stable)
    d, tt = integrate(los_displacement(phase, stack2.wavelength), net, n)
    del phase
    d, _ = epoch_screen_correction(d, fit_m, r, model="linear", weights=mean_cc)
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], fit_m, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        d[k] -= scr
    phase_mm = (d[j] - d[0]) * 1000.0                          # mm over the same gap
    del d
    stack2.close()

    # patch centres on the decimated grid the masks live on
    pr_rows = np.clip(off.rows // args.decimate, 0, ice.shape[0] - 1)
    pr_cols = np.clip(off.cols // args.decimate, 0, ice.shape[1] - 1)
    ice_p = ice[np.ix_(pr_rows, pr_cols)]
    rock_p = held_m[np.ix_(pr_rows, pr_cols)]
    phase_p = phase_mm[np.ix_(pr_rows, pr_cols)]
    good = off.valid(args.min_correlation)

    print(f"\n{'population':18s} {'patches':>8s} {'range offset':>26s} "
          f"{'LOS (mm)':>20s} {'cross-range (m)':>18s} {'phase LOS (mm)':>16s}")
    rows_out = {}
    for label, m in (("RGI ice", ice_p & good), ("held-out rock", rock_p & good)):
        if not m.any():
            print(f"{label:18s} {'0':>8s}  no patches")
            continue
        med_r = np.nanmedian(off.range[m])
        sd_r = np.nanstd(off.range[m])
        rows_out[label] = (med_r, sd_r, np.nanmedian(los_mm[m]), np.nanstd(los_mm[m]),
                           np.nanmedian(cross_m[m]), np.nanstd(cross_m[m]),
                           np.nanmedian(phase_p[m]))
        print(f"{label:18s} {m.sum():8,d} {med_r:+9.3f} +/- {sd_r:6.3f} samples "
              f"{np.nanmedian(los_mm[m]):+9.1f} +/- {np.nanstd(los_mm[m]):6.1f} "
              f"{np.nanmedian(cross_m[m]):+8.2f} +/- {np.nanstd(cross_m[m]):5.2f} "
              f"{np.nanmedian(phase_p[m]):+15.2f}")
    print(f"\nthe tracking noise floor over {dt_h:.1f} h is the rock row's "
          f"scatter: {rows_out.get('held-out rock', [np.nan] * 7)[3]:.0f} mm in "
          f"line of sight, against the phase's {np.nanstd(phase_p[rock_p & good]):.2f} mm")
    if "RGI ice" in rows_out:
        floor = rows_out["held-out rock"][3] if "held-out rock" in rows_out else np.nan
        print(f"a glacier would have to move {3 * floor / dt_h * 24 / 1000:.1f} m/day "
              f"in line of sight for tracking to see it at three sigma over "
              f"this interval")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=140)
    shown = np.where(good, los_mm, np.nan)
    lim = float(np.nanpercentile(np.abs(shown[np.isfinite(shown)]), 95)) or 1.0
    im0 = axes[0].imshow(shown, origin="lower", cmap="RdBu_r", aspect="auto",
                         vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], label="Tracked LOS (mm)", fraction=0.046)
    im1 = axes[1].imshow(np.where(good, cross_m, np.nan), origin="lower",
                         cmap="RdBu_r", aspect="auto", interpolation="nearest",
                         vmin=-np.nanpercentile(np.abs(cross_m[good]), 95),
                         vmax=np.nanpercentile(np.abs(cross_m[good]), 95))
    fig.colorbar(im1, ax=axes[1], label="Cross-range (m)", fraction=0.046)
    for ax in axes[:2]:
        ax.set_xlabel("Range (patch)")
        ax.set_ylabel("Azimuth (patch)")
    for label, m, colour in (("ice", ice_p & good, "tab:blue"),
                             ("rock", rock_p & good, "tab:red")):
        if m.any():
            axes[2].plot(phase_p[m], los_mm[m], ".", ms=2, alpha=0.3,
                         color=colour, label=label)
    axes[2].set_xlabel("Phase LOS (mm)")
    axes[2].set_ylabel("Tracked LOS (mm)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"36_tracking_{scene.name}_{dt_h:.0f}h.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")

    root = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
    npz = root / f"tracking_u_{dt_h:.0f}h.npz"
    np.savez(npz, azimuth=off.azimuth, range=off.range,
             correlation=off.correlation, rows=off.rows, cols=off.cols,
             los_mm=los_mm, cross_m=cross_m, phase_mm=phase_p,
             ice=ice_p, rock=rock_p, hours=dt_h, d_range=d_range)
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
