#!/usr/bin/env python3
"""What the diurnal waveform's amplitude follows, pixel by pixel.

    python examples/baker_pixels.py --scene 20170803_full

The population series (``baker_population.py``) says the ice as a whole
carries a diurnal waveform that bedrock does not.  This script asks *which*
ice carries it, and how much, by projecting every pixel's corrected series
onto the population waveform (:func:`gpri_tools.diurnal.waveform_share`): a
share of 1 is a pixel that moves like the median, 0 is one that does not
move with it.  The share is then binned by what is known per pixel.

Three hypotheses make three different predictions:

* a **fractional speed-up** of the flow — the share is proportional to the
  pixel's own secular LOS rate, and stagnant ice has none;
* **residual atmosphere** — the share is a smooth function of range and
  height, and the held-out bedrock at the same range and height carries it
  too, at least before the turbulence screen interpolates it away there
  (so the comparison is made at stage B, after the range-linear epoch
  screen and *before* the turbulence screen, as well as at stage C);
* something **on the ice surface** — the share follows height (the snow),
  is absent on bare rock at the same range, and is independent of the flow.

:func:`gpri_tools.diurnal.slope_within` gives the fixed-effects version:
how the share varies with the secular rate among pixels that agree on range
and height, and with height among pixels that agree on range and rate.

The radar also records the surface's *brightness* every epoch, which is
independent of the phase.  Wet snow is dark at Ku band and refrozen snow is
bright, so the per-epoch backscatter of the ice by height band
(:meth:`gpri_tools.stack.SlcPairStack.backscatter`, bedrock as the gain
reference) says whether the surface itself cycled over the day.

Everything per pixel is cached in ``work/<scene>/pixels_u_dec16.npz``; the
tables and figure are rebuilt from it.
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
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, integrate, load, split_mask          # noqa: E402
from baker_north_side import decimated_par                          # noqa: E402
from baker_population import detrend_pixels                          # noqa: E402

from gpri_tools.aps import epoch_screen_correction, turbulence_screen    # noqa: E402
from gpri_tools.diurnal import (DIURNAL, MIN_CYCLES, m_per_yr, secular_slope,  # noqa: E402
                                slope_within, waveform_share)
from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry          # noqa: E402
from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask  # noqa: E402
from gpri_tools.heading import scene_heading, target_heights              # noqa: E402
from gpri_tools.timeseries import los_displacement                        # noqa: E402

RATE_EDGES = np.array([-30, -5, 0, 5, 10, 20, 30, 45, 60, 90, 150])   # m/yr
HEIGHT_STEP = 200.0                                                   # m
RANGE_STEP = 1000.0                                                   # m


def pixels_path(scene: Path, antenna: str, dec: int) -> Path:
    root = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
    return root / scene.name / f"pixels_{antenna[0].lower()}_dec{dec}.npz"


def compute(scene, args):
    """Per-pixel shares, rates, geometry and backscatter for one scene."""
    stack, net, phase, cc, r, az, n = load(scene, args.decimate, 0,
                                           antenna=args.antenna)
    lam = stack.wavelength
    mean_cc = cc.mean(axis=0)
    del cc
    geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                         heading=scene_heading(scene, default=BAKERBEND1_HEADING))
    la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1], cols=[0, geom.shape[1] - 1])
    bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"), bbox=bbox)
    stable, _ = stable_ground_mask(mean_cc, geom, gdf, threshold=args.stable_coherence)
    ice = (mean_cc >= args.ice_coherence) & glacier_mask(geom, gdf)
    fit_m, held_m = split_mask(stable)
    dem = os.environ.get("GPRI_DEM", "")
    if not Path(dem).exists():
        sys.exit("baker_pixels.py needs GPRI_DEM to point at a DEM tile")
    z = target_heights(geom, dem)
    # how far each pixel is from the bedrock the screens are built on, in
    # units of the turbulence screen's kernel
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~fit_m, sampling=(1 / args.sigma[0], 1 / args.sigma[1]))

    d, times = integrate(los_displacement(phase, lam), net, n)
    del phase
    t = np.asarray(times, float)
    tc = t - t.mean()
    stages = {}
    # stage B: the range-linear epoch screen from the fit half of the bedrock
    d, _ = epoch_screen_correction(d, fit_m, r, model="linear", weights=mean_cc)
    stages["B"] = detrend_pixels(t, d)[0] * 1000            # mm
    # stage C: the turbulence screen on top, as every other product has it
    t0 = time.time()
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], fit_m, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        d[k] -= scr
    print(f"corrections in {time.time() - t0:.0f} s")
    da, rate_px = detrend_pixels(t, d)
    del d
    stages["C"] = da * 1000

    # the population waveform, from the same line the population script uses
    lin_ice = np.nanmedian(stages["C"][:, ice], axis=1)
    tol = (1 - MIN_CYCLES) * DIURNAL
    try:
        tilt = secular_slope(t, lin_ice, tolerance=tol)[0]
    except ValueError:
        tilt = 0.0
    template = lin_ice - tilt * tc
    out = {"template": template, "hours": t * 24,
           "origin": net.epochs[0].hour + net.epochs[0].minute / 60.0,
           "epoch0": np.datetime64(net.epochs[0]).astype("datetime64[s]")}
    for tag, da in stages.items():
        da[:, ice] -= tilt * tc[:, None]
        share, se = waveform_share(da, template)
        out["share_" + tag], out["se_" + tag] = share, se
        out["ice_" + tag] = np.nanmedian(da[:, ice], axis=1)
        out["rock_" + tag] = np.nanmedian(da[:, held_m], axis=1)
    del stages
    print(f"waveform RMS {np.nanstd(template):.2f} mm; per-pixel share: ice median "
          f"{np.nanmedian(out['share_C'][ice]):.2f}, held-out bedrock "
          f"{np.nanmedian(out['share_C'][held_m]):+.2f}")

    # ---- the surface's brightness, epoch by epoch ------------------------
    bands = {}
    edges = [-np.inf] + list(args.bands) + [np.inf]
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        bands[f"ice_{lo_:.0f}_{hi_:.0f}"] = ice & (z >= lo_) & (z < hi_)
    bands["rock_held"], bands["rock_fit"] = held_m, fit_m
    db = {k: np.full(t.size, np.nan) for k in bands}
    c = template - np.nanmean(template)
    # per pixel: the mean brightness, its spread over the record, and its
    # projection on the waveform (dB per mm), the fit-half bedrock's median
    # taken out of every frame as the instrument's gain
    mean_db, sq_db, n_db = (np.zeros(ice.shape) for _ in range(3))
    # the projection is a covariance over the epochs the pixel actually has:
    # the waveform sums to zero over the whole record but not over what is
    # left of it once a dead azimuth line has taken some epochs away
    n_c, sum_y, sum_c, sum_yc, sum_cc = (np.zeros(ice.shape) for _ in range(5))
    if hasattr(stack, "backscatter"):
        t0 = time.time()
        for e in range(t.size):
            frame = stack.backscatter(e, looks=(1, args.decimate))[:, :ice.shape[1]]
            frame[frame < -200] = np.nan      # zero power: an empty line, not a surface
            for k, m in bands.items():
                db[k][e] = np.nanmedian(frame[m])
            frame = frame - db["rock_fit"][e]
            ok = np.isfinite(frame)
            frame = np.where(ok, frame, 0.0)
            mean_db += frame
            sq_db += frame * frame
            n_db += ok
            if np.isfinite(c[e]):
                n_c += ok
                sum_y += frame
                sum_c += ok * c[e]
                sum_yc += frame * c[e]
                sum_cc += ok * (c[e] * c[e])
            if e % 200 == 0:
                print(f"  backscatter {e + 1}/{t.size} ({time.time() - t0:.0f} s)")
        n = np.where(n_db > 0, n_db, np.nan)
        mean_db /= n
        sd_db = np.sqrt(np.maximum(sq_db / n - mean_db ** 2, 0.0))
        nc = np.where(n_c > 1, n_c, np.nan)
        var_c = sum_cc - sum_c * sum_c / nc
        gamma = np.where(var_c > 0, (sum_yc - sum_y * sum_c / nc) / var_c, np.nan)
    else:
        print("no SLCs behind this stack (GAMMA diff0 products): no backscatter")
        sd_db = np.full(ice.shape, np.nan)
        gamma = np.full(ice.shape, np.nan)
        mean_db[:] = np.nan
    out.update({"db_" + k: v for k, v in db.items()})
    out.update(rate=m_per_yr(rate_px * 1000 + np.where(ice, tilt, 0.0), "mm"),
               r=np.broadcast_to(r, ice.shape).copy(), z=z, cc=mean_cc,
               dist=dist, ice=ice, fit=fit_m, held=held_m, gamma=gamma,
               mean_db=mean_db, sd_db=sd_db, bands=np.asarray(args.bands, float))
    return out


def binned(x, y, edges, mask, min_count=20):
    """Median, quartiles and count of ``y`` in bins of ``x`` over ``mask``."""
    mid, med, q1, q3, cnt = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = mask & (x >= a) & (x < b) & np.isfinite(y)
        if m.sum() < min_count:
            continue
        p = np.percentile(y[m], [25, 50, 75])
        mid.append(np.median(x[m])); med.append(p[1]); q1.append(p[0]); q3.append(p[2])
        cnt.append(int(m.sum()))
    return (np.array(mid), np.array(med), np.array(q1), np.array(q3), np.array(cnt))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803_full")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--bands", type=float, nargs="+", default=(1800, 2200, 2600),
                    help="height edges of the backscatter bands, m")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene, args.scene))
    day = scene.name + ("" if args.antenna == "upper" else f"_{args.antenna}")

    npz = pixels_path(scene, args.antenna, args.decimate)
    if npz.exists() and not args.recompute:
        p = dict(np.load(npz, allow_pickle=True))
        print(f"read {npz}")
    else:
        p = compute(scene, args)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **p)
        print(f"wrote {npz}")

    ice, held = p["ice"], p["held"]
    v, r, z, dist = p["rate"], p["r"] / 1000, p["z"], p["dist"]
    sB, sC = p["share_B"], p["share_C"]
    good = np.isfinite(sB) & np.isfinite(sC) & np.isfinite(v) & np.isfinite(z)
    I, R = ice & good, held & good
    rms = np.nanstd(p["template"])
    hours = p["hours"]
    print(f"\n{day}: waveform RMS {rms:.2f} mm over {hours[-1]:.1f} h; "
          f"{I.sum():,} ice px, {R.sum():,} held-out bedrock px")
    print(f"  stage B (range screen only): ice RMS {np.nanstd(p['ice_B']):.2f} mm, "
          f"held-out bedrock {np.nanstd(p['rock_B']):.2f} mm;  "
          f"stage C (+ turbulence): ice {np.nanstd(p['ice_C']):.2f}, "
          f"bedrock {np.nanstd(p['rock_C']):.2f} mm")

    def table(x, edges, label):
        print(f"\n  share of the waveform by {label}: median (n)")
        print(f"  {'bin':>14s} {'ice B':>7s} {'ice C':>7s} {'n':>6s}   {'rock B':>7s} "
              f"{'rock C':>7s} {'n':>5s}   {'rate':>6s} {'z':>5s} {'r':>4s}")
        for a, b in zip(edges[:-1], edges[1:]):
            mi = I & (x >= a) & (x < b)
            mr = R & (x >= a) & (x < b)
            if mi.sum() < 20 and mr.sum() < 20:
                continue
            f = lambda m, s: f"{np.median(s[m]):7.2f}" if m.sum() >= 20 else "      ."
            print(f"  {a:6.4g}..{b:<7.4g} {f(mi, sB)} {f(mi, sC)} {mi.sum():6d}   "
                  f"{f(mr, sB)} {f(mr, sC)} {mr.sum():5d}   "
                  f"{np.median(v[mi]) if mi.sum() else np.nan:6.1f} "
                  f"{np.median(z[mi]) if mi.sum() else np.nan:5.0f} "
                  f"{np.median(r[mi]) if mi.sum() else np.nan:4.1f}")

    table(v, RATE_EDGES, "secular LOS rate (m/yr)")
    table(z, np.arange(1000, 3400, HEIGHT_STEP), "height (m)")
    table(r, np.arange(0, 12000, RANGE_STEP) / 1000, "slant range (km)")

    # the fixed-effects slopes: what the share does with one variable among
    # pixels that agree on the others
    rc, zc, vc = np.floor(r / 0.5), np.floor(z / 100), np.floor(v / 10)
    print("\n  holding the rest fixed (ice pixels, stage C):")
    for what, x, keys, unit, scale in (
            ("secular rate", v, (rc, zc), "per 10 m/yr", 10),
            ("height", z, (rc, vc), "per 100 m", 100),
            ("range", r, (zc, vc), "per km", 1),
            ("distance from bedrock", dist, (rc, zc), "per screen sigma", 1)):
        y = np.where(I, sC, np.nan)
        s, corr, cells, npx = slope_within(y, x, *keys)
        print(f"    share vs {what:22s} {s * scale:+.3f} {unit:16s} r {corr:+.2f} "
              f"({cells} cells, {npx:,} px)")
    if np.isfinite(p["gamma"]).any():
        # the surface's own diurnal cycle, per pixel: does the ice that
        # brightens and darkens most carry the most waveform, among pixels
        # at one range and height?
        g = np.where(I, p["gamma"] * 10, np.nan)               # dB per 10 mm
        s, corr, cells, npx = slope_within(np.where(I, sC, np.nan), g, rc, zc)
        print(f"    share vs {'backscatter cycle':22s} {s:+.3f} {'per dB/10 mm':16s} "
              f"r {corr:+.2f} ({cells} cells, {npx:,} px)")
    prop = np.median(sC[I & (v > 5)]) / np.median(v[I & (v > 5)])
    slow, away = I & (v >= 0) & (v < 5), I & (v < -5)
    print(f"    a fractional speed-up would give {prop * 10:.2f} per 10 m/yr through "
          f"the origin; ice moving under 5 m/yr carries {np.median(sC[slow]):.2f} "
          f"of the waveform ({slow.sum():,} px)")
    if away.sum() >= 20:
        # a modulation of the flow changes sign with the flow's LOS direction
        print(f"    ice flowing away from the radar (under -5 m/yr) carries "
              f"{np.median(sC[away]):+.2f} ({away.sum():,} px): a modulation of the "
              f"flow would carry the waveform inverted there")

    # ---- backscatter --------------------------------------------------------
    ref = p["db_rock_fit"] - np.nanmean(p["db_rock_fit"])      # the instrument
    c = p["template"] - np.nanmean(p["template"])
    keys = [k[3:] for k in p if k.startswith("db_") and k != "db_rock_fit"]
    have_db = np.isfinite(p["db_rock_fit"]).any()
    if have_db:
        print(f"\n  backscatter anomaly by band, bedrock gain removed:")
        print(f"  {'band':16s} {'RMS (dB)':>9s} {'r(dB, ice mm)':>14s} {'dB per 10 mm':>13s} "
              f"{'peak-to-peak':>13s}")
        for k in keys:
            y = p["db_" + k] - np.nanmean(p["db_" + k]) - ref
            ok = np.isfinite(y) & np.isfinite(c)
            slope = (y[ok] * c[ok]).sum() / (c[ok] ** 2).sum() * 10
            print(f"  {k:16s} {np.nanstd(y):9.2f} {np.corrcoef(y[ok], c[ok])[0, 1]:+14.2f} "
                  f"{slope:+13.2f} {np.nanmax(y) - np.nanmin(y):13.2f}")

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ext = [r[0, 0], r[0, -1], 0, ice.shape[0]]
    show = np.where(ice | held, sC, np.nan)
    ax = axes[0, 0]
    im = ax.imshow(show, extent=ext, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-2, vmax=2, interpolation="nearest")
    ax.contour(np.broadcast_to(r[0], ice.shape), np.arange(ice.shape[0])[:, None]
               * np.ones(ice.shape[1]), (p["fit"] | held).astype(float), levels=[0.5],
               colors="k", linewidths=0.3)
    ax.set_title("Share of the ice waveform per pixel (bedrock outlined)", fontsize=10)
    ax.set_xlabel("Slant range (km)"); ax.set_ylabel("Azimuth (px)")
    ax.set_xlim(0, min(11, ext[1]))
    plt.colorbar(im, ax=ax, label="Share")
    ax = axes[0, 1]
    im = ax.imshow(np.where(ice, v, np.nan), extent=ext, aspect="auto", origin="lower",
                   cmap="viridis", vmin=-20, vmax=100, interpolation="nearest")
    ax.set_title("Secular LOS rate, positive towards the radar", fontsize=10)
    ax.set_xlabel("Slant range (km)"); ax.set_ylabel("Azimuth (px)")
    ax.set_xlim(0, min(11, ext[1]))
    plt.colorbar(im, ax=ax, label="Rate (m/yr)")

    ax = axes[0, 2]
    mid, med, q1, q3, cnt = binned(v, sC, RATE_EDGES, I)
    ax.fill_between(mid, q1, q3, color="0.85", label="ice quartiles")
    ax.plot(mid, med, "k.-", label="ice median")
    vv = np.linspace(0, max(mid), 50)
    ax.plot(vv, prop * vv, "k:", label="proportional")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Rate (m/yr)"); ax.set_ylabel("Share")
    ax.set_title("Share against each pixel's secular rate\n"
                 "(dotted: what a fractional speed-up would give)", fontsize=10)
    ax.legend(fontsize=8)

    for ax, x, edges, xlabel, title in (
            (axes[1, 0], z, np.arange(1000, 3400, HEIGHT_STEP), "Height (m)",
             "Share against target height"),
            (axes[1, 1], r, np.arange(0, 12, 1.0), "Slant range (km)",
             "Share against slant range")):
        for m, s, colour, lab, ls in ((I, sB, "0.5", None, "--"),
                                      (I, sC, "k", "ice", "-"),
                                      (R, sB, "tab:red", None, "--"),
                                      (R, sC, "tab:red", "held-out bedrock", "-")):
            mid, med, q1, q3, cnt = binned(x, s, edges, m)
            if mid.size:
                ax.plot(mid, med, color=colour, ls=ls, marker=".", label=lab)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(xlabel); ax.set_ylabel("Share")
        ax.set_title(title + "\n(dashed: range screen only; solid: + turbulence screen)",
                     fontsize=10)
        ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(hours, p["template"], "k", lw=1.2, label="ice anomaly")
    ax.set_ylabel("Anomaly (mm)"); ax.set_xlabel("Elapsed time (hr)")
    ax.axhline(0, color="k", lw=0.5)
    if have_db:
        ax2 = ax.twinx()
        icekeys = [k for k in keys if k.startswith("ice_")]
        colours = plt.cm.coolwarm(np.linspace(0, 1, len(icekeys)))
        for k, colour in zip(icekeys, colours):
            y = p["db_" + k] - np.nanmean(p["db_" + k]) - ref
            lo_, hi_ = k.split("_")[1:]
            lab = (f"ice below {hi_} m" if lo_ == "-inf" else
                   f"ice above {lo_} m" if hi_ == "inf" else f"ice {lo_}-{hi_} m")
            ax2.plot(hours, y, color=colour, lw=0.8, label=lab)
        y = p["db_rock_held"] - np.nanmean(p["db_rock_held"]) - ref
        ax2.plot(hours, y, color="0.5", lw=0.8, label="held-out bedrock")
        ax2.set_ylabel("Backscatter anomaly (dB)")
        ax2.legend(fontsize=7, loc="upper right")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Ice anomaly and the surface's brightness by height band", fontsize=10)
    origin = float(p["origin"])
    top = ax.secondary_xaxis("top")
    utc = np.arange(np.ceil(origin / 6) * 6, origin + hours[-1] + 1e-9, 6)
    top.set_xticks(utc - origin)
    top.set_xticklabels([f"{int(u) % 24:02d}" for u in utc])
    top.set_xlabel("UTC (hr)")
    fig.suptitle(f"{day}: which ice carries the diurnal waveform, from "
                 f"{p['epoch0']} UTC", fontsize=11)
    plt.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"23_pixels_{day}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
