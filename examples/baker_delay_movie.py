#!/usr/bin/env python3
"""The air and the ice side by side, frame by frame.

    python examples/baker_delay_movie.py --scene 20170913 \\
           --lags 1 2 3 --looks 3 15 --decimate 1

Two panels on one clock. Left, the atmospheric path delay this acquisition
carries, from the double-difference inversion (`gpri_tools.pathdelay`) run on
multilooked pairs over several temporal baselines — the quantity the method
estimates, mapped rather than averaged. Right, the line-of-sight deformation
left once the correction ladder and that delay have both been taken off.
Below, the two read as curves over the coherent ice, with a cursor on the
frame being shown.

The delay is fitted per pixel and then passed through the same turbulence
screen the ladder uses, on the pixels the analysis trusts, so what is mapped
is the part neighbouring pixels agree on rather than each pixel's own noise.
The weight is floored (`--protect-period`) so the correction cannot take a
diurnal signal out of the deformation panel with it.

Multilooking is what makes this worth doing: at single look the longer
baselines close with the chain exactly and carry nothing, and the per-pixel
field is mostly noise. See `docs/pathdelay.md`. With `--rewrap` each longer
baseline is first moved by whole cycles onto the chain it spans
(`gpri_tools.pathdelay.rewrap_to_chain`), as `baker_pathdelay.py --rewrap`
does.

Display smoothing is for the eye only and is declared on the frame.
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
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, integrate, load                        # noqa: E402
from baker_brightness import shade_local_nights                      # noqa: E402
from baker_movie import Resampler, decimated_geom                    # noqa: E402

from gpri_tools.aps import epoch_screen_correction, turbulence_screen    # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask  # noqa: E402
from gpri_tools.heading import scene_heading                              # noqa: E402
from gpri_tools.pathdelay import pair_delay_field, rewrap_to_chain        # noqa: E402
from gpri_tools.timeseries import los_displacement                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170913")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=1)
    ap.add_argument("--looks", type=int, nargs=2, default=(3, 15))
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--rewrap", action="store_true",
                    help="move each longer baseline by whole cycles onto the "
                         "chain it spans (gpri_tools.pathdelay.rewrap_to_chain) "
                         "before the delay is fitted")
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--protect-period", type=float, default=1.0)
    ap.add_argument("--max-response", type=float, default=0.01)
    ap.add_argument("--stable-coherence", type=float, default=0.6)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--show-coherence", type=float, default=0.4)
    ap.add_argument("--spacing", type=float, default=40.0,
                    help="output map spacing, metres")
    ap.add_argument("--t-smooth", type=int, default=5,
                    help="rolling mean over this many epochs, display only")
    ap.add_argument("--s-smooth", type=float, nargs=2, default=(1.0, 1.0),
                    help="Gaussian sigma on each frame, display only")
    ap.add_argument("--refractivity", action="store_true",
                    help="show the delay as the equivalent uniform refractivity "
                         "along the path, N-units, rather than mm of LOS")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--utc-offset", type=float, default=-7.0,
                    help="local time minus UTC, hours; local nights 00-06 are "
                         "shaded on the time strip")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)

    stack, net, phase, cc, r, az, n = load(
        scene, args.decimate, 0, antenna=args.antenna,
        lags=tuple(int(l) for l in args.lags),
        looks=tuple(int(l) for l in args.looks))
    mean_cc = cc.mean(axis=0)
    del cc

    geom = decimated_geom(stack, args.decimate, heading)
    la_, lo_ = geom.geodetic(rows=[0, geom.shape[0] - 1],
                             cols=[0, geom.shape[1] - 1])
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                        bbox=(lo_.min() - .02, la_.min() - .02,
                              lo_.max() + .02, la_.max() + .02))
    stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                   threshold=args.stable_coherence)
    ice = (mean_cc >= args.ice_coherence) & glacier_mask(geom, gdf)
    show = mean_cc >= args.show_coherence
    trusted = ice | stable
    print(f"{scene.name}: {n:,} pairs on a {phase.shape[1]} x {phase.shape[2]} "
          f"grid; bedrock {stable.sum():,} px, ice {ice.sum():,} px")

    obs = (los_displacement(phase, stack.wavelength) * 1000.0).astype(np.float32)
    del phase
    pairs = np.asarray(net.pairs[:n], int)
    times = np.asarray(net.times, float)
    chain = np.array([k for k, (i, j) in enumerate(pairs) if j == i + 1])

    # ---- the ladder, on the consecutive chain: that is the product ---------
    class _Chain:
        pass
    chain_net = _Chain()
    chain_net.pairs = [tuple(pairs[k]) for k in chain]
    chain_net.times = times
    d, _ = integrate(obs[chain], chain_net, chain.size)
    d, _ = epoch_screen_correction(d, stable, r, model="linear", weights=mean_cc)
    t0 = time.time()
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], stable, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        d[k] -= scr
    print(f"ladder in {time.time() - t0:.0f} s")

    # ---- the delay, from every measured baseline --------------------------
    if args.rewrap:
        # A pair unwrapped on its own is known modulo half a wavelength; put
        # each longer baseline on the cycle nearest the chain it spans.
        obs, moved = rewrap_to_chain(obs, pairs, stack.wavelength / 2 * 1000.0)
        lag = pairs[:, 1] - pairs[:, 0]
        print("rewrapped onto the chain: " + (", ".join(
            f"lag {L} moved {100 * moved[lag == L].mean():.1f} % of samples"
            for L in np.unique(lag) if L > 1) or "nothing to move at lag 1"))
        del moved
    t0 = time.time()
    field, lam = pair_delay_field(obs, pairs, times, trusted, weights=mean_cc,
                                  sigma=tuple(args.sigma), protect_period=args.protect_period,
                                  max_response=args.max_response)
    field = field.astype(np.float32)
    del obs
    print(f"path delay (lambda {lam:.4g}) in {time.time() - t0:.0f} s; "
          f"field sd {np.nanstd(field):.3f} mm")
    d = d - field                         # the deformation panel

    # ---- display smoothing, declared on the frame -------------------------
    from scipy.ndimage import gaussian_filter, uniform_filter1d
    W = max(1, args.t_smooth)
    if W > 1:
        d = uniform_filter1d(d, W, axis=0, mode="nearest")
        field = uniform_filter1d(field, W, axis=0, mode="nearest")
    if max(args.s_smooth) > 0:
        good = show & np.isfinite(d).all(axis=0)
        den = gaussian_filter(good.astype(np.float32), args.s_smooth)
        for k in range(d.shape[0]):
            for arr in (d, field):
                num = gaussian_filter(np.where(good, arr[k], 0.0).astype(np.float32),
                                      args.s_smooth)
                with np.errstate(invalid="ignore", divide="ignore"):
                    arr[k] = np.where(den > 0.05, num / den, np.nan)
    d[:, ~show] = np.nan
    field[:, ~show] = np.nan

    # ---- render ------------------------------------------------------------
    res = Resampler(geom, args.spacing)
    keep = range(0, d.shape[0], max(1, args.stride))

    # the delay is path-integrated, so the refractivity it implies is the
    # uniform dn that would produce it over that pixel's range: d = -dn r
    delay_label = "Path delay (mm)"
    if args.refractivity:
        with np.errstate(invalid="ignore", divide="ignore"):
            field = (-1000.0 * field / np.maximum(r[None, None, :], 1.0)
                     ).astype(np.float32)
        delay_label = "Refractivity (N)"

    dlim = float(np.nanpercentile(np.abs(field), 99)) or 1.0
    ulim = float(np.nanpercentile(np.abs(d), 99)) or 1.0
    ice_delay = np.array([np.nanmean(x[ice]) for x in field])
    ice_disp = np.array([np.nanmean(x[ice]) for x in d])
    import datetime as dt
    t_utc = [net.epochs[0] + dt.timedelta(days=float(t - times[0])) for t in times]

    fig = plt.figure(figsize=(11.0, 6.4), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[4.4, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_i = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])
    im_a = ax_a.imshow(res(field[0]), origin="lower", cmap="RdBu_r",
                       vmin=-dlim, vmax=dlim, interpolation="nearest")
    im_i = ax_i.imshow(res(d[0]), origin="lower", cmap="RdBu_r",
                       vmin=-ulim, vmax=ulim, interpolation="nearest")
    fig.colorbar(im_a, ax=ax_a, label=delay_label, fraction=0.046)
    fig.colorbar(im_i, ax=ax_i, label="LOS displacement (mm)", fraction=0.046)
    # crop both panels to where there is data: the coherent swath is a small
    # part of the map frame and the rest is white
    seen = np.isfinite(res(np.where(show, 0.0, np.nan)))
    if seen.any():
        rows_, cols_ = np.where(seen)
        pad = 4
        for ax in (ax_a, ax_i):
            ax.set_xlim(max(cols_.min() - pad, 0), min(cols_.max() + pad, seen.shape[1] - 1))
            ax.set_ylim(max(rows_.min() - pad, 0), min(rows_.max() + pad, seen.shape[0] - 1))
    for ax in (ax_a, ax_i):
        ax.set_xticks([]); ax.set_yticks([])
    if args.refractivity:
        ice_delay = np.array([np.nanmean(x[ice]) for x in field])
    ax_t.plot(t_utc, ice_delay, color="tab:green", lw=0.9, label="delay")
    ax_t.plot(t_utc, ice_disp, color="k", lw=0.9, label="displacement")
    ax_t.set_ylabel("Ice mean" if args.refractivity else "Ice mean (mm)")
    ax_t.set_xlabel("Time (UTC)")
    ax_t.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax_t.grid(alpha=0.3)
    shade_local_nights(ax_t, t_utc[0], t_utc[-1], args.utc_offset)
    ax_t.set_xlim(t_utc[0], t_utc[-1])
    cursor = ax_t.axvline(t_utc[0], color="0.4", lw=1.0)
    stamp = ax_a.text(0.02, 0.97, "", transform=ax_a.transAxes, va="top",
                      fontsize=9, color="0.15")
    note = (f"{args.looks[0]}x{args.looks[1]} looks, lags "
            f"{'+'.join(map(str, args.lags))}{' rewrapped' if args.rewrap else ''}; "
            f"display: {W}-epoch mean, Gaussian {args.s_smooth[0]:g}x{args.s_smooth[1]:g} px")
    ax_i.text(0.02, 0.03, note, transform=ax_i.transAxes, fontsize=7, color="0.35")
    fig.tight_layout()

    tag = f"_lk{args.looks[0]}x{args.looks[1]}" + ("_N" if args.refractivity else "")
    out = args.outdir / f"29_delay_movie_{scene.name}{tag}.mp4"
    args.outdir.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, codec="h264",
                          extra_args=["-pix_fmt", "yuv420p"])
    t0 = time.time()
    with writer.saving(fig, str(out), dpi=100):
        for k in keep:
            im_a.set_data(res(field[k]))
            im_i.set_data(res(d[k]))
            cursor.set_xdata([t_utc[k], t_utc[k]])
            stamp.set_text(t_utc[k].strftime("%Y-%m-%d %H:%M UTC"))
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out}  ({len(list(keep))} frames, "
          f"{out.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
