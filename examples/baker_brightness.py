#!/usr/bin/env python3
"""The surface's brightness through a GPRI day, as simply as it can be shown.

    python examples/baker_brightness.py --scene 20190719

Two products per campaign, one panel each, both on the UTC clock:

* ``26_db_movie_<scene>.mp4`` — the radar image itself, every epoch,
  geocoded: backscatter in dB from black to white on a grey scale fixed
  over the record, so the glacier going dark by day and bright by night is
  watched directly, and bedrock is in the frame as the control.
* ``26_db_series_<scene>.png`` — one line: the mean backscatter over the
  coherent ice (mean coherence >= 0.5) in the glacier outline against time,
  with the local night (00-06) shaded.

The panels carry no titles and nothing but two-word axis labels; what each
shows is written here and in ``docs/baker.md``.

Nothing is referenced or differenced: these are the numbers as they come
off the SLC.  Both read what ``baker_melt.py --scene X`` left behind — its
cache and the float16 stack of frames beside it — so nothing here touches
an SLC.  The movie is smoothed for display only, and the frame says by how
much: a rolling mean over a few epochs and a light spatial Gaussian, which
take the speckle flicker out of a single-look image.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, open_stack                            # noqa: E402
from baker_melt import db_stack_path, load_melt, melt_path            # noqa: E402
from baker_movie import Resampler, decimated_geom                    # noqa: E402

from gpri_tools.geocode import BAKERBEND1_HEADING                    # noqa: E402
from gpri_tools.heading import scene_heading                         # noqa: E402


def utc_epochs(p):
    """Each epoch as a UTC datetime, from the cache's clock."""
    t0 = np.datetime64(p["epoch0"], "s").astype(dt.datetime)
    return [t0 + dt.timedelta(hours=float(h)) for h in p["epoch_hours"]]


def shade_local_nights(ax, t_lo, t_hi, utc_offset, until=6):
    """Shade 00:00-``until``:00 local of every night the record touches."""
    shift = dt.timedelta(hours=float(utc_offset))
    day = (t_lo + shift).replace(hour=0, minute=0, second=0, microsecond=0) - shift
    while day < t_hi:
        ax.axvspan(max(day, t_lo), min(day + dt.timedelta(hours=until), t_hi),
                   color="0.85", lw=0, zorder=0)
        day += dt.timedelta(days=1)


def series(p, stack_frames, name, args):
    """The mean backscatter over the coherent ice in the glacier outline.

    One line against time; coherent is ``mean coherence >= 0.5``, the mask
    ``baker_melt.py`` took the mean under.
    """
    if "db_ice" in p:
        db_ice = p["db_ice"]
    else:
        ice = p["ice"]
        db_ice = np.array([float(np.nanmean(np.asarray(f, np.float32)[ice]))
                           for f in stack_frames])
    t = utc_epochs(p)
    fig, ax = plt.subplots(figsize=(7.68, 3.4), dpi=110)
    shade_local_nights(ax, t[0], t[-1], p["utc_offset"])
    ax.plot(t, db_ice, color="k", lw=1.0)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Backscatter (dB)")
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.outdir / f"26_db_series_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (glacier mean {np.nanmin(db_ice):.1f} to "
          f"{np.nanmax(db_ice):.1f} dB)")


def movie(p, stack_frames, scene, name, args):
    """The radar image through the day, one geocoded panel with a clock."""
    d = np.asarray(stack_frames, np.float32)           # epochs x azimuth x range

    # ---------------- display smoothing (declared on the frame) ------------
    # both NaN-aware, so a line that is empty in some epochs is averaged
    # over the epochs and neighbours it has rather than lost
    W = max(1, args.t_smooth)
    ok = np.isfinite(d)
    if W > 1:
        from scipy.ndimage import uniform_filter1d
        num = uniform_filter1d(np.where(ok, d, 0.0), W, axis=0, mode="nearest")
        den = uniform_filter1d(ok.astype(np.float32), W, axis=0, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(den > 0.5, num / den, np.nan)
        del num, den
    from scipy.ndimage import gaussian_filter
    if max(args.s_smooth) > 0:
        for k in range(d.shape[0]):
            ok = np.isfinite(d[k])
            den = gaussian_filter(ok.astype(np.float32), args.s_smooth)
            num = gaussian_filter(np.where(ok, d[k], 0.0), args.s_smooth)
            with np.errstate(invalid="ignore", divide="ignore"):
                d[k] = np.where(den > 0.05, num / den, np.nan)

    # ---------------- geocode once, sample per frame -----------------------
    stack = open_stack(scene, antenna=args.antenna)
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    geom = decimated_geom(stack, args.decimate, heading)
    rs = Resampler(geom, args.spacing)
    x0, y0 = geom.origin_xy()

    frames = range(0, d.shape[0], max(1, args.stride))
    # one grey scale for the whole record, from the terrain the analysis
    # uses, so a surface that darkens is seen to darken
    shown = p["cc"] >= args.show_coherence
    sample = d[::max(1, d.shape[0] // 40)][:, shown]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(sample, args.stretch)
    del sample
    print(f"grey scale {lo:.1f} (black) to {hi:.1f} dB (white), {len(frames)} frames")

    # the view: the coherent terrain with a margin, as the LOS movies show it
    first = rs(np.where(shown, d[len(d) // 2], np.nan))
    rr, ccx = np.nonzero(np.isfinite(first))
    xmin, sx, _, ymax, _, sy = rs.transform
    pad = int(1200 / args.spacing)
    if ccx.size == 0:
        rr = np.array([0, first.shape[0] - 1])
        ccx = np.array([0, first.shape[1] - 1])
    x_lo = (xmin + sx * max(ccx.min() - pad, 0)) / 1000
    x_hi = (xmin + sx * min(ccx.max() + pad, first.shape[1])) / 1000
    y_lo = (ymax + sy * min(rr.max() + pad, first.shape[0])) / 1000
    y_hi = (ymax + sy * max(rr.min() - pad, 0)) / 1000
    extent = [xmin / 1000, (xmin + sx * first.shape[1]) / 1000,
              (ymax + sy * first.shape[0]) / 1000, ymax / 1000]

    fig, ax = plt.subplots(figsize=(7.68, 6.40), dpi=100)
    im = ax.imshow(rs(d[len(d) // 2]), cmap="gray", vmin=lo, vmax=hi, extent=extent,
                   origin="upper", interpolation="nearest")
    ax.plot(x0 / 1000, y0 / 1000, "^", ms=9, mfc="w", mec="k", mew=1.4)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("Backscatter (dB)")
    clock = ax.text(0.02, 0.975, "", transform=ax.transAxes, va="top",
                    fontsize=11, family="monospace",
                    bbox=dict(fc="w", alpha=0.8, ec="none"))
    hours = np.asarray(p["epoch_hours"], float)
    cadence = float(np.median(np.diff(hours))) * 60 if hours.size > 1 else 0.0
    ax.text(0.02, 0.02,
            f"smoothed for display: {W} epochs (~{W * cadence:.0f} min), "
            f"gaussian {args.s_smooth} px",
            transform=ax.transAxes, fontsize=7, color="0.35",
            bbox=dict(fc="w", alpha=0.7, ec="none"))
    fig.tight_layout()

    t_utc = utc_epochs(p)
    out = args.outdir / f"26_db_movie_{name}.mp4"
    writer = FFMpegWriter(fps=args.fps, codec="h264",
                          extra_args=["-crf", "24", "-pix_fmt", "yuv420p"])
    t0 = time.time()
    with writer.saving(fig, str(out), dpi=100):
        for i, k in enumerate(frames):
            im.set_data(rs(d[k]))
            clock.set_text(f"{t_utc[k]:%Y-%m-%d %H:%M} UTC   +{hours[k]:5.2f} hr")
            writer.grab_frame()
            if i % 200 == 0:
                print(f"  frame {i + 1}/{len(frames)}  ({time.time() - t0:.0f} s)")
    plt.close(fig)
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(frames)} frames, {size:.1f} MB, "
          f"{len(frames) / args.fps:.0f} s at {args.fps} fps)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--spacing", type=float, default=40.0, help="map grid, m")
    ap.add_argument("--show-coherence", type=float, default=0.4,
                    help="mean coherence of the terrain that sets the view and the grey scale")
    ap.add_argument("--stretch", type=float, nargs=2, default=(1.0, 99.0),
                    help="percentiles of that terrain's dB at black and white")
    ap.add_argument("--t-smooth", type=int, default=5,
                    help="rolling mean over this many epochs, for display")
    ap.add_argument("--s-smooth", type=float, nargs=2, default=(1.0, 2.0),
                    help="gaussian sigma (azimuth, range) px, for display")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--no-movie", action="store_true", help="only the series")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    name = args.scene
    scene = Path(SCENES.get(name, name))
    cache = melt_path(scene, args.antenna, args.decimate)
    spath = db_stack_path(scene, args.antenna, args.decimate)
    p = load_melt(scene, args.antenna, args.decimate)
    if p is None:
        sys.exit(f"no brightness for {name}: run baker_melt.py --scene {name} --recompute")
    stack_frames = np.load(spath, mmap_mode="r")
    if stack_frames.shape[0] != p["epoch_hours"].size:
        sys.exit(f"{spath} and {cache} disagree on the epochs: rerun baker_melt.py "
                 f"--scene {name} --recompute")
    args.outdir.mkdir(parents=True, exist_ok=True)
    series(p, stack_frames, name, args)
    if not args.no_movie:
        movie(p, stack_frames, scene, name, args)


if __name__ == "__main__":
    main()
