#!/usr/bin/env python3
"""What the corrected bedrock still does together, and whether it transfers.

    python examples/baker_modes.py --scene 20170803_full

After the correction ladder (stages C and D of `baker_aps.py`, fitted here on
one half of the bedrock) the other half keeps a 24 h harmonic in its mean
series.  This script decomposes the post-ladder residual over the fitted
half into temporal modes and per-pixel loadings (`gpri_tools.modes`) and
puts two questions to the held-out half, which never enters the fit:

    interpolated   each mode's loading is smoothed from the fitted pixels
                   onto the held-out ones with the ladder's own screen
                   (`--sigma`), and the correction that predicts is
                   subtracted.  This is the honest score.
    self-fit       each held-out pixel is regressed on the mode series
                   itself: the most the modes could remove if every pixel's
                   loading were measured rather than interpolated.  The
                   interpolated score is read against this bound.

The table gives, for the ladder alone and for each variant, the sd of the
held-out mean series and its 24 h swing (peak to peak, from
`gpri_tools.diurnal.fit_harmonics` with an offset and a trend); the sd
pooled over every held-out pixel; the median over held-out pixels of the
2 h velocity scatter; and, on the coherent ice, the mean series' 24 h swing
and its rate.  Singular values are printed against a shuffled-epoch null,
with the 24 h swing of each mode series.

Outputs `figures/30_modes_<scene>.png` and caches the modes, the table and
the figure's series in `work/<scene>/modes_u_dec16.npz`.
"""
from __future__ import annotations

import argparse
import gc
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
from baker_aps import SCENES, integrate, load                          # noqa: E402
from baker_brightness import shade_local_nights, utc_epochs           # noqa: E402
from baker_pathdelay import masks_for                                 # noqa: E402

from gpri_tools.aps import epoch_screen_correction, turbulence_screen      # noqa: E402
from gpri_tools.diurnal import fit_harmonics, m_per_yr                     # noqa: E402
from gpri_tools.modes import (fit_modes, mode_correction, mode_projection,  # noqa: E402
                              mode_screens)
from gpri_tools.timeseries import los_displacement                         # noqa: E402

MODES_CACHE_VERSION = 1

#: flags that change what the cached numbers answer
CACHE_ARGS = ("ice_coherence", "stable_coherence", "sigma", "loading_sigma", "k")

#: ranks scored with interpolated loadings, and with each pixel's own
RANKS = (1, 2, 3, 5, 10)
SELF_RANKS = (1, 3, 10)
COLUMNS = ("held sd", "held 24 h", "held px sd", "held px 2 h",
           "ice 24 h", "ice rate")


def run_tag(args) -> str:
    """Loading screens narrower or wider than the ladder's get their own files."""
    if args.loading_sigma is None:
        return ""
    return "_ls{:g}x{:g}".format(*args.loading_sigma)


def modes_path(scene: Path, antenna: str, dec: int, tag="") -> Path:
    root = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
    return root / scene.name / f"modes_{antenna[0].lower()}_dec{dec}{tag}.npz"


def load_modes(scene: Path, args):
    """The cached run, or ``(None, reason)`` when it answers another question."""
    cache = modes_path(scene, args.antenna, args.decimate, run_tag(args))
    if not cache.exists():
        return None, "no cache"
    c = dict(np.load(cache, allow_pickle=False))
    if int(c.get("cache_version", 0)) < MODES_CACHE_VERSION:
        return None, f"older than cache version {MODES_CACHE_VERSION}"
    for k in CACHE_ARGS:
        want = np.atleast_1d(np.asarray(cache_arg(args, k), float))
        have = np.atleast_1d(np.asarray(c.get(k, np.nan), float))
        if have.shape != want.shape or not np.array_equal(have, want):
            return None, f"built with a different --{k.replace('_', '-')}"
    return c, ""


def cache_arg(args, k):
    v = getattr(args, k)
    return args.sigma if (k == "loading_sigma" and v is None) else v


def ladder(d, fit, r, mean_cc, sigma):
    """Stages C and D of `baker_aps.py`, fitted on ``fit`` alone."""
    d, _ = epoch_screen_correction(d, fit, r, model="linear", weights=mean_cc)
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], fit, sigma=tuple(sigma), weights=mean_cc,
                                   wrapped=False)
        d[k] -= scr
    return d


def swing(series, times):
    """Peak-to-peak 24 h harmonic of a series (offset and trend beside it),
    its explained variance, and the trend per day."""
    ok = np.isfinite(series)
    h = fit_harmonics(series[ok], times[ok])
    return (2.0 * float(h.amplitude()), float(h.explained_variance()),
            float(h.secular))


def velocity_scatter(hours, X, window_h=2.0):
    """Median over pixels of the sd of each pixel's ``window_h`` velocity.

    The central difference of `baker_lag.velocity_anomaly`, every pixel at
    once: the window's end epochs are the same for all of them.
    """
    t = np.asarray(hours, float)
    a = np.searchsorted(t, t - window_h / 2)
    b = np.searchsorted(t, t + window_h / 2) - 1
    ok = (b > a) & (t[b] > t[a])
    a, b = a[ok], b[ok]
    v = (X[b] - X[a]) / (t[b] - t[a])[:, None]
    return float(np.nanmedian(np.nanstd(v, axis=0)))


def score(d, times, hours, held, ice):
    """One row of the table, plus the held-out mean series."""
    H = d[:, held].astype(np.float64)
    rs = np.nanmean(H, axis=1)
    sd = float(np.nanstd(rs))
    swing_h, _, _ = swing(rs, times)
    H -= np.nanmean(H, axis=0)
    px_sd = float(np.sqrt(np.nanmean(H ** 2)))
    px_2h = velocity_scatter(hours, H)
    del H
    pop = np.nanmean(d[:, ice].astype(np.float64), axis=1)
    swing_i, _, trend = swing(pop, times)
    return np.array([sd, swing_h, px_sd, px_2h, swing_i,
                     float(m_per_yr(trend, "mm"))]), rs


def compute(scene, name, args):
    stack, net, phase, cc, r, az, n = load(scene, args.decimate, 0, antenna=args.antenna)
    mean_cc = cc.mean(axis=0)
    del cc
    masks = masks_for(scene, stack, mean_cc, args)
    fit, held, ice = masks["fit rock"], masks["held rock"], masks["ice"]
    print("pixels: " + ", ".join(f"{k} {v.sum():,}" for k, v in masks.items()))
    pair = (los_displacement(phase, stack.wavelength) * 1000.0).astype(np.float32)
    del phase

    t0 = time.time()
    d, times = integrate(pair, net, n)
    del pair
    times = np.asarray(times, float)
    hours = times * 24.0
    d = ladder(d, fit, r, mean_cc, args.sigma)
    gc.collect()
    print(f"ladder C+D on the fit half: {d.shape[0]} epochs, {time.time() - t0:.0f} s")

    modes = fit_modes(d, fit, weights=mean_cc, k=args.k)
    screens = mode_screens(modes, fit, sigma=tuple(cache_arg(args, "loading_sigma")),
                           weights=mean_cc)
    mode_swing = np.array([swing(modes.temporal[:, i], times)[:2]
                           for i in range(modes.k)])
    print(f"{modes.k} modes over {modes.n_pixels:,} fitted pixels, "
          f"{time.time() - t0:.0f} s")

    rows, labels, series = [], [], {}
    row, rs = score(d, times, hours, held, ice)
    rows.append(row); labels.append("ladder"); series["ladder"] = rs
    for k in (k for k in RANKS if k <= modes.k):
        corr = mode_correction(modes, screens[:k]).astype(np.float32)
        row, rs = score(d - corr, times, hours, held, ice)
        rows.append(row); labels.append(f"+ modes k={k}")
        series[f"interpolated k={k}"] = rs
        del corr
        gc.collect()
    proj, own = mode_projection(modes, d, held, coefficients=True)
    del proj
    for k in (k for k in SELF_RANKS if k <= modes.k):
        proj = np.nan_to_num(mode_projection(modes, d, held, k=k), copy=False)
        row, rs = score(d - proj.astype(np.float32), times, hours, held, ice)
        rows.append(row); labels.append(f"self-fit k={k}")
        series[f"self-fit k={k}"] = rs
        del proj
        gc.collect()
    print(f"scored {len(rows)} variants, {time.time() - t0:.0f} s")

    return {"table": np.array(rows), "labels": np.array(labels),
            "columns": np.array(COLUMNS),
            "temporal": modes.temporal.astype(np.float32),
            "singular": modes.singular, "null": modes.null,
            "explained": modes.explained, "n_pixels": modes.n_pixels,
            "mode_swing": mode_swing[:, 0], "mode_r2": mode_swing[:, 1],
            "loading": modes.loading.astype(np.float32),
            "interpolated": screens.astype(np.float32),
            "own": own.astype(np.float32),
            "held": held, "fit": fit, "ice": ice,
            **{f"series {k}": v for k, v in series.items()},
            "epoch_hours": hours,
            "epoch0": np.datetime64(net.epochs[0]).astype("datetime64[s]"),
            "cache_version": MODES_CACHE_VERSION, "antenna": args.antenna,
            "decimate": args.decimate, "utc_offset": args.utc_offset,
            **{k: np.asarray(cache_arg(args, k), float) for k in CACHE_ARGS}}


def diurnal_mode(c):
    """Index of the kept mode whose series has the largest 24 h swing."""
    return int(np.nanargmax(c["mode_swing"]))


def transfer(c, i):
    """Own against interpolated loading of mode ``i`` on the held-out pixels."""
    held = c["held"]
    own, interp = c["own"][i][held], c["interpolated"][i][held]
    ok = np.isfinite(own) & np.isfinite(interp)
    r = float(np.corrcoef(own[ok], interp[ok])[0, 1]) if ok.sum() > 2 else np.nan
    return own[ok], interp[ok], r


def report(c, name):
    print(f"\n{name}: {c['n_pixels']:,} fitted bedrock pixels, "
          f"{c['held'].sum():,} held out, {c['ice'].sum():,} ice")
    print("  mode   singular      null   ratio  explained   24 h swing of series")
    for i in range(len(c["singular"])):
        print(f"  {i + 1:4d}  {c['singular'][i]:9.2f} {c['null'][i]:9.2f} "
              f"{c['singular'][i] / c['null'][i]:7.2f}   {100 * c['explained'][i]:6.1f} %"
              f"   {c['mode_swing'][i]:6.2f} mm (r2 {c['mode_r2'][i]:.2f})")
    i = diurnal_mode(c)
    _, _, r = transfer(c, i)
    print(f"  mode {i + 1}: own against interpolated loading on the held-out "
          f"pixels, r = {r:.2f}")
    print("\n  " + " " * 16 + "  ".join(f"{k:>12s}" for k in c["columns"]))
    base = c["table"][0]
    for lab, row in zip(c["labels"], c["table"]):
        cells = []
        for j, v in enumerate(row):
            if lab == "ladder":
                cells.append(f"{v:12.3f}")
            elif j == len(row) - 1:
                cells.append(f"{v - base[j]:+12.3f}")
            else:
                cells.append(f"{100 * (v / base[j] - 1):+11.1f}%")
        print(f"  {lab:16s}" + "  ".join(cells))
    print("  held sd, held px sd in mm; 24 h swings peak to peak in mm; "
          "px 2 h in mm/hr; ice rate in m/yr (change in m/yr)")


def figure(c, name, args):
    t = utc_epochs(c)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), dpi=110)

    ax = axes[0, 0]
    idx = np.arange(1, len(c["singular"]) + 1)
    ax.semilogy(idx, c["singular"], "o-", color="k", ms=4, lw=0.9, label="bedrock")
    ax.semilogy(idx, c["null"], "s--", color="0.6", ms=3, lw=0.8, label="shuffled")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Singular value")
    ax.set_xticks(idx)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i, col in zip(range(min(3, c["temporal"].shape[1])),
                      ("k", "tab:blue", "tab:orange")):
        ax.plot(t, c["temporal"][:, i], color=col, lw=0.8, label=f"mode {i + 1}")
    ax.set_ylabel("Mode series (mm)")
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=3)

    ax = axes[1, 0]
    i = diurnal_mode(c)
    own, interp, _ = transfer(c, i)
    lim = float(np.nanpercentile(np.abs(own), 99)) or 1.0
    ax.scatter(interp, own, s=2, color="k", alpha=0.25, lw=0)
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.7, ls=":")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Interpolated loading")
    ax.set_ylabel("Own loading")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    k = max(kk for kk in (1, 2, 3) if f"series interpolated k={kk}" in c)
    ax.plot(t, c["series ladder"], color="0.6", lw=0.8, label="ladder")
    ax.plot(t, c[f"series interpolated k={k}"], color="k", lw=0.8,
            label=f"+ modes (k={k})")
    ks = max(kk for kk in SELF_RANKS if f"series self-fit k={kk}" in c and kk <= k)
    ax.plot(t, c[f"series self-fit k={ks}"], color="tab:red", lw=0.8,
            label=f"self-fit (k={ks})")
    ax.set_ylabel("Held-out rock (mm)")
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=3)

    for ax in (axes[0, 1], axes[1, 1]):
        shade_local_nights(ax, t[0], t[-1], c["utc_offset"])
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time (UTC)")
        ax.set_xlim(t[0], t[-1])
        loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    fig.tight_layout()
    out = args.outdir / f"30_modes_{name}{run_tag(args)}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803_full",
                    help="campaign key from site.env, or a directory")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--k", type=int, default=10, help="modes to keep")
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0),
                    help="Gaussian sigma (azimuth, range) of the ladder's "
                         "turbulence screen")
    ap.add_argument("--loading-sigma", type=float, nargs=2, default=None,
                    help="the same for the loading interpolation; the ladder's "
                         "--sigma unless given, and a run with its own value "
                         "is cached and drawn beside the default one")
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--stable-coherence", type=float, default=0.6)
    ap.add_argument("--utc-offset", type=float, default=-7.0)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    name = args.scene
    scene = Path(SCENES.get(name, name))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")

    c, why = (None, "recompute") if args.recompute else load_modes(scene, args)
    if c is None:
        cache = modes_path(scene, args.antenna, args.decimate, run_tag(args))
        print(f"{cache}: {why}; computing")
        c = compute(scene, name, args)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **c)
        print(f"cached {cache}")

    report(c, name)
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure(c, name, args)


if __name__ == "__main__":
    main()
