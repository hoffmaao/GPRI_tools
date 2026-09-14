#!/usr/bin/env python3
"""Atmospheric path delay from double differences, and what it is worth here.

    python examples/baker_pathdelay.py --scene 20170913

The correction ladder in `baker_aps.py` separates air from ice **spatially**:
a screen fitted on bedrock, taken off everywhere.  This script does it
**temporally** instead, the way the terrestrial-radar literature does it
(`gpri_tools.pathdelay`): two interferograms that share their middle epoch,
each divided by its own span, differ by exactly nothing if the surface moved
at a steady rate, so whatever is left is the air.  Stacking every such
triplet gives an overdetermined system for one delay per acquisition, solved
with Tikhonov regularisation.  No bedrock is needed to *estimate* it.

Bedrock is still needed to *score* it, and this script keeps the ladder's
rule: the stable mask is split in half, one half feeds the estimate and the
other is never touched by it and does the scoring.  Three estimators are run
and reported side by side, because the honest number and the flattering one
differ by a factor of fifteen:

    scene     one delay series for the whole frame, from the spatial mean of
              the fit-half bedrock.  Scored on the held-out half, which the
              estimate never saw.
    pixel     one delay series per pixel.  This has one unknown per epoch and
              one observation per pair, so it fits the pixel's own noise; the
              scatter it removes from a pixel is largely that pixel's own.
              It is here as the control that says so.
    smooth    the per-pixel field, Gaussian-smoothed (`--sigma`) so only the
              spatially coherent part survives — the part a real atmosphere
              would have.

The delay is pinned so that it contributes nothing to the mean apparent
velocity (`gpri_tools.pathdelay.pin_rate`).  Without that, the affine part
the system cannot see leaks into the rate: on bedrock, which does not move,
the plain pinning shifts the apparent velocity by 1.0 to 1.8 m/yr.  The
trend so discarded is reported per campaign — it is the size of what this
method cannot observe, in the units it would be mistaken for.

What the correction can reach is set by the operator, not by the data.  A
component of period T enters the double differences weighted by
4 sin^2(pi dt / T) / dt, which vanishes as T grows; with the regularisation
chosen here the gain is ~1 at an hour and 0.002 to 0.010 at a day, measured
on the operator actually inverted, and the figure's response panel says so
per campaign.  A diurnal signal, of either origin, passes through this
correction untouched.

Outputs `figures/28_pathdelay_<scene>.png` and caches the series, the maps
and the scoring table in `work/<scene>/pathdelay_u_dec16.npz`.
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
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, load, split_mask                       # noqa: E402
from baker_brightness import shade_local_nights, utc_epochs           # noqa: E402
from baker_north_side import decimated_par                           # noqa: E402

from gpri_tools.closure import closure_rms, correct_bias, estimate_bias   # noqa: E402
from gpri_tools.diurnal import m_per_yr                                   # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry           # noqa: E402
from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask  # noqa: E402
from gpri_tools.heading import scene_heading                               # noqa: E402
from gpri_tools.pathdelay import (discarded_rate, double_difference,       # noqa: E402
                                  double_difference_row_weights,
                                  invert_path_delay, lambda_for_system_response,
                                  pair_variance_from_coherence, pin_rate,
                                  rewrap_to_chain, select_lambda,
                                  system_response)
from gpri_tools.refractivity import specific_humidity                      # noqa: E402
from gpri_tools.timeseries import los_displacement                         # noqa: E402

# version 5: the rows are weighted by the pairs' coherence, so a cached run
# from before the weighting answers a different question
PATHDELAY_CACHE_VERSION = 5

#: flags that change what the cached numbers answer
CACHE_ARGS = ("ice_coherence", "stable_coherence", "sigma", "lags", "looks",
              "protect_period", "max_response", "debias", "rewrap")

ESTIMATORS = ("scene", "pixel", "smooth")
MASKS = ("fit rock", "held rock", "ice")


def pathdelay_path(scene: Path, antenna: str, dec: int, looks=(1, 1),
                   debias=False) -> Path:
    """Where one run is cached.  Multilooked and de-biased runs sit beside the
    single-look one, because they are answers to a different question."""
    root = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
    la, lr = (int(looks[0]), int(looks[1]))
    tag = "" if (la, lr) == (1, 1) else f"_lk{la}x{lr}"
    tag += "_db" if debias else ""
    return root / scene.name / f"pathdelay_{antenna[0].lower()}_dec{dec}{tag}.npz"


def load_pathdelay(scene: Path, args):
    """The cached run, or ``(None, reason)`` when it answers another question."""
    cache = pathdelay_path(scene, args.antenna, args.decimate,
                           args.looks, args.debias)
    if not cache.exists():
        return None, "no cache"
    c = dict(np.load(cache, allow_pickle=False))
    if int(c.get("cache_version", 0)) < PATHDELAY_CACHE_VERSION:
        return None, f"older than cache version {PATHDELAY_CACHE_VERSION}"
    for k in CACHE_ARGS:
        want = np.atleast_1d(np.asarray(getattr(args, k), float))
        have = np.atleast_1d(np.asarray(c.get(k, np.nan), float))
        if have.shape != want.shape or not np.array_equal(have, want):
            return None, f"built with a different --{k.replace('_', '-')}"
    return c, ""


def masks_for(scene, stack, mean_cc, args):
    """Fit-half bedrock, held-out bedrock, and the coherent ice."""
    geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                         heading=scene_heading(scene, default=BAKERBEND1_HEADING))
    la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1], cols=[0, geom.shape[1] - 1])
    bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"), bbox=bbox)
    stable, _ = stable_ground_mask(mean_cc, geom, gdf, threshold=args.stable_coherence)
    fit, held = split_mask(stable)
    ice = (mean_cc >= args.ice_coherence) & glacier_mask(geom, gdf)
    return {"fit rock": fit, "held rock": held, "ice": ice}


def per_pixel_delay(d, system, lam, times, pairs, chunk_rows=24, row_w=None):
    """Invert every pixel with one factorisation, a block of rows at a time.

    ``row_w`` weights the double differences, each by the worse of the two
    pairs it is built from (:func:`double_difference_row_weights`).
    """
    sw = None if row_w is None else np.sqrt(row_w)
    A = system.A if sw is None else system.A * sw[:, None]
    M = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T)
    out = np.empty((A.shape[1],) + d.shape[1:], np.float32)
    for s in range(0, d.shape[1], chunk_rows):
        e = min(s + chunk_rows, d.shape[1])
        b = system.apply(d[:, s:e].astype(np.float64))
        if sw is not None:
            b = b * sw.reshape((-1,) + (1,) * (b.ndim - 1))
        x = np.tensordot(M, b.reshape(b.shape[0], -1), axes=(1, 0))
        x = x.reshape((A.shape[1],) + b.shape[1:])
        out[:, s:e] = pin_rate(x, times, pairs).astype(np.float32)
    return out


def score(d, pair_delay, mask, dt, per_pixel):
    """Apparent LOS velocity before and after, over one mask.

    ``per_pixel`` scores each pixel's own series and averages the spreads;
    otherwise the mask is averaged first, which is what the scene estimator
    is fitted on.  Rates come back in m/yr, positive toward the radar.
    """
    if per_pixel:
        before = d[:, mask] / dt[:, None]
        after = (d[:, mask] - pair_delay[:, mask]) / dt[:, None]
        sd = (np.nanstd(before, axis=0).mean(), np.nanstd(after, axis=0).mean())
        mean = (np.nanmean(before), np.nanmean(after))
    else:
        before = np.array([np.nanmean(x[mask]) for x in d]) / dt
        after = np.array([np.nanmean(x[mask]) for x in d - pair_delay]) / dt
        sd = (float(np.nanstd(before)), float(np.nanstd(after)))
        mean = (float(np.nanmean(before)), float(np.nanmean(after)))
    return [m_per_yr(v, "mm") for v in (*sd, *mean)]


def humidity_at_epochs(name):
    """Specific humidity (g/kg) at each radar epoch, from the met cache."""
    metf = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "met" / f"met_{name}.npz"
    if not metf.exists():
        return None
    m = np.load(metf, allow_pickle=True)
    need = ("era5_temperature_2m_at_epochs", "era5_relative_humidity_2m_at_epochs",
            "era5_surface_pressure_at_epochs")
    if not all(k in m.files for k in need):
        return None
    t, rh, p = (np.asarray(m[k], float) for k in need)
    return specific_humidity(t, rh, p)


def compute(scene, name, args):
    stack, net, phase, cc, r, az, n = load(scene, args.decimate, 0,
                                           antenna=args.antenna,
                                           lags=tuple(int(l) for l in args.lags),
                                           looks=tuple(int(l) for l in args.looks))
    mean_cc = cc.mean(axis=0)
    masks = masks_for(scene, stack, mean_cc, args)
    print("pixels: " + ", ".join(f"{k} {v.sum():,}" for k, v in masks.items()))
    # each pair is worth what its coherence says, read over the pixels the
    # fit reads: held-out bedrock scores the answer and never feeds it
    pair_var = pair_variance_from_coherence(cc[:n], masks["fit rock"])
    del cc
    print(f"pair variance {pair_var.min():.3f}-{pair_var.max():.3f} "
          "(Cramer-Rao, from the coherence over the fit half)")

    if args.debias:
        # Multilooking makes each baseline a distinct estimate, and with it the
        # short-baseline bias of De Zan et al. -- which is not epoch-separable
        # and so cannot be written as a per-epoch delay.  It has to come off
        # before the double differences, or the inversion explains it with
        # delays it has invented.
        net.pairs = np.asarray(net.pairs[:n], int)
        t0 = time.time()
        before = float(np.nanmean(closure_rms(phase, net)))
        model = estimate_bias(phase, net, robust=2, wavelength=stack.wavelength)
        phase = correct_bias(phase, model)
        after = float(np.nanmean(closure_rms(phase, net)))
        print(f"closure bias: rms {before:.4f} -> {after:.4f} rad over "
              f"{model.n_triplets:,} triangles in {time.time() - t0:.0f} s; "
              f"bias {np.nanmin(model.bias) * 1e3:.2f} to "
              f"{np.nanmax(model.bias) * 1e3:.2f} mrad over "
              f"{len(model.centers)} baseline bins")

    d = (los_displacement(phase, stack.wavelength) * 1000.0).astype(np.float32)
    del phase
    pairs = np.asarray(net.pairs[:n], int)
    times = np.asarray(net.times, float)
    dt = times[pairs[:, 1]] - times[pairs[:, 0]]
    if args.rewrap:
        # A pair unwrapped on its own is known modulo half a wavelength; put
        # each longer baseline on the cycle nearest the chain it spans.
        d, moved = rewrap_to_chain(d, pairs, stack.wavelength / 2 * 1000.0)
        lag = pairs[:, 1] - pairs[:, 0]
        print("rewrapped onto the chain: " + (", ".join(
            f"lag {L} moved {100 * moved[lag == L].mean():.1f} % of samples"
            for L in np.unique(lag) if L > 1) or "nothing to move at lag 1"))
    system = double_difference(pairs, times)
    print(f"{system.n_rows:,} double differences over {len(times):,} epochs, "
          f"baselines {system.spans.min() * 1440:.1f}-{system.spans.max() * 1440:.1f} min")
    # the operator actually inverted is the weighted one, so the weight
    # choice, the response and the per-pixel solve all read A_w
    row_w = double_difference_row_weights(system.rows, pair_var, len(pairs))
    A_w = system.A * np.sqrt(row_w)[:, None]

    scene_series = np.array([np.nanmean(x[masks["fit rock"]]) for x in d])
    if args.lam is None:
        lam, _ = select_lambda(A_w, system.apply(scene_series) * np.sqrt(row_w),
                               method="gcv")
    else:
        lam = float(args.lam)
    if args.protect_period:
        floor = lambda_for_system_response(A_w, times, args.protect_period,
                                           args.max_response)
        if floor > lam:
            print(f"lambda {lam:.4g} from GCV raised to {floor:.4g} to hold "
                  f"{args.protect_period * 24:.0f} h at {args.max_response:.1%}")
        lam = max(lam, floor)
    labels = ("10 min", "1 h", "2 h", "12 h", "24 h")
    gains = system_response(A_w, times,
                            np.array([1 / 144, 1 / 24, 1 / 12, 0.5, 1.0]), lam)
    print(f"lambda {lam:.4g}; the system returns "
          + ", ".join(f"{g:.3f} at {lab}" for g, lab in zip(gains, labels)))

    cadence = float(np.median(np.diff(times)))
    resp_periods = np.logspace(np.log10(2 * cadence), np.log10(2.0), 200)
    response = np.asarray(system_response(A_w, times, resp_periods, lam), float)

    w_pair = 1.0 / pair_var
    loose = invert_path_delay(scene_series, pairs, times, lam=lam, weights=w_pair)
    trend = float(discarded_rate(loose.delay, times, pairs))
    delays = {"scene": invert_path_delay(scene_series, pairs, times, lam=lam,
                                         weights=w_pair, pin="rate").delay}
    print(f"trend the pinning discards: {m_per_yr(trend, 'mm'):+.2f} m/yr "
          "(unobservable: the affine part this system cannot see)")
    t0 = time.time()
    cube = per_pixel_delay(d, system, lam, times, pairs, row_w=row_w)
    print(f"per-pixel inversion in {time.time() - t0:.0f} s")
    smooth = np.empty_like(cube)
    t0 = time.time()
    for k in range(cube.shape[0]):
        smooth[k] = gaussian_filter(np.nan_to_num(cube[k]), sigma=tuple(args.sigma),
                                    mode="nearest")
    smooth = pin_rate(smooth, times, pairs).astype(np.float32)
    print(f"spatial smoothing in {time.time() - t0:.0f} s")

    table = np.full((len(ESTIMATORS), len(MASKS), 4), np.nan)
    for ei, est in enumerate(ESTIMATORS):
        if est == "scene":
            pd_pairs = (delays["scene"][pairs[:, 1]] - delays["scene"][pairs[:, 0]])
            pd_pairs = pd_pairs[:, None, None] * np.ones((1,) + d.shape[1:], np.float32)
        else:
            cube_e = cube if est == "pixel" else smooth
            pd_pairs = cube_e[pairs[:, 1]] - cube_e[pairs[:, 0]]
        for mi, mk in enumerate(MASKS):
            table[ei, mi] = score(d, pd_pairs, masks[mk], dt,
                                  per_pixel=(est != "scene"))
        del pd_pairs

    ice_series = np.array([np.nanmean(x[masks["ice"]]) for x in smooth])
    rock_series = np.array([np.nanmean(x[masks["held rock"]]) for x in smooth])
    picks = np.linspace(0, len(times) - 1, args.n_maps + 2)[1:-1].round().astype(int)

    before = np.array([np.nanmean(x[masks["held rock"]]) for x in d]) / dt
    after = before - (delays["scene"][pairs[:, 1]]
                      - delays["scene"][pairs[:, 0]]) / dt

    return {"delay_scene": delays["scene"].astype(np.float32),
            "delay_ice": ice_series.astype(np.float32),
            "delay_rock": rock_series.astype(np.float32),
            "delay_rms": np.nanstd(smooth, axis=0).astype(np.float32),
            "maps": smooth[picks].astype(np.float32), "map_epochs": picks,
            "table": table, "estimators": np.array(ESTIMATORS),
            "masks": np.array(MASKS),
            "v_before": m_per_yr(before, "mm").astype(np.float32),
            "v_after": m_per_yr(after, "mm").astype(np.float32),
            "pair_hours": (times[pairs[:, 0]] * 24.0).astype(np.float32),
            "epoch_hours": times * 24.0,
            "epoch0": np.datetime64(net.epochs[0]).astype("datetime64[s]"),
            "range_km": (r / 1000.0).astype(np.float32),
            "humidity": (lambda q: np.full(len(times), np.nan) if q is None
                         else q[:len(times)])(humidity_at_epochs(name)),
            "lam": float(lam), "cadence_days": cadence,
            "response_periods": resp_periods, "response": response,
            "discarded_rate": m_per_yr(trend, "mm"),
            "n_pixels": np.array([masks[k].sum() for k in MASKS]),
            "cache_version": PATHDELAY_CACHE_VERSION, "antenna": args.antenna,
            "decimate": args.decimate, "utc_offset": args.utc_offset,
            **{k: np.asarray(getattr(args, k), float) for k in CACHE_ARGS}}


def report(c, name):
    """The scoring table: what each estimator does to the apparent velocity."""
    print(f"\n{name}: apparent LOS velocity over each mask, m/yr")
    print("  estimator  mask         sd before   sd after   reduction    "
          "mean before  mean after")
    for ei, est in enumerate(c["estimators"]):
        for mi, mk in enumerate(c["masks"]):
            sb, sa, mb, ma = c["table"][ei, mi]
            print(f"  {str(est):10s} {str(mk):11s} {sb:10.1f} {sa:10.1f} "
                  f"{100 * (1 - sa / sb):9.1f} % {mb:12.2f} {ma:11.2f}")
    print(f"  the pinning discarded {float(c['discarded_rate']):+.2f} m/yr, which "
          "this system cannot observe")
    q = c["humidity"]
    if np.isfinite(q).any():
        ok = np.isfinite(q) & np.isfinite(c["delay_scene"])
        r = np.corrcoef(q[ok], c["delay_scene"][ok])[0, 1]
        print(f"  delay against specific humidity at the epochs: r {r:+.2f} "
              f"over {q[ok].min():.1f}-{q[ok].max():.1f} g/kg (ERA5 is hourly; "
              "the delay recovered here is faster than that)")


def figure(c, name, args):
    t = utc_epochs(c)
    tp = utc_epochs({"epoch0": c["epoch0"], "epoch_hours": c["pair_hours"]})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), dpi=110)

    ax = axes[0, 0]
    ax.plot(t, c["delay_scene"], color="k", lw=0.9, label="bedrock mean")
    ax.plot(t, c["delay_ice"], color="tab:blue", lw=0.8, label="ice mean")
    ax.set_ylabel("Path delay (mm)")
    q = c["humidity"]
    if np.isfinite(q).any():
        tw = ax.twinx()
        tw.plot(t, q, color="tab:green", lw=0.8, label="humidity")
        tw.set_ylabel("Specific humidity (g/kg)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = axes[0, 1]
    ax.plot(tp, c["v_before"], color="0.6", lw=0.7, label="uncorrected")
    ax.plot(tp, c["v_after"], color="k", lw=0.7, label="corrected")
    ax.set_ylabel("LOS velocity (m/yr)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    for ax in (axes[0, 0], axes[0, 1]):
        shade_local_nights(ax, t[0], t[-1], c["utc_offset"])
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time (UTC)")
        ax.set_xlim(t[0], t[-1])
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    ax = axes[1, 0]
    m = c["maps"][0]
    lim = float(np.nanpercentile(np.abs(m), 98)) or 1.0
    im = ax.imshow(m, extent=[c["range_km"][0], c["range_km"][-1], 0, m.shape[0]],
                   aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_xlabel("Slant range (km)")
    ax.set_ylabel("Azimuth (px)")
    fig.colorbar(im, ax=ax, label="Path delay (mm)")

    ax = axes[1, 1]
    ax.semilogx(c["response_periods"] * 24.0, c["response"], color="k", lw=1.0)
    for hours in (1.0, 24.0):
        ax.axvline(hours, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("Period (hr)")
    ax.set_ylabel("Response")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    la, lr = (int(args.looks[0]), int(args.looks[1]))
    tag = "" if (la, lr) == (1, 1) else f"_lk{la}x{lr}"
    tag += "_db" if args.debias else ""
    out = args.outdir / f"28_pathdelay_{name}{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170913",
                    help="campaign key from site.env, or a directory")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--looks", type=int, nargs=2, default=(1, 1),
                    help="azimuth and range looks. Multilooking is what makes "
                         "the longer baselines independent measurements; at "
                         "single look they close with the chain exactly and "
                         "add nothing. Drop --decimate when using it")
    ap.add_argument("--debias", action="store_true",
                    help="estimate and remove the closure-phase bias before "
                         "the double differences (gpri_tools.closure). Needs "
                         "multilooked pairs to have anything to remove")
    ap.add_argument("--lags", type=int, nargs="+", default=[1],
                    help="temporal baselines to form pairs over; more than one "
                         "carries less noise into the delay, at the cost of "
                         "reading the stack again")
    ap.add_argument("--rewrap", action="store_true",
                    help="move each longer baseline by whole cycles onto the "
                         "chain it spans (gpri_tools.pathdelay.rewrap_to_chain); "
                         "at single look the two are the same phase and only "
                         "the wrapping separates them")
    ap.add_argument("--lam", type=float, default=None,
                    help="Tikhonov weight; chosen by GCV when omitted")
    ap.add_argument("--protect-period", type=float, default=1.0,
                    help="period (days) the correction must leave alone; the "
                         "weight is raised until the system returns no more "
                         "than --max-response of it. 0 disables the floor")
    ap.add_argument("--max-response", type=float, default=0.01,
                    help="how much of --protect-period may come through")
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0),
                    help="Gaussian sigma (azimuth, range) of the smooth estimator")
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--stable-coherence", type=float, default=0.6)
    ap.add_argument("--n-maps", type=int, default=3)
    ap.add_argument("--utc-offset", type=float, default=-7.0)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    name = args.scene
    scene = Path(SCENES.get(name, name))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")

    c, why = (None, "recompute") if args.recompute else load_pathdelay(scene, args)
    if c is None:
        print(f"{pathdelay_path(scene, args.antenna, args.decimate, args.looks, args.debias)}"
              f": {why}; computing")
        c = compute(scene, name, args)
        cache = pathdelay_path(scene, args.antenna, args.decimate,
                               args.looks, args.debias)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **c)
        print(f"cached {cache}")

    report(c, name)
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure(c, name, args)


if __name__ == "__main__":
    main()
