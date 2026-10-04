#!/usr/bin/env python3
"""The 24 h harmonic per pixel: amplitude, hour of peak, and what they are worth.

    python examples/baker_harmonics.py --scene 20170827 --decimate 16 --rgi

`baker_population.py` reports the median of the ice against the median of the
held-out bedrock; this fits the same 24 h harmonic to **every pixel**
(:func:`gpri_tools.diurnal.fit_harmonics`) on the same corrected series —
reference, per-epoch drift, per-epoch turbulence screen, then the temporal
path delay of :mod:`gpri_tools.pathdelay` — and maps what comes out.

Three things are mapped, geocoded at ``--spacing`` metres: the amplitude in
mm, the hour of day the harmonic peaks, and the amplitude divided by its own
one-sigma (:meth:`~gpri_tools.diurnal.HarmonicFit.amplitude_sigma`, the
least-squares error propagated from each pixel's residual scatter).  The
third is the one that says whether the first two mean anything: a pixel whose
amplitude is not a few times its sigma has fitted its own noise, and the
phase of such a pixel is arbitrary.

Held-out bedrock is carried through every panel as the control.  It does not
move, so its amplitude distribution is the floor an ice amplitude has to
clear, and the printed table gives both distributions and the share of ice
pixels above the bedrock's 84th percentile.  The fourth panel is the
range test of :func:`gpri_tools.diurnal.range_dependence`: residual
refractivity puts an amplitude that grows with slant range, ice motion has no
reason to, so amplitude against range is drawn for both populations.

A record shorter than ``MIN_CYCLES`` of a day cannot carry a 24 h harmonic
and is refused rather than fitted.
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
from gpri_tools.diurnal import (DIURNAL, MIN_CYCLES,                     # noqa: E402
                                effective_sample_factor, fit_harmonics,
                                m_per_yr, range_dependence)
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.glaciers import (glacier_mask, load_outlines,            # noqa: E402
                                 stable_ground_mask)
from gpri_tools.heading import scene_heading                             # noqa: E402
from gpri_tools.pathdelay import (displacement_delay_field,              # noqa: E402
                                  pair_variance_from_coherence)
from gpri_tools.timeseries import los_displacement                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170827")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--rgi", action="store_true",
                    help="tie the bedrock mask to the glacier outlines")
    ap.add_argument("--no-path-delay", dest="path_delay", action="store_false",
                    help="leave the temporal path delay in: the ladder alone")
    ap.add_argument("--protect-period", type=float, default=1.0)
    ap.add_argument("--max-response", type=float, default=0.01)
    ap.add_argument("--spacing", type=float, default=40.0,
                    help="map pixel, metres")
    ap.add_argument("--snr", type=float, default=3.0,
                    help="amplitude over sigma a pixel needs before its phase "
                         "is drawn")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    stack, net, phase, cc, r, az, n = load(scene, args.decimate, 0,
                                           antenna=args.antenna)
    mean_cc = cc.mean(axis=0)
    geom = decimated_geom(stack, args.decimate, heading)
    usable = mean_cc >= args.ice_coherence
    stable = mean_cc >= args.stable_coherence
    if args.rgi:
        la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1],
                               cols=[0, geom.shape[1] - 1])
        gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                            bbox=(lo.min() - .02, la.min() - .02,
                                  lo.max() + .02, la.max() + .02))
        stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                       threshold=args.stable_coherence)
        ice = usable & glacier_mask(geom, gdf)
    else:
        ice = usable & ~stable
    fit_m, held_m = split_mask(stable)
    pair_var = (pair_variance_from_coherence(cc[:n], ice | fit_m)
                if args.path_delay else None)
    del cc

    times = np.asarray(net.times, float)
    span = float(times[-1] - times[0])
    day = scene.name + ("" if args.antenna == "upper" else f"_{args.antenna}")
    print(f"{day}: {n} pairs over {span * 24:.1f} h ({span:.2f} cycles); "
          f"ice {ice.sum():,} px, bedrock {fit_m.sum():,} fit + "
          f"{held_m.sum():,} held out")
    if span < DIURNAL * MIN_CYCLES:
        sys.exit(f"a 24 h harmonic needs {MIN_CYCLES:g} of a cycle; this record "
                 f"has {span:.2f}")

    d, times = integrate(los_displacement(phase, stack.wavelength), net, n)
    del phase
    d, _ = epoch_screen_correction(d, fit_m, r, model="linear", weights=mean_cc)
    t0 = time.time()
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], fit_m, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        d[k] -= scr
    print(f"corrections in {time.time() - t0:.0f} s")
    if args.path_delay:
        t0 = time.time()
        trusted = ice | fit_m
        field, plam = displacement_delay_field(
            d, np.asarray(net.pairs[:n], int), np.asarray(times, float),
            trusted, weights=mean_cc, sigma=tuple(args.sigma),
            protect_period=args.protect_period, max_response=args.max_response,
            pair_variance=pair_var)
        d -= field.astype(d.dtype)
        print(f"path delay (lambda {plam:.4g}) removed in {time.time() - t0:.0f} s; "
              f"field sd {1000 * np.nanstd(field):.3f} mm")
        del field

    t = np.asarray(times, float)
    t0 = time.time()
    series = d * 1000.0                                     # mm
    fit = fit_harmonics(series, t)
    amp = fit.amplitude()
    # a white-noise error bar counts every epoch as independent, which a
    # random-walk series is not: correct it by the autocorrelation of each
    # pixel's own residual before anything is called significant
    resid = series - fit.evaluate()
    inflation = effective_sample_factor(resid)
    sig_white = fit.amplitude_sigma()
    sig = fit.amplitude_sigma(inflation=inflation)
    del resid, series
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where(sig > 0, amp / sig, np.nan)
    origin = (net.epochs[0].hour + net.epochs[0].minute / 60.0
              + net.epochs[0].second / 3600.0)
    peak = fit.peak_time(origin_hour=origin)
    rate = m_per_yr(fit.secular, "mm")
    print(f"per-pixel harmonic fit in {time.time() - t0:.0f} s; the residual's "
          f"autocorrelation inflates the error bar by a median "
          f"{np.nanmedian(inflation[ice]):.1f}x on ice and "
          f"{np.nanmedian(inflation[held_m]):.1f}x on held-out bedrock "
          f"(white-noise sigma would be {np.nanmedian(sig_white[ice]):.2f} mm "
          f"on ice, corrected {np.nanmedian(sig[ice]):.2f} mm)")

    # ---- what the numbers say ---------------------------------------------
    print(f"\n{'population':18s} {'pixels':>8s} {'amplitude (mm)':>22s} "
          f"{'sigma':>7s} {'A/sigma':>9s} {'peak UTC':>10s} {'rate (m/yr)':>12s}")
    rows = {}
    for label, m in (("RGI ice", ice), ("held-out bedrock", held_m),
                     ("fit-half bedrock", fit_m)):
        a, s_, sn = amp[m], sig[m], snr[m]
        p16, p50, p84 = np.nanpercentile(a, [16, 50, 84])
        strong = np.isfinite(sn) & (sn >= args.snr)
        ph = peak[m][strong]
        circ = (np.angle(np.mean(np.exp(2j * np.pi * ph / 24.0))) / (2 * np.pi) * 24
                if ph.size else np.nan)
        rows[label] = (p50, np.nanmedian(s_), np.nanmedian(sn), np.mod(circ, 24),
                       float(np.nanmedian(rate[m])), strong.mean())
        print(f"{label:18s} {m.sum():8,d} {p50:8.2f} (p16-p84 {p16:5.2f}-{p84:5.2f}) "
              f"{np.nanmedian(s_):7.2f} {np.nanmedian(sn):9.2f} "
              f"{np.mod(circ, 24):9.1f} h {np.nanmedian(rate[m]):12.2f}")
    floor = np.nanpercentile(amp[held_m], 84)
    above = np.nanmean(amp[ice] > floor)
    print(f"\nthe held-out bedrock's 84th percentile amplitude is {floor:.2f} mm; "
          f"{100 * above:.1f} % of ice pixels are above it")
    for label, m in (("RGI ice", ice), ("held-out bedrock", held_m)):
        print(f"{label:18s} {100 * rows[label][5]:5.1f} % of pixels reach "
              f"A/sigma >= {args.snr:g}")
    rd = range_dependence(amp, r, mask=ice)
    rd_rock = range_dependence(amp, r, mask=held_m)
    print(f"\namplitude against slant range (the refractivity test): "
          f"ice r = {rd['correlation']:+.2f}, {rd['slope'] * 1000:+.2f} mm/km; "
          f"held-out bedrock r = {rd_rock['correlation']:+.2f}, "
          f"{rd_rock['slope'] * 1000:+.2f} mm/km")
    print(f"  {rd['verdict']}")

    # ---- the figure --------------------------------------------------------
    res = Resampler(geom, args.spacing)
    show = ice | stable
    strong = np.isfinite(snr) & (snr >= args.snr)

    def mapped(x, where):
        return res(np.where(where, x, np.nan))

    fig = plt.figure(figsize=(13.5, 8.0), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_a, ax_p = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_s, ax_r = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    alim = float(np.nanpercentile(amp[show], 98))
    im_a = ax_a.imshow(mapped(amp, show), origin="lower", cmap="viridis",
                       vmin=0, vmax=alim, interpolation="nearest")
    fig.colorbar(im_a, ax=ax_a, label="Amplitude (mm)", fraction=0.046)
    im_p = ax_p.imshow(mapped(peak, show & strong), origin="lower",
                       cmap="twilight", vmin=0, vmax=24, interpolation="nearest")
    fig.colorbar(im_p, ax=ax_p, label="Peak (hr UTC)", fraction=0.046)
    # a long record makes the formal error small, so the scale comes from the
    # data rather than from the threshold
    slim = float(np.nanpercentile(snr[show], 98))
    im_s = ax_s.imshow(mapped(snr, show), origin="lower", cmap="magma",
                       vmin=0, vmax=max(slim, 3 * args.snr),
                       interpolation="nearest")
    fig.colorbar(im_s, ax=ax_s, label="Amplitude / sigma", fraction=0.046)
    seen = np.isfinite(res(np.where(show, 0.0, np.nan)))
    if seen.any():
        rr, cc_ = np.where(seen)
        for ax in (ax_a, ax_p, ax_s):
            ax.set_xlim(max(cc_.min() - 4, 0), min(cc_.max() + 4, seen.shape[1] - 1))
            ax.set_ylim(max(rr.min() - 4, 0), min(rr.max() + 4, seen.shape[0] - 1))
            ax.set_xticks([]); ax.set_yticks([])

    km = np.broadcast_to(np.asarray(r, float) / 1000.0, amp.shape)
    for m, colour, label in ((ice, "tab:blue", "ice"),
                             (held_m, "tab:red", "held-out rock")):
        sel = m & np.isfinite(amp)
        ax_r.plot(km[sel][::7], amp[sel][::7], ".", ms=1.2, alpha=0.25,
                  color=colour, label=label)
        edges = np.arange(np.floor(km[sel].min()), np.ceil(km[sel].max()) + 0.5, 0.5)
        idx = np.digitize(km[sel], edges) - 1
        med = np.array([np.nanmedian(amp[sel][idx == i]) if (idx == i).sum() > 20
                        else np.nan for i in range(edges.size - 1)])
        ax_r.plot(edges[:-1] + 0.25, med, "-", lw=2.0, color=colour)
    ax_r.set_xlabel("Slant range (km)")
    ax_r.set_ylabel("Amplitude (mm)")
    ax_r.set_ylim(0, alim)
    ax_r.grid(alpha=0.3)
    ax_r.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.path_delay else "_ladder"
    out = args.outdir / f"32_harmonics_{day}{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")

    root = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
    root.mkdir(parents=True, exist_ok=True)
    npz = root / f"harmonics_{args.antenna[0].lower()}_dec{args.decimate}{tag}.npz"
    np.savez(npz, amplitude=amp.astype(np.float32), sigma=sig.astype(np.float32),
             sigma_white=sig_white.astype(np.float32),
             inflation=inflation.astype(np.float32),
             peak=peak.astype(np.float32), rate=rate.astype(np.float32),
             ice=ice, held=held_m, fit=fit_m, slant_range=r.astype(np.float32),
             origin=origin, span_hours=span * 24.0)
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
