#!/usr/bin/env python3
"""The two antennas as one interferometer: single-pass phase, and what it sees.

    python examples/baker_antenna_interferogram.py --scene 20170803_full

`baker_antennas.py` uses the GPRI-II's second antenna as a replicate — the
same scene measured twice, so the difference is noise.  This uses the two as
an **interferometer**.  Upper and lower are 25 cm apart on one mast and
sampled in the same sweep, so ``s_upper * conj(s_lower)`` at one epoch has

* no temporal decorrelation, because there is no time between them,
* no deformation, because nothing moved between them,
* no atmospheric delay beyond the difference over 25 cm,

which leaves the geometry: the path-length difference to each target, set by
the target's elevation angle.  That is a topographic measurement made by the
radar itself, and this script compares it against the external DEM that
`gpri_tools.heading.target_heights` reads — the DEM the height screen and
the geocoding already depend on.

The comparison is a fit, not an assertion.  The phase a vertical baseline
``B`` puts on a target at elevation angle ``theta`` is ``k * 2 pi B
sin(theta) / lambda`` with ``k = 1`` for a one-way path difference (transmit
on one antenna, receive on both) and ``k = 2`` for two-way.  Rather than
assume which this instrument is, the script regresses the measured phase on
``sin(theta)`` from the DEM and reports the slope it finds, the implied
``k``, and the scatter about the line.  A slope near one of those two
values, with the measured and predicted phase patterns agreeing across the
scene, says the interferogram is reading topography; a slope near zero says
it is reading nothing.

Phase is compared where it is unambiguous, so the script works on the
**wrapped difference** between measured and predicted and reports its
circular statistics.  The height ambiguity — the height change that turns
the phase by ``2 pi`` — is printed per range bin, because it grows with
range and says how coarse this measurement is.
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
from baker_aps import SCENES, open_stack                                 # noqa: E402
from baker_movie import decimated_geom                                   # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.heading import scene_heading, target_heights             # noqa: E402

ANTENNA_SEPARATION = 0.25            # metres, upper to lower on the mast


def multilook(z, looks):
    la, lr = int(looks[0]), int(looks[1])
    if (la, lr) == (1, 1):
        return z
    na, nr = z.shape[0] // la * la, z.shape[1] // lr * lr
    return z[:na, :nr].reshape(na // la, la, nr // lr, lr).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803_full")
    ap.add_argument("--epochs", type=int, default=24,
                    help="epochs averaged; topography is static, so averaging "
                         "the complex interferogram beats the noise down")
    ap.add_argument("--stride", type=int, default=None,
                    help="epoch spacing; spread over the record by default")
    ap.add_argument("--looks", type=int, nargs=2, default=(3, 15),
                    help="spatial multilooking of the cross-antenna product")
    ap.add_argument("--baseline", type=float, default=ANTENNA_SEPARATION,
                    help="antenna separation, metres")
    ap.add_argument("--coherence", type=float, default=0.3,
                    help="cross-antenna coherence a pixel needs to be read")
    ap.add_argument("--shift-search", type=int, nargs=2, default=(12, 4),
                    help="azimuth and range pixels either way the predicted "
                         "field is moved when testing the DEM's registration")
    ap.add_argument("--spacing", type=float, default=40.0)
    ap.add_argument("--dem", default=None,
                    help="DEM for the predicted phase; GPRI_DEM by default")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")
    dem = args.dem or os.environ.get("GPRI_DEM", "")
    if not dem or not Path(dem).exists():
        sys.exit("this needs a DEM to compare against: set GPRI_DEM or --dem")

    upper = open_stack(scene, "upper", lags=(1,), looks=(1, 1))
    lower = open_stack(scene, "lower", lags=(1,), looks=(1, 1))
    n_epochs = min(upper.network.n_epochs, lower.network.n_epochs)
    stride = args.stride or max(1, n_epochs // max(args.epochs, 1))
    epochs = list(range(0, n_epochs, stride))[:args.epochs]
    print(f"{scene.name}: {n_epochs} epochs on each antenna; averaging "
          f"{len(epochs)} of them (stride {stride}) at {args.looks[0]} x "
          f"{args.looks[1]} looks")

    t0 = time.time()
    acc, coh, offsets = None, None, []
    for e in epochs:
        su = upper.read_slc(e)
        sl = lower.read_slc(e)
        na = min(su.shape[0], sl.shape[0])
        nr = min(su.shape[1], sl.shape[1])
        a, b = su[:na, :nr], sl[:na, :nr]
        # the multilooked interferogram and its coherence, per epoch
        num = multilook(a * np.conj(b), args.looks)
        den = np.sqrt(multilook(np.abs(a) ** 2, args.looks)
                      * multilook(np.abs(b) ** 2, args.looks))
        with np.errstate(invalid="ignore", divide="ignore"):
            g_e = np.where(den > 0, np.abs(num) / den, 0.0)
        # the two receive chains carry an arbitrary relative phase that is
        # constant across the scene and changes sweep to sweep; it is not
        # geometry, and averaging epochs without removing it cancels the
        # geometry instead
        z_e = np.where(np.abs(num) > 0, num / np.maximum(np.abs(num), 1e-30), 0)
        bright = g_e >= np.nanpercentile(g_e, 90)
        const = np.angle(np.mean(z_e[bright])) if bright.any() else 0.0
        offsets.append(const)
        z_e = z_e * np.exp(-1j * const)
        acc = z_e * g_e if acc is None else acc + z_e * g_e
        coh = g_e if coh is None else coh + g_e
    upper.close(); lower.close()
    with np.errstate(invalid="ignore", divide="ignore"):
        ifg = np.where(coh > 0, acc / np.maximum(coh, 1e-30), np.nan)
    gamma = coh / len(epochs)                # mean per-epoch coherence
    consistency = np.abs(acc) / np.maximum(coh, 1e-30)
    phase = np.angle(ifg)
    off = np.unwrap(np.array(offsets))
    print(f"cross-antenna interferogram in {time.time() - t0:.0f} s; "
          f"grid {ifg.shape[1]} x {ifg.shape[2] if ifg.ndim > 2 else ifg.shape[1]}"
          if False else
          f"cross-antenna interferogram in {time.time() - t0:.0f} s; "
          f"grid {ifg.shape[0]} x {ifg.shape[1]}")
    print(f"  per-epoch coherence, median {np.nanmedian(gamma):.3f}; "
          f"the scene-constant chain phase spans {np.ptp(off):.2f} rad over "
          f"the {len(epochs)} epochs and is removed")
    print(f"  after removing it the phase agrees across epochs at "
          f"{np.nanmedian(consistency):.3f} (1 is perfect)")

    # ---- what the DEM predicts --------------------------------------------
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    geom = decimated_geom(upper, 1, heading)
    la, lr = args.looks
    rows = ((np.arange(ifg.shape[0]) + 0.5) * la).astype(int)
    cols = ((np.arange(ifg.shape[1]) + 0.5) * lr).astype(int)
    h = target_heights(geom, dem, rows=rows, cols=cols)
    slant = np.asarray(geom.slant_range(), float)[cols]
    R = np.broadcast_to(slant, h.shape)
    # elevation angle of each target as the radar sees it
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_theta = np.clip((h - geom.alt0) / np.maximum(R, 1.0), -1.0, 1.0)
    wavelength = upper.wavelength
    unit = 2 * np.pi * args.baseline / wavelength          # k = 1, radians
    predicted = unit * sin_theta

    ok = (np.isfinite(sin_theta) & np.isfinite(phase)
          & (gamma >= args.coherence) & (consistency >= args.coherence))
    print(f"{ok.sum():,} pixels have DEM height and cross-antenna consistency "
          f">= {args.coherence:g} ({100 * ok.mean():.1f} % of the grid)")
    if ok.sum() < 100:
        sys.exit("too few pixels to compare; lower --coherence")

    # The phase is wrapped, so the slope is read by asking which multiple of
    # the DEM's sin(theta) leaves the flattest residual.  That scan is
    # ALIASED: with unit = 2 pi B / lambda of order ninety radians, slopes
    # differing by about 2 pi / (unit * spread of sin theta) fit nearly as
    # well, and which alias wins is decided by noise.  So the candidates the
    # geometry allows are scored explicitly and the scan is reported beside
    # them rather than trusted.
    z = np.exp(1j * phase[ok])
    s = sin_theta[ok]

    def resultant_at(k):
        return float(np.abs(np.mean(z * np.exp(-1j * k * unit * s))))

    candidates = {"one-way (k = 1)": 1.0, "two-way (k = 2)": 2.0,
                  "half (k = 0.5)": 0.5}
    scored = {name: resultant_at(k) for name, k in candidates.items()}
    best_name = max(scored, key=scored.get)
    k_hat = candidates[best_name]
    # a local refinement, narrow enough not to cross into the next alias
    width = 2 * np.pi / (unit * max(float(np.std(s)), 1e-6))
    local = np.linspace(k_hat - 0.3 * width, k_hat + 0.3 * width, 201)
    local_res = np.array([resultant_at(k) for k in local])
    k_ref = float(local[int(np.argmax(local_res))])
    scales = np.linspace(-4.0, 4.0, 3201)
    resultant = np.array([resultant_at(k) for k in scales])
    k_scan = float(scales[int(np.argmax(resultant))])
    R_hat = float(local_res.max())
    k_hat = k_ref
    offset = float(np.angle(np.mean(z * np.exp(-1j * k_hat * unit * s))))
    resid = np.angle(np.exp(1j * (phase - k_hat * predicted - offset)))
    circ_sd = float(np.sqrt(-2 * np.log(max(R_hat, 1e-12))))
    print(f"\nregressing the wrapped phase on the DEM's sin(theta):")
    for name, k in candidates.items():
        print(f"  {name:16s} resultant {scored[name]:.3f}"
              + ("   <- taken" if candidates[name] == candidates[best_name] else ""))
    print(f"  refined near it: slope {k_hat:+.4f} x (2 pi B / lambda), "
          f"offset {offset:+.2f} rad, resultant {R_hat:.3f}, "
          f"circular sd {circ_sd:.2f} rad")
    print(f"  an unconstrained scan peaks at {k_scan:+.3f} with "
          f"{resultant.max():.3f}; the scan is aliased at intervals of about "
          f"{width:.2f} in k, so it is reported, not used")

    # A poor fit at k = 1 is as likely to be the DEM sitting in the wrong
    # place as the geometry being wrong: a heading error of a degree puts the
    # predicted phase tens of pixels away in azimuth, and at these height
    # ambiguities that destroys the match.  Scan the registration and say
    # where the agreement actually peaks.
    best = (R_hat, 0, 0)
    for da in range(-int(args.shift_search[0]), int(args.shift_search[0]) + 1):
        for dr in range(-int(args.shift_search[1]), int(args.shift_search[1]) + 1):
            shifted = np.roll(np.roll(sin_theta, da, axis=0), dr, axis=1)
            m = ok & np.isfinite(shifted)
            if m.sum() < 100:
                continue
            val = float(np.abs(np.mean(np.exp(1j * phase[m])
                                       * np.exp(-1j * k_hat * unit * shifted[m]))))
            if val > best[0]:
                best = (val, da, dr)
    print(f"  registration scan: the agreement peaks at {best[0]:.3f} with the "
          f"DEM moved {best[1]:+d} azimuth and {best[2]:+d} range pixels of the "
          f"multilooked grid (0, 0 gives {R_hat:.3f})")
    flat = np.abs(np.mean(z))
    print(f"  the same statistic with no DEM term at all: {flat:.3f}")
    r_med = float(np.nanmedian(R[ok]))
    amb_med = wavelength * r_med / (max(abs(k_hat), 1e-6) * args.baseline)
    print(f"  at the median range of {r_med / 1000:.1f} km a cycle is "
          f"{amb_med:.0f} m of height, so the circular sd is "
          f"{circ_sd / (2 * np.pi) * amb_med:.0f} m of height against the DEM")

    print(f"\n{'range (km)':>11} {'pixels':>8} {'height ambiguity (m)':>21} "
          f"{'|mean phasor|':>14}")
    edges = np.arange(np.floor(slant.min() / 1000), np.ceil(slant.max() / 1000) + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = ok & (R >= lo * 1000) & (R < hi * 1000)
        if band.sum() < 50:
            continue
        r_mid = float(np.nanmedian(R[band]))
        # d(phase)/d(height) = k * 2 pi B / (lambda R); a cycle costs
        amb = wavelength * r_mid / (max(abs(k_hat), 1e-6) * args.baseline)
        val = float(np.abs(np.mean(np.exp(1j * resid[band]))))
        print(f"{lo:5.0f}-{hi:<5.0f} {band.sum():8,d} {amb:21.0f} {val:14.3f}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=140)
    shown = np.where(ok, phase, np.nan)
    im0 = axes[0].imshow(shown, origin="lower", cmap="twilight", aspect="auto",
                         vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], label="Measured phase (rad)", fraction=0.046)
    pred_w = np.angle(np.exp(1j * (k_hat * predicted + offset)))
    im1 = axes[1].imshow(np.where(ok, pred_w, np.nan), origin="lower",
                         cmap="twilight", aspect="auto", vmin=-np.pi, vmax=np.pi,
                         interpolation="nearest")
    fig.colorbar(im1, ax=axes[1], label="DEM phase (rad)", fraction=0.046)
    for ax in axes[:2]:
        ax.set_xlabel("Range (px)")
        ax.set_ylabel("Azimuth (px)")
    sel = np.flatnonzero(ok.ravel())
    take = sel[:: max(1, sel.size // 40000)]
    axes[2].plot(sin_theta.ravel()[take], phase.ravel()[take], ".", ms=1.0,
                 alpha=0.2, color="0.4")
    xs = np.linspace(np.nanpercentile(s, 0.5), np.nanpercentile(s, 99.5), 200)
    axes[2].plot(xs, np.angle(np.exp(1j * (k_hat * unit * xs + offset))), ".",
                 ms=2.0, color="tab:red")
    axes[2].set_xlabel("sin elevation")
    axes[2].set_ylabel("Measured phase (rad)")
    axes[2].set_ylim(-np.pi, np.pi)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"35_antenna_ifg_{scene.name}_lk{la}x{lr}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")

    root = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
    npz = root / f"antenna_ifg_lk{la}x{lr}.npz"
    np.savez(npz, phase=phase.astype(np.float32), gamma=gamma.astype(np.float32),
             predicted=predicted.astype(np.float32), residual=resid.astype(np.float32),
             sin_theta=sin_theta.astype(np.float32), k=k_hat, offset=offset,
             resultant=R_hat, k_scan=k_scan, scan_k=scales, scan_resultant=resultant,
             candidate_resultants=np.array([scored[n] for n in candidates]),
             candidate_k=np.array([candidates[n] for n in candidates]),
             registration=np.array(best[1:]), registration_resultant=best[0],
             baseline=args.baseline, wavelength=wavelength,
             epochs=np.array(epochs), looks=np.array(args.looks))
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
