#!/usr/bin/env python3
"""The joint path / ice-motion inversion on every Baker campaign.

    python examples/baker_joint.py                  # cells (cached), solve, figures
    python examples/baker_joint.py --stage figures  # redraw from the saved results

:mod:`gpri_tools.jointinv` does the work; this script only reads the Baker
data into it.  Per campaign it integrates both antennas' daisy chains, draws
the masks (RGI ice at coherence >= 0.5; coherent ground off the buffered
outlines, split into fit and held-out halves by whole 200 m cells; candidate
reference ground off the outlines at coherence >= 0.6, used at or above
``--cand-min-z``), samples heights from ``GPRI_DEM_LIDAR`` (the 2015 USGS
lidar; ``GPRI_DEM`` fills its gaps) and caches the cell series with
:meth:`gpri_tools.jointinv.Campaign.save` (rebuilt when the cell size or DEM
changes), adding the ladder run on each antenna and averaged on the same
held-out cells, for the scores.  Then one call,
:func:`gpri_tools.jointinv.invert`, solves every campaign; the results go to
``$GPRI_WORK_ROOT/joint/`` (:meth:`~gpri_tools.jointinv.Inversion.load` reads
them back).

Printed: the held-out rock scores of the ladder, the joint inversion without
candidates and with them, on the held-out half the stationarity calibration
did not use; how the stationarity decision does on ground of
known state; and the ice's 24 h terms.  Drawn: ``40_joint_<campaign>.png``
(24 h amplitude and hour of peak on the cells) and
``41_joint_bands_<campaign>.png`` (ice motion by elevation band), into
``docs/figures`` at the default settings; any other setting draws them beside
its own results, in ``$GPRI_WORK_ROOT/joint/<setting>/figures``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baker_aps import SCENES, integrate, load                       # noqa: E402
from baker_north_side import decimated_par                           # noqa: E402
from gpri_tools import jointinv as J                                 # noqa: E402
from gpri_tools.aps import epoch_screen_correction, turbulence_screen  # noqa: E402
from gpri_tools.diurnal import fit_harmonics, periodic_detrend       # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry     # noqa: E402
from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask  # noqa: E402
from gpri_tools.heading import scene_heading, target_heights         # noqa: E402
from gpri_tools.plot import joint_elevation_time, joint_harmonic_map  # noqa: E402
from gpri_tools.timeseries import los_displacement                   # noqa: E402

CAMPAIGNS = ["20170713_full", "20170803_full", "20170827", "20170913", "20180709",
             "20180808", "20190719"]
UTC_OFFSET = -7.0
WORK = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
CACHE_VERSION = 3


def cells_path(name, dec):
    return WORK / name / f"joint_cells_dec{dec}.npz"


def cache_settings(args):
    """What the cell cache depends on besides the scene and the decimation."""
    return {"cache_version": CACHE_VERSION, "size": float(args.size), "dem": str(args.dem),
            "fallback_dem": str(args.fallback_dem)}


def cache_is_current(name, args):
    p = cells_path(name, args.decimate)
    if not p.exists():
        return False
    with np.load(p, allow_pickle=False) as c:
        return all(k in c.files and c[k].item() == v for k, v in cache_settings(args).items())


def build(name, args):
    """One campaign's cell series, from the pair caches of both antennas."""
    scene = Path(SCENES[name])
    stack, net, phase, cc, r, _, n = load(scene, args.decimate, 0, antenna="upper")
    mean_cc = cc.mean(axis=0)
    del cc
    geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                         heading=scene_heading(scene, default=BAKERBEND1_HEADING))
    la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1], cols=[0, geom.shape[1] - 1])
    rgi = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                        bbox=(lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02))
    stable, _ = stable_ground_mask(mean_cc, geom, rgi, threshold=0.85)
    on_ice = glacier_mask(geom, rgi)
    x, y = geom.map_coordinates()
    z = target_heights(geom, args.dem)
    if args.fallback_dem:
        z = np.where(np.isfinite(z), z, target_heights(geom, args.fallback_dem))
    # the two halves of the stable ground are whole cells, so no held-out cell
    # sits on the same ground as fit rock
    fit, held = J.split_cells(stable & np.isfinite(z), x, y, args.size)
    ice = (mean_cc >= 0.5) & on_ice
    # every height is cached; --cand-min-z picks them at solve time
    cand = ~on_ice & ~stable & (mean_cc >= 0.6)

    held_groups, held_keys = J.group_cells(held & np.isfinite(z), x, y, args.size, 3)

    def ladder(d):
        """The ladder on one antenna (metres), as cell series on the held-out cells (mm)."""
        d, _ = epoch_screen_correction(d, fit, r, model="linear", weights=mean_cc)
        for k in range(d.shape[0]):
            d[k] -= turbulence_screen(d[k], fit, sigma=(5.0, 25.0), weights=mean_cc,
                                      wrapped=False)[0]
        return J.cell_series(d * 1000.0, held_groups)

    up, times = integrate(los_displacement(phase, stack.wavelength), net, n)
    del phase
    _, _, phase_l, _, _, _, n_l = load(scene, args.decimate, 0, antenna="lower")
    low, _ = integrate(los_displacement(phase_l, stack.wavelength), net, n_l)
    del phase_l
    if low.shape != up.shape:
        sys.exit(f"{name}: the antennas have {up.shape[0]} and {low.shape[0]} epochs")
    e0 = net.epochs[0]
    camp = J.build_campaign(np.asarray(times, float), up * 1000.0, low * 1000.0, x, y, r, z,
                            fit=fit, ice=ice, held=held, cand=cand, size=args.size,
                            origin_hour=e0.hour + e0.minute / 60.0 + e0.second / 3600.0)
    # the ladder run on each antenna and averaged, as the joint residual averages
    # both, on exactly the held-out cells the campaign kept
    lad = 0.5 * (ladder(up) + ladder(low))
    col = {int(k): i for i, k in enumerate(held_keys)}
    lad = lad[:, [col[int(k)] for k in camp.held.key]]
    keep = J.complete_cells(lad)
    camp = J.Campaign(camp.t, camp.fit, camp.ice, camp.cand, camp.held.subset(np.flatnonzero(keep)),
                      camp.origin_hour)
    camp.save(cells_path(name, args.decimate), ladder_held=lad[:, keep],
              epoch0=np.datetime64(e0).astype("datetime64[s]"), **cache_settings(args))
    print(f"{name}: " + ", ".join(f"{k} {getattr(camp, k).n}" for k in
                                  ("fit", "ice", "cand", "held"))
          + " cells (cells missing at any epoch dropped)", flush=True)


def campaigns(args):
    out = {}
    for name in args.scenes:
        if not cache_is_current(name, args):
            sys.exit(f"{name}: no cell cache for these settings; run --stage cells")
        camp, extra = J.Campaign.load(cells_path(name, args.decimate))
        keep = np.flatnonzero(camp.cand.z >= args.cand_min_z)
        out[name] = (J.Campaign(camp.t, camp.fit, camp.ice, camp.cand.subset(keep), camp.held,
                                camp.origin_hour), extra)
    return out


def score(t, series, weights, full):
    """Anomaly RMS (about the secular line) and 24 h amplitude of a weighted mean, mm."""
    s = np.average(series, axis=1, weights=weights)
    a = (periodic_detrend(t, s, tolerance=0.02)[0] if full
         else s - np.polyval(np.polyfit(t, s, 1), t))
    return float(np.std(a)), (float(fit_harmonics(s[:, None], t).amplitude()[0]) if full else None)


def report(inv, data):
    for kind, label in (("held", "held-out rock not used to calibrate"), ("ice", "ice")):
        if kind in inv.validation:
            p = inv.validation[kind][1]
            print(f"{label}: {100 * np.mean(p > 0.5):.0f} % of {p.size} locations called still")
    print(f"candidates: {np.sum(inv.p > 0.5)} of {inv.p.size} locations still")
    for name, (camp, extra) in data.items():
        h, full = camp.held, camp.spans_a_day
        print(f"\n{name}  (spread calibration x" + ",".join(f"{k:.0f}" for k in inv.kappa[name]) + ")")
        # scored only on the held cells the stationary spread was not calibrated on
        test = ~J.calibration_mask(h.n)
        for lab, sel in (("above 2500 m", test & (h.z >= 2500)), ("below 2300 m", test & (h.z < 2300))):
            if not sel.any():
                continue
            rows = [score(camp.t, m[:, sel], h.n_px[sel], full) for m in
                    (extra["ladder_held"], inv.baseline[name].held_residual,
                     inv.results[name].held_residual)]
            print(f"  held rock {lab} ({sel.sum()} cells), ladder / joint / + candidates: "
                  f"RMS {' / '.join(f'{v[0]:.2f}' for v in rows)} mm"
                  + (f"; 24 h {' / '.join(f'{v[1]:.2f}' for v in rows)} mm" if full else ""))
        if not full:
            continue
        res = inv.results[name]
        amp, sig, _ = res.harmonic(camp.origin_hour)
        sig = sig * np.sqrt(np.mean(inv.kappa[name][1:3]))
        for lab, sel in (("below 2300 m", camp.ice.z < 2300), ("above 2500 m", camp.ice.z >= 2500)):
            w = camp.ice.n_px[sel]
            zc = np.average(res.ice_theta[2][sel] + 1j * res.ice_theta[3][sel], weights=w)
            hr = (camp.origin_hour + np.angle(zc) / (2 * np.pi) * 24) % 24
            print(f"  ice {lab} ({sel.sum()} cells): 24 h {abs(zc):.2f} mm at {hr:04.1f} h UTC; "
                  f"median cell {np.median(amp[sel]):.2f} +/- {np.median(sig[sel]):.2f} mm; "
                  f"rate {np.average(res.ice_theta[1][sel], weights=w) * 365.25 / 1000:+.1f} m/yr")


def figures(inv, data, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    still = dict(zip(inv.keys.tolist(), inv.p > 0.5))
    for name, (camp, extra) in data.items():
        res = inv.results[name]
        if camp.spans_a_day:
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=130)
            st = np.array([still.get(int(k), False) for k in camp.cand.key])
            scale = np.sqrt(np.mean(inv.kappa[name][1:3]))
            for ax, which, label in ((axes[0], "amplitude", "Amplitude (mm)"),
                                     (axes[1], "peak", "Peak (hr UTC)")):
                sc = joint_harmonic_map(ax, res, camp, which, sigma_scale=scale, stationary=st)
                cb = fig.colorbar(sc, ax=ax, label=label, fraction=0.04)
                if which == "peak":
                    cb.set_ticks([0, 6, 12, 18, 24])
                ax.set_xlabel("Easting (km)")
                ax.set_ylabel("Northing (km)")
            fig.tight_layout()
            fig.savefig(outdir / f"40_joint_{name}.png")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 4.2), dpi=130)
        e0 = np.datetime64(extra["epoch0"], "s").astype(dt.datetime)
        mesh = joint_elevation_time(ax, res, camp, e0, utc_offset=UTC_OFFSET)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Elevation (m)")
        fig.colorbar(mesh, ax=ax, label="Ice motion (mm)", fraction=0.03, pad=0.01)
        fig.tight_layout()
        fig.savefig(outdir / f"41_joint_bands_{name}.png")
        plt.close(fig)


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=("cells", "solve", "figures", "all"))
    ap.add_argument("--scenes", nargs="+", default=CAMPAIGNS)
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--size", type=float, default=200.0, help="cell size, m")
    ap.add_argument("--cand-min-z", type=float, default=2300.0,
                    help="candidate reference ground only at or above this height, m")
    ap.add_argument("--shared-noise", default="global", choices=("global", "cell"))
    ap.add_argument("--dem", default=os.environ.get("GPRI_DEM_LIDAR") or os.environ.get("GPRI_DEM"))
    ap.add_argument("--fallback-dem", default=os.environ.get("GPRI_DEM"))
    ap.add_argument("--workers", type=int, default=len(CAMPAIGNS))
    ap.add_argument("--outdir", type=Path, default=None,
                    help="figures; by default docs/figures at the default settings "
                         "and beside the results at any other")
    return ap


def folders(args):
    """The results folder of these settings, and the folder their figures go to.

    Only the default settings draw into ``docs/figures``; a comparison setting
    draws beside its own results, so it never replaces the documented figures.
    """
    tag = f"z{int(args.cand_min_z)}_{args.shared_noise}"
    results = WORK / "joint" / tag
    if args.outdir is not None:
        return results, args.outdir
    default = parser().parse_args([])
    if tag == f"z{int(default.cand_min_z)}_{default.shared_noise}":
        return results, Path("docs/figures")
    return results, results / "figures"


def main():
    args = parser().parse_args()
    results, outdir = folders(args)
    if args.stage in ("cells", "all"):
        for name in args.scenes:
            if args.stage == "cells" or not cache_is_current(name, args):
                build(name, args)
        if args.stage == "cells":
            return
    data = campaigns(args)
    if args.stage in ("solve", "all"):
        with ProcessPoolExecutor(args.workers) as pool:
            inv = J.invert({n: c for n, (c, _) in data.items()}, shared=args.shared_noise,
                           map=pool.map)
        inv.save(results)
    else:
        inv = J.Inversion.load(results)
    report(inv, data)
    figures(inv, data, outdir)


if __name__ == "__main__":
    main()
