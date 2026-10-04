#!/usr/bin/env python3
"""One inversion for the path delay and the ice's motion, every campaign.

    python examples/baker_joint.py --stage all

:mod:`gpri_tools.jointinv` replaces the ladder and the temporal path delay
with one linear model of the uncorrected, integrated series of **both
antennas**, solved for a campaign at once: a per-epoch trend in range and
height, a spatially correlated path field over rock, ice and candidate ground,
per ice cell an offset, a rate and a 24 h harmonic, a spatially correlated ice
remainder, and antenna noise sized from the upper - lower difference.
Held-out rock never enters; the path predicted there scores the model.

Candidate reference ground — coherent pixels off the glacier outlines that
the reference rules exclude, inside the outline buffer or below the
coherence threshold, at or above ``--cand-min-z`` — enters with its own
motion terms.  Whether each location is stationary is decided once from every
campaign together (:func:`gpri_tools.jointinv.shared_stationarity`), against a
stationary spread calibrated on the held-out rock
(:func:`gpri_tools.jointinv.calibrated_spike`); the final solve then treats a
stationary candidate as rock.

Stages, each cached under ``$GPRI_WORK_ROOT/<scene>/``:

``cells``   cell-median series of both antennas from the pair caches (and,
            for scoring, the ladder run on each antenna, on the same held
            cells), heights from the DEM (``--dem``; the 2015 lidar by default
            when ``GPRI_DEM_LIDAR`` is set); the stable ground is split into
            fit and held-out halves by whole cells; rebuilt when the cache
            was made with other settings
``A``       the path prior by marginal likelihood on the rock, and the
            calibration solve that measures each candidate's motion terms
``EM``      stationarity of every candidate location, all campaigns at once
``B``       the final solve; prints the scores and draws the figures
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates                                     # noqa: E402
import matplotlib.pyplot as plt                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baker_aps import SCENES, integrate, load                         # noqa: E402
from baker_north_side import decimated_par                            # noqa: E402
from gpri_tools import jointinv as J                                  # noqa: E402
from gpri_tools.aps import epoch_screen_correction, turbulence_screen  # noqa: E402
from gpri_tools.diurnal import fit_harmonics, periodic_detrend        # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry      # noqa: E402
from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask  # noqa: E402
from gpri_tools.heading import scene_heading, target_heights          # noqa: E402
from gpri_tools.timeseries import los_displacement                    # noqa: E402

CAMPAIGNS = ["20170713_full", "20170803_full", "20170827", "20170913", "20180709",
             "20180808", "20190719"]
UTC_OFFSET = -7.0
CACHE_VERSION = 2


def root_of(scene):
    return Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene


def cells_path(scene, dec):
    return root_of(scene) / f"joint_cells_dec{dec}.npz"


def cache_settings(args):
    """What the cell cache depends on besides the scene and the decimation."""
    return {"cache_version": CACHE_VERSION, "size": float(args.size), "dem": str(args.dem),
            "fallback_dem": str(args.fallback_dem), "stable_coherence": float(args.stable_coherence),
            "ice_coherence": float(args.ice_coherence), "cand_coherence": float(args.cand_coherence)}


def cache_is_current(scene, args):
    p = cells_path(scene, args.decimate)
    if not p.exists():
        return False
    with np.load(p, allow_pickle=False) as c:
        return all(k in c.files and c[k].item() == v for k, v in cache_settings(args).items())


def calibration_half(n):
    """The held cells that calibrate the stationary spread; the rest score and test."""
    return (np.arange(n) % 2) == 0


# ------------------------------------------------------------------ cells
def build_cells(scene_name, args):
    t0 = time.time()
    scene = Path(SCENES[scene_name])
    dec = args.decimate
    stack, net, phase, cc, r, az, n = load(scene, dec, 0, antenna="upper")
    lam = stack.wavelength
    mean_cc = cc.mean(axis=0)
    del cc
    geom = RadarGeometry(decimated_par(stack.par, dec),
                         heading=scene_heading(scene, default=BAKERBEND1_HEADING))
    la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1], cols=[0, geom.shape[1] - 1])
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                        bbox=(lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02))
    stable, _ = stable_ground_mask(mean_cc, geom, gdf, threshold=args.stable_coherence)
    on_glacier = glacier_mask(geom, gdf)
    x, y = geom.map_coordinates()
    z = target_heights(geom, args.dem)
    if args.fallback_dem and np.isnan(z).any():
        z = np.where(np.isfinite(z), z, target_heights(geom, args.fallback_dem))
    rr = np.broadcast_to(np.asarray(r, float), x.shape)
    ok = np.isfinite(z)
    fit, held = J.split_cells(stable & ok, x, y, args.size)
    ice = (mean_cc >= args.ice_coherence) & on_glacier & ok
    cand = ~on_glacier & ~stable & ok & (mean_cc >= args.cand_coherence)
    classes = {"fit": (fit & ok, 3), "ice": (ice, 8), "cand": (cand, 3), "held": (held & ok, 3)}
    groups = {k: J.group_cells(m, x, y, args.size, nmin) for k, (m, nmin) in classes.items()}

    def series_of(antenna):
        if antenna == "upper":
            ph, nn = phase, n
        else:
            _, _, ph, _, _, _, nn = load(scene, dec, 0, antenna="lower")
        d, times = integrate(los_displacement(ph, lam), net, nn)
        d *= 1000.0                                                     # mm
        return d, np.asarray(times, float)

    def ladder_of(d):
        dl, _ = epoch_screen_correction(d / 1000.0, fit, r, model="linear", weights=mean_cc)
        for k in range(dl.shape[0]):
            scr, _ = turbulence_screen(dl[k], fit, sigma=(5.0, 25.0), weights=mean_cc, wrapped=False)
            dl[k] -= scr
        dl *= 1000.0
        return {k: J.cell_series(dl, groups[k][0]) for k in ("held", "ice")}

    # the ladder, on the same cells, for the scores: run on each antenna and
    # averaged, as the joint inversion's residual averages both
    out = {}
    d, times = series_of("upper")
    up = {k: J.cell_series(d, g) for k, (g, _) in groups.items()}
    lad_u = ladder_of(d)
    del d, phase
    d, times_l = series_of("lower")
    if times_l.size != times.size:
        sys.exit(f"{scene_name}: the antennas have {times.size} and {times_l.size} epochs")
    lw = {k: J.cell_series(d, g) for k, (g, _) in groups.items()}
    lad_l = ladder_of(d)
    del d
    dropped = {}
    for k, (g, keys) in groups.items():
        extra = (lad_u[k], lad_l[k]) if k in lad_u else ()
        keep = np.flatnonzero(J.complete_cells(up[k], lw[k], *extra))
        dropped[k] = len(g) - keep.size
        g = [g[i] for i in keep]
        cs = J.CellSet.build(g, keys[keep], x, y, rr, z, up[k][:, keep], lw[k][:, keep])
        out.update(cs.as_dict(k))
        groups[k] = (g, keys[keep])
        if extra:
            out[f"ladder_{k}"] = 0.5 * (lad_u[k][:, keep] + lad_l[k][:, keep])
    e0 = net.epochs[0]
    np.savez_compressed(cells_path(scene_name, dec), times=times,
                        epoch0=np.datetime64(e0).astype("datetime64[s]"),
                        origin=e0.hour + e0.minute / 60.0 + e0.second / 3600.0,
                        **cache_settings(args), **out)
    print(f"{scene_name}: " + ", ".join(f"{k} {len(g)} cells ({dropped[k]} dropped, missing at an epoch)"
                                        for k, (g, _) in groups.items())
          + f" [{time.time() - t0:.0f} s]", flush=True)


def campaign_of(scene_name, args):
    c = dict(np.load(cells_path(scene_name, args.decimate), allow_pickle=False))
    cs = {k: J.CellSet.from_dict(c, k) for k in ("fit", "ice", "cand", "held")}
    cs["cand"] = cs["cand"].subset(np.flatnonzero(cs["cand"].z >= args.cand_min_z))
    return J.Campaign(c["times"], cs["fit"], cs["ice"], cs["cand"], cs["held"],
                      float(c["origin"])), c


# ---------------------------------------------------------------- A, EM, B
def noise_of(camp, args):
    """Independent noise from the antennas and, with --shared-noise cell, the shared part per cell."""
    cl = (camp.fit, camp.ice, camp.cand)
    Y = [np.hstack([getattr(c, a) for c in cl]) for a in ("yu", "yl")]
    n_px = np.concatenate([c.n_px for c in cl])
    alpha, rho = J.antenna_noise(Y[0], Y[1], n_px, reference=np.arange(camp.fit.n + camp.ice.n))
    shared = J.shared_noise(cl) if args.shared_noise == "cell" else None
    return (alpha, rho), shared


def tag_of(args):
    return f"_z{int(args.cand_min_z)}{'_sn' if args.shared_noise == 'cell' else ''}"


def stage_a(scene_name, args):
    t0 = time.time()
    camp, _ = campaign_of(scene_name, args)
    noise, shared = noise_of(camp, args)
    prior = J.fit_path_prior(camp.fit, noise[0], noise[1][:camp.fit.n])
    kw = {}
    if shared is not None:
        # the calibration solve stacks the held cells after the candidates; they
        # take the median shared level of the fit rock
        held_sh = np.full(camp.held.n, np.median(shared[:camp.fit.n]))
        a_, r_ = noise
        kw = {"noise": (a_, np.concatenate([r_, np.full(camp.held.n, np.median(r_[:camp.fit.n]))])),
              "shared": np.concatenate([shared, held_sh])}
    # half the held-out rock calibrates the stationary spread; the other half,
    # and the ice, are put through the classification afterwards as ground of
    # known state
    calib = calibration_half(camp.held.n)
    spike, kappa, cal = J.calibrated_spike(camp, prior, calibrate=calib, **kw)
    var = J.motion_variance(cal)
    ni, nc = camp.ice.n, camp.cand.n
    test = ~calib
    np.savez(root_of(scene_name) / f"joint_A_dec{args.decimate}{tag_of(args)}.npz",
             length=prior.length, range_growth=prior.range_growth, common=prior.common,
             cand_key=camp.cand.key, cand_terms=cal.cand_theta[1:, :nc], spike=spike,
             kappa=kappa,
             held_key=camp.held.key[test], held_z=camp.held.z[test],
             held_terms=cal.cand_theta[1:, nc:][:, test],
             held_spike=(kappa[None] * var[:, ni + nc:][:, test].T),
             ice_key=camp.ice.key, ice_z=camp.ice.z, ice_terms=cal.ice_theta[1:],
             ice_spike=(kappa[None] * var[:, :ni].T))
    print(f"{scene_name}: path length {prior.length:.0f} m, range growth {prior.range_growth:.1f}/km, "
          f"shared noise {prior.common:.0f}x; stationary spread calibrated x"
          + ",".join(f"{k:.1f}" for k in kappa)
          + (f"; shared noise per cell: rock x{np.median(shared[:camp.fit.n]):.1f}, candidates "
             f"x{np.median(shared[camp.fit.n + camp.ice.n:]) if camp.cand.n else float('nan'):.1f}"
             if shared is not None else "") + f" [{time.time() - t0:.0f} s]", flush=True)


def stage_em(args):
    obs, names = [], []
    for s in args.scenes:
        p = root_of(s) / f"joint_A_dec{args.decimate}{tag_of(args)}.npz"
        if p.exists():
            a = np.load(p)
            obs.append((a["cand_key"], a["cand_terms"], a["spike"]))
            names.append(s)
    keys, p, share, slab = J.shared_stationarity(obs)
    # ground of known state through the same mixture: held-out rock the
    # calibration did not use should come out still, ice should not
    for label, kk, zk in (("held-out rock (not calibrated on)", "held", 2300.0), ("ice", "ice", 0.0)):
        o, zs = [], {}
        for s in names:
            a = np.load(root_of(s) / f"joint_A_dec{args.decimate}{tag_of(args)}.npz")
            o.append((a[f"{kk}_key"], a[f"{kk}_terms"], a[f"{kk}_spike"]))
            zs.update(zip(a[f"{kk}_key"].astype(np.int64).tolist(), a[f"{kk}_z"]))
        vk, vp = J.stationarity_probability(o, share, slab)
        vz = np.array([zs[int(k)] for k in vk])
        print(f"  {label}: {100 * np.mean(vp > 0.5):.0f} % of {vk.size} locations called still"
              + "".join(f"; {lo:.0f}-{hi:.0f} m {100 * np.mean(vp[(vz >= lo) & (vz < hi)] > 0.5):.0f} % "
                        f"of {np.sum((vz >= lo) & (vz < hi))}"
                        for lo, hi in ((0, 2300), (2300, 2500), (2500, 4000)) if np.any((vz >= lo) & (vz < hi))))
    np.savez(root_of("") / f"joint_stationarity_dec{args.decimate}{tag_of(args)}.npz", keys=keys, p=p, share=share,
             campaigns=np.array(names))
    print(f"{keys.size} candidate locations over {len(names)} campaigns: stationary share {share:.2f}, "
          f"{np.sum(p > 0.5)} with p > 0.5")


def score(t, res, w, full, origin):
    sr = np.average(res, axis=1, weights=w)
    # the secular line from same-hour differences where the record allows one,
    # with the harmonic fits' allowance for a record minutes short of a day
    a = (periodic_detrend(t, sr, tolerance=0.02)[0] if full
         else sr - np.polyval(np.polyfit(t, sr, 1), t))
    out = [float(np.std(a))]
    if full:
        f = fit_harmonics(sr[:, None], t)
        out.append(float(f.amplitude()[0]))
    return out


def stage_b(scene_name, args):
    t0 = time.time()
    camp, c = campaign_of(scene_name, args)
    a = np.load(root_of(scene_name) / f"joint_A_dec{args.decimate}{tag_of(args)}.npz")
    st = np.load(root_of("") / f"joint_stationarity_dec{args.decimate}{tag_of(args)}.npz")
    noise, shared = noise_of(camp, args)
    pmap = dict(zip(st["keys"].astype(np.int64).tolist(), st["p"]))
    p = np.array([pmap.get(int(k), 0.0) for k in camp.cand.key])
    prior = J.PathPrior(float(a["length"]), float(a["range_growth"]), float(a["common"]))
    res = J.solve(camp, prior, cand_p=p, cand_spike=a["spike"], noise=noise, shared=shared)
    nfi = camp.fit.n + camp.ice.n
    base = J.solve(J.Campaign(camp.t, camp.fit, camp.ice, camp.cand.subset(np.zeros(0, int)),
                              camp.held, camp.origin_hour), prior,
                   noise=(noise[0], noise[1][:nfi]),
                   shared=None if shared is None else shared[:nfi])
    t, h, full = camp.t, camp.held, camp.spans_a_day
    ladder = c["ladder_held"]
    rows = []
    # scored only on the held cells the stationary spread was not calibrated on
    test = ~calibration_half(h.n)
    for lab, sel in (("above 2500 m", test & (h.z >= 2500)), ("below 2300 m", test & (h.z < 2300))):
        if not sel.any():
            continue
        lad = score(t, ladder[:, sel], h.n_px[sel], full, camp.origin_hour)
        b0 = score(t, base.held_residual[:, sel], h.n_px[sel], full, camp.origin_hour)
        jn = score(t, res.held_residual[:, sel], h.n_px[sel], full, camp.origin_hour)
        rows.append(f"  held rock {lab} ({sel.sum()} cells): RMS ladder {lad[0]:.2f}, joint {b0[0]:.2f}, "
                    f"+ candidates {jn[0]:.2f} mm"
                    + (f"; 24 h {lad[1]:.2f} / {b0[1]:.2f} / {jn[1]:.2f} mm" if full else ""))
    print(f"{scene_name}: {np.sum(p > 0.5)} of {camp.cand.n} candidates stationary "
          f"[{time.time() - t0:.0f} s]\n" + "\n".join(rows))
    if full:
        amp, sig, peak = res.harmonic(camp.origin_hour)
        # the formal sigma treats path and remainder as independent between
        # epochs; held-out rock says by how much that under-states the error
        sig = sig * np.sqrt(np.mean(a["kappa"][1:3]))
        for lab, sel in (("below 2300 m", camp.ice.z < 2300), ("above 2500 m", camp.ice.z >= 2500)):
            zc = np.average(res.ice_theta[2][sel] + 1j * res.ice_theta[3][sel], weights=camp.ice.n_px[sel])
            hr = (camp.origin_hour + np.angle(zc) / (2 * np.pi) * 24) % 24
            print(f"  ice {lab} ({sel.sum()} cells): 24 h {abs(zc):.2f} mm at {hr:04.1f} h UTC, "
                  f"median cell {np.median(amp[sel]):.2f} mm (sigma {np.median(sig[sel]):.2f}, calibrated on held rock), "
                  f"rate {np.average(res.ice_theta[1][sel], weights=camp.ice.n_px[sel]) * 365.25 / 1000:+.1f} m/yr")
    np.savez_compressed(root_of(scene_name) / f"joint_B_dec{args.decimate}{tag_of(args)}.npz",
                        theta=res.theta, cov_diag=np.diag(res.cov), n_ice=res.n_ice, p=p,
                        held_residual=res.held_residual, ice_remainder=res.ice_remainder,
                        basis=res.basis, ice_xy=camp.ice.xy, ice_z=camp.ice.z, ice_n=camp.ice.n_px,
                        cand_xy=camp.cand.xy, held_xy=camp.held.xy, fit_xy=camp.fit.xy,
                        times=camp.t, origin=camp.origin_hour, epoch0=c["epoch0"])
    if args.figures:
        figures(scene_name, camp, res, p, c, args, np.sqrt(np.mean(a["kappa"][1:3])) if full else 1.0)


# ---------------------------------------------------------------- figures
def figures(scene_name, camp, res, p, c, args, sigma_scale=1.0):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ic = camp.ice
    if camp.spans_a_day:
        amp, sig, peak = res.harmonic(camp.origin_hour)
        sig = sig * sigma_scale
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=130)
        for ax, v, cmap, lim, lab in ((axes[0], amp, "viridis", (0, np.percentile(amp, 97)), "Amplitude (mm)"),
                                      (axes[1], peak, "hsv", (0, 24), "Peak (hr UTC)")):
            for xy, col, mk, s in ((camp.fit.xy, "0.75", "o", 8), (camp.held.xy, "0.45", "o", 8),
                                   (camp.cand.xy[p > 0.5], "k", "^", 22)):
                ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, s=s, color=col, marker=mk, lw=0)
            # an hour of peak is only drawn solid where the amplitude is twice its
            # calibrated sigma
            alpha = None if lab.startswith("Amp") else np.where(amp > 2 * sig, 1.0, 0.2)
            sc = ax.scatter(ic.xy[:, 0] / 1000, ic.xy[:, 1] / 1000, c=v, cmap=cmap, vmin=lim[0],
                            vmax=lim[1], s=30, marker="s", alpha=alpha, lw=0)
            cb = fig.colorbar(sc, ax=ax, label=lab, fraction=0.04)
            if lab.startswith("Peak"):
                cb.set_ticks([0, 6, 12, 18, 24])
            ax.set_aspect("equal")
            ax.set_xlabel("Easting (km)")
            ax.set_ylabel("Northing (km)")
        fig.tight_layout()
        fig.savefig(out / f"40_joint_{scene_name}.png")
        plt.close(fig)
    # elevation against time: each cell's harmonic and remainder, by 100 m band
    mot = res.ice_motion()
    mot = mot - mot.mean(axis=0)
    edges = np.arange(1300, 3401, 100.0)
    band = np.digitize(ic.z, edges) - 1
    hours = camp.t * 24.0
    tb = np.arange(0, hours[-1] + 0.5, 0.5)
    bi = np.digitize(hours, tb) - 1
    M = np.full((edges.size - 1, tb.size - 1), np.nan)
    for k in range(edges.size - 1):
        sel = band == k
        if sel.sum() < 2:
            continue
        s = np.average(mot[:, sel], axis=1, weights=ic.n_px[sel])
        for i in range(tb.size - 1):
            m = bi == i
            if m.any():
                M[k, i] = s[m].mean()
    rows = np.where(np.isfinite(M).any(axis=1))[0]
    lo, hi = rows.min(), rows.max() + 1
    lim = float(np.nanpercentile(np.abs(M), 98))
    e0 = np.datetime64(c["epoch0"], "s").astype(dt.datetime)
    te = [e0 + dt.timedelta(hours=float(h)) for h in tb]
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=130)
    im = ax.pcolormesh(mdates.date2num(te), edges[lo:hi + 1], M[lo:hi], cmap="RdBu_r",
                       vmin=-lim, vmax=lim, shading="flat")
    day = (te[0] + dt.timedelta(hours=UTC_OFFSET)).replace(hour=0, minute=0, second=0, microsecond=0)
    day -= dt.timedelta(hours=UTC_OFFSET)
    while day < te[-1]:
        a_, b_ = max(day, te[0]), min(day + dt.timedelta(hours=6), te[-1])
        if a_ < b_:
            ax.axvspan(mdates.date2num(a_), mdates.date2num(b_), ymin=0.975, ymax=1.0, color="0.3", lw=0)
        day += dt.timedelta(days=1)
    loc = mdates.HourLocator(byhour=range(0, 24, 3 if hours[-1] <= 30 else 6))
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Elevation (m)")
    fig.colorbar(im, ax=ax, label="Ice motion (mm)", fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(out / f"41_joint_bands_{scene_name}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=("cells", "A", "EM", "B", "all"))
    ap.add_argument("--scenes", nargs="+", default=CAMPAIGNS)
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--size", type=float, default=200.0, help="cell size, m")
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--cand-coherence", type=float, default=0.6)
    ap.add_argument("--cand-min-z", type=float, default=2300.0,
                    help="candidate reference ground only at or above this height, m")
    ap.add_argument("--dem", default=os.environ.get("GPRI_DEM_LIDAR") or os.environ.get("GPRI_DEM"))
    ap.add_argument("--fallback-dem", default=os.environ.get("GPRI_DEM"),
                    help="heights where --dem has none")
    ap.add_argument("--shared-noise", default="global", choices=("global", "cell"),
                    help="noise both antennas share: one level fitted on the rock, or "
                         "measured per cell (gpri_tools.jointinv.shared_noise)")
    ap.add_argument("--no-figures", dest="figures", action="store_false")
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()
    if args.stage in ("cells", "all"):
        for s in args.scenes:
            if args.stage == "cells" or not cache_is_current(s, args):
                build_cells(s, args)
    if args.stage in ("A", "all"):
        for s in args.scenes:
            stage_a(s, args)
    if args.stage in ("EM", "all"):
        stage_em(args)
    if args.stage in ("B", "all"):
        for s in args.scenes:
            stage_b(s, args)


if __name__ == "__main__":
    main()
