#!/usr/bin/env python3
"""One calibration of the two-antenna surfaces across all campaigns.

    python -m baker.radar_joint                 # after baker.radar_dem on every scene

:mod:`baker.radar_dem` calibrates each campaign on its own ice-free ground.
That leaves, in every campaign, the same change relative to the lidar with
elevation, in metres, although the campaigns' effective baselines differ by
a factor of three.  A term that is the same in metres in every campaign
enters each one's phase scaled by its own baseline, ``k_c unit E / R``,
while a fault of one campaign's receivers enters its phase directly; fitting
all campaigns at once on the ice-free ground separates the two.

Each campaign keeps its own ``k``, planar ramp, ``1/R`` term, constant and
lidar shift (:func:`baker.radar_dem.fit_geometry`).  Shared, in metres:

    E = e1 z^2 + e2 z R   (+ e3 z^3 with --cubic)

with ``z = (h - 2000 m) / 1 km`` and ``R = (R - 7.5 km) / 1 km``.  A term
linear in ``z`` alone is each campaign's ``k`` and a term in ``R^2`` alone
is each campaign's range ramp and ``1/R``, so neither can be shared.

**Held out.**  The ice-free ground is cut into 1 km blocks; each fold is
fitted without its blocks and scored on them.  The calibration ground lies
between 1700 and 2300 m, so a second test fits below 2150 m and scores
above it.  Both are reported per campaign for the per-campaign fit (``E =
0``) and for the joint one.

Writes ``$GPRI_WORK_ROOT/baker/radar_joint/radar_dh_<scene>.npz`` in the
layout of :mod:`baker.radar_dem`, and ``radar_joint/summary.npz``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .radar_dem import fit_geometry, smooth_on_grid

Z0, R0 = 2000.0, 7500.0


def load(work):
    looks = {}
    for p in sorted(work.glob("radar_looks_*.npz")):
        d = dict(np.load(p))
        name = p.stem[len("radar_looks_"):]
        shape = d["phase"].shape
        d["rows"] = np.broadcast_to(np.arange(shape[0])[:, None], shape).astype(float)
        d["cols"] = np.broadcast_to(np.arange(shape[1])[None, :], shape).astype(float)
        looks[name] = d
    return looks


def shared_basis(h, R, cubic=False):
    z, r = (h - Z0) / 1e3, (R - R0) / 1e3
    cols = [z * z, z * r] + ([z ** 3] if cubic else [])
    return np.stack(cols, axis=-1)


def single(d, mask):
    """The per-campaign calibration of :mod:`baker.radar_dem` on ``mask``."""
    k, coef, _ = fit_geometry(d["phase"], d["s"], d["rows"], d["cols"], 1e3 / d["R"], mask,
                              d["unit"], d["k0"], d["ds_row"], d["ds_col"])
    return k, coef


def model_phase(d, coef):
    s, R = d["s"], d["R"]
    return (coef[0] * d["unit"] * s + coef[1] * d["rows"] + coef[2] * d["cols"]
            + coef[3] * 1e3 / R + coef[4]
            + d["k0"] * d["unit"] * (coef[6] * d["ds_row"] + coef[7] * d["ds_col"]))


def joint(looks, masks, cubic=False, iters=3):
    """Per-campaign geometry plus a shared height term ``E``, fitted on ``masks``.

    Starts from each campaign's own fit and solves for the corrections to it
    and the shared ``E`` coefficients together, weighting each campaign by
    its ice-free phase scatter.  Returns ``({name: (k, coef)}, e)``.
    """
    names = list(looks)
    base = {n: single(looks[n], masks[n]) for n in names}
    ne = 3 if cubic else 2
    npc = 7                                    # dk, ramp rows, ramp cols, 1/R, const, shift az, rg
    keep = {n: masks[n].copy() for n in names}
    for _ in range(iters):
        blocks, rhs = [], []
        for c, n in enumerate(names):
            d, (k, coef) = looks[n], base[n]
            m = keep[n]
            r = np.angle(np.exp(1j * (d["phase"] - model_phase(d, coef))))[m]
            sd = 1.4826 * np.median(np.abs(r))
            local = np.stack([d["unit"] * d["s"], d["rows"], d["cols"], 1e3 / d["R"],
                              np.ones_like(d["s"]), d["k0"] * d["unit"] * d["ds_row"],
                              d["k0"] * d["unit"] * d["ds_col"]], axis=-1)[m]
            E = k * d["unit"] * shared_basis(d["h"], d["R"], cubic)[m] / d["R"][m][:, None]
            A = np.zeros((m.sum(), npc * len(names) + ne))
            A[:, c * npc:(c + 1) * npc] = local
            A[:, npc * len(names):] = E
            blocks.append(A / sd)
            rhs.append(r / sd)
        A, b = np.concatenate(blocks), np.concatenate(rhs)
        scale = np.maximum(np.sqrt((A * A).mean(axis=0)), 1e-30)
        x, *_ = np.linalg.lstsq(A / scale, b, rcond=None)
        x /= scale
        e = x[npc * len(names):]
        # drop 3-sigma outliers per campaign and refit
        for c, n in enumerate(names):
            res = phase_residual(looks[n], *update(base[n], x[c * npc:(c + 1) * npc]), e, cubic)
            sd = 1.4826 * np.median(np.abs(res[masks[n]]))
            keep[n] = masks[n] & (np.abs(res) < 3 * sd)
    out = {n: update(base[n], x[c * npc:(c + 1) * npc]) for c, n in enumerate(names)}
    return out, e


def update(kc, dx):
    k, coef = kc
    coef = coef.copy()
    coef[0] += dx[0]
    coef[1:5] += dx[1:5]
    coef[6:8] += dx[5:7]
    return k + dx[0], coef


def phase_residual(d, k, coef, e, cubic=False):
    E = shared_basis(d["h"], d["R"], cubic) @ e if len(e) else 0.0
    return np.angle(np.exp(1j * (d["phase"] - model_phase(d, coef)
                                 - k * d["unit"] * E / d["R"])))


def heights(d, k, coef, e, cubic=False):
    """The change relative to the lidar, metres, at every look."""
    return phase_residual(d, k, coef, e, cubic) * d["R"] / (k * d["unit"])


def stable(d, coherence, min_z):
    return (d["valid"] & d["off_ice"] & (d["coherence"] >= coherence) & (d["h"] >= min_z))


def robust_sd(v):
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else np.nan


def held_out(looks, masks, cubic, shared, block=1000.0, folds=5, seed=0):
    """Out-of-block dh on the ice-free ground, every look scored once."""
    rng = np.random.default_rng(seed)
    cell = {n: (np.floor(d["x"] / block).astype(np.int64) * 100003
                + np.floor(d["y"] / block).astype(np.int64)) for n, d in looks.items()}
    ids = np.unique(np.concatenate([cell[n][masks[n]] for n in looks]))
    fold_of = dict(zip(ids, rng.integers(0, folds, ids.size)))
    lookup = {n: np.vectorize(fold_of.get, otypes=[float])(cell[n]) for n in looks}
    dh = {n: np.full(d["phase"].shape, np.nan) for n, d in looks.items()}
    for f in range(folds):
        train = {n: masks[n] & (lookup[n] != f) for n in looks}
        test = {n: masks[n] & (lookup[n] == f) for n in looks}
        fit, e = (joint(looks, train, cubic) if shared
                  else ({n: single(looks[n], train[n]) for n in looks}, np.zeros(0)))
        for n, d in looks.items():
            h = heights(d, *fit[n], e, cubic)
            dh[n][test[n]] = h[test[n]]
    return dh


def height_held_out(looks, masks, cubic, shared, split=2150.0):
    train = {n: masks[n] & (looks[n]["h"] < split) for n in looks}
    test = {n: masks[n] & (looks[n]["h"] >= split) for n in looks}
    fit, e = (joint(looks, train, cubic) if shared
              else ({n: single(looks[n], train[n]) for n in looks}, np.zeros(0)))
    return {n: np.where(test[n], heights(looks[n], *fit[n], e, cubic), np.nan) for n in looks}


def to_cells(grid, d, values, weight):
    """Bin looks onto the model grid by their map position (weighted mean per cell)."""
    x, y = grid["x"], grid["y"]
    j = np.rint((d["x"] - x[0]) / (x[1] - x[0])).astype(int)
    i = np.rint((d["y"] - y[0]) / (y[1] - y[0])).astype(int)
    ok = (np.isfinite(values) & (weight > 0) & (i >= 0) & (i < y.size) & (j >= 0) & (j < x.size))
    vw = np.zeros((y.size, x.size))
    w = np.zeros((y.size, x.size))
    np.add.at(vw, (i[ok], j[ok]), values[ok] * weight[ok])
    np.add.at(w, (i[ok], j[ok]), weight[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, vw / w, np.nan), w


def report(label, looks, dh, masks, bands=((1700, 2000), (2000, 2150), (2150, 2300), (2300, 3300))):
    print(f"  {label}")
    for n, d in looks.items():
        m = masks[n] & np.isfinite(dh[n])
        meds = " ".join(f"{np.median(dh[n][m & (d['h'] >= lo) & (d['h'] < hi)]) if (m & (d['h'] >= lo) & (d['h'] < hi)).sum() >= 5 else np.nan:+6.2f}"
                        for lo, hi in bands)
        print(f"    {n:15s} sd {robust_sd(dh[n][m]):5.2f} m  median by band {meds}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coherence", type=float, default=0.3)
    ap.add_argument("--stable-min-z", type=float, default=1700.0)
    ap.add_argument("--smooth", type=float, default=200.0, help="averaging width on the grid, m")
    ap.add_argument("--cubic", action="store_true", help="also share a z^3 term")
    ap.add_argument("--skip-held-out", action="store_true")
    args = ap.parse_args()
    work = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "baker"
    looks = load(work)
    masks = {n: stable(d, args.coherence, args.stable_min_z) for n, d in looks.items()}
    print(f"{len(looks)} campaigns; ice-free looks " +
          ", ".join(f"{n} {m.sum()}" for n, m in masks.items()), flush=True)

    if not args.skip_held_out:
        held_out_report(looks, masks, args)
    fit, e = joint(looks, masks, args.cubic)
    single_fit = {n: single(looks[n], masks[n]) for n in looks}
    names = ["z^2", "z R"] + (["z^3"] if args.cubic else [])
    print("shared height term: " + ", ".join(f"{a} {b:+.3f} m" for a, b in zip(names, e)))
    zz = np.arange(1500, 3301, 300.0)
    for r in (6000.0, 7500.0, 9000.0):
        E = shared_basis(zz, np.full_like(zz, r), args.cubic) @ e
        print(f"  E at R {r / 1e3:.1f} km: " + " ".join(f"{z:.0f} m {v:+.1f}" for z, v in zip(zz, E)))
    for n in looks:
        print(f"  {n:15s} k {single_fit[n][0]:+.4f} -> {fit[n][0]:+.4f} "
              f"(B {single_fit[n][0] * 0.25:+.4f} -> {fit[n][0] * 0.25:+.4f} m)")
    grid = dict(np.load(work / "grid.npz"))
    print("per campaign -> radar_single/")
    write(work / "radar_single", grid, looks, masks, single_fit, np.zeros(0), [], args)
    print("joint -> radar_joint/")
    write(work / "radar_joint", grid, looks, masks, fit, e, names, args)


def held_out_report(looks, masks, args):
    print("held out by 1 km blocks (ice-free dh, m; bands 1700-2000, 2000-2150, 2150-2300, >2300 m)")
    report("per campaign", looks, held_out(looks, masks, args.cubic, False), masks)
    report("joint", looks, held_out(looks, masks, args.cubic, True), masks)
    print("held out above 2150 m (fitted below)")
    report("per campaign", looks, height_held_out(looks, masks, args.cubic, False), masks)
    report("joint", looks, height_held_out(looks, masks, args.cubic, True), masks)



def write(out, grid, looks, masks, fit, e, names, args):
    out.mkdir(exist_ok=True)
    width = max(int(round(args.smooth / (grid["x"][1] - grid["x"][0]))), 1)
    for n, d in looks.items():
        k, coef = fit[n]
        dh = heights(d, k, coef, e, args.cubic)
        m = masks[n]
        bands = np.quantile(d["R"][m], np.linspace(0, 1, 6))
        noise = np.full(dh.shape, np.nan)
        for lo, hi in zip(bands[:-1], bands[1:]):
            mb = m & (d["R"] >= lo) & (d["R"] < hi)
            inb = ((d["R"] >= lo) | (lo == bands[0])) & ((d["R"] < hi) | (hi == bands[-1]))
            noise[inb] = robust_sd(dh[mb])
        use = d["valid"] & (d["coherence"] >= args.coherence)
        outside = use & d["on_ice"] & ((d["R"] < bands[0]) | (d["R"] > bands[-1]))
        print(f"  {n:15s} coherent ice looks outside the ice-free range span: "
              f"{outside.sum()} of {(use & d['on_ice']).sum()}", flush=True)
        w = np.where(use, 1.0 / np.maximum(noise, 0.1) ** 2, 0.0)
        v, wg = to_cells(grid, d, np.where(use, dh, np.nan), w)
        dh_g, n_eff = smooth_on_grid(v, wg, width)
        sd_g = np.where(n_eff > 0, 1.0 / np.sqrt(np.maximum(n_eff, 1e-30)), np.nan)
        ice = grid["ice"]
        stable_g = ~ice & np.isfinite(dh_g) & (grid["surface"] >= args.stable_min_z)
        mi = use & d["on_ice"]
        print(f"  {n:15s} on the ice {np.median(dh[mi]):+.2f} m (5/95 {np.percentile(dh[mi], 5):+.1f}/"
              f"{np.percentile(dh[mi], 95):+.1f}); grid ice cells {np.isfinite(dh_g[ice]).sum()}, "
              f"ice-free cells sd {robust_sd(dh_g[stable_g]):.2f} m", flush=True)
        np.savez_compressed(out / f"radar_dh_{n}.npz", dh=dh_g, surface=grid["surface"] + dh_g, sd=sd_g, n_eff=n_eff, k=k,
                            baseline=k * 0.25, coef=coef, shared=e, shared_terms=np.array(names),
                            phase_sd=robust_sd(phase_residual(d, k, coef, e, args.cubic)[m]),
                            smooth=args.smooth, calibration="joint" if len(e) else "single")
    np.savez_compressed(out / "summary.npz", shared=e, shared_terms=np.array(names),
                        names=np.array(list(looks)), k=np.array([fit[n][0] for n in looks]))


if __name__ == "__main__":
    main()
