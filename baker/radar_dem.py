#!/usr/bin/env python3
"""Each campaign's surface from the radar itself, as a correction to the lidar.

    python -m baker.radar_dem --scene 20180808        # or --scene all

The GPRI-II receives on two antennas on one mast in the same sweep, so the
phase of ``upper * conj(lower)`` at one epoch is set by geometry alone: the
target's elevation angle seen across the antennas' vertical separation.
That makes it a topographic measurement taken at the campaign's own date.

**The baseline is calibrated, not assumed.**  The nominal 0.25 m receiver
spacing fits only some campaigns; the others fit effective baselines of
about 0.37 m and ±0.85 m (the antennas were not mounted or wired the same
way on every trip).  So each campaign's phase is fitted, on ice-free ground
only, to

    phi = k (2 pi B0 / lambda) (h_lidar - z_radar) / R + a row + b col + c / R + d

``h_lidar`` the 2015 lidar under each pixel, ``R`` the slant range, ``B0 =
0.25 m``: a scale ``k`` (the effective baseline is ``k B0``), a planar ramp
(the receive chains drift in phase over the sweep), a ``1/R`` term (the
radar's height in the lidar's datum), a constant and a sub-look shift of the
lidar under the radar (``--tilt`` adds a ``cos(theta)`` term, a horizontal
component of the baseline).

**Limits.**  Calibration ground (ice-free, above treeline, below 45
degrees) lies almost all between 1700 and 2300 m, so anything in the
geometry that varies with elevation is fitted there and extrapolated onto
the high ice: on ice-free ground the change relative to the lidar still
varies with elevation by metres, and between campaigns with the same
antennas by 3-10 m above 2300 m.  A coarse scan of ``k``
with the ramp found from the 2-D spectrum picks the branch; a linear fit of
the small residual refines everything.

**What is left on the ice is the change since 2015.**  With the fitted
geometry removed, the residual phase on the glacier is ``k (2 pi B0 /
lambda) dh / R``, so

    dh = residual R / (k 2 pi B0 / lambda),

the campaign's surface minus the lidar's.  A few metres is a small fraction
of a cycle (tens to hundreds of metres), so no unwrapping is needed.  The
ice-free ground's scatter about zero, per range band, is the noise.

Terrain steeper than ``--max-slope`` (45 degrees) is left out everywhere
(layover and shadow).  Ice-free ground below ``--stable-min-z`` (1700 m) is left out of the
calibration and the noise: the lidar is bare earth and the radar sees the
forest canopy.  The epochs are stacked after each one's scene-constant chain phase is
removed, the lidar's predicted phase is taken off at full resolution before
multilooking (the topographic fringe is dense at 0.85 m), and the result is
binned onto the flow model's grid (``grid.npz``, :mod:`baker.prepare`).
Writes ``$GPRI_WORK_ROOT/baker/radar_dh_<scene>.npz``: ``dh`` and its
``sd`` on the grid (NaN where the radar has no coherent data), the number of
looks in each cell, and the calibration.  Runs where the examples do.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
B0 = 0.25                       # m, the nominal receiver spacing


def multilook(z, looks):
    la, lr = looks
    na, nr = z.shape[0] // la * la, z.shape[1] // lr * lr
    return z[:na, :nr].reshape(na // la, la, nr // lr, lr).mean(axis=(1, 3))


def stack_interferogram(upper, lower, epochs, flat_phase, looks):
    """The coherence-weighted stack of the flattened cross-antenna interferogram.

    ``flat_phase`` (full resolution) is removed from every epoch before the
    looks; each epoch's scene-constant phase is removed before stacking.
    Returns ``(phase, coherence)`` on the multilooked grid.
    """
    flat = np.exp(-1j * flat_phase).astype(np.complex64)
    acc = coh = None
    for e in epochs:
        a, b = upper.read_slc(e), lower.read_slc(e)
        na, nr = min(a.shape[0], b.shape[0], flat.shape[0]), min(a.shape[1], b.shape[1], flat.shape[1])
        a, b = a[:na, :nr], b[:na, :nr]
        num = multilook(a * np.conj(b) * flat[:na, :nr], looks)
        den = np.sqrt(multilook(np.abs(a) ** 2, looks) * multilook(np.abs(b) ** 2, looks))
        g = np.where(den > 0, np.abs(num) / np.maximum(den, 1e-30), 0.0)
        z = np.where(np.abs(num) > 0, num / np.maximum(np.abs(num), 1e-30), 0)
        bright = g >= np.nanpercentile(g, 90)
        z = z * np.exp(-1j * np.angle(np.mean(z[bright])))
        acc = z * g if acc is None else acc + z * g
        coh = g if coh is None else coh + g
    return np.angle(acc), coh / len(epochs)


def fit_geometry(phase, s, rows, cols, inv_r, mask, unit, k0, ds_row=None, ds_col=None, tilt=False):
    """Refine ``k`` and fit the ramp, 1/R term and constant on ``mask``.

    ``phase`` is the residual after ``k0 unit s`` came off; returns
    ``(k, coef, residual)`` with ``residual`` the phase after everything.
    """
    # the planar ramp from the 2-D spectrum, so the residual is small enough to fit linearly
    z = np.where(mask, np.exp(1j * phase), 0)
    F = np.fft.fft2(z, s=(z.shape[0] * 4, z.shape[1] * 4))
    i, j = np.unravel_index(np.abs(F).argmax(), F.shape)
    fa = 2 * np.pi * np.fft.fftfreq(F.shape[0])[i]
    fr = 2 * np.pi * np.fft.fftfreq(F.shape[1])[j]
    r = np.angle(np.exp(1j * (phase - fa * rows - fr * cols)))
    r0 = np.angle(np.mean(np.exp(1j * r[mask])))
    r = np.angle(np.exp(1j * (r - r0)))
    # [k - k0, ramp in rows, ramp in cols, 1/R, constant, horizontal baseline (cos theta),
    #  then the lidar's sub-look shift in rows and cols]
    cos_t = unit * np.sqrt(np.clip(1 - s * s, 0, 1)) if tilt else np.zeros_like(s)
    cols_ = [unit * s, rows, cols, inv_r, np.ones_like(s), cos_t]
    if ds_row is not None:                   # a sub-look shift of the lidar under the radar
        cols_ += [k0 * unit * ds_row, k0 * unit * ds_col]
    A = np.stack(cols_, axis=-1)
    m = mask.copy()
    for _ in range(3):                       # refit, dropping 3-sigma outliers
        use = [i for i in range(A.shape[-1]) if tilt or i != 5]
        c, *_ = np.linalg.lstsq(A[m][:, use], r[m], rcond=None)
        coef = np.zeros(A.shape[-1])
        coef[use] = c
        res = np.angle(np.exp(1j * (r - A @ coef)))
        sd = 1.4826 * np.median(np.abs(res[m]))
        m &= np.abs(res) < 3 * sd
    k = k0 + coef[0]
    total = np.array([coef[0], fa + coef[1], fr + coef[2], coef[3], r0 + coef[4], *coef[5:]])
    return k, total, res


def scan_k(phase, s, mask, unit, ks):
    """Best ``k`` with a free planar ramp: the 2-D spectrum's peak at each ``k``."""
    best = (0.0, None)
    for k in ks:
        z = np.where(mask, np.exp(1j * (phase - k * unit * s)), 0)
        R = np.abs(np.fft.fft2(z, s=(z.shape[0] * 2, z.shape[1] * 2))).max() / mask.sum()
        if R > best[0]:
            best = (R, k)
    return best[1], best[0]


def to_grid(geom, grid, values, weight, looks):
    """Bin multilooked radar values onto the model grid (weighted mean per cell)."""
    from pyproj import Transformer
    x, y = grid["x"], grid["y"]
    X, Y = np.meshgrid(x, y)
    to_local = Transformer.from_crs(str(grid["crs"]), geom.crs, always_xy=True)
    lx, ly = to_local.transform(X, Y)
    row, col = geom.radar_coordinates(lx, ly)
    i = np.floor(np.asarray(row) / looks[0]).astype(int)
    j = np.floor(np.asarray(col) / looks[1]).astype(int)
    row, col = np.asarray(row, float), np.asarray(col, float)
    na, nr = geom.shape
    # points outside the swath come back pinned to its edges: drop them
    ok = (np.isfinite(row) & np.isfinite(col) & (row > 0.5) & (row < na - 1.5)
          & (col > 0.5) & (col < nr - 1.5))
    ok &= (i >= 0) & (i < values.shape[0]) & (j >= 0) & (j < values.shape[1])
    # and the far side of the swath: the radar's slant range at the point must be inside it
    ok &= (col < values.shape[1] * looks[1] - 1.5) & (row < values.shape[0] * looks[0] - 1.5)
    v = np.full(X.shape, np.nan)
    w = np.zeros(X.shape)
    vi, wi = values[i[ok], j[ok]], weight[i[ok], j[ok]]
    good = np.isfinite(vi) & (wi > 0)
    idx = np.flatnonzero(ok)[good]
    v.flat[idx] = vi[good]
    w.flat[idx] = wi[good]
    return v, w


def smooth_on_grid(v, w, width):
    """Weighted box average of a gridded field over ``width`` cells; NaN where no weight."""
    from scipy.ndimage import uniform_filter
    vw = uniform_filter(np.nan_to_num(v) * w, width, mode="constant")
    ww = uniform_filter(w, width, mode="constant")
    # the separable filter leaves round-off (~1e-16 of the weights) where there is no data
    ww = np.where(ww > 1e-6 * ww.max(), ww, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ww > 0, vw / ww, np.nan), ww * width * width


def run_scene(name, args):
    from baker_aps import SCENES, open_stack                      # examples/
    from baker_movie import decimated_geom
    from gpri_tools.geocode import BAKERBEND1_HEADING
    from gpri_tools.heading import scene_heading, target_heights
    from shapely import vectorized
    from shapely.ops import unary_union
    from gpri_tools.glaciers import load_outlines
    from pyproj import Transformer

    work = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "baker"
    grid = dict(np.load(work / "grid.npz"))
    scene = Path(SCENES[name])
    upper = open_stack(scene, "upper", lags=(1,), looks=(1, 1))
    lower = open_stack(scene, "lower", lags=(1,), looks=(1, 1))
    geom = decimated_geom(upper, 1, scene_heading(scene, default=BAKERBEND1_HEADING))
    n = min(upper.network.n_epochs, lower.network.n_epochs)
    epochs = np.unique(np.linspace(0, n - 1, min(args.epochs, n)).astype(int))
    looks = tuple(args.looks)
    unit = 2 * np.pi * B0 / upper.wavelength

    na, nr = geom.shape
    h = target_heights(geom, args.dem, rows=np.arange(na), cols=np.arange(nr))
    R_full = np.broadcast_to(np.asarray(geom.slant_range(), float)[None, :], h.shape)
    s_full = np.nan_to_num((h - geom.alt0) / np.maximum(R_full, 1.0))

    # the multilooked geometry, and ice / ice-free masks on it
    la, lr = looks
    rows_c = (np.arange(na // la) + 0.5) * la
    cols_c = (np.arange(nr // lr) + 0.5) * lr
    s = multilook(s_full, looks)
    hl = multilook(np.nan_to_num(h, nan=-9999.0), looks)
    R = np.broadcast_to(np.asarray(geom.slant_range(), float)[cols_c.astype(int)][None, :], s.shape)
    lat, lon = geom.geodetic(rows=rows_c.astype(int), cols=cols_c.astype(int))
    o = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                      bbox=(lon.min() - .02, lat.min() - .02, lon.max() + .02, lat.max() + .02))
    utm = Transformer.from_crs("EPSG:4326", str(grid["crs"]), always_xy=True)
    ux, uy = utm.transform(lon, lat)
    ice_poly = unary_union(list(o.to_crs(str(grid["crs"])).geometry))
    on_ice = vectorized.contains(ice_poly, ux, uy)
    off_ice = ~vectorized.contains(ice_poly.buffer(100.0), ux, uy)
    # layover and shadow: leave out the steepest terrain (slope from the lidar on the grid of looks)
    gx = np.gradient(np.where(np.isfinite(h), h, np.nanmedian(h)), axis=1) / (
        np.asarray(geom.ground_range(), float)[1] - np.asarray(geom.ground_range(), float)[0])
    gy = np.gradient(np.where(np.isfinite(h), h, np.nanmedian(h)), axis=0) / np.maximum(
        R_full * np.deg2rad(abs(geom.par.float("GPRI_az_angle_step", 0.2))), 1.0)
    slope = np.degrees(np.arctan(multilook(np.hypot(gx, gy), looks)))
    valid = (hl > -9000) & np.isfinite(s) & (slope < args.max_slope)
    # the lidar is bare earth and the radar sees canopy tops: stable ground is above the trees
    off_ice &= hl >= args.stable_min_z

    # 1. the branch of k, from a short stack flattened by nothing
    coarse = epochs[:: max(1, len(epochs) // 16)]
    ph0, g0 = stack_interferogram(upper, lower, coarse, np.zeros_like(s_full), looks)
    m0 = valid & off_ice & (g0 >= args.coherence)
    k0, R0 = scan_k(ph0, s, m0, unit, np.arange(-5, 5.0001, 0.01))
    print(f"{name}: {len(epochs)} epochs; k scan {k0:+.2f} (B {k0 * B0:+.3f} m, resultant {R0:.3f})",
          flush=True)

    # 2. register the lidar to the radar: a shift (azimuth lines, range samples) estimated
    #    on the short stack and applied at full resolution, until it stops moving
    from scipy.ndimage import shift as nd_shift
    rows = np.broadcast_to(np.arange(s.shape[0])[:, None], s.shape).astype(float)
    cols = np.broadcast_to(np.arange(s.shape[1])[None, :], s.shape).astype(float)
    shift_az = shift_rg = 0.0
    s_reg, sr = s_full, s
    for it in range(4):
        ph1, g1 = stack_interferogram(upper, lower, coarse, k0 * unit * s_reg, looks)
        m1 = valid & off_ice & (g1 >= args.coherence)
        dsr, dsc = np.gradient(sr)
        _, c1, _ = fit_geometry(ph1, sr, rows, cols, 1e3 / R, m1, unit, k0, dsr, dsc, args.tilt)
        d_az, d_rg = c1[6] * looks[0], c1[7] * looks[1]
        shift_az += d_az
        shift_rg += d_rg
        s_reg = nd_shift(s_full, (-shift_az, -shift_rg), order=1, mode="nearest")
        sr = multilook(s_reg, looks)
        print(f"  registration {it + 1}: lidar shifted {shift_az:+.2f} lines, {shift_rg:+.1f} samples",
              flush=True)
        if abs(d_az) < 0.3 and abs(d_rg) < 2.0:
            break
    s = sr

    # 3. the full stack, flattened at full resolution with the registered lidar
    ph, g = stack_interferogram(upper, lower, epochs, k0 * unit * s_reg, looks)
    upper.close(); lower.close()
    m = valid & off_ice & (g >= args.coherence)
    ds_row, ds_col = np.gradient(s)
    k, coef, res = fit_geometry(ph, s, rows, cols, 1e3 / R, m, unit, k0, ds_row, ds_col, args.tilt)
    # residual everywhere after the fitted geometry (k0 already off; k - k0 in coef[0])
    model = (coef[0] * unit * s + coef[1] * rows + coef[2] * cols + coef[3] * 1e3 / R + coef[4]
             + coef[5] * unit * np.sqrt(np.clip(1 - s * s, 0, 1))
             + k0 * unit * (coef[6] * ds_row + coef[7] * ds_col))
    resid = np.angle(np.exp(1j * (ph - model)))
    dh = resid * R / (k * unit)
    sd_phase = 1.4826 * np.median(np.abs(resid[m]))
    # noise by range: the ice-free scatter of dh in range bands
    bands = np.quantile(R[m], np.linspace(0, 1, 6))
    noise = np.full(s.shape, np.nan)
    print(f"  calibrated: k {k:+.4f} (B {k * B0:+.4f} m); ramp {coef[1]:+.4f} rad/look-row, "
          f"{coef[2]:+.5f} rad/look-col; horizontal baseline {coef[5] * B0:+.4f} m; "
          f"lidar shift {coef[6]:+.2f} az, {coef[7]:+.2f} rg looks; "
          f"ice-free phase sd {sd_phase:.2f} rad", flush=True)
    for lo, hi in zip(bands[:-1], bands[1:]):
        mb = m & (R >= lo) & (R < hi)
        sd = 1.4826 * np.median(np.abs(dh[mb] - np.median(dh[mb])))
        noise[(R >= lo) & (R < hi)] = sd
        amb = upper.wavelength * (lo + hi) / 2 / (abs(k) * B0)
        print(f"   {lo / 1e3:4.1f}-{hi / 1e3:4.1f} km: ice-free dh median {np.median(dh[mb]):+.2f} m, "
              f"sd {sd:5.1f} m per look (height per cycle {amb:5.0f} m)", flush=True)
    for lo, hi in ((1700, 2000), (2000, 2300), (2300, 2600), (2600, 3300)):
        mf = m & (hl >= lo) & (hl < hi)
        mi_ = valid & on_ice & (g >= args.coherence) & (hl >= lo) & (hl < hi)
        print(f"   {lo}-{hi} m: ice-free dh median {np.median(dh[mf]) if mf.any() else np.nan:+.2f} "
              f"(n {mf.sum()}), ice {np.median(dh[mi_]) if mi_.any() else np.nan:+.2f} (n {mi_.sum()})",
              flush=True)
    mi = valid & on_ice & (g >= args.coherence)
    # the looks themselves, for the joint calibration across campaigns (baker.radar_joint)
    np.savez_compressed(work / f"radar_looks_{name}.npz", phase=ph.astype(np.float32),
                        coherence=g.astype(np.float32), s=s.astype(np.float32),
                        ds_row=ds_row.astype(np.float32), ds_col=ds_col.astype(np.float32),
                        R=np.asarray(R, np.float32), h=hl.astype(np.float32),
                        x=np.asarray(ux, np.float64), y=np.asarray(uy, np.float64),
                        on_ice=on_ice, off_ice=off_ice, valid=valid, slope=slope.astype(np.float32),
                        k0=k0, unit=unit, alt0=geom.alt0, looks=np.asarray(looks),
                        shift_lines=shift_az, shift_samples=shift_rg)
    print(f"  on the ice:{mi.sum()} looks, dh median {np.median(dh[mi]):+.2f} m "
          f"(5/95 {np.percentile(dh[mi], 5):+.1f}/{np.percentile(dh[mi], 95):+.1f})", flush=True)

    # 4. onto the model grid: inverse-variance weights, then smoothed over --smooth metres
    w = np.where(valid & (g >= args.coherence), 1.0 / np.maximum(noise, 0.1) ** 2, 0.0)
    v, wg = to_grid(geom, grid, np.where(w > 0, dh, np.nan), w, looks)
    width = max(int(round(args.smooth / (grid["x"][1] - grid["x"][0]))), 1)
    dh_g, n_eff = smooth_on_grid(v, wg, width)
    sd_g = np.where(n_eff > 0, 1.0 / np.sqrt(np.maximum(n_eff, 1e-30)), np.nan)
    ice = grid["ice"]
    stable_g = ~ice & np.isfinite(dh_g) & (grid["surface"] >= args.stable_min_z)
    print(f"  on the grid ({args.smooth:.0f} m): ice cells {np.isfinite(dh_g[ice]).sum()} of {ice.sum()}, "
          f"dh median {np.nanmedian(dh_g[ice]):+.2f} m; ice-free cells sd "
          f"{1.4826 * np.nanmedian(np.abs(dh_g[stable_g] - np.nanmedian(dh_g[stable_g]))):.2f} m", flush=True)
    np.savez_compressed(work / f"radar_dh_{name}.npz", dh=dh_g, sd=sd_g, n_eff=n_eff,
                        k=k, baseline=k * B0, coef=coef, epochs=len(epochs),
                        shift_lines=shift_az, shift_samples=shift_rg,
                        phase_sd=sd_phase, smooth=args.smooth)
    return dh_g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", nargs="+", default=["all"])
    ap.add_argument("--epochs", type=int, default=96, help="epochs stacked, spread over the record")
    ap.add_argument("--looks", type=int, nargs=2, default=(3, 15))
    ap.add_argument("--coherence", type=float, default=0.3)
    ap.add_argument("--smooth", type=float, default=200.0, help="averaging width on the grid, m")
    ap.add_argument("--stable-min-z", type=float, default=1700.0,
                    help="ice-free ground used for calibration lies above this height (treeline), m")
    ap.add_argument("--max-slope", type=float, default=45.0,
                    help="terrain steeper than this (layover, shadow) is left out, degrees")
    ap.add_argument("--tilt", action="store_true",
                    help="also fit a horizontal baseline (cos theta); off by default, because the "
                         "ice-free calibration ground barely reaches above 2300 m")
    ap.add_argument("--dem", default=os.environ.get("GPRI_DEM_LIDAR"))
    args = ap.parse_args()
    work = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "baker"
    names = args.scene
    if names == ["all"]:
        names = [p.name[:-len("_obs.npz")] for p in sorted(work.glob("*_obs.npz"))
                 if not p.name.startswith("surface")]
    for name in names:
        run_scene(name, args)


if __name__ == "__main__":
    main()
