#!/usr/bin/env python3
"""The surface at each data set's date, from every late-summer elevation source.

    python -m baker.surface                 # after baker.prepare

The flow model needs the surface at the time of each campaign and ITS_LIVE
year.  The 2015 lidar (26 Aug – 27 Sep 2015) is the reference; every other
late-summer (15 Jul – 15 Oct) elevation over the grid measures its change:

* **ASTER DEMs** (AST14DEM v004, one per stereo scene, 30 m), in
  ``$GPRI_ASTER``; each is co-registered to the lidar on ice-free ground,
  horizontally (Nuth & Kääb 2011, iterated) and vertically (median).  A
  scene's remaining bias varies over it by metres (against co-dated
  ICESat-2 and GEDI); a smooth bias fitted on ice-free ground does not
  carry over to the ice, so instead each pixel's sd is inflated for the
  errors' spatial correlation (``--aster-corr``);
* **ICESat-2 ATL06 and GEDI L2A** (``*_latesummer.npz`` in
  ``$GPRI_ALTIMETRY``, written by the SlideRule fetch in the README); each
  track gets its own vertical offset on ice-free ground, which also takes
  the ellipsoid to the lidar's NAVD88.

Each source's standard deviation is its NMAD against the lidar on ice-free
ground (with the altimeters' formal errors).  The change is fitted as a rate,

    s(x, t) = s_2015(x) + r(x) (t - t_2015),

``r`` a field on the grid, zero off the RGI ice, by weighted least squares
with a curvature penalty (``--smooth``), an ASTER pixel's sd inflated by
``--aster-corr`` over the grid spacing for its errors' spatial correlation, twice, dropping observations more
than ``--reject`` standard deviations from the first fit.  Only data within
``--years`` enter.  Writes ``$GPRI_WORK_ROOT/baker/surface.npz``: ``dhdt``
(m/yr) on the grid, its formal standard deviation, ``t_ref``, and a table
of the sources used; and ``surface_points.npz``, every observation that
entered (grid position, date, change, sd, source, kept or rejected).
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

T_LIDAR = 2015.0 + (datetime(2015, 9, 11) - datetime(2015, 1, 1)).days / 365.25  # mid-survey
LATE_SUMMER = ((7, 15), (10, 15))


def decimal_year(dt):
    start = datetime(dt.year, 1, 1)
    return dt.year + (dt - start).total_seconds() / (365.25 * 86400)


def late_summer(dt):
    (m0, d0), (m1, d1) = LATE_SUMMER
    return (m0, d0) <= (dt.month, dt.day) <= (m1, d1)


def aster_date(path):
    """Acquisition time from an AST14DEM file name (``AST14DEM_004MMDDYYYYHHMMSS_...``)."""
    m = re.search(r"AST14DEM_004(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})", Path(path).name)
    mo, d, y, H, M, S = map(int, m.groups())
    return datetime(y, mo, d, H, M, S)


def nmad(a):
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a))) if a.size else np.nan


# ------------------------------------------------------------- the reference
class Reference:
    """The lidar on the grid and finer, its slope and aspect, and the stable-ground mask."""

    def __init__(self, grid, dem, ice_buffer=100.0, rgi=None):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
        from shapely import vectorized
        from shapely.ops import unary_union
        from gpri_tools.glaciers import load_outlines
        from pyproj import Transformer

        self.grid = grid
        x, y = grid["x"], grid["y"]
        pad = 500.0
        with rasterio.open(dem) as d:
            to = Transformer.from_crs(str(grid["crs"]), d.crs, always_xy=True)
            u, w = to.transform([x[0] - pad, x[-1] + pad], [y[0] - pad, y[-1] + pad])
            win = from_bounds(min(u), min(w), max(u), max(w), d.transform)
            step = 10.0                                          # m: fine enough for 30 m DEMs
            shape = (int(win.height * d.res[1] / step), int(win.width * d.res[0] / step))
            z = d.read(1, window=win, out_shape=shape, masked=True,
                       resampling=Resampling.average).astype(float).filled(np.nan)
            t = d.window_transform(win) * d.window_transform(win).scale(
                win.width / shape[1], win.height / shape[0])
            self.crs = d.crs
        self.z, self.transform = z, t
        gy, gx = np.gradient(z, -t.e, t.a)                       # rows go south
        self.slope = np.arctan(np.hypot(gx, gy))
        self.aspect = np.arctan2(-gx, gy)                        # downslope direction, from north
        cols, rows = np.meshgrid(np.arange(z.shape[1]) + 0.5, np.arange(z.shape[0]) + 0.5)
        X, Y = t * (cols, rows)
        lon_lat = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lo, la = lon_lat.transform([X.min(), X.max()], [Y.min(), Y.max()])
        o = load_outlines(rgi or os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                          bbox=(min(lo), min(la), max(lo), max(la))).to_crs(self.crs)
        ice = unary_union(list(o.geometry))
        self.ice_fine = vectorized.contains(ice, X, Y)
        self.stable = ~vectorized.contains(ice.buffer(ice_buffer), X, Y) & np.isfinite(z)
        self.outline = ice
        self.X, self.Y = X, Y

    def sample(self, x, y, field=None):
        """Bilinear sample of ``field`` (the lidar by default) at points in the DEM's CRS."""
        from scipy.ndimage import map_coordinates
        f = self.z if field is None else field
        col, row = ~self.transform * (np.asarray(x), np.asarray(y))
        return map_coordinates(f, [row - 0.5, col - 0.5], order=1, mode="constant", cval=np.nan)


# ------------------------------------------------------------- the sources
def coregister_raster(ref, path, iterations=5, max_nmad=20.0):
    """A DEM's change from the lidar on the reference's fine grid, after co-registration.

    Returns ``(dh, sigma, info)`` or ``None`` when the DEM has too little
    stable ground or too large a spread there.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    src = np.full(ref.z.shape, np.nan)
    with rasterio.open(path) as d:
        reproject(rasterio.band(d, 1), src, dst_transform=ref.transform, dst_crs=ref.crs,
                  src_nodata=d.nodata if d.nodata is not None else -9999, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    src[(src < -500) | (src > 9000)] = np.nan
    dx = dy = 0.0
    dem = src
    for _ in range(iterations):
        dh = dem - ref.z
        m = ref.stable & np.isfinite(dh) & (ref.slope > np.deg2rad(5)) & (np.abs(dh) < 100)
        if m.sum() < 2000:
            return None
        dh0 = dh[m] - np.median(dh[m])
        keep = np.abs(dh0) < 3 * nmad(dh0)
        y_ = dh[m][keep] / np.tan(ref.slope[m][keep])
        A = np.column_stack([np.cos(ref.aspect[m][keep]), np.sin(ref.aspect[m][keep]),
                             np.ones(keep.sum())])
        (a, b, c), *_ = np.linalg.lstsq(A, y_, rcond=None)
        # a DEM displaced by (e, n) has dh / tan(slope) = e sin(aspect) + n cos(aspect)
        # (aspect: downslope azimuth); it is moved back by (-e, -n)
        sx, sy = -b, -a
        if np.hypot(sx, sy) < 0.5:
            break
        dx, dy = dx + sx, dy + sy
        from scipy.ndimage import shift as nd_shift
        dem = nd_shift(np.nan_to_num(src, nan=-1e4), (dy / ref.transform.e, dx / ref.transform.a),
                       order=1, mode="constant", cval=-1e4)
        dem[dem < -500] = np.nan
    dh = dem - ref.z
    off = np.nanmedian(dh[ref.stable & (np.abs(dh) < 100)])
    dh -= off
    s = nmad(dh[ref.stable])
    if not np.isfinite(s) or s > max_nmad:
        return None
    return dh, s, {"shift_e": dx, "shift_n": dy, "offset": off, "nmad": s}


def coregister_points(ref, d, min_stable=20):
    """Altimetry changes from the lidar, each track offset on stable ground.

    Returns ``(x, y, t, dh, sigma, tracks used)`` in the reference's CRS.
    """
    from pyproj import Transformer
    from shapely import vectorized
    to = Transformer.from_crs("EPSG:4326", ref.crs, always_xy=True)
    x, y = to.transform(d["lon"], d["lat"])
    dh = d["h"] - ref.sample(x, y)
    stable = ~vectorized.contains(ref.outline.buffer(100.0), x, y) & np.isfinite(dh)
    out = np.zeros(len(dh), bool)
    sig = np.full(len(dh), np.nan)
    used = 0
    for tr in np.unique(d["track"]):
        m = d["track"] == tr
        s = m & stable
        if s.sum() < min_stable:
            continue
        off = np.median(dh[s])
        dh[m] -= off
        sig[m] = np.hypot(d["sigma"][m], nmad(dh[s]))
        out |= m
        used += 1
    keep = out & np.isfinite(dh)
    return x[keep], y[keep], d["t"][keep], dh[keep], sig[keep], used


# ----------------------------------------------------------------- the fit
def fit_rate(grid_shape, rows, cols, dt, dh, sigma, ice, smooth, damp=1e-3):
    """``r`` on the grid's ice minimising sum ((r dt - dh)/sigma)^2 + smooth |lap r|^2 + damp |r|^2."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    ny, nx = grid_shape
    idx = -np.ones(grid_shape, int)
    idx[ice] = np.arange(ice.sum())
    n = ice.sum()
    k = idx[rows, cols]
    ok = k >= 0
    w = 1.0 / sigma[ok] ** 2
    G = sp.csr_matrix((dt[ok], (np.arange(ok.sum()), k[ok])), shape=(ok.sum(), n))
    AtA = (G.T @ sp.diags(w) @ G).tocsr()
    Atb = G.T @ (w * dh[ok])
    # five-point Laplacian over ice neighbours
    r_, c_, v_ = [], [], []
    J, I = np.nonzero(ice)
    for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        jj, ii = J + di, I + dj
        inside = (jj >= 0) & (jj < ny) & (ii >= 0) & (ii < nx)
        nb = np.full(J.shape, -1)
        nb[inside] = idx[jj[inside], ii[inside]]
        has = nb >= 0
        r_ += [idx[J, I][has], idx[J, I][has]]
        c_ += [idx[J, I][has], nb[has]]
        v_ += [np.full(has.sum(), -1.0), np.full(has.sum(), 1.0)]
    L = sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))), shape=(n, n))
    M = AtA + smooth * (L.T @ L) + damp * sp.identity(n)
    r = spsolve(M.tocsc(), Atb)
    # formal sd from the data term alone per pixel (diagonal approximation)
    diag = np.asarray(M.diagonal())
    out = np.zeros(grid_shape)
    sd = np.full(grid_shape, np.nan)
    out[ice] = r
    sd[ice] = 1.0 / np.sqrt(diag)
    return out, sd, ok


def main():
    from pyproj import Transformer
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default=os.environ.get("GPRI_DEM_LIDAR"))
    ap.add_argument("--aster", default=os.environ.get("GPRI_ASTER"),
                    help="directory of AST14DEM v004 GeoTIFFs")
    ap.add_argument("--altimetry", default=os.environ.get("GPRI_ALTIMETRY"),
                    help="directory of *_latesummer.npz (ICESat-2 ATL06, GEDI L2A)")
    ap.add_argument("--years", type=float, nargs=2, default=[2013.0, 2024.0],
                    help="only data between these decimal years enter the fit")
    ap.add_argument("--smooth", type=float, default=3e3, help="curvature penalty")
    ap.add_argument("--aster-corr", type=float, default=500.0,
                    help="correlation length of an ASTER DEM's errors, m")
    ap.add_argument("--only", choices=["aster", "altimetry"],
                    help="fit one kind of source alone (for comparison)")
    ap.add_argument("--out", default="surface.npz", help="file name under work/baker")
    ap.add_argument("--reject", type=float, default=3.0, help="outlier threshold, sigmas")
    args = ap.parse_args()

    work = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "baker"
    grid = dict(np.load(work / "grid.npz"))
    ref = Reference(grid, args.dem)
    to_grid = Transformer.from_crs(ref.crs, str(grid["crs"]), always_xy=True)
    x, y = grid["x"], grid["y"]
    sp_ = x[1] - x[0]

    def to_rc(px, py):
        gx, gy = to_grid.transform(px, py)
        c = np.round((gx - x[0]) / sp_).astype(int)
        r = np.round((gy - y[0]) / sp_).astype(int)
        ok = (r >= 0) & (r < len(y)) & (c >= 0) & (c < len(x))
        return r, c, ok

    obs = {k: [] for k in ("r", "c", "t", "dh", "s", "src")}
    table = []
    ice_fine = ref.ice_fine

    def add(r, c, t, dh, s, src):
        for k, v in zip(("r", "c", "t", "dh", "s"), (r, c, t, dh, s)):
            obs[k].append(np.asarray(v, float) if k in ("t", "dh", "s") else np.asarray(v, int))
        obs["src"].append(np.full(len(dh), src))

    use_aster = args.only in (None, "aster")
    use_alt = args.only in (None, "altimetry")
    for path in sorted(Path(args.aster).glob("AST14DEM_*.tif")) if args.aster and use_aster else []:
        when = aster_date(path)
        t = decimal_year(when)
        if not late_summer(when) or not (args.years[0] <= t <= args.years[1]):
            continue
        res = coregister_raster(ref, path)
        if res is None:
            table.append((path.name[:30], t, "rejected", 0, np.nan))
            continue
        dh, s, info = res
        m = ice_fine & np.isfinite(dh) & (np.abs(dh) < 60)
        r, c, ok = to_rc(ref.X[m], ref.Y[m])
        # one observation per grid pixel: the median of the fine samples in it
        key = r[ok] * len(x) + c[ok]
        vals = dh[m][ok]
        order = np.argsort(key)
        uk, start = np.unique(key[order], return_index=True)
        med = np.array([np.median(v) for v in np.split(vals[order], start[1:])])
        # a scene's errors are correlated over ~corr metres: inflate each pixel's sd so
        # the scene counts as about (area / corr^2) independent observations
        inflate = max(args.aster_corr / sp_, 1.0)
        add(uk // len(x), uk % len(x), np.full(uk.size, t), med,
            np.full(uk.size, s * inflate), len(table))
        table.append((path.name[:30], t, f"shift {info['shift_e']:+.0f},{info['shift_n']:+.0f} m",
                      uk.size, s))
    for path in (sorted(Path(args.altimetry).glob("*_latesummer.npz"))
                 if args.altimetry and use_alt else []):
        d = dict(np.load(path))
        sel = (d["t"] >= args.years[0]) & (d["t"] <= args.years[1])
        d = {k: v[sel] for k, v in d.items()}
        px, py, t, dh, s, used = coregister_points(ref, d)
        on = ref.sample(px, py, ref.ice_fine.astype(float)) > 0.5
        r, c, ok = to_rc(px[on], py[on])
        add(r[ok], c[ok], t[on][ok], dh[on][ok], s[on][ok], len(table))
        table.append((path.stem, float(np.median(t)) if t.size else np.nan,
                      f"{used} tracks", int(ok.sum()), float(np.median(s)) if s.size else np.nan))

    o = {k: np.concatenate(v) for k, v in obs.items()}
    ice = grid["ice"]
    dt = o["t"] - T_LIDAR
    rate, sd, used = fit_rate(ice.shape, o["r"], o["c"], dt, o["dh"], o["s"], ice, args.smooth)
    res = (rate[o["r"], o["c"]] * dt - o["dh"]) / o["s"]
    good = np.abs(res) < args.reject
    rate, sd, used = fit_rate(ice.shape, o["r"][good], o["c"][good], dt[good], o["dh"][good],
                              o["s"][good], ice, args.smooth)
    res2 = (rate[o["r"], o["c"]] * dt - o["dh"]) / o["s"]
    np.savez_compressed(work / args.out, dhdt=rate, dhdt_sd=sd, t_ref=T_LIDAR,
                        years=np.array(args.years), smooth=args.smooth,
                        sources=np.array([str(r) for r in table]))
    gx, gy = x[o["c"]], y[o["r"]]
    np.savez_compressed(work / args.out.replace(".npz", "_points.npz"), x=gx, y=gy, t=o["t"],
                        dh=o["dh"], sigma=o["s"], source=o["src"], kept=good,
                        kind=np.array(["aster" if str(r[0]).startswith("AST") else str(r[0])
                                       for r in table])[o["src"]] if table else np.array([]))
    print(f"{len(table)} sources, {o['dh'].size} observations on the ice "
          f"({(~good).sum()} rejected at {args.reject} sd)")
    for i, (name, t, note, n, s) in enumerate(table):
        m = (o["src"] == i) & good
        rr = res2[m]
        print(f"  {name:32s} {t:8.2f}  {note:22s} n {n:6d}  sd {s:5.2f} m  "
              f"residual rms {np.sqrt(np.mean(rr ** 2)) if rr.size else np.nan:5.2f} sd")
    z = grid["surface"]
    for lo, hi in ((1500, 2000), (2000, 2500), (2500, 3000), (3000, 3300)):
        m = ice & (z >= lo) & (z < hi)
        if m.any():
            print(f"  {lo}-{hi} m: dh/dt median {np.median(rate[m]):+.2f} m/yr "
                  f"(5/95 {np.percentile(rate[m], 5):+.2f}/{np.percentile(rate[m], 95):+.2f})")
    print(f"-> {work / args.out}")


if __name__ == "__main__":
    main()
