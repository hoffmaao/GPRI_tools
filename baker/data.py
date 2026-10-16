"""The observations and the geometry the flow inversion needs, from one campaign.

:func:`prepare` reads the joint path / ice-motion inversion of
:mod:`gpri_tools.jointinv` (``examples/baker_joint.py`` writes it) and turns
it into what :mod:`baker.inverse` assimilates:

* **LOS strain rates.**  Each ice cell's line-of-sight velocity at any time of
  day follows from its rate and 24 h terms, ``v(t) = rate + w (-c sin wt + s
  cos wt)``.  For every pair of neighbouring ice cells the observation is
  ``(v_i - v_j) / d_ij``, the LOS velocity difference over the cells'
  separation, at a set of times of day.  A difference cancels anything the
  two cells share, which is why strain rates rather than velocities are
  assimilated.  Its variance follows from the cells' (calibrated)
  variances; the pairs sharing a cell are treated as independent.  A
  record shorter than a day has no 24 h terms; its strain rates are those
  of the mean rate, at one "time" whose hour is NaN.
* **Geometry.**  The cells' map position and surface height, the radar's
  position, and from them each cell's unit vector toward the radar.  All
  campaigns are put in one map frame (UTM zone 10 N, :data:`COMMON_CRS`), so
  their cells sit on one mesh and see one bed.
* **ITS_LIVE.**  The annual mosaics' surface velocity (``vx``, ``vy``,
  120 m) at the pixels on the RGI glaciers of the grid
  (:func:`its_live`).  The mosaics' formal errors are an m/yr or less;
  each pixel's standard deviation is that error combined with a floor, the
  RMS of ``vx`` and ``vy`` over the ice-free pixels of the same window.
* **Gridded fields.**  The surface (``--dem``, the 2015 lidar), the prior
  thickness (Farinotti et al. 2019), and the RGI outlines, on one regular
  grid of that frame covering every campaign's cells.

Each campaign is written to its own ``.npz`` and the grid to another.
Velocities are m/yr and positive toward the radar, times are UTC hours,
lengths metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

YEAR_DAYS = 365.25
COMMON_CRS = "EPSG:32610"      # UTM 10 N: one frame for every campaign


@dataclass
class Observations:
    """LOS strain rates between neighbouring ice cells at a set of times of day."""
    hours: np.ndarray          # (nt,) UTC hour of each time sample
    xy: np.ndarray             # (nc, 2) cell centres, m
    z: np.ndarray              # (nc,) surface height at the cells, m
    los: np.ndarray            # (nc, 3) unit vector from the cell toward the radar
    pairs: np.ndarray          # (np, 2) cell indices
    dist: np.ndarray           # (np,) pair separation, m
    rate: np.ndarray           # (nt, np) observed strain rate, 1/yr
    sigma: np.ndarray          # (nt, np) its standard deviation, 1/yr
    velocity: np.ndarray       # (nt, nc) cell LOS velocity, m/yr (for reference)
    radar: np.ndarray          # (3,) radar position, m

    def save(self, path, **extra):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__},
                            **extra)

    @classmethod
    def load(cls, path):
        d = dict(np.load(path, allow_pickle=False))
        obs = cls(**{k: d.pop(k) for k in cls.__dataclass_fields__})
        return obs, d


def campaign_epoch(name):
    """Decimal year of a campaign from its name (``YYYYMMDD...``), at midday."""
    from datetime import datetime
    when = datetime.strptime(name[:8], "%Y%m%d")
    return when.year + ((when - datetime(when.year, 1, 1)).days + 0.5) / 365.25


def neighbour_pairs(xy, size):
    """Index pairs of cells whose centres are within 1.5 cell sizes (edges and corners)."""
    from scipy.spatial import cKDTree
    pairs = np.array(sorted(cKDTree(xy).query_pairs(1.5 * size)), dtype=int).reshape(-1, 2)
    return pairs, np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)


def los_vectors(xy, z, radar):
    """Unit vectors from each cell toward the radar at ``radar = (x, y, z)``."""
    d = np.column_stack([radar[0] - xy[:, 0], radar[1] - xy[:, 1],
                         radar[2] - np.asarray(z, float)])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def cell_velocity(theta, var, hours, origin_hour, t0_days=0.0):
    """LOS velocity (m/yr) and its variance at the given UTC hours, from the 24 h terms.

    ``theta`` is ``(4, n)``: offset (mm), rate (mm/day), cos, sin (mm) of
    :func:`gpri_tools.jointinv.solve`, whose ``t`` counts days from the first
    epoch at UTC hour ``origin_hour``; ``var`` the variances of those terms.
    """
    w = 2 * np.pi
    t = (np.asarray(hours, float) - origin_hour) / 24.0          # days, mod a day
    sin, cos = np.sin(w * t)[:, None], np.cos(w * t)[:, None]
    v = theta[1][None] + w * (-theta[2][None] * sin + theta[3][None] * cos)
    vv = var[1][None] + w ** 2 * (var[2][None] * sin ** 2 + var[3][None] * cos ** 2)
    scale = YEAR_DAYS / 1000.0                                   # mm/day -> m/yr
    return v * scale, vv * scale ** 2


def strain_rates(velocity, variance, pairs, dist):
    """Pair LOS strain rates (1/yr) and their standard deviations."""
    i, j = pairs[:, 0], pairs[:, 1]
    rate = (velocity[:, i] - velocity[:, j]) / dist[None]
    sigma = np.sqrt(variance[:, i] + variance[:, j]) / dist[None]
    return rate, sigma


def prepare(campaign, result, kappa, radar, size, hours=(0, 4, 8, 12, 16, 20), to_map=None):
    """:class:`Observations` from a :class:`gpri_tools.jointinv.Campaign` and its result.

    ``kappa`` is the held-out calibration of the motion terms' variance
    (:attr:`gpri_tools.jointinv.Inversion.kappa`), applied to the formal
    variances; ``radar`` the radar's ``(x, y, z)`` in the map frame, its
    height in the frame of the cells' ``z``; ``to_map(x, y)`` takes the
    campaign's local cell coordinates to the map frame (identity if None).
    """
    ice = campaign.ice
    q, n = result.theta.shape
    theta = result.ice_theta
    var = np.diag(result.cov).reshape(q, n)[:, :result.n_ice].copy()
    var[1:] *= np.asarray(kappa, float)[:, None]
    if q >= 4:
        hours = np.asarray(hours, float)
        v, vv = cell_velocity(theta, var, hours, campaign.origin_hour)
    else:                                       # no 24 h terms: the mean rate only
        hours = np.array([np.nan])
        scale = YEAR_DAYS / 1000.0
        v, vv = theta[1][None] * scale, var[1][None] * scale ** 2
    xy = ice.xy if to_map is None else np.column_stack(to_map(ice.xy[:, 0], ice.xy[:, 1]))
    pairs, dist = neighbour_pairs(xy, size)
    rate, sigma = strain_rates(v, vv, pairs, dist)
    radar = np.asarray(radar, float)
    return Observations(hours=hours, xy=xy, z=ice.z, los=los_vectors(xy, ice.z, radar),
                        pairs=pairs, dist=dist, rate=rate, sigma=sigma, velocity=v, radar=radar)


@dataclass
class SurfaceVelocity:
    """A year of ITS_LIVE surface velocity at pixels on the ice (m/yr)."""
    year: np.ndarray           # () the mosaic's year
    xy: np.ndarray             # (np, 2) pixel centres, m
    vx: np.ndarray             # (np,)
    vy: np.ndarray             # (np,)
    sigma: np.ndarray          # (np,) standard deviation of each component
    floor: np.ndarray          # () the ice-free RMS folded into sigma

    save = Observations.save

    @classmethod
    def load(cls, path):
        d = dict(np.load(path, allow_pickle=False))
        return cls(**{k: d.pop(k) for k in cls.__dataclass_fields__}), d


def its_live(path, bounds, inside, crs=COMMON_CRS):
    """:class:`SurfaceVelocity` from one ITS_LIVE annual mosaic (netCDF, read through GDAL).

    ``bounds`` ``(xmin, ymin, xmax, ymax)`` in ``crs``; ``inside(x, y)`` is
    True for the points kept (on the glaciers being modelled).
    """
    import re
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    def read(name):
        with rasterio.open(f"NETCDF:{path}:{name}") as d:
            to = Transformer.from_crs(crs, d.crs, always_xy=True)
            u, w = to.transform([bounds[0], bounds[2]], [bounds[1], bounds[3]])
            win = from_bounds(min(u), min(w), max(u), max(w), d.transform).round_offsets().round_lengths()
            a = d.read(1, window=win, masked=True).astype(float).filled(np.nan)
            t = d.window_transform(win)
            cols, rows = np.meshgrid(np.arange(a.shape[1]) + 0.5, np.arange(a.shape[0]) + 0.5)
            X, Y = t * (cols, rows)
            back = Transformer.from_crs(d.crs, crs, always_xy=True)
            return a, np.stack(back.transform(X, Y), axis=-1)

    vx, xy = read("vx")
    vy, _ = read("vy")
    ex, _ = read("vx_error")
    ey, _ = read("vy_error")
    ice, _ = read("landice")
    ok = np.isfinite(vx) & np.isfinite(vy)
    off = ok & (ice != 1)
    floor = float(np.sqrt(np.mean(np.r_[vx[off], vy[off]] ** 2))) if off.any() else 0.0
    keep = ok & (ice == 1) & inside(xy[..., 0], xy[..., 1])
    err = np.nan_to_num(np.maximum(ex, ey), nan=0.0)
    year = int(re.search(r"_(\d{4})_", Path(path).name).group(1))
    return SurfaceVelocity(year=np.array(year), xy=xy[keep], vx=vx[keep], vy=vy[keep],
                           sigma=np.sqrt(err[keep] ** 2 + floor ** 2), floor=np.array(floor))


def thickness_rasters(source, rgi_ids, out_dir):
    """The Farinotti et al. (2019) rasters of the given glaciers.

    ``source`` is the composite-thickness zip of an RGI region (the rasters
    of the glaciers asked for are extracted into ``out_dir``) or a directory
    holding ``<RGIId>_thickness.tif`` files.
    """
    import zipfile
    source, out_dir = Path(source), Path(out_dir)
    names = {f"{i}_thickness.tif" for i in rgi_ids}
    if source.is_dir():
        return sorted(p for p in source.rglob("*_thickness.tif") if p.name in names)
    out = []
    with zipfile.ZipFile(source) as z:
        for member in z.namelist():
            if Path(member).name in names:
                target = out_dir / Path(member).name
                if not target.exists():
                    out_dir.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(member))
                out.append(target)
    return sorted(out)


def grid_fields(bounds, spacing, lat0, lon0, dem, fallback_dem, thickness_tifs, outlines,
                crs=COMMON_CRS):
    """Surface, prior thickness and an ice mask on a regular grid of the map frame.

    ``bounds`` is ``(xmin, ymin, xmax, ymax)`` in metres of ``crs``;
    ``(lat0, lon0)`` a point near the middle, where the DEMs are read.
    Thickness rasters (one per glacier, any CRS) are sampled where they have
    data and are 0 elsewhere.
    """
    import rasterio
    from pyproj import Transformer
    from shapely import vectorized
    from shapely.ops import unary_union
    from gpri_tools.heading import _dem_sampler

    x = np.arange(bounds[0], bounds[2] + spacing, spacing)
    y = np.arange(bounds[1], bounds[3] + spacing, spacing)
    X, Y = np.meshgrid(x, y)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(X, Y)
    surface = _dem_sampler(dem, lat0, lon0)[0](lat.ravel(), lon.ravel()).reshape(X.shape)
    if fallback_dem:
        fb = _dem_sampler(fallback_dem, lat0, lon0)[0](lat.ravel(), lon.ravel())
        surface = np.where(np.isfinite(surface), surface, fb.reshape(X.shape))
    thick = np.zeros(X.shape)
    for tif in thickness_tifs:
        with rasterio.open(tif) as d:
            tr = Transformer.from_crs("EPSG:4326", d.crs, always_xy=True)
            u, w = tr.transform(lon, lat)
            row, col = d.index(u.ravel(), w.ravel())
            row, col = np.asarray(row).reshape(X.shape), np.asarray(col).reshape(X.shape)
            ok = (row >= 0) & (row < d.height) & (col >= 0) & (col < d.width)
            band = d.read(1, masked=True).filled(np.nan)
            vals = np.full(X.shape, np.nan)
            vals[ok] = band[row[ok], col[ok]]
            thick = np.where(np.isfinite(vals) & (vals > 0), vals, thick)
    union = unary_union(list(outlines.to_crs(crs).geometry.values))
    ice = vectorized.contains(union, X, Y)
    return {"x": x, "y": y, "surface": surface, "thickness": thick, "ice": ice}
