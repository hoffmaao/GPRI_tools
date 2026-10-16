#!/usr/bin/env python3
"""Inputs for the flow inversion: every campaign's strain rates, and one grid.

    python -m baker.prepare                      # every campaign with results
    python -m baker.prepare --scenes 20170913 20180808

Reads the joint inversion that ``examples/baker_joint.py`` saved under
``$GPRI_WORK_ROOT/joint/<setting>/`` and each campaign's cell cache, and
writes to ``$GPRI_WORK_ROOT/baker/``:

* ``<scene>_obs.npz`` — the campaign's LOS strain rates and their geometry
  (:mod:`baker.data`), in the common map frame (UTM 10 N);
* ``its_live_<year>.npz`` — each ITS_LIVE annual mosaic's surface velocity
  at the pixels on the glaciers of the grid (``--its-live``);
* ``grid.npz`` — the surface, the prior thickness (Farinotti et al. 2019,
  the rasters of the RGI glaciers on the grid, taken from ``--thickness``)
  and the ice mask on one grid covering every campaign's cells, so all the
  campaigns share one mesh and one bed.

Runs in the same environment as the examples (it needs :mod:`gpri_tools`
and the scenes).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))


def main():
    from pyproj import Transformer
    from baker_aps import SCENES, open_stack                      # examples/
    from baker_north_side import decimated_par
    from gpri_tools import jointinv as J
    from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry
    from gpri_tools.glaciers import load_outlines
    from gpri_tools.heading import _dem_sampler, scene_heading
    from shapely import vectorized
    from shapely.ops import unary_union
    from baker.data import COMMON_CRS, grid_fields, its_live, prepare, thickness_rasters

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", help="campaigns (default: all with results)")
    ap.add_argument("--setting", default="z2300_global", help="joint inversion results to read")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--size", type=float, default=200.0, help="the cells' size, m")
    ap.add_argument("--spacing", type=float, default=50.0, help="grid spacing, m")
    ap.add_argument("--margin", type=float, default=1000.0, help="grid beyond the ice cells, m")
    ap.add_argument("--hours", type=float, nargs="+", default=[0, 4, 8, 12, 16, 20])
    ap.add_argument("--dem", default=os.environ.get("GPRI_DEM_LIDAR") or os.environ.get("GPRI_DEM"))
    ap.add_argument("--fallback-dem", default=os.environ.get("GPRI_DEM"))
    ap.add_argument("--thickness", default=os.environ.get("GPRI_THICKNESS"),
                    help="Farinotti et al. (2019) composite-thickness zip of RGI region 02, "
                         "or a directory of <RGIId>_thickness.tif")
    ap.add_argument("--its-live", nargs="*",
                    default=sorted(Path(os.environ["GPRI_ITS_LIVE"]).glob("ITS_LIVE_velocity_*.nc"))
                    if os.environ.get("GPRI_ITS_LIVE") else [],
                    help="ITS_LIVE annual mosaics (default: every one in $GPRI_ITS_LIVE)")
    args = ap.parse_args()

    work = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
    out = work / "baker"
    out.mkdir(parents=True, exist_ok=True)
    inv = J.Inversion.load(work / "joint" / args.setting)
    to_utm = Transformer.from_crs("EPSG:4326", COMMON_CRS, always_xy=True)

    corners, centre = [], None
    for name in args.scenes or sorted(inv.results):
        camp, _ = J.Campaign.load(work / name / f"joint_cells_dec{args.decimate}.npz")
        scene = Path(SCENES[name])
        geom = RadarGeometry(decimated_par(open_stack(scene, "upper").par, args.decimate),
                             heading=scene_heading(scene, default=BAKERBEND1_HEADING))
        to_map = Transformer.from_crs(geom.crs, COMMON_CRS, always_xy=True).transform
        # the radar 2 m above the ground, in the DEM's own datum, as gpri heading does
        rz = float(_dem_sampler(args.dem, geom.lat0, geom.lon0)[1]) + 2.0
        radar = (*to_utm.transform(geom.lon0, geom.lat0), rz)
        obs = prepare(camp, inv.results[name], inv.kappa[name], radar, args.size,
                      args.hours, to_map=to_map)
        obs.save(out / f"{name}_obs.npz", scene=name, origin_hour=camp.origin_hour)
        corners += [obs.xy.min(axis=0), obs.xy.max(axis=0)]
        centre = centre or (geom.lat0, geom.lon0)
        print(f"{name}: {obs.xy.shape[0]} ice cells, {obs.pairs.shape[0]} pairs, "
              f"{'mean rate only' if np.isnan(obs.hours[0]) else f'{obs.hours.size} times'}; strain-rate "
              f"sigma median {np.median(obs.sigma):.4f} /yr, |rate| median "
              f"{np.median(np.abs(obs.rate)):.4f} /yr")

    lo = np.min(corners, axis=0) - args.margin
    hi = np.max(corners, axis=0) + args.margin
    to_ll = Transformer.from_crs(COMMON_CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform([lo[0], hi[0]], [lo[1], hi[1]])
    outlines = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                             bbox=(min(lon), min(lat), max(lon), max(lat)))
    tifs = (thickness_rasters(args.thickness, outlines["RGIId"], out / "thickness")
            if args.thickness else [])
    grid = grid_fields((lo[0], lo[1], hi[0], hi[1]), args.spacing, *centre, args.dem,
                       args.fallback_dem, tifs, outlines)
    np.savez_compressed(out / "grid.npz", crs=COMMON_CRS, **grid)
    union = unary_union(list(outlines.to_crs(COMMON_CRS).geometry.values))
    for path in args.its_live:
        sv = its_live(path, (lo[0], lo[1], hi[0], hi[1]),
                      lambda x, y: vectorized.contains(union, x, y))
        sv.save(out / f"its_live_{int(sv.year)}.npz")
        speed = np.hypot(sv.vx, sv.vy)
        print(f"ITS_LIVE {int(sv.year)}: {speed.size} pixels on the ice, speed median "
              f"{np.median(speed):.1f} m/yr, 95th {np.percentile(speed, 95):.1f}; sigma median "
              f"{np.median(sv.sigma):.2f} m/yr (ice-free floor {float(sv.floor):.2f})")
    h = grid["thickness"]
    print(f"grid {h.shape} at {args.spacing:.0f} m; {len(outlines)} RGI glaciers, "
          f"{len(tifs)} thickness rasters; prior thickness on ice median "
          f"{np.median(h[grid['ice'] & (h > 0)]) if np.any(h > 0) else 0:.0f} m, "
          f"max {h.max():.0f} m -> {out}")


if __name__ == "__main__":
    main()
