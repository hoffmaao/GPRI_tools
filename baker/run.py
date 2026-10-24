#!/usr/bin/env python3
"""Invert every campaign and ITS_LIVE year for one bed and their effective pressures.

    python -m baker.run                       # all the inputs baker.prepare wrote
    python -m baker.run --scenes 20180808 --years 2018 --maxiter 20

Reads ``$GPRI_WORK_ROOT/baker/`` (:mod:`baker.prepare`) and writes, there,

* ``fields.h5`` — the mesh and every control (Firedrake checkpoint);
* ``fields.npz`` — on the 50 m grid of ``grid.npz``: thickness under the
  2015 surface ``H`` and bed ``bed`` (shared), each data set's thickness
  at its own date ``<name>_H``, the stationary effective pressure
  ``N_stationary`` (MPa, ``exp(q_bar)``, with ``--friction shared``), and for
  each data set ``<name>_N_mean`` (MPa, ``exp(q_bar + q0)``), ``<name>_dlogN``
  (``q0``, its departure from the stationary field) and, for campaigns with 24 h terms, ``<name>_logN_amp`` and
  ``<name>_logN_peak`` (the 24 h amplitude of ``log N`` and the UTC hour of
  its maximum); with continuity (the default), ``melt`` (m/yr of ice, the
  melt-only surface source) and for each radar campaign ``<name>_ds_flux``
  and ``<name>_ds_model`` (m: the change since the lidar from the flux
  divergence alone, and with the melt) beside ``<name>_ds_obs``, the radar's.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def on_grid(f, x, y, mask):
    """A mesh field sampled at the grid points in ``mask`` (NaN elsewhere)."""
    X, Y = np.meshgrid(x, y)
    out = np.full(X.shape, np.nan)
    pts = np.column_stack([X[mask], Y[mask]])
    out[mask] = np.asarray(f.at(pts, dont_raise=True, tolerance=1e-6), dtype=float)
    return out


def main():
    import firedrake
    from .inverse import DIURNAL_W, T_LIDAR, Problem, Regularization

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", help="radar campaigns (default: all)")
    ap.add_argument("--years", nargs="*", type=int, help="ITS_LIVE years (default: all)")
    ap.add_argument("--spacing", type=float, default=150.0, help="mesh spacing, m")
    ap.add_argument("--maxiter", type=int, default=50)
    ap.add_argument("--weight", type=float, default=1.0, help="regularization weight")
    ap.add_argument("--friction", choices=["shared", "separate"], default="shared",
                    help="a stationary log N under every data set plus each one's departure "
                         "(default), or a separate log N for each")
    ap.add_argument("--radar-dir", default="radar_single",
                    help="folder of radar_dh_<scene>.npz under the work folder (radar_single, "
                         "radar_joint; '' for the work folder itself)")
    ap.add_argument("--out", default="", help="subfolder of the work folder for the results")
    ap.add_argument("--surface", choices=["radar", "dhdt", "lidar"], default="radar",
                    help="each data set's surface: the lidar plus the radar's own height "
                         "change (default), plus the ASTER/altimetry rate, or the lidar alone")
    ap.add_argument("--continuity", action=argparse.BooleanOptionalAction, default=True,
                    help="fit each radar campaign's surface change with ice continuity and a "
                         "melt-only source (default; needs --surface radar)")
    ap.add_argument("--continuity-max-slope", type=float, default=30.0,
                    help="measured change on terrain steeper than this (degrees, on the lidar) "
                         "is left out of continuity")
    ap.add_argument("--melt-prior", type=float, default=1.0,
                    help="prior melt rate m_0, m/yr of ice")
    args = ap.parse_args()

    folder = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "baker"
    P = Problem.load(folder, spacing=args.spacing, scenes=args.scenes, years=args.years,
                     reg=Regularization(weight=args.weight, m_0=args.melt_prior),
                     surface=args.surface, radar_dir=args.radar_dir or None,
                     friction=args.friction, continuity=args.continuity,
                     max_slope=args.continuity_max_slope)
    out_dir = folder / args.out if args.out else folder
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(P.datasets)} data sets, {P.n_obs:.0f} observations, {len(P.states())} momentum "
          f"balances per evaluation, {P.model.mesh.num_cells()} triangles")
    ctrl = P.invert(maxiter=args.maxiter)

    theta_H, qs = P.split(ctrl)
    H = firedrake.Function(P.model.Q, name="thickness").interpolate(P.thickness(theta_H))
    with firedrake.CheckpointFile(str(out_dir / "fields.h5"), "w") as cf:
        cf.save_mesh(P.model.mesh)
        for f in [H, *ctrl]:
            cf.save_function(f)

    g = np.load(folder / "grid.npz")
    x, y, ice = g["x"], g["y"], g["ice"]
    out = {"x": x, "y": y, "H": on_grid(H, x, y, ice), "surface_source": np.array(args.surface),
           "radar_dir": np.array(args.radar_dir), "friction": np.array(args.friction),
           "continuity": np.array(args.continuity),
           "continuity_max_slope": np.array(args.continuity_max_slope)}
    q_bar, own = P.friction_terms(ctrl)
    if q_bar is not None:
        out["N_stationary"] = np.exp(on_grid(q_bar, x, y, ice))
    out["bed"] = np.where(ice, g["surface"], np.nan) - out["H"]
    for d, q in zip(P.datasets, qs):
        if d.ds is not None:
            Hk = firedrake.Function(P.model.Q).interpolate(P.thickness(theta_H, d))
            out[f"{d.name}_H"] = on_grid(Hk, x, y, ice)
        mean = firedrake.Function(P.model.Q).interpolate(q[0])
        out[f"{d.name}_N_mean"] = np.exp(on_grid(mean, x, y, ice))
        if q_bar is not None:
            out[f"{d.name}_dlogN"] = on_grid(own[P.datasets.index(d)][0], x, y, ice)
        if d.diurnal:
            c, s = on_grid(q[1], x, y, ice), on_grid(q[2], x, y, ice)
            out[f"{d.name}_logN_amp"] = np.hypot(c, s)
            out[f"{d.name}_logN_peak"] = (np.arctan2(s, c) / DIURNAL_W) % 24.0
    if P.continuity:
        melt = firedrake.Function(P.model.Q, name="melt").interpolate(P.melt(ctrl))
        out["melt"] = on_grid(melt, x, y, ice)
        zs, states = P.solve_all(ctrl), P.states()
        DG0 = firedrake.FunctionSpace(P.model.mesh, "DG", 0)
        for i, d in enumerate(P.datasets):
            if d.ds_w is None:
                continue
            Hk = firedrake.Function(P.model.Q).interpolate(P.thickness(theta_H, d))
            u = P.depth_averaged([z for z, (j, _) in zip(zs, states) if j == i])
            flux = firedrake.Function(DG0).interpolate(P.flux_change(d, Hk, u))
            out[f"{d.name}_ds_flux"] = on_grid(flux, x, y, ice)
            out[f"{d.name}_ds_model"] = out[f"{d.name}_ds_flux"] - (d.epoch - T_LIDAR) * out["melt"]
            seen = on_grid(d.ds_w, x, y, ice) > 0
            out[f"{d.name}_ds_obs"] = np.where(seen, on_grid(d.ds_obs, x, y, ice), np.nan)
    np.savez_compressed(out_dir / "fields.npz", **out)
    print(f"-> {out_dir / 'fields.h5'}, {out_dir / 'fields.npz'}")


if __name__ == "__main__":
    main()
