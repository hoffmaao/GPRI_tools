#!/usr/bin/env python3
"""Figures of an inversion :mod:`baker.run` saved.

    python -m baker.figures --predict  # Firedrake environment: the model's observations
    python -m baker.figures            # plotting (any environment with matplotlib)
    python -m baker.figures --run radar_friction --predict   # a run saved with --out

``--predict`` runs the forward model once at the saved fields for every
data set and time and caches what it sees in ``predictions.npz``; the
plotting then writes to ``fig/`` beside the fields
(``$GPRI_WORK_ROOT/baker/``, or ``--run``'s subfolder of it), one panel per
figure, the RGI outlines on every map:

* ``thickness.png``, ``thickness_prior.png``, ``thickness_ratio.png``,
  ``bed.png`` — the shared thickness, its Farinotti prior, their ratio,
  and the bed;
* ``n_mean_<data set>.png`` — each data set's mean effective pressure, all
  on one colour scale, with ``n_stationary.png`` (the stationary field) on
  the same scale and ``dlogn_<data set>.png`` (each data set's departure
  from it, in ``log N``) when the run shared one; ``logn_amp_<campaign>.png`` and
  ``logn_peak_<campaign>.png`` — the 24 h amplitude of ``log N`` and the
  UTC hour of its maximum;
* with continuity, ``melt.png`` — the melt rate; ``ds_obs_<campaign>.png``,
  ``ds_model_<campaign>.png`` and ``ds_misfit_<campaign>.png`` — the radar's
  surface change since the lidar, continuity's, and their difference, on one
  colour scale;
* ``speed_model_<data set>.png`` — the model's surface speed over the
  whole domain, averaged over the data set's times;
* ``speed_obs_<year>.png``, ``speed_at_obs_<year>.png``,
  ``speed_fit_<year>.png`` and ``speed_misfit_<year>.png`` — ITS_LIVE's
  surface speed, the model's at the same pixels, one against the other,
  and each pixel's misfit ``|u - u_obs| / sigma``;
* ``los_obs_<campaign>.png``, ``los_model_<campaign>.png`` — the radar
  cells' LOS velocity (positive toward the radar) and the model's,
  averaged over the times; ``strain_fit_<campaign>.png`` — the strain
  rates against the model's, every pair and time; ``strain_misfit_<campaign>.png``
  — each cell's RMS normalized strain-rate residual over its pairs and
  times;
* ``misfit.png`` — each data set's misfit per observation against the
  iteration, from ``run.log``.

``--radar`` draws the radar's own surfaces (:mod:`baker.radar_dem`), into
``fig/radar/`` (or ``fig/<--radar-source>/`` for ``radar_single`` or
``radar_joint``, :mod:`baker.radar_joint`): ``radar_dh_<campaign>.png`` and
``radar_sd_<campaign>.png``, the surface minus the 2015 lidar and its sd on
the 50 m grid; ``radar_z_<campaign>.png``, the lidar plus that change, and
``radar_z_lidar.png``, the lidar over the same cells on the same scale; and
``radar_dh_elevation.png``, each campaign's median change on the ice by
elevation band.

``--docs DIR`` (with ``--run``) copies the figures ``docs/flow.md`` shows
into ``DIR`` (``docs/figures``) under their names there, once the radar
figures and the run's plots are drawn.

``--surface`` instead draws the surfaces the flow model is given
(:mod:`baker.surface`), into ``fig/surface/``:

* ``surface_2015.png`` — the 2015 lidar surface (the reference);
* ``dhdt.png``, ``dhdt_altimetry.png``, ``dhdt_aster.png``,
  ``dhdt_sd.png`` — the fitted rate of surface change from every source,
  from ICESat-2 and GEDI alone, from ASTER alone (when ``surface_altimetry.npz``
  and ``surface_aster.npz`` exist), and its formal sd;
* ``coverage.png`` — where the ICESat-2 and GEDI observations on the ice
  are, by year;
* ``dh_time.png`` — every source's median change since 2015 in each
  elevation band against its date, with the fitted rate's band median;
* ``ds_<data set>.png`` — the surface change since 2015 that each campaign
  and ITS_LIVE year is given.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ forward
def predict(folder, run=None):
    """The model's observations at the saved fields, every data set and time.

    The fields are read from ``run`` (default ``folder``), which
    :mod:`baker.run` wrote with the data sets, surface and friction recorded.
    """
    import firedrake
    from .inverse import Problem, SurfaceVelocities
    from .observe import pair_rates

    run = run or folder
    with firedrake.CheckpointFile(str(run / "fields.h5"), "r") as cf:
        mesh = cf.load_mesh()
        f = np.load(run / "fields.npz")
        source = str(f["surface_source"]) if "surface_source" in f.files else "dhdt"
        radar_dir = str(f["radar_dir"]) if "radar_dir" in f.files else ""
        friction = str(f["friction"]) if "friction" in f.files else "separate"
        continuity = bool(f["continuity"]) if "continuity" in f.files else False
        max_slope = float(f["continuity_max_slope"]) if "continuity_max_slope" in f.files else 90.0
        names = [k[:-len("_N_mean")] for k in f.files if k.endswith("_N_mean")]
        scenes = [n for n in names if not n.startswith("its_live_")]
        years = [int(n[len("its_live_"):]) for n in names if n.startswith("its_live_")]
        P = Problem.load(folder, mesh=mesh, surface=source, radar_dir=radar_dir or None,
                         friction=friction, scenes=scenes, years=years, continuity=continuity,
                         max_slope=max_slope)
        names = [f.name() for f in P.controls()]
        saved = [cf.load_function(mesh, n) for n in names]
    ctrl = [firedrake.Function(P.model.Q, name=n).assign(f) for n, f in zip(names, saved)]
    theta_H, qs = P.split(ctrl)
    hs = [firedrake.Function(P.model.Q).interpolate(P.thickness(theta_H, d)) for d in P.datasets]
    zs = P.solve_all(ctrl)
    g = np.load(folder / "grid.npz")
    X, Y = np.meshgrid(g["x"], g["y"])
    pts = np.column_stack([X.ravel(), Y.ravel()])
    out = {}
    for z, (i, k) in zip(zs, P.states()):
        d = P.datasets[i]
        N = firedrake.Function(P.model.Q).interpolate(P.effective_pressure(qs[i], d.hours[k]))
        us = P.model.surface_velocity(z, hs[i], N)
        speed = firedrake.Function(P.model.Q).interpolate(firedrake.sqrt(firedrake.inner(us, us)))
        on_grid = np.asarray(speed.at(pts, dont_raise=True), dtype=float).reshape(X.shape)
        out.setdefault(f"{d.name}_speed_grid", []).append(on_grid)
        if isinstance(d, SurfaceVelocities):
            U = firedrake.Function(d.points.W).interpolate(us)
            V = firedrake.Function(firedrake.VectorFunctionSpace(d.points.vom.input_ordering,
                                                                 "DG", 0)).interpolate(U)
            out[f"{d.name}_model"] = V.dat.data_ro.copy()
        else:
            v = d.points.los_velocity(us, P.surface(d)).dat.data_ro.copy()
            out.setdefault(f"{d.name}_model", []).append(pair_rates(v, d.pairs, d.dist))
            out.setdefault(f"{d.name}_los", []).append(v)
    out = {k: np.asarray(v) for k, v in out.items()}
    for k in [k for k in out if k.endswith("_speed_grid")]:
        out[k] = out[k].mean(axis=0)
    return out


# --------------------------------------------------------------------- maps
def _outlines(grid):
    import geopandas as gpd  # noqa: F401  (load_outlines needs it)
    from pyproj import Transformer
    from gpri_tools.glaciers import load_outlines
    x, y = grid["x"], grid["y"]
    to_ll = Transformer.from_crs(str(grid["crs"]), "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform([x[0], x[-1]], [y[0], y[-1]])
    o = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                      bbox=(min(lon), min(lat), max(lon), max(lat)))
    return o.to_crs(str(grid["crs"]))


def _map(path, grid, outlines, field, label, cmap="viridis", vmin=None, vmax=None,
         points=None, norm=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = grid["x"] / 1e3, grid["y"] / 1e3
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if points is None:
        im = ax.pcolormesh(x, y, field, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                           shading="auto")
    else:
        im = ax.scatter(points[:, 0] / 1e3, points[:, 1] / 1e3, c=field, s=6, marker="s",
                        cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    for geom in outlines.geometry:
        for poly in getattr(geom, "geoms", [geom]):
            ax.plot(*(np.asarray(poly.exterior.coords)[:, :2].T / 1e3), color="k", lw=0.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    fig.colorbar(im, ax=ax, shrink=0.85).set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(path, obs, model, label_obs, label_model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(obs, model, s=3, alpha=0.4, color="k", lw=0)
    lim = np.nanpercentile(np.abs(np.r_[obs, model]), 99.5) * 1.05
    lo = -lim if np.nanmin(np.r_[obs, model]) < 0 else 0.0
    ax.plot([lo, lim], [lo, lim], color="0.5", lw=0.8)
    ax.set_xlim(lo, lim)
    ax.set_ylim(lo, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(label_obs)
    ax.set_ylabel(label_model)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _misfit(path, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [line for line in log.read_text().splitlines() if line.startswith("iteration")]
    it = [int(r.split()[1]) for r in rows]
    series = {}
    for r in rows:
        tail = r.split("misfit/obs:")[1]
        for name, v in re.findall(r"(\S+) ([0-9.e+-]+)(?=\s|$)", tail):
            series.setdefault(name, []).append(float(v))
        mass = re.search(r"\(mass ([0-9.e+-]+)\)", tail)
        if mass:
            series.setdefault("continuity", []).append(float(mass.group(1)))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for name, v in series.items():
        ax.semilogy(it, v, lw=1.2, label=name,
                    ls="--" if name.startswith("its_live") else ":" if name == "continuity" else "-")
    ax.axhline(0.5, color="0.5", lw=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Misfit (per obs.)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def surface_figures(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from .data import campaign_epoch

    out = folder / "fig" / "surface"
    out.mkdir(parents=True, exist_ok=True)
    grid = np.load(folder / "grid.npz")
    sf = np.load(folder / "surface.npz")
    outlines = _outlines(grid)
    ice, z = grid["ice"], grid["surface"]
    t_ref = float(sf["t_ref"])

    _map(out / "surface_2015.png", grid, outlines, z, "Surface elevation (m)", cmap="terrain")
    rates = {"dhdt": sf["dhdt"]}
    for kind in ("altimetry", "aster"):
        if (folder / f"surface_{kind}.npz").exists():
            rates[f"dhdt_{kind}"] = np.load(folder / f"surface_{kind}.npz")["dhdt"]
    lim = max(np.nanpercentile(np.abs(np.concatenate([r[ice] for r in rates.values()])), 99), 0.1)
    for name, r in rates.items():
        _map(out / f"{name}.png", grid, outlines, np.where(ice, r, np.nan),
             "Elevation change (m/yr)", cmap="RdBu", norm=TwoSlopeNorm(0.0, -lim, lim))
    _map(out / "dhdt_sd.png", grid, outlines, np.where(ice, sf["dhdt_sd"], np.nan),
         "Rate sd (m/yr)", cmap="magma")

    ob = np.load(folder / "surface_points.npz")
    alt = ob["kept"] & (ob["kind"] != "aster")
    _map(out / "coverage.png", grid, outlines, np.floor(ob["t"][alt]), "Year", cmap="viridis",
         points=np.column_stack([ob["x"][alt], ob["y"][alt]]))

    # change since 2015 against date, per source and elevation band
    col = np.round((ob["x"] - grid["x"][0]) / (grid["x"][1] - grid["x"][0])).astype(int)
    row = np.round((ob["y"] - grid["y"][0]) / (grid["y"][1] - grid["y"][0])).astype(int)
    zo = z[row, col]
    bands = [(1500, 2000), (2000, 2500), (2500, 3000)]
    colours = plt.get_cmap("viridis")(np.linspace(0.1, 0.8, len(bands)))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    tt = np.linspace(t_ref, ob["t"].max() + 0.2, 2)
    for (lo, hi), c in zip(bands, colours):
        inb = ob["kept"] & (zo >= lo) & (zo < hi)
        for src in np.unique(ob["source"][inb]):
            m = inb & (ob["source"] == src)
            aster = ob["kind"][m][0] == "aster"
            if aster:
                ax.plot(np.median(ob["t"][m]), np.median(ob["dh"][m]), "s", color=c, ms=6,
                        mfc="none")
            else:                                          # one point per year of altimetry
                for yr in np.unique(np.floor(ob["t"][m])):
                    my = m & (np.floor(ob["t"]) == yr)
                    if my.sum() >= 10:
                        ax.plot(np.median(ob["t"][my]), np.median(ob["dh"][my]), "o", color=c,
                                ms=6, mfc=c if "atl06" in ob["kind"][my][0] else "none")
        mz = ice & (z >= lo) & (z < hi)
        ax.plot(tt, np.median(sf["dhdt"][mz]) * (tt - t_ref), color=c, lw=1.5,
                label=f"{lo}-{hi} m")
    ax.plot([t_ref], [0], "k*", ms=10, label="2015 lidar")
    ax.plot([], [], "ks", mfc="none", label="ASTER")
    ax.plot([], [], "ko", label="ICESat-2")
    ax.plot([], [], "ko", mfc="none", label="GEDI")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Surface change (m)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out / "dh_time.png", dpi=150)
    plt.close(fig)

    epochs = {}
    for path in sorted(folder.glob("*_obs.npz")):
        name = path.name[:-len("_obs.npz")]
        epochs[name] = campaign_epoch(name)
    for path in sorted(folder.glob("its_live_*.npz")):
        epochs[path.stem] = int(path.stem[-4:]) + 0.5
    ds = {k: np.where(ice, sf["dhdt"] * (t - t_ref), np.nan) for k, t in epochs.items()}
    dlim = max(np.nanpercentile(np.abs(np.concatenate([d[ice] for d in ds.values()])), 99), 0.1)
    for k, d in ds.items():
        _map(out / f"ds_{k}.png", grid, outlines, d, "Surface change (m)", cmap="RdBu",
             norm=TwoSlopeNorm(0.0, -dlim, dlim))
    print(f"-> {out} ({len(list(out.glob('*.png')))} figures)")
    return epochs


def radar_figures(folder, source=None):
    """The radar's surfaces from ``folder`` (or ``folder / source``, e.g. radar_joint)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = folder / "fig" / (source or "radar")
    out.mkdir(parents=True, exist_ok=True)
    grid = np.load(folder / "grid.npz")
    outlines = _outlines(grid)
    ice, z = grid["ice"], grid["surface"]
    files = sorted((folder / source if source else folder).glob("radar_dh_*.npz"))
    fields = {p.stem[len("radar_dh_"):]: np.load(p) for p in files}
    seen = np.any([np.isfinite(r["dh"]) for r in fields.values()], axis=0)
    zlim = np.nanpercentile(np.where(seen, z, np.nan), [1, 99])
    _map(out / "radar_z_lidar.png", grid, outlines, np.where(seen, z, np.nan),
         "Surface elevation (m)", cmap="terrain", vmin=zlim[0], vmax=zlim[1])
    for name, r in fields.items():
        _map(out / f"radar_dh_{name}.png", grid, outlines, r["dh"], "Height change (m)",
             cmap="RdBu", vmin=-20, vmax=20)
        _map(out / f"radar_z_{name}.png", grid, outlines, z + r["dh"], "Surface elevation (m)",
             cmap="terrain", vmin=zlim[0], vmax=zlim[1])
        _map(out / f"radar_sd_{name}.png", grid, outlines, r["sd"], "Height sd (m)",
             cmap="magma", vmin=0, vmax=np.nanpercentile(r["sd"], 98))
    edges = np.arange(1400, 3301, 100)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for name, r in fields.items():
        med = [np.nanmedian(r["dh"][ice & (z >= lo) & (z < lo + 100)])
               if np.isfinite(r["dh"][ice & (z >= lo) & (z < lo + 100)]).sum() >= 20 else np.nan
               for lo in edges[:-1]]
        ax.plot(np.asarray(med), edges[:-1] + 50, lw=1.4, marker="o", ms=3, label=name)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Height change (m)")
    ax.set_ylabel("Elevation (m)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out / "radar_dh_elevation.png", dpi=150)
    plt.close(fig)
    print(f"-> {out} ({len(list(out.glob('*.png')))} figures)")


def docs_figures(folder, run, dest):
    """Copy the figures ``docs/flow.md`` shows into ``dest``, under their names there.

    The radar's come from ``fig/radar_single`` and ``fig/radar_joint``
    (``--radar``), the inversion's from the run's ``fig/``; every one has to
    be drawn already.  Returns the copies.
    """
    import shutil

    radar, fig = folder / "fig", run / "fig"
    names = sorted(p.stem[len("radar_dh_"):] for p in (folder / "radar_single").glob("radar_dh_*.npz"))
    pairs = [(radar / "radar_single" / "radar_dh_elevation.png", "42_radar_dh_elevation.png"),
             (radar / "radar_joint" / "radar_dh_elevation.png", "42_radar_dh_elevation_joint.png"),
             (radar / "radar_single" / "radar_sd_20170803_full.png", "43_radar_sd_20170803_full.png")]
    pairs += [(radar / "radar_single" / f"radar_dh_{n}.png", f"43_radar_dh_{n}.png") for n in names]
    pairs += [(fig / f"{k}.png", f"{number}_flow_{k}.png") for number, k in
              (("44", "thickness"), ("44", "bed"), ("45", "n_stationary"), ("48", "misfit"))]
    pairs += [(p, f"46_flow_{p.name}") for p in sorted(fig.glob("dlogn_*.png"))]
    pairs += [(p, f"47_flow_{p.name}") for p in sorted(fig.glob("logn_*_*.png"))]
    missing = [str(p) for p, _ in pairs if not p.exists()] or (
        [] if names else [str(folder / "radar_single" / "radar_dh_*.npz")])
    if missing:
        raise SystemExit("draw these first: " + ", ".join(missing))
    dest.mkdir(parents=True, exist_ok=True)
    for p, name in pairs:
        shutil.copyfile(p, dest / name)
    return [dest / name for _, name in pairs]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", action="store_true",
                    help="run the model at the saved fields and cache its observations")
    ap.add_argument("--surface", action="store_true",
                    help="draw the surfaces the flow model is given (baker.surface)")
    ap.add_argument("--radar", action="store_true",
                    help="draw the radar's own surfaces (baker.radar_dem)")
    ap.add_argument("--radar-source", default=None,
                    help="with --radar: a subfolder holding the surfaces, e.g. radar_joint")
    ap.add_argument("--run", default="",
                    help="subfolder holding the inversion's fields (baker.run --out)")
    ap.add_argument("--docs", type=Path, default=None,
                    help="copy the figures docs/flow.md shows into this folder (docs/figures)")
    args = ap.parse_args()
    folder = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "baker"
    run = folder / args.run if args.run else folder
    if args.docs:
        written = docs_figures(folder, run, args.docs)
        print(f"-> {args.docs} ({len(written)} figures)")
        return
    if args.radar:
        radar_figures(folder, args.radar_source)
        return
    if args.surface:
        surface_figures(folder)
        return
    cache = run / "predictions.npz"
    if args.predict:
        np.savez_compressed(cache, **predict(folder, run))
        print(f"-> {cache}")
        return
    if not cache.exists() or cache.stat().st_mtime < (run / "fields.h5").stat().st_mtime:
        raise SystemExit(f"{cache} is missing or older than fields.h5: run with --predict first")
    plot(folder, dict(np.load(cache)), run)


def plot(folder, pred, run=None):
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    from .data import Observations, SurfaceVelocity

    run = run or folder
    out = run / "fig"
    out.mkdir(exist_ok=True)
    grid, f = np.load(folder / "grid.npz"), np.load(run / "fields.npz")
    outlines = _outlines(grid)
    ice = grid["ice"]

    prior = np.where(ice, grid["thickness"], np.nan)
    hmax = np.nanpercentile(np.r_[f["H"][ice], prior[ice]], 99)
    _map(out / "thickness.png", grid, outlines, f["H"], "Thickness (m)", vmin=0, vmax=hmax)
    _map(out / "thickness_prior.png", grid, outlines, prior, "Thickness (m)", vmin=0, vmax=hmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prior > 1, f["H"] / prior, np.nan)
    _map(out / "thickness_ratio.png", grid, outlines, ratio, "Thickness ratio", cmap="RdBu_r",
         norm=LogNorm(0.25, 4.0))
    _map(out / "bed.png", grid, outlines, f["bed"], "Bed elevation (m)", cmap="terrain")

    n_keys = [k for k in f.files if k.endswith("_N_mean")] + (
        ["N_stationary"] if "N_stationary" in f.files else [])
    n_lo, n_hi = np.nanpercentile(np.concatenate([f[k][ice] for k in n_keys]), [1, 99])
    if "N_stationary" in f.files:
        _map(out / "n_stationary.png", grid, outlines, f["N_stationary"],
             "Effective pressure (MPa)", cmap="RdBu",
             norm=TwoSlopeNorm(1.0, min(n_lo, 0.99), max(n_hi, 1.01)))
        n_keys.remove("N_stationary")
    d_keys = [k for k in f.files if k.endswith("_dlogN")]
    if d_keys:
        d_hi = max(np.nanpercentile(np.abs(np.concatenate([f[k][ice] for k in d_keys])), 99), 1e-3)
        for k in d_keys:
            _map(out / f"dlogn_{k[:-len('_dlogN')]}.png", grid, outlines, f[k], "Departure (log N)",
                 cmap="RdBu", vmin=-d_hi, vmax=d_hi)
    for k in n_keys:
        name = k[:-len("_N_mean")]
        _map(out / f"n_mean_{name}.png", grid, outlines, f[k], "Effective pressure (MPa)",
             cmap="RdBu", norm=TwoSlopeNorm(1.0, min(n_lo, 0.99), max(n_hi, 1.01)))
    amp_keys = [k for k in f.files if k.endswith("_logN_amp")]
    a_hi = max(np.nanpercentile(np.concatenate([f[k][ice] for k in amp_keys]), 99), 1e-3)
    for k in amp_keys:
        name = k[:-len("_logN_amp")]
        _map(out / f"logn_amp_{name}.png", grid, outlines, f[k], "Amplitude (log N)",
             cmap="magma", vmin=0, vmax=a_hi)
        peak = np.where(f[k] > 0.1 * a_hi, f[f"{name}_logN_peak"], np.nan)
        _map(out / f"logn_peak_{name}.png", grid, outlines, peak, "Peak time (UTC hr)",
             cmap="twilight", vmin=0, vmax=24)

    if "melt" in f.files:
        _map(out / "melt.png", grid, outlines, f["melt"], "Melt (m/yr)", cmap="viridis", vmin=0,
             vmax=np.nanpercentile(f["melt"][ice], 99))
    ds_keys = [k[:-len("_ds_obs")] for k in f.files if k.endswith("_ds_obs")]
    if ds_keys:
        ds_hi = np.nanpercentile(np.abs(np.concatenate(
            [f[f"{n}_{s}"][ice] for n in ds_keys for s in ("ds_obs", "ds_model")])), 99)
        for name in ds_keys:
            seen = np.isfinite(f[f"{name}_ds_obs"])
            for s, label in (("ds_obs", "Measured (m)"), ("ds_model", "Continuity (m)")):
                _map(out / f"{s}_{name}.png", grid, outlines, np.where(seen, f[f"{name}_{s}"], np.nan),
                     label, cmap="RdBu", vmin=-ds_hi, vmax=ds_hi)
            _map(out / f"ds_misfit_{name}.png", grid, outlines,
                 np.where(seen, f[f"{name}_ds_model"] - f[f"{name}_ds_obs"], np.nan),
                 "Misfit (m)", cmap="RdBu", vmin=-ds_hi, vmax=ds_hi)

    speed_keys = [k for k in pred if k.endswith("_speed_grid")]
    s_hi = np.nanpercentile(np.concatenate([pred[k][ice] for k in speed_keys]), 99.5)
    snorm = LogNorm(1.0, s_hi)
    for k in speed_keys:
        name = k[:-len("_speed_grid")]
        _map(out / f"speed_model_{name}.png", grid, outlines,
             np.where(ice, np.maximum(pred[k], 1.0), np.nan), "Surface speed (m/yr)", norm=snorm)

    for path in sorted(folder.glob("its_live_*.npz")):
        sv, _ = SurfaceVelocity.load(path)
        name, year = path.stem, int(sv.year)
        if f"{name}_model" not in pred:
            continue
        model = pred[f"{name}_model"]
        so, sm = np.hypot(sv.vx, sv.vy), np.hypot(model[:, 0], model[:, 1])
        _map(out / f"speed_obs_{year}.png", grid, outlines, np.maximum(so, 1.0),
             "Surface speed (m/yr)", points=sv.xy, norm=snorm)
        _map(out / f"speed_at_obs_{year}.png", grid, outlines, np.maximum(sm, 1.0),
             "Surface speed (m/yr)", points=sv.xy, norm=snorm)
        _scatter(out / f"speed_fit_{year}.png", so, sm,
                 "Observed speed (m/yr)", "Modelled speed (m/yr)")
        r = np.hypot(model[:, 0] - sv.vx, model[:, 1] - sv.vy) / sv.sigma
        _map(out / f"speed_misfit_{year}.png", grid, outlines, r, "Normalized misfit",
             cmap="magma", points=sv.xy, vmin=0, vmax=np.percentile(r, 99))

    los_hi = None
    for path in sorted(folder.glob("*_obs.npz")):
        o, _ = Observations.load(path)
        name = path.name[:-len("_obs.npz")]
        if f"{name}_model" not in pred:
            continue
        vo, vm = o.velocity.mean(axis=0), pred[f"{name}_los"].mean(axis=0)
        # the radar sees velocity relative to its stable ground; the model's has
        # no such offset, so both are shown about their own medians
        vo, vm = vo - np.median(vo), vm - np.median(vm)
        los_hi = los_hi or np.percentile(np.abs(np.r_[vo, vm]), 98)
        for tag, v in (("obs", vo), ("model", vm)):
            _map(out / f"los_{tag}_{name}.png", grid, outlines, v, "LOS velocity (m/yr)",
                 cmap="RdBu_r", points=o.xy, vmin=-los_hi, vmax=los_hi)
        model = pred[f"{name}_model"]
        _scatter(out / f"strain_fit_{name}.png", o.rate.ravel(), model.ravel(),
                 "Observed strain (1/yr)", "Modelled strain (1/yr)")
        r2 = ((model - o.rate) / o.sigma) ** 2
        tot, cnt = np.zeros(len(o.xy)), np.zeros(len(o.xy))
        for col in (0, 1):
            np.add.at(tot, o.pairs[:, col], r2.sum(axis=0))
            np.add.at(cnt, o.pairs[:, col], r2.shape[0])
        rms = np.sqrt(tot / np.maximum(cnt, 1))
        _map(out / f"strain_misfit_{name}.png", grid, outlines, rms, "Normalized misfit",
             cmap="magma", points=o.xy, vmin=0, vmax=max(np.percentile(rms, 99), 1.0))
    if (run / "run.log").exists():
        _misfit(out / "misfit.png", run / "run.log")
    print(f"-> {out} ({len(list(out.glob('*.png')))} figures)")


if __name__ == "__main__":
    main()
