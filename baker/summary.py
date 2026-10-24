#!/usr/bin/env python3
"""The numbers ``docs/flow.md`` quotes, from the radar surfaces and a run's fields.

    python -m baker.summary --run radar_friction     # any environment with numpy

Everything on the 50 m grid's RGI ice (``grid.npz``):

* each campaign's radar surface in ``radar_single/`` (:mod:`baker.radar_joint`):
  its effective baseline and phase sd, the ice cells with a value, ``dh``
  there (median, 5th and 95th percentile), its median formal sd, and the
  robust sd of ``dh`` on the ice-free cells at or above ``--stable-min-z``;
* the spread between campaigns in the same ice cell (the sd across
  campaigns, median over the cells with a value in every campaign) by
  elevation band, for ``radar_single/`` and ``radar_joint/``, across all
  campaigns and across the 2017 ones;
* the run's fields (:mod:`baker.run`): the thickness, its prior, the bed
  and the stationary effective pressure, on the ice and at the distinct
  grid cells nearest the cells the campaigns observe (each counted once); for each campaign its
  departure ``q0``, its mean ``N`` and the 24 h amplitude of ``log N`` at its
  own cells; and from ``run.log`` the objective and each campaign's misfit
  per observation at the first and the last iteration.

The held-out scores and the shared height term are printed by
:mod:`baker.radar_joint`.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

from .radar_joint import robust_sd

BANDS = ((1500.0, 2000.0), (2000.0, 2300.0), (2300.0, 2600.0), (2600.0, 3300.0))


def percentiles(values, q=(5, 50, 95)):
    """Percentiles of the finite values (NaN when there are none)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return np.percentile(v, q) if v.size else np.full(len(q), np.nan)


def spread(dh, z, ice, groups, bands=BANDS):
    """The sd across campaigns in each ice cell with a value in every campaign of ``dh``.

    ``dh`` is (campaigns, rows, cols) and ``z`` each cell's elevation;
    ``groups`` maps a label to the campaigns (indices into ``dh``) whose sd
    is taken.  Returns the number of those cells and, for each band
    ``[lo, hi)``, its cells and each group's median sd.
    """
    ok = ice & np.isfinite(dh).all(axis=0)
    sd = {g: np.std(dh[list(k)], axis=0) for g, k in groups.items()}
    rows = []
    for lo, hi in bands:
        m = ok & (z >= lo) & (z < hi)
        rows.append((lo, hi, int(m.sum()),
                     {g: float(np.median(s[m])) if m.any() else np.nan for g, s in sd.items()}))
    return int(ok.sum()), rows


def nearest(x, y, xy):
    """The row and column indices ``(j, i)`` of the grid cell nearest each point of ``xy``."""
    i = np.abs(x[None, :] - xy[:, :1]).argmin(axis=1)
    j = np.abs(y[None, :] - xy[:, 1:]).argmin(axis=1)
    return j, i


def at_cells(field, x, y, xy):
    """``field`` (rows along ``y``) at the grid cell nearest each point of ``xy``."""
    return field[nearest(x, y, xy)]


def distinct_cells(x, y, xys):
    """The distinct grid cells ``(j, i)`` nearest the points of every array in ``xys``, each once."""
    j, i = nearest(x, y, np.vstack(list(xys)))
    jc, ic = np.unique(np.stack([j, i], axis=1), axis=0).T
    return jc, ic


def read_log(log):
    """``run.log``'s iterations: their numbers, the objective, and each data set's misfit per observation."""
    its, J, misfit = [], [], {}
    for row in Path(log).read_text().splitlines():
        if not row.startswith("iteration"):
            continue
        head, tail = row.split("misfit/obs:")
        its.append(int(head.split()[1]))
        J.append(float(head.split()[3]))
        for name, v in re.findall(r"(\S+) ([0-9.e+-]+)(?=\s|$)", tail):
            misfit.setdefault(name, []).append(float(v))
        mass = re.search(r"\(mass ([0-9.e+-]+)\)", tail)
        if mass:
            misfit.setdefault("mass", []).append(float(mass.group(1)))
    return np.array(its), np.array(J), {k: np.array(v) for k, v in misfit.items()}


def _q(p, fmt="{:+.1f}"):
    return f"{fmt.format(p[1])} ({fmt.format(p[0])} to {fmt.format(p[2])})"


def surfaces(folder, grid, stable_min_z):
    ice, z = grid["ice"], grid["surface"]
    print("radar surfaces, radar_single/: baseline (m), phase sd (rad), ice cells, dh on the ice "
          "median (5th to 95th) (m), median formal sd on the ice (m), ice-free cells robust sd (m)")
    for p in sorted((folder / "radar_single").glob("radar_dh_*.npz")):
        r = np.load(p)
        dh, sd = r["dh"], r["sd"]
        on = ice & np.isfinite(dh)
        stable = ~ice & np.isfinite(dh) & (z >= stable_min_z)
        print(f"  {p.stem[len('radar_dh_'):]:15s} {float(r['baseline']):+.3f}  "
              f"{float(r['phase_sd']):.2f}  {on.sum():5d}  {_q(percentiles(dh[on]))}  "
              f"{np.median(sd[on]):.2f}  {robust_sd(dh[stable]):.2f}")


def spreads(folder, grid):
    for source in ("radar_single", "radar_joint"):
        files = sorted((folder / source).glob("radar_dh_*.npz"))
        names = [p.stem[len("radar_dh_"):] for p in files]
        groups = {"all": range(len(names)),
                  "2017": [k for k, n in enumerate(names) if n.startswith("2017")]}
        n, rows = spread(np.stack([np.load(p)["dh"] for p in files]), grid["surface"], grid["ice"],
                         groups)
        print(f"spread between campaigns, {source}/: {n} ice cells with a value in all "
              f"{len(names)}; median sd (m) across all / across the 2017 ones")
        for lo, hi, cells, med in rows:
            print(f"  {lo:.0f}-{hi:.0f} m  cells {cells:5d}  {med['all']:.1f} / {med['2017']:.1f}")


def fields(folder, run, grid):
    from .data import Observations

    f = np.load(run / "fields.npz")
    x, y, ice = f["x"], f["y"], grid["ice"]
    prior = np.where(ice, grid["thickness"], np.nan)
    cells = {}
    for p in sorted(folder.glob("*_obs.npz")):
        name = p.name[:-len("_obs.npz")]
        if f"{name}_N_mean" in f.files:
            cells[name] = Observations.load(p)[0].xy
    j, i = distinct_cells(x, y, cells.values())
    on = ice[j, i]
    j, i, n_cells = j[on], i[on], len(on)
    H, Hc, Pc = f["H"], f["H"][j, i], prior[j, i]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio, ratio_c = np.where(prior > 1, H / prior, np.nan), np.where(Pc > 1, Hc / Pc, np.nan)
    print(f"thickness (m) 5/50/95: on the ice {_q(percentiles(H[ice]), '{:.0f}')}, prior "
          f"{_q(percentiles(prior[ice]), '{:.0f}')}, median ratio {np.nanmedian(ratio[ice]):.2f}")
    print(f"  at the {len(j)} distinct grid cells on the ice nearest the observed cells "
          f"(of {n_cells}, from {sum(len(c) for c in cells.values())} campaign-cells): "
          f"{_q(percentiles(Hc), '{:.0f}')}, prior {_q(percentiles(Pc), '{:.0f}')}, median ratio "
          f"{np.nanmedian(ratio_c):.2f}")
    print(f"bed (m) 5/50/95 on the ice: {_q(percentiles(f['bed'][ice]), '{:.0f}')}")
    if "N_stationary" in f.files:
        print(f"stationary N (MPa): on the ice {_q(percentiles(f['N_stationary'][ice]), '{:.3f}')}, "
              f"at those cells {_q(percentiles(f['N_stationary'][j, i]), '{:.3f}')}")
    its, J, misfit = read_log(run / "run.log")
    tail = f"; over the last ten iterations {J[-1] - J[-11]:+.1f}" if len(J) > 10 else ""
    print(f"objective: iteration {its[0]} {J[0]:.1f}, iteration {its[-1]} {J[-1]:.1f}{tail}")
    print("per campaign at its own cells: cells on the ice, q0, mean N (MPa), 24 h amplitude of "
          "log N median / 95th, misfit per observation first -> last")
    for name, c in cells.items():
        q0 = at_cells(f[f"{name}_dlogN"], x, y, c) if f"{name}_dlogN" in f.files else np.full(len(c), np.nan)
        line = (f"  {name:15s} {np.isfinite(at_cells(f['H'], x, y, c)).sum():4d}  "
                f"{_q(percentiles(q0), '{:+.3f}')}  "
                f"{_q(percentiles(at_cells(f[f'{name}_N_mean'], x, y, c)), '{:.2f}')}")
        if f"{name}_logN_amp" in f.files:
            a = percentiles(at_cells(f[f"{name}_logN_amp"], x, y, c), (50, 95))
            line += f"  {a[0]:.3f} / {a[1]:.3f}"
        if name in misfit:
            line += f"  {misfit[name][0]:.3g} -> {misfit[name][-1]:.3g}"
        print(line)
    if "melt" in f.files:
        print(f"melt (m/yr of ice): on the ice {_q(percentiles(f['melt'][ice]), '{:.2f}')}, "
              f"at those cells {_q(percentiles(f['melt'][j, i]), '{:.2f}')}")
        if "mass" in misfit:
            print(f"continuity misfit per mesh cell of observed change: "
                  f"{misfit['mass'][0]:.3g} -> {misfit['mass'][-1]:.3g}")
        print("surface change since the lidar (m) where the radar has one: measured, flux "
              "divergence alone, with the melt, model minus measured (5/50/95)")
        for name in cells:
            if f"{name}_ds_obs" not in f.files:
                continue
            obs, seen = f[f"{name}_ds_obs"], np.isfinite(f[f"{name}_ds_obs"])
            print(f"  {name:15s} {seen.sum():5d}  {_q(percentiles(obs[seen]))}  "
                  f"{_q(percentiles(f[f'{name}_ds_flux'][seen]))}  "
                  f"{_q(percentiles(f[f'{name}_ds_model'][seen]))}  "
                  f"{_q(percentiles(f[f'{name}_ds_model'][seen] - obs[seen]))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="", help="subfolder holding the inversion's fields (baker.run --out)")
    ap.add_argument("--stable-min-z", type=float, default=1700.0,
                    help="ice-free cells this high and above score the surfaces, m")
    args = ap.parse_args()
    folder = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "baker"
    grid = np.load(folder / "grid.npz")
    surfaces(folder, grid, args.stable_min_z)
    spreads(folder, grid)
    fields(folder, folder / args.run if args.run else folder, grid)


if __name__ == "__main__":
    main()
