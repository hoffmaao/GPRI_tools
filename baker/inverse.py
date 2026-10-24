"""One bed under every data set; an effective pressure for each.

The data are the radar campaigns' LOS strain rates (:mod:`baker.data`,
``<scene>_obs.npz``) and ITS_LIVE's annual surface velocities
(``its_live_<year>.npz``), all on one mesh over the north-side glaciers.
The unknowns, each a continuous piecewise-linear field on the mesh, are

* ``theta_H`` — **shared by every data set**: the thickness under the 2015
  lidar surface, ``H_2015 = H_prior exp(theta_H)``, and so one bed
  ``b = s_2015 - H_2015`` across all the glaciers and all the time
  periods.  Each data set sees its own surface ``s_k = s_2015 + ds_k``,
  by default ``ds_k`` from the campaign's own two-antenna height change
  (:mod:`baker.radar_dem`; see :meth:`Problem.load` for the others), and so
  the thickness ``H_k = max(H_2015 + ds_k, 0)`` over that same bed.  ``H`` stays zero wherever the prior is zero (off the ice);
* ``q_bar`` — **shared by every data set** (with ``friction="shared"``, the
  default): the stationary part of log effective pressure, and so of the
  Coulomb strength ``mu N``;
* for **each** data set its own departure from it: a campaign with 24 h
  terms has ``N(t) = exp(q_bar + q0 + qc cos wt + qs sin wt)`` (MPa, ``t``
  the UTC hour); a campaign shorter than a day, and each ITS_LIVE year, a
  constant ``N = exp(q_bar + q0)``.  With ``friction="separate"`` there is
  no ``q_bar`` and each ``q0`` is that data set's whole mean;
* ``theta_m`` — with ``continuity=True``, the melt rate
  ``m = m_0 exp(theta_m)`` (m/yr of ice), shared by every campaign.  It is
  positive everywhere, so the surface mass balance is melt only and never
  accumulation.

**Mass continuity.**  With ``continuity=True`` each radar campaign's
measured surface change since the 2015 lidar is also an observation, of

    ds_k = dt_k ( -div(H_k u_k) - m ),

the change that ice continuity gives over the ``dt_k`` years since the lidar
(:data:`T_LIDAR`) with the flux of that campaign's own thickness and
depth-averaged velocity (``u_k``, the mean over its times of day) held
through the interval, and the melt ``m`` as the only surface source.  Its
misfit, ``1/2 int w_k (ds_k - ds_obs)^2 dx / spacing^2`` with ``w_k`` the
inverse variance of the radar's change, counts each mesh cell's area of
observed ice once.  Only terrain less steep than ``max_slope`` (30 degrees,
on the lidar) enters, which leaves out the steep summit, seen at grazing
angles; the melt there is its prior.

For every data set and time of day the momentum balance
(:mod:`baker.physics`) is solved with that data set's surface, thickness
and ``N``, and compared with the observations, weighted by their standard
deviations.  The objective adds

    R = 1/2 int [ l_H^2 |grad theta_H|^2 + theta_H^2 / s_H^2
                  + l_N^2 |grad q_bar|^2
                  + sum over data sets ( l_N^2 sum_q |grad q|^2 + q0^2 / s_0^2
                                         + (qc^2 + qs^2) / s_q^2 ) ] * weight * n_obs / area

(the ``q0^2 / s_0^2`` term only with a shared ``q_bar``), and with
continuity ``l_m^2 |grad theta_m|^2 + theta_m^2 / s_m^2``.

``n_obs`` the number of observations assimilated, so ``weight = 1`` gives
the regularization's mean density the weight of the misfit's mean per
observation.  scipy's L-BFGS-B minimises the sum, with the gradient from
pyadjoint.  Every evaluation first solves each time off the tape, from the
last solution (:meth:`baker.physics.ForwardModel.solve` continues the
solution when a step fails), then tapes one Newton solve from there, so a
step the optimizer tries into a region with no solution returns a large
value and is backtracked rather than ending the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import firedrake
from firedrake import Constant, Function, assemble, dx, exp, grad, inner
from firedrake.adjoint import (Control, compute_gradient, continue_annotation,
                               get_working_tape, pause_annotation)

from .data import campaign_epoch
from .observe import CellPoints, VelocityPoints, pair_misfit
from .physics import ForwardModel, Parameters
from .surface import T_LIDAR

DIURNAL_W = 2 * np.pi / 24.0


@dataclass
class Regularization:
    l_H: float = 500.0            # m, smoothing length of log thickness
    s_H: float = 0.5              # prior spread of log thickness (about 50 %)
    l_N: float = 1000.0           # m, smoothing length of the log N fields
    s_q: float = 1.0              # prior spread of the 24 h terms of log N
    s_0: float = 0.5              # prior spread of each data set's mean log N about q_bar
    m_0: float = 1.0              # m/yr of ice, prior melt rate
    l_m: float = 1000.0           # m, smoothing length of log melt
    s_m: float = 1.0              # prior spread of log melt
    weight: float = 1.0           # R per unit area, against the misfit per observation


def rectangle_mesh(bounds, spacing):
    x0, y0, x1, y1 = bounds
    nx, ny = int(np.ceil((x1 - x0) / spacing)), int(np.ceil((y1 - y0) / spacing))
    mesh = firedrake.RectangleMesh(nx, ny, x1 - x0, y1 - y0)
    mesh.coordinates.dat.data[:] += np.array([x0, y0])
    return mesh


def grid_to_mesh(Q, x, y, values, fill=0.0, smooth=0.0):
    """A gridded field (rows along y) on the CG1 space ``Q``, bilinearly.

    ``smooth`` (m) is the width of a box average applied first, so a field
    gridded finer than the mesh is not aliased onto it.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import uniform_filter
    values = np.where(np.isfinite(values), values, fill)
    width = int(round(smooth / (x[1] - x[0])))
    if width > 1:
        values = uniform_filter(values, width, mode="nearest")
    f = RegularGridInterpolator((y, x), values, bounds_error=False, fill_value=fill)
    X = Function(firedrake.VectorFunctionSpace(Q.mesh(), Q.ufl_element()))
    X.interpolate(firedrake.SpatialCoordinate(Q.mesh()))
    out = Function(Q)
    out.dat.data[:] = f(X.dat.data_ro[:, ::-1])
    return out


# ------------------------------------------------------------------ data sets
@dataclass
class RadarStrainRates:
    """One campaign's LOS strain rates; ``hours`` is ``[nan]`` for a mean-rate-only record."""
    name: str
    points: CellPoints
    hours: np.ndarray
    pairs: np.ndarray
    dist: np.ndarray
    rate: np.ndarray              # (nt, np)
    sigma: np.ndarray             # (nt, np)
    epoch: float = np.nan         # decimal year
    ds: Function | None = None    # surface change since 2015, m
    surface: Function | None = None
    ds_obs: Function | None = None  # the measured change where observed, for continuity
    ds_w: Function | None = None    # its inverse variance, 1/m^2 (zero where not observed)
    n_mass: float = 0.0             # mesh cells' worth of observed change

    @property
    def diurnal(self):
        return not np.isnan(self.hours[0])

    @property
    def n_obs(self):
        return self.rate.size + self.n_mass

    def misfit(self, u_surface, surface, k):
        v = self.points.los_velocity(u_surface, surface)
        return pair_misfit(v, self.pairs, self.dist, self.rate[k], self.sigma[k])


@dataclass
class SurfaceVelocities:
    """A year of ITS_LIVE surface velocity: one annual-mean state."""
    name: str
    points: VelocityPoints
    hours: np.ndarray = field(default_factory=lambda: np.array([np.nan]))
    diurnal: bool = False
    epoch: float = np.nan
    ds: Function | None = None
    surface: Function | None = None
    ds_obs: Function | None = None
    ds_w: Function | None = None
    n_mass: float = 0.0

    @property
    def n_obs(self):
        return self.points.size

    def misfit(self, u_surface, surface, k):
        return self.points.misfit(u_surface)


# ------------------------------------------------------------------- problem
@dataclass
class Problem:
    model: ForwardModel
    h_prior: Function
    datasets: list
    reg: Regularization = field(default_factory=Regularization)
    shared_friction: bool = True
    continuity: bool = False
    spacing: float = 150.0

    @classmethod
    def load(cls, folder, spacing=150.0, scenes=None, years=None, params=None, reg=None,
             mesh=None, surface="radar", radar_dir=None, friction="shared", continuity=False,
             max_slope=30.0):
        """From the files :mod:`baker.prepare` wrote to ``folder``.

        ``scenes`` and ``years`` pick campaigns and ITS_LIVE years (default:
        all there); pass an empty list to leave one kind out.  ``mesh``
        replaces the rectangle of ``spacing`` (e.g. one read back from a
        checkpoint).  ``surface`` is where each data set's surface comes from:
        ``"radar"``, the lidar plus that campaign's own height correction from
        the two antennas (``radar_dh_<scene>.npz``, :mod:`baker.radar_dem`; an
        ITS_LIVE year takes the mean of its year's campaigns, and a data set
        without one keeps the lidar); ``"dhdt"``, the lidar plus the rate of
        :mod:`baker.surface` times the time since 2015; ``"lidar"``, the lidar.
        ``radar_dir`` is the folder under ``folder`` holding the radar's
        ``radar_dh_<scene>.npz`` (e.g. ``radar_single``, :mod:`baker.radar_joint`).
        ``friction`` is ``"shared"`` (a stationary ``q_bar`` under every data
        set) or ``"separate"``.  ``continuity`` adds the melt rate and each
        radar campaign's measured change as an observation of ice continuity
        (needs ``surface="radar"``), only where the 2015 lidar surface is less
        steep than ``max_slope`` (degrees).
        """
        from .data import Observations, SurfaceVelocity
        folder = Path(folder)
        g = np.load(folder / "grid.npz")
        x, y = g["x"], g["y"]
        if continuity and surface != "radar":
            raise ValueError("continuity needs the radar's measured surface change")
        if mesh is None:
            mesh = rectangle_mesh((x[0], y[0], x[-1], y[-1]), spacing)
        Q = firedrake.FunctionSpace(mesh, "CG", 1)
        s = grid_to_mesh(Q, x, y, g["surface"], fill=np.nanmedian(g["surface"]), smooth=spacing)
        h = grid_to_mesh(Q, x, y, np.where(g["ice"], g["thickness"], 0.0), smooth=spacing)
        model = ForwardModel(mesh, s, params or Parameters())
        datasets = []
        for path in sorted(folder.glob("*_obs.npz")):
            name = path.name[:-len("_obs.npz")]
            if scenes is not None and name not in scenes:
                continue
            o, _ = Observations.load(path)
            datasets.append(RadarStrainRates(name, CellPoints(mesh, o.xy, o.los), o.hours,
                                             o.pairs, o.dist, o.rate, o.sigma))
        for path in sorted(folder.glob("its_live_*.npz")):
            sv, _ = SurfaceVelocity.load(path)
            if years is not None and int(sv.year) not in years:
                continue
            datasets.append(SurfaceVelocities(f"its_live_{int(sv.year)}",
                                              VelocityPoints(mesh, sv.xy, sv.vx, sv.vy, sv.sigma),
                                              epoch=int(sv.year) + 0.5))
        for d in datasets:
            if isinstance(d, RadarStrainRates):
                d.epoch = campaign_epoch(d.name)
        changes = {}
        if surface == "dhdt":
            if not (folder / "surface.npz").exists():
                raise FileNotFoundError(f"--surface dhdt needs {folder / 'surface.npz'}")
            sf = np.load(folder / "surface.npz")
            for d in datasets:
                changes[d.name] = np.where(g["ice"], sf["dhdt"], 0.0) * (d.epoch - float(sf["t_ref"]))
        elif surface == "radar":
            gy, gx = np.gradient(g["surface"], y, x)
            slope = np.degrees(np.arctan(np.hypot(gx, gy)))
            radar = {}
            where = folder / radar_dir if radar_dir else folder
            for p in where.glob("radar_dh_*.npz"):
                r = np.load(p)
                radar[p.stem[len("radar_dh_"):]] = (np.where(g["ice"], r["dh"], np.nan), r["sd"])
            if not radar:
                raise FileNotFoundError(f"--surface radar found no radar_dh_*.npz in {where}")
            for d in datasets:
                if d.name in radar:
                    changes[d.name] = np.nan_to_num(radar[d.name][0])
                    if continuity:
                        dh, sd = radar[d.name]
                        seen = np.isfinite(dh) & (slope < max_slope)
                        w = np.where(seen, 1.0 / np.maximum(np.nan_to_num(sd, nan=1.0), 0.1) ** 2, 0.0)
                        # box averages over the mesh spacing; a node counts where most of
                        # its box is observed, with the mean weight and the weighted mean
                        num = grid_to_mesh(Q, x, y, np.where(seen, dh, 0.0) * w, smooth=spacing)
                        den = grid_to_mesh(Q, x, y, w, smooth=spacing)
                        cov = grid_to_mesh(Q, x, y, seen.astype(float), smooth=spacing)
                        kept = cov.dat.data_ro > 0.5
                        d.ds_w = Function(Q, name=f"{d.name}_surface_change_weight")
                        d.ds_w.dat.data[kept] = den.dat.data_ro[kept] / cov.dat.data_ro[kept]
                        d.ds_obs = Function(Q, name=f"{d.name}_surface_change_observed")
                        d.ds_obs.dat.data[kept] = num.dat.data_ro[kept] / den.dat.data_ro[kept]
                        d.n_mass = float(seen.sum()) * (x[1] - x[0]) * (y[1] - y[0]) / spacing ** 2
                elif d.name.startswith("its_live_"):
                    same = [v for k, v in radar.items() if k.startswith(d.name[-4:])]
                    if same:
                        w = [np.where(np.isfinite(v[0]), 1.0 / np.maximum(v[1], 0.1) ** 2, 0.0)
                             for v in same]
                        num = sum(np.nan_to_num(v[0]) * wi for v, wi in zip(same, w))
                        den = sum(w)
                        changes[d.name] = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
        elif surface != "lidar":
            raise ValueError(f"unknown surface source {surface!r}")
        print(f"surface change ({surface}): "
              + (", ".join(d.name for d in datasets if d.name in changes) or "none"), flush=True)
        for d in datasets:
            if d.name in changes:
                d.ds = Function(Q, name=f"{d.name}_surface_change").assign(
                    grid_to_mesh(Q, x, y, changes[d.name], smooth=spacing))
                d.surface = Function(Q, name=f"{d.name}_surface").interpolate(s + d.ds)
        if friction not in ("shared", "separate"):
            raise ValueError(f"unknown friction {friction!r}")
        return cls(model, h, datasets, reg or Regularization(), friction == "shared",
                   continuity, spacing)

    @property
    def n_obs(self):
        return sum(d.n_obs for d in self.datasets)

    # -------------------------------------------------------------- controls
    def controls(self, N0=1.0):
        """``[theta_H, (q_bar,) then per data set q0 (, qc, qs)]``, at the prior."""
        Q = self.model.Q
        ctrl = [Function(Q, name="log_thickness_ratio")]
        if self.shared_friction:
            ctrl.append(Function(Q, name="log_N_stationary").assign(np.log(N0)))
        for d in self.datasets:
            q0 = Function(Q, name=f"{d.name}_log_N_mean")
            ctrl.append(q0 if self.shared_friction else q0.assign(np.log(N0)))
            if d.diurnal:
                ctrl += [Function(Q, name=f"{d.name}_log_N_cos"),
                         Function(Q, name=f"{d.name}_log_N_sin")]
        if self.continuity:
            ctrl.append(Function(Q, name="log_melt_ratio"))
        return ctrl

    def _friction_ctrl(self, ctrl):
        """The controls of ``log N``: all but ``theta_H`` and, with continuity, ``theta_m``."""
        return list(ctrl[1:-1] if self.continuity else ctrl[1:])

    def melt(self, ctrl):
        """The melt rate ``m_0 exp(theta_m)`` (m/yr of ice); zero without continuity."""
        if not self.continuity:
            return Constant(0.0)
        return Constant(self.reg.m_0) * exp(ctrl[-1])

    @staticmethod
    def depth_averaged(zs):
        """The mean over a data set's times of the depth-averaged velocity."""
        return sum(firedrake.split(z)[0] for z in zs) / len(zs)

    def flux_change(self, d, H, u):
        """``dt_k (-div(H u))``, the change continuity gives without melt, m."""
        return Constant(d.epoch - T_LIDAR) * -firedrake.div(H * u)

    def mass_misfit(self, d, H, u, melt):
        """``1/2 int w (ds_model - ds_obs)^2 dx / spacing^2``, assembled."""
        r = self.flux_change(d, H, u) - Constant(d.epoch - T_LIDAR) * melt - d.ds_obs
        return assemble(Constant(0.5 / self.spacing ** 2) * d.ds_w * r ** 2 * dx(self.model.mesh))

    def split(self, ctrl):
        """``(theta_H, [q terms of each data set])``; with a shared ``q_bar``
        each data set's first term is ``q_bar + q0``."""
        theta_H, rest, out = ctrl[0], self._friction_ctrl(ctrl), []
        if self.shared_friction:
            q_bar, rest = rest[0], rest[1:]
        for d in self.datasets:
            k = 3 if d.diurnal else 1
            q = rest[:k]
            out.append([q_bar + q[0], *q[1:]] if self.shared_friction else q)
            rest = rest[k:]
        return theta_H, out

    def friction_terms(self, ctrl):
        """``(q_bar or None, [the data sets' own terms q0 (, qc, qs)])``."""
        rest, out = self._friction_ctrl(ctrl), []
        q_bar = None
        if self.shared_friction:
            q_bar, rest = rest[0], rest[1:]
        for d in self.datasets:
            k = 3 if d.diurnal else 1
            out.append(rest[:k])
            rest = rest[k:]
        return q_bar, out

    def thickness(self, theta_H, d=None):
        """``H_2015``, or with a data set ``d`` its thickness at its own date."""
        H = self.h_prior * exp(theta_H)
        if d is None or d.ds is None:
            return H
        return firedrake.max_value(H + d.ds, 0.0)

    def surface(self, d):
        return self.model.s if d.surface is None else d.surface

    @staticmethod
    def effective_pressure(q, hour):
        if len(q) == 1:
            return exp(q[0])
        return exp(q[0] + q[1] * Constant(np.cos(DIURNAL_W * hour))
                   + q[2] * Constant(np.sin(DIURNAL_W * hour)))

    def states(self):
        """``(data set index, time index)`` of every momentum balance solved."""
        return [(i, k) for i, d in enumerate(self.datasets) for k in range(len(d.hours))]

    # ------------------------------------------------------------ forward run
    def solve_all(self, ctrl, guesses=None):
        """Solve every state; returns the solutions, in :meth:`states` order."""
        theta_H, qs = self.split(ctrl)
        hs = [Function(self.model.Q).interpolate(self.thickness(theta_H, d)) for d in self.datasets]
        zs = []
        for j, (i, k) in enumerate(self.states()):
            d = self.datasets[i]
            N = Function(self.model.Q).interpolate(self.effective_pressure(qs[i], d.hours[k]))
            z = None if guesses is None else guesses[j]
            zs.append(self.model.solve(hs[i], N, z=z, continuation=guesses is None,
                                       s=self.surface(d)))
        return zs

    def regularization(self, ctrl):
        theta_H = ctrl[0]
        q_bar, qs = self.friction_terms(ctrl)
        r = self.reg
        area = assemble(Constant(1.0) * dx(self.model.mesh))
        R = (Constant(r.l_H ** 2) * inner(grad(theta_H), grad(theta_H))
             + theta_H ** 2 / Constant(r.s_H ** 2))
        if q_bar is not None:
            R = R + Constant(r.l_N ** 2) * inner(grad(q_bar), grad(q_bar))
        for q in qs:
            R = R + Constant(r.l_N ** 2) * sum(inner(grad(f), grad(f)) for f in q)
            if q_bar is not None:
                R = R + q[0] ** 2 / Constant(r.s_0 ** 2)
            if len(q) == 3:
                R = R + (q[1] ** 2 + q[2] ** 2) / Constant(r.s_q ** 2)
        if self.continuity:
            theta_m = ctrl[-1]
            R = R + (Constant(r.l_m ** 2) * inner(grad(theta_m), grad(theta_m))
                     + theta_m ** 2 / Constant(r.s_m ** 2))
        return Constant(0.5 * r.weight * self.n_obs / area) * R * dx

    # ------------------------------------------------------------- inversion
    def objective(self, ctrl, guesses):
        """Taped: the misfits of every data set and time, plus the regularization.

        ``guesses`` are converged solutions near ``ctrl``; each is overwritten.
        Returns ``(J, misfit of each data set)``; with continuity a data set's
        misfit includes its mass term, and one more entry is the sum of those.
        """
        theta_H, qs = self.split(ctrl)
        hs = [Function(self.model.Q).interpolate(self.thickness(theta_H, d)) for d in self.datasets]
        J = assemble(self.regularization(ctrl))
        parts = [0.0] * len(self.datasets)
        solved = [[] for _ in self.datasets]
        for j, (i, k) in enumerate(self.states()):
            d = self.datasets[i]
            N = Function(self.model.Q).interpolate(self.effective_pressure(qs[i], d.hours[k]))
            z = self.model.solve_from(hs[i], N, guesses[j], s=self.surface(d))
            solved[i].append(z)
            m = d.misfit(self.model.surface_velocity(z, hs[i], N), self.surface(d), k)
            parts[i] += float(m)
            J = J + m
        if self.continuity:
            melt, mass = self.melt(ctrl), 0.0
            for i, d in enumerate(self.datasets):
                if d.ds_w is None:
                    continue
                m = self.mass_misfit(d, hs[i], self.depth_averaged(solved[i]), melt)
                parts[i] += float(m)
                mass += float(m)
                J = J + m
            parts.append(mass)
        return J, parts

    def invert(self, ctrl=None, maxiter=50, log=print):
        """Minimise the objective from ``ctrl`` (the prior by default); returns the controls."""
        from scipy.optimize import minimize as scipy_minimize
        ctrl = ctrl or self.controls()
        cuts = np.cumsum([c.dat.data_ro.size for c in ctrl])[:-1]
        pause_annotation()
        guesses = self.solve_all(ctrl)
        last = {"J": None, "g": None, "parts": None}

        def assign(x):
            for c, part in zip(ctrl, np.split(x, cuts)):
                c.dat.data[:] = part

        def evaluate(x):
            assign(x)
            trial = [g.copy(deepcopy=True) for g in guesses]
            try:
                self.solve_all(ctrl, trial)
                get_working_tape().clear_tape()
                continue_annotation()
                J, parts = self.objective(ctrl, trial)
                pause_annotation()
                grads = compute_gradient(J, [Control(c) for c in ctrl])
            except firedrake.ConvergenceError:
                pause_annotation()
                get_working_tape().clear_tape()
                if last["J"] is None:
                    raise
                log("  no solution at this step; backtracking")
                return 10.0 * abs(last["J"]), last["g"]
            get_working_tape().clear_tape()
            for g, t in zip(guesses, trial):
                g.assign(t)
            last.update(J=float(J), g=np.concatenate([g.dat.data_ro for g in grads]),
                        parts=parts)
            return last["J"], last["g"].copy()

        state = {"it": 0}

        def report(x=None):
            terms = "  ".join(f"{d.name} {p / d.n_obs:.3g}"
                              for d, p in zip(self.datasets, last["parts"]))
            if self.continuity:
                n_mass = sum(d.n_mass for d in self.datasets)
                terms += f"  (mass {last['parts'][-1] / max(n_mass, 1):.3g})"
            log(f"iteration {state['it']:3d}  J {last['J']:.6g}  misfit/obs: {terms}")
            state["it"] += 1

        x0 = np.concatenate([c.dat.data_ro for c in ctrl])
        evaluate(x0)
        report()
        res = scipy_minimize(evaluate, x0, jac=True, method="L-BFGS-B", callback=report,
                             options={"maxiter": maxiter})
        assign(res.x)
        log(f"{res.message} after {res.nit} iterations, {res.nfev} evaluations")
        return ctrl
