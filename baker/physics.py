"""Momentum balance at zero thickness, with a regularized Coulomb sliding law.

The model is icepack2's dual (mixed) form of the shallow stream equations:
velocity ``u``, membrane stress ``M`` and basal stress ``tau`` are all
unknowns, which keeps the problem solvable where the thickness ``h`` is zero,
so the domain is a plain rectangle and ice-free ground is ``h = 0``.  Units
are icepack2's: MPa, metres, years.

**Flow law.**  Glen's law for temperate ice, ``n = 3``, rate factor at
273.15 K (:func:`icepack.rate_factor`).

**Sliding law.**  The regularized Coulomb law of Joughin et al. (2019),
``|tau| = mu N (|u_b| / (|u_b| + u_0))^(1/m)``, written the way the dual form
wants it, as the sliding velocity in terms of the basal stress:

    u_b = -u_0 r^m / (1 - r^m) tau / |tau|,   r = |tau| / (mu N)

so sliding grows without bound as the basal stress approaches the Coulomb
limit ``mu N``.  ``1 - r^m`` is smoothed (``d -> (d + sqrt(d^2 + delta^2)) /
2``) so a Newton step that crosses the limit stays finite.  Only the product
``mu N`` enters; ``mu`` is fixed and ``N`` is read through it.

**Internal deformation.**  As icepack2 does it, through the same basal term:
a shearing velocity ``h A / (n + 2) |tau|^(n - 1) tau`` is added to the sliding
velocity, so ``u`` is the depth-averaged velocity.  The surface velocity the
radar sees is the sliding velocity plus ``(n + 2) / (n + 1)`` of that
deformation part (:meth:`ForwardModel.surface_velocity`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import firedrake
from firedrake import (Constant, Function, FunctionSpace, VectorFunctionSpace,
                       TensorFunctionSpace, inner, sqrt, grad, dx, derivative, replace)

from icepack2.model.variational import flow_law, friction_law, momentum_balance
from icepack2.model import utilities


@dataclass
class Parameters:
    """Material and sliding parameters, icepack2 units (MPa, m, yr)."""
    n: float = 3.0                 # Glen exponent
    m: float = 3.0                 # sliding exponent
    temperature: float = 273.15    # K, temperate ice
    mu: float = 0.5                # Coulomb friction coefficient; only mu N is seen
    u0: float = 300.0              # m/yr, regularized Coulomb speed scale
    delta: float = 0.05            # smoothing of 1 - r^m near the Coulomb limit
    tau_c: float = 0.1             # MPa, stress scale of the linear regularization
    h_scale: float = 100.0         # m, thickness standing in for h in the flow law
    alpha: float = 1e-4            # weight of the linear regularization
    h_min: float = 1e-3            # m, floor of h where it multiplies a power law


def regularized_coulomb(u, tau, h, N, A, p: Parameters, n=None, m=None):
    r"""The sliding-plus-deformation relation as a residual form, test function from ``tau``."""
    sigma = utilities.get_test_function(tau)
    n = Constant(p.n) if n is None else n
    m = Constant(p.m) if m is None else m
    tau2 = inner(tau, tau) + Constant(1e-12)
    r_m = (tau2 / (Constant(p.mu) * N) ** 2) ** (m / 2)
    d = 1 - r_m
    d_reg = (d + sqrt(d ** 2 + Constant(p.delta) ** 2)) / 2
    # sliding: u0 r^m / (1 - r^m) along tau, written without dividing by |tau|
    k_slide = Constant(p.u0) * tau2 ** ((m - 1) / 2) / ((Constant(p.mu) * N) ** m * d_reg)
    k_def = h * A / (n + 2) * tau2 ** ((n - 1) / 2)
    return inner((k_slide + k_def) * tau + u, sigma) * dx


class ForwardModel:
    """The momentum balance on one mesh, solved for given thickness and effective pressure."""

    def __init__(self, mesh, surface, params: Parameters | None = None):
        import icepack
        self.mesh = mesh
        self.p = params or Parameters()
        self.V = VectorFunctionSpace(mesh, "CG", 1)
        self.Sigma = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
        self.T = VectorFunctionSpace(mesh, "CG", 1)
        self.Q = FunctionSpace(mesh, "CG", 1)
        self.Z = self.V * self.Sigma * self.T
        self.s = surface
        self.A = Constant(float(icepack.rate_factor(Constant(self.p.temperature))))
        self.n = Constant(self.p.n)
        self.m = Constant(self.p.m)
        self.N_scale = Constant(1.0)
        self.qdegree = 6
        self.sparams = {
            "snes_type": "newtonls", "snes_max_it": 200, "snes_linesearch_type": "nleqerr",
            "snes_rtol": 1e-8, "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    def residual(self, z, h, N, s=None):
        p = self.p
        u, M, tau = firedrake.split(z)
        tau_c = Constant(p.tau_c)
        eps_c = self.A * tau_c ** self.n
        glen = {"flow_law_exponent": self.n, "flow_law_coefficient": eps_c / tau_c ** self.n}
        linear = {"flow_law_exponent": 1, "flow_law_coefficient": eps_c / tau_c}
        fields = {"velocity": u, "membrane_stress": M, "basal_stress": tau,
                  "thickness": h, "surface": self.s if s is None else s}
        H = Constant(p.h_scale)
        hp = firedrake.max_value(Constant(p.h_min), h)
        # icepack2's small linear friction, which keeps Newton's early steps well posed
        u_c = Constant(p.u0) + hp * self.A * tau_c ** self.n / (self.n + 2)
        slide = {"sliding_exponent": 1, "sliding_coefficient": u_c / tau_c}
        return (momentum_balance(**fields)
                + replace(flow_law(**fields, **glen), {h: H})
                + Constant(p.alpha) * replace(flow_law(**fields, **linear), {h: H})
                + regularized_coulomb(u, tau, hp, self.N_scale * N, self.A, p, self.n, self.m)
                + Constant(p.alpha) * friction_law(**fields, **slide))

    def solve(self, h, N, z=None, continuation=True, s=None):
        """Solve for ``(u, M, tau)``; ``z`` is the starting guess and is overwritten.

        From a cold start the Glen and sliding exponents are stepped from 1 to
        ``n`` and ``m``, as the icepack2 demos do, at ten times ``N`` (far from
        the Coulomb limit), and ``N`` is then stepped down to its value.  A
        warm start goes straight to the target, and falls back to a cold
        start if that fails.  A step that fails is halved, down to 1/64.
        Raises :class:`firedrake.ConvergenceError` if the target is not
        reached.
        """
        cold = z is None or continuation
        z = Function(self.Z) if z is None else z
        F = self.residual(z, h, N, s)
        problem = firedrake.NonlinearVariationalProblem(
            F, z, form_compiler_parameters={"quadrature_degree": self.qdegree})
        solver = firedrake.NonlinearVariationalSolver(problem, solver_parameters=self.sparams)
        target = (self.p.n, self.p.m, 1.0)
        if not cold:
            try:
                self._set(target)
                solver.solve()
                return z
            except firedrake.ConvergenceError:
                z.assign(0.0)
        else:
            z.assign(0.0)
        start = (1.0, 1.0, 10.0)
        self._set(start)
        solver.solve()
        self._walk(solver, z, start, (self.p.n, self.p.m, 10.0))
        self._walk(solver, z, (self.p.n, self.p.m, 10.0), target)
        return z

    def solve_from(self, h, N, z, s=None):
        """One Newton solve at the target parameters from ``z``, on the tape if annotating."""
        problem = firedrake.NonlinearVariationalProblem(
            self.residual(z, h, N, s), z,
            form_compiler_parameters={"quadrature_degree": self.qdegree})
        self._set((self.p.n, self.p.m, 1.0))
        firedrake.NonlinearVariationalSolver(problem, solver_parameters=self.sparams).solve()
        return z

    def _set(self, state):
        n, m, scale = state
        self.n.assign(n)
        self.m.assign(m)
        self.N_scale.assign(scale)

    def _walk(self, solver, z, a, b, steps=4, depth=6):
        """Continue the solution from parameters ``a`` to ``b`` (n, m, log-stepped N scale)."""
        def at(f):
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]),
                    a[2] ** (1 - f) * b[2] ** f)

        fs, done = list(np.linspace(0, 1, steps + 1)[1:]), 0.0
        while fs:
            f = fs[0]
            saved = z.copy(deepcopy=True)
            try:
                self._set(at(f))
                solver.solve()
                done = fs.pop(0)
            except firedrake.ConvergenceError:
                z.assign(saved)
                if f - done < 2.0 ** -depth:
                    raise
                fs.insert(0, (done + f) / 2)
        self._set(b)

    def surface_velocity(self, z, h, N):
        """Horizontal surface velocity (m/yr): sliding plus (n+2)/(n+1) of the deformation."""
        p = self.p
        u, M, tau = firedrake.split(z)
        n = Constant(p.n)
        tau2 = inner(tau, tau) + Constant(1e-12)
        hp = firedrake.max_value(Constant(p.h_min), h)
        u_def = -hp * self.A / (n + 2) * tau2 ** ((n - 1) / 2) * tau
        return u + (Constant(1.0) / (n + 1)) * u_def
