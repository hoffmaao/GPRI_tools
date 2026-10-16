"""The model's LOS strain rates, and their misfit, with an exact adjoint.

The radar sees a cell's line-of-sight velocity, ``v = l . (u_x, u_y, w)``
toward the radar, where ``w = u . grad(s)`` is the vertical velocity of flow
parallel to the surface.  The observation is the difference of two cells'
``v`` over their separation.  Firedrake evaluates ``v`` at the cell centres
(a vertex-only mesh); the pair differences and the weighted misfit are
formed in numpy by :class:`PairMisfitBlock`, which pyadjoint differentiates
exactly: the gradient of ``J = sum ((D v - e) / sigma)^2 / 2`` with respect
to ``v`` is ``D^T ((D v - e) / sigma^2)``.
"""
from __future__ import annotations

import numpy as np

import firedrake
from firedrake import Function, FunctionSpace, VectorFunctionSpace, VertexOnlyMesh, dot
from pyadjoint import AdjFloat, Block
from pyadjoint.tape import annotate_tape, get_working_tape


class CellPoints:
    """The cell centres as a vertex-only mesh, with each cell's unit vector toward the radar."""

    def __init__(self, mesh, xy, los):
        self.vom = VertexOnlyMesh(mesh, np.asarray(xy, float), missing_points_behaviour="error")
        self.P = FunctionSpace(self.vom, "DG", 0)
        self.W = VectorFunctionSpace(self.vom, "DG", 0)
        # numpy works in the order the cells were given; the vom may reorder them
        self.P_in = FunctionSpace(self.vom.input_ordering, "DG", 0)
        given = []
        for k in range(3):
            f = Function(self.P_in)
            f.dat.data[:] = np.asarray(los, float)[:, k]
            given.append(Function(self.P).interpolate(f))
        self.lx, self.ly, self.lz = given

    def los_velocity(self, u_surface, surface):
        """LOS velocity at every cell, in the input order (m/yr, toward the radar)."""
        U = Function(self.W).interpolate(u_surface)
        G = Function(self.W).interpolate(firedrake.grad(surface))
        v = Function(self.P).interpolate(self.lx * U[0] + self.ly * U[1]
                                         + self.lz * dot(U, G))
        return Function(self.P_in).interpolate(v)


def _misfit(v, pairs, dist, obs, sigma):
    d = (v[pairs[:, 0]] - v[pairs[:, 1]]) / dist
    r = (d - obs) / sigma
    return 0.5 * float(np.sum(r * r)), r / sigma


class PairMisfitBlock(Block):
    def __init__(self, v, pairs, dist, obs, sigma):
        super().__init__()
        self.add_dependency(v)
        self.pairs, self.dist, self.obs, self.sigma = pairs, dist, obs, sigma

    def __str__(self):
        return "PairMisfitBlock"

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return AdjFloat(_misfit(inputs[0].dat.data_ro, self.pairs, self.dist,
                                self.obs, self.sigma)[0])

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        v = inputs[0]
        _, w = _misfit(v.dat.data_ro, self.pairs, self.dist, self.obs, self.sigma)
        g = np.zeros_like(v.dat.data_ro)
        np.add.at(g, self.pairs[:, 0], w / self.dist)
        np.add.at(g, self.pairs[:, 1], -w / self.dist)
        out = firedrake.Cofunction(v.function_space().dual())
        out.dat.data[:] = float(adj_inputs[0]) * g
        return out


def pair_misfit(v, pairs, dist, obs, sigma):
    """``sum(((v_i - v_j) / d_ij - obs) / sigma)^2 / 2`` over the pairs, on the tape."""
    pairs = np.asarray(pairs, int)
    J = AdjFloat(_misfit(v.dat.data_ro, pairs, dist, obs, sigma)[0])
    if annotate_tape():
        block = PairMisfitBlock(v, pairs, np.asarray(dist, float), np.asarray(obs, float),
                                np.asarray(sigma, float))
        get_working_tape().add_block(block)
        block.add_output(J.block_variable)
    return J


def pair_rates(v, pairs, dist):
    """The strain rates themselves (numpy), for synthetic data and plots."""
    v = np.asarray(v)
    return (v[..., pairs[:, 0]] - v[..., pairs[:, 1]]) / dist


class VelocityPoints:
    """Points where a horizontal surface velocity is observed (ITS_LIVE pixels)."""

    def __init__(self, mesh, xy, vx, vy, sigma):
        self.vom = VertexOnlyMesh(mesh, np.asarray(xy, float), missing_points_behaviour="error")
        self.P = FunctionSpace(self.vom, "DG", 0)
        self.W = VectorFunctionSpace(self.vom, "DG", 0)
        P_in = FunctionSpace(self.vom.input_ordering, "DG", 0)
        given = []
        for values in (vx, vy, 1.0 / np.asarray(sigma, float) ** 2):
            f = Function(P_in)
            f.dat.data[:] = values
            given.append(Function(self.P).interpolate(f))
        self.vx, self.vy, self.weight = given
        self.size = 2 * len(vx)

    def misfit(self, u_surface):
        """``sum(((u - u_obs) / sigma)^2) / 2`` over the points and both components, taped."""
        U = Function(self.W).interpolate(u_surface)
        return firedrake.assemble(0.5 * self.weight * ((U[0] - self.vx) ** 2
                                                       + (U[1] - self.vy) ** 2) * firedrake.dx)
