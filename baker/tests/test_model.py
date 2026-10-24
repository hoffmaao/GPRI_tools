"""The flow model and the misfit's adjoint, on a small slab; needs Firedrake and icepack2."""
import numpy as np
import pytest

firedrake = pytest.importorskip("firedrake")
pytest.importorskip("icepack2")

from firedrake import Constant, Function, SpatialCoordinate  # noqa: E402
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,  # noqa: E402
                               get_working_tape, pause_annotation, taylor_test)

from baker.inverse import Problem, RadarStrainRates  # noqa: E402
from baker.observe import CellPoints, pair_misfit, pair_rates  # noqa: E402
from baker.physics import ForwardModel  # noqa: E402


@pytest.fixture(scope="module")
def slab():
    mesh = firedrake.RectangleMesh(10, 10, 2000.0, 2000.0)
    x, y = SpatialCoordinate(mesh)
    Q = firedrake.FunctionSpace(mesh, "CG", 1)
    s = Function(Q).interpolate(2000.0 - 0.1 * x)            # slopes down toward +x
    model = ForwardModel(mesh, s)
    h = Function(Q).interpolate(firedrake.max_value(0, 100.0 * (1 - ((x - 1000) / 800) ** 2)))
    return model, h


def test_ice_flows_down_slope_and_ground_stays_still(slab):
    model, h = slab
    N = Function(model.Q).assign(1.0)
    z = model.solve(h, N)
    u = z.subfunctions[0]
    assert u.at((1000.0, 1000.0))[0] > 1.0
    assert abs(u.at((100.0, 1000.0))[0]) < 0.05 * u.at((1000.0, 1000.0))[0]


def test_lower_effective_pressure_slides_faster(slab):
    model, h = slab
    speeds = []
    for value in (2.0, 1.0):
        z = model.solve(h, Function(model.Q).assign(value))
        speeds.append(z.subfunctions[0].at((1000.0, 1000.0))[0])
    assert speeds[1] > speeds[0]


def test_pair_misfit_gradient_is_exact(slab):
    model, h = slab
    xy = np.array([[800.0, 900.0], [1000.0, 900.0], [1200.0, 1100.0]])
    los = np.tile([-0.6, -0.8, 0.0], (3, 1))
    points = CellPoints(model.mesh, xy, los)
    pairs, dist = np.array([[0, 1], [1, 2]]), np.array([200.0, 283.0])
    N = Function(model.Q).assign(1.0)
    z = model.solve(h, N)
    obs = np.array([0.01, -0.02])
    continue_annotation()
    q = Function(model.Q).assign(0.0)
    Nq = Function(model.Q).interpolate(N * firedrake.exp(q))
    model.solve(h, Nq, z=z, continuation=False)
    v = points.los_velocity(model.surface_velocity(z, h, Nq), model.s)
    J = pair_misfit(v, pairs, dist, obs, np.array([0.01, 0.01]))
    rf = ReducedFunctional(J, Control(q))
    pause_annotation()
    assert np.allclose(pair_rates(v.dat.data_ro, pairs, dist).shape, (2,))
    d = Function(model.Q).assign(0.05)
    assert taylor_test(rf, q.copy(deepcopy=True), d) > 1.9
    get_working_tape().clear_tape()


def test_continuity_is_melt_only_and_its_gradient_is_exact(slab):
    model, h = slab
    xy = np.array([[800.0, 900.0], [1000.0, 900.0], [1200.0, 1100.0]])
    los = np.tile([-0.6, -0.8, 0.0], (3, 1))
    d = RadarStrainRates("20170803", CellPoints(model.mesh, xy, los), np.array([np.nan]),
                         np.array([[0, 1], [1, 2]]), np.array([200.0, 283.0]),
                         np.array([[0.01, -0.02]]), np.array([[0.01, 0.01]]), epoch=2017.6)
    d.ds = Function(model.Q).assign(0.0)
    d.ds_w = Function(model.Q).assign(1.0)
    P = Problem(model, h, [d], continuity=True, spacing=200.0)
    ctrl = P.controls()
    ctrl[-1].assign(-30.0)
    assert float(firedrake.assemble(P.melt(ctrl) * firedrake.dx(model.mesh))) > 0.0
    ctrl[-1].assign(0.0)
    zs = P.solve_all(ctrl)
    H = Function(model.Q).interpolate(P.thickness(ctrl[0], d))
    u = P.depth_averaged([z.copy(deepcopy=True) for z in zs])
    dt = d.epoch - 2015.6927
    # observed change = continuity with the prior melt: no mass misfit there, some elsewhere
    d.ds_obs = P.flux_change(d, H.copy(deepcopy=True), u) - Constant(dt * P.reg.m_0)
    assert abs(P.mass_misfit(d, H, u, P.melt(ctrl))) < 1e-6
    assert P.mass_misfit(d, H, u, 2 * P.melt(ctrl)) > 1.0
    continue_annotation()
    J, parts = P.objective(ctrl, zs)
    controls = [ctrl[0], ctrl[-1]]
    rf = ReducedFunctional(J, [Control(c) for c in controls])
    pause_annotation()
    assert len(parts) == 2
    d0 = [Function(model.Q).assign(0.02), Function(model.Q).assign(0.05)]
    assert taylor_test(rf, [c.copy(deepcopy=True) for c in controls], d0) > 1.9
    get_working_tape().clear_tape()
