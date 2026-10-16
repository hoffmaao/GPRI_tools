"""The observations: LOS velocity from the 24 h terms, strain rates, geometry."""
import numpy as np

from baker.data import (YEAR_DAYS, cell_velocity, los_vectors, neighbour_pairs,
                        strain_rates)


def test_velocity_is_the_time_derivative_of_the_harmonic_fit():
    rng = np.random.default_rng(1)
    theta = rng.standard_normal((4, 5))                      # offset, rate, cos, sin
    origin = 3.0
    hours = np.array([0.0, 7.5, 13.0])
    v, _ = cell_velocity(theta, np.ones_like(theta), hours, origin)

    def y(t):                                                # mm, t in days from origin
        w = 2 * np.pi
        return theta[0] + theta[1] * t + theta[2] * np.cos(w * t) + theta[3] * np.sin(w * t)

    dt = 1e-6
    t = (hours - origin) / 24.0
    numeric = (y(t[:, None] + dt) - y(t[:, None] - dt)) / (2 * dt) * YEAR_DAYS / 1000
    assert np.allclose(v, numeric, rtol=1e-6)


def test_strain_rate_and_its_sigma_follow_from_the_cells():
    v = np.array([[1.0, 3.0, 6.0]])
    var = np.array([[0.04, 0.09, 0.16]])
    pairs = np.array([[0, 1], [1, 2]])
    dist = np.array([200.0, 100.0])
    rate, sigma = strain_rates(v, var, pairs, dist)
    assert np.allclose(rate, [[-2 / 200, -3 / 100]])
    assert np.allclose(sigma, [[np.sqrt(0.13) / 200, np.sqrt(0.25) / 100]])


def test_neighbours_are_edges_and_corners_only():
    g = np.arange(3) * 200.0
    xy = np.array([(x, y) for y in g for x in g])
    pairs, dist = neighbour_pairs(xy, 200.0)
    assert len(pairs) == 12 + 8                              # 12 edges, 8 diagonals
    assert np.allclose(np.sort(np.unique(dist.round(6))), [200.0, 200.0 * np.sqrt(2)])


def test_los_vectors_point_at_the_radar():
    xy = np.array([[300.0, 400.0], [-1000.0, 0.0]])
    l = los_vectors(xy, np.array([100.0, 50.0]), radar=(0.0, 0.0, 100.0))
    assert np.allclose(np.linalg.norm(l, axis=1), 1.0)
    assert np.allclose(l[0], [-0.6, -0.8, 0.0])
    assert l[1, 0] > 0 and l[1, 2] > 0


def test_smoothing_leaves_cells_without_data_empty():
    """Round-off in the box filter must not spread values into empty cells."""
    from baker.radar_dem import smooth_on_grid
    v = np.full((60, 60), np.nan)
    w = np.zeros((60, 60))
    rng = np.random.default_rng(1)
    v[:10, :10] = rng.normal(0.0, 5.0, (10, 10))
    w[:10, :10] = rng.uniform(0.01, 1.0, (10, 10))
    smoothed, n_eff = smooth_on_grid(v, w, 4)
    far = np.zeros_like(w, dtype=bool)
    far[20:, :] = True
    far[:, 20:] = True
    assert np.isnan(smoothed[far]).all()
    assert (n_eff[far] == 0).all()
    assert np.isfinite(smoothed[:10, :10]).all()
