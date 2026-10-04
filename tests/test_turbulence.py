"""The structure function of a field, and the power law fitted to it."""
import numpy as np
import pytest

from gpri_tools.turbulence import (FlooredPowerLaw, PowerLaw, power_law,
                                   power_law_with_floor, structure_function)


def _points(n=4000, seed=0):
    g = np.random.default_rng(seed)
    return g.uniform(0, 1000, n), g.uniform(0, 1000, n), g


def test_white_noise_has_a_flat_structure_function():
    x, y, g = _points()
    v = g.normal(0, 1.0, x.size)
    bins = np.array([10.0, 50.0, 200.0, 600.0])
    r, D, n = structure_function(v, x, y, bins, max_pairs=20_000, rng=g)
    assert np.all(n > 1000)
    # uncorrelated points: D(r) = 2 var, at every separation
    assert np.allclose(D, 2.0, rtol=0.15)
    fit = power_law(r, D, r0=100.0)
    assert abs(fit.exponent) < 0.1


def test_a_linear_ramp_gives_an_exponent_of_two():
    x, y, g = _points(6000, seed=1)
    v = 0.01 * x                                   # a pure ramp in x
    bins = np.array([20.0, 60.0, 180.0, 540.0])
    r, D, _ = structure_function(v, x, y, bins, max_pairs=30_000, rng=g)
    fit = power_law(r, D, r0=100.0)
    assert fit.exponent == pytest.approx(2.0, abs=0.15)


def test_the_estimator_needs_finite_points_and_ascending_bins():
    x, y, g = _points(50)
    with pytest.raises(ValueError, match="same length"):
        structure_function(np.ones(10), x, y, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="ascending"):
        structure_function(np.ones(50), x, y, np.array([2.0, 1.0]))
    v = np.full(50, np.nan)
    with pytest.raises(ValueError, match="two finite"):
        structure_function(v, x, y, np.array([1.0, 2.0]))


def test_non_finite_values_are_dropped_not_propagated():
    x, y, g = _points(2000, seed=2)
    v = g.normal(0, 1.0, x.size)
    v[::7] = np.nan
    r, D, n = structure_function(v, x, y, np.array([10.0, 300.0]),
                                 max_pairs=20_000, rng=g)
    assert np.isfinite(D).all() and n[0] > 1000


def test_power_law_round_trips_its_own_curve():
    r = np.logspace(1, 3, 12)
    truth = PowerLaw(exponent=5 / 3, amplitude=4.0, r0=1000.0, rms=0.0, n=12)
    fit = power_law(r, truth(r), r0=1000.0)
    assert fit.exponent == pytest.approx(5 / 3, rel=1e-6)
    assert fit.amplitude == pytest.approx(4.0, rel=1e-6)
    assert fit.rms < 1e-9
    assert fit(1000.0) == pytest.approx(4.0, rel=1e-6)


def test_power_law_refuses_a_single_point():
    with pytest.raises(ValueError, match="two positive"):
        power_law(np.array([100.0]), np.array([1.0]))


def test_the_floor_separates_noise_from_a_real_exponent():
    """A turbulent field plus white noise: the floor is the noise, not a flat air."""
    r = np.logspace(1.5, 3.5, 12)
    noise_rms, alpha, amp = 2.0, 5 / 3, 4.0
    D = 2 * noise_rms ** 2 + amp * (r / 1000.0) ** alpha
    plain = power_law(r, D, r0=1000.0)
    floored = power_law_with_floor(r, D, r0=1000.0)
    assert isinstance(floored, FlooredPowerLaw)
    assert floored.exponent == pytest.approx(alpha, abs=0.05)
    assert floored.amplitude == pytest.approx(amp, rel=0.1)
    assert floored.noise_rms == pytest.approx(noise_rms, rel=0.1)
    # the unfloored fit is dragged well below the true exponent by the floor
    assert plain.exponent < alpha - 0.3


def test_pure_noise_gives_all_floor_and_no_structure():
    r = np.logspace(1.5, 3.5, 10)
    D = np.full(r.size, 8.0)
    f = power_law_with_floor(r, D, r0=1000.0)
    assert f.noise_rms == pytest.approx(2.0, rel=0.05)
    assert abs(f.amplitude) < 0.2


def test_the_floor_can_be_pinned_and_stays_non_negative():
    r = np.logspace(2, 3.5, 8)
    D = 3.0 * (r / 1000.0) ** 1.0            # no floor at all
    f = power_law_with_floor(r, D, r0=1000.0)
    assert f.floor >= 0.0
    pinned = power_law_with_floor(r, D + 5.0, r0=1000.0, floor=5.0)
    assert pinned.floor == 5.0
    assert pinned.exponent == pytest.approx(1.0, abs=0.05)


def test_floored_fit_needs_three_points():
    with pytest.raises(ValueError, match="three points"):
        power_law_with_floor(np.array([10.0, 100.0]), np.array([1.0, 2.0]))
