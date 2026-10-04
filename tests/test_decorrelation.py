"""Coherence against temporal baseline: classes, crossing time, fitted curve."""
import numpy as np
import pytest

from gpri_tools.decorrelation import (DecorrelationFit, coherence_by_baseline,
                                      decorrelation_time,
                                      exponential_decorrelation)


def _stack(n_epochs=40, lags=(1, 2, 5), cadence=1 / 720):
    times = np.arange(n_epochs) * cadence
    pairs = np.array([(i, i + k) for i in range(n_epochs)
                      for k in lags if i + k < n_epochs])
    return pairs, times


def test_classes_collapse_to_the_lags_the_pairs_were_formed_at():
    pairs, times = _stack()
    cc = np.ones((pairs.shape[0], 3, 4))
    b, g, n = coherence_by_baseline(cc, pairs, times)
    assert b.size == 3 and g.shape == (3, 3, 4)
    assert np.all(np.diff(b) > 0)
    assert n.sum() == pairs.shape[0]
    assert b == pytest.approx(np.array([1, 2, 5]) / 720, rel=1e-6)


def test_each_class_averages_its_own_pairs():
    pairs, times = _stack(n_epochs=10, lags=(1, 3))
    lag = pairs[:, 1] - pairs[:, 0]
    cc = np.where(lag[:, None, None] == 1, 0.9, 0.4) * np.ones((1, 2, 2))
    b, g, _ = coherence_by_baseline(cc, pairs, times)
    assert g[0] == pytest.approx(0.9) and g[1] == pytest.approx(0.4)


def test_baselines_must_run_forward():
    pairs, times = _stack(n_epochs=6, lags=(1,))
    with pytest.raises(ValueError, match="forward in time"):
        coherence_by_baseline(np.ones((pairs.shape[0], 2, 2)),
                              pairs[:, ::-1], times)


def test_crossing_time_recovers_an_exponential():
    tau = 0.25
    b = np.logspace(-2, 0.5, 12)
    g = np.exp(-b / tau)[:, None] * np.ones((1, 5))
    tau_hat = decorrelation_time(b, g)
    assert tau_hat == pytest.approx(tau, rel=0.05)


def test_a_pixel_that_never_decorrelates_comes_back_nan():
    b = np.array([0.001, 0.01, 0.1])
    g = np.array([0.95, 0.94, 0.93])[:, None] * np.ones((1, 3))
    assert np.all(np.isnan(decorrelation_time(b, g)))


def test_crossing_time_needs_ascending_baselines():
    g = np.ones((3, 2))
    with pytest.raises(ValueError, match="ascend"):
        decorrelation_time(np.array([1.0, 0.5, 2.0]), g)
    with pytest.raises(ValueError, match="at least two"):
        decorrelation_time(np.array([1.0]), np.ones((1, 2)))


def test_exponential_fit_recovers_tau_and_the_floor():
    tau, g0, g_inf = 0.2, 0.9, 0.25
    b = np.logspace(-2.5, 0.5, 20)
    g = (g0 - g_inf) * np.exp(-b / tau) + g_inf
    f = exponential_decorrelation(b, g)
    assert isinstance(f, DecorrelationFit)
    assert f.tau == pytest.approx(tau, rel=0.1)
    assert f.g_inf == pytest.approx(g_inf, abs=0.02)
    assert f.g0 == pytest.approx(g0, abs=0.02)
    assert f.rms < 1e-3
    assert f(0.0) == pytest.approx(g0, abs=0.02)


def test_exponential_fit_refuses_two_points():
    with pytest.raises(ValueError, match="three points"):
        exponential_decorrelation(np.array([1.0, 2.0]), np.array([0.9, 0.5]))


def test_crossing_at_an_absolute_level():
    b = np.array([0.1, 1.0, 10.0])
    g = np.array([[0.9, 0.9], [0.6, 0.4], [0.3, 0.2]])
    tau = decorrelation_time(b, g, level=0.5)
    # the first pixel crosses 0.5 between 1 and 10 h, the second between 0.1 and 1
    assert 1.0 < tau[0] < 10.0
    assert 0.1 < tau[1] < 1.0
    # and the level, not the pixel's own start, is what it crossed
    assert np.all(np.isnan(decorrelation_time(b, g, level=0.05)))


def test_level_overrides_the_fraction():
    b = np.array([0.1, 1.0, 10.0])
    g = np.array([[0.9], [0.6], [0.3]])
    by_level = decorrelation_time(b, g, level=0.5)
    by_fraction = decorrelation_time(b, g, fraction=0.5)      # target 0.45
    assert np.isfinite(by_level[0]) and np.isfinite(by_fraction[0])
    assert by_level[0] < by_fraction[0]


def test_lag_classing_survives_cadence_jitter():
    """A wobbling cadence must not split one lag into many classes."""
    rng = np.random.default_rng(3)
    n = 60
    times = np.arange(n) / 720 + rng.normal(0, 1e-6, n)     # jitter in the clock
    times.sort()
    pairs = np.array([(i, i + k) for i in range(n) for k in (1, 3) if i + k < n])
    cc = np.ones((pairs.shape[0], 2, 2))
    b, g, counts = coherence_by_baseline(cc, pairs, times)          # by='lag'
    assert b.size == 2 and counts.min() > 50
    b_t, _, counts_t = coherence_by_baseline(cc, pairs, times, by="time")
    assert b_t.size >= b.size            # the time path may split; lag does not


def test_singleton_classes_can_be_dropped():
    n = 30
    times = np.arange(n) / 720
    pairs = np.array([(i, i + k) for i in range(n) for k in (1, 2) if i + k < n]
                     + [(0, n - 1)])                    # one very long pair
    cc = np.ones((pairs.shape[0], 2, 2))
    b_all, _, c_all = coherence_by_baseline(cc, pairs, times)
    b_cut, _, c_cut = coherence_by_baseline(cc, pairs, times, min_pairs=5)
    assert c_all.min() == 1 and b_all.size == 3
    assert c_cut.min() >= 5 and b_cut.size == 2


def test_an_unknown_classing_is_refused():
    pairs, times = _stack(n_epochs=8, lags=(1,))
    with pytest.raises(ValueError, match="'lag' or 'time'"):
        coherence_by_baseline(np.ones((pairs.shape[0], 2, 2)), pairs, times,
                              by="whatever")
