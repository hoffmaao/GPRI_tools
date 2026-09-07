"""Double-difference path delay: what it recovers, and what it cannot see."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from gpri_tools.network import Network
from gpri_tools.pathdelay import (DoubleDifference, PathDelay, discarded_rate,
                                  double_difference, frequency_response,
                                  invert_path_delay, pin_affine, pin_rate,
                                  select_lambda, shared_epoch_triplets, tikhonov)

CADENCE = 4.0 / (60.0 * 24.0)          # four minutes, in days


def _chain(n_epochs=40, lags=(1,), cadence=CADENCE):
    """A daisy chain at fixed cadence, optionally with longer baselines too."""
    t0 = datetime(2017, 8, 3, 22, 0, 0)
    epochs = [t0 + timedelta(days=cadence * k) for k in range(n_epochs)]
    pairs = [(i, i + lag) for lag in lags
             for i in range(n_epochs - lag)]
    return Network(epochs, pairs)


def _observe(net, delay, rate=0.0, curvature=0.0):
    """Pair observations from a per-epoch delay plus a motion history."""
    t = net.times
    motion = rate * t + 0.5 * curvature * t ** 2
    total = motion + np.asarray(delay, float)
    return np.array([total[j] - total[i] for i, j in net.pairs])


# ------------------------------------------------------------------ triplets
def test_triplets_are_pairs_sharing_a_middle_epoch():
    net = _chain(n_epochs=5)
    rows = shared_epoch_triplets(net.pairs)
    assert len(rows) == 3                                  # (0,1)+(1,2), ...
    for p, q in rows:
        assert net.pairs[p][1] == net.pairs[q][0]


def test_single_reference_network_has_no_triplet():
    epochs = [datetime(2017, 8, 3, 22, 0) + timedelta(minutes=4 * k)
              for k in range(5)]
    net = Network(epochs, [(0, j) for j in range(1, 5)])
    assert len(shared_epoch_triplets(net.pairs)) == 0
    with pytest.raises(ValueError, match="shares a middle epoch"):
        double_difference(net.pairs, net.times)


def test_max_span_drops_the_long_triplets():
    net = _chain(n_epochs=10, lags=(1, 3))
    everything = shared_epoch_triplets(net.pairs, net.times)
    short = shared_epoch_triplets(net.pairs, net.times, max_span=2.5 * CADENCE)
    assert 0 < len(short) < len(everything)
    for p, q in short:
        i, k = net.pairs[p][0], net.pairs[q][1]
        assert abs(net.times[k] - net.times[i]) <= 2.5 * CADENCE + 1e-12


def test_max_triplets_thins_the_list():
    net = _chain(n_epochs=30, lags=(1, 2))
    rows = shared_epoch_triplets(net.pairs, net.times, max_triplets=10)
    assert len(rows) <= 10


# ------------------------------------------------------- the operator itself
def test_affine_delay_is_in_the_null_space():
    net = _chain(n_epochs=25, lags=(1, 2, 3))
    sys = double_difference(net.pairs, net.times)
    t = net.times
    for a in (np.ones_like(t), t, 3.2 - 17.0 * t):
        assert np.allclose(sys.A @ a, 0.0, atol=1e-9)


def test_steady_motion_cancels_at_any_spacing():
    rng = np.random.default_rng(3)
    t0 = datetime(2017, 8, 3, 22, 0)
    # deliberately uneven: a dropped acquisition here and there
    steps = np.cumsum(rng.integers(1, 4, 24)) * CADENCE
    epochs = [t0 + timedelta(days=float(s)) for s in np.r_[0.0, steps]]
    net = Network(epochs, [(i, i + 1) for i in range(len(epochs) - 1)])
    sys = double_difference(net.pairs, net.times)

    obs = _observe(net, np.zeros(net.n_epochs), rate=53.0)
    assert np.allclose(sys.apply(obs), 0.0, atol=1e-9)


def test_unnormalised_rows_only_cancel_even_spacing():
    t0 = datetime(2017, 8, 3, 22, 0)
    epochs = [t0 + timedelta(days=float(s) * CADENCE) for s in (0, 1, 3)]
    net = Network(epochs, [(0, 1), (1, 2)])
    obs = _observe(net, np.zeros(3), rate=11.0)

    assert np.allclose(double_difference(net.pairs, net.times).apply(obs), 0.0,
                       atol=1e-9)
    plain = double_difference(net.pairs, net.times, normalise=False)
    assert abs(float(plain.apply(obs)[0])) > 1e-3


def test_curvature_is_what_leaks_through():
    """A double difference measures the second derivative of the motion."""
    net = _chain(n_epochs=3)
    sys = double_difference(net.pairs, net.times)
    obs = _observe(net, np.zeros(3), rate=5.0, curvature=2.0)
    # (-d_i + 2 d_j - d_k)/dt for d = c t^2/2 is -c * dt
    assert float(sys.apply(obs)[0]) == pytest.approx(-2.0 * CADENCE, rel=1e-6)


def test_more_baselines_carry_less_noise_into_the_delay():
    """Why the method wants several temporal baselines, not just the chain."""
    def noise_gain(lags):
        net = _chain(n_epochs=40, lags=lags)
        t = net.times
        M = np.linalg.pinv(double_difference(net.pairs, t).A)
        basis = np.column_stack([np.ones_like(t), t - t.mean()])
        pinned = (np.eye(t.size) - basis @ np.linalg.pinv(basis)) @ M
        return np.sqrt(np.trace(pinned @ pinned.T) / t.size)

    gains = [noise_gain(l) for l in ((1,), (1, 2), (1, 2, 3))]
    assert gains[2] < gains[1] < gains[0]
    assert gains[2] < 0.25 * gains[0]


# ------------------------------------------------------------------ recovery
def test_noiseless_recovery_up_to_the_affine_part():
    rng = np.random.default_rng(0)
    net = _chain(n_epochs=50, lags=(1, 2))
    a = rng.normal(0, 2.0, net.n_epochs)
    obs = _observe(net, a, rate=37.0)

    pd = invert_path_delay(obs, net.pairs, net.times, lam=0.0)
    assert isinstance(pd, PathDelay)
    assert np.allclose(pd.delay, pin_affine(a, net.times), atol=1e-6)


def test_the_affine_part_is_unrecoverable_by_construction():
    """Adding a trend to the delay changes nothing the system can see."""
    rng = np.random.default_rng(1)
    net = _chain(n_epochs=40, lags=(1, 2))
    a = rng.normal(0, 1.0, net.n_epochs)
    t = net.times
    plain = invert_path_delay(_observe(net, a, rate=12.0), net.pairs, t, lam=0.0)
    tilted = invert_path_delay(_observe(net, a + 5.0 - 90.0 * t, rate=12.0),
                               net.pairs, t, lam=0.0)
    assert np.allclose(plain.delay, tilted.delay, atol=1e-6)


def test_correct_leaves_pure_steady_motion():
    rng = np.random.default_rng(2)
    net = _chain(n_epochs=45, lags=(1, 2))
    a = pin_affine(rng.normal(0, 3.0, net.n_epochs), net.times)
    obs = _observe(net, a, rate=25.0)

    pd = invert_path_delay(obs, net.pairs, net.times, lam=0.0)
    left = pd.correct(obs)
    dt = np.array([net.times[j] - net.times[i] for i, j in net.pairs])
    assert np.allclose(left, 25.0 * dt, atol=1e-6)


def test_a_delay_trend_is_absorbed_into_the_rate():
    """The affine part it cannot see comes straight off the velocity."""
    rng = np.random.default_rng(2)
    net = _chain(n_epochs=45, lags=(1, 2))
    a = pin_affine(rng.normal(0, 3.0, net.n_epochs), net.times)
    dt = np.array([net.times[j] - net.times[i] for i, j in net.pairs])

    trend = 6.0
    obs = _observe(net, a + trend * net.times, rate=25.0)
    left = invert_path_delay(obs, net.pairs, net.times, lam=0.0).correct(obs)
    assert np.allclose(left, (25.0 + trend) * dt, atol=1e-6)


def test_pair_delay_matches_the_epoch_differences():
    net = _chain(n_epochs=20, lags=(1, 2))
    rng = np.random.default_rng(5)
    pd = invert_path_delay(_observe(net, rng.normal(0, 1, net.n_epochs), rate=3.0),
                           net.pairs, net.times, lam=0.0)
    expect = np.array([pd.delay[j] - pd.delay[i] for i, j in net.pairs])
    assert np.allclose(pd.pair_delay(), expect)


def test_it_removes_the_scatter_of_apparent_velocity_not_its_mean():
    """The reduction this buys is in the fluctuations, not in the rate."""
    rng = np.random.default_rng(7)
    net = _chain(n_epochs=80, lags=(1, 2))
    a = np.cumsum(rng.normal(0, 0.5, net.n_epochs))        # a wandering delay
    rate = 40.0
    obs = _observe(net, a, rate=rate)
    dt = np.array([net.times[j] - net.times[i] for i, j in net.pairs])

    pd = invert_path_delay(obs, net.pairs, net.times, lam=1e-6)
    before, after = obs / dt, pd.correct(obs) / dt
    assert after.std() < 0.05 * before.std()

    # the mean moves only by the trend the inversion could not see
    trend = float(np.polyfit(net.times, a, 1)[0])
    assert after.mean() == pytest.approx(rate + trend, rel=1e-3)


# --------------------------------------------------------------- many pixels
def test_pixels_are_solved_together_and_one_at_a_time_alike():
    rng = np.random.default_rng(11)
    net = _chain(n_epochs=30, lags=(1, 2))
    a = rng.normal(0, 1.5, (net.n_epochs, 4, 3))
    obs = np.stack([a[j] - a[i] for i, j in net.pairs])
    obs = obs + 8.0 * np.array([net.times[j] - net.times[i]
                                for i, j in net.pairs])[:, None, None]

    field = invert_path_delay(obs, net.pairs, net.times, lam=1e-3)
    assert field.delay.shape == (net.n_epochs, 4, 3)
    for r in range(4):
        for c in range(3):
            one = invert_path_delay(obs[:, r, c], net.pairs, net.times, lam=1e-3)
            assert np.allclose(field.delay[:, r, c], one.delay, atol=1e-8)


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(12)
    net = _chain(n_epochs=20)
    obs = rng.normal(0, 1, (net.n_pairs, 50))
    whole = tikhonov(double_difference(net.pairs, net.times).A,
                     double_difference(net.pairs, net.times).T @ obs, 1e-2)
    chunked = tikhonov(double_difference(net.pairs, net.times).A,
                       double_difference(net.pairs, net.times).T @ obs, 1e-2,
                       chunk=7)
    assert np.allclose(whole, chunked)


# ------------------------------------------------------------ regularisation
def test_tikhonov_shrinks_as_lambda_grows():
    rng = np.random.default_rng(13)
    net = _chain(n_epochs=30, lags=(1, 2))
    sys = double_difference(net.pairs, net.times)
    b = rng.normal(0, 1, sys.n_rows)
    norms = [np.linalg.norm(tikhonov(sys.A, b, lam))
             for lam in (0.0, 1e-2, 1.0, 1e2, 1e12)]
    assert np.all(np.diff(norms) <= 1e-12)
    assert norms[-1] < 1e-3 * norms[0]


def test_negative_lambda_is_refused():
    net = _chain(n_epochs=10)
    sys = double_difference(net.pairs, net.times)
    with pytest.raises(ValueError, match="lam must be"):
        tikhonov(sys.A, np.zeros(sys.n_rows), -1.0)


@pytest.mark.parametrize("method", ["gcv", "lcurve"])
def test_lambda_selection_beats_both_extremes_on_noisy_data(method):
    rng = np.random.default_rng(17)
    net = _chain(n_epochs=60, lags=(1,))
    a = rng.normal(0, 2.0, net.n_epochs)
    obs = _observe(net, a, rate=30.0)
    obs = obs + rng.normal(0, 1.0, obs.shape)              # phase noise
    truth = pin_affine(a, net.times)

    sys = double_difference(net.pairs, net.times)
    b = sys.apply(obs)
    lam, curve = select_lambda(sys.A, b, method=method)
    assert lam in set(curve["lams"])

    def err(l):
        return np.sqrt(np.mean((pin_affine(tikhonov(sys.A, b, l), net.times)
                                - truth) ** 2))
    assert err(lam) < err(0.0)
    assert err(lam) < err(float(curve["lams"][-1]))
    best = min(err(float(l)) for l in curve["lams"])
    assert err(lam) < 1.5 * best


def test_select_lambda_rejects_an_unknown_method():
    net = _chain(n_epochs=12)
    sys = double_difference(net.pairs, net.times)
    with pytest.raises(ValueError, match="unknown method"):
        select_lambda(sys.A, np.zeros(sys.n_rows), method="magic")


def test_lambda_is_chosen_when_none_is_given():
    rng = np.random.default_rng(19)
    net = _chain(n_epochs=40, lags=(1, 2))
    obs = _observe(net, rng.normal(0, 1, net.n_epochs), rate=9.0)
    pd = invert_path_delay(obs, net.pairs, net.times)
    assert pd.lam > 0


# ------------------------------------------------------------------- pinning
def test_pin_affine_removes_offset_and_trend_and_is_idempotent():
    t = np.linspace(0, 1, 40)
    rng = np.random.default_rng(23)
    a = rng.normal(0, 1, 40) + 4.0 - 2.5 * t
    pinned = pin_affine(a, t)
    assert abs(pinned.mean()) < 1e-9
    assert abs(float(np.polyfit(t, pinned, 1)[0])) < 1e-9
    assert np.allclose(pin_affine(pinned, t), pinned)


def test_pin_affine_works_along_a_chosen_axis():
    t = np.linspace(0, 1, 12)
    a = np.tile((3.0 - 2.0 * t)[:, None], (1, 5))
    assert np.allclose(pin_affine(a, t), 0.0, atol=1e-9)
    assert np.allclose(pin_affine(a.T, t, axis=1), 0.0, atol=1e-9)


def test_unpinned_solution_keeps_the_regularised_offset():
    rng = np.random.default_rng(29)
    net = _chain(n_epochs=30, lags=(1, 2))
    obs = _observe(net, rng.normal(0, 1, net.n_epochs), rate=5.0)
    free = invert_path_delay(obs, net.pairs, net.times, lam=1e-3, pin=False)
    assert not free.pinned
    assert not np.allclose(free.delay.mean(), 0.0, atol=1e-12)


# -------------------------------------------------------- frequency response
def test_response_is_unity_without_regularisation():
    assert frequency_response(1.0, CADENCE) == pytest.approx(1.0)
    assert frequency_response(2 * CADENCE, CADENCE) == pytest.approx(1.0)


def test_regularisation_suppresses_slow_components_first():
    fast = frequency_response(2 * CADENCE, CADENCE, lam=1.0)
    diurnal = frequency_response(1.0, CADENCE, lam=1.0)
    assert fast > 0.99
    assert diurnal < 0.02
    periods = np.array([2 * CADENCE, 0.01, 0.1, 1.0])
    gains = frequency_response(periods, CADENCE, lam=1.0)
    assert np.all(np.diff(gains) < 0)


def test_path_delay_reports_its_own_response():
    net = _chain(n_epochs=30, lags=(1, 2))
    rng = np.random.default_rng(31)
    pd = invert_path_delay(_observe(net, rng.normal(0, 1, net.n_epochs), rate=2.0),
                           net.pairs, net.times, lam=1.0)
    assert pd.frequency_response(1.0) == pytest.approx(
        frequency_response(1.0, CADENCE, lam=1.0), rel=1e-6)


# -------------------------------------------------------- weights and gaps
def test_per_pair_weights_are_folded_onto_the_rows():
    rng = np.random.default_rng(37)
    net = _chain(n_epochs=25, lags=(1, 2))
    a = rng.normal(0, 1, net.n_epochs)
    obs = _observe(net, a, rate=4.0)
    obs[3] += 50.0                                        # one wrecked pair
    w = np.ones(net.n_pairs)
    w[3] = 0.0

    bad = invert_path_delay(obs, net.pairs, net.times, lam=1e-6)
    good = invert_path_delay(obs, net.pairs, net.times, lam=1e-6, weights=w)
    truth = pin_affine(a, net.times)
    assert (np.sqrt(np.mean((good.delay - truth) ** 2))
            < np.sqrt(np.mean((bad.delay - truth) ** 2)))


def test_wrong_sized_weights_are_refused():
    net = _chain(n_epochs=15)
    obs = _observe(net, np.zeros(net.n_epochs), rate=1.0)
    with pytest.raises(ValueError, match="weights has"):
        invert_path_delay(obs, net.pairs, net.times, lam=1.0,
                          weights=np.ones(3))


def test_nan_policy_controls_what_happens_to_gaps():
    rng = np.random.default_rng(41)
    net = _chain(n_epochs=30, lags=(1, 2))
    obs = _observe(net, rng.normal(0, 1, net.n_epochs), rate=6.0)
    obs[5] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        invert_path_delay(obs, net.pairs, net.times, lam=1.0, nan_policy="raise")

    dropped = invert_path_delay(obs, net.pairs, net.times, lam=1.0)
    assert dropped.dropped_rows > 0
    assert np.isfinite(dropped.delay).all()

    zeroed = invert_path_delay(obs, net.pairs, net.times, lam=1.0,
                               nan_policy="zero")
    assert np.isfinite(zeroed.delay).all()
    assert not np.allclose(zeroed.delay, dropped.delay)


def test_mismatched_observation_count_is_refused():
    net = _chain(n_epochs=12)
    with pytest.raises(ValueError, match="observations for"):
        invert_path_delay(np.zeros(3), net.pairs, net.times, lam=1.0)


# --------------------------------------------------------------- the network
def test_it_runs_straight_off_a_network_object():
    net = _chain(n_epochs=35, lags=(1, 2))
    rng = np.random.default_rng(43)
    a = rng.normal(0, 1.0, net.n_epochs)
    obs = _observe(net, a, rate=15.0)

    pd = invert_path_delay(obs, net.pairs, net.times, lam=0.0)
    assert pd.n_epochs == net.n_epochs
    assert np.allclose(pd.delay, pin_affine(a, net.times), atol=1e-6)
    assert isinstance(pd.system, DoubleDifference)
    assert pd.system.null_space().shape == (net.n_epochs, 2)


# --------------------------------------------- pinning so a rate cannot move
def test_pin_rate_leaves_every_pair_velocity_alone():
    rng = np.random.default_rng(47)
    net = _chain(n_epochs=50, lags=(1, 2))
    dt = np.array([net.times[j] - net.times[i] for i, j in net.pairs])
    a = rng.normal(0, 2.0, net.n_epochs)
    obs = _observe(net, a, rate=18.0)

    pinned = invert_path_delay(obs, net.pairs, net.times, lam=1e-6, pin="rate")
    before, after = obs / dt, pinned.correct(obs) / dt
    assert after.mean() == pytest.approx(before.mean(), abs=1e-9)
    assert after.std() < 0.05 * before.std()
    assert pinned.pinned == "rate"


def test_pin_rate_survives_uneven_spacing():
    rng = np.random.default_rng(53)
    t0 = datetime(2017, 8, 3, 22, 0)
    steps = np.cumsum(rng.integers(1, 7, 40)) * CADENCE
    epochs = [t0 + timedelta(days=float(s)) for s in np.r_[0.0, steps]]
    net = Network(epochs, [(i, i + 1) for i in range(len(epochs) - 1)])
    dt = np.array([net.times[j] - net.times[i] for i, j in net.pairs])
    obs = _observe(net, rng.normal(0, 2.0, net.n_epochs), rate=9.0)

    kept = invert_path_delay(obs, net.pairs, net.times, lam=1e-6, pin="rate")
    assert (kept.correct(obs) / dt).mean() == pytest.approx((obs / dt).mean(),
                                                            abs=1e-9)


def test_affine_pinning_can_move_the_rate_and_rate_pinning_cannot():
    """The two representatives differ by exactly the discarded trend."""
    rng = np.random.default_rng(59)
    t0 = datetime(2017, 8, 3, 22, 0)
    steps = np.cumsum(rng.integers(1, 9, 30)) * CADENCE
    epochs = [t0 + timedelta(days=float(s)) for s in np.r_[0.0, steps]]
    net = Network(epochs, [(i, i + 1) for i in range(len(epochs) - 1)])
    dt = np.array([net.times[j] - net.times[i] for i, j in net.pairs])
    obs = _observe(net, rng.normal(0, 3.0, net.n_epochs), rate=20.0)

    loose = invert_path_delay(obs, net.pairs, net.times, lam=1e-6)
    tight = invert_path_delay(obs, net.pairs, net.times, lam=1e-6, pin="rate")
    shift = discarded_rate(loose.delay, net.times, net.pairs)
    assert abs(shift) > 1e-6
    assert ((loose.correct(obs) / dt).mean()
            == pytest.approx((tight.correct(obs) / dt).mean() - shift, abs=1e-9))


def test_discarded_rate_is_zero_once_it_has_been_pinned():
    rng = np.random.default_rng(61)
    net = _chain(n_epochs=40, lags=(1, 2))
    obs = _observe(net, rng.normal(0, 1.0, net.n_epochs), rate=7.0)
    pinned = invert_path_delay(obs, net.pairs, net.times, lam=1e-4, pin="rate")
    assert discarded_rate(pinned.delay, net.times, net.pairs) == pytest.approx(0.0,
                                                                              abs=1e-9)


def test_pin_rate_handles_a_stack_of_pixels():
    rng = np.random.default_rng(67)
    net = _chain(n_epochs=25, lags=(1, 2))
    a = rng.normal(0, 1.0, (net.n_epochs, 3, 2))
    obs = np.stack([a[j] - a[i] for i, j in net.pairs])
    pinned = pin_rate(a, net.times, net.pairs)
    assert pinned.shape == a.shape
    assert np.allclose(discarded_rate(pinned, net.times, net.pairs), 0.0, atol=1e-9)
