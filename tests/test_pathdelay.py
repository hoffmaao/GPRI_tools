"""Double-difference path delay: what it recovers, and what it cannot see."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from gpri_tools.network import Network
from gpri_tools.pathdelay import (DoubleDifference, PathDelay, discarded_rate,
                                  displacement_delay_field, double_difference,
                                  double_difference_row_weights,
                                  pair_delay_field, pair_variance_from_coherence,
                                  frequency_response,
                                  gls_path_delay, gls_resolution,
                                  invert_path_delay, joint_design,
                                  lambda_for_response,
                                  lambda_for_system_response, pin_affine,
                                  pin_rate, response_from_resolution,
                                  rewrap_to_chain, select_lambda,
                                  shared_epoch_triplets, system_response,
                                  tikhonov)

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


def test_max_span_without_times_is_refused():
    net = _chain(n_epochs=10, lags=(1, 3))
    with pytest.raises(ValueError, match="needs times"):
        shared_epoch_triplets(net.pairs, max_span=2.5 * CADENCE)


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


def test_pin_affine_pins_pixels_beside_one_that_is_never_finite():
    """An incoherent pixel must not leave the whole cube unpinned."""
    t = np.linspace(0, 1, 5)
    a = np.zeros((5, 2, 2))
    a[:, 0, 0] = np.nan                     # masked at every epoch
    a[:, 1, 1] = 3.0 * t + 7.0
    out = pin_affine(a, t)
    assert np.allclose(out[:, 1, 1], 0.0, atol=1e-9)
    assert np.all(np.isnan(out[:, 0, 0]))


def test_pin_affine_fits_each_pixel_on_its_own_finite_epochs():
    rng = np.random.default_rng(11)
    t = np.linspace(0, 1, 24)
    a = np.column_stack([2.0 - 0.5 * t, rng.normal(0, 1.0, t.size) + 4.0 * t])
    a[:12, 0] = np.nan                      # a gap in one pixel only
    out = pin_affine(a, t)
    for col in range(a.shape[1]):
        ok = np.isfinite(out[:, col])
        assert abs(float(out[ok, col].mean())) < 1e-9
        assert abs(float(np.polyfit(t[ok], out[ok, col], 1)[0])) < 1e-9


def test_pin_affine_takes_the_mean_off_a_single_epoch_pixel():
    t = np.linspace(0, 1, 6)
    a = np.full((6, 2), np.nan)
    a[2, 0] = 5.0                           # one finite epoch: no trend to fit
    a[:, 1] = 1.0 + 2.0 * t
    out = pin_affine(a, t)
    assert out[2, 0] == 0.0
    assert np.allclose(out[:, 1], 0.0, atol=1e-9)


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


# ------------------------------------------- keeping the correction off a band
def test_lambda_for_response_hits_the_response_it_promises():
    for period in (1.0, 0.5, 1 / 6):
        for rho in (0.5, 0.01, 1e-4):
            lam = lambda_for_response(period, CADENCE, rho)
            assert frequency_response(period, CADENCE, lam) == pytest.approx(rho)


def test_protecting_a_slow_period_is_cheap_at_fast_ones():
    lam = lambda_for_response(1.0, CADENCE, 0.01)
    assert frequency_response(1 / 12, CADENCE, lam) > 0.99      # 2 h
    assert frequency_response(1 / 144, CADENCE, lam) > 0.999    # 10 min
    assert frequency_response(0.5, CADENCE, lam) < 0.2          # 12 h is not free


def test_bad_response_targets_are_refused():
    for rho in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="max_response"):
            lambda_for_response(1.0, CADENCE, rho)


def test_protect_period_raises_lambda_but_never_lowers_it():
    rng = np.random.default_rng(71)
    net = _chain(n_epochs=60, lags=(1, 2))
    obs = _observe(net, rng.normal(0, 2.0, net.n_epochs), rate=11.0)

    free = invert_path_delay(obs, net.pairs, net.times)
    kept = invert_path_delay(obs, net.pairs, net.times, protect_period=1.0)
    assert kept.lam >= free.lam
    assert system_response(kept.system.A, net.times, 1.0, kept.lam) <= 0.0101

    # a lam already above the floor is left alone
    huge = invert_path_delay(obs, net.pairs, net.times, lam=1e9, protect_period=1.0)
    assert huge.lam == 1e9


def test_protecting_the_diurnal_leaves_a_diurnal_delay_in_place():
    """The correction must not take out what it was told to protect."""
    rng = np.random.default_rng(73)
    net = _chain(n_epochs=400, lags=(1, 2))          # 400 x 4 min = 26.7 h
    t = net.times
    diurnal = 6.0 * np.sin(2 * np.pi * t / 1.0)
    fast = rng.normal(0, 1.0, net.n_epochs)
    obs = _observe(net, diurnal + fast, rate=15.0)

    kept = invert_path_delay(obs, net.pairs, net.times, protect_period=1.0,
                             pin="rate")
    left = obs - kept.pair_delay()
    # what the correction removed, projected back onto the diurnal it protected
    removed = kept.delay
    w = 2 * np.pi
    G = np.column_stack([np.ones_like(t), t, np.cos(w * t), np.sin(w * t)])
    c, *_ = np.linalg.lstsq(G, removed, rcond=None)
    assert 2 * np.hypot(c[2], c[3]) < 0.05 * 2 * 6.0     # under a twentieth
    # and it still took the fast part out
    assert np.std(left) < np.std(obs)


def test_system_response_exceeds_the_single_baseline_formula():
    """Why the floor is measured on A: mixed baselines let more through."""
    net = _chain(n_epochs=200, lags=(1, 2, 3))
    A = double_difference(net.pairs, net.times).A
    lam = lambda_for_response(1.0, CADENCE, 0.01)         # the closed form
    assert system_response(A, net.times, 1.0, lam) > 0.01


def test_system_response_takes_a_period_array():
    net = _chain(n_epochs=30, lags=(1, 2))
    A = double_difference(net.pairs, net.times).A
    periods = np.array([1 / 24, 1 / 12, 0.5, 1.0])
    curve = system_response(A, net.times, periods, 1.0)
    assert curve.shape == periods.shape
    assert np.allclose(curve, [system_response(A, net.times, T, 1.0)
                               for T in periods])
    assert np.all(np.diff(curve) < 0)                    # falls with period


def test_lambda_for_system_response_is_the_smallest_that_works():
    net = _chain(n_epochs=200, lags=(1, 2))
    A = double_difference(net.pairs, net.times).A
    lam = lambda_for_system_response(A, net.times, 1.0, 0.01)
    assert system_response(A, net.times, 1.0, lam) <= 0.0101
    assert system_response(A, net.times, 1.0, lam / 2.0) > 0.01


def test_no_regularisation_is_needed_when_the_target_is_already_met():
    net = _chain(n_epochs=30, lags=(1,))
    A = double_difference(net.pairs, net.times).A
    assert lambda_for_system_response(A, net.times, 1.0, 1.0) == 0.0


# ------------------------------------------------- the displacement-cube path
def _cube(net, rng, shape=(12, 9)):
    """A displacement cube: steady motion + a delay that is smooth in space."""
    t = net.times
    az = np.linspace(-1, 1, shape[0])[:, None]
    rg = np.linspace(-1, 1, shape[1])[None, :]
    smooth = np.exp(-(az ** 2 + rg ** 2))
    delay = np.array([a * smooth for a in rng.normal(0, 1.0, net.n_epochs)])
    motion = 20.0 * t[:, None, None] * np.ones((1,) + shape)
    return motion + delay, delay


def test_displacement_delay_field_finds_the_coherent_part():
    rng = np.random.default_rng(83)
    net = _chain(n_epochs=60, lags=(1, 2))
    d, delay = _cube(net, rng)
    mask = np.ones(d.shape[1:], bool)

    field, lam = displacement_delay_field(d, net.pairs, net.times, mask,
                                          sigma=(1.0, 1.0), protect_period=None)
    assert field.shape == d.shape
    assert lam >= 0
    # what it removes correlates with the delay that was put in
    a = pin_rate(delay, net.times, net.pairs).ravel()
    b = field.ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.5


def test_displacement_delay_field_cannot_move_a_rate():
    rng = np.random.default_rng(89)
    net = _chain(n_epochs=40, lags=(1, 2))
    d, _ = _cube(net, rng)
    mask = np.ones(d.shape[1:], bool)
    field, _ = displacement_delay_field(d, net.pairs, net.times, mask,
                                        sigma=(1.0, 1.0), protect_period=None)
    assert np.allclose(discarded_rate(field, net.times, net.pairs), 0.0, atol=1e-9)


def test_displacement_delay_field_honours_the_protected_period():
    rng = np.random.default_rng(97)
    net = _chain(n_epochs=400, lags=(1, 2))
    d, _ = _cube(net, rng)
    mask = np.ones(d.shape[1:], bool)
    free, lam_free = displacement_delay_field(d, net.pairs, net.times, mask,
                                              sigma=(1.0, 1.0), protect_period=None)
    kept, lam_kept = displacement_delay_field(d, net.pairs, net.times, mask,
                                              sigma=(1.0, 1.0), protect_period=1.0)
    assert lam_kept >= lam_free


# ----------------------------------------------------- the pair-domain GLS
def test_joint_design_is_the_pair_model():
    net = _chain(n_epochs=6, lags=(1, 2))
    G = joint_design(net.pairs, net.times)
    assert G.shape == (net.n_pairs, net.n_epochs + 1)
    a = np.arange(net.n_epochs, dtype=float) ** 2
    v = 7.0
    predicted = G @ np.r_[a, v]
    expect = np.array([(a[j] - a[i]) + v * (net.times[j] - net.times[i])
                       for i, j in net.pairs])
    assert np.allclose(predicted, expect)


def test_gls_recovers_delay_and_rate_without_noise():
    rng = np.random.default_rng(101)
    net = _chain(n_epochs=50, lags=(1, 2))
    a = pin_rate(rng.normal(0, 2.0, net.n_epochs), net.times, net.pairs)
    obs = _observe(net, a, rate=31.0)

    delay, motion, lam = gls_path_delay(obs, net.pairs, net.times, lam=0.0,
                                        protect_period=None)
    assert np.allclose(delay, a, atol=1e-6)
    assert motion == pytest.approx(31.0, rel=1e-6)


def test_gls_matches_the_double_difference_up_to_the_null_space():
    """With equal weights and no regularisation the two see the same thing."""
    rng = np.random.default_rng(103)
    net = _chain(n_epochs=40, lags=(1, 2))
    obs = _observe(net, rng.normal(0, 2.0, net.n_epochs), rate=12.0)

    dd = invert_path_delay(obs, net.pairs, net.times, lam=0.0, pin="rate").delay
    gls, _, _ = gls_path_delay(obs, net.pairs, net.times, lam=0.0,
                               protect_period=None, pin="rate")
    assert np.allclose(dd, gls, atol=1e-6)


def test_gls_cannot_move_a_rate_when_pinned():
    rng = np.random.default_rng(107)
    net = _chain(n_epochs=45, lags=(1, 2))
    obs = _observe(net, rng.normal(0, 2.0, net.n_epochs), rate=9.0)
    delay, _, _ = gls_path_delay(obs, net.pairs, net.times, lam=1e-3,
                                 protect_period=None, pin="rate")
    assert np.allclose(discarded_rate(delay, net.times, net.pairs), 0.0, atol=1e-9)


def test_gls_downweights_a_bad_pair():
    rng = np.random.default_rng(109)
    net = _chain(n_epochs=40, lags=(1, 2))
    a = pin_rate(rng.normal(0, 1.0, net.n_epochs), net.times, net.pairs)
    obs = _observe(net, a, rate=5.0)
    obs[7] += 40.0                                     # one wrecked pair

    var = np.ones(net.n_pairs)
    flat, _, _ = gls_path_delay(obs, net.pairs, net.times, lam=1e-4,
                                protect_period=None)
    var[7] = 1e6
    told, _, _ = gls_path_delay(obs, net.pairs, net.times, variance=var,
                                lam=1e-4, protect_period=None)
    assert (np.sqrt(np.mean((told - a) ** 2))
            < np.sqrt(np.mean((flat - a) ** 2)))


def test_gls_protects_the_period_it_is_told_to():
    rng = np.random.default_rng(113)
    net = _chain(n_epochs=400, lags=(1, 2))
    obs = _observe(net, rng.normal(0, 1.0, net.n_epochs), rate=8.0)
    _, _, lam = gls_path_delay(obs, net.pairs, net.times, protect_period=1.0)
    G = joint_design(net.pairs, net.times)
    R = gls_resolution(G, np.ones(net.n_pairs), lam, net.n_epochs)
    assert response_from_resolution(R, net.times, 1.0) <= 0.0101
    # the pair design differences once, not twice, so protecting a slow period
    # costs more at short ones here than it does on double differences
    assert response_from_resolution(R, net.times, 1 / 12) > 0.2


def test_roughness_prior_is_low_pass_and_says_so():
    """The wrong prior for this job, kept so the choice is visible.

    Compared at matched protection of the diurnal, not at matched lam: the
    two penalties have quite different scales, so only the shape is at issue.
    """
    from gpri_tools.pathdelay import _bisect_lambda
    net = _chain(n_epochs=300, lags=(1, 2))
    G = joint_design(net.pairs, net.times)
    w = np.ones(net.n_pairs)

    def fast_response(penalty):
        lam = _bisect_lambda(
            lambda c: response_from_resolution(
                gls_resolution(G, w, c, net.n_epochs, penalty), net.times, 1.0),
            0.01, 1e14)
        return response_from_resolution(
            gls_resolution(G, w, lam, net.n_epochs, penalty), net.times, 1 / 12)

    assert fast_response("roughness") < 0.1 * fast_response("ridge")


def test_unknown_penalty_is_refused():
    net = _chain(n_epochs=10)
    obs = _observe(net, np.zeros(net.n_epochs), rate=1.0)
    with pytest.raises(ValueError, match="penalty must be"):
        gls_path_delay(obs, net.pairs, net.times, lam=1.0, penalty="magic")


def test_gls_solves_many_pixels_like_one():
    rng = np.random.default_rng(127)
    net = _chain(n_epochs=25, lags=(1, 2))
    a = rng.normal(0, 1.0, (net.n_epochs, 3, 2))
    obs = np.stack([a[j] - a[i] for i, j in net.pairs])
    field, motion, _ = gls_path_delay(obs, net.pairs, net.times, lam=1e-3,
                                      protect_period=None)
    assert field.shape == a.shape and motion.shape == (3, 2)
    for rr in range(3):
        for cc in range(2):
            one, _, _ = gls_path_delay(obs[:, rr, cc], net.pairs, net.times,
                                       lam=1e-3, protect_period=None)
            assert np.allclose(field[:, rr, cc], one, atol=1e-8)


def test_gls_rejects_a_wrong_sized_variance():
    net = _chain(n_epochs=12)
    obs = _observe(net, np.zeros(net.n_epochs), rate=1.0)
    with pytest.raises(ValueError, match="variance has"):
        gls_path_delay(obs, net.pairs, net.times, variance=np.ones(3), lam=1.0)


def test_pair_variance_downweights_a_bad_pair_in_the_field():
    rng = np.random.default_rng(131)
    net = _chain(n_epochs=50, lags=(1, 2))
    a = pin_rate(rng.normal(0, 1.0, net.n_epochs), net.times, net.pairs)
    cube = np.tile(a[:, None, None], (1, 6, 6))
    obs = np.stack([cube[j] - cube[i] for i, j in net.pairs])
    obs[4] += 30.0
    mask = np.ones(obs.shape[1:], bool)

    flat, _ = pair_delay_field(obs, net.pairs, net.times, mask, sigma=(1.0, 1.0),
                               protect_period=None, lam=1e-4)
    var = np.ones(net.n_pairs)
    var[4] = 1e6
    told, _ = pair_delay_field(obs, net.pairs, net.times, mask, sigma=(1.0, 1.0),
                               protect_period=None, lam=1e-4, pair_variance=var)
    truth = np.tile(a[:, None, None], (1, 6, 6))
    assert (np.sqrt(np.nanmean((told - truth) ** 2))
            < np.sqrt(np.nanmean((flat - truth) ** 2)))


def test_pair_variance_of_the_wrong_length_is_refused():
    net = _chain(n_epochs=12)
    obs = _observe(net, np.zeros(net.n_epochs), rate=1.0)[:, None, None]
    with pytest.raises(ValueError, match="pair_variance has"):
        pair_delay_field(obs, net.pairs, net.times, np.ones((1, 1), bool),
                         pair_variance=np.ones(3), lam=1.0)


# ------------------------------------------------- wrapped rows and robust fits
AMBIGUITY = 8.7                                   # mm per cycle, Ku band two-way


def test_rewrap_moves_only_whole_cycles_and_only_off_the_chain():
    rng = np.random.default_rng(140)
    net = _chain(n_epochs=30, lags=(1, 2, 3))
    a = rng.normal(0, 0.5, net.n_epochs)
    obs = _observe(net, a, rate=40.0)[:, None, None] * np.ones((1, 4, 5))
    obs += rng.normal(0, 0.05, obs.shape)         # independent per-pair noise
    long = [p for p, (i, j) in enumerate(net.pairs) if j > i + 1]
    truth = obs.copy()
    wrapped = obs.copy()
    wrapped[long[0], 1, 2] += AMBIGUITY            # one pixel, one cycle up
    wrapped[long[3], 0, 0] -= 2 * AMBIGUITY        # another, two cycles down
    wrapped[long[5], 3, 4] += 0.3 * AMBIGUITY      # less than a cycle: stays

    fixed, moved = rewrap_to_chain(wrapped, net.pairs, AMBIGUITY)
    assert fixed.shape == obs.shape
    assert np.allclose(fixed[long[0], 1, 2], truth[long[0], 1, 2], atol=1e-9)
    assert np.allclose(fixed[long[3], 0, 0], truth[long[3], 0, 0], atol=1e-9)
    assert np.allclose(fixed[long[5], 3, 4], wrapped[long[5], 3, 4])
    assert moved[long[0]] == pytest.approx(1 / 20)
    assert moved[long[3]] == pytest.approx(1 / 20)
    assert moved[long[5]] == 0.0
    chain = [p for p, (i, j) in enumerate(net.pairs) if j == i + 1]
    assert np.array_equal(fixed[chain], wrapped[chain])
    assert wrapped[long[0], 1, 2] != fixed[long[0], 1, 2]   # input untouched


def test_rewrap_leaves_a_pair_with_no_chain_beneath_it():
    net = _chain(n_epochs=8, lags=(1, 2))
    pairs = [tuple(int(x) for x in p) for p in net.pairs]
    pairs.remove((2, 3))                                  # break the chain
    obs = np.zeros(len(pairs))
    k = pairs.index((2, 4))
    obs[k] = AMBIGUITY
    fixed, moved = rewrap_to_chain(obs, pairs, AMBIGUITY)
    assert fixed[k] == AMBIGUITY and moved[k] == 0.0
    j = pairs.index((0, 2))                               # this one is spanned
    obs[j] = AMBIGUITY
    fixed, moved = rewrap_to_chain(obs, pairs, AMBIGUITY)
    assert fixed[j] == 0.0 and moved[j] == 1.0


def test_rewrap_refuses_bad_input():
    net = _chain(n_epochs=6, lags=(1, 2))
    obs = np.zeros(net.n_pairs)
    with pytest.raises(ValueError, match="ambiguity"):
        rewrap_to_chain(obs, net.pairs, 0.0)
    with pytest.raises(ValueError, match="observations for"):
        rewrap_to_chain(obs[:-1], net.pairs, AMBIGUITY)


def test_robust_sweeps_shrug_off_a_wrapped_pair():
    rng = np.random.default_rng(151)
    net = _chain(n_epochs=40, lags=(1, 2))
    a = pin_affine(rng.normal(0, 1.0, net.n_epochs), net.times)
    obs = _observe(net, a, rate=30.0) + rng.normal(0, 0.02, net.n_pairs)
    long = [p for p, (i, j) in enumerate(net.pairs) if j > i + 1]
    obs[long[10]] += AMBIGUITY

    plain = invert_path_delay(obs, net.pairs, net.times, lam=1e-6)
    tough = invert_path_delay(obs, net.pairs, net.times, lam=1e-6, robust=3)
    err = lambda pd: np.sqrt(np.mean((pd.delay - a) ** 2))
    assert tough.robust == 3 and plain.robust == 0
    assert err(tough) < 0.3 * err(plain)


def test_robust_zero_is_the_plain_solve_and_clean_data_stay_put():
    rng = np.random.default_rng(152)
    net = _chain(n_epochs=30, lags=(1, 2))
    a = rng.normal(0, 1.0, net.n_epochs)
    obs = _observe(net, a, rate=5.0) + rng.normal(0, 0.01, net.n_pairs)
    plain = invert_path_delay(obs, net.pairs, net.times, lam=1e-3)
    same = invert_path_delay(obs, net.pairs, net.times, lam=1e-3, robust=0)
    assert np.array_equal(plain.delay, same.delay)
    # gaussian noise only: the sweeps change the answer by a small fraction
    tough = invert_path_delay(obs, net.pairs, net.times, lam=1e-3, robust=2)
    assert np.max(np.abs(tough.delay - plain.delay)) < 0.2 * np.std(plain.delay)


def test_robust_solves_each_pixel_on_its_own():
    """A wrapped pair in one pixel must not touch its neighbours."""
    rng = np.random.default_rng(153)
    net = _chain(n_epochs=30, lags=(1, 2))
    a = pin_affine(rng.normal(0, 1.0, net.n_epochs), net.times)
    one = _observe(net, a, rate=10.0) + rng.normal(0, 0.02, net.n_pairs)
    obs = np.tile(one[:, None], (1, 3))
    long = [p for p, (i, j) in enumerate(net.pairs) if j > i + 1]
    obs[long[4], 1] += AMBIGUITY
    pd = invert_path_delay(obs, net.pairs, net.times, lam=1e-6, robust=3)
    clean = invert_path_delay(one, net.pairs, net.times, lam=1e-6, robust=3)
    assert np.allclose(pd.delay[:, 0], clean.delay, atol=1e-9)
    assert np.allclose(pd.delay[:, 2], clean.delay, atol=1e-9)
    assert np.sqrt(np.mean((pd.delay[:, 1] - a) ** 2)) < 0.1


def test_pair_delay_field_refuses_a_series_with_no_spatial_axis():
    net = _chain(n_epochs=12, lags=(1, 2))
    rng = np.random.default_rng(7)
    obs = rng.normal(0, 1.0, len(net.pairs))
    with pytest.raises(ValueError, match="spatial axis"):
        pair_delay_field(obs, net.pairs, net.times, np.ones((1, 1), bool))


def test_pair_delay_field_takes_the_robust_path():
    rng = np.random.default_rng(154)
    net = _chain(n_epochs=40, lags=(1, 2))
    a = pin_rate(rng.normal(0, 1.0, net.n_epochs), net.times, net.pairs)
    cube = np.tile(a[:, None, None], (1, 5, 5))
    obs = np.stack([cube[j] - cube[i] for i, j in net.pairs])
    obs += rng.normal(0, 0.02, obs.shape)
    long = [p for p, (i, j) in enumerate(net.pairs) if j > i + 1]
    obs[long[6], 2, 2] += AMBIGUITY
    mask = np.ones(obs.shape[1:], bool)
    plain, _ = pair_delay_field(obs, net.pairs, net.times, mask, sigma=(0.5, 0.5),
                                protect_period=None, lam=1e-4)
    tough, _ = pair_delay_field(obs, net.pairs, net.times, mask, sigma=(0.5, 0.5),
                                protect_period=None, lam=1e-4, robust=3)
    err = lambda f: np.sqrt(np.nanmean((f[:, 2, 2] - a) ** 2))
    assert err(tough) < 0.5 * err(plain)


def test_robust_banded_and_dense_solves_agree(monkeypatch):
    from gpri_tools import pathdelay as pdm
    net = _chain(30, lags=(1, 2, 3))
    rng = np.random.default_rng(5)
    delay = rng.standard_normal(30) * 0.4
    obs = np.stack([_observe(net, delay) + 0.05 * rng.standard_normal(len(net.pairs))
                    for _ in range(4)], axis=1)
    obs[7, 2] += AMBIGUITY
    pairs = np.asarray(net.pairs, int)
    sysd = double_difference(pairs, net.times)
    args = (sysd.A, sysd.apply(obs), 1e-3, 2, sysd.rows, sysd.weights, pairs.shape[0])
    x_band, rms_band, w_band = pdm._robust_tikhonov(*args)
    monkeypatch.setattr(pdm, "solveh_banded", None)
    x_dense, rms_dense, w_dense = pdm._robust_tikhonov(*args)
    assert np.allclose(x_band, x_dense, atol=1e-8)
    assert np.allclose(rms_band, rms_dense) and np.allclose(w_band, w_dense)


def test_pair_variance_is_the_cramer_rao_form_of_the_pair_coherence():
    g = np.array([0.6, 0.9, 0.3])
    cc = np.stack([np.full((4, 5), x) for x in g])
    assert np.allclose(pair_variance_from_coherence(cc), (1 - g ** 2) / (2 * g ** 2))


def test_pair_variance_reads_only_the_masked_pixels():
    cc = np.zeros((2, 3, 3), float)
    cc[:, 0] = 0.8                       # the row the mask keeps
    cc[:, 1:] = 0.1                      # incoherent ground the fit never reads
    mask = np.zeros((3, 3), bool)
    mask[0] = True
    v = pair_variance_from_coherence(cc, mask)
    assert np.allclose(v, (1 - 0.8 ** 2) / (2 * 0.8 ** 2))


def test_pair_variance_clips_the_coherence_at_both_ends():
    cc = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
    v = pair_variance_from_coherence(cc, clip=(0.2, 0.95))
    assert np.allclose(v, [(1 - 0.2 ** 2) / (2 * 0.2 ** 2),
                           (1 - 0.95 ** 2) / (2 * 0.95 ** 2)])


def test_row_weights_take_the_worse_of_the_two_pairs():
    net = _chain(n_epochs=6, lags=(1, 2))
    pairs = np.asarray(net.pairs, int)
    sysd = double_difference(pairs, net.times)
    v = np.linspace(0.1, 1.0, pairs.shape[0])
    w = double_difference_row_weights(sysd.rows, v, pairs.shape[0])
    raw = 1.0 / np.maximum(v[sysd.rows[:, 0]], v[sysd.rows[:, 1]])
    assert np.allclose(w, raw / raw.mean())
    assert np.isclose(w.mean(), 1.0)


def test_row_weights_refuse_a_variance_of_the_wrong_length():
    net = _chain(n_epochs=5, lags=(1,))
    sysd = double_difference(np.asarray(net.pairs, int), net.times)
    with pytest.raises(ValueError, match="entries for"):
        double_difference_row_weights(sysd.rows, np.ones(3), len(net.pairs))


def test_weighting_a_field_follows_the_pair_the_variance_trusts():
    """A pair with a large variance is allowed to pull the field less."""
    net = _chain(n_epochs=24, lags=(1, 2))
    rng = np.random.default_rng(11)
    delay = pin_rate(rng.normal(0, 1.0, net.n_epochs), net.times, net.pairs)
    obs = np.stack([np.full((4, 4), delay[j] - delay[i]) for i, j in net.pairs])
    bad = 5                                     # one pair, badly wrong
    obs[bad] += 4.0
    mask = np.ones(obs.shape[1:], bool)
    v = np.full(len(net.pairs), 0.05)
    v[bad] = 50.0
    plain, _ = pair_delay_field(obs, net.pairs, net.times, mask, sigma=(0.5, 0.5),
                                protect_period=None, lam=1e-3)
    weighted, _ = pair_delay_field(obs, net.pairs, net.times, mask, sigma=(0.5, 0.5),
                                   protect_period=None, lam=1e-3, pair_variance=v)
    err = lambda f: np.sqrt(np.nanmean((f[:, 2, 2] - (delay - delay[0])) ** 2))
    assert err(weighted) < err(plain)


def test_displacement_delay_field_passes_the_pair_variance_through():
    net = _chain(n_epochs=12, lags=(1,))
    rng = np.random.default_rng(3)
    cube = rng.normal(0, 1.0, (net.n_epochs, 3, 3))
    mask = np.ones(cube.shape[1:], bool)
    v = np.linspace(0.1, 2.0, len(net.pairs))
    a, _ = displacement_delay_field(cube, net.pairs, net.times, mask,
                                    sigma=(0.5, 0.5), protect_period=None,
                                    lam=1e-3, pair_variance=v)
    b, _ = pair_delay_field(np.stack([cube[j] - cube[i] for i, j in net.pairs]),
                            net.pairs, net.times, mask, sigma=(0.5, 0.5),
                            protect_period=None, lam=1e-3, pair_variance=v)
    assert np.allclose(a, b)
