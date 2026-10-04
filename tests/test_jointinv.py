"""The joint path / ice-motion inversion on synthetic campaigns with known answers."""
import numpy as np
import pytest

from gpri_tools import jointinv as J

RNG = np.random.default_rng(7)


def _cells(n, xlim, ylim, r_of, z_of, nt, n_px=20, rng=RNG):
    xy = np.column_stack([rng.uniform(*xlim, n), rng.uniform(*ylim, n)])
    r = r_of(xy)
    z = z_of(xy)
    zeros = np.zeros((nt, n))
    return J.CellSet(np.arange(n, dtype=np.int64), xy, r, z, np.full(n, float(n_px)),
                     zeros.copy(), zeros.copy())


def _campaign(nt=73, span=1.5, amp=8.0, peak_frac=0.2, path_sd=3.0, noise_px=4.0,
              n_rock=50, n_ice=30, n_held=25, n_cand=0, cand_moving=None, rng=RNG):
    """Rock on the left, ice on the right; a smooth path field every epoch."""
    t = np.linspace(0.0, span, nt)
    r_of = lambda xy: np.hypot(xy[:, 0], xy[:, 1])
    z_of = lambda xy: 1500.0 + 0.15 * (r_of(xy) - 5000.0)
    fit = _cells(n_rock, (4000, 6000), (-1500, 1500), r_of, z_of, nt, rng=rng)
    held = _cells(n_held, (4000, 6000), (-1500, 1500), r_of, z_of, nt, rng=rng)
    ice = _cells(n_ice, (5500, 7500), (-1500, 1500), r_of, z_of, nt, rng=rng)
    cand = _cells(n_cand, (6000, 7500), (-1500, 1500), r_of, z_of, nt, rng=rng)
    allc = (fit, held, ice, cand)
    XY = np.vstack([c.xy for c in allc])
    R = np.concatenate([c.r for c in allc])
    K = J.PathPrior(length=1500.0).covariance(XY, R)
    Lk = np.linalg.cholesky(K + 1e-9 * np.eye(len(K)))
    w = 2 * np.pi
    truth_ice = amp * np.cos(w * t - 2 * np.pi * peak_frac)[:, None] * np.ones(ice.n)[None] \
        + 0.02 * 1000 * t[:, None]                               # 20 mm/day of flow too
    truth_cand = np.zeros((nt, n_cand))
    if cand_moving is not None:
        truth_cand[:, cand_moving] = 15.0 * np.cos(w * t)[:, None] + 30.0 * t[:, None]
    path = np.zeros((nt, len(R)))
    for k in range(nt):
        trend = rng.normal(0, 2.0) + rng.normal(0, 1.0) * (R - 6000) / 1000
        path[k] = trend + path_sd * (Lk @ rng.normal(size=len(R)))
    path -= path[0]
    o = np.cumsum([0] + [c.n for c in allc])
    motion = {id(ice): truth_ice - truth_ice[0], id(cand): truth_cand}
    for i, c in enumerate(allc):
        base = path[:, o[i]:o[i + 1]] + motion.get(id(c), 0.0)
        for att in ("yu", "yl"):
            n = rng.normal(0, noise_px / np.sqrt(c.n_px), size=(nt, c.n)).cumsum(axis=0) * 0.1
            setattr(c, att, base + n - n[0])
    return J.Campaign(t, fit, ice, cand, held, origin_hour=0.0), truth_ice


def test_group_cells_keys_are_a_fixed_lattice():
    x = np.array([10.0, 150.0, 250.0, 260.0, -30.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 10.0])
    groups, keys = J.group_cells(np.ones(5, bool), x, y, size=200.0)
    assert sum(g.size for g in groups) == 5
    assert len(groups) == 3                      # [-30], [10, 150], [250, 260]
    _, keys2 = J.group_cells(np.ones(5, bool), x + 5.0, y, size=200.0)
    np.testing.assert_array_equal(keys, keys2)   # a small shift keeps the lattice
    groups, _ = J.group_cells(np.ones(5, bool), x, y, size=200.0, min_pixels=2)
    assert len(groups) == 2


def test_cell_series_is_relative_to_the_first_epoch():
    s = np.arange(24, dtype=float).reshape(3, 2, 4)
    groups = [np.array([0, 1]), np.array([4, 5, 6, 7])]
    out = J.cell_series(s, groups)
    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[2], [16.0, 16.0])


def test_a_cell_missing_at_an_epoch_is_dropped_not_zero_filled():
    s = np.arange(24, dtype=float).reshape(3, 2, 4)
    s[0, 0, :2] = np.nan                         # cell 0 has no pixel at epoch 0
    s[1, 1, :] = np.nan                          # cell 1 has none at epoch 1
    groups = [np.array([0, 1]), np.array([4, 5, 6, 7]), np.array([2, 3])]
    out = J.cell_series(s, groups)
    assert np.isnan(out[:, 0]).all()
    assert np.isnan(out[1, 1]) and np.isfinite(out[[0, 2], 1]).all()
    np.testing.assert_array_equal(J.complete_cells(out, out), [False, False, True])
    x = np.arange(8.0)
    cs = J.CellSet.build(groups, np.array([10, 11, 12]), x, x, x, x, out, out)
    np.testing.assert_array_equal(cs.key, [12])
    assert cs.yu.shape == (3, 1) and np.isfinite(cs.yu).all()
    np.testing.assert_allclose(cs.xy, [[2.5, 2.5]])


def test_split_cells_assigns_whole_cells():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 2000, 4000), rng.uniform(0, 2000, 4000)
    mask = rng.uniform(size=4000) < 0.8
    a, b = J.split_cells(mask, x, y, size=200.0, seed=0)
    assert not (a & b).any()
    np.testing.assert_array_equal(a | b, mask)
    _, ka = J.group_cells(a, x, y, 200.0)
    _, kb = J.group_cells(b, x, y, 200.0)
    assert not np.intersect1d(ka, kb).size
    assert abs(ka.size - kb.size) <= 1
    a2, _ = J.split_cells(mask, x, y, size=200.0, seed=0)
    np.testing.assert_array_equal(a, a2)


def test_antenna_noise_reads_the_per_pixel_level_and_each_cell():
    rng = np.random.default_rng(1)
    nt, n = 400, 60
    n_px = np.full(n, 25.0)
    level = np.where(np.arange(n) < 30, 1.0, 3.0)        # second half three times noisier
    common = rng.normal(size=(nt, n)).cumsum(0)
    steps_u = rng.normal(size=(nt, n)) * 2.0 * level / np.sqrt(n_px)
    steps_l = rng.normal(size=(nt, n)) * 2.0 * level / np.sqrt(n_px)
    yu, yl = common + steps_u, common + steps_l
    alpha, rho = J.antenna_noise(yu, yl, n_px, reference=np.arange(30))
    assert np.median(alpha) == pytest.approx(4.0, rel=0.15)        # per-pixel variance 2^2
    assert np.median(rho[30:]) == pytest.approx(9.0, rel=0.2)


def test_solve_recovers_the_ice_harmonic_through_a_path_field():
    camp, truth = _campaign()
    prior = J.PathPrior(length=1500.0)
    res = J.solve(camp, prior)
    amp, sigma, peak = res.harmonic(camp.origin_hour)
    assert np.median(amp) == pytest.approx(8.0, abs=1.0)
    assert np.median(peak) == pytest.approx(0.2 * 24, abs=1.0)
    rate = res.ice_theta[1]
    assert np.median(rate) == pytest.approx(20.0, abs=3.0)
    # the path predicted at rock the solve never saw explains most of it
    obs = 0.5 * (camp.held.yu + camp.held.yl)
    assert np.std(res.held_residual) < 0.5 * np.std(obs)


def test_a_shared_signal_in_the_trend_does_not_leak_into_the_ice():
    camp, _ = _campaign()
    res = J.solve(camp, J.PathPrior(length=1500.0))
    inj, got = J.inject_shared(camp, res, lambda r, z: 10.0 * (z - 1500.0) / 300.0)
    assert np.max(np.abs(got)) < 0.05 * np.max(np.abs(inj)) + 0.05


def test_shared_stationarity_separates_still_from_moving_ground():
    rng = np.random.default_rng(3)
    keys = np.arange(80)
    moving = keys >= 50
    obs = []
    for _ in range(4):                              # four campaigns see the same ground
        spike = np.full((80, 3), 1.0)
        th = rng.normal(0, 1.0, size=(3, 80))
        th[:, moving] += rng.normal(0, 6.0, size=(3, moving.sum()))
        obs.append((keys, th, spike))
    k, p, share, slab = J.shared_stationarity(obs, n_iter=100)
    assert np.mean((p > 0.5) == ~moving) > 0.9
    assert share == pytest.approx(50 / 80, abs=0.15)


def test_stationary_candidates_act_as_reference():
    """One campaign cannot tell them apart reliably; three sharing the locations can."""
    prior = J.PathPrior(length=1500.0)
    camps, frees, obs = [], [], []
    for seed in (11, 12, 13):
        camp, _ = _campaign(n_cand=30, cand_moving=np.arange(15, 30), rng=np.random.default_rng(seed))
        spike, kappa, cal = J.calibrated_spike(camp, prior)
        assert np.all(kappa >= 1.0)
        free = J.solve(camp, prior)
        camps.append((camp, spike)); frees.append(free)
        obs.append((camp.cand.key, cal.cand_theta[1:, :camp.cand.n], spike))
    k, p, _, _ = J.shared_stationarity(obs)
    assert np.mean(p[:15] > 0.5) > 0.8 and np.mean(p[15:] < 0.5) > 0.8
    camp, spike = camps[0]
    used = J.solve(camp, prior, cand_p=p, cand_spike=spike)
    # stationary candidates are pulled to zero motion, moving ones keep theirs
    assert np.median(np.abs(used.cand_theta[1, :15])) < np.median(np.abs(frees[0].cand_theta[1, :15]))
    assert np.median(np.abs(used.cand_theta[2, 15:])) == pytest.approx(15.0, abs=4.0)


def test_fit_path_prior_finds_the_length_scale():
    camp, _ = _campaign(nt=40, n_rock=60, path_sd=4.0)
    alpha, rho = J.antenna_noise(camp.fit.yu, camp.fit.yl, camp.fit.n_px)
    prior = J.fit_path_prior(camp.fit, alpha, rho, lengths=(300, 1500, 6000), growths=(0.0,),
                             commons=(0.0,), n_epochs=30)
    assert prior.length == 1500


def test_shared_noise_finds_the_cells_with_extra_common_noise():
    rng = np.random.default_rng(5)
    nt, n = 300, 80
    xy = np.column_stack([rng.uniform(0, 3000, n), rng.uniform(0, 3000, n)])
    n_px = np.full(n, 20.0)
    smooth = np.cumsum(rng.normal(size=(nt, 1)), axis=0) * np.ones((1, n))     # the path: everywhere alike
    extra = np.where(np.arange(n) >= 40, 3.0, 0.0)                            # shared noise, half the cells
    common = rng.normal(size=(nt, n)).cumsum(0) * np.sqrt(extra / n_px)[None]
    yu = smooth + common + rng.normal(size=(nt, n)).cumsum(0) / np.sqrt(n_px)[None]
    yl = smooth + common + rng.normal(size=(nt, n)).cumsum(0) / np.sqrt(n_px)[None]
    c = J.CellSet(np.arange(n), xy, np.full(n, 6000.0), np.full(n, 2000.0), n_px, yu, yl)
    sh = J.shared_noise([c], radius=1500.0)
    assert np.median(sh[:40]) < 0.5
    assert np.median(sh[40:]) == pytest.approx(3.0, rel=0.4)


def test_stationarity_probability_reproduces_the_fit_on_its_own_data():
    rng = np.random.default_rng(9)
    keys = np.arange(60)
    obs = []
    for _ in range(3):
        th = rng.normal(0, 1.0, size=(3, 60))
        th[:, 30:] += rng.normal(0, 6.0, size=(3, 30))
        obs.append((keys, th, np.ones((60, 3))))
    k, p, share, slab = J.shared_stationarity(obs, n_iter=100)
    k2, p2 = J.stationarity_probability(obs, share, slab)
    np.testing.assert_array_equal(k, k2)
    np.testing.assert_allclose(p, p2, atol=1e-6)
