"""Modes of a residual: what transfers to pixels that were not fitted."""
import numpy as np
import pytest

from gpri_tools.modes import (RockModes, fit_modes, mode_correction,
                              mode_projection, mode_screens)

SHAPE = (24, 40)


def _grid(seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, 120)
    series = np.sin(2 * np.pi * t) - 0.3 * np.cos(4 * np.pi * t)
    fit = rng.random(SHAPE) < 0.5              # interleaved halves
    held = ~fit
    return rng, t, series, fit, held


def _residual(rng, series, loading, noise=0.05):
    return (series[:, None, None] * loading[None]
            + noise * rng.standard_normal((series.size,) + SHAPE))


def test_fit_modes_recovers_a_planted_mode():
    rng, t, series, fit, held = _grid()
    loading = rng.standard_normal(SHAPE)
    d = _residual(rng, series, loading)
    modes = fit_modes(d, fit, k=3)
    assert isinstance(modes, RockModes) and modes.k == 3
    assert modes.n_pixels == fit.sum()
    # one mode far above the null, the next at it
    assert modes.ratio[0] > 3 and modes.ratio[1] < 1.5
    assert modes.explained[0] > 0.9
    r = np.corrcoef(modes.temporal[:, 0], series)[0, 1]
    assert abs(r) > 0.99
    rl = np.corrcoef(modes.loading[0][fit], loading[fit])[0, 1]
    assert abs(rl) > 0.99 and np.sign(rl) == np.sign(r)
    assert np.isnan(modes.loading[0][held]).all()


def test_mode_scaling_is_unit_loading():
    rng, t, series, fit, held = _grid(1)
    loading = 3.0 * rng.standard_normal(SHAPE)
    d = _residual(rng, series, loading, noise=0.0)
    modes = fit_modes(d, fit, k=1)
    assert np.sqrt(np.nanmean(modes.loading[0][fit] ** 2)) == pytest.approx(1.0)
    # series x loading gives back the centred residual on the fitted pixels
    rebuilt = modes.temporal[:, 0][:, None] * modes.loading[0][fit][None, :]
    centred = d[:, fit] - d[:, fit].mean(axis=0)
    assert np.abs(rebuilt - centred).max() < 1e-8
    # the series have no mean: the offset is not a mode
    assert abs(modes.temporal[:, 0].mean()) < 1e-10


def test_loading_scale_is_the_weighted_mean_square_whatever_the_weights():
    rng, t, series, fit, held = _grid(3)
    loading = 2.0 * rng.standard_normal(SHAPE)
    d = _residual(rng, series, loading, noise=0.02)
    for w in (np.full(SHAPE, 0.7), rng.uniform(0.5, 0.9, SHAPE)):
        modes = fit_modes(d, fit, weights=w, k=1)
        L = modes.loading[0][fit]
        assert (w[fit] * L ** 2).sum() / fit.sum() == pytest.approx(1.0)
    # at a constant weight the plain RMS is 1 / sqrt(w), not 1
    modes = fit_modes(d, fit, weights=np.full(SHAPE, 0.7), k=1)
    plain = np.sqrt((modes.loading[0][fit] ** 2).mean())
    assert plain == pytest.approx(1.0 / np.sqrt(0.7))


def test_fit_modes_leaves_out_bad_pixels_and_zero_weights():
    rng, t, series, fit, held = _grid(2)
    loading = rng.standard_normal(SHAPE)
    d = _residual(rng, series, loading)
    d[5, 3, 3] = np.nan
    w = np.ones(SHAPE)
    w[7, 7] = 0.0
    fit[3, 3] = fit[7, 7] = True
    modes = fit_modes(d, fit, weights=w, k=2)
    assert modes.n_pixels == fit.sum() - 2
    assert np.isnan(modes.loading[0][3, 3]) and np.isnan(modes.loading[0][7, 7])
    with pytest.raises(ValueError):
        fit_modes(d, fit[:, :10], k=2)


def test_smooth_loading_transfers_to_the_held_out_half():
    rng, t, series, fit, held = _grid(3)
    x = np.linspace(-1, 1, SHAPE[1])[None, :] * np.ones(SHAPE)
    loading = 1.0 + 0.8 * x                    # smooth across the frame
    d = _residual(rng, series, loading, noise=0.02)
    modes = fit_modes(d, fit, k=2)
    scr = mode_screens(modes, fit, k=1, sigma=(1.0, 1.0))
    corr = mode_correction(modes, scr)
    assert corr.shape == d.shape
    before = np.sqrt(np.mean((d[:, held] - d[:, held].mean(axis=0)) ** 2))
    after = np.sqrt(np.mean((d - corr)[:, held] ** 2))
    assert after < 0.15 * before
    with pytest.raises(ValueError):
        mode_correction(modes, np.zeros((3,) + SHAPE))


def test_pixel_scale_loading_does_not_transfer_but_projects():
    rng, t, series, fit, held = _grid(4)
    loading = rng.standard_normal(SHAPE)       # changes from pixel to pixel
    d = _residual(rng, series, loading, noise=0.02)
    modes = fit_modes(d, fit, k=1)
    corr = mode_correction(modes, mode_screens(modes, fit, sigma=(1.0, 1.0)))
    before = np.sqrt(np.mean(d[:, held] ** 2))
    interpolated = np.sqrt(np.mean((d - corr)[:, held] ** 2))
    assert interpolated > 0.7 * before
    proj = mode_projection(modes, d, held, k=1)
    assert np.isnan(proj[:, fit]).all() and np.isfinite(proj[:, held]).all()
    own = np.sqrt(np.mean((d - proj)[:, held] ** 2))
    assert own < 0.1 * before
    # the offset stays in the pixel, not in the projection
    assert np.abs(proj[:, held].mean(axis=0)).max() < 1e-8


def test_projection_coefficients_are_the_pixels_own_loadings():
    rng, t, series, fit, held = _grid(5)
    loading = rng.standard_normal(SHAPE)
    d = _residual(rng, series, loading, noise=0.0)
    modes = fit_modes(d, fit, k=1)
    proj, own = mode_projection(modes, d, held, k=1, coefficients=True)
    assert own.shape == (1,) + SHAPE
    assert np.isnan(own[0][fit]).all()
    # the planted loading, in the modes' unit-RMS scaling
    scale = np.sqrt(np.mean(loading[fit] ** 2))
    sign = np.sign(np.corrcoef(modes.temporal[:, 0], series)[0, 1])
    assert np.allclose(own[0][held], sign * loading[held] / scale, atol=1e-6)
