"""Surface wetness from the brightness: binning, the diurnal swing, the duty cycle."""
from pathlib import Path

import numpy as np
import pytest

from gpri_tools.melt import (BinAccumulator, air_temperature_at, bin_by_hour, bin_mean,
                             diurnal_harmonic, diurnal_swing, pixel_correlation,
                             transfer_curve, wet_fraction)


def test_bins_start_at_the_first_epoch_and_absorb_the_endpoint():
    hours = 3.5 + np.arange(0, 5.0001, 1 / 30)          # 2-minute cadence from 03:30
    edges, idx = bin_by_hour(hours)
    assert edges[0] == pytest.approx(3.5)
    assert edges.size == 6 and idx.max() == 4 and idx.min() == 0
    assert (np.bincount(idx)[:-1] == 30).all()
    with pytest.raises(ValueError):
        bin_by_hour([])


def test_bin_mean_skips_the_gaps():
    m = bin_mean([1.0, np.nan, 3.0, 5.0], [0, 0, 1, 1], 3)
    assert np.allclose(m[:2], [1.0, 4.0]) and np.isnan(m[2])
    with pytest.raises(ValueError):
        bin_mean([1.0, 2.0], [0], 2)


def test_pixel_correlation_is_pearson_r_per_pixel():
    t = np.arange(10.0)
    a = np.stack([t, -t, np.full(10, 2.0), t], axis=1)
    b = np.stack([2 * t + 1, t, t, t], axis=1)
    b[:8, 3] = np.nan                                    # two finite pairs: too few
    r = pixel_correlation(a, b)
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(-1.0)
    assert np.isnan(r[2]) and np.isnan(r[3])
    with pytest.raises(ValueError):
        pixel_correlation(a, b[:-1])


def test_accumulator_is_a_nan_aware_running_mean():
    acc = BinAccumulator(2, (3,))
    acc.add(0, [1.0, np.nan, 2.0])
    acc.add(0, [3.0, 5.0, np.nan])
    acc.add(1, [np.nan, np.nan, np.nan])
    m = acc.mean()
    assert np.allclose(m[0], [2.0, 5.0, 2.0])
    assert np.isnan(m[1]).all()
    assert list(acc.epochs_per_bin) == [2, 1]


def test_swing_and_clock_hours_of_a_two_day_record():
    hours = 6.0 + np.arange(48)                          # hourly bins from 06:00 local
    # darkest at 14:00, brightest at 02:00, 3 dB peak to peak, one pixel of each
    clock = np.mod(hours, 24.0)
    cycle = 1.5 * np.cos(2 * np.pi * (clock - 2.0) / 24.0)
    hourly = np.stack([cycle, 2 * cycle, np.full(48, np.nan)], axis=1)
    swing, hmin, hmax = diurnal_swing(hourly, hours)
    assert swing[0] == pytest.approx(3.0, abs=0.05)
    assert swing[1] == pytest.approx(6.0, abs=0.1)
    assert np.isnan(swing[2]) and np.isnan(hmin[2])
    assert hmin[0] == pytest.approx(14.5, abs=0.51)
    assert hmax[0] == pytest.approx(2.5, abs=0.51)


def test_harmonic_fit_recovers_amplitude_and_clock_through_noise():
    rng = np.random.default_rng(1)
    hours = 6.0 + np.arange(48)
    cycle = 1.5 * np.cos(2 * np.pi * (hours - 2.0) / 24.0)
    hourly = np.stack([cycle, 2 * cycle + rng.normal(0, 0.3, 48), np.full(48, np.nan),
                       rng.normal(0, 0.3, 48)], axis=1)
    hourly[:40, 2] = cycle[:40]                          # 40 finite bins: enough
    amp, hmax, mean = diurnal_harmonic(hourly, hours)
    assert amp[0] == pytest.approx(1.5, abs=1e-6) and hmax[0] == pytest.approx(2.0, abs=1e-6)
    assert amp[1] == pytest.approx(3.0, abs=0.15) and hmax[1] == pytest.approx(2.0, abs=0.3)
    assert amp[2] == pytest.approx(1.5, abs=1e-6) and mean[2] == pytest.approx(0.0, abs=1e-6)
    assert amp[3] < 0.3                                   # noise alone: near the floor
    assert np.isnan(diurnal_harmonic(hourly[:8], hours[:8])[0]).all()
    with pytest.raises(ValueError):
        diurnal_harmonic(hourly, hours[:-1])


def test_a_record_shorter_than_the_window_gives_no_swing():
    hours = np.arange(8.0)
    swing, hmin, hmax = diurnal_swing(np.zeros((8, 2)), hours)
    assert np.isnan(swing).all() and np.isnan(hmin).all()
    with pytest.raises(ValueError):
        diurnal_swing(np.zeros((8, 2)), hours[:-1])


def test_wet_fraction_is_the_dark_duty_cycle():
    hourly = np.array([[0.0, 1.0], [-2.0, 1.0], [-2.0, 1.0], [0.0, 1.0]])
    frac = wet_fraction(hourly)
    assert frac[0] == pytest.approx(0.5)
    assert frac[1] == pytest.approx(0.0)                 # flat: nothing below its midpoint
    assert wet_fraction(hourly, threshold=-1.0)[0] == pytest.approx(0.5)
    assert np.isnan(wet_fraction(np.full((3, 1), np.nan)))[0]


def test_air_temperature_follows_the_lapse_from_the_station():
    T = np.array([10.0, 12.0])
    z = np.array([[1500.0, 2500.0]])
    Tz = air_temperature_at(z, T, 1500.0, lapse=-6.5)
    assert Tz.shape == (2, 1, 2)
    assert Tz[0, 0, 0] == pytest.approx(10.0)
    assert Tz[1, 0, 1] == pytest.approx(12.0 - 6.5)


def test_transfer_curve_bins_one_against_the_other():
    x = np.repeat(np.arange(4.0), 100)
    y = -0.5 * x + np.tile([-1, 0, 0, 1], 100)
    mid, med, q1, q3, cnt = transfer_curve(x, y, np.arange(-0.5, 4.0), min_count=50)
    assert list(cnt) == [100] * 4
    assert np.allclose(med, -0.5 * np.arange(4.0))
    assert (q1 <= med).all() and (med <= q3).all()
    with pytest.raises(ValueError):
        transfer_curve(x, y[:-1], np.arange(-0.5, 4.0))


def test_clock_median_stays_on_the_clock():
    """The middle of a set of hours is in [0, 24): symmetric about midnight is 0."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from baker_melt import clock_median

    assert clock_median(np.array([23.5, 0.5])) == pytest.approx(0.0)
    assert clock_median(np.array([23.5, 23.9, 0.1, 0.5])) < 1.0
    assert clock_median(np.array([11.8, 12.0, 12.2])) == pytest.approx(12.0, abs=0.1)
    # hours spread round the whole circle agree on none of them
    assert np.isnan(clock_median(np.arange(0.0, 24.0, 0.25)))
