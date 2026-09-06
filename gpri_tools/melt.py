"""Surface wetness from the radar's own brightness.

Every acquisition carries the surface's backscatter as well as its phase,
and at Ku band the two say different things.  Liquid water in the top
centimetres of a snowpack makes it dark — the loss tangent of wet snow puts
the penetration depth at a centimetre or two and the surface scatters
specularly away from a grazing radar — while snow that has drained or
refrozen overnight scatters from a volume again and brightens by decibels.
That is what QuikSCAT's diurnal backscatter difference maps over the ice
sheets (Nghiem et al. 2001), what Baffelli et al. (2019) saw under a Ku-band
terrestrial radar on an Alpine glacier, and what the ``db_*`` band series in
``examples/baker_pixels.py`` show on Mount Baker: dark through the warm
afternoon, brightest around dawn.

This module turns that into per-pixel numbers a melt model can be checked
against.  The radar's clock is minutes and its record a day or two, so the
natural unit is the *hourly mean*: :class:`BinAccumulator` builds one frame
per hour from a stream of epochs without holding the stack in memory, and
:func:`diurnal_swing` reads each pixel's day off it — how far it swings, and
the clock hour at which it is darkest (wettest) and brightest.
:func:`wet_fraction` is the duty cycle of the dark state, and
:func:`transfer_curve` puts the brightness against the air temperature at
the pixel's own height (:func:`air_temperature_at`), which is the curve a
degree-day melt model would need.

Two cautions.  The brightness is an *instrument* measurement before it is a
surface one — the receiver gain drifts by a decibel or more over a day — so
every frame should be referenced to bedrock, as ``baker_pixels.py`` does with
the fit-half bedrock's median, before anything here is read.  And "dark" is
relative: August firn that drained overnight is not January powder, and the
swing between a pixel's own wet and drained states is the measure, not its
distance from a dry-snow reference it never reaches.
"""
from __future__ import annotations

import numpy as np

__all__ = ["BinAccumulator", "air_temperature_at", "bin_by_hour", "bin_mean",
           "clock_composite", "clock_median", "diurnal_harmonic", "diurnal_swing",
           "pixel_correlation", "transfer_curve", "wet_fraction"]


def bin_by_hour(hours, width=1.0):
    """Edges and the per-epoch bin index for ``width``-hour bins.

    Bins start at the first epoch, so bin ``b`` covers
    ``[h0 + b*width, h0 + (b+1)*width)``; the last bin absorbs the endpoint.
    """
    hours = np.asarray(hours, float)
    if hours.size == 0:
        raise ValueError("no epochs to bin")
    span = hours.max() - hours.min()
    n = max(int(np.ceil(span / width)), 1)
    edges = hours.min() + width * np.arange(n + 1)
    idx = np.minimum(((hours - hours.min()) / width).astype(int), n - 1)
    return edges, idx


def bin_mean(values, idx, n_bins):
    """NaN-aware mean of a series per bin: the hourly air temperature, say."""
    values = np.asarray(values, float)
    idx = np.asarray(idx, int)
    if values.shape != idx.shape:
        raise ValueError("values and idx must pair element-wise")
    ok = np.isfinite(values)
    total = np.bincount(idx[ok], weights=values[ok], minlength=n_bins)
    count = np.bincount(idx[ok], minlength=n_bins)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


class BinAccumulator:
    """NaN-aware running mean of frames, one slot per bin.

    ``add(b, frame)`` folds one epoch's frame into bin ``b``; ``mean()`` is
    the per-bin mean with NaN where a bin never saw a finite value.  Frames
    are read one at a time, so a day of backscatter never has to exist as a
    single array.
    """

    def __init__(self, n_bins: int, shape):
        self.sum = np.zeros((n_bins,) + tuple(shape), np.float64)
        self.count = np.zeros((n_bins,) + tuple(shape), np.int32)
        self.epochs_per_bin = np.zeros(n_bins, int)

    def add(self, b: int, frame):
        frame = np.asarray(frame, float)
        ok = np.isfinite(frame)
        self.sum[b] += np.where(ok, frame, 0.0)
        self.count[b] += ok
        self.epochs_per_bin[b] += 1

    def mean(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.sum / self.count, np.nan)


def diurnal_swing(hourly, hours, period=24.0, min_bins=12):
    """Each pixel's day: its swing, and the clock hours it is darkest and brightest.

    ``hourly`` is ``(H, ...)`` binned values and ``hours`` the ``H`` bin
    centres on whatever clock the answer should be in (local hours since a
    midnight, say).  The swing is the peak-to-peak range within each
    ``period``-long window, averaged over the windows that have at least
    ``min_bins`` finite bins; the hours of the minimum and maximum come from
    the composite over the record by clock hour, so a two-day record answers
    with one clock.  Returns ``(swing, hour_min, hour_max)``, each of the
    trailing shape, with NaN where the record is too short.
    """
    hourly = np.asarray(hourly, float)
    hours = np.asarray(hours, float)
    if hourly.shape[0] != hours.size:
        raise ValueError("hourly and hours disagree on the number of bins")
    shape = hourly.shape[1:]
    swings, n_win = np.zeros(shape), np.zeros(shape, int)
    for start in np.arange(hours.min(), hours.max(), period):
        w = (hours >= start) & (hours < start + period)
        if w.sum() < min_bins:
            continue
        block = hourly[w]
        ok = np.isfinite(block).sum(axis=0) >= min_bins
        with np.errstate(invalid="ignore"):
            pp = np.nanmax(block, axis=0) - np.nanmin(block, axis=0)
        swings += np.where(ok, pp, 0.0)
        n_win += ok
    swing = np.where(n_win > 0, swings / np.maximum(n_win, 1), np.nan)

    # composite by clock hour, then the hour of each pixel's extremes
    clock = np.mod(hours, period)
    width = float(np.median(np.diff(hours))) if hours.size > 1 else period
    n_clock = max(int(round(period / width)), 1)
    slot = np.minimum((clock / period * n_clock).astype(int), n_clock - 1)
    comp = np.full((n_clock,) + shape, np.nan)
    for s in range(n_clock):
        m = slot == s
        if m.any():
            with np.errstate(invalid="ignore"):
                comp[s] = np.nanmean(hourly[m], axis=0)
    any_ok = np.isfinite(comp).any(axis=0)
    filled = np.where(np.isfinite(comp), comp, np.inf)
    hour_min = (np.argmin(filled, axis=0) + 0.5) * period / n_clock
    filled = np.where(np.isfinite(comp), comp, -np.inf)
    hour_max = (np.argmax(filled, axis=0) + 0.5) * period / n_clock
    hour_min = np.where(any_ok & (n_win > 0), hour_min, np.nan)
    hour_max = np.where(any_ok & (n_win > 0), hour_max, np.nan)
    return swing, hour_min, hour_max


def diurnal_harmonic(hourly, hours, period=24.0, min_bins=12):
    """Least-squares diurnal sinusoid per pixel: ``(amplitude, hour_max, mean)``.

    Fits ``c + A cos(2π (h − φ) / period)`` to each pixel's binned series
    and returns ``A``, the clock hour ``φ`` of its maximum, and ``c``.  Where
    :func:`diurnal_swing` reads the extremes, which a noisy pixel exaggerates,
    the harmonic amplitude averages the noise down over every bin, so the
    peak-to-peak ``2A`` of bedrock is the noise floor and an ice pixel's
    excess over it is the surface.  Pixels with fewer than ``min_bins``
    finite bins are NaN.
    """
    hourly = np.asarray(hourly, float)
    hours = np.asarray(hours, float)
    if hourly.shape[0] != hours.size:
        raise ValueError("hourly and hours disagree on the number of bins")
    w = 2 * np.pi / period
    X = np.stack([np.ones_like(hours), np.cos(w * hours), np.sin(w * hours)], axis=1)
    ok = np.isfinite(hourly)
    y = np.where(ok, hourly, 0.0)
    XtX = np.einsum("ti,tj,t...->...ij", X, X, ok.astype(float))
    Xty = np.einsum("ti,t...->...i", X, y)
    good = ok.sum(axis=0) >= min_bins
    beta = np.full(good.shape + (3,), np.nan)
    if good.any():
        beta[good] = np.einsum("...ij,...j->...i", np.linalg.pinv(XtX[good]), Xty[good])
    amplitude = np.hypot(beta[..., 1], beta[..., 2])
    hour_max = np.mod(np.arctan2(beta[..., 2], beta[..., 1]) / w, period)
    return amplitude, hour_max, beta[..., 0]


def clock_composite(series, hours, period=24.0):
    """Mean by clock hour of an hourly series: a two-day record on one clock.

    ``hours`` is the clock the bins sit on, so a record that spans two days
    folds onto one ``period``-long day and each returned bin is the mean of
    every hour of the record that fell in it.  Bins the record never reached
    are NaN.
    """
    series = np.asarray(series, float)
    slot = np.mod(np.floor(np.asarray(hours, float)), period).astype(int)
    comp = np.full(int(period), np.nan)
    for s in range(int(period)):
        m = (slot == s) & np.isfinite(series)
        if m.any():
            comp[s] = series[m].mean()
    return comp


def clock_median(hours, period=24.0, min_r=0.3):
    """Middle of a set of clock hours, as the phase of their mean unit vector.

    An hour wraps, so a plain median of one is wrong wherever the values
    straddle midnight and meaningless where they are spread round the whole
    circle.  The mean resultant length says which of the two it is; below
    ``min_r`` they agree on no hour and the answer is NaN rather than a number
    that only looks definite.  The result is on ``[0, period)``.
    """
    h = np.asarray(hours, float)
    h = h[np.isfinite(h)]
    if h.size == 0:
        return np.nan
    z = np.exp(2j * np.pi * h / period).mean()
    if np.abs(z) < min_r:
        return np.nan
    out = np.mod(np.angle(z) * period / (2 * np.pi), period)
    return float(out if out < period else 0.0)


def wet_fraction(hourly, threshold=None):
    """The dark duty cycle: the fraction of bins a pixel spends in its wet state.

    With ``threshold`` a bin is wet when its value is below that number (a
    referenced dB anomaly, say); without one the pixel's own midpoint
    between brightest and darkest bin is the line, so the answer is the
    fraction of the record nearer the dark state than the bright one.
    """
    hourly = np.asarray(hourly, float)
    with np.errstate(invalid="ignore"):
        if threshold is None:
            threshold = 0.5 * (np.nanmax(hourly, axis=0) + np.nanmin(hourly, axis=0))
        wet = hourly < threshold
    n = np.isfinite(hourly).sum(axis=0)
    return np.where(n > 0, wet.sum(axis=0) / np.maximum(n, 1), np.nan)


def pixel_correlation(a, b, min_count=3):
    """Pearson r along axis 0 of two ``(T, ...)`` fields, per trailing element.

    Only the times where both are finite count; elements with fewer than
    ``min_count`` of them, or no variance in either field, are NaN.  This is
    the per-pixel "does the brightness follow the air" map.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=0)
    a = np.where(ok, a, 0.0)
    b = np.where(ok, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = a.sum(axis=0) / n
        mb = b.sum(axis=0) / n
        da = np.where(ok, a - ma, 0.0)
        db = np.where(ok, b - mb, 0.0)
        r = (da * db).sum(axis=0) / np.sqrt((da * da).sum(axis=0) * (db * db).sum(axis=0))
    return np.where((n >= min_count) & np.isfinite(r), r, np.nan)


def air_temperature_at(z, station_temperature, station_height, lapse=-6.5):
    """Air temperature at height ``z`` from a station series and a lapse rate.

    ``station_temperature`` is ``(T,)`` in °C, ``z`` any shape in metres,
    ``lapse`` in °C/km (negative when it cools upward): the answer is
    ``(T,) + z.shape``.  A fixed environmental lapse is the honest default
    when every station sits below the glacier; a fitted one should be passed
    only where its stations bracket the heights asked for.
    """
    T = np.asarray(station_temperature, float)
    z = np.asarray(z, float)
    dz = (z - float(station_height)) / 1000.0
    return T.reshape((-1,) + (1,) * z.ndim) + lapse * dz


def transfer_curve(x, y, edges, min_count=50):
    """Median and quartiles of ``y`` in bins of ``x``: brightness against the air.

    Both are flattened and paired element-wise (a ``(T, N)`` temperature
    field against a ``(T, N)`` brightness field, say); bins with fewer than
    ``min_count`` finite pairs are dropped.  Returns ``(mid, median, q1, q3,
    count)``.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must pair element-wise")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    mid, med, q1, q3, cnt = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < min_count:
            continue
        p = np.percentile(y[m], [25, 50, 75])
        mid.append(0.5 * (a + b)); med.append(p[1]); q1.append(p[0]); q3.append(p[2])
        cnt.append(int(m.sum()))
    return (np.array(mid), np.array(med), np.array(q1), np.array(q3),
            np.array(cnt, int))
