"""The spatial statistics of what the screens take out.

:mod:`gpri_tools.aps` removes an atmosphere it never characterises: a linear
range screen, then a normalised convolution at a chosen ``sigma``.  The
choice of ``sigma`` is a claim about the scale over which the atmosphere is
coherent, and this module measures that claim rather than assuming it.

The estimator is the **structure function**

    D(r) = < [ phi(x + r) - phi(x) ]^2 >

over pairs of pixels a distance ``r`` apart.  It is the standard description
of a turbulent field, and it is the right one here for a reason worth
stating: a structure function needs no mean.  A screen fitted on bedrock has
an arbitrary constant in it, and an arbitrary linear ramp if a ramp was
fitted, and neither touches ``D(r)`` at separations short against the ramp.

A field whose phase follows Kolmogorov turbulence gives ``D(r) ~ r^alpha``
with ``alpha`` between 2/3 and 5/3 depending on whether the separation is
larger or smaller than the thickness of the turbulent layer, so the fitted
exponent says which regime the record is in — and a flat ``D(r)`` says the
field at that separation is noise rather than atmosphere.
:func:`power_law` fits ``alpha`` and the amplitude by least squares in
``log r``.

Both entry points work on masked, irregularly sampled pixels, because the
pixels worth measuring are the coherent ones and they are not a rectangle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["structure_function", "power_law", "PowerLaw"]


def structure_function(values, x, y, bins, max_pairs=200_000, rng=None):
    """Mean squared difference against separation, over sampled pixel pairs.

    Parameters
    ----------
    values : (n,) array
        The field at ``n`` scattered points, in whatever unit the answer
        should come back squared in.  Non-finite points are dropped.
    x, y : (n,) array
        Point coordinates, in the unit ``bins`` is given in.
    bins : (n_bins + 1,) array
        Separation bin edges, ascending.
    max_pairs : int
        Pairs to sample per bin.  The full pair set is quadratic in ``n``;
        sampling keeps the estimator cheap and its error falls as
        ``1 / sqrt(pairs)``.
    rng : numpy.random.Generator, optional

    Returns
    -------
    centres : (n_bins,) array
        Mean separation actually sampled in each bin, NaN where empty.
    D : (n_bins,) array
        Mean squared difference, NaN where a bin had no pairs.
    counts : (n_bins,) int array
    """
    v = np.asarray(values, float).reshape(-1)
    xx = np.asarray(x, float).reshape(-1)
    yy = np.asarray(y, float).reshape(-1)
    if not (v.size == xx.size == yy.size):
        raise ValueError("values, x and y must be the same length")
    ok = np.isfinite(v) & np.isfinite(xx) & np.isfinite(yy)
    v, xx, yy = v[ok], xx[ok], yy[ok]
    if v.size < 2:
        raise ValueError("need at least two finite points")
    edges = np.asarray(bins, float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be ascending edges")

    g = np.random.default_rng() if rng is None else rng
    n_bins = edges.size - 1
    centres = np.full(n_bins, np.nan)
    D = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, int)

    # one pass of random pairs per bin: draw, keep what lands in the bin.
    # Drawing per bin rather than globally keeps the long separations, which
    # are rare in a compact mask, from being swamped by the short ones.
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        kept_d2, kept_r, tries = [], [], 0
        target = int(max_pairs)
        while sum(len(k) for k in kept_r) < target and tries < 12:
            m = min(4 * target, 2_000_000)
            i = g.integers(0, v.size, m)
            j = g.integers(0, v.size, m)
            r = np.hypot(xx[i] - xx[j], yy[i] - yy[j])
            sel = (r >= lo) & (r < hi) & (i != j)
            if sel.any():
                kept_r.append(r[sel])
                kept_d2.append((v[i[sel]] - v[j[sel]]) ** 2)
            tries += 1
        if kept_r:
            r = np.concatenate(kept_r)[:target]
            d2 = np.concatenate(kept_d2)[:target]
            counts[b] = r.size
            centres[b] = r.mean()
            D[b] = d2.mean()
    return centres, D, counts


@dataclass
class PowerLaw:
    """``D(r) = amplitude * (r / r0) ** exponent`` fitted in log-log."""

    exponent: float
    amplitude: float
    r0: float
    rms: float
    n: int

    def __call__(self, r):
        return self.amplitude * (np.asarray(r, float) / self.r0) ** self.exponent

    def __repr__(self):
        return (f"PowerLaw(exponent={self.exponent:.3f}, "
                f"amplitude={self.amplitude:.4g} at r0={self.r0:g}, "
                f"rms={self.rms:.3f} in log10, n={self.n})")


def power_law(r, D, r0=1000.0, weights=None):
    """Least-squares ``log D = log A + alpha log(r / r0)``.

    ``r0`` is the separation the amplitude is quoted at, so that the two
    numbers are not correlated through a lever arm: quote the amplitude
    where the data are, not at ``r = 1``.
    """
    rr = np.asarray(r, float)
    dd = np.asarray(D, float)
    ok = np.isfinite(rr) & np.isfinite(dd) & (rr > 0) & (dd > 0)
    if ok.sum() < 2:
        raise ValueError("need two positive (r, D) points to fit a power law")
    lr = np.log10(rr[ok] / float(r0))
    ld = np.log10(dd[ok])
    w = np.ones(ok.sum()) if weights is None else np.asarray(weights, float)[ok]
    W = np.sqrt(np.maximum(w, 0.0))
    A = np.column_stack([np.ones(lr.size), lr])
    coef, *_ = np.linalg.lstsq(A * W[:, None], ld * W, rcond=None)
    rms = float(np.sqrt(np.mean((A @ coef - ld) ** 2)))
    return PowerLaw(exponent=float(coef[1]), amplitude=float(10 ** coef[0]),
                    r0=float(r0), rms=rms, n=int(ok.sum()))
