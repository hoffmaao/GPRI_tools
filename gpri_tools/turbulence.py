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

__all__ = ["structure_function", "power_law", "power_law_with_floor",
           "PowerLaw", "FlooredPowerLaw"]


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


@dataclass
class FlooredPowerLaw:
    """``D(r) = floor + amplitude * (r / r0) ** exponent``.

    The floor is the part of the structure function that does not depend on
    separation, which is what uncorrelated per-pixel noise contributes:
    two independent pixels differ by ``2 * variance`` however far apart they
    are.  Fitting it explicitly is the difference between measuring the air
    and measuring the receiver — without it, a noise-dominated field returns
    an exponent near zero and looks like a flat atmosphere.
    """

    exponent: float
    amplitude: float
    floor: float
    r0: float
    rms: float
    n: int

    @property
    def noise_rms(self):
        """The per-pixel noise the floor implies, in the field's units."""
        return float(np.sqrt(max(self.floor, 0.0) / 2.0))

    def __call__(self, r):
        return (self.floor
                + self.amplitude * (np.asarray(r, float) / self.r0) ** self.exponent)

    def __repr__(self):
        return (f"FlooredPowerLaw(exponent={self.exponent:.3f}, "
                f"amplitude={self.amplitude:.4g} at r0={self.r0:g}, "
                f"floor={self.floor:.4g} ({self.noise_rms:.3g} rms), "
                f"rms={self.rms:.4g}, n={self.n})")


def power_law_with_floor(r, D, r0=1000.0, exponents=None, floor=None):
    """``D(r) = floor + A (r / r0) ** alpha``, fitted in the data's own units.

    The exponent is scanned and the two linear coefficients solved exactly at
    each one, the same trick :func:`gpri_tools.decorrelation.exponential_decorrelation`
    uses, because a three-parameter nonlinear fit to a handful of points
    finds local minima.  The floor is constrained non-negative — a negative
    one is not a noise variance — and can be pinned with ``floor=``.

    Fit this rather than :func:`power_law` whenever the field may be
    noise-dominated at short separations, which for a per-pixel
    interferometric phase is always.
    """
    rr = np.asarray(r, float)
    dd = np.asarray(D, float)
    ok = np.isfinite(rr) & np.isfinite(dd) & (rr > 0)
    if ok.sum() < 3:
        raise ValueError("a floor plus a power law needs three points")
    rr, dd = rr[ok], dd[ok]
    # the scan starts above zero on purpose: at exactly zero the power-law
    # term is a constant and is perfectly degenerate with the floor, so the
    # fit would split a flat curve arbitrarily between the two
    grid = (np.linspace(0.05, 3.0, 296) if exponents is None
            else np.asarray(exponents, float))
    best = None
    for alpha in grid:
        basis = (rr / float(r0)) ** alpha
        if floor is None:
            A = np.column_stack([np.ones(rr.size), basis])
            coef, *_ = np.linalg.lstsq(A, dd, rcond=None)
            c0, c1 = float(coef[0]), float(coef[1])
            if c0 < 0:                       # refit with the floor pinned at 0
                c1 = float(np.dot(basis, dd) / max(np.dot(basis, basis), 1e-30))
                c0 = 0.0
        else:
            c0 = float(floor)
            c1 = float(np.dot(basis, dd - c0) / max(np.dot(basis, basis), 1e-30))
        resid = c0 + c1 * basis - dd
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(alpha), c1, c0)
    rms, alpha, amp, c0 = best
    return FlooredPowerLaw(exponent=alpha, amplitude=amp, floor=c0,
                           r0=float(r0), rms=rms, n=int(rr.size))
