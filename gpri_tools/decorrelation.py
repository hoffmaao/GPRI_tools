"""How fast the surface stops looking like itself: coherence against baseline.

Every product in this package treats coherence as a weight.  It is also an
observation in its own right: the rate at which ``gamma`` falls as the
temporal baseline grows says how quickly the scattering surface rearranges,
and it is measured on the same pairs the deformation is measured on, at no
extra cost.

Two estimators live here, and they answer different questions.

:func:`coherence_by_baseline` reduces a pair stack to one coherence per
temporal baseline class per pixel — the empirical decorrelation curve.

:func:`decorrelation_time` turns that curve into a single number per pixel:
the baseline at which coherence first falls through a target, interpolated
in log baseline.  The target is a fraction of the pixel's own
shortest-baseline coherence, or an absolute level when one is given — the
level is what compares between pixels, masks and campaigns, and on a record
that ends before most pixels have lost ``1 - 1/e`` the ratio form is mostly
NaN.  It assumes no model, which matters
because real decorrelation on a glacier is not a single exponential — wet
snow, firn and rock fall away at different rates and a fitted exponential
splits the difference.  :func:`exponential_decorrelation` does fit the usual
model ``gamma(t) = (g0 - g_inf) exp(-t / tau) + g_inf`` for the cases where
a curve is wanted rather than a map; it is meant for the mask-averaged
curves, not for every pixel.

The caveat that governs both: a coherence estimated from ``L`` looks is
biased high, the more so as the true coherence approaches zero, so a
long-baseline coherence is a ceiling rather than a measurement.  The bias
depends only on ``L``, so it moves every baseline class the same way and
leaves the *shape* — which is what these estimators read — alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["coherence_by_baseline", "decorrelation_time",
           "exponential_decorrelation", "DecorrelationFit"]


def coherence_by_baseline(coherence, pairs, times, by="lag", tolerance=0.05,
                          min_pairs=1):
    """Mean coherence per temporal baseline class, per pixel.

    Parameters
    ----------
    coherence : (n_pairs, ...) array
        Per-pair coherence on the pixel grid.
    pairs : (n_pairs, 2) int array
    times : (n_epochs,) array
        Epoch times in days.
    by : {'lag', 'time'}
        ``'lag'`` classes pairs by ``j - i``, the number of epochs they span,
        which is exactly how a pair stack is built and is immune to the
        cadence jitter that splits a nominal four-minute baseline into two
        classes, one of them holding a single pair.  ``'time'`` clusters the
        elapsed times themselves, for a stack whose epochs are not on a
        regular cadence.
    tolerance : float
        ``by='time'`` only: baselines within this fraction of the running
        class mean join it.
    min_pairs : int
        Classes with fewer pairs than this are dropped.  A class of one pair
        is a coherence estimate from one interferogram and carries the
        wrong weight in a curve every other point of which averages hundreds.

    Returns
    -------
    baselines : (n_classes,) array
        Class mean baseline, in days, ascending.
    gamma : (n_classes, ...) array
        Mean coherence of the pairs in each class.
    counts : (n_classes,) int array
    """
    cc = np.asarray(coherence)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    t = np.asarray(times, float)
    if cc.shape[0] != pr.shape[0]:
        raise ValueError(f"{cc.shape[0]} coherence planes for {pr.shape[0]} pairs")
    dt = t[pr[:, 1]] - t[pr[:, 0]]
    if not np.all(dt > 0):
        raise ValueError("every pair must run forward in time")

    if by not in ("lag", "time"):
        raise ValueError(f"by must be 'lag' or 'time', not {by!r}")
    if by == "lag":
        lag = pr[:, 1] - pr[:, 0]
        classes = [np.flatnonzero(lag == L) for L in np.unique(lag)]
    else:
        order = np.argsort(dt)
        classes, current = [], [order[0]]
        for k in order[1:]:
            ref = dt[current].mean()
            if abs(dt[k] - ref) <= tolerance * ref:
                current.append(k)
            else:
                classes.append(np.asarray(current))
                current = [k]
        classes.append(np.asarray(current))

    classes = [c for c in classes if len(c) >= int(min_pairs)]
    if not classes:
        raise ValueError(f"no baseline class has {min_pairs} pairs")
    baselines = np.array([dt[c].mean() for c in classes])
    order = np.argsort(baselines)
    classes = [classes[i] for i in order]
    baselines = baselines[order]
    counts = np.array([len(c) for c in classes], int)
    gamma = np.stack([np.nanmean(cc[c], axis=0) for c in classes])
    return baselines, gamma, counts


def decorrelation_time(baselines, gamma, fraction=1.0 / np.e, level=None,
                       axis=0):
    """Baseline at which ``gamma`` first falls through a target.

    The target is ``fraction`` of each pixel's shortest-baseline coherence,
    or the absolute ``level`` when one is given.  Which to use is a real
    choice: the ratio asks how far each pixel has fallen from its own
    starting point, the level asks when it reached a coherence a reader can
    compare between pixels, masks and campaigns.  On a record that ends
    before most pixels have lost ``1 - 1/e`` of their coherence the ratio
    form returns mostly NaN, which is the honest answer to the question it
    was asked and a poor map.

    The crossing is interpolated in ``log`` baseline, which is the scale the
    classes are spaced on.  Returns NaN where the curve never crosses — a
    pixel still above the target at the longest baseline observed, one
    already below it at the shortest, or one with nothing finite to read.

    Parameters
    ----------
    baselines : (n_classes,) array
        Ascending, in whatever time unit the answer should come back in.
    gamma : (n_classes, ...) array
    fraction : float
        Of the shortest-baseline coherence.  ``1/e`` by default.  Ignored
        when ``level`` is given.
    level : float, optional
        An absolute coherence to cross instead.
    axis : int
        Which axis of ``gamma`` the classes lie on.

    Returns
    -------
    tau : array
        Shaped like ``gamma`` with the class axis removed.
    """
    b = np.asarray(baselines, float)
    g = np.moveaxis(np.asarray(gamma, float), axis, 0)
    if b.size != g.shape[0]:
        raise ValueError(f"{b.size} baselines for {g.shape[0]} classes")
    if b.size < 2:
        raise ValueError("a crossing needs at least two baseline classes")
    if np.any(np.diff(b) <= 0):
        raise ValueError("baselines must ascend")

    target = (np.full(g.shape[1:], float(level)) if level is not None
              else g[0] * float(fraction))
    lb = np.log(b)
    tau = np.full(g.shape[1:], np.nan)
    below = np.zeros(g.shape[1:], bool)
    for i in range(1, g.shape[0]):
        # the first descent through the target, pixel by pixel
        crossed = (~below) & np.isfinite(g[i]) & np.isfinite(g[i - 1]) \
            & (g[i] <= target) & (g[i - 1] > target)
        if crossed.any():
            hi, lo = g[i - 1][crossed], g[i][crossed]
            w = (hi - target[crossed]) / np.maximum(hi - lo, 1e-12)
            tau[crossed] = np.exp(lb[i - 1] + w * (lb[i] - lb[i - 1]))
        below |= crossed
    return tau


@dataclass
class DecorrelationFit:
    """``gamma(t) = (g0 - g_inf) exp(-t / tau) + g_inf`` fitted to a curve."""

    tau: float
    g0: float
    g_inf: float
    rms: float
    n: int

    def __call__(self, t):
        t = np.asarray(t, float)
        return (self.g0 - self.g_inf) * np.exp(-t / self.tau) + self.g_inf

    def __repr__(self):
        return (f"DecorrelationFit(tau={self.tau:.4g}, g0={self.g0:.3f}, "
                f"g_inf={self.g_inf:.3f}, rms={self.rms:.4f}, n={self.n})")


def exponential_decorrelation(baselines, gamma, taus=None):
    """Least-squares ``(g0 - g_inf) exp(-t / tau) + g_inf`` for one curve.

    ``tau`` is found by scanning a grid (log-spaced across the observed
    baselines by default) and solving the two linear coefficients exactly at
    each one, which avoids the local minima a joint nonlinear solve falls
    into on a three-point curve.  Meant for a mask-averaged curve; for a map,
    use :func:`decorrelation_time`.
    """
    b = np.asarray(baselines, float)
    g = np.asarray(gamma, float)
    ok = np.isfinite(b) & np.isfinite(g)
    b, g = b[ok], g[ok]
    if b.size < 3:
        raise ValueError("an exponential with a floor needs three points")
    if taus is None:
        taus = np.logspace(np.log10(b.min() / 5), np.log10(b.max() * 5), 400)
    best = None
    for tau in np.asarray(taus, float):
        A = np.column_stack([np.exp(-b / tau), np.ones(b.size)])
        coef, *_ = np.linalg.lstsq(A, g, rcond=None)
        rms = float(np.sqrt(np.mean((A @ coef - g) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(tau), float(coef[0] + coef[1]), float(coef[1]))
    rms, tau, g0, g_inf = best
    return DecorrelationFit(tau=tau, g0=g0, g_inf=g_inf, rms=rms, n=int(b.size))
