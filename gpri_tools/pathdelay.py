"""Per-acquisition atmospheric path delay from double-differenced pairs.

A tripod radar sees two things in every interferogram and cannot tell them
apart pixel by pixel: the surface moved, and the air the beam crossed
changed.  :mod:`gpri_tools.aps` separates them *spatially* — a screen fitted
on stable ground, subtracted everywhere.  That needs bedrock in the scene,
and it only removes what the screen model can express.

This module separates them *temporally* instead, following the terrestrial
radar interferometry (TRI) literature (Wild et al., double-difference DInSAR
on Grenzgletscher).  Nothing is fitted on rock, and no spatial model is
assumed.

The model
---------
Write each pair's line-of-sight displacement in the ``d_j - d_i`` convention
:func:`gpri_tools.timeseries.invert_network` uses::

    b_p = (d_j - d_i) + (a_j - a_i)          p = (i, j)

``d`` is the surface's own motion and ``a`` is the apparent displacement the
path delay puts on epoch ``i``.  Both enter as an epoch difference, so a
single pair can never split them.

Take two pairs that share their middle epoch, ``p = (i, j)`` and
``q = (j, k)``, and difference them after dividing by their spans::

    D = b_p / dt_p - b_q / dt_q

If the surface moves at a steady rate over the three epochs, ``d = v t``
contributes ``v - v = 0``: the motion is gone, exactly, for any spacing.
What is left is linear in the per-epoch delays::

    D = -a_i / dt_p + a_j (1 / dt_p + 1 / dt_q) - a_k / dt_q

which is one row of :func:`double_difference`'s ``A``.  Equally spaced
epochs give the familiar ``(-1, 2, -1) / dt`` second difference.  Stacking
every such triplet over several temporal baselines gives an overdetermined
system ``A a = b``, solved here with Tikhonov regularisation::

    a = (A^T A + lam I)^-1 A^T b

Two things this cannot do
-------------------------
**It cannot see an affine delay.**  ``A`` annihilates ``a_i = alpha + beta
t_i`` — a constant offset and a linear drift — because a delay that grows
linearly with time is indistinguishable from steady motion, which is what
the double difference was built to remove.  The null space is exactly two
dimensional, and :func:`pin_affine` fixes the representative by removing the
constant and the trend.  Whatever linear-in-time delay was really there is
absorbed into the velocity, and no amount of data in this system will get it
back; that part is :func:`gpri_tools.aps.epoch_screen_correction`'s job,
which uses bedrock and can see it.

**It cannot tell slow atmosphere from unsteady flow.**  Motion and delay
enter identically, so the only thing separating them is the assumption that
the motion is linear in time.  Everything in the phase that is *not* linear
in time is called atmosphere by this method, whatever put it there.

How much of a given timescale even reaches the system is set by the
operator: a component of period ``T``, sampled at spacing ``dt``, arrives in
the double differences multiplied by ``4 sin^2(pi dt / T) / dt``, which goes
to zero as ``T`` grows.  At the four-minute cadence of a GPRI campaign that
weight is 1440 per day at the Nyquist period and 0.11 per day at 24 h — a
diurnal component is present in the data four orders of magnitude more
weakly than an epoch-to-epoch one.  An unregularised solve inverts that gain
and returns the slow part anyway, at whatever noise it costs; any ``lam``
above zero shrinks it instead.  :func:`frequency_response` puts a number on
which timescales survive for a given ``lam``, and it is what to quote before
claiming this correction did or did not touch a diurnal signal.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DoubleDifference", "PathDelay", "discarded_rate", "double_difference",
    "frequency_response", "invert_path_delay", "lambda_for_response",
    "lambda_for_system_response", "pin_affine", "pin_rate", "select_lambda",
    "shared_epoch_triplets", "system_response", "tikhonov",
]

#: Rows whose observation is non-finite in any pixel are dropped by default.
NAN_POLICY = ("drop", "raise", "zero")


def shared_epoch_triplets(pairs, times=None, max_span=None, max_triplets=None):
    """Index every ``(p, q)`` where pair ``p`` ends on the epoch pair ``q`` starts.

    Parameters
    ----------
    pairs : (n_pairs, 2) int array
        ``(i, j)`` epoch indices, as :attr:`gpri_tools.network.Network.pairs`.
    times : (n_epochs,) array, optional
        Needed only with ``max_span``.
    max_span : float, optional
        Skip triplets whose outer span ``t_k - t_i`` exceeds this, in the
        units of ``times`` (days, for a :class:`~gpri_tools.network.Network`).
        The motion is only assumed steady *within* a triplet, so a shorter
        span is a weaker assumption.
    max_triplets : int, optional
        Keep at most this many, evenly spaced through the list, when the
        stack is large enough that the full set is unwieldy.

    Returns
    -------
    (m, 2) int array of pair indices.
    """
    pr = np.asarray(pairs, int).reshape(-1, 2)
    starts = {}
    for q, (i, _) in enumerate(pr):
        starts.setdefault(i, []).append(q)

    rows = []
    for p, (i, j) in enumerate(pr):
        for q in starts.get(j, ()):
            k = pr[q, 1]
            if k == i:
                continue                       # a closed loop, not a triplet
            if max_span is not None:
                if times is None:
                    raise ValueError("max_span needs times")
                t = np.asarray(times, float)
                if abs(t[k] - t[i]) > max_span:
                    continue
            rows.append((p, q))
    rows = np.asarray(rows, int).reshape(-1, 2)
    if max_triplets is not None and len(rows) > max_triplets:
        keep = np.linspace(0, len(rows) - 1, int(max_triplets)).round().astype(int)
        rows = rows[np.unique(keep)]
    return rows


@dataclass
class DoubleDifference:
    """The two operators of a double-difference system.

    ``b = T @ observations`` turns per-pair observations into double
    differences; ``A @ delay`` predicts the same quantity from per-epoch
    delays.  Both are built by :func:`double_difference`.
    """

    A: np.ndarray                 # (m, n_epochs)
    T: np.ndarray                 # (m, n_pairs), the dense form of `apply`
    rows: np.ndarray              # (m, 2) pair indices
    times: np.ndarray             # (n_epochs,)
    spans: np.ndarray             # (m, 2) the two temporal baselines
    epochs: np.ndarray            # (m, 3) the i, j, k of each row
    weights: np.ndarray           # (m, 2) the 1/dt each pair enters with

    @property
    def n_rows(self) -> int:
        return self.A.shape[0]

    def apply(self, observations) -> np.ndarray:
        """Double-difference per-pair observations.

        Equivalent to ``T @ observations``, but evaluated through the two
        pair indices of each row so that a non-finite pair spoils only the
        rows it actually enters — ``0 * nan`` in the dense product would put
        a NaN in every row of the system.
        """
        obs = np.asarray(observations, float)
        wp, wq = self.weights[:, 0], self.weights[:, 1]
        shape = (-1,) + (1,) * (obs.ndim - 1)
        return (wp.reshape(shape) * obs[self.rows[:, 0]]
                - wq.reshape(shape) * obs[self.rows[:, 1]])

    def null_space(self) -> np.ndarray:
        """The two-column affine null space ``[1, t]``, orthonormalised."""
        t = self.times - self.times.mean()
        basis = np.column_stack([np.ones_like(t), t])
        q, _ = np.linalg.qr(basis)
        return q

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return (f"DoubleDifference({self.n_rows} rows, "
                f"{self.A.shape[1]} epochs, "
                f"spans {self.spans.min():.4g}-{self.spans.max():.4g})")


def double_difference(pairs, times, max_span=None, max_triplets=None,
                      normalise=True):
    """Build ``A`` and ``T`` for every triplet sharing a middle epoch.

    Parameters
    ----------
    pairs : (n_pairs, 2) int array
    times : (n_epochs,) array
        Epoch times, any consistent unit; days is what
        :attr:`gpri_tools.network.Network.times` gives.
    normalise : bool
        Divide each pair by its temporal baseline before differencing, so the
        motion cancels for *any* spacing.  With ``False`` the rows are plain
        differences ``b_p - b_q``, which cancel motion only where the two
        baselines match — the equally spaced case.

    Returns
    -------
    :class:`DoubleDifference`

    Raises
    ------
    ValueError
        If no triplet exists.  A single-reference network (``1-2``, ``1-3``,
        ``1-4`` …) has none by construction; it needs a daisy chain, which is
        what a GPRI campaign records.
    """
    pr = np.asarray(pairs, int).reshape(-1, 2)
    t = np.asarray(times, float)
    rows = shared_epoch_triplets(pr, t, max_span=max_span,
                                 max_triplets=max_triplets)
    if len(rows) == 0:
        raise ValueError(
            "no pair of interferograms shares a middle epoch; a double "
            "difference needs a chain (1-2, 2-3, ...), not a single-reference "
            "network")

    m, n = len(rows), t.size
    A = np.zeros((m, n))
    T = np.zeros((m, pr.shape[0]))
    spans = np.zeros((m, 2))
    epochs = np.zeros((m, 3), int)
    w = np.zeros((m, 2))
    for r, (p, q) in enumerate(rows):
        i, j = pr[p]
        _, k = pr[q]
        dp = t[j] - t[i]
        dq = t[k] - t[j]
        if dp == 0 or dq == 0:
            raise ValueError(f"pair {p} or {q} spans zero time")
        wp, wq = (1.0 / dp, 1.0 / dq) if normalise else (1.0, 1.0)
        A[r, i] -= wp
        A[r, j] += wp + wq
        A[r, k] -= wq
        T[r, p] += wp
        T[r, q] -= wq
        spans[r] = (dp, dq)
        epochs[r] = (i, j, k)
        w[r] = (wp, wq)
    return DoubleDifference(A=A, T=T, rows=rows, times=t, spans=spans,
                            epochs=epochs, weights=w)


# ------------------------------------------------------------------- solving
def tikhonov(A, b, lam, chunk=None):
    """Solve ``(A^T A + lam I) x = A^T b`` for one or many right-hand sides.

    ``b`` may be ``(m,)`` or ``(m, ...)``; the extra axes are pixels and are
    solved with a single factorisation.  ``lam`` is in the units of
    ``A^T A``'s diagonal, so scale-free only if ``A`` is.  Use
    :func:`select_lambda` rather than guessing.
    """
    A = np.asarray(A, float)
    lam = float(lam)
    if lam < 0:
        raise ValueError("lam must be >= 0")
    B = np.asarray(b, float)
    if B.shape[0] != A.shape[0]:
        raise ValueError(f"b has {B.shape[0]} rows, A has {A.shape[0]}")
    spatial = B.shape[1:]
    Y = B.reshape(B.shape[0], -1)

    n = A.shape[1]
    normal = A.T @ A + lam * np.eye(n)
    try:
        chol = np.linalg.cholesky(normal)
        solve = lambda R: np.linalg.solve(chol.T, np.linalg.solve(chol, R))
    except np.linalg.LinAlgError:                     # lam = 0 and rank short
        solve = lambda R: np.linalg.lstsq(normal, R, rcond=None)[0]

    if chunk is None or Y.shape[1] <= chunk:
        X = solve(A.T @ Y)
    else:
        X = np.empty((n, Y.shape[1]))
        for s in range(0, Y.shape[1], int(chunk)):
            e = min(s + int(chunk), Y.shape[1])
            X[:, s:e] = solve(A.T @ Y[:, s:e])
    return X.reshape((n,) + spatial)


def select_lambda(A, b, lams=None, method="gcv"):
    """Pick the regularisation weight from the data.

    Parameters
    ----------
    A : (m, n) array
    b : (m,) or (m, k) array
        One or a few representative right-hand sides — a spatial mean series,
        or a random subset of pixels.  Both criteria are cheap once ``A`` is
        decomposed, but they are meant for a handful of columns, not a whole
        scene.
    lams : sequence, optional
        Grid to search.  Defaults to 40 points log-spaced over eight decades
        around the largest singular value squared.
    method : {'gcv', 'lcurve'}
        ``gcv`` minimises generalised cross-validation,
        ``m ||b - A x||^2 / trace(I - A M)^2``.  ``lcurve`` takes the point of
        maximum curvature of ``log||A x - b||`` against ``log||x||``.

    Returns
    -------
    lam : float
    curve : dict
        ``lams``, ``residual``, ``norm`` and the criterion actually minimised,
        so the choice can be plotted rather than trusted.
    """
    A = np.asarray(A, float)
    B = np.asarray(b, float).reshape(A.shape[0], -1)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if lams is None:
        top = float(s[0]) ** 2
        lams = np.logspace(np.log10(top) - 8, np.log10(top), 40)
    lams = np.asarray(lams, float)

    beta = U.T @ B                       # (n, k)
    b2 = float(np.sum(B ** 2))
    s2 = s ** 2
    resid, norm, crit = [], [], []
    m = A.shape[0]
    for lam in lams:
        f = s2 / (s2 + lam)              # filter factors
        # ||A x - b||^2 = sum((1-f)^2 beta^2) + (b2 - sum(beta^2))
        r2 = float(np.sum(((1.0 - f) ** 2)[:, None] * beta ** 2)
                   + b2 - float(np.sum(beta ** 2)))
        x2 = float(np.sum(((f / np.where(s > 0, s, 1.0)) ** 2)[:, None] * beta ** 2))
        resid.append(np.sqrt(max(r2, 0.0)))
        norm.append(np.sqrt(x2))
        if method == "gcv":
            trace = m - float(np.sum(f))
            crit.append(m * r2 / trace ** 2 if trace > 0 else np.inf)
    resid, norm = np.asarray(resid), np.asarray(norm)

    if method == "gcv":
        crit = np.asarray(crit)
        lam = float(lams[int(np.argmin(crit))])
        curve = {"lams": lams, "residual": resid, "norm": norm, "gcv": crit}
    elif method == "lcurve":
        x, y = np.log(np.maximum(resid, 1e-300)), np.log(np.maximum(norm, 1e-300))
        d1x, d1y = np.gradient(x), np.gradient(y)
        d2x, d2y = np.gradient(d1x), np.gradient(d1y)
        kappa = (d1x * d2y - d1y * d2x) / np.maximum(
            (d1x ** 2 + d1y ** 2) ** 1.5, 1e-300)
        lam = float(lams[int(np.argmax(kappa))])
        curve = {"lams": lams, "residual": resid, "norm": norm, "curvature": kappa}
    else:
        raise ValueError(f"unknown method {method!r}; use 'gcv' or 'lcurve'")
    return lam, curve


def pin_affine(delay, times, axis=0):
    """Remove the constant and the trend — the part ``A`` cannot see.

    The inversion's answer is unique only up to ``alpha + beta t``, so the
    number worth quoting is the one with both removed.  Everything the
    affine part carried is in the velocity instead.
    """
    a = np.asarray(delay, float)
    t = np.asarray(times, float)
    if a.shape[axis] != t.size:
        raise ValueError(f"delay has {a.shape[axis]} epochs, times {t.size}")
    a = np.moveaxis(a, axis, 0)
    basis = np.column_stack([np.ones_like(t), t - t.mean()])
    flat = a.reshape(t.size, -1)
    finite = np.isfinite(flat).all(axis=1)
    coef, *_ = np.linalg.lstsq(basis[finite], flat[finite], rcond=None)
    out = (flat - basis @ coef).reshape(a.shape)
    return np.moveaxis(out, 0, axis)


def pin_rate(delay, times, pairs, axis=0):
    """Pin the trend so the delay contributes nothing to the mean rate.

    :func:`pin_affine` removes the least-squares trend *in time*, which is one
    arbitrary choice out of the two-dimensional null space; on an uneven
    stack it still leaves the correction free to move an apparent velocity —
    measured on Mount Baker bedrock, which does not move, by 1.0 to 1.8 m/yr.

    Adding ``beta * t`` to the delay adds exactly ``beta`` to every pair's
    apparent velocity ``(a_j - a_i) / dt``, whatever the spacing.  So there is
    a ``beta`` that makes the mean of that contribution zero, and this
    function subtracts it: the correction then removes scatter and provably
    leaves the mean rate of every pair set alone.  It is no more *true* than
    any other representative — the affine part is unobservable either way —
    but it is the one that cannot be mistaken for a rate measurement.
    """
    a = np.moveaxis(np.asarray(delay, float), axis, 0)
    t = np.asarray(times, float)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    if a.shape[0] != t.size:
        raise ValueError(f"delay has {a.shape[0]} epochs, times {t.size}")
    dt = t[pr[:, 1]] - t[pr[:, 0]]
    shape = (-1,) + (1,) * (a.ndim - 1)
    v = (a[pr[:, 1]] - a[pr[:, 0]]) / dt.reshape(shape)
    beta = np.nanmean(v, axis=0)
    out = a - beta * (t - t.mean()).reshape(shape)
    out = out - np.nanmean(out, axis=0)
    return np.moveaxis(out, 0, axis)


def discarded_rate(delay, times, pairs, axis=0):
    """The mean apparent velocity the pinning throws away, in delay units/time.

    The size of the part of the affine null space this inversion cannot see,
    expressed the way it would be mistaken for signal.
    """
    a = np.moveaxis(np.asarray(delay, float), axis, 0)
    t = np.asarray(times, float)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    dt = t[pr[:, 1]] - t[pr[:, 0]]
    shape = (-1,) + (1,) * (a.ndim - 1)
    return np.nanmean((a[pr[:, 1]] - a[pr[:, 0]]) / dt.reshape(shape), axis=0)


@dataclass
class PathDelay:
    """Per-epoch apparent displacement from the path delay, and its provenance."""

    delay: np.ndarray                    # (n_epochs, ...) same unit as the input
    times: np.ndarray                    # (n_epochs,)
    pairs: np.ndarray                    # (n_pairs, 2)
    lam: float
    system: DoubleDifference = field(repr=False)
    residual_rms: np.ndarray | None = None
    pinned: object = "affine"
    dropped_rows: int = 0

    @property
    def n_epochs(self) -> int:
        return self.delay.shape[0]

    def pair_delay(self) -> np.ndarray:
        """What the delay contributes to each pair: ``a_j - a_i``."""
        pr = np.asarray(self.pairs, int)
        return self.delay[pr[:, 1]] - self.delay[pr[:, 0]]

    def correct(self, observations) -> np.ndarray:
        """Take the delay out of per-pair observations."""
        obs = np.asarray(observations, float)
        if obs.shape[0] != len(self.pairs):
            raise ValueError(
                f"{obs.shape[0]} observations for {len(self.pairs)} pairs")
        return obs - self.pair_delay()

    def frequency_response(self, period, spacing=None):
        """Gain this inversion applies at ``period`` — see the module docstring."""
        if spacing is None:
            spacing = float(np.median(np.abs(np.diff(np.sort(self.times)))))
        return frequency_response(period, spacing, self.lam)


def frequency_response(period, spacing, lam=0.0):
    """Fraction of a delay component of ``period`` the inversion recovers.

    One triplet of equally spaced epochs applies ``g = 4 sin^2(pi dt / T) / dt``
    to a component of period ``T``, and the Tikhonov solution scales the
    recovered amplitude by ``g^2 / (g^2 + lam)``.  With ``lam = 0`` that is 1
    at every period — least squares inverts the gain, however small — so the
    regularisation weight is the whole story::

        >>> round(float(frequency_response(1.0, 1.0 / 360)), 3)          # lam = 0
        1.0
        >>> round(float(frequency_response(1.0, 1.0 / 360, lam=1.0)), 3)  # diurnal
        0.012
        >>> round(float(frequency_response(1.0 / 180, 1.0 / 360, lam=1.0)), 6)
        1.0

    ``period`` and ``spacing`` share units (days, to match
    :attr:`~gpri_tools.network.Network.times`).  This is the response of the
    idealised equally spaced system: an actual stack mixes baselines, so read
    it as the scale of what survives, not a calibration.
    """
    T = np.asarray(period, float)
    dt = float(spacing)
    g = 4.0 * np.sin(np.pi * dt / T) ** 2 / dt
    return g ** 2 / (g ** 2 + float(lam))


def lambda_for_response(period, spacing, max_response=0.01):
    """The smallest ``lam`` whose response at ``period`` is at most that.

    Inverts :func:`frequency_response`: with ``g = 4 sin^2(pi dt / T) / dt``
    the solve returns ``g^2 / (g^2 + lam)`` of a component, so

        lam = g^2 (1 - rho) / rho

    is the weight that holds it to ``rho``.  This is how to keep the
    correction off a signal you mean to measure rather than hoping the
    cross-validated weight lands somewhere harmless — on the Baker
    campaigns the GCV weight spans three orders of magnitude, and with it
    the attenuation of a diurnal ranges from nothing to a quarter.

    Because ``g`` falls as ``1 / T^2``, protecting a slow component costs
    almost nothing fast: at a two-minute cadence, holding 24 h to 1 % still
    returns 99.5 % at 2 h and 100 % at 10 minutes.  The period to think
    about is the semidiurnal one, which keeps only 14 %.

    >>> lam = lambda_for_response(1.0, 2.0 / 1440)
    >>> round(float(frequency_response(1.0, 2.0 / 1440, lam)), 4)
    0.01
    >>> round(float(frequency_response(1 / 12, 2.0 / 1440, lam)), 4)
    0.9952
    """
    rho = float(max_response)
    if not 0.0 < rho <= 1.0:
        raise ValueError("max_response must be in (0, 1]")
    dt = float(spacing)
    g = 4.0 * np.sin(np.pi * dt / np.asarray(period, float)) ** 2 / dt
    return g ** 2 * (1.0 - rho) / rho


def system_response(A, times, period, lam):
    """Gain the regularised solve applies at ``period``, for *this* ``A``.

    :func:`frequency_response` is the closed form for one equally spaced
    triplet; a real stack mixes temporal baselines, and the longer ones are
    more sensitive at long periods, so the closed form understates what gets
    through.  This measures it on the operator actually being inverted:
    ``M A`` is ``V diag(s^2 / (s^2 + lam)) V^T``, and this returns the largest
    gain that filter applies to any phase of a harmonic of ``period``.
    """
    t = np.asarray(times, float)
    w = 2.0 * np.pi / float(period)
    H = np.column_stack([np.cos(w * t), np.sin(w * t)])
    H = H / np.linalg.norm(H, axis=0)
    _, s, Vt = np.linalg.svd(np.asarray(A, float), full_matrices=False)
    f = s ** 2 / (s ** 2 + float(lam))
    return float(np.linalg.norm(Vt.T @ (f[:, None] * (Vt @ H)), axis=0).max())


def lambda_for_system_response(A, times, period, max_response=0.01,
                               lo=None, hi=None, tol=1e-3):
    """Smallest ``lam`` holding :func:`system_response` at or under the target.

    Bisects on ``log lam``; the response is monotone decreasing in ``lam``, so
    the bracket only has to be wide enough.  Returns ``0.0`` when even an
    unregularised solve already meets the target.
    """
    rho = float(max_response)
    if not 0.0 < rho <= 1.0:
        raise ValueError("max_response must be in (0, 1]")
    A = np.asarray(A, float)
    if system_response(A, times, period, 0.0) <= rho:
        return 0.0
    spacing = float(np.median(np.abs(np.diff(np.sort(np.asarray(times, float))))))
    lo = float(lo if lo is not None else 1e-12)
    hi = float(hi if hi is not None else
               max(lambda_for_response(period, spacing, rho), 1.0) * 1e6)
    while system_response(A, times, period, hi) > rho:
        hi *= 100.0
        if hi > 1e30:                                  # pragma: no cover
            raise RuntimeError("no lam holds the response; check the period")
    while hi / max(lo, 1e-300) > 1.0 + tol:
        mid = np.sqrt(lo * hi) if lo > 0 else hi / 10.0
        if system_response(A, times, period, mid) > rho:
            lo = mid
        else:
            hi = mid
    return hi


def invert_path_delay(observations, pairs, times, lam=None, max_span=None,
                      max_triplets=None, normalise=True, weights=None,
                      pin="affine", nan_policy="drop", chunk=200_000,
                      lambda_method="gcv", protect_period=None,
                      max_response=0.01):
    """Per-epoch path delay from a stack of pair observations.

    Parameters
    ----------
    observations : (n_pairs, ...) array
        Line-of-sight displacement per pair, ``d_j - d_i`` — the convention of
        :func:`gpri_tools.timeseries.invert_network`, i.e.
        :func:`~gpri_tools.timeseries.los_displacement` applied to unwrapped
        phase.  Raw phase inverts the sign of the answer.
    pairs, times :
        As :attr:`~gpri_tools.network.Network.pairs` and
        :attr:`~gpri_tools.network.Network.times`.  Pass ``net.pairs,
        net.times`` for a :class:`~gpri_tools.network.Network`.
    lam : float, optional
        Regularisation weight.  Chosen by :func:`select_lambda` on the spatial
        mean series when omitted.
    protect_period : float, optional
        A period, in the units of ``times``, the correction must leave alone:
        ``lam`` is raised until :func:`system_response` of the operator
        actually being inverted is at most ``max_response`` there.  Pass
        ``1.0`` (a day) to keep the correction off a diurnal signal; it still
        returns essentially everything at an hour or less.
    weights : (n_rows,) or (n_pairs,) array, optional
        Row confidence.  A per-pair vector is turned into a per-row one by
        taking the smaller of the two pairs' weights.
    pin : {'affine', 'rate', False}
        Which representative of the two-dimensional null space to return.
        ``'affine'`` (``True``) removes the least-squares trend in time;
        ``'rate'`` removes the trend that would move the mean apparent
        velocity, so the correction cannot change a rate (:func:`pin_rate`);
        ``False`` leaves whatever the regularised solve produced.
    nan_policy : {'drop', 'raise', 'zero'}
        What to do with a double difference that is non-finite anywhere.

    Returns
    -------
    :class:`PathDelay`
    """
    obs = np.asarray(observations, float)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    t = np.asarray(times, float)
    if obs.shape[0] != pr.shape[0]:
        raise ValueError(f"{obs.shape[0]} observations for {pr.shape[0]} pairs")

    sys = double_difference(pr, t, max_span=max_span, max_triplets=max_triplets,
                            normalise=normalise)
    b = sys.apply(obs)

    keep = np.ones(sys.n_rows, bool)
    bad = ~np.isfinite(b.reshape(sys.n_rows, -1)).all(axis=1)
    if bad.any():
        if nan_policy == "raise":
            raise ValueError(
                f"{int(bad.sum())} of {sys.n_rows} double differences are "
                "non-finite; mask the stack to finite pixels or pass "
                "nan_policy='drop'")
        if nan_policy == "drop":
            keep = ~bad
        elif nan_policy == "zero":
            b = np.nan_to_num(b)
        else:
            raise ValueError(f"nan_policy must be one of {NAN_POLICY}")

    A, B, rows = sys.A[keep], b[keep], sys.rows[keep]
    if A.shape[0] < 3:
        raise ValueError(
            f"only {A.shape[0]} usable double differences; nothing to invert")

    if weights is not None:
        w = np.asarray(weights, float)
        if w.size == pr.shape[0]:
            w = np.minimum(w[rows[:, 0]], w[rows[:, 1]])
        if w.size != A.shape[0]:
            raise ValueError(
                f"weights has {w.size} entries, need {A.shape[0]} rows or "
                f"{pr.shape[0]} pairs")
        s = np.sqrt(np.maximum(w, 0.0))
        A = A * s[:, None]
        B = B * s.reshape((-1,) + (1,) * (B.ndim - 1))

    if lam is None:
        mean = B.reshape(B.shape[0], -1).mean(axis=1)
        lam, _ = select_lambda(A, mean, method=lambda_method)
    if protect_period is not None:
        lam = max(float(lam), float(lambda_for_system_response(
            A, t, protect_period, max_response)))

    delay = tikhonov(A, B, lam, chunk=chunk)
    resid = B - np.tensordot(A, delay, axes=(1, 0))
    rms = np.sqrt(np.nanmean(resid.reshape(resid.shape[0], -1) ** 2, axis=0))
    rms = rms.reshape(delay.shape[1:]) if delay.ndim > 1 else float(rms[0])

    if pin == "rate":
        delay = pin_rate(delay, t, pr)
    elif pin:
        delay = pin_affine(delay, t)
    if bad.any() and nan_policy == "drop" and bad.sum() > 0.5 * sys.n_rows:
        warnings.warn(
            f"dropped {int(bad.sum())} of {sys.n_rows} double differences as "
            "non-finite; the remaining chain may not constrain every epoch",
            stacklevel=2)

    return PathDelay(delay=delay, times=t, pairs=pr, lam=float(lam), system=sys,
                     residual_rms=rms,
                     pinned=("affine" if pin is True else pin),
                     dropped_rows=int(bad.sum()))
