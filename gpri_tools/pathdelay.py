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

try:
    from scipy.linalg import solveh_banded
except ImportError:  # pragma: no cover
    solveh_banded = None

__all__ = [
    "DoubleDifference", "PathDelay", "discarded_rate", "displacement_delay_field",
    "pair_delay_field",
    "double_difference", "frequency_response", "invert_path_delay",
    "gls_path_delay", "gls_resolution", "joint_design", "lambda_for_response",
    "roughness_penalty",
    "lambda_for_system_response", "pin_affine", "pin_rate",
    "response_from_resolution", "rewrap_to_chain", "select_lambda",
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
    if max_span is not None and times is None:
        raise ValueError("max_span needs times")
    t = None if times is None else np.asarray(times, float)
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


def _robust_tikhonov(A, B, lam, iterations, rows, coef, n_pairs, huber=3.0,
                     chunk=64):
    """Tikhonov solve with Huber weights on the *pairs*, per pixel.

    Errors in this system live on pairs, not rows — a wrapped or badly
    unwrapped interferogram is wrong in every double difference it enters —
    so the weight is attached to the pair.  Each row's residual, divided by
    the coefficient a pair enters it with, is that row's estimate of the
    pair's error; a pair's statistic is the **smallest** of those over the
    rows it appears in, so a good pair next to a bad one (large in one row,
    small in the rest) keeps its weight while the bad one, large in all of
    them, loses it.  Row-level Huber weights do not do this: the least
    squares start bends toward the bad pair and its neighbours' rows carry
    the leak, so they are weighted down with it and the sweeps crawl.  The
    pair statistic converges in two.

    Weights are ``min(1, huber * scale / stat)`` with ``scale`` the 1.4826
    MAD of the statistic over the pixel's pairs, recomputed each sweep (the
    weight is capped, so a shrinking scale cannot run away), and a row takes
    the smaller of its two pairs' weights.  A non-finite observation gets
    weight zero in its own pixel only.  With a single baseline there is no
    redundancy and a bad pair is fitted exactly, weight one: robustness
    needs the longer baselines.

    Every pixel gets its own normal matrix.  A row touches three epochs a
    few lags apart, so the matrix is assembled from nine products per row
    and is banded, and each pixel's solve is a banded Cholesky — linear in
    the epochs, not cubic.  (A design whose rows reach across the whole
    record falls back to a dense solve.)  Returns ``(x, rms, pair_weight)``
    with ``rms`` the per-pixel residual rms after the last sweep and
    ``pair_weight`` the mean weight of each pair over the pixels.
    """
    A = np.asarray(A, float)
    m, n = A.shape
    B = np.asarray(B, float)
    Y = B.reshape(m, -1)
    P = Y.shape[1]
    rows = np.asarray(rows, int)
    coef = np.asarray(coef, float)
    finite = np.isfinite(Y)
    Y0 = np.where(finite, Y, 0.0)
    X = tikhonov(A, Y0, lam, chunk=200_000)
    rms = np.empty(P)
    weight_sum = np.zeros(n_pairs)
    nz = [np.flatnonzero(A[r]) for r in range(m)]
    q = max(len(c) for c in nz)
    cols = np.zeros((m, q), int)
    vals = np.zeros((m, q))
    for r, c in enumerate(nz):
        cols[r, :c.size] = c
        vals[r, :c.size] = A[r, c]
    bw = max(int(c.max() - c.min()) for c in nz if c.size)
    banded = solveh_banded is not None and bw < n // 4
    # the (i, j) products of each row, as flat indices into either the
    # upper band form scipy wants (row bw + i - j, column j, i <= j) or the
    # full matrix
    ii, jj = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    ci, cj = cols[:, ii.ravel()], cols[:, jj.ravel()]           # (m, q^2)
    vv = vals[:, ii.ravel()] * vals[:, jj.ravel()]
    if banded:
        keep = ci <= cj
        ci, cj, vv = ci[keep], cj[keep], vv[keep]
        ncell = (bw + 1) * n
        cell = (bw + ci - cj) * n + cj
        rr = np.broadcast_to(np.arange(m)[:, None], keep.shape)[keep]
    else:
        ncell = n * n
        cell = (ci * n + cj).ravel()
        vv = vv.ravel()
        rr = np.repeat(np.arange(m), q * q)
    for s in range(0, P, int(chunk)):
        e = min(s + int(chunk), P)
        p = e - s
        Yc, Y0c, fc, Xc = Y[:, s:e], Y0[:, s:e], finite[:, s:e], X[:, s:e]
        base = (np.arange(p) * ncell)[:, None]                 # (p, 1)
        for _ in range(int(iterations)):
            R = np.where(fc, Yc - A @ Xc, np.nan)
            stat = np.full((n_pairs, e - s), np.inf)
            for slot in (0, 1):
                with np.errstate(invalid="ignore", divide="ignore"):
                    est = np.abs(R) / coef[:, slot][:, None]
                np.minimum.at(stat, rows[:, slot], np.where(np.isfinite(est), est, np.inf))
            stat = np.where(np.isfinite(stat), stat, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scale = 1.4826 * np.nanmedian(stat, axis=0)
            scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.inf)
            with np.errstate(invalid="ignore", divide="ignore"):
                wp = np.minimum(1.0, float(huber) * scale[None, :]
                                / np.maximum(stat, 1e-300))
            wp = np.where(np.isfinite(wp), wp, 0.0)
            W = np.where(fc, np.minimum(wp[rows[:, 0]], wp[rows[:, 1]]), 0.0)
            Wc = W.T                                             # (p, m)
            N = np.bincount((base + cell[None, :]).ravel(),
                            weights=(Wc[:, rr] * vv[None, :]).ravel(),
                            minlength=p * ncell)
            rhs = (Wc * Y0c.T) @ A                               # (p, n)
            if banded:
                ab = N.reshape(p, bw + 1, n)
                ab[:, bw, :] += float(lam)
                for ip in range(p):
                    Xc[:, ip] = solveh_banded(ab[ip], rhs[ip])
            else:
                N = N.reshape(p, n, n) + float(lam) * np.eye(n)
                Xc = np.linalg.solve(N, rhs[..., None])[..., 0].T
        X[:, s:e] = Xc
        R = np.where(fc, Yc - A @ Xc, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rms[s:e] = np.sqrt(np.nanmean(R ** 2, axis=0))
        weight_sum += np.nansum(wp, axis=1) if iterations else p
    return (X.reshape((n,) + B.shape[1:]), rms.reshape(B.shape[1:]),
            weight_sum / max(P, 1))


def rewrap_to_chain(observations, pairs, ambiguity):
    """Put every longer baseline on the cycle nearest the chain it spans.

    A pair unwrapped on its own is known only modulo ``ambiguity`` — half a
    wavelength of line-of-sight displacement on a two-way path — and a
    baseline long enough for the phase to cross that bound comes back a
    cycle away from where the chain of shorter pairs, which never crossed
    it, puts the same interval.  Measured at single look on
    `20170803_full`, the two-epoch pairs sit a whole cycle (8.7 mm) from
    the chain on 11 % of the bedrock and ice samples and the three-epoch
    pairs on 17 %, half of them each way; at 3 x 15 looks on `20170913`
    it is 3-4 % of the bedrock and about 1 % of the ice.  Those rows
    steer a least-squares fit.

    Each pair ``(i, k)`` with ``k > i + 1`` is moved by the whole number of
    cycles that brings it closest to the sum of the consecutive pairs
    ``(i, i+1) ... (k-1, k)``.  Nothing smaller than a cycle is touched, so a
    multilooked baseline keeps whatever independent value it measured.  A
    pair with no complete chain beneath it is left alone.

    Parameters
    ----------
    observations : (n_pairs, ...) array
        Line-of-sight displacement per pair, in the unit of ``ambiguity``.
    ambiguity : float
        The displacement one phase cycle stands for: ``wavelength / 2`` for
        the two-way path of ``los_displacement``.

    Returns
    -------
    rewrapped : array
        A copy, in the input's floating dtype.
    moved : (n_pairs,) array
        Fraction of finite pixels shifted, per pair; 0 on the chain itself.
    """
    obs = np.array(observations, copy=True)
    if not np.issubdtype(obs.dtype, np.floating):
        obs = obs.astype(float)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    if obs.shape[0] != pr.shape[0]:
        raise ValueError(f"{obs.shape[0]} observations for {pr.shape[0]} pairs")
    amb = float(ambiguity)
    if not amb > 0:
        raise ValueError("ambiguity must be positive")
    link = {int(i): p for p, (i, j) in enumerate(pr) if j == i + 1}
    moved = np.zeros(pr.shape[0])
    for p, (i, k) in enumerate(pr):
        if k <= i + 1:
            continue
        beneath = [link.get(int(e)) for e in range(int(i), int(k))]
        if any(q is None for q in beneath):
            continue
        predicted = obs[beneath].sum(axis=0)
        with np.errstate(invalid="ignore"):
            cycles = np.round((obs[p] - predicted) / amb)
        ok = np.isfinite(cycles)
        cycles = np.where(ok, cycles, 0.0)
        moved[p] = float(np.mean(cycles[ok] != 0)) if ok.any() else 0.0
        obs[p] = obs[p] - (cycles * amb).astype(obs.dtype, copy=False)
    return obs, moved


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

    Each pixel is fitted on its own finite epochs, so a pixel that is masked
    at every epoch — an incoherent one — costs nothing but itself; a pixel
    with fewer than two distinct finite epochs keeps its trend, there being
    nothing to fit it on, and only its mean comes off.
    """
    a = np.asarray(delay, float)
    t = np.asarray(times, float)
    if a.shape[axis] != t.size:
        raise ValueError(f"delay has {a.shape[axis]} epochs, times {t.size}")
    a = np.moveaxis(a, axis, 0)
    x = t - t.mean()
    flat = a.reshape(t.size, -1)
    good = np.isfinite(flat)
    y = np.where(good, flat, 0.0)
    w = good.astype(float)
    s0, s1, s2 = w.sum(axis=0), x @ w, (x ** 2) @ w
    sy, sxy = y.sum(axis=0), x @ y
    det = s0 * s2 - s1 ** 2
    fit = det > 1e-12 * np.maximum(s0 * s2, 1.0)       # two distinct epochs
    mean_only = ~fit & (s0 > 0)
    safe = np.where(fit, det, 1.0)
    alpha = np.where(fit, (s2 * sy - s1 * sxy) / safe, 0.0)
    beta = np.where(fit, (s0 * sxy - s1 * sy) / safe, 0.0)
    alpha[mean_only] = sy[mean_only] / s0[mean_only]
    out = (flat - (alpha + beta * x[:, None])).reshape(a.shape)
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
    robust: int = 0

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


def pair_delay_field(observations, pairs, times, mask, weights=None,
                     sigma=(5.0, 25.0), lam=None, protect_period=1.0,
                     max_response=0.01, chunk_rows=24, min_support=0.02,
                     pair_variance=None, robust=0, huber=3.0):
    """Per-pixel path delay from measured pair observations, kept where trusted.

    Each pixel's series is inverted for a per-epoch delay, and each epoch's
    field is then passed through :func:`gpri_tools.aps.turbulence_screen` on
    ``mask`` — so the answer is the part neighbouring trusted pixels agree on,
    not each pixel's own noise. The result is pinned with :func:`pin_rate`, so
    subtracting it cannot move a rate.

    Pass the **measured** observations, one per interferogram: at single look
    the long baselines are algebraic combinations of the chain and add
    nothing, but multilooked they are independent estimates and are most of
    what makes the per-pixel field worth having.

    Parameters
    ----------
    observations : (n_pairs, ...) array
        LOS displacement per pair, ``d_j - d_i``, in any consistent unit.
    mask : bool array
        Where the delay may be fitted — the coherent pixels. Smoothing the raw
        per-pixel field over the whole frame instead drags in delays fitted on
        incoherent ground, which is worse than doing nothing.
    weights : array, optional
        Per-pixel confidence for the spatial fit; mean coherence is the
        intended one.
    pair_variance : (n_pairs,) array, optional
        Per-pair error variance, shared across pixels — the Cramer-Rao form
        ``(1 - g^2) / (2 g^2)`` from each pair's coherence is the intended
        source.  Measured on `20170913`, weighting by it takes the held-out
        bedrock scatter from -56.2 % to -58.8 % at no cost in selectivity.
    protect_period : float, optional
        Passed to :func:`lambda_for_system_response`; ``1.0`` keeps the
        correction off a diurnal signal.
    robust : int
        Huber reweighting sweeps per pixel after the shared solve, as in
        :func:`invert_path_delay` (two converge).  Zero keeps the one
        factorisation for every pixel; anything above it solves each pixel
        of ``mask`` on its own, the rest keeping the shared solve since the
        screen never reads them.

    Returns
    -------
    field : (n_epochs, ...) array
        Subtract it from the per-epoch displacement.
    lam : float
    """
    from .aps import turbulence_screen

    obs = np.asarray(observations, float)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    t = np.asarray(times, float)
    sysd = double_difference(pr, t)

    # a per-pair variance becomes a per-row weight: each double difference is
    # only as good as the worse of the two pairs it is built from
    row_w = None
    if pair_variance is not None:
        v = np.maximum(np.asarray(pair_variance, float), 1e-30)
        if v.size != pr.shape[0]:
            raise ValueError(f"pair_variance has {v.size} entries for "
                             f"{pr.shape[0]} pairs")
        row_w = 1.0 / np.maximum(v[sysd.rows[:, 0]], v[sysd.rows[:, 1]])
        row_w = row_w / row_w.mean()

    A = sysd.A if row_w is None else sysd.A * np.sqrt(row_w)[:, None]
    coef = sysd.weights if row_w is None else sysd.weights * np.sqrt(row_w)[:, None]
    if lam is None:
        series = np.array([np.nanmean(x[mask]) for x in obs])
        b_ = sysd.apply(series)
        lam, _ = select_lambda(A, b_ if row_w is None else b_ * np.sqrt(row_w),
                               method="gcv")
    if protect_period is not None:
        lam = max(float(lam), float(lambda_for_system_response(
            A, t, protect_period, max_response)))

    M = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T)
    cube = np.empty((A.shape[1],) + obs.shape[1:], float)
    rows = obs.shape[1] if obs.ndim > 1 else 1
    for s in range(0, rows, int(chunk_rows)):
        e = min(s + int(chunk_rows), rows)
        b = sysd.apply(obs[:, s:e])
        if row_w is not None:
            b = b * np.sqrt(row_w).reshape((-1,) + (1,) * (b.ndim - 1))
        b2 = b.reshape(b.shape[0], -1)
        x = np.tensordot(M, b2, axes=(1, 0))
        if int(robust) > 0:
            # only the trusted pixels are read by the screen, so only they
            # get the per-pixel sweeps
            sel = (np.flatnonzero(np.asarray(mask, bool)[s:e].ravel())
                   if obs.ndim > 1 else np.arange(b2.shape[1]))
            if sel.size:
                x[:, sel], _, _ = _robust_tikhonov(
                    A, b2[:, sel], lam, int(robust), sysd.rows, coef,
                    pr.shape[0], huber=huber)
        cube[:, s:e] = pin_rate(x.reshape((A.shape[1],) + b.shape[1:]), t, pr)

    field = np.empty_like(cube)
    for k in range(cube.shape[0]):
        scr, _ = turbulence_screen(cube[k], mask, sigma=tuple(sigma),
                                   weights=weights, wrapped=False,
                                   min_support=min_support)
        field[k] = scr
    return pin_rate(field, t, pr), float(lam)


def displacement_delay_field(displacement, pairs, times, mask, weights=None,
                             sigma=(5.0, 25.0), lam=None, protect_period=1.0,
                             max_response=0.01, chunk_rows=24, min_support=0.02):
    """The path delay of a displacement cube — :func:`pair_delay_field` after
    re-differencing the cube into the given pairs.

    The pipeline this is for: run the spatial ladder, then hand its output
    here to take out what is left that is fast and spatially coherent.  Note
    that pairs re-differenced from a cube are not independent measurements,
    so only the shortest baseline carries anything; hand
    :func:`pair_delay_field` the measured multilooked pairs when there are
    several baselines to exploit.
    """
    d = np.asarray(displacement, float)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    return pair_delay_field(d[pr[:, 1]] - d[pr[:, 0]], pr, times, mask,
                            weights=weights, sigma=sigma, lam=lam,
                            protect_period=protect_period,
                            max_response=max_response, chunk_rows=chunk_rows,
                            min_support=min_support)


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


def response_from_resolution(R, times, period):
    """Gain a resolution operator applies to a harmonic of ``period``.

    ``R`` maps the true per-epoch delay to the estimated one, so this is what
    the estimator returns of a component at that period — the number to quote
    before claiming a correction did or did not touch a signal.  Every
    estimator here is judged by the same measure; only the way ``R`` is built
    differs.

    ``period`` may be an array, and the gain then comes back with its shape —
    a response curve costs one ``R`` rather than one per period.
    """
    t = np.asarray(times, float)
    R = np.asarray(R, float)
    T = np.asarray(period, float)
    gains = np.empty(T.size)
    for i, p in enumerate(T.reshape(-1)):
        w = 2.0 * np.pi / float(p)
        H = np.column_stack([np.cos(w * t), np.sin(w * t)])
        H = H / np.linalg.norm(H, axis=0)
        gains[i] = float(np.linalg.norm(R @ H, axis=0).max())
    return float(gains[0]) if T.ndim == 0 else gains.reshape(T.shape)


def system_response(A, times, period, lam):
    """Gain the regularised double-difference solve applies at ``period``.

    :func:`frequency_response` is the closed form for one equally spaced
    triplet; a real stack mixes temporal baselines, and the longer ones are
    more sensitive at long periods, so the closed form understates what gets
    through.  This measures it on the operator actually being inverted:
    ``M A`` is ``V diag(s^2 / (s^2 + lam)) V^T``.

    ``period`` may be an array — the whole curve for one SVD of ``A``.
    """
    _, s_, Vt = np.linalg.svd(np.asarray(A, float), full_matrices=False)
    f = s_ ** 2 / (s_ ** 2 + float(lam))
    return response_from_resolution(Vt.T @ (f[:, None] * Vt), times, period)


def _bisect_lambda(response_at, max_response, hi_guess, tol=1e-3):
    """Smallest ``lam`` with ``response_at(lam) <= max_response``."""
    rho = float(max_response)
    if not 0.0 < rho <= 1.0:
        raise ValueError("max_response must be in (0, 1]")
    if response_at(0.0) <= rho:
        return 0.0
    lo, hi = 1e-12, float(hi_guess)
    while response_at(hi) > rho:
        hi *= 100.0
        if hi > 1e30:                                  # pragma: no cover
            raise RuntimeError("no lam holds the response; check the period")
    while hi / max(lo, 1e-300) > 1.0 + tol:
        mid = np.sqrt(lo * hi)
        if response_at(mid) > rho:
            lo = mid
        else:
            hi = mid
    return hi


def lambda_for_system_response(A, times, period, max_response=0.01,
                               lo=None, hi=None, tol=1e-3):
    """Smallest ``lam`` holding :func:`system_response` at or under the target.

    Bisects on ``log lam``; the response is monotone decreasing in ``lam``.
    Returns ``0.0`` when even an unregularised solve already meets the target.
    """
    A = np.asarray(A, float)
    spacing = float(np.median(np.abs(np.diff(np.sort(np.asarray(times, float))))))
    guess = (hi if hi is not None else
             max(lambda_for_response(period, spacing, max_response), 1.0) * 1e6)
    return _bisect_lambda(lambda c: system_response(A, times, period, c),
                          max_response, guess, tol=tol)


def joint_design(pairs, times):
    """``[D | dt]``: per-epoch delay differences beside a steady-motion column.

    A pair reads ``b_p = (a_j - a_i) + v dt_p``.  Stacking that is the whole
    model, and estimating ``v`` alongside the delays eliminates it once,
    inside the weighted solve, instead of twice — which is what forming
    double differences and then undoing their induced correlation amounts to.
    """
    pr = np.asarray(pairs, int).reshape(-1, 2)
    t = np.asarray(times, float)
    D = np.zeros((pr.shape[0], t.size))
    D[np.arange(pr.shape[0]), pr[:, 1]] = 1.0
    D[np.arange(pr.shape[0]), pr[:, 0]] -= 1.0
    return np.column_stack([D, t[pr[:, 1]] - t[pr[:, 0]]])


def roughness_penalty(n_epochs):
    """``L^T L`` for the second difference of the delay — a smoothness prior.

    Penalising curvature is **low-pass**: it hits short periods hardest, which
    is the opposite of what a correction aimed at fast fluctuation wants.
    Measured on a 400-epoch chain at two-minute cadence, holding 24 h to 1 %
    with this prior leaves 0.003 at 2 h against 0.293 for a ridge and 0.946
    for the double difference.  The double difference is selective because its
    *data* term differences twice and so barely constrains slow components at
    all — no choice of prior in the pair domain reproduces that.
    """
    n = int(n_epochs)
    L = np.zeros((max(n - 2, 0), n))
    for k in range(n - 2):
        L[k, k], L[k, k + 1], L[k, k + 2] = 1.0, -2.0, 1.0
    return L.T @ L


def gls_resolution(G, weight, lam, n_epochs, penalty="ridge"):
    """Resolution operator of the joint solve, for the delay block.

    This is a different estimator from the double-difference solve, not a
    reweighting of it: there the data term carries the operator's ``1/T^2``
    emphasis, so holding a slow period costs almost nothing fast, while the
    pair design differences only once and the same protection shrinks short
    periods too.  :func:`response_from_resolution` measures both on the same
    footing.
    """
    W = np.asarray(weight, float)
    GtWG = G.T @ (W[:, None] * G)
    P = _penalty_matrix(penalty, n_epochs)
    A_ = GtWG + float(lam) * P
    try:
        return np.linalg.solve(A_, GtWG)[:n_epochs, :n_epochs]
    except np.linalg.LinAlgError:               # pragma: no cover
        return np.linalg.lstsq(A_, GtWG, rcond=None)[0][:n_epochs, :n_epochs]


def _penalty_matrix(penalty, n_epochs):
    """The prior's quadratic form, over ``[delay, motion]``.

    The motion term is never penalised: it is a parameter of the model, not
    something to shrink.
    """
    n = int(n_epochs)
    P = np.zeros((n + 1, n + 1))
    if penalty == "ridge":
        P[:n, :n] = np.eye(n)
    elif penalty == "roughness":
        P[:n, :n] = roughness_penalty(n)
    else:
        raise ValueError(f"penalty must be 'ridge' or 'roughness', not {penalty!r}")
    return P


def gls_path_delay(observations, pairs, times, variance=None, lam=None,
                   protect_period=1.0, max_response=0.01, pin="rate",
                   penalty="ridge", chunk=200_000):
    """Delay and steady motion together, weighted by the pair variances.

    The poster's estimator — :func:`invert_path_delay` — solves
    ``(A^T A + lam I)^-1 A^T b`` on double differences, which treats every row
    as an independent measurement of equal variance.  They are neither: each
    pair enters two triplets, so neighbouring rows are correlated at -0.5 even
    when the pair errors are independent, and coherence makes the pair
    variances unequal besides.  Solving in the pair domain instead never
    manufactures that correlation.

    Parameters
    ----------
    observations : (n_pairs, ...) array
        LOS displacement per pair, ``d_j - d_i``.
    penalty : {'ridge', 'roughness'}
        The prior.  ``ridge`` penalises the delay's amplitude and is the
        default; with the pair design's single difference it still suppresses
        slow periods faster than short ones, but less sharply than the double
        difference does.  ``roughness`` penalises the second time difference
        and is **low-pass** — measured on a 400-epoch chain, holding 24 h to
        1 % leaves 0.003 at 2 h, so it is the wrong prior for a correction
        meant to remove fast fluctuation.  It is here because the choice
        should be visible rather than assumed.
    variance : (n_pairs,) array, optional
        Per-pair error variance, shared across pixels — coherence is the
        intended source.  ``None`` weights every pair alike, which is still
        not the poster's estimator, because the pair domain has no induced
        correlation to ignore.
    lam : float, optional
        Chosen by generalised cross-validation on the mask mean when omitted.
    protect_period : float, optional
        Raised until :func:`response_from_resolution` of this estimator is at
        most ``max_response`` there.

    Returns
    -------
    delay : (n_epochs, ...) array
    motion : (...) array
        The steady rate, in the observations' unit per unit of ``times``.
    lam : float
    """
    obs = np.asarray(observations, float)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    t = np.asarray(times, float)
    if obs.shape[0] != pr.shape[0]:
        raise ValueError(f"{obs.shape[0]} observations for {pr.shape[0]} pairs")
    n = t.size
    G = joint_design(pr, t)
    w = (np.ones(pr.shape[0]) if variance is None
         else 1.0 / np.maximum(np.asarray(variance, float), 1e-30))
    if w.size != pr.shape[0]:
        raise ValueError(f"variance has {w.size} entries for {pr.shape[0]} pairs")

    GtWG = G.T @ (w[:, None] * G)
    P = _penalty_matrix(penalty, n)
    if lam is None:
        mean = obs.reshape(obs.shape[0], -1).mean(axis=1)
        rhs = G.T @ (w * mean)
        lams = np.logspace(np.log10(max(np.trace(GtWG) / (n + 1), 1e-12)) - 8,
                           np.log10(max(np.trace(GtWG) / (n + 1), 1e-12)) + 2, 40)
        best, lam = np.inf, float(lams[0])
        m_obs = mean.size
        for c in lams:
            x = np.linalg.solve(GtWG + c * P, rhs)
            r2 = float(np.sum(w * (mean - G @ x) ** 2))
            trace = m_obs - float(np.trace(
                np.linalg.solve(GtWG + c * P, GtWG)))
            if trace > 0:
                g = m_obs * r2 / trace ** 2
                if g < best:
                    best, lam = g, float(c)
    if protect_period is not None:
        lam = max(float(lam), _bisect_lambda(
            lambda c: response_from_resolution(
                gls_resolution(G, w, c, n, penalty), t, protect_period),
            max_response, max(np.trace(GtWG) / (n + 1), 1.0) * 1e6))

    spatial = obs.shape[1:]
    Y = obs.reshape(obs.shape[0], -1)
    normal = GtWG + float(lam) * P
    try:                                   # lam = 0 leaves the affine null space
        chol = np.linalg.cholesky(normal)
        solve = lambda R_: np.linalg.solve(chol.T, np.linalg.solve(chol, R_))
    except np.linalg.LinAlgError:
        solve = lambda R_: np.linalg.lstsq(normal, R_, rcond=None)[0]
    X = np.empty((n + 1, Y.shape[1]))
    step = Y.shape[1] if chunk is None else int(chunk)
    for s0 in range(0, Y.shape[1], step):
        e = min(s0 + step, Y.shape[1])
        X[:, s0:e] = solve(G.T @ (w[:, None] * Y[:, s0:e]))
    delay = X[:n].reshape((n,) + spatial)
    motion = X[n].reshape(spatial) if spatial else float(X[n, 0])
    # the affine null space is shared between the delay's trend and the rate:
    # taking a trend out of the delay puts it into the motion, or the two stop
    # predicting the same pairs
    if pin == "rate":
        shift = discarded_rate(delay, t, pr)
        delay = pin_rate(delay, t, pr)
        motion = motion + shift
    elif pin:
        tc = t - t.mean()
        flat = np.moveaxis(delay, 0, 0).reshape(n, -1)
        slope = (tc @ flat) / float(tc @ tc)
        delay = pin_affine(delay, t)
        motion = motion + slope.reshape(spatial) if spatial else motion + float(slope[0])
    return delay, motion, float(lam)


def invert_path_delay(observations, pairs, times, lam=None, max_span=None,
                      max_triplets=None, normalise=True, weights=None,
                      pin="affine", nan_policy="drop", chunk=200_000,
                      lambda_method="gcv", protect_period=None,
                      max_response=0.01, robust=0, huber=3.0):
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
    robust : int
        Number of Huber reweighting sweeps after the least-squares solve
        (0, the default, is plain least squares).  The weight sits on the
        pair: one whose implied error exceeds ``huber`` robust scales in
        every row it enters is weighted down in that pixel, so a pair a
        whole cycle off — see :func:`rewrap_to_chain` for the cheaper fix
        when the chain is there to tell — stops steering the fit.  Two
        sweeps converge; each costs a solve per pixel, and a single baseline
        has no redundancy to be robust with.

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

    A, B, rows, coef = sys.A[keep], b[keep], sys.rows[keep], sys.weights[keep]
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
        coef = coef * s[:, None]

    if lam is None:
        mean = B.reshape(B.shape[0], -1).mean(axis=1)
        lam, _ = select_lambda(A, mean, method=lambda_method)
    if protect_period is not None:
        lam = max(float(lam), float(lambda_for_system_response(
            A, t, protect_period, max_response)))

    if int(robust) > 0:
        delay, rms, _ = _robust_tikhonov(A, B, lam, int(robust), rows, coef,
                                         pr.shape[0], huber=huber)
        rms = rms if delay.ndim > 1 else float(rms)
    else:
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
                     dropped_rows=int(bad.sum()), robust=int(robust))
