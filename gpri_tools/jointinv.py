"""One inversion for the path delay and the ice's motion: every epoch, both antennas.

The correction ladder (:mod:`gpri_tools.aps`) and the temporal path delay
(:mod:`gpri_tools.pathdelay`) are separate steps, each fitted and then
subtracted.  This module replaces them with a single linear-Gaussian model of
the uncorrected, integrated series, solved once for a campaign.

The data are cell medians: the scene is cut into ground cells (200 m by
default, :func:`group_cells`) of four kinds — fit-half rock, ice, candidate
reference ground, and held-out rock — and each antenna's integrated LOS
series is reduced to one median per cell and epoch (:func:`cell_series`).
At every epoch ``k`` each cell-antenna value of the fit rock, the ice and the
candidates is modelled as

    y_k = F beta_k  +  s_k  +  E (G_k theta + a_k)  +  n_k

``F beta_k``
    a per-epoch trend over every cell in slant range, its square and height
    (:func:`trend_design`), flat prior;
``s_k``
    the path field, a Gaussian process over rock, ice and candidates alike
    with an exponential covariance whose amplitude may grow with range
    (:class:`PathPrior`), independent between epochs, scaled per epoch by
    the rock's own variance;
``theta``
    per ice or candidate cell and constant in time: an offset, a rate and,
    on a record of a day or more, a 24 h cosine and sine;
``a_k``
    ice motion that is neither, a Gaussian process over the ice cells; a
    candidate's own remainder is independent per cell;
``n_k``
    noise: an independent part per antenna, sized every epoch from the
    upper - lower difference and per cell from that cell's own difference
    (:func:`antenna_noise`), plus a part shared by both antennas whose size
    the rock's marginal likelihood picks (:func:`fit_path_prior`).

``theta`` is estimated by generalized least squares over every epoch at once,
with ``beta``, ``s``, ``a`` and ``n`` marginalised (:func:`solve`); the path
field is then predicted at the held-out rock, which never enters, and the
difference scores the whole model.

Candidates are coherent ground the reference rules leave out (inside the
outline buffer, or below the coherence threshold).  Whether one is stationary
is decided by the data: its motion terms are compared with a stationary
spread, its own formal variance from a free solve scaled by how far that
variance under-states the motion terms of held-out rock
(:func:`calibrated_spike`), in a two-component mixture whose evidence is summed
over every campaign that sees the location (:func:`shared_stationarity`).  The
posterior probability ``p`` then sets a prior precision ``p / spike`` on the
candidate's rate and harmonic and a remainder variance ``(1 - p) a2`` — a
stationary candidate acts as rock, a moving one only describes itself.

What it does not do: the path and the remainder are independent between
epochs, so the formal ``theta`` covariance is a lower bound; the held-out rock
residual is the honest error.  And ``theta`` carries no prior, so a signal
confined to the ice is returned at unit gain by construction — the
informative test is the opposite one, a signal shared by rock and ice, which
:func:`inject_shared` measures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

__all__ = [
    "CellSet", "group_cells", "split_cells", "cell_medians", "cell_series", "complete_cells",
    "antenna_noise", "shared_noise", "trend_design", "exponential_covariance", "PathPrior", "fit_path_prior",
    "Campaign", "JointResult", "solve", "motion_terms", "spike_by_range", "calibrated_spike",
    "shared_stationarity", "stationarity_probability", "motion_variance", "inject_shared", "CELL_ORIGIN", "DIURNAL",
]

#: Lower-left corner of the cell lattice in the radar's local map frame, m.
#: Fixed, so two campaigns from one tripod position share cell keys.
CELL_ORIGIN = -20000.0

DIURNAL = 1.0             # days


# ----------------------------------------------------------------- the cells
def _cell_keys(x, y, size, origin):
    ix = np.floor((np.asarray(x) - origin) / size).astype(np.int64)
    iy = np.floor((np.asarray(y) - origin) / size).astype(np.int64)
    return ix * 1_000_000 + iy


def group_cells(mask, x, y, size=200.0, min_pixels=1, origin=CELL_ORIGIN):
    """Pixels of ``mask`` grouped into ``size``-metre ground cells.

    Returns ``(groups, keys)``: a list of flat pixel-index arrays, and one
    integer key per group that identifies the cell on a fixed lattice, the
    same for every campaign whose ``x, y`` are in the same frame.  Groups with
    fewer than ``min_pixels`` pixels are dropped.
    """
    m = np.asarray(mask, bool)
    key = _cell_keys(np.asarray(x)[m], np.asarray(y)[m], size, origin)
    pix = np.flatnonzero(m.ravel())
    o = np.argsort(key, kind="stable")
    keys, start, count = np.unique(key[o], return_index=True, return_counts=True)
    keep = count >= min_pixels
    groups = [pix[o[s:s + c]] for s, c, k in zip(start, count, keep) if k]
    return groups, keys[keep]


def split_cells(mask, x, y, size=200.0, seed=0, origin=CELL_ORIGIN):
    """``mask`` split in two halves of whole ``size``-metre cells, at random.

    The cells are those of :func:`group_cells`; each is assigned entire to
    the first or the second half (about half the cells each, ``seed`` fixes
    the draw), so no cell of one half shares lattice ground with the other.
    Returns two boolean masks of the shape of ``mask``.
    """
    m = np.asarray(mask, bool)
    key = np.full(m.shape, -1, np.int64)
    key[m] = _cell_keys(np.asarray(x)[m], np.asarray(y)[m], size, origin)
    cells = np.unique(key[m])
    rng = np.random.default_rng(seed)
    first = rng.permutation(cells)[: cells.size // 2]
    a = m & np.isin(key, first)
    return a, m & ~a


def cell_medians(frame, groups):
    """The median of one image over each group (NaN-aware)."""
    flat = np.asarray(frame).ravel()
    return np.array([np.nanmedian(flat[g]) if g.size else np.nan for g in groups])


def cell_series(series, groups):
    """``(n_epochs, n_groups)`` medians of a ``(n_epochs, ...)`` series, relative to epoch 0.

    ``series`` may be a memory-mapped array; it is read one epoch at a time.
    A cell with no valid pixel at an epoch is NaN there (at every epoch, if
    that is epoch 0); :func:`complete_cells` finds the cells without a gap.
    """
    out = np.empty((len(series), len(groups)))
    for k in range(len(series)):
        out[k] = cell_medians(series[k], groups)
    out -= out[0]
    return out


def complete_cells(*series):
    """Cells whose median is finite at every epoch of every ``(n_epochs, n)`` series."""
    return np.logical_and.reduce([np.isfinite(np.asarray(s, float)).all(axis=0) for s in series])


@dataclass
class CellSet:
    """One class of cells: keys, position, range, height, pixel count, series."""
    key: np.ndarray
    xy: np.ndarray            # (n, 2) metres, local map frame
    r: np.ndarray             # slant range, m
    z: np.ndarray             # height, m
    n_px: np.ndarray          # pixels per cell
    yu: np.ndarray            # (n_epochs, n) upper antenna, mm, relative to epoch 0
    yl: np.ndarray            # (n_epochs, n) lower antenna

    @property
    def n(self):
        return int(self.key.size)

    @classmethod
    def build(cls, groups, keys, x, y, r, z, yu, yl):
        """Cells from pixel groups; a cell missing at any epoch of either antenna is dropped."""
        keep = complete_cells(yu, yl)
        if not keep.all():
            sel = np.flatnonzero(keep)
            groups = [groups[i] for i in sel]
            keys, yu, yl = np.asarray(keys)[sel], np.asarray(yu)[:, sel], np.asarray(yl)[:, sel]
        x, y, r, z = (np.asarray(v).ravel() for v in (x, y, r, z))
        return cls(key=np.asarray(keys, np.int64),
                   xy=np.array([[x[g].mean(), y[g].mean()] for g in groups]).reshape(-1, 2),
                   r=np.array([np.median(r[g]) for g in groups]),
                   z=np.array([np.nanmedian(z[g]) for g in groups]),
                   n_px=np.array([g.size for g in groups], float),
                   yu=np.asarray(yu, float), yl=np.asarray(yl, float))

    def subset(self, sel):
        sel = np.asarray(sel)
        return CellSet(self.key[sel], self.xy[sel], self.r[sel], self.z[sel],
                       self.n_px[sel], self.yu[:, sel], self.yl[:, sel])

    def as_dict(self, prefix):
        return {f"{prefix}_{k}": getattr(self, k) for k in
                ("key", "xy", "r", "z", "n_px", "yu", "yl")}

    @classmethod
    def from_dict(cls, d, prefix):
        return cls(*(np.asarray(d[f"{prefix}_{k}"]) for k in
                     ("key", "xy", "r", "z", "n_px", "yu", "yl")))


# ------------------------------------------------------------------ the noise
def _robust_var(a, axis):
    med = np.median(a, axis=axis, keepdims=True)
    return (np.median(np.abs(a - med), axis=axis) * 1.4826) ** 2


def antenna_noise(yu, yl, n_px, reference=None, floor=0.2):
    """Independent noise from the two antennas: per epoch and per cell.

    ``(u - l) sqrt(n_px)`` cancels everything the antennas share — motion,
    path, and noise common to both — and leaves twice the per-pixel variance
    of each antenna's own noise.  Returns ``(alpha, rho)``: ``alpha[k]`` is
    that per-pixel variance at epoch ``k`` over the ``reference`` cells, and
    ``rho[c]`` is cell ``c``'s own level relative to the reference, from the
    robust variance of its epoch-to-epoch steps (floored at ``floor``).  The
    noise variance of a cell's median is then ``alpha[k] rho[c] / n_px[c]``.
    """
    D = (np.asarray(yu, float) - np.asarray(yl, float)) * np.sqrt(np.asarray(n_px, float))[None]
    ref = np.arange(D.shape[1]) if reference is None else np.asarray(reference)
    alpha = np.maximum(_robust_var(D[:, ref], axis=1) / 2.0, 1e-6)
    steps = _robust_var(np.diff(D, axis=0), axis=0)
    rho = np.maximum(steps / np.median(steps[ref]), floor)
    return alpha, rho


def shared_noise(cells, radius=600.0, floor=0.0):
    """Noise both antennas share, per cell, as a multiple of each antenna's own noise.

    ``upper - lower`` cancels it, so it is measured on the epoch-to-epoch
    steps another way.  The path and the motion are spatially smooth, so a
    cell's step in the antenna mean, minus the pixel-weighted mean step of the
    cells within ``radius`` of it, is that cell's own noise; its robust
    variance (corrected for the neighbours' noise it borrows) less the
    independent part, a quarter of the variance of the ``upper - lower``
    step, is the shared part.  It is returned as a multiple of one antenna's
    independent step variance (half that of ``upper - lower``), which is how
    :func:`solve` takes it.  ``cells`` is a sequence of :class:`CellSet` in the
    order the solve stacks them.
    """
    xy = np.vstack([c.xy for c in cells])
    n_px = np.concatenate([c.n_px for c in cells])
    du = np.diff(np.hstack([c.yu for c in cells]), axis=0)
    dl = np.diff(np.hstack([c.yl for c in cells]), axis=0)
    mean_step, diff_step = 0.5 * (du + dl), du - dl
    d = cdist(xy, xy)
    near = (d <= radius) & (d > 0)
    w = near * n_px[None]
    m = np.maximum(near.sum(1), 1)
    local = mean_step @ (w / np.maximum(w.sum(1), 1e-9)[:, None]).T
    own = _robust_var(mean_step - local, axis=0) / (1.0 + 1.0 / m)
    indep_antenna = np.maximum(_robust_var(diff_step, axis=0) / 2.0, 1e-12)
    shared = np.maximum(own - indep_antenna / 2.0, 0.0)
    return np.maximum(shared / indep_antenna, floor)


# ----------------------------------------------------------- the path prior
def trend_design(r, z, terms=("1", "r", "r2", "z"), r0=6500.0, z0=2000.0):
    """Columns of the per-epoch trend: range and height in km about a centre."""
    rk = (np.asarray(r, float) - r0) / 1000.0
    zk = (np.asarray(z, float) - z0) / 1000.0
    cols = {"1": np.ones_like(rk), "r": rk, "r2": rk ** 2, "z": zk}
    return np.column_stack([cols[t] for t in terms])


def exponential_covariance(a, b, length):
    """``exp(-d / length)`` between two sets of points (n, 2) and (m, 2)."""
    return np.exp(-cdist(np.asarray(a, float), np.asarray(b, float)) / float(length))


@dataclass
class PathPrior:
    """Shape of the path field and of the noise shared by both antennas."""
    length: float = 1000.0          # m
    range_growth: float = 0.0       # amplitude x exp(range_growth per km from r0)
    common: float = 0.0             # shared noise as a multiple of the independent
    r0: float = 6500.0

    def amplitude(self, r):
        return np.exp(self.range_growth * (np.asarray(r, float) - self.r0) / 1000.0)

    def covariance(self, xy, r, xy2=None, r2=None):
        xy2 = xy if xy2 is None else xy2
        r2 = r if r2 is None else r2
        return (self.amplitude(r)[:, None] * self.amplitude(r2)[None]
                * exponential_covariance(xy, xy2, self.length))


def fit_path_prior(rock, alpha, rho=None, lengths=(250, 500, 1000, 2000, 4000),
                   growths=(0.0, 0.4, 0.8), commons=(0.0, 2.0, 9.0, 29.0),
                   terms=("1", "r", "r2", "z"), n_epochs=120):
    """Choose the path prior by marginal likelihood on the rock cells.

    The rock's antenna-mean series at ``n_epochs`` evenly spread epochs is
    modelled as trend + path + noise; per epoch the path's scale is profiled
    over a log grid and the trend is profiled out (restricted likelihood).
    Returns the :class:`PathPrior` with the highest summed likelihood.
    """
    y = 0.5 * (rock.yu + rock.yl)
    nt = y.shape[0]
    rho = np.ones(rock.n) if rho is None else np.asarray(rho, float)
    F = trend_design(rock.r, rock.z, terms)
    sub = np.unique(np.linspace(1, nt - 1, min(n_epochs, nt - 1)).astype(int))
    best = None
    for L in lengths:
        for g in growths:
            K0 = PathPrior(L, g).covariance(rock.xy, rock.r)
            for c in commons:
                ll = 0.0
                for k in sub:
                    nk = alpha[k] * rho / rock.n_px * (c + 0.5)
                    best_k = -np.inf
                    for s2 in np.geomspace(1e-3, 1e3, 13):
                        try:
                            cf = cho_factor(s2 * K0 + np.diag(nk), lower=True)
                        except np.linalg.LinAlgError:
                            continue
                        CiF = cho_solve(cf, F)
                        A = F.T @ CiF
                        beta = np.linalg.solve(A, CiF.T @ y[k])
                        res = y[k] - F @ beta
                        q = res @ cho_solve(cf, res)
                        logdet = 2 * np.log(np.diag(cf[0])).sum() + np.linalg.slogdet(A)[1]
                        best_k = max(best_k, -0.5 * (q + logdet))
                    ll += best_k
                if best is None or ll > best[0]:
                    best = (ll, L, g, c)
    return PathPrior(length=float(best[1]), range_growth=float(best[2]), common=float(best[3]))


# ---------------------------------------------------------------- the solve
@dataclass
class Campaign:
    """Everything one solve needs: epochs and the four cell classes."""
    t: np.ndarray                    # days from the first epoch
    fit: CellSet
    ice: CellSet
    cand: CellSet
    held: CellSet
    origin_hour: float = 0.0         # UTC hour of t = 0

    @property
    def spans_a_day(self):
        return float(self.t[-1] - self.t[0]) >= 0.98 * DIURNAL


@dataclass
class JointResult:
    """What :func:`solve` returns."""
    theta: np.ndarray                # (q, n_ice + n_cand): offset, rate[, cos, sin]
    cov: np.ndarray                  # (q (n_ice + n_cand)) square, formal
    held_prediction: np.ndarray      # (n_epochs, n_held) path predicted at held-out rock
    held_residual: np.ndarray        # antenna-mean held rock minus prediction
    ice_remainder: np.ndarray        # (n_epochs, n_ice) posterior a_k
    basis: np.ndarray                # (n_epochs, q) the time functions of theta
    n_ice: int
    prior: PathPrior
    path_scale: np.ndarray = field(repr=False, default=None)
    ice_scale: np.ndarray = field(repr=False, default=None)

    @property
    def ice_theta(self):
        return self.theta[:, :self.n_ice]

    @property
    def cand_theta(self):
        return self.theta[:, self.n_ice:]

    def ice_motion(self):
        """Per ice cell, motion beyond offset and rate: harmonic + remainder, mm."""
        q = self.basis.shape[1]
        per = self.basis[:, 2:] @ self.ice_theta[2:] if q > 2 else 0.0
        return per + self.ice_remainder

    def harmonic(self, origin_hour=0.0, cells="ice"):
        """24 h amplitude (mm), its formal sigma, and the UTC hour of peak."""
        th = self.ice_theta if cells == "ice" else self.cand_theta
        if th.shape[0] < 4:
            raise ValueError("the record is shorter than a day: no 24 h terms")
        q, n = self.theta.shape
        sl = slice(0, self.n_ice) if cells == "ice" else slice(self.n_ice, n)
        var = np.diag(self.cov).reshape(q, n)[:, sl]
        amp = np.hypot(th[2], th[3])
        sigma = np.sqrt(0.5 * (var[2] + var[3]))
        peak = (origin_hour + np.arctan2(th[3], th[2]) / (2 * np.pi) * 24.0) % 24.0
        return amp, sigma, peak


def _basis(t, harmonic):
    w = 2 * np.pi / DIURNAL
    cols = [np.ones_like(t), t]
    if harmonic:
        cols += [np.cos(w * t), np.sin(w * t)]
    return np.column_stack(cols)


def solve(camp, prior, ice_length=600.0, terms=("1", "r", "r2", "z"),
          cand_p=None, cand_spike=None, noise=None, harmonic=None, scales=None,
          shared=None):
    """The joint GLS solve for one campaign.

    Parameters
    ----------
    camp : :class:`Campaign`
    prior : :class:`PathPrior`, usually from :func:`fit_path_prior`
    ice_length : float
        Length scale of the ice remainder's covariance, m.
    cand_p : (n_cand,) array, optional
        Probability each candidate is stationary (0 by default: every
        candidate free, with its own remainder).
    cand_spike : (n_cand, q - 1) array, optional
        Variance of each motion term for stationary ground (from
        :func:`calibrated_spike`); needed when ``cand_p`` is given.
    noise : ``(alpha, rho)``, optional
        From :func:`antenna_noise` over fit + ice + candidates (in that order);
        computed here if not given.
    harmonic : bool, optional
        Fit the 24 h terms; default when the record spans a day.
    shared : (n_fit + n_ice + n_cand,) array, optional
        Noise both antennas share, per cell, as a multiple of the independent
        part (:func:`shared_noise`); without it every cell takes
        ``prior.common``.
    scales : ``(path_scale, ice_scale)``, optional
        Per-epoch prior scales of the path field and the ice remainder;
        estimated from the data when not given (pass a previous result's to
        solve other data with the same weights, as :func:`inject_shared` does).
    """
    fc, ic, kc, hc = camp.fit, camp.ice, camp.cand, camp.held
    nf, ni, nc = fc.n, ic.n, kc.n
    nth, ncell = ni + nc, nf + ni + nc
    t = np.asarray(camp.t, float)
    nt = t.size
    harmonic = camp.spans_a_day if harmonic is None else harmonic
    G = _basis(t, harmonic)
    q = G.shape[1]
    cl = (fc, ic, kc)
    Y = np.hstack([np.hstack([c.yu for c in cl]), np.hstack([c.yl for c in cl])])
    n_px = np.concatenate([c.n_px for c in cl])
    XY = np.vstack([c.xy for c in cl])
    R = np.concatenate([c.r for c in cl])
    Z = np.concatenate([c.z for c in cl])
    if noise is None:
        noise = antenna_noise(Y[:, :ncell], Y[:, ncell:], n_px, reference=np.arange(nf + ni))
    alpha, rho = noise
    n_eff = n_px / rho
    common = (np.full(ncell, prior.common) if shared is None
              else np.asarray(shared, float))
    Fc = trend_design(R, Z, terms)
    F = np.vstack([Fc, Fc])
    Ks = prior.covariance(XY, R)
    Ka = exponential_covariance(ic.xy, ic.xy, ice_length)
    Ksh = prior.covariance(hc.xy, hc.r, XY, R) if hc.n else np.zeros((0, ncell))
    Fh = trend_design(hc.r, hc.z, terms) if hc.n else np.zeros((0, F.shape[1]))

    # per-epoch scales: the path from the rock's scatter about its trend, the
    # ice remainder from what the ice has beyond its own rate and harmonic
    if scales is None:
        yf = 0.5 * (fc.yu + fc.yl)
        Ff = Fc[:nf]
        beta0 = np.linalg.lstsq(Ff, yf.T, rcond=None)[0]
        res_f = yf - (Ff @ beta0).T
        noise_f = alpha[:, None] * (rho[:nf] / fc.n_px)[None] * (common[:nf] + 0.5)
        s2 = np.maximum(np.mean(res_f ** 2, axis=1) - np.mean(noise_f, axis=1), 1e-3) \
            / np.mean(np.diag(Ks)[:nf])
        yi = 0.5 * (ic.yu + ic.yl)
        ri = yi - G @ np.linalg.lstsq(G, yi, rcond=None)[0]
        a2 = np.maximum(np.var(ri, axis=1) - s2 * np.mean(np.diag(Ks)[nf:nf + ni]), 1e-2)
    else:
        s2, a2 = (np.asarray(v, float) for v in scales)

    p = np.zeros(nc) if cand_p is None else np.asarray(cand_p, float)
    E = np.zeros((2 * ncell, nth))
    for a in range(2):
        E[a * ncell + nf:(a + 1) * ncell] = np.eye(nth)
    Ei, Ec = E[:, :ni], E[:, ni:]
    J = np.ones((2, 2))

    def sigma(k):
        return (np.kron(J, s2[k] * Ks) + Ei @ (a2[k] * Ka) @ Ei.T
                + Ec @ np.diag(a2[k] * (1.0 - p)) @ Ec.T
                + np.kron(J, np.diag(common * alpha[k] / n_eff))
                + np.diag(alpha[k] * np.tile(1.0 / n_eff, 2)))

    prec = np.full((q, nth), 1e-12)
    if cand_p is not None and nc:
        spike = np.asarray(cand_spike, float)
        for i in range(q - 1):
            prec[1 + i, ni:] = p / spike[:, i]

    N = np.zeros((q * nth, q * nth))
    rhs = np.zeros(q * nth)
    for k in range(1, nt):
        cf = cho_factor(sigma(k), lower=True)
        CiF, CiE = cho_solve(cf, F), cho_solve(cf, E)
        W = CiE.T - (E.T @ CiF) @ np.linalg.solve(F.T @ CiF, CiF.T)
        N += np.kron(np.outer(G[k], G[k]), W @ E)
        rhs += np.kron(G[k], W @ Y[k])
    Np = N + np.diag(prec.ravel())
    theta = np.linalg.solve(Np, rhs).reshape(q, nth)
    cov = np.linalg.inv(Np)

    pred = np.zeros((nt, hc.n))
    rem = np.zeros((nt, ni))
    for k in range(1, nt):
        cf = cho_factor(sigma(k), lower=True)
        rk = Y[k] - E @ (G[k] @ theta)
        CiF = cho_solve(cf, F)
        beta = np.linalg.solve(F.T @ CiF, CiF.T @ rk)
        e = cho_solve(cf, rk - F @ beta)
        if hc.n:
            pred[k] = Fh @ beta + np.tile(s2[k] * Ksh, (1, 2)) @ e
        rem[k] = (a2[k] * Ka) @ (Ei.T @ e)
    held_res = 0.5 * (hc.yu + hc.yl) - pred if hc.n else pred
    return JointResult(theta=theta, cov=cov, held_prediction=pred, held_residual=held_res,
                       ice_remainder=rem, basis=G, n_ice=ni, prior=prior,
                       path_scale=s2, ice_scale=a2)


# ------------------------------------------------------- deciding stationarity
def motion_terms(basis, series):
    """Least-squares rate (and 24 h cos, sin) of each column of ``series``."""
    return np.linalg.lstsq(np.asarray(basis, float), np.asarray(series, float), rcond=None)[0][1:]


def spike_by_range(held_terms, held_r, cand_r, edges=(0.0, 6500.0, 7500.0, np.inf), min_cells=4):
    """Per-candidate variance of each motion term on stationary ground.

    ``held_terms`` (q - 1, n_held) are the held-out rock's motion terms after
    its predicted path; their variance is taken in slant-range bins and
    assigned to each candidate by its range.  A bin with fewer than
    ``min_cells`` held cells uses the ``min_cells`` held cells nearest to it
    in range instead.
    """
    edges = np.asarray(edges, float)
    held_r = np.asarray(held_r, float)
    hb = np.digitize(held_r, edges) - 1
    var = np.zeros((edges.size - 1, held_terms.shape[0]))
    for b in range(edges.size - 1):
        sel = hb == b
        if sel.sum() < min_cells:
            gap = np.maximum(edges[b] - held_r, 0) + np.maximum(held_r - edges[b + 1], 0)
            sel = np.zeros(held_r.size, bool)
            sel[np.argsort(gap, kind="stable")[:min(min_cells, held_r.size)]] = True
        var[b] = np.var(held_terms[:, sel], axis=1) + 1e-6
    cb = np.clip(np.digitize(cand_r, edges) - 1, 0, edges.size - 2)
    return var[cb], var


def motion_variance(result):
    """Formal variance of every cell's motion terms (rate[, cos, sin]): (q - 1, n_ice + n_cand)."""
    q, n = result.theta.shape
    return np.diag(result.cov).reshape(q, n)[1:]


def calibrated_spike(camp, prior, calibrate=None, **solve_kwargs):
    """Each candidate's motion terms if it were stationary, and the free solve that measured them.

    A candidate beyond the rock carries the path field's extrapolation error
    in its motion terms, which held-out rock sitting among the fit rock never
    sees; so the stationary spread is taken per candidate from the solve's own
    covariance, scaled by how far that covariance under-states the truth on
    ground known to be still.  The held-out rock is entered for this one solve
    as free candidates (it is not in the solve that is scored), and

        kappa_j = mean over held cells of theta_j^2 / formal var_j

    for each motion term ``j``, over the held cells ``calibrate`` selects (a
    boolean mask; all by default) — leaving some out lets them test the
    classification afterwards.  Returns ``(spike, kappa, result)``: spike
    ``(n_cand, q - 1)`` is kappa times the candidates' formal variances, and
    ``result`` is the free solve (ice, then candidates, then held cells).
    """
    both = CellSet(*(np.concatenate([getattr(camp.cand, a), getattr(camp.held, a)],
                                    axis=1 if a in ("yu", "yl") else 0)
                     for a in ("key", "xy", "r", "z", "n_px", "yu", "yl")))
    empty = camp.held.subset(np.zeros(0, int))
    cal = Campaign(camp.t, camp.fit, camp.ice, both, empty, camp.origin_hour)
    res = solve(cal, prior, **solve_kwargs)
    var = motion_variance(res)[:, res.n_ice:]
    th = res.theta[1:, res.n_ice:]
    nc = camp.cand.n
    use = np.ones(camp.held.n, bool) if calibrate is None else np.asarray(calibrate, bool)
    kappa = np.maximum(np.mean(th[:, nc:][:, use] ** 2 / var[:, nc:][:, use], axis=1), 1.0)
    return (kappa[None] * var[:, :nc].T), kappa, res


def _diag_gauss_logpdf(x, var):
    return -0.5 * np.sum(x ** 2 / var + np.log(2 * np.pi * var), axis=0)


def shared_stationarity(observations, n_iter=300, prior_share=0.5):
    """Probability that each ground location is stationary, from every campaign at once.

    ``observations`` is a list, one entry per campaign, of
    ``(keys, terms, spike)``: candidate keys (n,), their motion terms
    (q - 1, n) and the stationary variance of each term (n, q - 1).  A
    location's log-likelihood ratio spike : slab is summed over the campaigns
    that see it; the slab (moving ground) has its own variance per campaign,
    and it and the stationary share are estimated by EM.

    Returns ``(keys, p, share, slab)``.
    """
    keys = np.unique(np.concatenate([np.asarray(o[0], np.int64) for o in observations]))
    index = {int(k): i for i, k in enumerate(keys)}
    rows = [np.array([index[int(k)] for k in o[0]], int) for o in observations]
    slab = [np.var(np.asarray(o[1], float), axis=1) for o in observations]
    share = prior_share
    p = np.full(keys.size, share)
    for _ in range(n_iter):
        llr = np.zeros(keys.size)
        for (k, th, sp), rw, T in zip(observations, rows, slab):
            th = np.asarray(th, float)
            sp = np.asarray(sp, float).T
            l0 = _diag_gauss_logpdf(th, sp)
            l1 = _diag_gauss_logpdf(th, sp + T[:, None])
            np.add.at(llr, rw, l0 - l1)
        p = 1.0 / (1.0 + np.exp(-np.clip(np.log(share / (1 - share)) + llr, -50, 50)))
        share = float(np.clip(p.mean(), 1e-3, 1 - 1e-3))
        for i, ((k, th, sp), rw) in enumerate(zip(observations, rows)):
            th = np.asarray(th, float)
            wv = 1.0 - p[rw]
            slab[i] = np.maximum((wv[None] * (th ** 2 - np.asarray(sp, float).T)).sum(1)
                                 / max(wv.sum(), 1e-9), 1e-6)
    return keys, p, share, slab


def stationarity_probability(observations, share, slab):
    """``p`` for further locations under a fitted :func:`shared_stationarity` mixture.

    ``observations`` as there, one entry per campaign in the same order as
    ``slab``; the mixture is not refitted, so ground of known state (held-out
    rock, ice) can be put through it to measure how often it is called still.
    Returns ``(keys, p)``.
    """
    keys = np.unique(np.concatenate([np.asarray(o[0], np.int64) for o in observations]))
    index = {int(k): i for i, k in enumerate(keys)}
    llr = np.zeros(keys.size)
    for (k, th, sp), T in zip(observations, slab):
        th = np.asarray(th, float)
        sp = np.asarray(sp, float).T
        rw = np.array([index[int(x)] for x in k], int)
        np.add.at(llr, rw, _diag_gauss_logpdf(th, sp) - _diag_gauss_logpdf(th, sp + np.asarray(T)[:, None]))
    return keys, 1.0 / (1.0 + np.exp(-np.clip(np.log(share / (1 - share)) + llr, -50, 50)))


# ------------------------------------------------------------------- testing
def inject_shared(camp, result, pattern, **solve_kwargs):
    """The 24 h term the solve returns on the ice for a signal shared by rock and ice.

    ``pattern`` is a function ``(r, z) -> amplitude (mm)`` evaluated at every
    fit-rock, ice and candidate cell; a 24 h cosine of that amplitude is the
    only thing in the data.  The solve is linear, so the returned ice cosine
    term is exactly how much of a path-like signal leaks into the ice's
    harmonic.  ``result`` is the real data's :class:`JointResult`, whose prior
    and per-epoch scales the injected solve reuses so the weights are the same.
    Returns ``(injected_on_ice, recovered_on_ice)``.
    """
    t = np.asarray(camp.t, float)
    sig = np.cos(2 * np.pi * t / DIURNAL)
    sig = sig - sig[0]

    def blank(c):
        a = np.asarray(pattern(c.r, c.z), float) * np.ones(c.n)
        y = sig[:, None] * a[None]
        return CellSet(c.key, c.xy, c.r, c.z, c.n_px, y, y.copy()), a

    fit, _ = blank(camp.fit)
    ice, a_ice = blank(camp.ice)
    cand, _ = blank(camp.cand)
    held, _ = blank(camp.held)
    fake = Campaign(t, fit, ice, cand, held, camp.origin_hour)
    nf = camp.fit.n
    # the noise level is a property of the real data, not of the injected signal
    real = np.hstack([c.yu for c in (camp.fit, camp.ice, camp.cand)])
    reall = np.hstack([c.yl for c in (camp.fit, camp.ice, camp.cand)])
    n_px = np.concatenate([c.n_px for c in (camp.fit, camp.ice, camp.cand)])
    noise = antenna_noise(real, reall, n_px, reference=np.arange(nf + camp.ice.n))
    res = solve(fake, result.prior, noise=noise, harmonic=True,
                scales=(result.path_scale, result.ice_scale), **solve_kwargs)
    return a_ice, res.ice_theta[2]
