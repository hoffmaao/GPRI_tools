"""Modes of the bedrock residual: what the corrected rock still does together.

After the correction ladder the bedrock is supposed to be noise.  On the
Baker campaigns it is not quite: the held-out half of the bedrock — pixels
the ladder never fitted — keeps a 24 h harmonic in its mean series
(``docs/atmosphere.md``).  The question this module asks is how that
residual is organised.  Its singular value decomposition over the fitted bedrock
splits it into **temporal modes**, epoch series the pixels share, and
**loadings**, how much of each series a pixel carries.  A few strong modes
against a shuffled null say the residual is low-rank in time; whether it is
also organised in space is a separate question, and it is decided on the
other half of the bedrock:

1. the loadings are interpolated from the fitted pixels to the held-out
   ones with :func:`gpri_tools.aps.turbulence_screen`, the same normalised
   convolution the ladder uses, and the correction that produces is scored
   on the held-out pixels (:func:`mode_screens`, :func:`mode_correction`);
2. the held-out pixels are also regressed on the mode series directly
   (:func:`mode_projection`) — the most the modes could take out if every
   pixel's loading were known, which is the self-fit bound the interpolated
   correction is read against.

Where the two agree the loading is a smooth field and the modes are a
correction; where the interpolated correction falls far short of the bound
the loading changes from one pixel to the next and the modes describe the
residual without offering a way to remove it from a pixel that was not
fitted.  The functions here report both; which it is on a given campaign is
in the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RockModes", "fit_modes", "mode_correction", "mode_projection",
           "mode_screens"]


@dataclass
class RockModes:
    """The leading modes of a residual over the fitted pixels.

    Attributes
    ----------
    temporal : (n_epochs, k) array
        Mode series, in the residual's units **at unit loading**: each mode's
        loading is scaled to RMS 1 over the fitted pixels *weighted*, so a
        series is the size of the mode in a pixel that carries it at the
        typical amount.  The plain RMS equals it only when the weights are
        equal; at coherence weights near 0.7 it is about 1.19.
    loading : (k, ...) array
        Per-pixel loading of each mode on the residual's grid, NaN where the
        pixel was not fitted.
    singular : (k,) array
        Singular values of the weighted, centred residual.
    null : (k,) array
        The same from the residual with each pixel's epochs shuffled — the
        temporal structure destroyed, every pixel's variance kept.  A mode
        whose singular value sits at the null is what noise of this size and
        count produces on its own.
    explained : (k,) array
        Fraction of the weighted variance each mode carries.
    n_pixels : int
        Pixels the decomposition was fitted on.
    """

    temporal: np.ndarray
    loading: np.ndarray
    singular: np.ndarray
    null: np.ndarray
    explained: np.ndarray
    n_pixels: int

    @property
    def k(self) -> int:
        return int(self.temporal.shape[1])

    @property
    def ratio(self) -> np.ndarray:
        """Singular value over its shuffled null, mode by mode."""
        return self.singular / self.null


def fit_modes(residual, mask, weights=None, k=10, null_seed=0):
    """Leading modes of ``residual`` over the pixels of ``mask``.

    Each fitted pixel's series is centred over the epochs (a pixel's mean is
    an offset, not a mode) and scaled by the square root of its weight, and
    the ``(n_epochs, n_pixels)`` matrix is decomposed.  Pixels with a
    non-finite epoch are left out.

    Parameters
    ----------
    residual : (n_epochs, ...) array
        The series left after the corrections, in any unit.
    mask : bool array
        The pixels to fit on.  For an honest score keep half the bedrock out
        of it and hand that half to :func:`mode_correction`'s scoring.
    weights : array, optional
        Per-pixel weight of the same shape as ``mask``; mean coherence is
        the intended one.
    k : int
        Modes to keep.
    null_seed : int
        Seed of the epoch shuffle behind :attr:`RockModes.null`.

    Returns
    -------
    :class:`RockModes`
    """
    d = np.asarray(residual, float)
    m = np.asarray(mask, bool)
    if d.shape[1:] != m.shape:
        raise ValueError(f"residual grid {d.shape[1:]} does not match the "
                         f"mask {m.shape}")
    cols = np.flatnonzero(m.ravel())
    X = d.reshape(d.shape[0], -1)[:, cols]
    ok = np.isfinite(X).all(axis=0)
    X, cols = X[:, ok], cols[ok]
    if weights is None:
        w = np.ones(cols.size)
    else:
        w = np.clip(np.nan_to_num(np.asarray(weights, float).ravel()[cols]), 0.0, None)
        ok = w > 0
        X, cols, w = X[:, ok], cols[ok], w[ok]
    k = int(min(k, X.shape[0], X.shape[1]))
    if k < 1:
        raise ValueError("no finite pixel to fit the modes on")
    X = X - X.mean(axis=0)
    sw = np.sqrt(w)
    Xw = X * sw[None, :]
    U, s, Vt = np.linalg.svd(Xw, full_matrices=False)
    total = float((s ** 2).sum()) or 1.0

    rng = np.random.default_rng(null_seed)
    Xn = Xw.copy()
    for c in range(Xn.shape[1]):
        Xn[:, c] = Xn[rng.permutation(Xn.shape[0]), c]
    sn = np.linalg.svd(Xn, compute_uv=False)

    scale = np.sqrt(cols.size)
    temporal = U[:, :k] * s[:k] / scale
    load_px = Vt[:k] / sw[None, :] * scale
    loading = np.full((k,) + d.shape[1:], np.nan)
    flat = loading.reshape(k, -1)
    flat[:, cols] = load_px
    return RockModes(temporal=temporal, loading=loading, singular=s[:k],
                     null=sn[:k], explained=(s[:k] ** 2) / total,
                     n_pixels=int(cols.size))


def mode_screens(modes, mask, k=None, sigma=(5.0, 25.0), weights=None,
                 min_support=0.02):
    """Each mode's loading interpolated over the frame from the fitted pixels.

    :func:`gpri_tools.aps.turbulence_screen` in its unwrapped form, on the
    loading with ``mask`` as the support — the pixels whose loadings are
    known.  A held-out pixel gets the loading its fitted neighbours agree on,
    which is the only loading it can be given without fitting it.

    Returns an ``(k, ...)`` array, 0 where the support is thinner than
    ``min_support``.
    """
    from .aps import turbulence_screen

    kk = modes.k if k is None else int(min(k, modes.k))
    out = np.empty((kk,) + modes.loading.shape[1:], float)
    for i in range(kk):
        out[i], _ = turbulence_screen(modes.loading[i], mask, sigma=tuple(sigma),
                                      weights=weights, wrapped=False,
                                      min_support=min_support)
    return out


def mode_correction(modes, screens):
    """The residual the modes and their interpolated loadings predict.

    ``sum_i temporal[:, i] * screens[i]`` over the modes in ``screens``,
    an ``(n_epochs, ...)`` array in the residual's units.  Subtract it from
    the residual; on pixels that were not fitted it is a prediction, and the
    score there is the honest one.
    """
    scr = np.asarray(screens, float)
    kk = scr.shape[0]
    if kk > modes.k:
        raise ValueError(f"{kk} screens for {modes.k} modes")
    return np.tensordot(modes.temporal[:, :kk], scr, axes=(1, 0))


def mode_projection(modes, residual, mask, k=None, coefficients=False):
    """Each pixel of ``mask`` regressed on the mode series: the self-fit bound.

    An offset plus the first ``k`` mode series are fitted to every pixel's
    own series, and the fitted part (without the offset) is returned, NaN off
    ``mask`` and on pixels with a non-finite epoch.  On pixels that were not
    in the decomposition this is the most the modes could remove if each
    pixel's loading were measured rather than interpolated; the difference
    between it and :func:`mode_correction` is what the interpolation loses.

    With ``coefficients`` the per-pixel loadings that regression found come
    back as well, a ``(k, ...)`` array in the same units as
    :attr:`RockModes.loading` — beside the interpolated loading of
    :func:`mode_screens` at the same pixels, that is the transfer test in
    one scatter.
    """
    d = np.asarray(residual, float)
    m = np.asarray(mask, bool)
    kk = modes.k if k is None else int(min(k, modes.k))
    T = modes.temporal[:, :kk]
    G = np.column_stack([np.ones(T.shape[0]), T])
    cols = np.flatnonzero(m.ravel())
    X = d.reshape(d.shape[0], -1)[:, cols]
    ok = np.isfinite(X).all(axis=0)
    coef = np.linalg.lstsq(G, X[:, ok], rcond=None)[0]
    out = np.full(d.shape, np.nan)
    flat = out.reshape(d.shape[0], -1)
    fitted = np.full(X.shape, np.nan)
    fitted[:, ok] = T @ coef[1:]
    flat[:, cols] = fitted
    if not coefficients:
        return out
    own = np.full((kk,) + d.shape[1:], np.nan)
    own_flat = own.reshape(kk, -1)
    own_cols = np.full((kk, cols.size), np.nan)
    own_cols[:, ok] = coef[1:]
    own_flat[:, cols] = own_cols
    return out, own
