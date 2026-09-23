"""Speckle tracking: displacement from the amplitude, with no phase at all.

Every other estimator in this package measures displacement from phase,
which is precise and fragile: it needs coherence, and it is ambiguous
beyond half a wavelength.  Cross-correlating the *amplitude* pattern of two
acquisitions measures the same displacement in a way that is coarse —
fractions of a resolution cell rather than fractions of a millimetre — but
carries no ambiguity and survives where the phase has decorrelated.

For a terrestrial radar the two are complementary in a specific way.  The
range sample is tens of centimetres and the azimuth cell is an arc, so
tracking will never see a millimetre of line-of-sight motion; what it can
see is the fast ice the phase cannot follow, and it can say whether a phase
rate of many metres a year is real motion or an unwrapping artefact.

:func:`patch_offsets` is the estimator: normalised cross-correlation of
intensity patches over a small search window, with the correlation peak
refined by a parabola in each axis.  It returns the shift that takes a
patch of ``b`` onto the matching patch of ``a``, so a positive range offset
means the target moved *away* from the radar between the two.

The usual cautions apply and are worth stating because they decide whether
an answer means anything: a patch needs texture, so a featureless snowfield
returns noise at a high correlation; the parabolic refinement biases
sub-pixel offsets toward integers ("peak locking"), so a distribution of
offsets piled at whole samples is an artefact of the estimator; and the
correlation peak's height is the only quality measure there is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["patch_offsets", "texture", "PatchOffsets"]


def texture(image, highpass=6.0, looks=(1, 1)):
    """Terrain pattern: high-passed dB intensity, with the speckle averaged.

    Correlating raw intensity matches **speckle**, which is a different
    random field in every acquisition and does not repeat over hours; what
    repeats is the ridge-and-shadow pattern of the ground.  Taking dB turns
    the multiplicative speckle additive, multilooking averages it down, and
    subtracting a Gaussian-smoothed copy removes the brightness trend that
    would otherwise dominate the correlation.  This is the same
    preprocessing :func:`gpri_tools.coregister.texture` uses to hold a
    campaign's heading, where it reaches correlations of 0.7.

    Parameters
    ----------
    image : 2-D array
        Complex SLC or real intensity.
    highpass : float
        Sigma of the Gaussian mean removed, in output cells.
    looks : (int, int)
        Intensity multilooking applied first.

    Returns
    -------
    2-D float array on the multilooked grid.
    """
    from scipy.ndimage import gaussian_filter

    p = np.abs(np.asarray(image)) ** 2 if np.iscomplexobj(image) \
        else np.asarray(image, float)
    la, lr = int(looks[0]), int(looks[1])
    if (la, lr) != (1, 1):
        na = p.shape[0] // la * la
        nr = p.shape[1] // lr * lr
        p = p[:na, :nr].reshape(na // la, la, nr // lr, lr).mean(axis=(1, 3))
    db = 10.0 * np.log10(np.maximum(p, 1e-12))
    return (db - gaussian_filter(db, float(highpass))).astype(np.float32)


@dataclass
class PatchOffsets:
    """Per-patch shifts from :func:`patch_offsets`, on the patch-centre grid."""

    azimuth: np.ndarray            # lines, b -> a
    range: np.ndarray              # samples, b -> a
    correlation: np.ndarray        # peak of the normalised correlation
    rows: np.ndarray               # patch centres, azimuth
    cols: np.ndarray               # patch centres, range
    patch: tuple
    search: tuple

    @property
    def shape(self):
        return self.azimuth.shape

    def valid(self, min_correlation=0.3):
        """Patches whose peak clears ``min_correlation`` and refined finitely."""
        return (np.isfinite(self.azimuth) & np.isfinite(self.range)
                & (self.correlation >= float(min_correlation)))

    def __repr__(self):
        v = self.valid()
        return (f"PatchOffsets(grid={self.shape}, patch={self.patch}, "
                f"search={self.search}, {100 * v.mean():.0f} % valid, "
                f"median correlation {np.nanmedian(self.correlation):.2f})")


def _parabolic(y0, y1, y2):
    """Sub-sample peak of three samples with the middle one largest."""
    curv = y0 - 2.0 * y1 + y2
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(curv < 0, 0.5 * (y0 - y2) / curv, 0.0)
    return np.clip(np.nan_to_num(d), -1.0, 1.0)


def patch_offsets(a, b, patch=(32, 128), step=None, search=(3, 6),
                  min_texture=1e-12):
    """Cross-correlate intensity patches of ``b`` against ``a``.

    Parameters
    ----------
    a, b : 2-D arrays
        Complex SLCs or real intensities, same grid.  ``a`` is the reference
        and ``b`` the repeat, so the returned shift is where ``b``'s pattern
        sits relative to ``a``'s.
    patch : (int, int)
        Patch size in (azimuth lines, range samples).
    step : (int, int), optional
        Patch spacing; the patch size by default, so patches tile without
        overlapping.
    search : (int, int)
        Integer shifts tried either way in each axis.
    min_texture : float
        A patch whose intensity variance is below this has no pattern to
        match and returns NaN rather than a spurious peak.

    Returns
    -------
    PatchOffsets
    """
    A = np.abs(np.asarray(a)) ** 2 if np.iscomplexobj(a) else np.asarray(a, float)
    B = np.abs(np.asarray(b)) ** 2 if np.iscomplexobj(b) else np.asarray(b, float)
    if A.shape != B.shape:
        raise ValueError(f"grids differ: {A.shape} against {B.shape}")
    pa, pr = int(patch[0]), int(patch[1])
    sa, sr = int(search[0]), int(search[1])
    if pa < 2 * sa + 2 or pr < 2 * sr + 2:
        raise ValueError("the patch must be larger than twice the search")
    st = (pa, pr) if step is None else (int(step[0]), int(step[1]))

    rows = np.arange(sa + pa // 2, A.shape[0] - sa - pa // 2, st[0])
    cols = np.arange(sr + pr // 2, A.shape[1] - sr - pr // 2, st[1])
    if rows.size == 0 or cols.size == 0:
        raise ValueError("the grid is too small for that patch and search")

    off_a = np.full((rows.size, cols.size), np.nan)
    off_r = np.full((rows.size, cols.size), np.nan)
    peak = np.full((rows.size, cols.size), np.nan)
    shifts_a = np.arange(-sa, sa + 1)
    shifts_r = np.arange(-sr, sr + 1)

    for i, r0 in enumerate(rows):
        for j, c0 in enumerate(cols):
            ref = A[r0 - pa // 2:r0 + pa // 2, c0 - pr // 2:c0 + pr // 2]
            ref = ref - ref.mean()
            nref = np.sqrt((ref ** 2).sum())
            if nref ** 2 < min_texture * ref.size:
                continue
            corr = np.full((shifts_a.size, shifts_r.size), -1.0)
            for m, da in enumerate(shifts_a):
                for n, dr in enumerate(shifts_r):
                    win = B[r0 - pa // 2 + da:r0 + pa // 2 + da,
                            c0 - pr // 2 + dr:c0 + pr // 2 + dr]
                    win = win - win.mean()
                    nwin = np.sqrt((win ** 2).sum())
                    if nwin ** 2 < min_texture * win.size:
                        continue
                    corr[m, n] = float((ref * win).sum() / (nref * nwin))
            m, n = np.unravel_index(int(np.argmax(corr)), corr.shape)
            peak[i, j] = corr[m, n]
            d_a = float(shifts_a[m])
            d_r = float(shifts_r[n])
            if 0 < m < shifts_a.size - 1:
                d_a += float(_parabolic(corr[m - 1, n], corr[m, n], corr[m + 1, n]))
            if 0 < n < shifts_r.size - 1:
                d_r += float(_parabolic(corr[m, n - 1], corr[m, n], corr[m, n + 1]))
            off_a[i, j] = d_a
            off_r[i, j] = d_r
    return PatchOffsets(azimuth=off_a, range=off_r, correlation=peak,
                        rows=rows, cols=cols, patch=(pa, pr), search=(sa, sr))
