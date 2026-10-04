"""Speckle tracking: does a known shift come back?"""
import numpy as np
import pytest

from gpri_tools.tracking import PatchOffsets, patch_offsets


def _speckle(shape=(160, 640), seed=0):
    g = np.random.default_rng(seed)
    return g.exponential(1.0, shape).astype(float)


def test_an_integer_shift_comes_back_exactly():
    a = _speckle()
    # b[i, j] = a[i - 2, j + 4]: the pattern sits two lines later in azimuth
    # and four samples nearer in range, so the offsets are +2 and -4
    b = np.roll(a, (2, -4), axis=(0, 1))
    off = patch_offsets(a, b, patch=(64, 256), search=(3, 6))
    v = off.valid(0.5)
    assert v.any()
    assert np.allclose(off.azimuth[v], 2.0, atol=0.2)
    assert np.allclose(off.range[v], -4.0, atol=0.2)
    assert np.nanmedian(off.correlation) > 0.9


def test_no_shift_reads_zero():
    a = _speckle(seed=1)
    off = patch_offsets(a, a.copy(), patch=(64, 256), search=(3, 6))
    v = off.valid(0.5)
    assert np.allclose(off.azimuth[v], 0.0, atol=0.01)
    assert np.allclose(off.range[v], 0.0, atol=0.01)
    assert np.allclose(off.correlation[v], 1.0, atol=1e-6)


def test_a_sub_sample_shift_is_refined_off_the_integer():
    """A half-sample shift must land between the integers, not on one."""
    g = np.random.default_rng(2)
    from scipy.ndimage import gaussian_filter, shift as ndshift
    a = gaussian_filter(g.normal(0, 1, (160, 640)), 2.0)
    # b[j] = a[j - 0.5]: half a sample further from the radar
    b = ndshift(a, (0.0, 0.5), order=3, mode="nearest")
    off = patch_offsets(a, b, patch=(64, 256), search=(2, 4))
    v = off.valid(0.5)
    assert np.nanmedian(off.range[v]) == pytest.approx(0.5, abs=0.2)
    assert abs(np.nanmedian(off.range[v])) > 0.2        # not locked to zero


def test_a_featureless_patch_returns_nothing():
    a = np.ones((160, 640))
    off = patch_offsets(a, a.copy(), patch=(64, 256), search=(3, 6))
    assert np.all(np.isnan(off.azimuth))
    assert not off.valid().any()


def test_complex_input_is_read_as_intensity():
    a = _speckle(seed=3)
    z = np.sqrt(a) * np.exp(1j * np.random.default_rng(4).uniform(-np.pi, np.pi, a.shape))
    off_r = patch_offsets(a, np.roll(a, 3, axis=1), patch=(64, 256), search=(2, 5))
    off_c = patch_offsets(z, np.roll(z, 3, axis=1), patch=(64, 256), search=(2, 5))
    assert np.allclose(off_r.range[off_r.valid(0.5)], 3.0, atol=0.2)
    v = off_c.valid(0.5)
    assert np.allclose(off_c.range[v], off_r.range[off_r.valid(0.5)], atol=0.05)


def test_shapes_and_search_are_checked():
    a = _speckle((80, 200))
    with pytest.raises(ValueError, match="grids differ"):
        patch_offsets(a, a[:, :100])
    with pytest.raises(ValueError, match="larger than twice"):
        patch_offsets(a, a.copy(), patch=(8, 8), search=(8, 8))
    with pytest.raises(ValueError, match="too small"):
        patch_offsets(a, a.copy(), patch=(64, 190), search=(3, 6), step=(64, 190))


def test_repr_says_how_many_patches_are_usable():
    a = _speckle(seed=5)
    off = patch_offsets(a, np.roll(a, 1, axis=0), patch=(64, 256), search=(2, 4))
    assert isinstance(off, PatchOffsets)
    assert "valid" in repr(off) and "patch=" in repr(off)
