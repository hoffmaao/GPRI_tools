#!/usr/bin/env python3
"""Persistent scatterers: where the phase survives when the field does not.

    python examples/baker_ps.py --scene 20170913 --hours 6

`gpri_tools.psinterp` has been in this package since the beginning and
nothing in the Baker analysis has used it.  The place it belongs is the one
[`baker.md`](../docs/baker.md) documents as the pipeline's weak spot: beyond
7 km, where 46 % of the ice sits against 308 of the 7,697 stable pixels, the
rock-fitted screens are extrapolating and the modelled stratification
residual there is 41 to 70 % of what it started as.  A sparse network of
bright, stable targets is exactly what a decorrelated far field leaves
behind.

The script selects persistent scatterers by amplitude dispersion over the
record (`gpri_tools.psinterp.amplitude_dispersion`, the stability of a
pixel's brightness), unwraps a long-baseline interferogram at those targets
along a minimum spanning tree, and interpolates the sparse answer back onto
the grid (`unwrap_with_ps`).  Coordinates are ground metres, because the
GPRI's pixel spacing is wildly anisotropic and a tree built in pixels would
prefer the wrong neighbours.

What comes out is reported three ways: how many PS there are and how they
are distributed in range, how the unwrapped phase compares with the chain's
answer over the same interval where both exist, and the suspect fraction —
the pixels where the interpolated field was off by more than half a fringe
and the unwrapping aliased.  That last number is the honest failure map, not
a quality score.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baker_aps import SCENES, load, open_stack, split_mask               # noqa: E402
from baker_movie import decimated_geom                                   # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.glaciers import (glacier_mask, load_outlines,            # noqa: E402
                                 stable_ground_mask)
from gpri_tools.heading import scene_heading                             # noqa: E402
from gpri_tools.psinterp import (amplitude_dispersion, ps_density,       # noqa: E402
                                 select_ps, unwrap_with_ps)
from gpri_tools.timeseries import los_displacement                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170913")
    ap.add_argument("--hours", type=float, default=6.0,
                    help="baseline of the interferogram unwrapped")
    ap.add_argument("--looks", type=int, nargs=2, default=(3, 15))
    ap.add_argument("--amplitude-epochs", type=int, default=40,
                    help="epochs the amplitude dispersion is measured over")
    ap.add_argument("--max-dispersion", type=float, default=0.25,
                    help="amplitude dispersion a pixel must be under to be a PS")
    ap.add_argument("--max-count", type=int, default=40000)
    ap.add_argument("--mask-lags", type=int, nargs="+", default=(1, 2, 3),
                    help="the pair cache the masks and weights are read from; "
                         "the default is the one the path delay already built")
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--stable-coherence", type=float, default=0.6)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    la, lr = args.looks

    # ---- amplitude dispersion over the record ------------------------------
    # open at native looks: asking for a multilooked pair stack would form
    # every pair from the SLCs, which this script does not need
    stack = open_stack(scene, "upper", lags=(1,), looks=(1, 1))
    n_epochs = stack.network.n_epochs
    times = np.asarray(stack.network.times, float)
    step = max(1, n_epochs // max(args.amplitude_epochs, 1))
    picked = list(range(0, n_epochs, step))[:args.amplitude_epochs]
    t0 = time.time()
    def look_mean(x):
        na_ = x.shape[0] // la * la
        nr_ = x.shape[1] // lr * lr
        return x[:na_, :nr_].reshape(na_ // la, la, nr_ // lr, lr).mean(axis=(1, 3))

    amps = np.stack([look_mean(np.abs(stack.read_slc(e))) for e in picked])
    disp = amplitude_dispersion(amps)
    print(f"{scene.name}: amplitude dispersion over {len(picked)} epochs in "
          f"{time.time() - t0:.0f} s; grid {disp.shape[0]} x {disp.shape[1]}, "
          f"median {np.nanmedian(disp):.3f}")

    # ---- the interferogram to unwrap ---------------------------------------
    want = min(args.hours, (times[-1] - times[0]) * 24.0)
    j = int(np.argmin(np.abs((times - times[0]) * 24.0 - want)))
    if j == 0:
        sys.exit("--hours does not fit inside this record")
    dt_h = float((times[j] - times[0]) * 24.0)
    a = stack.read_slc(0)
    b = stack.read_slc(j)
    na = min(a.shape[0], b.shape[0]) // la * la
    nr = min(a.shape[1], b.shape[1]) // lr * lr
    z = (a[:na, :nr] * np.conj(b[:na, :nr])).reshape(na // la, la, nr // lr, lr).sum(axis=(1, 3))
    del a, b
    wavelength = stack.wavelength
    stack.close()
    wrapped = np.angle(z)
    print(f"interferogram over {dt_h:.2f} h, epochs 0 and {j}")

    # ---- masks and ground coordinates --------------------------------------
    # the masks and weights come from a multilooked pair cache the scene
    # already has: forming a fresh one from the SLCs costs half an hour and
    # this script does not need the pairs themselves
    stack2, net, phase, cc, r, az, n = load(scene, 1, 0, antenna="upper",
                                            lags=tuple(args.mask_lags),
                                            looks=tuple(args.looks))
    pr = np.asarray(net.pairs[:n], int)
    mean_cc = cc[(pr[:, 1] - pr[:, 0]) <= 3].mean(axis=0)
    del cc, phase
    geom = decimated_geom(stack2, 1, heading)
    la_, lo_ = geom.geodetic(rows=[0, geom.shape[0] - 1],
                             cols=[0, geom.shape[1] - 1])
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                        bbox=(lo_.min() - .02, la_.min() - .02,
                              lo_.max() + .02, la_.max() + .02))
    stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                   threshold=args.stable_coherence)
    ice = (mean_cc >= args.ice_coherence) & glacier_mask(geom, gdf)
    # the interferogram is multilooked, so the geometry has to be sampled at
    # the centre of each look cell rather than sliced off the native grid
    shape = wrapped.shape
    gr_all = np.asarray(geom.ground_range(), float)
    sl_all = np.asarray(geom.slant_range(), float)
    bear_all = np.deg2rad(np.asarray(geom.bearings(), float))
    # `load` hands back a stack whose geometry is already the multilooked
    # one, so the grids line up sample for sample; only a size mismatch needs
    # the centres of the look cells
    if sl_all.size == shape[1]:
        col_c = np.arange(shape[1])
    else:
        col_c = np.clip(np.arange(shape[1]) * lr + lr // 2, 0, sl_all.size - 1)
    if bear_all.size == shape[0]:
        row_c = np.arange(shape[0])
    else:
        row_c = np.clip(np.arange(shape[0]) * la + la // 2, 0, bear_all.size - 1)
    gr = gr_all[col_c]
    bearing = bear_all[row_c]
    east = gr[None, :] * np.sin(bearing)[:, None]
    north = gr[None, :] * np.cos(bearing)[:, None]
    slant = np.broadcast_to(sl_all[col_c], shape)
    ice = ice[:shape[0], :shape[1]]
    stable = stable[:shape[0], :shape[1]]
    print(f"grid {shape[0]} x {shape[1]} over {slant.min() / 1000:.1f}-"
          f"{slant.max() / 1000:.1f} km; ice {int(ice.sum()):,} px, "
          f"bedrock {int(stable.sum()):,} px")
    stack2.close()

    ps = select_ps(dispersion=disp[:shape[0], :shape[1]],
                   max_dispersion=args.max_dispersion, max_count=args.max_count)
    print(f"\n{ps.sum():,} persistent scatterers "
          f"({100 * ps.mean():.2f} % of the grid, density "
          f"{ps_density(ps):.4f} per pixel); on ice {int((ps & ice).sum()):,}, "
          f"on bedrock {int((ps & stable).sum()):,}")
    print(f"{'range (km)':>11} {'pixels':>9} {'PS':>8} {'PS %':>7} "
          f"{'ice px':>9} {'PS on ice':>10}")
    edges = np.arange(np.floor(slant.min() / 1000), np.ceil(slant.max() / 1000) + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (slant >= lo * 1000) & (slant < hi * 1000)
        if band.sum() < 100:
            continue
        print(f"{lo:5.0f}-{hi:<5.0f} {band.sum():9,d} {int((ps & band).sum()):8,d} "
              f"{100 * (ps & band).sum() / band.sum():6.2f}% "
              f"{int((ice & band).sum()):9,d} {int((ps & ice & band).sum()):10,d}")

    coords = np.column_stack([north.ravel(), east.ravel()])
    t0 = time.time()
    res = unwrap_with_ps(wrapped, mask=ps, coords=coords,
                         weights=mean_cc[:shape[0], :shape[1]])
    print(f"\n{res} in {time.time() - t0:.0f} s")
    mm = los_displacement(res.unwrapped, wavelength) * 1000.0
    chain_mm = los_displacement(wrapped, wavelength) * 1000.0
    amb = wavelength / 2 * 1000.0
    for label, m in (("RGI ice", ice), ("bedrock", stable)):
        d_ = mm[m] - chain_mm[m]
        cycles = np.round(d_ / amb)
        print(f"{label:10s} unwrapped minus wrapped: "
              f"{100 * np.mean(np.abs(cycles) >= 1):5.1f} % of pixels moved by "
              f"a whole cycle or more, up to {np.max(np.abs(cycles)):.0f}; "
              f"suspect {100 * np.mean(res.suspect[m]):.2f} %")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=140)
    show = ice | stable
    im0 = axes[0].imshow(np.where(show, disp[:shape[0], :shape[1]], np.nan),
                         origin="lower", aspect="auto", cmap="viridis",
                         vmin=0, vmax=1.0, interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], label="Amplitude dispersion", fraction=0.046)
    axes[0].plot(*np.nonzero(ps)[::-1], ".", ms=0.4, color="tab:red", alpha=0.35)
    im1 = axes[1].imshow(np.where(show, wrapped, np.nan), origin="lower",
                         aspect="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi,
                         interpolation="nearest")
    fig.colorbar(im1, ax=axes[1], label="Wrapped phase (rad)", fraction=0.046)
    lim = float(np.nanpercentile(np.abs(mm[show]), 98)) or 1.0
    im2 = axes[2].imshow(np.where(show, mm, np.nan), origin="lower",
                         aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                         interpolation="nearest")
    fig.colorbar(im2, ax=axes[2], label="Unwrapped LOS (mm)", fraction=0.046)
    for ax in axes:
        ax.set_xlabel("Range (px)")
        ax.set_ylabel("Azimuth (px)")
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"37_ps_{scene.name}_{dt_h:.0f}h.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")

    root = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
    npz = root / f"ps_u_lk{la}x{lr}_{dt_h:.0f}h.npz"
    np.savez(npz, dispersion=disp[:shape[0], :shape[1]].astype(np.float32),
             ps=ps, wrapped=wrapped.astype(np.float32),
             unwrapped=res.unwrapped.astype(np.float32),
             suspect=res.suspect, ice=ice, stable=stable, hours=dt_h,
             n_ps=res.n_ps, n_unresolved=res.n_unresolved)
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
