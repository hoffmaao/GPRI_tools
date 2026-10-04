#!/usr/bin/env python3
"""Phase linking on a mini-stack, and the quality metric it brings with it.

    python examples/baker_phaselink.py --scene 20170913 --epochs 30

`gpri_tools.covariance` and `gpri_tools.phaselink` have been in this package
since early on and no Baker script has used them.  Phase linking replaces the
daisy chain's "difference consecutive epochs and integrate" with an estimate
that uses **every** pair at once: the N x N sample coherence matrix at each
pixel, factored for the per-epoch phases that best explain all of it (EMI,
EVD or the MLE sweep).

Two things come out, and the second is the reason to run it even if the first
disappoints.  The linked series itself is scored here against the chain on
held-out bedrock, the same way every other estimator in this repository is
scored.  And `temporal_coherence` — how well the rank-one model reproduces
every observed pair at a pixel — is a per-pixel quality number that nothing
else in the pipeline produces: the chain cannot tell a pixel whose steps are
consistent from one whose steps merely integrate.

The cost is the reason this runs on a mini-stack rather than a campaign: one
N x N matrix per pixel is ``N^2 * 16`` bytes, which at 723 epochs is 8.4 MB
*per pixel*.  `coherence_from_slcs` multilooks onto a coarse grid and refuses
to allocate past ``--max-gib``; ``--epochs`` sets how many acquisitions of the
record are linked, spread evenly across it.
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
from gpri_tools.covariance import coherence_from_slcs                    # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING                        # noqa: E402
from gpri_tools.glaciers import (glacier_mask, load_outlines,            # noqa: E402
                                 stable_ground_mask)
from gpri_tools.heading import scene_heading                             # noqa: E402
from gpri_tools.phaselink import phase_link, temporal_coherence          # noqa: E402
from gpri_tools.timeseries import los_displacement                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170913")
    ap.add_argument("--epochs", type=int, default=30,
                    help="acquisitions linked")
    ap.add_argument("--spread", default="consecutive",
                    choices=("consecutive", "even"),
                    help="'consecutive' links adjacent acquisitions, which is "
                         "what a mini-stack means and where the coherence is; "
                         "'even' spreads them over the record, which measures "
                         "how badly the rank-one model fits once the pairs "
                         "have decorrelated")
    ap.add_argument("--start", type=int, default=0,
                    help="first acquisition of a consecutive mini-stack")
    ap.add_argument("--looks", type=int, nargs=2, default=(3, 15),
                    help="multilooking of the coherence matrices")
    ap.add_argument("--method", default="emi", choices=("emi", "evd", "mle"))
    ap.add_argument("--max-gib", type=float, default=8.0)
    ap.add_argument("--mask-lags", type=int, nargs="+", default=(1, 2, 3))
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--stable-coherence", type=float, default=0.6)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    scene = Path(SCENES.get(args.scene, args.scene))
    if not scene.exists():
        sys.exit(f"no such scene {scene}; site.env knows {sorted(SCENES)}")
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    la, lr = args.looks

    stack = open_stack(scene, "upper", lags=(1,), looks=(1, 1))
    n_epochs = stack.network.n_epochs
    times = np.asarray(stack.network.times, float)
    if args.spread == "consecutive":
        step = 1
        picked = list(range(args.start, min(args.start + args.epochs, n_epochs)))
    else:
        step = max(1, n_epochs // max(args.epochs, 1))
        picked = list(range(0, n_epochs, step))[:args.epochs]
    t = times[picked]
    span = (t[-1] - t[0]) * 24.0
    print(f"{scene.name}: linking {len(picked)} of {n_epochs} acquisitions "
          f"(every {step}, {span:.1f} h, {(t[1] - t[0]) * 24 * 60:.0f} min apart)")

    t0 = time.time()
    slcs = np.stack([stack.read_slc(e) for e in picked])
    wavelength = stack.wavelength
    stack.close()
    print(f"  read {slcs.nbytes / 2**30:.1f} GiB of SLC in {time.time() - t0:.0f} s")

    t0 = time.time()
    Gamma = coherence_from_slcs(slcs, looks=tuple(args.looks),
                                max_gib=args.max_gib)
    del slcs
    print(f"  {Gamma.shape[0]} x {Gamma.shape[1]} coherence matrices of "
          f"{Gamma.shape[-1]} epochs in {time.time() - t0:.0f} s")

    t0 = time.time()
    theta = phase_link(Gamma, method=args.method)
    tcoh = temporal_coherence(Gamma, theta)
    print(f"  {args.method.upper()} linked in {time.time() - t0:.0f} s")
    del Gamma

    # the linked series, in the chain's convention: consecutive differences
    psi = np.angle(theta[..., :-1] * np.conj(theta[..., 1:]))
    linked = np.concatenate([np.zeros(psi.shape[:-1] + (1,)),
                             np.cumsum(los_displacement(psi, wavelength) * 1000.0,
                                       axis=-1)], axis=-1)

    # ---- masks, and the chain over the same acquisitions -------------------
    stack2, net, phase, cc, r, az, n = load(scene, 1, 0, antenna="upper",
                                            lags=tuple(args.mask_lags),
                                            looks=tuple(args.looks))
    pr = np.asarray(net.pairs[:n], int)
    mean_cc = cc[(pr[:, 1] - pr[:, 0]) <= 3].mean(axis=0)
    geom = decimated_geom(stack2, 1, heading)
    la_, lo_ = geom.geodetic(rows=[0, geom.shape[0] - 1],
                             cols=[0, geom.shape[1] - 1])
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                        bbox=(lo_.min() - .02, la_.min() - .02,
                              lo_.max() + .02, la_.max() + .02))
    stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                   threshold=args.stable_coherence)
    ice = (mean_cc >= args.ice_coherence) & glacier_mask(geom, gdf)
    fit_m, held_m = split_mask(stable)
    # the coherence grid is the SLC multilooked by hand and the cache is the
    # pair stack's own: they can differ by a cell at the edge, so crop both
    shape = (min(tcoh.shape[0], ice.shape[0]), min(tcoh.shape[1], ice.shape[1]))
    tcoh = tcoh[:shape[0], :shape[1]]
    linked = linked[:shape[0], :shape[1]]
    ice = ice[:shape[0], :shape[1]]
    held_m = held_m[:shape[0], :shape[1]]
    stable = stable[:shape[0], :shape[1]]

    # the chain's answer at the same acquisitions: integrate lag-1 pairs and
    # sample the epochs that were linked
    one = (pr[:, 1] - pr[:, 0]) == 1
    obs = los_displacement(phase[one], stack2.wavelength) * 1000.0
    chain = np.concatenate([np.zeros((1,) + obs.shape[1:]),
                            np.cumsum(obs, axis=0)])[picked]
    chain = np.moveaxis(chain, 0, -1)[:shape[0], :shape[1]]
    assert chain.shape[:2] == tcoh.shape == ice.shape
    del phase, cc, obs
    stack2.close()

    print(f"\n{'population':18s} {'pixels':>8s} {'temporal coherence':>20s} "
          f"{'linked sd':>11s} {'chain sd':>10s} {'linked - chain':>15s}")
    rows = {}
    for label, m in (("RGI ice", ice), ("held-out bedrock", held_m)):
        if not m.any():
            continue
        tc = tcoh[m]
        l_ = linked[m] - linked[m].mean(axis=-1, keepdims=True)
        c_ = chain[m] - chain[m].mean(axis=-1, keepdims=True)
        d_ = l_ - c_
        rows[label] = (np.nanmedian(tc), np.nanstd(l_), np.nanstd(c_), np.nanstd(d_))
        print(f"{label:18s} {m.sum():8,d} {np.nanmedian(tc):20.3f} "
              f"{np.nanstd(l_):8.2f} mm {np.nanstd(c_):7.2f} mm "
              f"{np.nanstd(d_):12.2f} mm")
    print(f"\ntemporal coherence is the per-pixel quality the chain cannot "
          f"give: {100 * np.mean(tcoh[ice] >= 0.6):.1f} % of ice and "
          f"{100 * np.mean(tcoh[held_m] >= 0.6):.1f} % of held-out bedrock "
          f"reach 0.6")

    # ---- figure ------------------------------------------------------------
    show = ice | stable
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=140)
    im0 = axes[0].imshow(np.where(show, tcoh, np.nan), origin="lower",
                         aspect="auto", cmap="magma", vmin=0, vmax=1,
                         interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], label="Temporal coherence", fraction=0.046)
    last = linked[..., -1] - chain[..., -1]
    lim = float(np.nanpercentile(np.abs(last[show]), 98)) or 1.0
    im1 = axes[1].imshow(np.where(show, last, np.nan), origin="lower",
                         aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                         interpolation="nearest")
    fig.colorbar(im1, ax=axes[1], label="Linked - chain (mm)", fraction=0.046)
    for ax in axes[:2]:
        ax.set_xlabel("Range (px)")
        ax.set_ylabel("Azimuth (px)")
    hours = (t - t[0]) * 24.0
    for label, m, colour in (("ice", ice, "tab:blue"),
                             ("held-out rock", held_m, "tab:red")):
        if m.any():
            axes[2].plot(hours, np.nanmedian(linked[m], axis=0), "-", lw=1.6,
                         color=colour, label=f"{label}, linked")
            axes[2].plot(hours, np.nanmedian(chain[m], axis=0), "--", lw=1.2,
                         color=colour, label=f"{label}, chain")
    axes[2].set_xlabel("Time (hr)")
    axes[2].set_ylabel("LOS (mm)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}{len(picked)}" + ("" if args.spread == "consecutive" else "_even")
    out = args.outdir / f"38_phaselink_{scene.name}_{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")

    root = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / scene.name
    npz = root / f"phaselink_u_lk{la}x{lr}_{tag}.npz"
    np.savez(npz, temporal_coherence=tcoh.astype(np.float32),
             linked=linked.astype(np.float32), chain=chain.astype(np.float32),
             hours=hours, ice=ice, held=held_m, epochs=np.array(picked))
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
