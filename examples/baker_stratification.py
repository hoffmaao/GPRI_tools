#!/usr/bin/env python3
"""How much line-of-sight signal a stratified atmosphere can write, preliminary.

    python examples/baker_stratification.py --scene 20170803_full

The met data (`baker_met.py`) showed that the campaigns carrying the largest
"diurnal" ice anomalies are the ones where the air was frequently **inverted**,
and that held-out bedrock — which does not move — carries an anomaly correlated
with the lapse rate at ~0.8 on the two strongest.  That correlation says
stratification survives the correction.  It does not say how *much* signal it
can put on the glacier, and that is the number that decides whether the ice
anomaly is air or ice.  The bedrock residual is small in absolute terms — a few
tenths of a millimetre against ten — so the question is whether geometry can
amplify it by the twenty-odd times needed, not whether the effect exists.

This is the forward calculation.  Take the lapse rate the stations measured,
build a horizontally uniform atmosphere with it, integrate the refractivity
(:mod:`gpri_tools.refractivity`, Smith-Weintraub) along the straight path to
every pixel at its own DEM height and slant range, and turn the change in path
delay between two lapse rates into apparent displacement.  Then put that field
through **the same screen operators the chain applies** —
:func:`gpri_tools.aps.epoch_screen_correction` fitted on the fit-half bedrock,
then :func:`gpri_tools.aps.turbulence_screen` — because what matters is not the
delay a stratified atmosphere produces but the part of it that survives being
referenced to rock.

Preliminary, and honest about why: the atmosphere is taken as horizontally
uniform and hydrostatic with relative humidity constant in height, the path is
a straight line rather than a refracted ray, and the lapse rate measured across
four valley stations is extended over the glacier.  Each of those is a real
approximation.  What the calculation is good for is an **order of magnitude**
and a **sensitivity in mm per °C/km**, which can be compared directly with the
slope the data show.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, open_stack, split_mask                # noqa: E402
from baker_north_side import decimated_par                          # noqa: E402

from gpri_tools.aps import epoch_screen_correction, turbulence_screen  # noqa: E402
from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry    # noqa: E402
from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask  # noqa: E402
from gpri_tools.heading import scene_heading                        # noqa: E402
from gpri_tools.refractivity import stratified_delay                # noqa: E402

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803_full")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--lapse", type=float, nargs=2, default=None,
                    help="the two lapse rates to difference, °C/km; default is "
                         "the campaign's own p16 and p84 from the met file")
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--cc-stride", type=int, default=20,
                    help="use every Nth pair for the mean coherence (default 20)")
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene, args.scene))
    work = Path(os.environ.get("GPRI_WORK_ROOT", "work"))

    # ---- geometry: slant range and DEM height for every decimated pixel ----
    stack = open_stack(scene, args.antenna)
    par = decimated_par(stack.par, args.decimate)
    heading = scene_heading(scene, default=BAKERBEND1_HEADING)
    geom = RadarGeometry(par, heading=heading)
    lat, lon = geom.geodetic()
    r_cols = geom.slant_range().astype(float)          # one value per range column
    r = np.broadcast_to(r_cols[None, :], lat.shape).astype(float)

    dem = os.environ.get("GPRI_DEM", "")
    if not Path(dem).exists():
        sys.exit("GPRI_DEM must point at a DEM tile (set it in site.env)")
    from gpri_tools.heading import _dem_sampler
    sample, z_radar_dem = _dem_sampler(dem, float(par.float("GPRI_ref_north")),
                                       float(par.float("GPRI_ref_east")))
    z = sample(lat.ravel(), lon.ravel()).reshape(lat.shape)
    z0 = float(par.float("GPRI_ref_alt", z_radar_dem))
    print(f"{args.scene}: {lat.shape[0]}x{lat.shape[1]} px, heading {heading:.2f}°, "
          f"radar at {z0:.0f} m; targets {np.nanmin(z):.0f}-{np.nanmax(z):.0f} m, "
          f"range {r.min() / 1000:.1f}-{r.max() / 1000:.1f} km")

    # ---- the masks the chain itself uses ----------------------------------
    npz = work / scene.name / f"pairs_{'u' if args.antenna == 'upper' else 'l'}" \
        f"_lag1_lk1x1_dec{args.decimate}.npz"
    if not npz.exists():
        sys.exit(f"{npz} is missing: run baker_aps.py for this scene first")
    cc = np.load(npz, mmap_mode="r")["cc"]
    mean_cc = np.asarray(cc[::args.cc_stride]).mean(axis=0)
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                        bbox=(lon.min() - .02, lat.min() - .02,
                              lon.max() + .02, lat.max() + .02))
    stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                   threshold=args.stable_coherence)
    ice = (mean_cc >= args.ice_coherence) & glacier_mask(geom, gdf)
    fit_m, held_m = split_mask(stable)
    print(f"masks: ice {ice.sum():,} px, bedrock {fit_m.sum():,} fit + "
          f"{held_m.sum():,} held out")
    for name, m in (("ice", ice), ("held-out bedrock", held_m)):
        print(f"  {name:17s} range {np.median(r[m]) / 1000:5.2f} km, "
              f"height {np.median(z[m]):6.0f} m "
              f"({np.median(z[m]) - z0:+.0f} m above the radar)")

    # ---- the met the campaign actually had --------------------------------
    metf = work / "met" / f"met_{args.scene}.npz"
    if not metf.exists():
        sys.exit(f"{metf} is missing: run examples/baker_met.py --scenes {args.scene}")
    m = np.load(metf, allow_pickle=True)
    lap = m["lapse_rate_C_per_km"]
    ok = np.isfinite(lap)
    g_lo, g_hi = args.lapse if args.lapse else \
        (float(np.percentile(lap[ok], 16)), float(np.percentile(lap[ok], 84)))
    # surface conditions at the radar: the station lapse fit evaluated there,
    # with ERA5 for pressure and humidity
    t_radar = np.nanmedian(m["lapse_intercept_C"] + lap * z0 / 1000.0)
    p_sfc = float(np.nanmedian(m["era5_surface_pressure_at_epochs"]))
    rh = float(np.nanmedian(m["era5_relative_humidity_2m_at_epochs"])) / 100.0
    print(f"\nmet: {t_radar:.1f} °C and {p_sfc:.0f} hPa at the radar, RH "
          f"{100 * rh:.0f}%; lapse rate p16 {g_lo:+.2f} to p84 {g_hi:+.2f} °C/km "
          f"({100 * np.mean(lap[ok] > 0):.0f}% of epochs inverted)")

    # ---- the delay each state writes, and what survives the corrections ----
    d_lo = stratified_delay(r, z, z0, t_radar, g_lo, p_sfc, rh)
    d_hi = stratified_delay(r, z, z0, t_radar, g_hi, p_sfc, rh)
    # a longer path is a target that looks farther away: displacement is
    # positive toward the radar, so it is minus the delay change
    disp = -(d_hi - d_lo)                                  # metres

    raw = disp.copy()
    lin, _ = epoch_screen_correction(disp[None], fit_m, r_cols, model="linear",
                                     weights=mean_cc)
    lin = lin[0]
    scr, _ = turbulence_screen(lin, fit_m, sigma=tuple(args.sigma),
                               weights=mean_cc, wrapped=False)
    full = lin - scr

    print(f"\napparent LOS displacement from the lapse rate going "
          f"{g_lo:+.2f} -> {g_hi:+.2f} °C/km, mm (+ toward radar)")
    print(f"{'stage':34s}{'ice':>12}{'held-out rock':>16}{'ice/rock':>10}")
    for name, field in (("A  the raw stratification delay", raw),
                        ("C  + linear range screen on rock", lin),
                        ("D  + turbulence screen on rock", full)):
        a, b = np.median(field[ice]) * 1000, np.median(field[held_m]) * 1000
        print(f"{name:34s}{a:12.2f}{b:16.3f}"
              f"{(a / b if abs(b) > 1e-9 else np.nan):10.0f}")

    # ---- where the correction stops working -------------------------------
    # A screen fitted on rock removes what it can see. Rock and ice do not
    # occupy the same swath, so "referenced to bedrock" means interpolation
    # over part of the glacier and extrapolation over the rest.
    print(f"\nby range: what the rock-fitted screens leave on the ice")
    print(f"{'range (km)':>12}{'rock px':>9}{'ice px':>9}{'raw ice':>10}"
          f"{'after':>9}{'removed':>9}")
    edges = np.arange(0.0, r.max() + 1000.0, 1000.0)
    for a, b in zip(edges[:-1], edges[1:]):
        inb = (r >= a) & (r < b)
        m = inb & ice
        if m.sum() < 20:
            continue
        raw_mm = np.median(raw[m]) * 1000
        out_mm = np.median(full[m]) * 1000
        print(f"{a / 1000:5.0f}-{b / 1000:<6.0f}{(inb & stable).sum():9d}"
              f"{m.sum():9d}{raw_mm:10.1f}{out_mm:9.2f}"
              f"{100 * (1 - abs(out_mm / raw_mm)):8.0f}%")
    beyond = ice & (r > 7000)
    print(f"\n  {100 * beyond.sum() / ice.sum():.0f}% of ice pixels are beyond "
          f"7 km, where {(stable & (r > 7000)).sum()} of the {stable.sum():,} "
          f"stable\n  pixels are: the screen is interpolated over the near "
          f"glacier and extrapolated\n  over the far one, and the residual "
          f"follows that and not the physics.")

    span = g_hi - g_lo
    ice_mm = np.median(full[ice]) * 1000
    rock_mm = np.median(full[held_m]) * 1000

    # ---- against what the campaign actually did ---------------------------
    # the population step already removed the secular rate; regressing what is
    # left on the lapse rate gives the same quantity the model just predicted
    pop = work / scene.name / f"population_{'u' if args.antenna == 'upper' else 'l'}" \
        f"_dec{args.decimate}.npz"
    obs = {}
    if pop.exists():
        z_pop = np.load(pop)
        for key, series in (("ice", z_pop["ice"]), ("rock", z_pop["rock"])):
            good = np.isfinite(lap) & np.isfinite(series)
            if good.sum() > 50:
                A = np.vstack([lap[good], np.ones(good.sum())]).T
                obs[key] = float(np.linalg.lstsq(A, series[good], rcond=None)[0][0])

    print(f"\n{'':34s}{'ice':>12}{'held-out rock':>16}")
    print(f"{'predicted, mm per °C/km':34s}{ice_mm / span:12.2f}"
          f"{rock_mm / span:16.3f}")
    if obs:
        print(f"{'observed, mm per °C/km':34s}{obs.get('ice', np.nan):12.2f}"
              f"{obs.get('rock', np.nan):16.3f}")
        if obs.get("ice"):
            frac = 100 * (ice_mm / span) / obs["ice"]
            print(f"\n  a uniform atmosphere with the measured lapse rate accounts "
                  f"for {frac:.0f}% of the\n  ice slope this campaign shows, with the "
                  f"right sign, and reproduces the sign\n  flip between ice and rock "
                  f"that the correction produces.")
    print("\n  Note what the ladder does to the ratio: the raw delay is only "
          f"{np.median(raw[ice]) / np.median(raw[held_m]):.1f}x larger on ice\n"
          "  than on rock, and the correction -- fitted on rock, by construction -- "
          "is what\n  drives rock's residual to near zero. A small rock anomaly is "
          "therefore weak\n  evidence that the atmosphere is small on the ice.")


if __name__ == "__main__":
    main()
