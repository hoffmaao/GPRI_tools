#!/usr/bin/env python3
"""The surface's wetness, read from the radar's own brightness.

    python examples/baker_melt.py --scene 20170803_full
    python examples/baker_melt.py --campaigns 20170713_full 20170803_full 20170827 20180808 20190719

At Ku band the top centimetres of a snowpack set the backscatter: liquid
water makes them dark, snow that has drained or refrozen overnight is bright
by decibels.  So the brightness of every acquisition, referenced to the
bedrock in the same frame for the receiver's gain, is a reading of how wet
the surface is — the melt gauge the phase cannot give.  This script builds
it per pixel, hour by hour (:mod:`gpri_tools.melt`):

* **hourly brightness** — one frame per hour of the referenced backscatter,
  built by streaming the epochs (:class:`~gpri_tools.melt.BinAccumulator`),
  and the same for the lag-1 coherence, which drops when the surface is
  changing between acquisitions;
* **the pixel's day** — the diurnal sinusoid fitted to its hourly
  brightness (:func:`~gpri_tools.melt.diurnal_harmonic`): the peak-to-peak
  swing, with bedrock's as the noise floor, and the local hour it is darkest
  (wettest); the raw extremes (:func:`~gpri_tools.melt.diurnal_swing`) and
  the dark duty cycle (:func:`~gpri_tools.melt.wet_fraction`) of the band
  composites; and how well each pixel's brightness follows the air at its
  own height (:func:`~gpri_tools.melt.pixel_correlation`);
* **the transfer curve** — brightness against the air temperature at the
  pixel's height, the nearest SNOTEL station carried up a lapse rate
  (:func:`~gpri_tools.melt.air_temperature_at`,
  :func:`~gpri_tools.melt.transfer_curve`), per height band.

``--campaigns`` puts the campaigns side by side in one table: their
upper-glacier swing and trough hour, the diurnal displacement anomaly of
``baker_pixels.py``, and the warmth of the day — the test of whether the
anomaly follows the melt.

Everything per pixel is cached in ``work/<scene>/melt_u_dec16.npz``; the
tables are rebuilt from it.  The backscatter of every epoch is kept beside
it as it comes off the SLC, unreferenced, as ``db_u_dec16.npy`` (float16,
epochs x azimuth x range), and that is where the figures of the brightness
come from — the plain ones of ``baker_brightness.py`` and
``baker_catchments.py``, which never have to read the SLCs again; the
cache's ``db_reference`` is the bedrock median that was taken off
everything here.  The pair is stamped with ``MELT_CACHE_VERSION``, and
:func:`load_melt` refuses a cache written to an older definition or one whose
frames are missing, so no consumer reads a stale one.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import ChainMap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, load, split_mask                       # noqa: E402
from baker_north_side import decimated_par                            # noqa: E402
from baker_pixels import load_pixels, pixels_path                     # noqa: E402

from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry          # noqa: E402
from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask  # noqa: E402
from gpri_tools.heading import scene_heading, target_heights              # noqa: E402
from gpri_tools.melt import (BinAccumulator, air_temperature_at, bin_by_hour,  # noqa: E402
                             bin_mean, clock_composite, clock_median,
                             diurnal_harmonic, diurnal_swing, pixel_correlation,
                             transfer_curve, wet_fraction)

HEIGHT_STEP = 200.0                                                   # m
REFERENCE_HEIGHT = 2600.0                                             # m: the upper glacier

# version 1 is the definition every cache on disk was written to, so a file
# carrying no stamp is read as 1 rather than thrown away
MELT_CACHE_VERSION = 1


def melt_path(scene: Path, antenna: str, dec: int) -> Path:
    root = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
    return root / scene.name / f"melt_{antenna[0].lower()}_dec{dec}.npz"


def db_stack_path(scene: Path, antenna: str, dec: int) -> Path:
    """Every epoch's backscatter in dB, unreferenced, as ``compute`` streamed it."""
    return melt_path(scene, antenna, dec).with_name(f"db_{antenna[0].lower()}_dec{dec}.npy")


def load_melt(scene: Path, antenna: str, dec: int):
    """The cached melt products, or None if there are none to trust.

    A cache stamped below ``MELT_CACHE_VERSION`` holds quantities computed to a
    definition the code no longer uses, and one whose companion ``db_*.npy`` is
    gone cannot answer for the frames; either way there is nothing to read.
    """
    cache = melt_path(scene, antenna, dec)
    if not cache.exists() or not db_stack_path(scene, antenna, dec).exists():
        return None
    p = np.load(cache, allow_pickle=True)
    version = int(p["cache_version"]) if "cache_version" in p.files else 1
    return None if version < MELT_CACHE_VERSION else p


def on_clock(p, utc_offset):
    """The cached products with every local hour moved onto ``utc_offset``.

    The offset enters :func:`compute` as a pure additive shift of the local
    clock, so a cache built on one offset answers for any other without being
    rebuilt; the swings, amplitudes, correlations and transfer curves do not
    depend on it at all.  The shifted keys are laid over the cache rather than
    copied out of it, so the hourly frames are never read.
    """
    if "utc_offset" not in p:
        return p
    delta = float(utc_offset) - float(p["utc_offset"])
    if not delta:
        return p
    shifted = {"hours_local": p["hours_local"] + delta,
               "origin_local": np.mod(float(p["origin_local"]) + delta, 24.0),
               "utc_offset": float(utc_offset)}
    for k in p:
        if k.startswith("hour_"):
            shifted[k] = np.mod(p[k] + delta, 24.0)
    return ChainMap(shifted, p)


def station_series(name: str, station: str):
    """The station's temperature at the epochs and its height, from the met cache."""
    metf = Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "met" / f"met_{name}.npz"
    if not metf.exists():
        return None, np.nan
    m = np.load(metf, allow_pickle=True)
    key = f"snotel_{station}_TOBS_at_epochs"
    if key not in m.files or not np.isfinite(m[key]).any():
        return None, np.nan
    z_st = np.nan
    if "lapse_stations" in m.files:
        for label, z in zip(m["lapse_stations"], m["lapse_elevations_m"]):
            if str(label).split()[0] == station:
                z_st = float(z)
    return np.asarray(m[key], float), z_st


def compute(scene, name, args):
    """Hourly brightness and coherence per pixel, and what each pixel's day looks like."""
    stack, net, phase, cc, r, az, n = load(scene, args.decimate, 0, antenna=args.antenna)
    del phase
    if not hasattr(stack, "backscatter"):
        sys.exit("no SLCs behind this stack (GAMMA diff0 products): no backscatter")
    mean_cc = cc.mean(axis=0)
    geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                         heading=scene_heading(scene, default=BAKERBEND1_HEADING))
    la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1], cols=[0, geom.shape[1] - 1])
    bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
    gdf = load_outlines(os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"), bbox=bbox)
    stable, _ = stable_ground_mask(mean_cc, geom, gdf, threshold=args.stable_coherence)
    ice = (mean_cc >= args.ice_coherence) & glacier_mask(geom, gdf)
    fit_m, held_m = split_mask(stable)
    dem = os.environ.get("GPRI_DEM", "")
    if not Path(dem).exists():
        sys.exit("baker_melt.py needs GPRI_DEM to point at a DEM tile")
    z = target_heights(geom, dem)

    # the clock: hours since the first epoch, binned; and the same on the
    # local clock, hours past the midnight before the first epoch
    n_ep = net.n_epochs
    hours = net.times[:n_ep] * 24.0
    edges, idx = bin_by_hour(hours, args.width)
    hc = 0.5 * (edges[:-1] + edges[1:])
    e0 = net.epochs[0]
    origin_local = (e0.hour + e0.minute / 60.0 + args.utc_offset) % 24.0
    hours_local = origin_local + hc

    bands = {}
    bedges = [-np.inf] + list(args.bands) + [np.inf]
    for lo_, hi_ in zip(bedges[:-1], bedges[1:]):
        bands[f"ice_{lo_:.0f}_{hi_:.0f}"] = ice & (z >= lo_) & (z < hi_)
    bands["rock_held"], bands["rock_fit"] = held_m, fit_m

    # ---- stream the epochs: hourly brightness, gain-referenced to bedrock --
    # every frame is also kept on disk as it comes off the SLC (float16,
    # epochs x azimuth x range) for the movie and the glacier-mean series,
    # and the plain mean over the glacier is taken while the frame is in hand
    db = {k: np.full(n_ep, np.nan) for k in bands}
    db_ice = np.full(n_ep, np.nan)
    acc = BinAccumulator(hc.size, ice.shape)
    spath = db_stack_path(scene, args.antenna, args.decimate)
    spath.parent.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(spath, mode="w+", dtype=np.float16,
                                       shape=(n_ep,) + ice.shape)
    t0 = time.time()
    for e in range(n_ep):
        frame = stack.backscatter(e, looks=(1, args.decimate))[:, :ice.shape[1]]
        frame[frame < -200] = np.nan          # zero power: an empty line, not a surface
        frames[e] = frame
        db_ice[e] = np.nanmean(frame[ice])
        for k, m in bands.items():
            db[k][e] = np.nanmedian(frame[m])
        acc.add(idx[e], frame - db["rock_fit"][e])
        if e % 200 == 0:
            print(f"  backscatter {e + 1}/{n_ep} ({time.time() - t0:.0f} s)")
    frames.flush()
    del frames
    print(f"kept every frame in {spath}")
    hourly_db = acc.mean().astype(np.float32)
    epochs_per_bin = acc.epochs_per_bin.copy()
    del acc
    # the lag-1 coherence of each pair, in the hour of its first epoch
    acc = BinAccumulator(hc.size, ice.shape)
    for p in range(n):
        acc.add(idx[net.pairs[p, 0]], cc[p])
    hourly_cc = acc.mean().astype(np.float32)
    del acc, cc
    print(f"{hc.size} bins of {args.width:g} h, {np.median(epochs_per_bin):.0f} epochs "
          f"each; gain reference drifted {np.ptp(db['rock_fit']):.2f} dB")

    # ---- the air at every pixel's height --------------------------------
    T_ep, z_st = station_series(name, args.station)
    z_st = args.station_height if args.station_height is not None else z_st
    if T_ep is None or not np.isfinite(z_st):
        print(f"no station {args.station} temperature for {name}: run baker_met.py "
              f"(or pass --station-height); the transfer curves are skipped")
        T_h = np.full(hc.size, np.nan)
        T_ep = np.full(n_ep, np.nan)
    else:
        T_h = bin_mean(T_ep, idx, hc.size)

    # ---- each pixel's day -----------------------------------------------
    T_edges = np.arange(args.t_range[0], args.t_range[1] + 1e-9, args.t_step)
    out = {"hourly_db": hourly_db, "hourly_cc": hourly_cc, "hours": hc,
           "hours_local": hours_local, "epochs_per_bin": epochs_per_bin,
           "epoch_hours": hours, "db_ice": db_ice, "db_reference": db["rock_fit"],
           "T_epochs": T_ep, "T_station": T_h,
           "station": args.station, "station_height": z_st, "lapse": args.lapse,
           "origin_local": origin_local, "utc_offset": args.utc_offset,
           "epoch0": np.datetime64(e0).astype("datetime64[s]"),
           "z": z, "ice": ice, "fit": fit_m, "held": held_m, "cc": mean_cc,
           "r": np.asarray(r, float), "bands": np.asarray(args.bands, float),
           "T_edges": T_edges}
    for k, m in bands.items():
        out["db_" + k] = db[k] - db["rock_fit"]              # referenced band medians
        out["hourly_" + k] = bin_mean(out["db_" + k], idx, hc.size)
        out["cc_" + k] = np.nanmean(hourly_cc[:, m], axis=1) if m.any() else \
            np.full(hc.size, np.nan)
    out.update(pixel_days(out, args))
    return out


def pixel_days(p, args):
    """The per-pixel statistics of the hourly frames: rebuilt from any cache."""
    hourly_db, hourly_cc, hours_local = p["hourly_db"], p["hourly_cc"], p["hours_local"]
    with np.errstate(invalid="ignore"):
        anom = hourly_db - np.nanmean(hourly_db, axis=0)
        cc_anom = hourly_cc - np.nanmean(hourly_cc, axis=0)
    out = {}
    # the extremes, and the fitted sinusoid whose trough is the wet hour
    out["swing"], out["hour_min"], out["hour_max"] = diurnal_swing(
        anom, hours_local, min_bins=args.min_bins)
    amp, hour_bright, _ = diurnal_harmonic(anom, hours_local, min_bins=args.min_bins)
    out["amp"], out["hour_bright"] = 2 * amp, hour_bright
    out["hour_dark"] = np.mod(hour_bright + 12.0, 24.0)
    out["wet"] = wet_fraction(anom)
    out["swing_cc"], out["hour_min_cc"], out["hour_max_cc"] = diurnal_swing(
        cc_anom, hours_local, min_bins=args.min_bins)
    amp, hour_hi, _ = diurnal_harmonic(cc_anom, hours_local, min_bins=args.min_bins)
    out["amp_cc"], out["hour_dark_cc"] = 2 * amp, np.mod(hour_hi + 12.0, 24.0)
    # the air at every pixel's height, and the brightness against it
    T_z = air_temperature_at(p["z"], p["T_station"], float(p["station_height"]),
                             float(p["lapse"]))
    out["r_T"] = pixel_correlation(anom, T_z)
    ice, z, held = p["ice"], p["z"], p["held"]
    bedges = [-np.inf] + list(p["bands"]) + [np.inf]
    bands = {f"ice_{lo_:.0f}_{hi_:.0f}": ice & (z >= lo_) & (z < hi_)
             for lo_, hi_ in zip(bedges[:-1], bedges[1:])}
    bands["rock_held"] = held
    for k, m in bands.items():
        if m.any():
            mid, med, q1, q3, cnt = transfer_curve(T_z[:, m], anom[:, m], p["T_edges"],
                                                   min_count=args.min_count)
            out["curve_" + k] = np.vstack([mid, med, q1, q3, cnt])
    return out


def upper(p):
    """The top band's hourly brightness, bedrock gain removed."""
    return p["hourly_ice_%.0f_inf" % p["bands"][-1]]


def curve_slope(curve):
    """dB per °C over the transfer curve's medians, weighted by their counts."""
    mid, med, cnt = curve[0], curve[1], curve[4]
    if mid.size < 3:
        return np.nan, np.nan
    A = np.vstack([mid, np.ones_like(mid)]).T
    w = np.sqrt(cnt)
    slope = np.linalg.lstsq(A * w[:, None], med * w, rcond=None)[0][0]
    return slope, np.corrcoef(mid, med)[0, 1]


def report(name, p, args):
    hours_local, hc = p["hours_local"], p["hours"]
    keys = [k[3:] for k in p if k.startswith("db_ice_")] + ["rock_held"]
    ice, held, z = p["ice"], p["held"], p["z"]
    T_h = p["T_station"]
    z_st, lapse = float(p["station_height"]), float(p["lapse"])
    have_T = np.isfinite(T_h).any()
    print(f"\n{name}: {hc[-1] - hc[0] + args.width:.1f} h in {hc.size} bins of "
          f"{args.width:g} h; first epoch {p['epoch0']} UTC, local {p['origin_local']:.1f} h")
    if have_T:
        for zz in (REFERENCE_HEIGHT, 3000.0):
            T = air_temperature_at(zz, T_h, z_st, lapse)
            print(f"  air at {zz:.0f} m (station {p['station']} at {z_st:.0f} m, "
                  f"{lapse:+.1f} °C/km): {np.nanmin(T):.1f} to {np.nanmax(T):.1f} °C, "
                  f"{100 * np.mean(T[np.isfinite(T)] < 0):.0f} % of hours below 0, "
                  f"{np.nansum(np.maximum(T, 0)) * args.width:.0f} positive degree-hours")
    print("\n  hourly composites, bedrock gain removed: the swing, the local hours of "
          "the extremes,\n  the fraction of hours in the dark half, the correlation with "
          "the air at the band's height")
    print("  band              swing (dB)  darkest   brightest   wet  r(dB, T)  "
          "coherence swing  darkest")
    for k in keys:
        y = p["hourly_" + k]
        c = p["cc_" + k]
        comp = clock_composite(y, hours_local)
        ccomp = clock_composite(c, hours_local)
        fin, cfin = np.isfinite(comp), np.isfinite(ccomp)
        if not fin.any() or not cfin.any():
            continue
        m = (ice if k != "rock_held" else held)
        if k != "rock_held":
            lo_, hi_ = [float(v) for v in k.split("_")[1:]]
            m = m & (z >= lo_) & (z < hi_)
        zb = np.nanmedian(z[m]) if m.any() else np.nan
        rT = np.nan
        if have_T and np.isfinite(zb):
            T = air_temperature_at(zb, T_h, z_st, lapse)
            ok = np.isfinite(T) & np.isfinite(y)
            if ok.sum() > 3:
                rT = np.corrcoef(T[ok], y[ok])[0, 1]
        print(f"  {k:16s} {np.nanmax(y) - np.nanmin(y):9.2f}  "
              f"{np.flatnonzero(fin)[np.nanargmin(comp[fin])] + 0.5:6.1f}h  "
              f"{np.flatnonzero(fin)[np.nanargmax(comp[fin])] + 0.5:6.1f}h  "
              f"{float(wet_fraction(y[:, None])[0]):4.2f}  {rT:+8.2f}  "
              f"{np.nanmax(c) - np.nanmin(c):13.3f}  "
              f"{np.flatnonzero(cfin)[np.nanargmin(ccomp[cfin])] + 0.5:6.1f}h")

    print("\n  per pixel, medians: the fitted diurnal sinusoid's peak to peak (bedrock's is "
          "the noise floor),\n  the local hour of its trough, r(dB, T), and the same for "
          "the coherence")
    print("  band              swing (dB)  darkest  r(dB,T)  cc swing  darkest      n")
    for k in keys:
        m = held if k == "rock_held" else ice
        if k != "rock_held":
            lo_, hi_ = [float(v) for v in k.split("_")[1:]]
            m = m & (z >= lo_) & (z < hi_)
        m = m & np.isfinite(p["amp"])
        if not m.any():
            continue
        print(f"  {k:16s} {np.nanmedian(p['amp'][m]):9.2f}  "
              f"{clock_median(p['hour_dark'][m]):6.1f}h  {np.nanmedian(p['r_T'][m]):+7.2f}  "
              f"{np.nanmedian(p['amp_cc'][m]):8.3f}  "
              f"{clock_median(p['hour_dark_cc'][m]):6.1f}h  {m.sum():6d}")

    if have_T:
        print("\n  transfer curves, brightness against the air at the pixel's height:")
        print("  band              dB per °C   r     T range (°C)    n")
        for k in keys:
            if "curve_" + k not in p:
                continue
            curve = p["curve_" + k]
            if curve.shape[1] == 0:
                continue
            slope, rr = curve_slope(curve)
            print(f"  {k:16s} {slope:+9.3f} {rr:+6.2f}   {curve[0].min():5.1f} to "
                  f"{curve[0].max():5.1f}  {int(curve[4].sum()):7d}")


def campaigns(args):
    """The campaigns side by side in one table: swing, trough hour, and the anomaly."""
    rows = []
    for name in args.campaigns:
        scene = Path(SCENES.get(name, name))
        mf = melt_path(scene, args.antenna, args.decimate)
        pf = pixels_path(scene, args.antenna, args.decimate)
        p = load_melt(scene, args.antenna, args.decimate)
        if p is None:
            print(f"{name}: no usable melt cache ({mf}); run --scene {name} first")
            continue
        p = on_clock(p, args.utc_offset)
        T_h, z_st, lapse = p["T_station"], float(p["station_height"]), float(p["lapse"])
        T = air_temperature_at(REFERENCE_HEIGHT, T_h, z_st, lapse)
        ice, z = p["ice"], p["z"]
        up = ice & (z >= p["bands"][-1]) & np.isfinite(p["amp"])
        row = {"name": name, "span": p["hours"][-1] - p["hours"][0] + args.width,
               "T_mean": np.nanmean(T), "T_min": np.nanmin(T),
               "below0": 100 * np.mean(T[np.isfinite(T)] < 0) if np.isfinite(T).any() else np.nan,
               "pdh": np.nansum(np.maximum(T, 0)) * args.width,
               "swing_up": np.nanmedian(p["amp"][up]),
               "hmin_up": clock_median(p["hour_dark"][up]),
               "wet_up": float(wet_fraction(upper(p)[:, None])[0]),
               "rT_up": np.nanmedian(p["r_T"][up]),
               "cc_up": np.nanmedian(p["amp_cc"][up]),
               "rock": np.nanmedian(p["amp"][p["held"]]),
               "rms": np.nan, "r_db": np.nan}
        q = load_pixels(scene, args.antenna, args.decimate)
        if q is None and pf.exists():
            print(f"{name}: stale pixels cache ({pf}); rerun baker_pixels.py "
                  f"for its columns")
        if q is not None:
            row["rms"] = float(np.nanstd(q["template"]))
            # the anomaly against the upper glacier's brightness, epoch by epoch
            key = "db_ice_%.0f_inf" % p["bands"][-1]
            if key in q and q[key].size == q["template"].size:
                y = q[key] - q["db_rock_fit"]
                ok = np.isfinite(y) & np.isfinite(q["template"])
                row["r_db"] = np.corrcoef(y[ok], q["template"][ok])[0, 1]
        rows.append(row)
    if not rows:
        sys.exit("nothing to compare")

    print("\ncampaign         span  anomaly  swing>%.0f  darkest   wet  r(dB,T)  cc swing  "
          "r(dB,mm)  T%.0f mean  min  %%<0    PDH   rock" % (args.bands[-1], REFERENCE_HEIGHT))
    print("                  (h)     (mm)      (dB)  (local)                             "
          "     (°C)  (°C)          (°C h)   (dB)")
    for w in rows:
        print(f"{w['name']:15s} {w['span']:5.1f} {w['rms']:8.2f} {w['swing_up']:9.2f} "
              f"{w['hmin_up']:7.1f}h {w['wet_up']:5.2f} {w['rT_up']:+8.2f} {w['cc_up']:9.3f} "
              f"{w['r_db']:+9.2f} {w['T_mean']:9.1f} {w['T_min']:5.1f} {w['below0']:5.0f} "
              f"{w['pdh']:6.0f} {w['rock']:6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", help="one campaign: build its hourly brightness")
    ap.add_argument("--campaigns", nargs="+", help="campaigns to put side by side")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--bands", type=float, nargs="+", default=[1800, 2200, 2600],
                    help="height edges (m) of the ice bands")
    ap.add_argument("--width", type=float, default=1.0, help="bin width (hours)")
    ap.add_argument("--min-bins", type=int, default=12,
                    help="finite bins a 24 h window needs before a swing is read")
    ap.add_argument("--utc-offset", type=float, default=-7.0,
                    help="local clock relative to UTC (PDT: -7)")
    ap.add_argument("--station", default="1011", help="SNOTEL station for the air")
    ap.add_argument("--station-height", type=float, default=None,
                    help="its height (m); default from the met cache")
    ap.add_argument("--lapse", type=float, default=-6.5, help="°C/km, negative upward")
    ap.add_argument("--t-range", type=float, nargs=2, default=[-10, 25])
    ap.add_argument("--t-step", type=float, default=1.0)
    ap.add_argument("--min-count", type=int, default=200,
                    help="pixel-hours a transfer-curve bin needs")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    if args.campaigns:
        campaigns(args)
    if not args.scene:
        if not args.campaigns:
            ap.error("give --scene or --campaigns")
        return
    name = args.scene
    scene = Path(SCENES.get(name, name))
    cache = melt_path(scene, args.antenna, args.decimate)
    p = None if args.recompute else load_melt(scene, args.antenna, args.decimate)
    if p is not None:
        print(f"loaded {cache}")
        if "amp" not in p.files:
            out = dict(p)
            out.update(pixel_days(out, args))
            out["cache_version"] = MELT_CACHE_VERSION
            np.savez(cache, **out)
            p = np.load(cache, allow_pickle=True)
            print(f"rebuilt the per-pixel statistics in {cache}")
    else:
        spath = db_stack_path(scene, args.antenna, args.decimate)
        if cache.exists() and not args.recompute:
            if not spath.exists():
                print(f"{spath} is missing; rebuilding {cache}")
            else:
                print(f"{cache} is older than cache version "
                      f"{MELT_CACHE_VERSION}; rebuilding")
        out = compute(scene, name, args)
        out["cache_version"] = MELT_CACHE_VERSION
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
        print(f"wrote {cache}")
        p = np.load(cache, allow_pickle=True)
    report(name, on_clock(p, args.utc_offset), args)


if __name__ == "__main__":
    main()
