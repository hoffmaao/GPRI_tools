#!/usr/bin/env python3
"""Weather beside the radar, for every campaign and a week either side.

    python examples/baker_met.py                      # download and cache
    python examples/baker_met.py --scenes 20190719    # one campaign

The diurnal result this project keeps arriving at — a population-level
night-time trough that repeats across August days and years, with no per-pixel
detection under it — has two readings, ice and air, and radar cannot separate
them.  This script fetches what the air was doing.

Two sources (:mod:`gpri_tools.met`): the SNOTEL stations around the site, of
which **MF Nooksack sits 0.8 km from the BakerBend1 tripod**, and ERA5 surface
reanalysis at the tripod's own coordinates.  Each campaign's window is padded
by a week at each end, because a diurnal cycle only means something against the
weather it sat in: a week of context says whether a campaign caught a settled
ridge or the second day of a frontal passage.

The four stations span 930 to 1506 m, which is the useful part.  A single
thermometer gives the temperature; four at different heights give the **lapse
rate**, and an inversion — warm air over cold — is the atmospheric state that
writes a diurnal ramp into radar phase with no ice moving at all.

Everything lands under ``$GPRI_WORK_ROOT/met/`` as JSON (the raw API answers,
cached so a rerun costs nothing) plus one ``met_<scene>.npz`` per campaign,
holding both the context window and the series interpolated onto that
campaign's radar epochs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES                                        # noqa: E402
from baker_population import population_path                       # noqa: E402

from gpri_tools import met                                          # noqa: E402
from gpri_tools.gamma import ParFile                                # noqa: E402

# BakerBend1, from the scene parameter files; the tripod moved a few metres
# between campaigns, which is nothing at a reanalysis cell's resolution
BAKERBEND1 = (48.8213, -121.9202)


def radar_position(scene: Path, default=BAKERBEND1):
    """The instrument's own latitude and longitude, out of an SLC parameter file.

    Every ``.slc.par`` carries ``GPRI_ref_north`` / ``GPRI_ref_east`` — the GPS
    fix written at acquisition — so the campaign says where it was rather than
    the script assuming it.
    """
    for par in sorted(Path(scene).glob("slc/*u.slc.par"))[:1]:
        p = ParFile.load(par)
        lat, lon = p.float("GPRI_ref_north"), p.float("GPRI_ref_east")
        if lat and lon:
            return float(lat), float(lon)
    return default


def campaign_window(scene: Path, antenna="upper", dec=16, pad_days=7):
    """``(first epoch, last epoch, window start, window end)`` for a scene."""
    npz = population_path(scene, antenna, dec)
    if not npz.exists():
        return None
    z = np.load(npz)
    t0 = z["epoch0"].astype("datetime64[s]")
    t1 = t0 + np.timedelta64(int(round(float(z["hours"][-1]) * 3600)), "s")
    day = np.timedelta64(pad_days, "D")
    return t0, t1, (t0.astype("datetime64[D]") - day), (t1.astype("datetime64[D]") + day)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="default: every scene with a population file")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--pad-days", type=int, default=7,
                    help="context either side of the campaign (default 7)")
    ap.add_argument("--radius-km", type=float, default=20.0,
                    help="how far to look for stations (default 20)")
    ap.add_argument("--lat", type=float, default=None,
                    help="default: the instrument position in the scene's par")
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default $GPRI_WORK_ROOT/met")
    args = ap.parse_args()

    import os
    out = args.outdir or Path(os.environ.get("GPRI_WORK_ROOT", "work")) / "met"
    cache = out / "raw"
    out.mkdir(parents=True, exist_ok=True)

    # ---- which campaigns, and over what windows ----------------------------
    names = args.scenes or sorted(SCENES)
    windows, seen = {}, set()
    for name in names:
        w = campaign_window(Path(SCENES.get(name, name)), args.antenna,
                            args.decimate, args.pad_days)
        if w is None:
            print(f"{name}: no population file, skipped")
            continue
        key = (str(w[0]), str(w[1]))
        if key in seen:                     # 20170803 and 20170803_full are one day
            print(f"{name}: same record as a scene already queued; its met file "
                  f"is written from the cache, at no further cost")
        seen.add(key)
        windows[name] = w
    if not windows:
        sys.exit("no campaign to fetch; run baker_population.py first")

    # ---- the stations ------------------------------------------------------
    lat, lon = (args.lat, args.lon) if args.lat is not None else \
        radar_position(Path(SCENES.get(names[0], names[0])))
    stations = met.nearby_stations(lat, lon, args.radius_km,
                                   state="WA", networks=("SNTL",),
                                   cache_dir=cache)
    print(f"\n{len(stations)} SNOTEL stations within {args.radius_km:g} km of "
          f"{lat:.4f}, {lon:.4f}")
    for s in stations:
        print(f"  {s['stationTriplet']:14s} {s['name']:<14s} "
              f"{s['distance_km']:5.1f} km  {s['elevation_m']:6.0f} m  "
              f"UTC{s['dataTimeZone']:+.0f}"
              if s["dataTimeZone"] is not None else "tz from API")
    if not stations:
        sys.exit("no stations found; widen --radius-km")

    # ---- fetch, campaign by campaign --------------------------------------
    for name, (t0, t1, w0, w1) in windows.items():
        print(f"\n=== {name}  {str(t0)[:16]} .. {str(t1)[:16]} UTC "
              f"(window {w0} .. {w1})")
        z = np.load(population_path(Path(SCENES.get(name, name)),
                                    args.antenna, args.decimate))
        epochs = (z["epoch0"].astype("datetime64[s]")
                  + (z["hours"] * 3600).astype("timedelta64[s]"))

        saved = {"epochs": epochs, "campaign_start": t0, "campaign_end": t1,
                 "window_start": np.datetime64(w0), "window_end": np.datetime64(w1)}
        temps, elevs, labels = [], [], []
        for s in stations:
            trip = s["stationTriplet"]
            try:
                d = met.fetch_snotel(trip, str(w0), str(w1),
                                     utc_offset_hours=s["dataTimeZone"],
                                     cache_dir=cache)
            except Exception as e:                    # one station down is not fatal
                print(f"  {trip}: {type(e).__name__}: {e}")
                continue
            key = trip.split(":")[0]
            have = [e for e in met.SNOTEL_HOURLY if e in d]
            n_camp = int(((d["time"] >= t0) & (d["time"] <= t1)).sum())
            print(f"  {trip:14s} {len(d['time']):5d} hourly samples, "
                  f"{n_camp:3d} in the campaign: {' '.join(have)}")
            saved[f"snotel_{key}_time"] = d["time"]
            for e in have:
                saved[f"snotel_{key}_{e}"] = d[e]
                saved[f"snotel_{key}_{e}_at_epochs"] = met.interp_to(
                    d["time"], d[e], epochs)
            if "TOBS" in d and s["elevation_m"] is not None:
                temps.append(met.interp_to(d["time"], d["TOBS"], epochs))
                elevs.append(s["elevation_m"])
                labels.append(f"{key} {s['name']}")

        era = met.fetch_era5(lat, lon, str(w0), str(w1), cache_dir=cache)
        print(f"  ERA5           {len(era['time']):5d} hourly samples, cell at "
              f"{era['elevation_m']:.0f} m ({era['latitude']:.3f}, "
              f"{era['longitude']:.3f})")
        saved["era5_time"] = era["time"]
        saved["era5_elevation_m"] = era["elevation_m"]
        for k in met.ERA5_HOURLY:
            if k in era:
                saved[f"era5_{k}"] = era[k]
                saved[f"era5_{k}_at_epochs"] = met.interp_to(era["time"],
                                                            era[k], epochs)

        # ---- the lapse rate, on the radar's own clock ----------------------
        if len(temps) >= 2:
            rate, icept = met.lapse_rate(elevs, np.vstack(temps))
            saved["lapse_rate_C_per_km"] = rate
            saved["lapse_intercept_C"] = icept
            saved["lapse_stations"] = np.array(labels)
            saved["lapse_elevations_m"] = np.array(elevs, float)
            ok = np.isfinite(rate)
            if ok.any():
                inv = 100.0 * np.mean(rate[ok] > 0)
                print(f"  lapse rate over {len(elevs)} stations "
                      f"({min(elevs):.0f}-{max(elevs):.0f} m): median "
                      f"{np.nanmedian(rate):+.2f} °C/km, "
                      f"p16-p84 {np.nanpercentile(rate[ok], 16):+.2f} to "
                      f"{np.nanpercentile(rate[ok], 84):+.2f}; "
                      f"inverted in {inv:.0f}% of the campaign's epochs")

        f = out / f"met_{name}.npz"
        np.savez(f, **saved)
        print(f"  wrote {f}")

    print(f"\nraw API responses cached under {cache}")


if __name__ == "__main__":
    main()
