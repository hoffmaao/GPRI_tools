"""Meteorology beside the radar: SNOTEL stations and ERA5 reanalysis.

The diurnal signal this package was written to find has one competing
explanation radar alone cannot dismiss: an atmosphere whose refractivity
follows the sun.  Ice that speeds up in the afternoon and air that warms in the
afternoon put the same sign of phase into a repeat-pass interferogram, and the
only way to tell them apart is to measure the air.

Two sources, neither needing credentials:

* **SNOTEL** (`fetch_snotel`), the USDA/NRCS network, through the AWDB REST
  API.  Hourly air temperature, precipitation, snow depth, snow water
  equivalent, solar radiation and wind, at stations that are often the only
  instrument for tens of kilometres.  Timestamps come in the station's own
  standard time — the API reports the offset as ``dataTimeZone`` — and are
  returned here in UTC, because everything else in this package is.
* **ERA5** (`fetch_era5`) through the Open-Meteo archive, which serves the
  reanalysis by latitude and longitude as JSON, hourly and already in UTC.
  Surface fields only; the pressure-level profile is not in that archive.

What the two are for is different.  A station is a point measurement with the
instrument's real weather in it; the reanalysis is a 31 km average that knows
nothing about the valley but is never missing.  Several stations at different
heights give the third thing, a **lapse rate** (`lapse_rate`), which is the
closest an ordinary network comes to measuring the stratification that bends a
radar beam.

Series arrive on their own clock and the radar's epochs fall between: see
:func:`interp_to`, and :mod:`gpri_tools.refractivity` for turning temperature,
humidity and pressure into the refractivity the phase actually responds to.
"""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/era5"

#: hourly SNOTEL elements worth having beside a radar campaign
SNOTEL_HOURLY = ("TOBS", "PREC", "SNWD", "WTEQ", "SRADV", "WSPDV", "WDIRV")

#: ERA5 surface fields: enough for refractivity, plus the sun and the wind
ERA5_HOURLY = ("temperature_2m", "relative_humidity_2m", "dew_point_2m",
               "surface_pressure", "pressure_msl", "wind_speed_10m",
               "wind_direction_10m", "precipitation", "cloud_cover",
               "shortwave_radiation")

# stored unit -> (SI-ish unit, conversion).  Anything not here is passed
# through with its unit recorded, so a new element cannot silently arrive in
# the wrong scale.
_UNITS = {
    "degF": ("degC", lambda v: (v - 32.0) * 5.0 / 9.0),
    "degC": ("degC", lambda v: v),
    "in": ("mm", lambda v: v * 25.4),
    "mph": ("m/s", lambda v: v * 0.44704),
    "W/m2": ("W/m2", lambda v: v),
    "degree": ("deg", lambda v: v),
    "pct": ("pct", lambda v: v),
}


def _get(url: str, cache: Path | None = None, timeout: float = 120.0):
    """GET JSON, through a cache file when one is named.

    Downloads are the slow, rate-limited and occasionally unavailable part of
    an analysis; a cached response makes a rerun free and makes the numbers
    reproducible when the service is down.
    """
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    with urllib.request.urlopen(url, timeout=timeout) as f:
        payload = f.read().decode()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(payload)
    return json.loads(payload)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    p = math.pi / 180.0
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def nearby_stations(lat, lon, radius_km=30.0, state=None, networks=("SNTL",),
                    active_only=False, cache_dir=None) -> list[dict]:
    """Network stations within ``radius_km``, nearest first.

    ``state`` narrows the query to one state code, which is the difference
    between a 3 MB answer and a 60 MB one; leave it None to search everywhere.
    Each station keeps its ``distance_km`` and its ``dataTimeZone``, the
    standard-time offset its timestamps are in.
    """
    q = {"networkCds": ",".join(networks),
         "activeOnly": "true" if active_only else "false"}
    if state:
        q["stateCds"] = state
    url = f"{AWDB}/stations?{urllib.parse.urlencode(q)}"
    cache = None if cache_dir is None else \
        Path(cache_dir) / f"stations_{state or 'all'}_{'-'.join(networks)}.json"
    out = []
    for s in _get(url, cache):
        if s.get("latitude") is None or s.get("longitude") is None:
            continue
        # the API honours stateCds but not networkCds, and answers with every
        # network in the state -- including the discontinued manual snow
        # courses, which have no hourly data at all
        if networks and s.get("networkCode") not in networks:
            continue
        d = haversine_km(lat, lon, s["latitude"], s["longitude"])
        if d <= radius_km:
            s = dict(s)
            s["distance_km"] = d
            s["elevation_m"] = None if s.get("elevation") is None \
                else s["elevation"] * 0.3048          # AWDB reports feet
            s.setdefault("dataTimeZone", None)
            out.append(s)
    out.sort(key=lambda s: s["distance_km"])
    return out


def fetch_snotel(triplet, begin, end, elements=SNOTEL_HOURLY,
                 utc_offset_hours=None, cache_dir=None) -> dict:
    """Hourly SNOTEL series for one station, in UTC and SI-ish units.

    ``begin`` and ``end`` are dates (``YYYY-MM-DD``) in the station's own
    standard time, which is how the API reads them; the window is padded by a
    day at each end so that the UTC conversion cannot clip the interval the
    caller asked for.  Returns ``{"time": datetime64[m] UTC, "<ELEMENT>":
    float array, "units": {...}, ...}``, with NaN wherever the station
    reported nothing.

    ``utc_offset_hours`` defaults to the station's own ``dataTimeZone``.
    """
    b = np.datetime64(begin, "D") - np.timedelta64(1, "D")
    e = np.datetime64(end, "D") + np.timedelta64(1, "D")
    if utc_offset_hours is None:
        meta = _get(f"{AWDB}/stations?stationTriplets={urllib.parse.quote(triplet)}",
                    None if cache_dir is None
                    else Path(cache_dir) / f"meta_{triplet.replace(':', '_')}.json")
        utc_offset_hours = float(meta[0].get("dataTimeZone", 0.0))
    q = {"stationTriplets": triplet, "elements": ",".join(elements),
         "duration": "HOURLY", "beginDate": str(b), "endDate": str(e),
         "periodRef": "END", "returnFlags": "false"}
    tag = f"snotel_{triplet.replace(':', '_')}_{b}_{e}.json"
    payload = _get(f"{AWDB}/data?{urllib.parse.urlencode(q)}",
                   None if cache_dir is None else Path(cache_dir) / tag)

    shift = np.timedelta64(int(round(-utc_offset_hours * 60)), "m")   # -> UTC
    series, units, stamps = {}, {}, []
    for station in payload:
        for block in station.get("data", []):
            el = block["stationElement"]["elementCode"]
            unit = block["stationElement"].get("storedUnitCode", "")
            when, what = [], []
            for v in block.get("values", []):
                # an hour the station reported empty is kept, as NaN: dropping
                # it would close the gap and let interpolation invent the
                # weather that was not measured
                t = np.datetime64(v["date"].replace(" ", "T"), "m") + shift
                stamps.append(t)
                if v.get("value") is None:
                    continue
                when.append(t)
                what.append(float(v["value"]))
            if not when:
                continue
            series[el] = (np.array(when), np.array(what))
            units[el] = unit

    # one clock for every element, so the caller gets a table and not a bag
    if not series:
        return {"time": np.array([], dtype="datetime64[m]"), "units": {}}
    grid = np.unique(np.array(stamps, dtype="datetime64[m]"))
    out = {"time": grid, "units": {}, "triplet": triplet,
           "utc_offset_hours": utc_offset_hours}
    for el, (t, y) in series.items():
        unit, conv = _UNITS.get(units[el], (units[el], lambda v: v))
        col = np.full(grid.shape, np.nan)
        col[np.searchsorted(grid, t)] = conv(y)          # grid contains every t
        out[el] = col
        out["units"][el] = unit
    return out


def fetch_era5(lat, lon, begin, end, variables=ERA5_HOURLY,
               cache_dir=None) -> dict:
    """Hourly ERA5 surface reanalysis at a point, in UTC.

    The archive is served by latitude and longitude and answers with the
    containing grid cell; ``elevation_m`` in the result is that cell's mean
    height, which is not the radar's and matters when a temperature is
    compared with a station's.
    """
    q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
         "start_date": str(begin), "end_date": str(end),
         "hourly": ",".join(variables), "models": "era5", "timezone": "GMT"}
    tag = f"era5_{lat:.3f}_{lon:.3f}_{begin}_{end}.json"
    d = _get(f"{OPEN_METEO}?{urllib.parse.urlencode(q)}",
             None if cache_dir is None else Path(cache_dir) / tag)
    if "hourly" not in d:
        raise RuntimeError(f"ERA5 archive returned no data: {d.get('reason', d)}")
    h = d["hourly"]
    out = {"time": np.array(h["time"], dtype="datetime64[m]"),
           "units": d.get("hourly_units", {}),
           "elevation_m": d.get("elevation"),
           "latitude": d.get("latitude"), "longitude": d.get("longitude")}
    for k, v in h.items():
        if k != "time":
            out[k] = np.array([np.nan if x is None else float(x) for x in v])
    return out


def interp_to(times, values, targets):
    """Linear interpolation of a met series onto radar epochs.

    Both clocks are ``datetime64``; the result is NaN outside the series and
    wherever the two bracketing samples are not both finite, so a gap in a
    station record stays a gap instead of being bridged silently.
    """
    t = np.asarray(times).astype("datetime64[s]").astype(np.int64)
    y = np.asarray(values, float)
    x = np.asarray(targets).astype("datetime64[s]").astype(np.int64)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full(x.shape, np.nan)
    out = np.interp(x, t[ok], y[ok], left=np.nan, right=np.nan)
    # the bracketing samples are looked up in the raw record, not in the finite
    # ones: an hour the station left empty is a gap, not a value to be spanned
    hi = np.searchsorted(t, x)
    lo = np.clip(hi - 1, 0, t.size - 1)
    hic = np.clip(hi, 0, t.size - 1)
    on = (t[hic] == x) & ok[hic]                 # the target lands on a sample
    between = (hi > 0) & (hi < t.size) & ok[lo] & ok[hic]
    return np.where(on | between, out, np.nan)


def lapse_rate(elevations_m, temperatures):
    """Least-squares lapse rate, °C/km, from stations at several heights.

    ``temperatures`` is ``(n_stations, n_times)`` on a common clock.  Returns
    ``(rate, sea_level_intercept)``; a time with fewer than two reporting
    stations gives NaN.  The sign convention is meteorological: a *negative*
    rate is the usual atmosphere, cooling with height, and a positive one is an
    inversion — cold air pooled in the valley under warm air, which is exactly
    the state that puts a diurnal ramp into radar phase without any ice moving.
    """
    z = np.asarray(elevations_m, float) / 1000.0
    T = np.atleast_2d(np.asarray(temperatures, float))
    rate = np.full(T.shape[1], np.nan)
    icept = np.full(T.shape[1], np.nan)
    for k in range(T.shape[1]):
        ok = np.isfinite(T[:, k])
        if ok.sum() >= 2:
            A = np.vstack([z[ok], np.ones(ok.sum())]).T
            rate[k], icept[k] = np.linalg.lstsq(A, T[ok, k], rcond=None)[0]
    return rate, icept
