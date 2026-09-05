#!/usr/bin/env python3
"""Does the ice lag the weather?  Cross-correlation at every lag.

    python examples/baker_lag.py --scenes 20170803_full 20180808 20190719

The stratification model (`baker_stratification.py`) accounts for about half
the ice anomaly's response to the lapse rate on the warm campaigns and none of
its variation between campaigns.  The obvious candidate for the rest is melt.
The two explanations differ in a way that does not depend on any amplitude:

* **Refractivity is instantaneous.**  The delay a stratified atmosphere adds
  depends on the state of the air *now*.  Its correlation with temperature
  peaks at **zero lag**.
* **Melt is not.**  Water generated at the surface has to reach the bed before
  it can change basal water pressure and let the glacier slide, and that
  takes hours.  A melt-driven speed-up must peak *after* the forcing.

So the lag is the measurement.  This cross-correlates each campaign's
population series against the weather at lags from -12 to +12 hours and reports
where the correlation peaks.

One subtlety decides what to correlate.  Sliding is a **velocity**, but the
population series is a **displacement**: if velocity follows melt, displacement
follows the integral of melt and lags it by a further quarter cycle — six hours
for a diurnal signal — for reasons of calculus and not glaciology.  Both are
reported.  Held-out bedrock, which does not move, runs through the identical
machinery as the null.

The lag is measured as a **phase difference between 24 h harmonics**, not as
the peak of a cross-correlation.  On a signal that is itself diurnal the two
are not the same thing: shifting a daily cycle by twelve hours inverts it, so
a correlation searched over ±12 h always finds its largest magnitude at the
ends, and the answer says nothing.  A phase difference is defined once, and it
is defined only **modulo 24 hours** — which this cannot resolve and does not
pretend to.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES                                        # noqa: E402
from baker_population import population_path                       # noqa: E402

from gpri_tools.diurnal import fit_harmonics                        # noqa: E402


def velocity_anomaly(hours, displacement_mm, window_h=2.0):
    """Central difference of a displacement series, mm/hr.

    Differencing over a window rather than between neighbouring epochs is what
    makes the result readable: at a two-minute cadence the epoch-to-epoch
    difference of a millimetre-level series is almost pure noise.
    """
    t = np.asarray(hours, float)
    d = np.asarray(displacement_mm, float)
    out = np.full(t.shape, np.nan)
    for i, ti in enumerate(t):
        a = np.searchsorted(t, ti - window_h / 2)
        b = np.searchsorted(t, ti + window_h / 2) - 1
        if b > a and np.isfinite(d[a]) and np.isfinite(d[b]) and t[b] > t[a]:
            out[i] = (d[b] - d[a]) / (t[b] - t[a])
    return out


def harmonic_phase(hours, y, period_h=24.0):
    """``(amplitude, peak hour, explained variance)`` of a 24 h harmonic.

    Thin wrapper over :func:`gpri_tools.diurnal.fit_harmonics`, which carries
    an offset and a linear trend beside the harmonic — so a secular drift the
    population step did not remove cannot be absorbed into the phase.  NaNs are
    dropped rather than zero-filled, because a gap is not a measurement of
    zero.  The peak hour is on the same clock as ``hours``.
    """
    t = np.asarray(hours, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan, np.nan, np.nan
    fit = fit_harmonics(y[ok, None], t[ok] / 24.0,
                        periods=(period_h / 24.0,), degree=1)
    ev = fit.explained_variance()
    return (float(fit.amplitude(period_h / 24.0)[0]),
            float(fit.peak_time(period_h / 24.0)[0]),
            float(np.ravel(ev)[0]) if ev is not None else np.nan)


def wrap_lag(hours, period_h=24.0):
    """A lag folded into ``(-period/2, +period/2]`` — the only range it means."""
    return (float(hours) + period_h / 2) % period_h - period_h / 2


def lagged_correlation(hours, y, forcing, lags_h):
    """``r`` between ``y`` and ``forcing`` shifted by each lag, in hours.

    A *positive* lag means the forcing is shifted forward in time to match
    ``y``: that is, ``y`` responds **after** the forcing.
    """
    t = np.asarray(hours, float)
    y = np.asarray(y, float)
    f = np.asarray(forcing, float)
    out = np.full(len(lags_h), np.nan)
    for k, lag in enumerate(lags_h):
        shifted = np.interp(t - lag, t, f, left=np.nan, right=np.nan)
        ok = np.isfinite(y) & np.isfinite(shifted)
        if ok.sum() > 50 and np.std(y[ok]) > 0 and np.std(shifted[ok]) > 0:
            out[k] = np.corrcoef(y[ok], shifted[ok])[0, 1]
    return out


def peak(lags_h, r):
    """``(lag, r)`` where the correlation is most *positive*.

    Deliberately not the largest magnitude: on a diurnal pair that is always
    the ±12 h end, where the signal has simply been inverted.
    """
    if not np.isfinite(r).any():
        return np.nan, np.nan
    i = int(np.nanargmax(np.where(np.isfinite(r), r, -np.inf)))
    return float(lags_h[i]), float(r[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--max-lag", type=float, default=12.0)
    ap.add_argument("--lag-step", type=float, default=0.5)
    ap.add_argument("--window", type=float, default=2.0,
                    help="hours over which the velocity anomaly is differenced")
    ap.add_argument("--station", default="1011",
                    help="SNOTEL id for the forcing (default 1011, MF Nooksack)")
    args = ap.parse_args()
    work = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
    lags = np.arange(-args.max_lag, args.max_lag + 1e-9, args.lag_step)

    names = args.scenes or sorted(SCENES)
    print(f"lag of the ice against the weather, {args.antenna} antenna\n"
          f"a positive lag means the ice responds AFTER the forcing\n")
    for name in names:
        scene = Path(SCENES.get(name, name))
        pop = population_path(scene, args.antenna, args.decimate)
        metf = work / "met" / f"met_{name}.npz"
        if not pop.exists() or not metf.exists():
            continue
        z, m = np.load(pop), np.load(metf, allow_pickle=True)
        hours = z["hours"]
        span = float(hours[-1] - hours[0])
        if span < 12:
            print(f"{name}: {span:.1f} h of record, too short for a lag\n")
            continue

        series = {"ice displacement": z["ice"],
                  "ice velocity": velocity_anomaly(hours, z["ice"], args.window),
                  "rock displacement": z["rock"],
                  "rock velocity": velocity_anomaly(hours, z["rock"], args.window)}
        forcings = {}
        for key, label in ((f"snotel_{args.station}_TOBS_at_epochs", "air temperature"),
                           (f"snotel_{args.station}_SRADV_at_epochs", "solar radiation"),
                           ("lapse_rate_C_per_km", "lapse rate")):
            if key in m.files and np.isfinite(m[key]).sum() > 50:
                forcings[label] = m[key]
        if "air temperature" in forcings:
            # melt happens above freezing and not below: the rectified forcing
            forcings["melt (T above 0)"] = np.maximum(forcings["air temperature"], 0.0)
        if not forcings:
            print(f"{name}: no forcing series in {metf.name}\n")
            continue

        origin = float(m["campaign_start"].astype("datetime64[h]").astype(int) % 24)
        print(f"=== {name}  {span:.1f} h ({span / 24:.2f} cycles), "
              f"record opens at {origin:02.0f} UTC")

        print(f"\n{'24 h harmonic':22s}{'amplitude':>12s}{'peak (UTC)':>12s}{'r2':>7s}")
        phase = {}
        for label, y, unit in ([(k, v, "mm/hr" if "velocity" in k else "mm")
                                for k, v in series.items()]
                               + [(k, v, "") for k, v in forcings.items()]):
            amp, pk, r2 = harmonic_phase(hours, y)
            phase[label] = pk
            if np.isfinite(pk):
                print(f"{label:22s}{amp:12.2f} {unit:5s}"
                      f"{np.mod(pk + origin, 24):7.1f} h{r2:7.2f}")

        print(f"\n{'lag of':22s}{'behind':22s}{'phase lag':>11s}{'r at 0':>9s}"
              f"{'best +r lag':>13s}")
        for rname in ("ice velocity", "ice displacement", "rock displacement"):
            for fname in forcings:
                if rname not in phase or fname not in phase:
                    continue
                lag_h = wrap_lag(phase[rname] - phase[fname])
                r = lagged_correlation(hours, series[rname], forcings[fname], lags)
                r0 = r[int(np.argmin(np.abs(lags)))]
                bl, br = peak(lags, r)
                print(f"{rname:22s}{fname:22s}{lag_h:+9.1f} h{r0:+9.2f}"
                      f"{bl:+10.1f} h{br:+5.2f}")
        print()


if __name__ == "__main__":
    main()
