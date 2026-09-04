"""Meteorology helpers: units, clocks, interpolation and the lapse rate.

Nothing here touches the network: :func:`gpri_tools.met._get` is replaced with
a synthetic payload, which is the only part of the module that talks to a
service.  What is tested is what a wrong answer would quietly corrupt — a
temperature left in Fahrenheit, a station clock read as UTC, a gap bridged by
interpolation, an inversion given the wrong sign.
"""
import json

import numpy as np
import pytest

from gpri_tools import met


def _snotel_payload():
    """Two hourly elements at a station whose clock is UTC-8."""
    return [{
        "stationTriplet": "1011:WA:SNTL",
        "data": [
            {"stationElement": {"elementCode": "TOBS", "durationName": "HOURLY",
                                "storedUnitCode": "degF"},
             "values": [{"date": "2017-08-03 00:00", "value": 32.0},
                        {"date": "2017-08-03 01:00", "value": 212.0},
                        {"date": "2017-08-03 02:00", "value": None}]},
            {"stationElement": {"elementCode": "PREC", "durationName": "HOURLY",
                                "storedUnitCode": "in"},
             "values": [{"date": "2017-08-03 00:00", "value": 1.0},
                        {"date": "2017-08-03 01:00", "value": 2.0}]},
        ]}]


def test_fetch_snotel_converts_units_and_clock(monkeypatch):
    monkeypatch.setattr(met, "_get", lambda url, cache=None, timeout=120.0:
                        _snotel_payload())
    d = met.fetch_snotel("1011:WA:SNTL", "2017-08-03", "2017-08-03",
                         utc_offset_hours=-8.0)
    # degF -> degC, in -> mm
    assert d["units"]["TOBS"] == "degC" and d["units"]["PREC"] == "mm"
    assert np.allclose(d["TOBS"][:2], [0.0, 100.0])
    assert np.allclose(d["PREC"][:2], [25.4, 50.8])
    # midnight on a UTC-8 clock is 08:00 UTC, not midnight
    assert d["time"][0] == np.datetime64("2017-08-03T08:00")
    # a missing value stays missing rather than becoming zero
    assert np.isnan(d["TOBS"][2])
    # elements that report at different times share one clock
    assert len(d["time"]) == 3 and len(d["PREC"]) == 3


def test_fetch_snotel_reads_offset_from_the_station(monkeypatch):
    """With no offset given, the station's own dataTimeZone is used."""
    calls = []

    def fake(url, cache=None, timeout=120.0):
        calls.append(url)
        if "stations?" in url:
            return [{"dataTimeZone": -8.0}]
        return _snotel_payload()

    monkeypatch.setattr(met, "_get", fake)
    d = met.fetch_snotel("1011:WA:SNTL", "2017-08-03", "2017-08-03")
    assert d["utc_offset_hours"] == -8.0
    assert d["time"][0] == np.datetime64("2017-08-03T08:00")
    assert any("stations?" in u for u in calls)


def test_nearby_stations_filters_network_and_radius(monkeypatch):
    """The API answers with every network in the state; the filter is ours."""
    monkeypatch.setattr(met, "_get", lambda url, cache=None, timeout=120.0: [
        {"stationTriplet": "1011:WA:SNTL", "networkCode": "SNTL", "name": "near",
         "latitude": 48.8244, "longitude": -121.9295, "elevation": 4940.0,
         "dataTimeZone": -8.0},
        {"stationTriplet": "21A23:WA:SNOW", "networkCode": "SNOW", "name": "course",
         "latitude": 48.83, "longitude": -121.93, "elevation": 3780.0},
        {"stationTriplet": "999:WA:SNTL", "networkCode": "SNTL", "name": "far",
         "latitude": 47.0, "longitude": -120.0, "elevation": 3520.0},
    ])
    out = met.nearby_stations(48.8213, -121.9202, radius_km=20.0, state="WA")
    assert [s["name"] for s in out] == ["near"]          # SNOW dropped, far dropped
    assert out[0]["elevation_m"] == pytest.approx(4940.0 * 0.3048)
    assert out[0]["distance_km"] < 1.0


def test_interp_to_lands_on_epochs_and_refuses_gaps():
    t = np.array(["2017-08-03T00:00", "2017-08-03T01:00", "2017-08-03T06:00"],
                 dtype="datetime64[m]")
    y = np.array([0.0, 10.0, 60.0])
    x = np.array(["2017-08-03T00:30", "2017-08-03T03:00", "2017-08-02T23:00",
                  "2017-08-03T07:00"], dtype="datetime64[m]")
    out = met.interp_to(t, y, x)
    assert out[0] == pytest.approx(5.0)         # halfway between samples
    assert np.isnan(out[2]) and np.isnan(out[3])   # outside the record
    assert np.isfinite(out[1])                  # inside, across a 5 h gap


def test_lapse_rate_sign_and_missing_stations():
    z = [1000.0, 1500.0, 2000.0]
    # -5 C/km: 10, 7.5, 5 degC.  Second column has one station only.
    T = np.array([[10.0, 10.0], [7.5, np.nan], [5.0, np.nan]])
    rate, icept = met.lapse_rate(z, T)
    assert rate[0] == pytest.approx(-5.0)
    assert icept[0] == pytest.approx(15.0)      # extrapolated to sea level
    assert np.isnan(rate[1]) and np.isnan(icept[1])
    # an inversion is a positive rate, and that is the state that mimics ice
    rate2, _ = met.lapse_rate(z, np.array([[5.0], [7.5], [10.0]]))
    assert rate2[0] > 0


def test_get_uses_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "x.json"
    cache.write_text(json.dumps({"cached": True}))

    def explode(*a, **k):
        raise AssertionError("went to the network with a cache present")

    monkeypatch.setattr(met.urllib.request, "urlopen", explode)
    assert met._get("https://example.invalid/x", cache) == {"cached": True}
