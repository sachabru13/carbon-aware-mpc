"""Checks on the data layer. The fake API response below copies the series names and the
structure I saw when querying the public_power endpoint for Switzerland (March 2025)."""
import numpy as np
import pandas as pd
import pytest

import data

FAKE_RESPONSE = {
    "unix_seconds": [1741561200, 1741564800, 1741568400],
    "production_types": [
        {"name": "Cross border electricity trading", "data": [2151.0, 1938.2, 2114.7]},
        {"name": "Nuclear", "data": [2947.2, 2948.9, 2949.5]},
        {"name": "Hydro Run-of-River", "data": [834.2, 865.0, 856.3]},
        {"name": "Hydro water reservoir", "data": [529.6, 448.6, 455.0]},
        {"name": "Hydro pumped storage", "data": [34.4, 35.9, 35.3]},
        {"name": "Others", "data": [425.1, 425.1, 425.1]},
        {"name": "Wind onshore", "data": [4.5, 6.0, 6.5]},
        {"name": "Solar", "data": [0.0, 0.0, 0.0]},
        {"name": "Load", "data": [7059.4, 6808.2, 6969.1]},
        {"name": "Residual load", "data": [7054.9, 6802.2, 6962.5]},
        {"name": "Renewable share of load", "data": [19.4, 19.4, 18.9]},
        {"name": "Renewable share of generation", "data": [28.9, 28.1, 28.1]},
    ],
}


def test_parse_maps_series_to_columns():
    mix = data.parse_response(FAKE_RESPONSE)
    assert set(mix.columns) == {"net_import", "nuclear", "hydro_ror", "hydro_res", "other", "wind", "solar", "load"}
    # pumped storage is added to reservoir hydro
    assert mix["hydro_res"].iloc[0] == pytest.approx(529.6 + 34.4)
    # shares and residual load must not leak in
    assert mix["load"].iloc[0] == pytest.approx(7059.4)
    assert mix.index.tz is not None and len(mix) == 3


def test_intensity_matches_hand_computation():
    mix = data.parse_response(FAKE_RESPONSE)
    ci = data.carbon_intensity(mix, ef_import=380.0)
    gen = {"nuclear": 2947.2, "hydro_ror": 834.2, "hydro_res": 564.0, "other": 425.1, "wind": 4.5, "solar": 0.0}
    emissions = sum(gen[k] * data.EF[k] for k in gen) + 2151.0 * 380.0
    expected = emissions / (sum(gen.values()) + 2151.0)
    assert ci.iloc[0] == pytest.approx(expected)
    assert 100 < ci.iloc[0] < 200


def test_exports_mean_domestic_mix_only():
    mix = data.parse_response(FAKE_RESPONSE)
    mix["net_import"] = -3000.0  # country is exporting
    ci = data.carbon_intensity(mix, ef_import=380.0)
    assert ci.max() < 100  # no import term, mostly nuclear and hydro


def test_missing_required_series_raises():
    bad = {"unix_seconds": [0, 3600], "production_types": [{"name": "Solar", "data": [1.0, 2.0]}]}
    with pytest.raises(ValueError):
        data.parse_response(bad)


def test_null_values_do_not_crash():
    js = {k: v for k, v in FAKE_RESPONSE.items()}
    js["production_types"] = [dict(s) for s in FAKE_RESPONSE["production_types"]]
    js["production_types"][1] = {"name": "Nuclear", "data": [None, 2948.9, 2949.5]}
    mix = data.parse_response(js)
    assert np.isnan(mix["nuclear"].iloc[0])  # not turned into 0 MW
    ci = data.carbon_intensity(mix)
    assert np.isnan(ci.iloc[0]) and ci.iloc[1:].notna().all()  # flagged, to be interpolated later


def test_synthetic_schema_matches_real_schema():
    real = data.parse_response(FAKE_RESPONSE)
    fake = data.synthetic_mix("2023-01-01", "2023-01-10")
    assert set(real.columns) == set(fake.columns)
    assert isinstance(fake.index, pd.DatetimeIndex) and fake.index.tz is not None
    ci = data.carbon_intensity(fake)
    assert ci.notna().all() and (ci > 0).all()


def test_fetch_real_chunks_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get_json(params, tries=3):
        calls.append((params["start"], params["end"]))
        start = pd.Timestamp(params["start"], tz="UTC")
        end = pd.Timestamp(params["end"], tz="UTC") + pd.Timedelta(hours=23)
        stamps = pd.date_range(start, end, freq="h")
        n = len(stamps)
        js = {"unix_seconds": [int(t.timestamp()) for t in stamps], "production_types": []}
        for s in FAKE_RESPONSE["production_types"]:
            js["production_types"].append({"name": s["name"], "data": [float(s["data"][0])] * n})
        return js

    monkeypatch.setattr(data, "_get_json", fake_get_json)
    mix = data.fetch_real("2024-01-01", "2024-07-18", cache_dir=str(tmp_path))  # 200 days
    assert len(calls) == 3  # 90 + 90 + 20 days
    assert len(mix) == 200 * 24
    assert mix.index.is_monotonic_increasing and not mix.index.duplicated().any()
    n_calls = len(calls)
    again = data.fetch_real("2024-01-01", "2024-07-18", cache_dir=str(tmp_path))
    assert len(calls) == n_calls  # second call served from the cache
    assert len(again) == len(mix)
