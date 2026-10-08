"""Swiss generation mix (Energy-Charts) and an approximate carbon intensity.

Two sources, same output format:
  fetch_real()      public Energy-Charts API, needs internet
  synthetic_mix()   made-up but plausible data, used to test the pipeline offline

Output: DataFrame indexed by hourly UTC timestamps, columns in MW.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# Emission factors in gCO2eq/kWh, life-cycle orders of magnitude (IPCC AR5 medians).
# "other" is the catch-all series of Energy-Charts for Switzerland (waste, biomass, gas CHP...),
# so 300 is a rough guess, not a measured value.
EF = {
    "nuclear": 12.0,
    "hydro_ror": 24.0,
    "hydro_res": 24.0,
    "solar": 45.0,
    "wind": 11.0,
    "other": 300.0,
}
# Carbon intensity of imported power, assumed constant. This is the weakest assumption of
# the whole project: see the README.
DEFAULT_EF_IMPORT = 380.0

# Energy-Charts series name (lowercase substring) -> column. First match wins.
# "Cross border electricity trading" is positive for net imports and negative for net exports.
NAME_MAP = [
    ("residual", None),
    ("share", None),
    ("cross border", "net_import"),
    ("run-of-river", "hydro_ror"),
    ("water reservoir", "hydro_res"),
    ("pumped storage", "hydro_res"),
    ("nuclear", "nuclear"),
    ("solar", "solar"),
    ("wind", "wind"),
    ("others", "other"),
]
SINGLE_SERIES = {"load", "net_import"}  # not summed when several names map to them

API = "https://api.energy-charts.info/public_power"


def _column_for(name: str) -> str | None:
    n = name.lower().strip()
    if n == "load":
        return "load"
    for sub, col in NAME_MAP:
        if sub in n:
            return col
    return None


def parse_response(js: dict) -> pd.DataFrame:
    """Turn one API response into an hourly DataFrame with our column names."""
    idx = pd.to_datetime(js["unix_seconds"], unit="s", utc=True)
    cols: dict[str, np.ndarray] = {}
    ignored = []
    for series in js["production_types"]:
        col = _column_for(series["name"])
        if col is None:
            ignored.append(series["name"])
            continue
        vals = np.array([np.nan if x is None else x for x in series["data"]], dtype=float)
        # NaN stays NaN: a missing hour must not turn into 0 MW of generation
        cols[col] = vals if (col in SINGLE_SERIES or col not in cols) else cols[col] + vals
    missing = {"load", "nuclear"} - set(cols)
    if missing:
        raise ValueError(f"expected series not found in the API response: {missing}")
    return pd.DataFrame(cols, index=idx).resample("1h").mean()


def _get_json(params: dict, tries: int = 3) -> dict:
    import requests

    for attempt in range(tries):
        try:
            r = requests.get(API, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_real(start: str, end: str, cache_dir: str = "data_cache") -> pd.DataFrame:
    """Download Swiss production by source, 90 days per request, cached as CSV."""
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    out = cache / f"ch_mix_{start}_{end}.csv"
    if out.exists():
        return pd.read_csv(out, index_col=0, parse_dates=True)

    frames = []
    cursor, stop = pd.Timestamp(start), pd.Timestamp(end)
    while cursor <= stop:
        chunk_end = min(cursor + pd.Timedelta(days=89), stop)
        js = _get_json({"country": "ch", "start": cursor.date().isoformat(), "end": chunk_end.date().isoformat()})
        frames.append(parse_response(js))
        cursor = chunk_end + pd.Timedelta(days=1)

    mix = pd.concat(frames)
    mix = mix[~mix.index.duplicated(keep="last")].sort_index()
    mix.to_csv(out)
    return mix


def synthetic_mix(start: str = "2023-01-01", end: str = "2024-12-31", seed: int = 0) -> pd.DataFrame:
    """Fake Swiss-like mix. The shapes are plausible (solar at noon, winter peak load,
    reservoir hydro following the residual load); the levels are not calibrated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", tz="UTC")
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    dow = idx.dayofweek.to_numpy()
    season = np.cos(2 * np.pi * (doy - 15) / 365)  # +1 in winter, -1 in summer

    def ar1(phi: float, sd: float) -> np.ndarray:
        x = np.zeros(n)
        e = rng.normal(0, sd, n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    load = (
        7000
        + 900 * season
        + 700 * np.exp(-(((hour - 11) / 3.5) ** 2))
        + 600 * np.exp(-(((hour - 19) / 2.5) ** 2))
        - 500 * np.exp(-(((hour - 3) / 3.0) ** 2))
        - 600 * (dow >= 5)
        + ar1(0.9, 80)
    )

    day_len = 12 - 3.5 * season
    sunrise = 12 - day_len / 2
    shape = np.clip(np.sin(np.pi * (hour - sunrise) / day_len), 0, None)
    amplitude = 0.55 - 0.35 * season
    cloud = 1 / (1 + np.exp(-(ar1(0.97, 0.25) * 2.0)))
    solar = 6500 * amplitude * shape * (1 - 0.75 * cloud)

    wind = np.clip(100 + 150 * ar1(0.98, 0.12), 0, None)
    nuclear = 2900 - 900 * ((doy > 150) & (doy < 210)) + rng.normal(0, 15, n)
    hydro_ror = np.clip(2200 - 1000 * season + ar1(0.99, 15), 800, None)
    other = np.full(n, 850.0)

    residual = load - (nuclear + hydro_ror + solar + wind + other)
    reservoir_cap = 3500 - 1000 * (season < 0)
    hydro_res = np.clip(0.7 * residual, 0, reservoir_cap)
    net_import = residual - hydro_res  # whatever is left is imported (negative = export)

    return pd.DataFrame(
        {
            "nuclear": nuclear,
            "hydro_ror": hydro_ror,
            "hydro_res": hydro_res,
            "solar": solar,
            "wind": wind,
            "other": other,
            "load": load,
            "net_import": net_import,
        },
        index=idx,
    )


def carbon_intensity(mix: pd.DataFrame, ef_import: float = DEFAULT_EF_IMPORT) -> pd.Series:
    """Approximate intensity of what is consumed in Switzerland, in gCO2/kWh.

    Domestic generation at its own emission factor plus net imports at a fixed factor.
    When the country exports, imports are zero and this is just the domestic mix.
    Hours where any input is missing give NaN (they are interpolated later, not set to zero).
    """
    gen_cols = [c for c in EF if c in mix.columns]
    gen = mix[gen_cols].clip(lower=0.0)  # pumping is consumption, not generation
    total_gen = gen.sum(axis=1, skipna=False)
    if "net_import" in mix.columns:
        imports = mix["net_import"].clip(lower=0.0)
    else:
        imports = (mix["load"] - total_gen).clip(lower=0.0)
    emissions = sum(gen[c] * EF[c] for c in gen_cols) + imports * ef_import
    ci = emissions / (total_gen + imports)
    ci.name = "ci_g_per_kwh"
    return ci


def load_intensity(source: str, start: str, end: str, seed: int = 0, ef_import: float = DEFAULT_EF_IMPORT) -> pd.Series:
    mix = fetch_real(start, end) if source == "real" else synthetic_mix(start, end, seed)
    ci = carbon_intensity(mix, ef_import)
    full = pd.date_range(ci.index[0], ci.index[-1], freq="h", tz="UTC")
    ci = ci.reindex(full)
    n_missing = int(ci.isna().sum())
    if n_missing:
        print(f"[data] {n_missing} missing hours filled by interpolation")
        ci = ci.interpolate(limit_direction="both")
    return ci
