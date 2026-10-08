"""Carbon intensity forecasts for the next 1 to 24 hours.

Two models: "same hour yesterday" and a gradient boosting regressor.
A forecast issued at hour t only uses data up to t (included), and the model is trained
on targets that all come before the test period, so there is no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

H = 24
MIN_ORIGIN = 168  # need a week of history for the weekly lag


def _features(v: np.ndarray, roll24: np.ndarray, base: pd.Timestamp, t: np.ndarray, h: np.ndarray) -> pd.DataFrame:
    """One row per (origin t, horizon h). The target is v[t + h]."""
    T = t + h
    ts = base + pd.to_timedelta(T.ravel(), unit="h")
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy()
    return pd.DataFrame(
        {
            "h": h.ravel(),
            "y_t": v[t.ravel()],
            "mean24": roll24[t.ravel()],
            "trend24": v[t.ravel()] - v[t.ravel() - 24],
            "same_hour_d1": v[(T - 24).ravel()],
            "same_hour_d2": v[(T - 48).ravel()],
            "same_hour_w1": v[(T - 168).ravel()],
            "hour": hour,
            "dow": ts.dayofweek.to_numpy(),
            "sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25),
            "sin_h": np.sin(2 * np.pi * hour / 24),
            "cos_h": np.cos(2 * np.pi * hour / 24),
        }
    )


def _prep(y: pd.Series):
    v = y.to_numpy(float)
    roll24 = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    return v, roll24, y.index[0]


def train(y: pd.Series, split: int) -> HistGradientBoostingRegressor:
    """Fit on rows whose target index is strictly before `split` (the first test hour)."""
    v, roll24, base = _prep(y)
    t, h = np.broadcast_arrays(np.arange(MIN_ORIGIN, split - 1)[:, None], np.arange(1, H + 1)[None, :])
    keep = (t + h) < split
    t, h = t[keep], h[keep]
    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31, early_stopping=False, random_state=0
    )
    model.fit(_features(v, roll24, base, t, h), v[t + h])
    return model


def predict(model: HistGradientBoostingRegressor, y: pd.Series, origins: np.ndarray) -> np.ndarray:
    """Array of shape (len(origins), H): forecasts for t+1..t+H issued at each origin t."""
    v, roll24, base = _prep(y)
    t, h = np.broadcast_arrays(origins[:, None], np.arange(1, H + 1)[None, :])
    pred = model.predict(_features(v, roll24, base, t, h))
    return np.clip(pred, 0, None).reshape(len(origins), H)


def persistence(y: pd.Series, origins: np.ndarray) -> np.ndarray:
    """Baseline: the value at the same hour yesterday."""
    v = y.to_numpy(float)
    t, h = np.broadcast_arrays(origins[:, None], np.arange(1, H + 1)[None, :])
    return v[t + h - 24]


def truth(y: pd.Series, origins: np.ndarray) -> np.ndarray:
    v = y.to_numpy(float)
    t, h = np.broadcast_arrays(origins[:, None], np.arange(1, H + 1)[None, :])
    return v[t + h]


def scores(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae_by_horizon": np.abs(err).mean(axis=0).tolist(),
    }
