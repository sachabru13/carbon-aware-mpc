"""Model predictive control of a flexible load and a battery, to cut CO2.

The system is a toy version of a data center:
  - an inflexible base load (base_kw)
  - deferrable work that arrives with a known daily profile and can wait in a queue `q`
    (the queue is capped, which stands in for deadlines), served at up to wmax_kw
  - a battery with a capacity, a power limit and a round-trip efficiency of eta^2
  - the grid: import >= 0, no export

Every hour I solve a linear program over the next 24 h that minimizes
sum(forecast intensity * grid import), apply only the first decision, then re-solve.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class Params:
    H: int = 24
    base_kw: float = 800.0
    flex_mean_kw: float = 400.0
    wmax_kw: float = 1000.0
    qmax_kwh: float = 2400.0  # about 6 hours of average deferrable work
    cap_kwh: float = 2000.0
    pmax_kw: float = 1000.0
    eta: float = 0.95  # charge and discharge efficiency, each way
    soc0: float = 0.5
    cycle_penalty: float = 0.1  # g/kWh of battery throughput, avoids pointless cycling


def arrivals(local_hours: np.ndarray, p: Params) -> np.ndarray:
    """Deferrable work arriving each hour (kW). Heavier in the afternoon, mean = flex_mean_kw."""
    return p.flex_mean_kw * (1 + 0.6 * np.sin(2 * np.pi * (local_hours - 9) / 24))


class Controller:
    def __init__(self, p: Params):
        self.p = p
        H = p.H
        # parameters change at every solve, so cvxpy can reuse the compiled problem
        self.I = cp.Parameter(H, nonneg=True)  # forecast intensity
        self.a = cp.Parameter(H, nonneg=True)  # work arrivals
        self.e0 = cp.Parameter(nonneg=True)  # battery energy now
        self.q0 = cp.Parameter(nonneg=True)  # queue now
        self.w = cp.Variable(H)  # work served
        self.c = cp.Variable(H)  # battery charge
        self.d = cp.Variable(H)  # battery discharge
        e = cp.Variable(H + 1)
        q = cp.Variable(H + 1)
        constraints = [
            e[0] == self.e0,
            q[0] == self.q0,
            e[1:] == e[:-1] + p.eta * self.c - self.d / p.eta,
            q[1:] == q[:-1] + self.a - self.w,
            e >= 0, e <= p.cap_kwh,
            q >= 0, q <= p.qmax_kwh,
            self.w >= 0, self.w <= p.wmax_kw,
            self.c >= 0, self.c <= p.pmax_kw,
            self.d >= 0, self.d <= p.pmax_kw,
            p.base_kw + self.w + self.c - self.d >= 0,
            # terminal conditions: do not end the horizon with an emptier battery or a longer queue
            e[H] >= self.e0,
            q[H] <= self.q0,
        ]
        objective = self.I @ (self.w + self.c - self.d) + p.cycle_penalty * cp.sum(self.c + self.d)
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def act(self, forecast: np.ndarray, a: np.ndarray, e0: float, q0: float):
        """Return (work, charge, discharge) for the current hour, or None if the solve failed."""
        p = self.p
        self.I.value = np.clip(forecast, 0, None)
        self.a.value = a
        self.e0.value = float(np.clip(e0, 0, p.cap_kwh))
        self.q0.value = float(np.clip(q0, 0, p.qmax_kwh))
        self.prob.solve(solver=cp.HIGHS)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        return (
            float(np.clip(self.w.value[0], 0, p.wmax_kw)),
            float(np.clip(self.c.value[0], 0, p.pmax_kw)),
            float(np.clip(self.d.value[0], 0, p.pmax_kw)),
        )


def simulate(local_hours: np.ndarray, steps: np.ndarray, forecasts: np.ndarray | None, p: Params):
    """Run the closed loop over `steps` (hourly indices).

    forecasts[i] holds the 24 forecast values for hours steps[i] .. steps[i]+23.
    forecasts=None is the reference case: work is served as it arrives, no battery.
    Returns the hourly log and the number of failed solves (those hours fall back to the reference action).
    """
    ctrl = Controller(p) if forecasts is not None else None
    e, q = p.soc0 * p.cap_kwh, 0.0
    log = {k: [] for k in ("grid", "w", "c", "d", "e", "q")}
    fails = 0
    for i, t in enumerate(steps):
        a_h = arrivals(local_hours[t : t + p.H], p)
        w, c, d = a_h[0], 0.0, 0.0
        if ctrl is not None:
            action = ctrl.act(forecasts[i], a_h, e, q)
            if action is None:
                fails += 1
            else:
                w, c, d = action
        grid = p.base_kw + w + c - d
        e += p.eta * c - d / p.eta
        q += a_h[0] - w
        e, q = float(np.clip(e, 0, p.cap_kwh)), float(np.clip(q, 0, p.qmax_kwh))
        for key, val in zip(log, (grid, w, c, d, e, q)):
            log[key].append(val)
    return {k: np.array(v) for k, v in log.items()}, fails


def carbon_kg(log: dict, ci_true: np.ndarray, steps: np.ndarray, p: Params) -> dict:
    """Emissions with the real intensity, in kg CO2.

    "adjusted" corrects for what is left at the end of the run, otherwise a controller could
    look good just by leaving work undone or by draining nothing: energy left in the battery
    was bought but not used (subtracted), work still in the queue was not paid yet (added),
    both valued at the mean intensity of the period.
    """
    ci = ci_true[steps]
    raw = float((ci * log["grid"]).sum() / 1000.0)
    mean_ci = float(ci.mean())
    stored = log["e"][-1] - p.soc0 * p.cap_kwh
    left = log["q"][-1]
    adjusted = raw - stored * mean_ci / 1000.0 + left * mean_ci / 1000.0
    return {"raw_kg": raw, "adjusted_kg": adjusted, "soc_gain_kwh": float(stored), "backlog_end_kwh": float(left)}
