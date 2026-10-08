"""Checks on the controller using cases where I know what the answer should be."""
import numpy as np

from mpc import Params, arrivals, carbon_kg, simulate

p = Params()
N = 24 * 20
local_hours = np.arange(N + p.H) % 24
steps = np.arange(0, N)


def run(ci, with_oracle=True):
    F = np.array([ci[t : t + p.H] for t in steps])
    base_log, _ = simulate(local_hours, steps, None, p)
    mpc_log, fails = simulate(local_hours, steps, F, p)
    return (carbon_kg(base_log, ci, steps, p), carbon_kg(mpc_log, ci, steps, p), fails, mpc_log)


def test_constant_intensity_gives_no_gain_and_no_loss():
    ci = np.full(N + p.H, 100.0)
    base, mpc, fails, _ = run(ci)
    assert fails == 0
    assert mpc["adjusted_kg"] <= base["adjusted_kg"] * 1.001


def test_strong_daily_swing_gives_a_clear_gain():
    ci = 100 + 80 * np.sin(2 * np.pi * (np.arange(N + p.H) - 6) / 24)
    base, mpc, fails, _ = run(ci)
    gain = 1 - mpc["adjusted_kg"] / base["adjusted_kg"]
    assert fails == 0
    assert gain > 0.03


def test_work_is_conserved():
    ci = 100 + 80 * np.sin(2 * np.pi * (np.arange(N + p.H) - 6) / 24)
    _, _, _, log = run(ci)
    arrived = sum(arrivals(local_hours[t : t + 1], p)[0] for t in steps)
    assert abs(arrived - log["w"].sum() - log["q"][-1]) < 1e-3


def test_battery_energy_balance():
    ci = 100 + 80 * np.sin(2 * np.pi * (np.arange(N + p.H) - 6) / 24)
    _, _, _, log = run(ci)
    expected = p.soc0 * p.cap_kwh + (p.eta * log["c"] - log["d"] / p.eta).sum()
    assert abs(log["e"][-1] - expected) < 1e-3


def test_limits_are_respected():
    ci = 100 + 80 * np.sin(2 * np.pi * (np.arange(N + p.H) - 6) / 24)
    _, _, _, log = run(ci)
    assert log["grid"].min() >= -1e-6
    assert log["e"].min() >= -1e-6 and log["e"].max() <= p.cap_kwh + 1e-6
    assert log["q"].min() >= -1e-6 and log["q"].max() <= p.qmax_kwh + 1e-6
    assert log["w"].max() <= p.wmax_kw + 1e-6
    assert log["c"].max() <= p.pmax_kw + 1e-6 and log["d"].max() <= p.pmax_kw + 1e-6
