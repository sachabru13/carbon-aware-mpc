"""Day-ahead carbon intensity forecast, then MPC of a flexible load and a battery.

  python run_demo.py --source synthetic
  python run_demo.py --source real --start 2024-01-01 --end 2026-09-30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import forecast as fc
from data import DEFAULT_EF_IMPORT, load_intensity
from mpc import Params, carbon_kg, simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--import-ef", type=float, default=DEFAULT_EF_IMPORT,
                    help="assumed gCO2/kWh of imported power")
    ap.add_argument("--out", default=None, help="default: results/<source>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "real":
        start, end = args.start or "2024-01-01", args.end or "2026-09-30"
    else:
        start, end = args.start or "2023-01-01", args.end or "2024-12-31"
    out = Path(args.out or f"results/{args.source}")
    out.mkdir(parents=True, exist_ok=True)
    p = Params()

    # 1) data
    y = load_intensity(args.source, start, end, args.seed, args.import_ef)
    n = len(y)
    first_test = n - args.test_days * 24
    steps = np.arange(first_test, n - p.H + 1)  # control steps
    origins = steps - 1  # forecasts are issued just before each step
    print(f"[data] {n} hours ({y.index[0].date()} to {y.index[-1].date()}), "
          f"mean intensity {y.mean():.0f} g/kWh, std {y.std():.0f}, test period {len(steps)} h")

    # 2) forecasts
    t0 = time.time()
    model = fc.train(y, first_test)
    F_ml = fc.predict(model, y, origins)
    F_pers = fc.persistence(y, origins)
    F_true = fc.truth(y, origins)
    print(f"[forecast] training and prediction took {time.time() - t0:.0f} s")
    sc = {"same hour yesterday": fc.scores(F_pers, F_true), "gradient boosting": fc.scores(F_ml, F_true)}
    for name, s in sc.items():
        print(f"  {name:20s} MAE {s['mae']:.1f}  RMSE {s['rmse']:.1f} g/kWh")
    gain = 1 - sc["gradient boosting"]["mae"] / sc["same hour yesterday"]["mae"]
    print(f"  MAE reduction vs yesterday: {100 * gain:.1f} %")

    # 3) closed-loop control
    v = y.to_numpy(float)
    local_hours = np.asarray(y.index.tz_convert("Europe/Zurich").hour)
    strategies = {
        "no control": None,
        "MPC, yesterday": F_pers,
        "MPC, gradient boosting": F_ml,
        "MPC, perfect forecast": F_true,
    }
    results, logs = {}, {}
    for name, F in strategies.items():
        t0 = time.time()
        log, fails = simulate(local_hours, steps, F, p)
        res = carbon_kg(log, v, steps, p)
        res["solve_failures"] = fails
        results[name], logs[name] = res, log
        print(f"[mpc] {name:24s} {res['adjusted_kg']:10.1f} kg CO2 (raw {res['raw_kg']:.1f}), "
              f"failed solves {fails}, {time.time() - t0:.0f} s")
    ref = results["no control"]["adjusted_kg"]
    for r in results.values():
        r["reduction_pct"] = 100 * (1 - r["adjusted_kg"] / ref)
    print("\nReduction vs no control (adjusted):")
    for name, r in results.items():
        print(f"  {name:24s} {r['reduction_pct']:6.2f} %")

    # 4) physical consistency of every run
    for name, lg in logs.items():
        assert lg["grid"].min() >= -1e-6, f"negative grid import ({name})"
        assert lg["e"].min() >= -1e-6 and lg["e"].max() <= p.cap_kwh + 1e-6, f"battery out of bounds ({name})"
        assert lg["q"].min() >= -1e-6 and lg["q"].max() <= p.qmax_kwh + 1e-6, f"queue out of bounds ({name})"

    json.dump(
        {"source": args.source, "start": start, "end": end, "import_ef": args.import_ef,
         "n_hours": n, "test_hours": int(len(steps)), "mean_intensity": float(y.mean()),
         "std_intensity": float(y.std()), "forecast": sc, "control": results},
        open(out / "results.json", "w"), indent=2,
    )

    # 5) figures
    plt.rcParams.update({"font.size": 9})
    mid = len(steps) // 2
    week = slice(mid, mid + 7 * 24)
    h_plot = 12
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(week.stop - week.start)
    ax.plot(x, F_true[week, h_plot - 1], color="black", lw=1.5, label="actual")
    ax.plot(x, F_pers[week, h_plot - 1], color="tab:orange", lw=1, label="same hour yesterday")
    ax.plot(x, F_ml[week, h_plot - 1], color="tab:blue", lw=1, label="gradient boosting")
    ax.set(title=f"{h_plot}-hour-ahead forecast, one week of the test period", xlabel="hours", ylabel="gCO2/kWh")
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "forecast_week.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    names = list(results)
    vals = [results[k]["adjusted_kg"] for k in names]
    ax.bar(range(len(names)), vals, color=["#888", "tab:orange", "tab:blue", "tab:green"])
    for i, k in enumerate(names):
        ax.text(i, vals[i], f"{results[k]['reduction_pct']:.1f} %" if i else "ref.", ha="center", va="bottom")
    ax.set_xticks(range(len(names)), [k.replace(", ", ",\n") for k in names])
    ax.set(ylabel=f"kg CO2 over {args.test_days} days", title="Emissions by strategy")
    fig.tight_layout()
    fig.savefig(out / "co2_by_strategy.png", dpi=150)
    plt.close(fig)

    lg, base = logs["MPC, gradient boosting"], logs["no control"]
    w0 = min(30 * 24, max(0, len(steps) - 72))
    win = slice(w0, w0 + 72)
    xs = np.arange(72)
    fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axs[0].plot(xs, v[steps][win], color="black")
    axs[0].set_ylabel("gCO2/kWh")
    axs[0].set_title("Three days of control (MPC with gradient boosting forecasts)")
    axs[1].plot(xs, base["grid"][win], color="#888", label="no control")
    axs[1].plot(xs, lg["grid"][win], color="tab:blue", label="MPC")
    axs[1].set_ylabel("grid import (kW)")
    axs[1].legend(frameon=False, ncol=2)
    axs[2].plot(xs, lg["e"][win] / p.cap_kwh * 100, color="tab:green", label="battery (% of capacity)")
    axs[2].plot(xs, lg["q"][win] / p.qmax_kwh * 100, color="tab:red", label="work queue (% of max)")
    axs[2].set(xlabel="hours", ylabel="%")
    axs[2].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "schedule_3days.png", dpi=150)
    plt.close(fig)
    print(f"\nResults and figures written to {out}/")


if __name__ == "__main__":
    main()
