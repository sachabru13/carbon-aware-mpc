# Carbon-aware control of a flexible load and a battery

A small project to see how far a simple forecast plus model predictive control (MPC) can go
in shifting electricity use towards low-carbon hours in Switzerland. It forecasts the carbon
intensity of the Swiss grid for the next 24 hours, then uses that forecast to decide when a
battery should charge and when deferrable work should run.

It is a toy model. It is not a model of any real installation (a data center, for example),
and the limitations section below matters as much as the code.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                      # unit tests, about 10 s
python run_demo.py --source synthetic    # offline, uses generated data
python run_demo.py --source real --start 2024-01-01 --end 2026-09-30
```

The real-data run downloads hourly generation by source from the public
[Energy-Charts](https://api.energy-charts.info) API (Fraunhofer ISE) and caches it in
`data_cache/`. Outputs go to `results/<source>/`: `results.json` and three figures.
Useful options: `--test-days` (default 90) and `--import-ef`, the assumed intensity of imported
power in gCO2/kWh (default 380).

## How it works

**1. Carbon intensity (`data.py`).** For each hour, domestic generation (nuclear, run-of-river
hydro, reservoir and pumped hydro, wind, solar, "others") is weighted by life-cycle emission
factors. Net imports, taken from the cross-border trading series, are added at one fixed
factor. When Switzerland exports, the intensity is just the domestic mix.

**2. Forecast (`forecast.py`).** For every hour t, I predict the intensity at t+1 ... t+24
using only data up to t. Two models are compared:
- the same hour yesterday (baseline);
- a gradient boosting regressor (scikit-learn) with lags at 1, 2 and 7 days, the last 24 h
  mean and trend, hour, weekday, day of year and the forecast horizon.

The model is trained on targets that all lie before the test period (the last 90 days by
default), so nothing from the test period is seen during training.

**3. Control (`mpc.py`).** Toy system: 800 kW of inflexible load, deferrable work averaging
400 kW that can wait in a queue (capped at about 6 hours of work, which stands in for
deadlines), and a 2 MWh / 1 MW battery with 0.95 efficiency each way. Every hour a linear
program (cvxpy, HiGHS solver) minimizes forecast intensity times grid import over the next
24 h. Only the first decision is applied, then it is solved again with the new state. The
battery cannot end the horizon emptier than it started and the queue cannot end longer.

**4. Comparison.** Four runs on the same period, scored with the real intensity: no control
(work is served as it arrives, no battery); MPC with the yesterday forecast; MPC with the
gradient boosting forecast; MPC with a perfect forecast, which is an upper bound on what better
forecasting could bring.

**Accounting.** Emissions are reported raw and adjusted. A controller can look better just by
leaving work in the queue or energy in the battery at the end of the run, so the adjusted
figure adds the leftover queue and subtracts the leftover battery energy, both valued at the
mean intensity of the period. That correction is my choice, and the raw numbers are kept in
`results.json`.

## Results

The only results included (`results/synthetic/`) come from the synthetic generator. They show
that the pipeline runs end to end and that the pieces behave sensibly, nothing more.

| Strategy (synthetic data, 90 days) | CO2 reduction vs no control |
|---|---|
| MPC, same hour yesterday | 1.5 % |
| MPC, gradient boosting | 1.8 % |
| MPC, perfect forecast | 2.0 % |

The ordering is the expected one, and the forecast MAE is about 22 % lower than the
baseline. The magnitudes mean nothing because I generated the data, and the synthetic
intensity is smooth and very predictable. Results on real data have to be produced with
`--source real`.

Checks in `tests/`: constant intensity gives no gain and no loss; a strong daily swing gives a
clear gain; work and battery energy are conserved exactly; all power, energy and queue limits are
respected; the API parser and the intensity formula are checked against a response that copies
the structure and series names of the real endpoint.

## Limitations

**The intensity signal is a proxy.**
- It is not the official carbon intensity of Swiss supply. Imports are charged at one constant
  factor, so on real data the signal mostly tracks how much is imported. The real intensity of
  imports changes hour by hour with the mix in neighbouring countries. Conclusions can depend
  strongly on `--import-ef`, and it is worth running a few values.
- Emission factors are life-cycle orders of magnitude. The "others" series (waste, biomass, gas
  and so on) has no breakdown, so its factor of 300 g/kWh is a guess.
- Pumped storage is counted as hydro, and pumping consumption is ignored.

**The forecast is basic.**
- Only past intensity and calendar variables are used. No weather, no solar or wind forecast,
  no published day-ahead load forecast, no market prices. That is where most of the accuracy
  would come from.
- A forecast is issued every hour for the next 24 h. In practice a day-ahead forecast is
  often issued once a day, which would be less accurate for the late hours.
- One test window (90 days by default), one random seed, no confidence intervals, no
  cross-validation across seasons, and no hyperparameter tuning. A single 90-day window can
  be unrepresentative.

**The control model is a toy.**
- The profile of deferrable work is known in advance; only the intensity is uncertain.
- Deadlines are replaced by a cap on the queue. Battery ageing is represented only by a small
  throughput penalty, and efficiency is constant.
- There is no thermal side at all: no heat recovery, no heat pump, no district heating. For a
  real data center, using the waste heat is a large part of the problem.
- No grid tariffs, no peak-power charges, no export, no ancillary services.
- The terminal conditions (battery and queue not worse at the end of the horizon) are
  conservative and probably understate the potential gain.

**Data pipeline status.**
- The response format was checked against live responses of the `public_power` endpoint for two
  days (March 2025 and September 2026), and the parser and the 90-day chunked download are
  tested with mocked responses.
- A full multi-year download has not been run yet. If the first real run fails, the series
  mapping in `data.py` (`NAME_MAP`) is the first place to look.
- Missing hours are interpolated, not set to zero. Data is hourly (15-minute data would be
  averaged to hours).
- The workload profile follows Swiss local time. The forecast model uses UTC calendar
  features, which is consistent but ignores daylight saving effects on demand.

## Possible next steps

- Run on real data and report the sensitivity to the import factor.
- Replace the constant import factor with the generation mix of the neighbouring countries
  (the same API offers it per country).
- Add weather-based features and compare with the day-ahead forecasts from ENTSO-E.
- Add a thermal model (heat recovery and storage) to the control problem.
- Test across several seasons and with several random seeds, and report confidence intervals.

## Files

```
data.py          generation mix download and parsing, synthetic data, carbon intensity
forecast.py      baseline and gradient boosting forecasts
mpc.py           controller (cvxpy) and closed-loop simulation
run_demo.py      full experiment, figures
tests/           unit tests
results/         outputs of past runs
```
