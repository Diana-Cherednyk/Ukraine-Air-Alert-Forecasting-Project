# Ukraine Air Raid Alert Forecasting

A reproducible Python mini-project for **research-only** hourly air-raid-alert probability forecasting across 25 Ukrainian regional identifiers (including Kyiv City). **Never use these predictions instead of official air-raid warnings.**

## Quick start (the commands used to launch the project)

Extract the original `ukraine-air-alert-project-no-forecasts.zip`, open a terminal **in the folder containing `main.py`**, and run:

```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
```

Requires Python 3.11+ and the included `volunteer_data_en.csv`. `python main.py` prepares the training-only data, performs historical backtests, trains the models, and generates frozen forecasts. `streamlit run app.py` opens the interactive dashboard. The project ZIP intentionally does not contain precomputed forecasts. If rerunning, first archive/remove the existing `outputs/blind_forecast.csv` and `outputs/forecast_lock.json`; the training script refuses to overwrite a frozen forecast.

## Objective and data

Source: user-provided `volunteer_data_en.csv` [ukrainian airaid sirens dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) , containing regional alert intervals with `region`, `started_at`, and `finished_at`. For each region and local hour, estimate:
- `prob_started`: probability at least one new alert starts during the hour;
- `prob_active`: probability an alert is active during any part of the hour.

Intervals are treated as half-open `[started_at, finished_at)`; an alert lasting 12:00–14:20 is active in the 12:00, 13:00, and 14:00 hourly bins. This version works at **regional level only**, not district level.

## Forecast experiment and leakage controls

- Source history begins in February 2022; the cutoff is **2026-10-05 00:00 Europe/Kyiv**.
- Blind forecast: October 5–12, 2026, 192 hourly bins per region.
- `scripts/prepare.py` retains only alert starts strictly before the cutoff and censors finish times crossing the cutoff.
- `scripts/train_forecast.py` reads only `data/train_only.csv`; it does not receive the raw holdout dataset.
- Forecast inputs are region identity and known-in-advance calendar features; no observed October 5–12 alert data are used as features.
- The forecast is saved as `outputs/blind_forecast.csv`, and its SHA-256 is recorded in `outputs/forecast_lock.json`.
- Holdout evaluation is **not** called by `main.py`. After freezing predictions, it can be explicitly invoked with `python scripts/evaluate_holdout.py --raw volunteer_data_en.csv`.

**Important qualification:** Although the training-only CSV contains pre-cutoff records back to 2022, the current training script builds its hourly modeling panel starting **425 days before the cutoff** (`start = CUTOFF - pd.Timedelta(days=425)`). Thus the fitted models do **not** use all 2022–2026 records. This is an honest description of the retained baseline, not a claim of full-history training. The original CSV remains in the project folder, so isolation is enforced by pipeline design rather than by operating-system file permissions.

## Modeling and evaluation

Two scikit-learn `HistGradientBoostingClassifier` models are trained separately for alert starts and active alerts. Features are regional identity and calendar features (hour, weekday, annual cyclic features, weekend). No Telegram or other future external information is used.

The script performs three chronological, eight-day rolling-origin backtests at 24, 16, and 8 days before the cutoff. Each fold fits on observations strictly earlier than its origin and evaluates on the next eight days. Metrics include Brier score, log loss, ROC AUC, and a constant-prevalence Brier baseline. Final October 5–12 actuals must not influence model selection.

This is a baseline, not a reliable operational prediction system. Alert dynamics depend on future events absent from historical calendar features; recording practices and coverage may change. Predictions can be miscalibrated.

## Dashboard

Run `streamlit run app.py` to inspect regional probabilities by time, interactive hourly charts, heatmaps, regional comparisons, and available backtest outputs. Geographic map rendering requires a verified `assets/oblasts.geojson` with `feature.properties.region` matching CSV region names; it is not bundled. District forecasts are not supported by this dataset.

## Project structure

```text
main.py                        integrated preparation + training + forecast
app.py                         Streamlit dashboard
volunteer_data_en.csv          user-provided source dataset
requirements.txt               dependencies
src/pipeline.py                timestamp, feature, interval processing
scripts/prepare.py             cutoff-safe data extraction
scripts/train_forecast.py      backtests, fitting, frozen forecast
scripts/evaluate_holdout.py    optional, explicit holdout evaluation
tests/test_pipeline.py         unit tests
data/                          generated training-only dataset
outputs/                       generated forecast and metrics
```

## Tests

```bash
python -m pytest -q tests
```
