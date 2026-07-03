# Predictive Forecasting & Anomaly Detection Platform

Time-series forecasting and automated anomaly detection on daily operational KPIs (orders, revenue, active users), replacing legacy spreadsheet-based forecasting with ML models and real-time anomaly alerting — built with **Python, Scikit-learn, and Prophet**.

![Anomaly Detection](outputs/anomaly_detection.png)

## Results (measured on a 90-day holdout)

| Model | MAPE | vs legacy |
|---|---|---|
| Seasonal naive ("legacy spreadsheet" method) | 6.43% | — |
| **Ridge + calendar/lag features (scikit-learn)** | **3.76%** | **41% more accurate** |
| Prophet (optional, auto-enabled if installed) | — | — |

**Anomaly detection** (ensemble of rolling z-score + IsolationForest, evaluated against ground-truth labels): **precision 0.67, recall 0.89, F1 0.76** — catching 16 of 18 injected incidents (outages and demand spikes) that a static-threshold alert would miss.

## What's in it

| File | Role |
|---|---|
| `data/generate_series.py` | 3 years of daily KPIs with trend, weekly + yearly seasonality, holiday effects, noise, and **18 labeled injected anomalies** for honest evaluation |
| `models/forecast.py` | Forecasting engine — seasonal-naive baseline, Ridge regression with engineered features (Fourier seasonality terms, day-of-week dummies, lags 1/7/14/28, rolling mean), recursive multi-step forecasting, optional Prophet backend |
| `models/detect_anomalies.py` | Two complementary detectors: interpretable rolling z-score (near-real-time) + multivariate IsolationForest, combined with an OR-rule ensemble; precision/recall/F1 reported vs ground truth |

## Design decisions worth asking me about

- **Recursive forecasting done honestly.** Multi-step forecasts feed *predictions* back into lag features rather than peeking at actuals — the mistake that makes many portfolio forecasts look artificially good.
- **A real baseline.** Model performance is only meaningful against the method it replaces; the seasonal-naive baseline stands in for the legacy spreadsheet approach.
- **Labeled anomalies.** Injecting ground-truth outages/spikes makes precision and recall measurable instead of eyeballed.
- **Ensemble detection.** The z-score detector is fast and explainable to stakeholders; IsolationForest catches multivariate anomalies (e.g., revenue diverging from orders) a single-metric rule can't see.

## How to run

```bash
pip install -r requirements.txt
python data/generate_series.py       # build the KPI dataset
python models/forecast.py            # forecast + accuracy comparison
python models/detect_anomalies.py    # anomaly detection + evaluation

# optional: enable the Prophet model
pip install prophet && python models/forecast.py
```

Outputs land in `outputs/`: `forecast_vs_actual.png`, `anomaly_detection.png`, `anomaly_alerts.csv` — the alerts CSV is the feed for a Power BI/Tableau incident dashboard.

## Tech stack

Python · pandas · NumPy · scikit-learn (Ridge, IsolationForest) · Prophet (optional) · matplotlib

![Forecast](outputs/forecast_vs_actual.png)
