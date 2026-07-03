"""
Predictive Forecasting & Anomaly Detection Platform
---------------------------------------------------
Generates 3 years of daily operational KPI data (orders, revenue,
active_users) with trend, weekly + yearly seasonality, holidays,
noise, and injected anomalies (outages, spikes) with ground-truth
labels for evaluating the anomaly detector.

Usage:
    python data/generate_series.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(11)
OUT = Path(__file__).parent
DAYS = 365 * 3

def build():
    idx = pd.date_range("2023-07-01", periods=DAYS, freq="D")
    t = np.arange(DAYS)

    trend = 1000 + 0.9 * t                                   # steady growth
    weekly = 140 * np.sin(2 * np.pi * (t % 7) / 7 - 0.8)     # weekday cycle
    yearly = 220 * np.sin(2 * np.pi * t / 365.25 - 1.2)      # seasonal cycle
    noise = RNG.normal(0, 45, DAYS)
    orders = trend + weekly + yearly + noise

    # holiday dips (approx Christmas/New Year windows)
    holiday = ((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 2))
    orders = np.where(holiday, orders * 0.55, orders)

    # ---- inject labeled anomalies -----------------------------------
    labels = np.zeros(DAYS, dtype=int)
    anomaly_days = RNG.choice(np.arange(60, DAYS - 10), 18, replace=False)
    for d in anomaly_days:
        kind = RNG.choice(["outage", "spike"])
        if kind == "outage":
            orders[d] *= RNG.uniform(0.25, 0.55)   # system outage
        else:
            orders[d] *= RNG.uniform(1.6, 2.2)     # flash-sale spike
        labels[d] = 1

    df = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"),
        "orders": orders.round(0).astype(int),
        "revenue": (orders * RNG.uniform(38, 44, DAYS)).round(0).astype(int),
        "active_users": (orders * RNG.uniform(2.1, 2.5, DAYS)).round(0).astype(int),
        "is_anomaly": labels,
    })
    df.to_csv(OUT / "daily_kpis.csv", index=False)
    print(f"Wrote {len(df):,} days of KPI data | {labels.sum()} injected anomalies")

if __name__ == "__main__":
    build()
