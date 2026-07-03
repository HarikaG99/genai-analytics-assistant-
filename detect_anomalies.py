"""
Anomaly detection engine.

Two complementary detectors on daily order volume, evaluated against
the injected ground-truth labels:

    1. Rolling z-score      - residual vs 28-day rolling mean/std;
                              interpretable, near-real-time
    2. IsolationForest      - multivariate (orders, revenue,
                              active_users + day-of-week), catches
                              anomalies invisible to a single metric

An alert fires when either detector flags a day (ensemble OR-rule).
Reports precision / recall / F1 and writes anomaly_detection.png.

Usage:
    python models/detect_anomalies.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

Z_THRESHOLD = 3.2
WINDOW = 28


def main():
    df = pd.read_csv(ROOT / "data" / "daily_kpis.csv", parse_dates=["date"])

    # ---- detector 1: rolling z-score ---------------------------------
    roll_mean = df.orders.rolling(WINDOW, min_periods=14).mean().shift(1)
    roll_std = df.orders.rolling(WINDOW, min_periods=14).std().shift(1)
    df["zscore"] = (df.orders - roll_mean) / roll_std
    df["flag_z"] = (df.zscore.abs() > Z_THRESHOLD).astype(int)

    # ---- detector 2: IsolationForest ---------------------------------
    feats = df[["orders", "revenue", "active_users"]].copy()
    feats["dow"] = df.date.dt.dayofweek
    iso = IsolationForest(contamination=0.012, random_state=42)
    df["flag_iso"] = (iso.fit_predict(feats) == -1).astype(int)

    # ---- ensemble + evaluation ---------------------------------------
    df["alert"] = ((df.flag_z + df.flag_iso) > 0).astype(int)
    evald = df.iloc[WINDOW:]  # skip warm-up window
    p = precision_score(evald.is_anomaly, evald.alert)
    r = recall_score(evald.is_anomaly, evald.alert)
    f1 = f1_score(evald.is_anomaly, evald.alert)

    print("Anomaly detection vs ground truth:")
    print(f"  true anomalies: {int(evald.is_anomaly.sum())} | alerts fired: {int(evald.alert.sum())}")
    print(f"  precision {p:.2f} | recall {r:.2f} | F1 {f1:.2f}")

    alerts = df[df.alert == 1][["date", "orders", "zscore", "flag_z", "flag_iso", "is_anomaly"]]
    alerts.to_csv(OUT / "anomaly_alerts.csv", index=False)

    # ---- chart --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df.date, df.orders, linewidth=0.8, label="Daily orders")
    hits = df[(df.alert == 1) & (df.is_anomaly == 1)]
    fp = df[(df.alert == 1) & (df.is_anomaly == 0)]
    missed = df[(df.alert == 0) & (df.is_anomaly == 1)]
    ax.scatter(hits.date, hits.orders, color="red", s=45, zorder=3,
               label=f"Detected anomalies ({len(hits)})")
    ax.scatter(fp.date, fp.orders, color="orange", s=30, zorder=3,
               label=f"False positives ({len(fp)})")
    if len(missed):
        ax.scatter(missed.date, missed.orders, facecolors="none", edgecolors="purple",
                   s=60, zorder=3, label=f"Missed ({len(missed)})")
    ax.set_title(f"Anomaly Detection on Daily Orders — precision {p:.2f}, recall {r:.2f}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "anomaly_detection.png", dpi=150)
    print(f"Chart written to {OUT / 'anomaly_detection.png'}")


if __name__ == "__main__":
    main()
