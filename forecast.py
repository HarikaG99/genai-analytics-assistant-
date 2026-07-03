"""
Forecasting engine.

Trains on the first ~2.5 years and forecasts the final 90 days, comparing:

    1. Naive baseline          - "legacy spreadsheet" method: same value
                                 as 7 days ago (seasonal naive)
    2. Scikit-learn model      - Ridge regression on engineered calendar
                                 + lag features (trend, day-of-week,
                                 month, Fourier terms, lags 1/7/14/28)
    3. Prophet (optional)      - used automatically if installed

Reports MAPE per model and writes forecast_vs_actual.png.

Usage:
    python models/forecast.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
HORIZON = 90


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "daily_kpis.csv", parse_dates=["date"])
    return df.set_index("date")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    t = np.arange(len(df))
    f["t"] = t
    f["dow"] = df.index.dayofweek
    f = pd.get_dummies(f, columns=["dow"], drop_first=True)
    for k in (1, 2, 3):  # yearly Fourier terms
        f[f"sin{k}"] = np.sin(2 * np.pi * k * t / 365.25)
        f[f"cos{k}"] = np.cos(2 * np.pi * k * t / 365.25)
    for lag in (1, 7, 14, 28):
        f[f"lag{lag}"] = df.orders.shift(lag)
    f["roll7"] = df.orders.shift(1).rolling(7).mean()
    return f


def mape(actual, pred) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def recursive_forecast(model, df, feats, train_end, horizon):
    """Forecast day-by-day, feeding predictions back into lag features."""
    history = df.orders.astype(float).copy()
    preds = []
    cols = feats.columns
    for i in range(horizon):
        day = train_end + i
        row = feats.iloc[[day]].copy()
        for lag in (1, 7, 14, 28):
            row[f"lag{lag}"] = history.iloc[day - lag]
        row["roll7"] = history.iloc[day - 7:day].mean()
        p = float(model.predict(row[cols])[0])
        preds.append(p)
        history.iloc[day] = p
    return np.array(preds)


def main():
    df = load()
    feats = make_features(df)
    train_end = len(df) - HORIZON
    valid = feats.notna().all(axis=1)
    X_tr = feats[valid].iloc[: train_end - 28]
    y_tr = df.orders[valid].iloc[: train_end - 28]

    actual = df.orders.iloc[train_end:].values
    dates = df.index[train_end:]

    results = {}

    # 1) seasonal-naive baseline (the "legacy spreadsheet" approach)
    naive = df.orders.iloc[train_end - 7: train_end - 7 + HORIZON].values
    results["Seasonal naive (legacy)"] = (naive, mape(actual, naive))

    # 2) scikit-learn Ridge with engineered features
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    ridge_pred = recursive_forecast(model, df, feats, train_end, HORIZON)
    results["Ridge + calendar/lag features"] = (ridge_pred, mape(actual, ridge_pred))

    # 3) Prophet, if available
    try:
        from prophet import Prophet
        pdf = df.orders.iloc[:train_end].reset_index()
        pdf.columns = ["ds", "y"]
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        m.fit(pdf)
        future = m.make_future_dataframe(periods=HORIZON)
        fc = m.predict(future).yhat.iloc[-HORIZON:].values
        results["Prophet"] = (fc, mape(actual, fc))
    except ImportError:
        print("(Prophet not installed - skipping. `pip install prophet` to enable.)")

    # ---- report ------------------------------------------------------
    print(f"\n90-day holdout forecast accuracy (orders):")
    base = results["Seasonal naive (legacy)"][1]
    for name, (_, err) in results.items():
        improvement = (base - err) / base * 100
        extra = "" if name.startswith("Seasonal") else f"  ({improvement:+.0f}% vs legacy)"
        print(f"  {name:<32} MAPE {err:5.2f}%{extra}")

    # ---- chart -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[train_end - 120: train_end], df.orders.iloc[train_end - 120: train_end],
            color="gray", label="History")
    ax.plot(dates, actual, color="black", linewidth=2, label="Actual")
    for name, (pred, err) in results.items():
        ax.plot(dates, pred, "--", label=f"{name} (MAPE {err:.1f}%)")
    ax.set_title("90-Day Order Forecast vs Actual", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_vs_actual.png", dpi=150)
    print(f"\nChart written to {OUT / 'forecast_vs_actual.png'}")


if __name__ == "__main__":
    main()
