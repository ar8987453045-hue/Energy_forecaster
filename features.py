import numpy as np
import pandas as pd

LAGS = [1, 2, 24, 48, 168]   # hours back: 1h, 2h, 1 day, 2 days, 1 week

# The exact list of clues the model uses. Training and the app both use this list.
FEATURES = (
    ["hour", "dayofweek", "month", "is_weekend"]
    + [f"lag_{l}" for l in LAGS]
    + ["roll_24", "roll_168"]
)


def add_features(s):
    """Takes hourly usage (indexed by time) and builds a table of clues."""
    df = pd.DataFrame({"y": s})
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek          # Monday=0 ... Sunday=6
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    for l in LAGS:
        df[f"lag_{l}"] = s.shift(l)               # value from l hours ago

    # shift(1) first, so the average never includes the hour we are predicting
    df["roll_24"] = s.shift(1).rolling(24).mean()
    df["roll_168"] = s.shift(1).rolling(168).mean()
    return df


def forecast(model, history, horizon):
    """Predicts the next `horizon` hours, one at a time.
    Each prediction is added to the history and used to predict the next hour."""
    hist = history.astype(float).copy()
    preds = {}
    for _ in range(horizon):
        next_time = hist.index[-1] + pd.Timedelta(hours=1)
        hist.loc[next_time] = np.nan                       # empty slot for the new hour
        row = add_features(hist.iloc[-200:]).iloc[[-1]]    # clues for that hour
        value = float(model.predict(row[FEATURES])[0])
        hist.iloc[-1] = value                              # store the prediction
        preds[next_time] = value
    return pd.Series(preds, name="predicted_kwh")