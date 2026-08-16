from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "uac_data.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model.pkl"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "year",
    "month",
    "quarter",
    "dayofweek",
    "dayofyear",
    "lag1",
    "lag2",
    "lag3",
    "lag7",
    "lag14",
    "lag30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7"
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Children in HHS Care"] = (
        df["Children in HHS Care"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False
        )
    )

    df["Children in HHS Care"] = pd.to_numeric(
        df["Children in HHS Care"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Date",
            "Children in HHS Care"
        ]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# FORECAST
# ============================================================

def forecast_future(
    model,
    df,
    steps=7
):

    history = list(
        df["Children in HHS Care"]
        .astype(float)
        .values
    )

    last_date = df["Date"].max()

    predictions = []

    future_dates = []

    for step in range(steps):

        future_date = (
            last_date
            + pd.Timedelta(
                days=step + 1
            )
        )

        # Calendar
        year = future_date.year
        month = future_date.month
        quarter = future_date.quarter
        dayofweek = future_date.dayofweek
        dayofyear = future_date.dayofyear

        # Lags
        lag1 = history[-1]
        lag2 = history[-2]
        lag3 = history[-3]
        lag7 = history[-7]
        lag14 = history[-14]
        lag30 = history[-30]

        # Rolling
        rolling_mean_7 = np.mean(
            history[-7:]
        )

        rolling_mean_14 = np.mean(
            history[-14:]
        )

        rolling_std_7 = np.std(
            history[-7:],
            ddof=1
        )

        X_future = pd.DataFrame(
            [[
                year,
                month,
                quarter,
                dayofweek,
                dayofyear,
                lag1,
                lag2,
                lag3,
                lag7,
                lag14,
                lag30,
                rolling_mean_7,
                rolling_mean_14,
                rolling_std_7
            ]],
            columns=FEATURES
        )

        prediction = float(
            model.predict(
                X_future
            )[0]
        )

        prediction = max(
            0,
            prediction
        )

        predictions.append(
            prediction
        )

        future_dates.append(
            future_date
        )

        history.append(
            prediction
        )

    return (
        np.array(predictions),
        pd.to_datetime(future_dates)
    )


# ============================================================
# RISK
# ============================================================

def calculate_risk(
    forecast,
    capacity=25000
):

    peak = float(
        np.max(forecast)
    )

    utilization = (
        peak / capacity
    ) * 100

    if utilization >= 95:

        level = "HIGH"

    elif utilization >= 80:

        level = "MEDIUM"

    else:

        level = "LOW"

    return {
        "level": level,
        "percentage": utilization,
        "peak": peak
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Loading model...")

    model = load_model()

    print("Loading data...")

    df = load_data()

    print("Generating forecast...")

    forecast, dates = forecast_future(
        model,
        df,
        steps=7
    )

    print("\nForecast:")

    for date, value in zip(
        dates,
        forecast
    ):

        print(
            f"{date.date()} : "
            f"{value:.2f}"
        )

    risk = calculate_risk(
        forecast
    )

    print("\nRisk:")

    print(risk)