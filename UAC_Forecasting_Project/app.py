import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UAC Forecasting Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Dataset stays inside UAC_Forecasting_Project/data
DATA_PATH = (
    BASE_DIR
    / "data"
    / "uac_data.csv"
)

# IMPORTANT:
# Your working model is one level ABOVE UAC_Forecasting_Project.
#
# Do NOT move best_model.pkl.
MODEL_PATH = (
    BASE_DIR.parent
    / "models"
    / "best_model.pkl"
)

DATA_DIR = (
    BASE_DIR
    / "data"
)


# ============================================================
# MODEL FEATURES
# EXACTLY MATCHES RANDOM FOREST TRAINING
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE VALIDATION
# ============================================================

if not DATA_PATH.exists():

    st.error(
        f"""
        ❌ Dataset not found.

        Expected:

        {DATA_PATH}
        """
    )

    st.stop()


if not MODEL_PATH.exists():

    st.error(
        f"""
        ❌ Random Forest model not found.

        Expected:

        {MODEL_PATH}

        Your model should remain in the parent models folder.
        """
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    required_columns = {
        "Date",
        "Children in HHS Care"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Date",
            "Children in HHS Care"
        ]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Need at least 30 previous observations
    if len(df) < 31:

        raise ValueError(
            "At least 31 observations are required "
            "for lag-30 forecasting."
        )

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    try:

        model = joblib.load(
            MODEL_PATH
        )

    except Exception as error:

        st.error(
            "❌ Could not load best_model.pkl"
        )

        st.code(
            str(error)
        )

        st.stop()

    if not hasattr(
        model,
        "predict"
    ):

        st.error(
            "❌ best_model.pkl does not contain "
            "a valid prediction model."
        )

        st.stop()

    return model


# ============================================================
# LOAD
# ============================================================

try:

    df = load_data()

    model = load_model()

except Exception as error:

    st.error(
        "❌ Application initialization failed."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# VERIFY MODEL FEATURES
# ============================================================

if hasattr(
    model,
    "feature_names_in_"
):

    model_features = list(
        model.feature_names_in_
    )

    if model_features != FEATURES:

        st.error(
            "❌ Model feature mismatch."
        )

        st.write(
            "Features expected by application:"
        )

        st.code(
            FEATURES
        )

        st.write(
            "Features stored in model:"
        )

        st.code(
            model_features
        )

        st.stop()


# ============================================================
# READ METRICS
# ============================================================

def safe_number(value):

    if pd.isna(value):

        return np.nan

    text = str(value).strip()

    text = (
        text
        .replace(",", "")
        .replace("%", "")
    )

    try:

        return float(text)

    except:

        return np.nan


def read_metric_file(
    filename
):

    path = (
        DATA_DIR
        / filename
    )

    if not path.exists():

        return None

    try:

        metrics_df = pd.read_csv(
            path
        )

        if metrics_df.empty:

            return None

        result = {}

        # ----------------------------------------------------
        # Format 1:
        # MAE, RMSE, MAPE as columns
        # ----------------------------------------------------

        for metric in [
            "MAE",
            "RMSE",
            "MAPE"
        ]:

            matching_columns = [

                column

                for column
                in metrics_df.columns

                if str(column)
                .strip()
                .upper()
                == metric
            ]

            if matching_columns:

                value = safe_number(
                    metrics_df.iloc[0][
                        matching_columns[0]
                    ]
                )

                if not np.isnan(value):

                    result[metric] = value

        # ----------------------------------------------------
        # Format 2:
        # Metric / Value rows
        # ----------------------------------------------------

        if not result:

            lower_columns = {
                str(column)
                .strip()
                .lower():
                column

                for column
                in metrics_df.columns
            }

            metric_column = (
                lower_columns.get("metric")
            )

            value_column = (
                lower_columns.get("value")
            )

            if (
                metric_column
                and value_column
            ):

                for _, row in metrics_df.iterrows():

                    metric_name = str(
                        row[metric_column]
                    ).strip().upper()

                    if metric_name in [
                        "MAE",
                        "RMSE",
                        "MAPE"
                    ]:

                        result[
                            metric_name
                        ] = safe_number(
                            row[value_column]
                        )

        return (
            result
            if result
            else None
        )

    except Exception:

        return None


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

def load_model_comparison():

    comparison_path = (
        DATA_DIR
        / "model_comparison.csv"
    )

    # --------------------------------------------------------
    # First try model_comparison.csv
    # --------------------------------------------------------

    if comparison_path.exists():

        try:

            comparison = pd.read_csv(
                comparison_path
            )

            if not comparison.empty:

                return comparison

        except Exception:

            pass

    # --------------------------------------------------------
    # Otherwise build comparison from individual files
    # --------------------------------------------------------

    model_files = {

        "Random Forest":
            "random_forest_metrics.csv",

        "Gradient Boosting":
            "gradient_boosting_metrics.csv",

        "ARIMA":
            "arima_metrics.csv",

        "SARIMA":
            "sarima_metrics.csv"
    }

    rows = []

    for model_name, filename in (
        model_files.items()
    ):

        metrics = read_metric_file(
            filename
        )

        if metrics:

            rows.append(
                {
                    "Model":
                        model_name,

                    **metrics
                }
            )

    if rows:

        return pd.DataFrame(
            rows
        )

    return pd.DataFrame()


comparison = load_model_comparison()


# ============================================================
# CURRENT POPULATION
# ============================================================

current_population = float(
    df[
        "Children in HHS Care"
    ].iloc[-1]
)


# ============================================================
# FORECAST FUNCTION
# ============================================================

def forecast_random_forest(
    model,
    dataframe,
    steps
):

    history = list(

        dataframe[
            "Children in HHS Care"
        ]
        .astype(float)
        .values
    )

    last_date = (
        dataframe["Date"].max()
    )

    predictions = []

    future_dates = []

    for step in range(
        steps
    ):

        # ----------------------------------------------------
        # Future date
        # ----------------------------------------------------

        future_date = (

            last_date

            + pd.Timedelta(
                days=step + 1
            )
        )

        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

        year = (
            future_date.year
        )

        month = (
            future_date.month
        )

        quarter = (
            future_date.quarter
        )

        dayofweek = (
            future_date.dayofweek
        )

        dayofyear = (
            future_date.dayofyear
        )

        # ----------------------------------------------------
        # Lag features
        # ----------------------------------------------------

        lag1 = history[-1]

        lag2 = history[-2]

        lag3 = history[-3]

        lag7 = history[-7]

        lag14 = history[-14]

        lag30 = history[-30]

        # ----------------------------------------------------
        # Rolling features
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EXACT MODEL INPUT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = float(

            model.predict(
                X_future
            )[0]
        )

        # No negative population
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

        # Recursive forecasting
        history.append(
            prediction
        )

    return (

        np.array(
            predictions
        ),

        pd.to_datetime(
            future_dates
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Forecast Controls"
)

st.sidebar.success(
    "🏆 Selected Model: Random Forest"
)

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [7, 14, 30],
    index=0
)


# ============================================================
# DATA-DRIVEN CAPACITY
# ============================================================

historical_values = (
    df[
        "Children in HHS Care"
    ]
    .astype(float)
)

historical_mean = float(
    historical_values.mean()
)

historical_max = float(
    historical_values.max()
)

historical_95 = float(
    np.percentile(
        historical_values,
        95
    )
)

default_capacity = int(

    np.ceil(
        historical_95
        / 100
    )
    * 100
)


capacity = st.sidebar.number_input(

    "Reference Capacity",

    min_value=100,

    value=max(
        default_capacity,
        100
    ),

    step=100
)

st.sidebar.caption(
    "Default = historical 95th percentile. "
    "Change this to your real operational capacity if known."
)


# ============================================================
# GENERATE FORECAST
# ============================================================

forecast, future_dates = (
    forecast_random_forest(
        model,
        df,
        horizon
    )
)


# ============================================================
# FORECAST ANALYTICS
# ============================================================

peak_forecast = float(
    np.max(
        forecast
    )
)

average_forecast = float(
    np.mean(
        forecast
    )
)

minimum_forecast = float(
    np.min(
        forecast
    )
)

final_forecast = float(
    forecast[-1]
)


if current_population != 0:

    forecast_change = (

        (
            final_forecast
            - current_population
        )
        / current_population

    ) * 100

else:

    forecast_change = 0


# ============================================================
# CAPACITY RISK
# ============================================================

capacity_utilization = (

    peak_forecast
    / capacity

) * 100


if capacity_utilization >= 95:

    risk_level = "HIGH"

elif capacity_utilization >= 80:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📊 UAC Forecasting Analytics
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-powered forecasting, model analytics
        and capacity-risk monitoring
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# FORECAST OVERVIEW
# ============================================================

st.markdown(
    "## 📌 Forecast Overview"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Current Population",
        f"{current_population:,.0f}"
    )

with col2:

    st.metric(
        "Peak Forecast",
        f"{peak_forecast:,.0f}"
    )

with col3:

    st.metric(
        "Average Forecast",
        f"{average_forecast:,.0f}"
    )

with col4:

    st.metric(
        "Forecast Change",
        f"{forecast_change:+.2f}%"
    )


# ============================================================
# HISTORICAL VS FORECAST
# ============================================================

st.markdown("---")

st.markdown(
    "## 📈 Historical vs Forecast"
)

fig = go.Figure()


# Historical
fig.add_trace(

    go.Scatter(

        x=df["Date"],

        y=df[
            "Children in HHS Care"
        ],

        mode="lines",

        name="Historical",

        line=dict(
            width=2
        )
    )
)


# Forecast
fig.add_trace(

    go.Scatter(

        x=future_dates,

        y=forecast,

        mode="lines+markers",

        name="Random Forest Forecast",

        line=dict(
            width=3
        )
    )
)


# Reference capacity
fig.add_hline(

    y=capacity,

    line_dash="dash",

    annotation_text=(
        "Reference Capacity"
    ),

    annotation_position=(
        "top right"
    )
)


fig.update_layout(

    template="plotly_dark",

    height=520,

    xaxis_title="Date",

    yaxis_title=(
        "Children in HHS Care"
    ),

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# FORECAST DETAILS
# ============================================================

st.markdown("---")

st.markdown(
    "## 📅 Forecast Details"
)

forecast_table = pd.DataFrame(

    {

        "Date":
            future_dates.strftime(
                "%Y-%m-%d"
            ),

        "Predicted Population":
            np.rint(
                forecast
            ).astype(int),

        "Capacity Utilization":
            np.round(

                (
                    forecast
                    / capacity
                )
                * 100,

                2
            )
    }
)


st.dataframe(

    forecast_table,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# CAPACITY RISK ANALYTICS
# ============================================================

st.markdown("---")

st.markdown(
    "## ⚠️ Capacity Risk Analytics"
)

r1, r2, r3 = (
    st.columns(3)
)

with r1:

    st.metric(
        "Reference Capacity",
        f"{capacity:,.0f}"
    )

with r2:

    st.metric(
        "Peak Forecast",
        f"{peak_forecast:,.0f}"
    )

with r3:

    st.metric(
        "Peak Utilization",
        f"{capacity_utilization:.2f}%"
    )


if risk_level == "HIGH":

    st.error(

        f"""
        🔴 HIGH RISK

        Forecast peak reaches
        {capacity_utilization:.2f}%
        of reference capacity.
        """
    )

elif risk_level == "MEDIUM":

    st.warning(

        f"""
        🟡 MEDIUM RISK

        Forecast peak reaches
        {capacity_utilization:.2f}%
        of reference capacity.
        """
    )

else:

    st.success(

        f"""
        🟢 LOW RISK

        Forecast peak reaches
        {capacity_utilization:.2f}%
        of reference capacity.
        """
    )


# ============================================================
# HISTORICAL ANALYTICS
# ============================================================

st.markdown("---")

st.markdown(
    "## 📊 Historical Analytics"
)

h1, h2, h3, h4 = (
    st.columns(4)
)

with h1:

    st.metric(
        "Historical Mean",
        f"{historical_mean:,.0f}"
    )

with h2:

    st.metric(
        "Historical Maximum",
        f"{historical_max:,.0f}"
    )

with h3:

    st.metric(
        "95th Percentile",
        f"{historical_95:,.0f}"
    )

with h4:

    st.metric(
        "Forecast Minimum",
        f"{minimum_forecast:,.0f}"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("---")

st.markdown(
    "## 🏆 Model Comparison"
)

if comparison.empty:

    st.warning(
        """
        Model comparison files were not found.
        Make sure these files exist inside data/:
        
        model_comparison.csv
        random_forest_metrics.csv
        gradient_boosting_metrics.csv
        arima_metrics.csv
        sarima_metrics.csv
        """
    )

else:

    # --------------------------------------------------------
    # Normalize model column
    # --------------------------------------------------------

    model_column = None

    for column in comparison.columns:

        if (
            str(column)
            .strip()
            .lower()
            == "model"
        ):

            model_column = column

            break

    if model_column is None:

        model_column = (
            comparison.columns[0]
        )

    comparison = comparison.rename(

        columns={
            model_column:
                "Model"
        }
    )


    # --------------------------------------------------------
    # Normalize metric values
    # --------------------------------------------------------

    for metric in [
        "MAE",
        "RMSE",
        "MAPE"
    ]:

        if metric in comparison.columns:

            comparison[metric] = (

                comparison[metric]
                .apply(
                    safe_number
                )
            )


    # --------------------------------------------------------
    # Sort by MAE
    # --------------------------------------------------------

    if "MAE" in comparison.columns:

        comparison = (

            comparison
            .sort_values(
                "MAE",
                na_position="last"
            )
            .reset_index(
                drop=True
            )
        )


    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    ranks = []

    for index in range(
        len(comparison)
    ):

        if index == 0:

            ranks.append("🥇")

        elif index == 1:

            ranks.append("🥈")

        elif index == 2:

            ranks.append("🥉")

        else:

            ranks.append(
                str(index + 1)
            )


    comparison.insert(
        0,
        "Rank",
        ranks
    )


    # --------------------------------------------------------
    # Display copy
    # --------------------------------------------------------

    display_comparison = (
        comparison.copy()
    )


    if "MAE" in display_comparison.columns:

        display_comparison["MAE"] = (

            display_comparison["MAE"]
            .apply(

                lambda value:

                f"{value:.2f}"

                if pd.notna(value)

                else "—"
            )
        )


    if "RMSE" in display_comparison.columns:

        display_comparison["RMSE"] = (

            display_comparison["RMSE"]
            .apply(

                lambda value:

                f"{value:.2f}"

                if pd.notna(value)

                else "—"
            )
        )


    if "MAPE" in display_comparison.columns:

        display_comparison["MAPE"] = (

            display_comparison["MAPE"]
            .apply(

                lambda value:

                f"{value:.2f}%"

                if pd.notna(value)

                else "—"
            )
        )


    st.dataframe(

        display_comparison,

        use_container_width=True,

        hide_index=True
    )


    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    if (
        "Model" in comparison.columns
        and len(comparison) > 0
    ):

        best_model_name = (
            comparison.iloc[0]["Model"]
        )

        st.success(

            f"""
            🏆 Best Overall Model:
            **{best_model_name}**

            It ranks first according to the
            saved evaluation results.
            """
        )


# ============================================================
# RANDOM FOREST PERFORMANCE
# ============================================================

st.markdown("---")

st.markdown(
    "## 🎯 Random Forest Performance"
)

rf_metrics = read_metric_file(
    "random_forest_metrics.csv"
)

p1, p2, p3 = (
    st.columns(3)
)

if rf_metrics:

    with p1:

        st.metric(
            "MAE",
            f"{rf_metrics.get('MAE', np.nan):.2f}"
        )

    with p2:

        st.metric(
            "RMSE",
            f"{rf_metrics.get('RMSE', np.nan):.2f}"
        )

    with p3:

        st.metric(
            "MAPE",
            f"{rf_metrics.get('MAPE', np.nan):.2f}%"
        )

else:

    with p1:

        st.metric(
            "MAE",
            "N/A"
        )

    with p2:

        st.metric(
            "RMSE",
            "N/A"
        )

    with p3:

        st.metric(
            "MAPE",
            "N/A"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    "## 🤖 Selected Model"
)

m1, m2 = (
    st.columns(2)
)

with m1:

    st.markdown(
        """
        ### Random Forest Regressor

        - **Trees:** 300
        - **Max Depth:** 15
        - **Random State:** 42
        - **Features:** 14
        - **Forecasting:** Recursive multi-step
        """
    )


with m2:

    st.markdown(
        """
        ### Feature Engineering

        - Calendar features
        - Lag 1 / 2 / 3
        - Lag 7 / 14 / 30
        - 7-day rolling mean
        - 14-day rolling mean
        - 7-day rolling standard deviation
        """
    )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown("---")

st.markdown(
    "## 📋 Executive Summary"
)

st.info(

    f"""
    **Selected Model:** Random Forest

    **Forecast Horizon:** {horizon} days

    **Current Population:** {current_population:,.0f}

    **Peak Forecast:** {peak_forecast:,.0f}

    **Average Forecast:** {average_forecast:,.0f}

    **Reference Capacity:** {capacity:,.0f}

    **Peak Utilization:** {capacity_utilization:.2f}%

    **Risk Level:** {risk_level}

    **Forecast Change:** {forecast_change:+.2f}%

    The dashboard uses the saved Random Forest model
    and the same feature structure used during model training.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "UAC Forecasting Project • "
    "Random Forest Predictive Analytics"
)