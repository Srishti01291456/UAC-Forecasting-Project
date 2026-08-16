# Predictive Forecasting of Care Load & Placement Demand

A machine learning and time-series forecasting system designed to predict short-term changes in the population of children under HHS care and support operational capacity planning.

---

## Overview

The UAC (Unaccompanied Alien Children) program operates in a highly uncertain environment where changes in border activity, policy enforcement, and humanitarian conditions can rapidly affect the number of children entering federal care.

Historical analytics can explain what has already happened, but operational planning requires forward-looking intelligence.

This project develops a predictive forecasting pipeline that transforms historical daily UAC operational data into forecasts of future care load and placement demand.

The system combines:

- Exploratory Data Analysis
- Time-Series Analysis
- Feature Engineering
- Statistical Forecasting
- Machine Learning
- Model Comparison
- Forecast Evaluation
- Capacity Risk Analysis
- Interactive Streamlit Visualization

---

## Problem Statement

Decision-makers need to anticipate changes in care demand before they occur.

This project addresses questions such as:

- How many children are likely to be under HHS care in the coming days?
- What is the expected short-term care-load trend?
- Is the care population increasing or decreasing?
- What level of placement or discharge activity may be required?
- Which forecasting model provides the most reliable predictions?
- Can forecast trends provide an early indication of potential capacity pressure?

---

## Project Objectives

### Primary Objectives

- Forecast the number of children in HHS care.
- Predict short-term care-load trends.
- Estimate future placement/discharge demand.
- Compare statistical and machine-learning forecasting approaches.
- Select the best-performing forecasting model.
- Support operational capacity planning.

### Secondary Objectives

- Identify important forecasting features.
- Provide early-warning indicators for increasing care demand.
- Compare models using multiple evaluation metrics.
- Visualize historical and predicted trends.
- Provide an interactive decision-support dashboard.

---

## Dataset

The project uses aggregate daily UAC operational data.

The dataset contains operational indicators such as:

| Feature | Description |
|---|---|
| Date | Reporting date |
| Children apprehended and placed in CBP custody | Daily intake volume |
| Children in CBP custody | Active CBP care load |
| Children transferred out of CBP custody | Flow from CBP custody toward the HHS system |
| Children in HHS Care | Active HHS care population |
| Children discharged from HHS Care | Children discharged or placed from HHS care |

The project works with **aggregate operational time-series data** rather than individual-level records.

---

## Methodology

The overall forecasting pipeline is:

```text
Historical UAC Data
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Time-Series Preparation
        |
        v
Feature Engineering
        |
        v
Baseline Forecasting
        |
        +------------------+
        |                  |
        v                  v
      ARIMA              SARIMA
        |                  |
        +--------+---------+
                 |
                 v
        Machine Learning Models
                 |
        +--------+---------+
        |                  |
        v                  v
   Random Forest     Gradient Boosting
        |                  |
        +--------+---------+
                 |
                 v
          Model Comparison
                 |
                 v
        Best Model Selection
                 |
                 v
          Future Forecast
                 |
                 v
      Capacity / Risk Analytics
                 |
                 v
        Streamlit Dashboard