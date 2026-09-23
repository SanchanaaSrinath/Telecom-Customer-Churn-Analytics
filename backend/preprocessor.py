# -*- coding: utf-8 -*-

"""
backend/preprocessor.py
=======================
Feature engineering helpers used by both train_model.py (fit) and
app.py / predictor.py (transform-only).
"""

from __future__ import annotations
import pandas as pd
import numpy as np


CATEGORICAL_COLS = [
    "Gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = [
    "Tenure", "MonthlyCharges", "TotalCharges",
    "AvgMonthlyCharge", "HasMultipleServices",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns in-place (returns copy)."""
    df = df.copy()
    df["AvgMonthlyCharge"] = df["TotalCharges"] / df["Tenure"].replace(0, 1)
    df["HasMultipleServices"] = (
        (df["OnlineSecurity"]  == "Yes").astype(int)
        + (df["OnlineBackup"]  == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"]   == "Yes").astype(int)
        + (df["StreamingTV"]   == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    return df


def build_feature_matrix(df: pd.DataFrame, drop_target: bool = True) -> pd.DataFrame:
    """Return the full feature matrix (target excluded by default).

    Parameters
    ----------
    df : pd.DataFrame
        Raw (but cleaned) dataframe from data_loader.load_data().
    drop_target : bool
        Whether to drop the 'Churn' column.
    """
    df = add_derived_features(df)
    df = df.drop(columns=["Customer_ID"], errors="ignore")
    if drop_target:
        df = df.drop(columns=["Churn"], errors="ignore")
    return df


def single_record_to_df(record: dict) -> pd.DataFrame:
    """Convert a single-customer dict (from Streamlit form) to a 1-row DataFrame
    with derived features already added."""
    df = pd.DataFrame([record])
    # Ensure numeric types
    for col in ["Tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_derived_features(df)
    return df
