# -*- coding: utf-8 -*-

"""
backend/data_loader.py
======================
CSV ingestion, basic validation, and type coercion.
"""

from __future__ import annotations
import pandas as pd

DATA_PATH = "Telecom Customer Churn Analysis – Data Cleaning - Cleaned_Data.csv"

REQUIRED_COLS = {
    "Customer_ID", "Gender", "SeniorCitizen", "Partner", "Dependents",
    "Tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and lightly validate the cleaned churn CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with corrected dtypes.

    Raises
    ------
    FileNotFoundError
        If the CSV is not found at *path*.
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce numeric columns that may arrive as strings
    df["TotalCharges"]  = pd.to_numeric(df["TotalCharges"],  errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["Tenure"]         = pd.to_numeric(df["Tenure"],         errors="coerce")

    # Fill rare NaN from coercion
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    return df


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column summary: dtype, null count, nunique, sample."""
    rows = []
    for col in df.columns:
        rows.append({
            "Column":   col,
            "Dtype":    str(df[col].dtype),
            "Nulls":    df[col].isna().sum(),
            "Unique":   df[col].nunique(),
            "Sample":   df[col].dropna().iloc[0] if len(df[col].dropna()) else None,
        })
    return pd.DataFrame(rows)
