# -*- coding: utf-8 -*-

"""
backend/eda_utils.py
====================
Reusable EDA and visualisation helpers consumed by app.py (Streamlit pages)
and train_model.py (static report plots).

All functions return Matplotlib Figure objects so callers can either:
  - display them in Streamlit via st.pyplot(fig)
  - save them to disk via fig.savefig(...)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {"No Churn": "#4C72B0", "Churn": "#DD8452"}
SEQ_PALETTE = "viridis"


# ══════════════════════════════════════════════════════════════════════════════
# Tabular helpers
# ══════════════════════════════════════════════════════════════════════════════

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric columns."""
    return df.select_dtypes(include="number").describe().T.round(2)


def churn_rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return churn rate (%) per category value for *col*."""
    tmp = df.copy()
    tmp["_churn"] = (tmp["Churn"] == "Yes").astype(int)
    grp = tmp.groupby(col)["_churn"].agg(["mean", "count"]).reset_index()
    grp.columns = [col, "ChurnRate", "Count"]
    grp["ChurnRate"] = (grp["ChurnRate"] * 100).round(2)
    return grp.sort_values("ChurnRate", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib figures (for static reports & Streamlit st.pyplot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_churn_pie(df: pd.DataFrame) -> plt.Figure:
    counts = df["Churn"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(counts.values,
           labels=counts.index,
           autopct="%1.1f%%",
           colors=[COLORS.get(l, "#888") for l in counts.index],
           startangle=90)
    ax.set_title("Overall Churn Distribution")
    return fig


def plot_numeric_hist(df: pd.DataFrame, col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, grp in df.groupby("Churn"):
        ax.hist(grp[col].dropna(), bins=30, alpha=0.6,
                label=label, color=COLORS.get(label, "#888"))
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"{col} Distribution by Churn Status")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_category_churn(df: pd.DataFrame, col: str) -> plt.Figure:
    rate_df = churn_rate_by_category(df, col)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(rate_df[col].astype(str), rate_df["ChurnRate"],
                   color="#DD8452")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title(f"Churn Rate by {col}")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num_df = df.select_dtypes(include="number").copy()
    # Encode target if present
    if "Churn" in df.columns:
        num_df["Churn_enc"] = (df["Churn"] == "Yes").astype(int)
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig


def plot_monthly_charges_boxplot(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    data_no = df[df["Churn"] == "No"]["MonthlyCharges"].dropna()
    data_yes = df[df["Churn"] == "Yes"]["MonthlyCharges"].dropna()
    ax.boxplot([data_no, data_yes],
               labels=["No Churn", "Churn"],
               patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Monthly Charges by Churn Status")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Plotly figures (for interactive Streamlit tabs)
# ══════════════════════════════════════════════════════════════════════════════

def plotly_churn_pie(df: pd.DataFrame):
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    return px.pie(counts, names="Churn", values="Count",
                  title="Churn Distribution",
                  color="Churn",
                  color_discrete_map=COLORS)


def plotly_tenure_histogram(df: pd.DataFrame):
    return px.histogram(df, x="Tenure", color="Churn",
                        barmode="overlay", nbins=40,
                        title="Tenure Distribution by Churn",
                        color_discrete_map=COLORS,
                        opacity=0.7)


def plotly_monthly_charges(df: pd.DataFrame):
    return px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                  title="Monthly Charges vs Churn",
                  color_discrete_map=COLORS,
                  points="outliers")


def plotly_contract_churn(df: pd.DataFrame):
    rate_df = churn_rate_by_category(df, "Contract")
    return px.bar(rate_df, x="Contract", y="ChurnRate",
                  text="ChurnRate",
                  title="Churn Rate by Contract Type",
                  labels={"ChurnRate": "Churn Rate (%)"},
                  color="ChurnRate",
                  color_continuous_scale="OrRd")


def plotly_internet_churn(df: pd.DataFrame):
    rate_df = churn_rate_by_category(df, "InternetService")
    return px.bar(rate_df, x="InternetService", y="ChurnRate",
                  text="ChurnRate",
                  title="Churn Rate by Internet Service",
                  labels={"ChurnRate": "Churn Rate (%)"},
                  color="ChurnRate",
                  color_continuous_scale="Blues")


def plotly_payment_churn(df: pd.DataFrame):
    rate_df = churn_rate_by_category(df, "PaymentMethod")
    return px.bar(rate_df, x="ChurnRate", y="PaymentMethod",
                  orientation="h",
                  title="Churn Rate by Payment Method",
                  labels={"ChurnRate": "Churn Rate (%)"},
                  color="ChurnRate",
                  color_continuous_scale="Oranges",
                  text="ChurnRate")


def plotly_scatter_tenure_charges(df: pd.DataFrame):
    return px.scatter(df, x="Tenure", y="MonthlyCharges",
                      color="Churn", opacity=0.5,
                      title="Tenure vs Monthly Charges (coloured by Churn)",
                      color_discrete_map=COLORS)
