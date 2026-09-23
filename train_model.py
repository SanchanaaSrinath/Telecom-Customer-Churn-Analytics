# -*- coding: utf-8 -*-
"""
train_model.py
==============
End-to-end ML pipeline for Telecom Customer Churn prediction.

Steps
-----
1. Load & validate cleaned CSV
2. Feature engineering & preprocessing
3. SMOTE oversampling to handle class imbalance
4. Train / evaluate three classifiers
5. Select best model by F1-score (macro)
6. Serialize artefacts -> models/churn_model.pkl  +  models/preprocessor.pkl
7. Save evaluation plots -> reports/
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend - safe for CI / server
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "Telecom Customer Churn Analysis \u2013 Data Cleaning - Cleaned_Data.csv"
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    churn_dist = df["Churn"].value_counts(normalize=True).to_dict()
    print(f"[load]  Shape: {df.shape}  |  Churn rate: {churn_dist}")
    return df


# ==============================================================================
# 2. PREPROCESSING
# ==============================================================================
def preprocess(df: pd.DataFrame):
    """Return X (DataFrame), y (Series), preprocessor, num_cols, cat_cols."""
    df = df.copy()

    # Drop non-predictive ID column
    df.drop(columns=["Customer_ID"], inplace=True, errors="ignore")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # TotalCharges may be stored as string in some CSV exports
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Derived features
    df["AvgMonthlyCharge"] = df["TotalCharges"] / df["Tenure"].replace(0, 1)
    df["HasMultipleServices"] = (
        (df["OnlineSecurity"]   == "Yes").astype(int)
        + (df["OnlineBackup"]   == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"]    == "Yes").astype(int)
        + (df["StreamingTV"]    == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="passthrough",
    )

    return X, y, preprocessor, num_cols, cat_cols


# ==============================================================================
# 3. PLOTS
# ==============================================================================
def plot_confusion_matrix(y_test, y_pred, name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    safe_name = name.replace(" ", "_")
    fname = os.path.join(REPORTS_DIR, f"cm_{safe_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved {fname}")


def plot_roc_curves(results: dict, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fname = os.path.join(REPORTS_DIR, "roc_curves.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved {fname}")


def plot_churn_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    churn_counts = df["Churn"].value_counts()
    axes[0].bar(["No Churn", "Churn"], churn_counts.values,
                color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Churn Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

    no_churn = df[df["Churn"] == "No"]["Tenure"]
    churned  = df[df["Churn"] == "Yes"]["Tenure"]
    axes[1].hist(no_churn, bins=30, alpha=0.6, label="No Churn", color="#4C72B0")
    axes[1].hist(churned,  bins=30, alpha=0.6, label="Churn",    color="#DD8452")
    axes[1].set_title("Tenure Distribution by Churn")
    axes[1].set_xlabel("Tenure (months)")
    axes[1].legend()

    plt.tight_layout()
    fname = os.path.join(REPORTS_DIR, "churn_distribution.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved {fname}")


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    print("\n" + "=" * 55)
    print("  TELECOM CHURN - MODEL TRAINING PIPELINE")
    print("=" * 55)

    # ── Load ──────────────────────────────────────────────────
    df = load_data(DATA_PATH)
    plot_churn_distribution(df)

    # ── Preprocess ────────────────────────────────────────────
    X, y, preprocessor, num_cols, cat_cols = preprocess(df)
    print(f"[prep]  X shape: {X.shape}  |  "
          f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    # ── Train / test split ────────────────────────────────────
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit preprocessor on train only, then transform both splits
    X_train_t = preprocessor.fit_transform(X_train_raw, y_train_raw)
    X_test_t  = preprocessor.transform(X_test_raw)

    # ── Impute any residual NaNs produced by OHE / passthrough ──
    # (median for numeric columns, most_frequent for any remaining object-
    #  encoded slots; after OHE everything is numeric so 'median' covers all)
    imputer = SimpleImputer(strategy="median")
    X_train_t = imputer.fit_transform(X_train_t)
    X_test_t  = imputer.transform(X_test_t)
    print(f"[impute] NaN in train after imputation: "
          f"{int(np.isnan(X_train_t).sum())}")

    # ── SMOTE on train only (no leakage into test) ────────────
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_t, y_train_raw)
    bal_dist = pd.Series(y_train_bal).value_counts().to_dict()
    print(f"[smote] After resampling -> {bal_dist}")

    # Save fitted preprocessor + imputer
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(imputer,      os.path.join(MODELS_DIR, "imputer.pkl"))
    print(f"[save]  Preprocessor -> {MODELS_DIR}/preprocessor.pkl")
    print(f"[save]  Imputer      -> {MODELS_DIR}/imputer.pkl")

    # ── Classifiers ───────────────────────────────────────────
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            eval_metric="logloss",
            scale_pos_weight=3,
            random_state=42,
            verbosity=0,
        ),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\n{'-' * 55}")
        print(f"  Training: {name}")
        print(f"{'-' * 55}")
        clf.fit(X_train_bal, y_train_bal)
        y_pred  = clf.predict(X_test_t)
        y_proba = clf.predict_proba(X_test_t)[:, 1]
        f1  = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(y_test, y_proba)
        print(classification_report(y_test, y_pred,
                                    target_names=["No Churn", "Churn"]))
        print(f"  ROC-AUC : {auc:.4f}   |   F1-macro : {f1:.4f}")
        plot_confusion_matrix(y_test, y_pred, name)
        results[name] = {
            "clf": clf, "f1": f1, "auc": auc,
            "y_pred": y_pred, "y_proba": y_proba,
        }

    plot_roc_curves(results, y_test)

    # ── Select best model ─────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_clf  = results[best_name]["clf"]
    print(f"\n[best]  Best model: {best_name}  "
          f"(F1={results[best_name]['f1']:.4f}  "
          f"AUC={results[best_name]['auc']:.4f})")

    # ── Feature importance plot ────────────────────────────────
    if hasattr(best_clf, "feature_importances_"):
        ohe_features = (
            preprocessor
            .named_transformers_["cat"]
            .get_feature_names_out(cat_cols)
            .tolist()
        )
        all_features = num_cols + ohe_features
        importances  = best_clf.feature_importances_
        n = min(len(importances), len(all_features))
        imp_df = (
            pd.DataFrame({"Feature": all_features[:n], "Importance": importances[:n]})
            .sort_values("Importance", ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=imp_df, y="Feature", x="Importance",
                    palette="viridis", ax=ax)
        ax.set_title(f"Top-15 Feature Importances - {best_name}")
        plt.tight_layout()
        fname = os.path.join(REPORTS_DIR, "feature_importance.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"[plot]  Saved {fname}")

    # ── Serialize artefacts ───────────────────────────────────
    # Build the feature list in the same order the ColumnTransformer expects:
    # numeric columns first, then categorical columns, then derived columns.
    features = list(X_train_raw.columns)   # exact column order seen at fit time

    model_dict = {
        # Core inference objects
        "model":            best_clf,
        "preprocessor":     preprocessor,
        "imputer":          imputer,
        "features":         features,
        # Metadata / metrics
        "best_model_name":  best_name,
        "f1_score":         results[best_name]["f1"],
        "roc_auc":          results[best_name]["auc"],
        "num_cols":         num_cols,
        "cat_cols":         cat_cols,
    }

    model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
    joblib.dump(model_dict, model_path)
    print(f"[save]  model_dict -> {model_path}")
    print(f"        keys: {list(model_dict.keys())}")

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
