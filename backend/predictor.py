# -*- coding: utf-8 -*-

"""
backend/predictor.py
====================
Load serialised artefacts and run inference for the Streamlit UI.
"""

from __future__ import annotations
import os
import joblib
import pandas as pd
import numpy as np

from .preprocessor import single_record_to_df

MODELS_DIR = "models"


class ChurnPredictor:
    """Wraps the trained sklearn classifier + preprocessor for live predictions.

    Usage
    -----
    >>> p = ChurnPredictor()
    >>> p.predict_single({"Tenure": 12, "MonthlyCharges": 65.5, ...})
    {'churn_probability': 0.73, 'prediction': 'Churn', 'risk_level': 'High'}
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self._models_dir = models_dir
        self._model      = None
        self._prep       = None
        self._imputer    = None
        self._meta       = None

    # ── Lazy-load (Streamlit caching handles this at the app layer) ──────────
    def _load(self):
        if self._model is not None:
            return
        model_path   = os.path.join(self._models_dir, "churn_model.pkl")
        prep_path    = os.path.join(self._models_dir, "preprocessor.pkl")
        imputer_path = os.path.join(self._models_dir, "imputer.pkl")
        meta_path    = os.path.join(self._models_dir, "model_meta.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run `python train_model.py` first."
            )

        self._model   = joblib.load(model_path)
        self._prep    = joblib.load(prep_path)
        self._imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
        self._meta    = joblib.load(meta_path)    if os.path.exists(meta_path)    else {}

    # ── Public API ────────────────────────────────────────────────────────────
    @property
    def meta(self) -> dict:
        self._load()
        return self._meta

    def predict_single(self, record: dict) -> dict:
        """Predict churn for a single customer record.

        Parameters
        ----------
        record : dict
            Raw customer feature values (matching the training schema).

        Returns
        -------
        dict
            Keys: churn_probability, prediction, risk_level
        """
        self._load()
        df = single_record_to_df(record)
        X  = self._prep.transform(df)
        if self._imputer is not None:
            X = self._imputer.transform(X)
        prob = float(self._model.predict_proba(X)[0, 1])
        label = "Churn" if prob >= 0.5 else "No Churn"
        if prob < 0.35:
            risk = "Low"
        elif prob < 0.65:
            risk = "Medium"
        else:
            risk = "High"
        return {"churn_probability": round(prob, 4),
                "prediction": label,
                "risk_level": risk}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run batch predictions on a DataFrame (must include feature columns)."""
        self._load()
        X = self._prep.transform(df)
        if self._imputer is not None:
            X = self._imputer.transform(X)
        probs  = self._model.predict_proba(X)[:, 1]
        labels = ["Churn" if p >= 0.5 else "No Churn" for p in probs]
        out = df.copy()
        out["ChurnProbability"] = probs
        out["Prediction"]       = labels
        return out
