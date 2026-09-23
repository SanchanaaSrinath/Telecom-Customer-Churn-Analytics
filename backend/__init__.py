"""backend/__init__.py"""
from .data_loader  import load_data
from .preprocessor import build_feature_matrix
from .predictor    import ChurnPredictor
from .eda_utils    import (
    churn_rate_by_category,
    numeric_summary,
    plot_churn_pie,
    plot_numeric_hist,
    plot_category_churn,
    plot_correlation_heatmap,
    plot_monthly_charges_boxplot,
)

__all__ = [
    "load_data",
    "build_feature_matrix",
    "ChurnPredictor",
    "churn_rate_by_category",
    "numeric_summary",
    "plot_churn_pie",
    "plot_numeric_hist",
    "plot_category_churn",
    "plot_correlation_heatmap",
    "plot_monthly_charges_boxplot",
]
