# Telecom Customer Churn Prediction System

An end-to-end Machine Learning web application designed to predict customer churn risk in telecom networks, built with Streamlit and Scikit-Learn.

## Features
- **Interactive Dashboard**: Explore key performance indicators and EDA distributions.
- **Real-Time Prediction**: Input customer parameters to compute instant churn probabilities.
- **Handling Class Imbalance**: Integrated SMOTE oversampling to maintain high recall on churning customers.
- **Automated Preprocessing**: Numerical imputation and categorical encoding via Scikit-Learn `ColumnTransformer`.
- **Model Artefact Dict**: All pipeline components (model, preprocessor, imputer, metadata) bundled in a single `models/churn_model.pkl`.

## Directory Structure

```
Telecom Cust Churn-Project/
├── app.py                    # Streamlit multi-page dashboard
├── train_model.py            # ML training pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── backend/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── predictor.py
│   └── eda_utils.py
├── models/
│   └── churn_model.pkl       # Artefact dict: model, preprocessor, imputer, metadata
├── reports/                  # Training plots (PNG)
└── Telecom_Churn_Project_Report.docx
```

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Telecom-Cust-Churn-Project.git
cd Telecom-Cust-Churn-Project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```
Generates `models/churn_model.pkl` and plots in `reports/`. Takes ~60–90 seconds.

### 5. Launch the dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | KPI cards, dataset snapshot |
| **EDA** | Churn distribution, tenure histogram, monthly charges boxplot, contract churn rate |
| **Predict** | Real-time single-customer churn probability + gauge chart |
| **Model Insights** | F1 / AUC metrics + training report plots |

## Models Evaluated

| Model | F1 Macro | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~0.71 | ~0.84 |
| Random Forest | ~0.70 | ~0.82 |
| XGBoost | ~0.70 | ~0.82 |

Best model selected by macro F1-score on held-out 20% test set (SMOTE applied to training split only).

## Dataset

Cleaned IBM Telco Customer Churn dataset — 1,500 records, 21 features.  
Target: `Churn` (Yes / No).

## License

MIT

Dataset

https://docs.google.com/spreadsheets/d/1vr6aRM84fabb-c6RDUAIJKt02c2TEcvBHVm5mKqKiYU/edit?gid=1965156557#gid=1965156557