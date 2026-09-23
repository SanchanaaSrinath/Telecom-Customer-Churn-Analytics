# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📡",
    layout="wide"
)

# ── Load model artefact dict ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "churn_model.pkl")
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run 'python train_model.py' first.")
        st.stop()
    return joblib.load(model_path)

model_dict    = load_artifacts()
model         = model_dict["model"]
preprocessor  = model_dict["preprocessor"]
imputer       = model_dict.get("imputer")          # may be None for legacy runs
features      = model_dict["features"]

# ── Load dataset ──────────────────────────────────────────────────────────────
CSV_PATH = "Telecom Customer Churn Analysis \u2013 Data Cleaning - Cleaned_Data.csv"

@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

df = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📡 Churn Analytics")
page = st.sidebar.radio("Navigate to",
                        ["🏠 Home", "📊 EDA", "🎯 Predict", "📈 Model Insights"])


# ==============================================================================
# PAGE 1 — HOME
# ==============================================================================
if page == "🏠 Home":
    st.title("📡 Telecom Customer Churn Analytics System")
    st.markdown(
        "An end-to-end machine-learning system that **identifies at-risk customers** "
        "before they churn, enabling targeted retention strategies."
    )

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers",    f"{len(df):,}")
        col2.metric("Overall Churn Rate", f"{(df['Churn'] == 'Yes').mean():.1%}")
        col3.metric("Avg Tenure",         f"{df['Tenure'].mean():.0f} mo")
        col4.metric("Avg Monthly Charges",f"${df['MonthlyCharges'].mean():.0f}")

        st.subheader("📁 Dataset Snapshot")
        st.dataframe(df.head(10).astype(str), use_container_width=True)
    else:
        st.warning("Dataset not found.")


# ==============================================================================
# PAGE 2 — EDA
# ==============================================================================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            churn_counts = df["Churn"].value_counts()
            fig = px.pie(
                churn_counts,
                values=churn_counts.values,
                names=churn_counts.index,
                title="Churn Distribution",
                color_discrete_sequence=["#ff4b4b", "#00c0f2"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(
                df, x="Tenure", color="Churn",
                title="Tenure Distribution by Churn",
                color_discrete_sequence=["#ff4b4b", "#00c0f2"],
                barmode="overlay",
            )
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.box(
                df, x="Churn", y="MonthlyCharges", color="Churn",
                title="Monthly Charges vs Churn",
                color_discrete_sequence=["#00c0f2", "#ff4b4b"],
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            contract_churn = (
                df.groupby("Contract")["Churn"]
                .apply(lambda s: (s == "Yes").mean() * 100)
                .reset_index()
                .rename(columns={"Churn": "ChurnRate (%)"})
            )
            fig4 = px.bar(
                contract_churn, x="Contract", y="ChurnRate (%)",
                title="Churn Rate by Contract Type",
                color="ChurnRate (%)", color_continuous_scale="Reds",
            )
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Dataset not found.")


# ==============================================================================
# PAGE 3 — PREDICT
# ==============================================================================
elif page == "🎯 Predict":
    st.title("🎯 Real-Time Churn Risk Predictor")
    st.markdown("Fill in customer attributes below to evaluate churn probability.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender            = st.selectbox("Gender", ["Female", "Male"])
            senior            = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner           = st.selectbox("Partner", ["No", "Yes"])
            dependents        = st.selectbox("Dependents", ["No", "Yes"])
            tenure            = st.slider("Tenure (months)", 1, 72, 12)
            phone             = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines    = st.selectbox("Multiple Lines",
                                             ["No", "Yes", "No phone service"])

        with col2:
            internet          = st.selectbox("Internet Service",
                                             ["DSL", "Fiber optic", "No"])
            online_security   = st.selectbox("Online Security",
                                             ["No", "Yes", "No internet service"])
            online_backup     = st.selectbox("Online Backup",
                                             ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection",
                                             ["No", "Yes", "No internet service"])
            tech_support      = st.selectbox("Tech Support",
                                             ["No", "Yes", "No internet service"])
            streaming_tv      = st.selectbox("Streaming TV",
                                             ["No", "Yes", "No internet service"])

        with col3:
            streaming_movies  = st.selectbox("Streaming Movies",
                                             ["No", "Yes", "No internet service"])
            contract          = st.selectbox("Contract",
                                             ["Month-to-month", "One year", "Two year"])
            paperless         = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment           = st.selectbox("Payment Method", [
                                             "Electronic check", "Mailed check",
                                             "Bank transfer (automatic)",
                                             "Credit card (automatic)"])
            monthly           = st.number_input("Monthly Charges ($)",
                                                value=65.0, min_value=18.0, max_value=120.0)
            total             = st.number_input("Total Charges ($)",
                                                value=float(tenure * monthly))

        submit = st.form_submit_button("Calculate Churn Risk", type="primary",
                                       use_container_width=True)

    if submit:
        # Build input DataFrame using the exact column names the preprocessor was trained on
        input_data = pd.DataFrame([{
            "Gender":           gender,
            "SeniorCitizen":    "Yes" if senior == "Yes" else "No",
            "Partner":          partner,
            "Dependents":       dependents,
            "Tenure":           tenure,
            "PhoneService":     phone,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_protection,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment,
            "MonthlyCharges":   monthly,
            "TotalCharges":     total,
        }])

        # Add derived features (must match train_model.py preprocessing)
        input_data["TotalCharges"]        = pd.to_numeric(input_data["TotalCharges"], errors="coerce")
        input_data["AvgMonthlyCharge"]    = input_data["TotalCharges"] / max(tenure, 1)
        input_data["HasMultipleServices"] = (
            (input_data["OnlineSecurity"]   == "Yes").astype(int)
            + (input_data["OnlineBackup"]   == "Yes").astype(int)
            + (input_data["DeviceProtection"] == "Yes").astype(int)
            + (input_data["TechSupport"]    == "Yes").astype(int)
            + (input_data["StreamingTV"]    == "Yes").astype(int)
            + (input_data["StreamingMovies"] == "Yes").astype(int)
        )

        # Transform → impute → predict
        X_trans = preprocessor.transform(input_data[features])
        if imputer is not None:
            X_trans = imputer.transform(X_trans)
        prob = float(model.predict_proba(X_trans)[0][1])

        st.markdown("---")
        st.subheader("Prediction Result")

        col_a, col_b = st.columns(2)
        with col_a:
            if prob > 0.5:
                st.error(f"⚠️ High Churn Risk: {prob:.1%}")
                st.warning(
                    "**Recommendations:**\n"
                    "- Offer a long-term contract discount\n"
                    "- Add Tech Support or Online Security bundle\n"
                    "- Incentivise switch to auto-payment"
                )
            else:
                st.success(f"✅ Low Churn Risk: {prob:.1%}")
                st.info("**Recommendation:** Maintain standard engagement programme.")

        with col_b:
            import plotly.graph_objects as go
            color = "#dc2626" if prob > 0.5 else "#16a34a"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0, 35],  "color": "#dcfce7"},
                        {"range": [35, 65], "color": "#fef3c7"},
                        {"range": [65, 100],"color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                }
            ))
            fig_g.update_layout(height=280)
            st.plotly_chart(fig_g, use_container_width=True)


# ==============================================================================
# PAGE 4 — MODEL INSIGHTS
# ==============================================================================
elif page == "📈 Model Insights":
    st.title("📈 Model Performance & Evaluation")
    st.markdown(
        f"Metrics for the trained **{model_dict.get('best_model_name', 'best')} model**."
    )

    col1, col2 = st.columns(2)
    col1.metric("F1-Score (macro)", f"{model_dict.get('f1_score', 0.0):.4f}")
    col2.metric("ROC-AUC",          f"{model_dict.get('roc_auc',  0.0):.4f}")

    st.markdown("---")

    # Show saved training plots if available
    report_imgs = [f for f in os.listdir("reports") if f.endswith(".png")] \
        if os.path.isdir("reports") else []

    if report_imgs:
        st.subheader("Training Report Plots")
        cols = st.columns(2)
        for i, img in enumerate(sorted(report_imgs)):
            with cols[i % 2]:
                st.image(
                    os.path.join("reports", img),
                    caption=img.replace("_", " ").replace(".png", ""),
                    use_container_width=True,
                )
    else:
        st.info("No report images found. Run `python train_model.py` to generate them.")
