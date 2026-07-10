import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

# custom styling
st.markdown("""
<style>
.main { background-color: #f8f9fb; }
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #eaeaea;
    border-radius: 12px;
    padding: 15px;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# load model and preprocessing objects saved from training
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    binary_map = joblib.load("binary_map.pkl")
    multi_cat_cols = joblib.load("multi_cat_cols.pkl")
    num_cols = joblib.load("num_cols.pkl")
    return model, scaler, feature_columns, binary_map, multi_cat_cols, num_cols

model, scaler, feature_columns, binary_map, multi_cat_cols, num_cols = load_artifacts()

st.title("📉 Telco Customer Churn Predictor")
st.caption("Fill in customer details and get an instant churn risk score with an explanation.")

tab1, tab2 = st.tabs(["🔮 Predict", "📊 About the Model"])

with tab1:
    st.markdown("### Customer Details")
    d_col, s_col, b_col = st.columns(3)

    with d_col:
        st.markdown("**👤 Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with s_col:
        st.markdown("**📡 Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with b_col:
        st.markdown("**💳 Billing**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

    predict_btn = st.button("🔍 Predict Churn Risk", type="primary")

    def preprocess_input(raw: dict) -> pd.DataFrame:
        row = pd.DataFrame([raw])

        for col, mapping in binary_map.items():
            row[col] = row[col].map(mapping)

        # pd.get_dummies on a single row is unreliable, so build one-hot columns manually
        for col in multi_cat_cols:
            val = row.at[0, col]
            dummy_col = f"{col}_{val}"
            row[dummy_col] = 1 if dummy_col in feature_columns else 0
            row.drop(columns=[col], inplace=True)

        row = row.reindex(columns=feature_columns, fill_value=0)

        # scale numeric columns using the same scaler from training
        row[num_cols] = scaler.transform(row[num_cols])
        return row

    raw_input = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }

    st.markdown("---")
    st.markdown("### Result")

    if predict_btn:
        X_input = preprocess_input(raw_input)
        churn_proba = model.predict_proba(X_input)[0, 1]
        churn_pred = "Yes" if churn_proba >= 0.5 else "No"

        if churn_proba >= 0.6:
            risk_label, bar_color = "HIGH RISK", "#F44336"
        elif churn_proba >= 0.3:
            risk_label, bar_color = "MEDIUM RISK", "#FF9800"
        else:
            risk_label, bar_color = "LOW RISK", "#4CAF50"

        gauge_col, info_col = st.columns([1, 1.3])

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_proba * 100,
                number={"suffix": "%"},
                title={"text": f"Churn Probability — {risk_label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": bar_color},
                    "steps": [
                        {"range": [0, 30], "color": "#e8f5e9"},
                        {"range": [30, 60], "color": "#fff3e0"},
                        {"range": [60, 100], "color": "#ffebee"},
                    ],
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with info_col:
            m1, m2 = st.columns(2)
            m1.metric("Predicted Churn", churn_pred)
            m2.metric("Risk Level", risk_label)

            st.markdown("**Business Recommendation**")
            if churn_proba >= 0.6:
                st.error("Immediately offer a retention discount or contract upgrade call.")
            elif churn_proba >= 0.3:
                st.warning("Add to monitoring list; consider a proactive check-in email.")
            else:
                st.success("No action needed — customer is stable.")

        st.markdown("**Why this prediction? (SHAP)**")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

            shap_df = pd.DataFrame({
                "Feature": X_input.columns,
                "SHAP Value": sv
            }).sort_values("SHAP Value", key=abs, ascending=False).head(8)
            shap_df["Effect"] = np.where(shap_df["SHAP Value"] > 0, "Increases risk", "Decreases risk")

            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP Value"],
                y=shap_df["Feature"],
                orientation="h",
                marker_color=["#F44336" if v > 0 else "#4CAF50" for v in shap_df["SHAP Value"]],
                text=shap_df["Effect"],
            ))
            fig_shap.update_layout(
                height=320, margin=dict(t=20, b=20, l=10, r=10),
                xaxis_title="Impact on prediction", yaxis={"autorange": "reversed"}
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception:
            st.info("SHAP explanation is only available for tree-based models (Random Forest / XGBoost).")
    else:
        st.info("⬆️ Fill in the customer details above and click **Predict Churn Risk** to see results here.")

with tab2:
    st.markdown("### How this model was built")
    st.write(
        "Five algorithms were trained as a baseline comparison — Logistic Regression, "
        "Decision Tree, Random Forest, Naive Bayes, and XGBoost — using 5-fold cross-validation, "
        "SMOTE for class imbalance, and feature scaling. The best-performing model was then "
        "tuned further with GridSearchCV before being deployed here."
    )
    st.markdown("**Metrics used:** Accuracy, Precision, Recall, F1-Score, ROC-AUC")
    st.markdown("**Explainability:** SHAP values show which features push a prediction toward or away from churn.")
