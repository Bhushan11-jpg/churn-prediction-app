# Telco Customer Churn - Streamlit Deployment App
# Run locally with:  streamlit run app.py
# Deploy free on: https://share.streamlit.io  or  Hugging Face Spaces


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# Load trained artifacts (produced by churn_prediction.py)

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
st.caption("Predicts churn risk for a customer and explains WHY, using SHAP.")

# Sidebar - Customer Inputs

st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

predict_btn = st.sidebar.button("Predict Churn Risk", type="primary")

# Build a single-row dataframe matching the training pipeline

def preprocess_input(raw: dict) -> pd.DataFrame:
    row = pd.DataFrame([raw])

    # Binary mapping
    for col, mapping in binary_map.items():
        row[col] = row[col].map(mapping)

    # One-hot encode multi-category columns, then align to training columns
    row = pd.get_dummies(row, columns=multi_cat_cols, drop_first=True)
    row = row.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric columns with the SAME scaler used in training
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

# Main panel - Prediction + SHAP explanation

col1, col2 = st.columns([1, 1.4])

if predict_btn:
    X_input = preprocess_input(raw_input)
    churn_proba = model.predict_proba(X_input)[0, 1]
    churn_pred = "Yes" if churn_proba >= 0.5 else "No"

    with col1:
        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{churn_proba*100:.1f}%")
        if churn_proba >= 0.6:
            st.error(f"⚠️ HIGH RISK — Predicted Churn: {churn_pred}")
        elif churn_proba >= 0.3:
            st.warning(f"🟠 MEDIUM RISK — Predicted Churn: {churn_pred}")
        else:
            st.success(f"✅ LOW RISK — Predicted Churn: {churn_pred}")

        st.markdown("**Business Recommendation:**")
        if churn_proba >= 0.6:
            st.write("Immediately offer a retention discount or contract upgrade call.")
        elif churn_proba >= 0.3:
            st.write("Add to monitoring list; consider a proactive check-in email.")
        else:
            st.write("No action needed — customer is stable.")

    with col2:
        st.subheader("Why this prediction? (SHAP Explanation)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            # Handle both binary-classifier output formats
            sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

            shap_df = pd.DataFrame({
                "Feature": X_input.columns,
                "SHAP Value": sv
            }).sort_values("SHAP Value", key=abs, ascending=False).head(8)

            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ["#F44336" if v > 0 else "#4CAF50" for v in shap_df["SHAP Value"]]
            ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
            ax.set_xlabel("Impact on Churn Prediction (red = increases risk, green = decreases risk)")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info("SHAP explanation is only available for tree-based models (Random Forest / XGBoost).")
else:
    st.info("⬅️ Fill in the customer details in the sidebar and click **Predict Churn Risk**.")

# Footer - project context for anyone viewing the deployed app

st.markdown("---")
st.caption(
    "Built with scikit-learn + XGBoost + SHAP. "
    "Model selected via 5-fold cross-validation and GridSearchCV hyperparameter tuning "
    "across Logistic Regression, Decision Tree, Random Forest, Naive Bayes, and XGBoost."
)
