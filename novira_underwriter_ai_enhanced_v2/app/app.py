import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap

# ---------------------------------------------------------
# 📌 LOAD ARTIFACTS
# ---------------------------------------------------------
ARTIFACT_DIR = "/mount/src/novira-underwriter-ai/novira_underwriter_ai_enhanced_v2/model"

MODEL_PATH = os.path.join(ARTIFACT_DIR, "risk_model.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")

# Load artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# ---------------------------------------------------------
# 📌 TITLE & HERO
# ---------------------------------------------------------
st.set_page_config(page_title="Novira.ai — Underwriter Risk Scorecard", layout="wide")

st.markdown("""
# 🧠 Novira.ai — Underwriter Risk Scorecard (Enhanced v2)
### Real-time risk scoring, explainability, eligibility & premium estimation.
---
""")

# ---------------------------------------------------------
# 📌 INPUT FORM
# ---------------------------------------------------------
st.subheader("📦 Shipment Details")

with st.form("score_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        weight = st.number_input("Weight (kg)", 1, 50000, 1200)
        distance = st.number_input("Distance (km)", 1, 20000, 850)
        temp = st.slider("Temperature (°C)", -20, 50, 5)

    with col2:
        humidity = st.slider("Humidity (%)", 0, 100, 45)
        shocks = st.number_input("Shock Events", 0, 50, 2)
        route_risk = st.slider("Route Risk Score", 1, 10, 4)

    with col3:
        packaging_score = st.slider("Packaging Quality (1–10)", 1, 10, 7)
        past_claims = st.number_input("Past Claims", 0, 20, 1)
        carrier_score = st.slider("Carrier Reliability (1–10)", 1, 10, 8)

    # 🔥 NEW FEATURES FOR FULL VERSION
    traffic_risk = st.slider("Traffic Density Risk (1–10)", 1, 10, 5)
    port_delay_risk = st.slider("Port Delay Probability (1–10)", 1, 10, 4)

    submitted = st.form_submit_button("⚡ Score Shipment")


# ---------------------------------------------------------
# 📌 SCORE + SHAP + PREMIUM CALC
# ---------------------------------------------------------
if submitted:

    st.markdown("## 📊 Results")

    # Build feature row
    row = pd.DataFrame([[
        weight,
        distance,
        temp,
        humidity,
        shocks,
        route_risk,
        packaging_score,
        past_claims,
        carrier_score,
        traffic_risk,
        port_delay_risk
    ]], columns=feature_columns)

    # Scale
    X_scaled = scaler.transform(row)

    # Predict risk index (0–100)
    risk_prob = model.predict_proba(X_scaled)[0][1]
    risk_index = round(risk_prob * 100, 1)

    # Eligibility rule
    eligibility = "Approve" if risk_index < 60 else "Review" if risk_index < 80 else "Decline"
    elig_color = "#30C955" if eligibility == "Approve" else "#F5C542" if eligibility == "Review" else "#EF4444"

    # Premium estimation model
    base_price = 500
    premium = base_price * (1 + (risk_index / 100))
    premium = round(premium, 2)

    # ---------------------------------------------------------
    # 📌 Display Results UI
    # ---------------------------------------------------------
    st.markdown(f"""
    <div style="padding:20px;background:#111827;border-radius:12px;">
        <h2 style="color:white;margin-bottom:10px;">Risk Index</h2>
        <div style="font-size:42px;font-weight:900;color:#60A5FA;">{risk_index} / 100</div>
        <br>

        <h3 style="color:white;margin-bottom:5px;">Eligibility</h3>
        <div style="background:{elig_color};padding:8px 18px;color:white;
            display:inline-block;border-radius:8px;font-weight:800;font-size:20px;">
            {eligibility}
        </div>

        <br><br>
        <h3 style="color:white;margin-bottom:5px;">Premium Estimate</h3>
        <div style="font-size:32px;font-weight:800;color:#FBBF24;">${premium}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 📌 SHAP EXPLAINABILITY
    # ---------------------------------------------------------
    st.markdown("### 🔍 SHAP Explainability")

    with st.spinner("Generating explainability visuals..."):
        explainer = shap.LinearExplainer(model, X_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_scaled)

        fig = shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            row.iloc[0],
            show=False
        )
        st.pyplot(fig)

    # ---------------------------------------------------------
    # 📌 DEBUG SECTION
    # ---------------------------------------------------------
    with st.expander("🛠 Debug: feature row"):
        st.write(row)

