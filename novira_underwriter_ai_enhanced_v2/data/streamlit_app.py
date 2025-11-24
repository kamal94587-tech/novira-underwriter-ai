import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Novira.ai — Underwriter Risk Scorecard (Enhanced)", layout="wide")

st.title("🧠 Novira.ai — Underwriter Risk Scorecard (Enhanced)")
st.write("This enhanced version calculates a synthetic shipment, predicts risk, and provides premium suggestions.")

MODEL_PATH = "./model/model.pkl"
SCALER_PATH = "./model/scaler.pkl"
FEATURE_COLS_PATH = "./feature_columns.pkl"

@st.cache_data(show_spinner=False)
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, scaler, feature_cols

model, scaler, feature_columns = load_artifacts()

def generate_synthetic_shipment():
    shipment = {
        "temp_avg": np.random.uniform(28, 65),
        "temp_max": np.random.uniform(40, 75),
        "humidity_avg": np.random.uniform(30, 90),
        "shock_events": np.random.randint(0, 8),
        "tilt_events": np.random.randint(0, 5),
        "duration_hours": np.random.uniform(10, 300),
        "distance_km": np.random.uniform(50, 2500),
        "lane_risk_score": np.random.uniform(0.1, 0.9),
        "package_density": np.random.uniform(0.1, 1.0),
        "provider_reliability": np.random.uniform(0.5, 1.0),
        "weather_severity_index": np.random.uniform(0.0, 1.0),
        "historical_claim_prob": np.random.uniform(0.01, 0.25)
    }
    return shipment

def predict(model, scaler, record, feature_cols):
    df = pd.DataFrame([record])
    df = df[feature_cols]
    X = scaler.transform(df)
    risk = model.predict_proba(X)[0][1]
    return risk

def premium_from_risk(r):
    base_price = 35
    multiplier = 1 + (r * 2.5)
    return round(base_price * multiplier, 2)

st.subheader("🔥 Test Shipment (Auto-Generated)")

if st.button("⚡ Score test shipment"):
    shipment = generate_synthetic_shipment()
    st.write("### 📦 Synthetic Shipment Generated")
    st.json(shipment)

    if model is None:
        st.error("❌ Model & scaler not found. Upload your real artifacts into `/model/` folder.")
    else:
        risk_score = predict(model, scaler, shipment, feature_columns)
        premium = premium_from_risk(risk_score)

        st.success("Risk score calculated successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🔮 Risk Score", f"{risk_score:.3f}")

        with col2:
            st.metric("💰 Suggested Premium", f"${premium}")

        st.write("---")

        st.write("### 📊 Interpretation")
        if risk_score < 0.25:
            st.success("🟢 **LOW RISK** — Good shipment profile.")
        elif risk_score < 0.60:
            st.warning("🟡 **MEDIUM RISK** — Some factors need monitoring.")
        else:
            st.error("🔴 **HIGH RISK** — High risk of loss or delay.")

st.write("---")

st.info("Upload real model artifacts into `/model/model.pkl` and `/model/scaler.pkl` when ready.")
