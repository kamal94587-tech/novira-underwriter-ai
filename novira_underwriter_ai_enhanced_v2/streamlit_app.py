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

if st.button("⚡ Score test shipment"):
    df = generate_synthetic_shipment()
    risk, decision = predict_risk(df)

    if risk is None:
        st.error("Model artifacts not found. Upload them to the /model folder.")
    else:
        st.subheader("Results")

        # Risk bar
        st.progress(min(risk/100, 1.0))

        # Show Risk Score
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; margin-top:10px;">
                {risk} / 100
            </div>
        """, unsafe_allow_html=True)

        # Decision pill
        pill_color = "#22cc55" if decision == "Approve" else "#cc2255"
        st.markdown(f"""
            <div style="
                display:inline-block;
                padding:6px 16px;
                border-radius:999px;
                background:{pill_color};
                color:white;
                font-weight:700;
                font-size:14px;
                margin-top:6px;
            ">
                {decision}
            </div>
        """, unsafe_allow_html=True)
