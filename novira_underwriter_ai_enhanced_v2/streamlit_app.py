import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------
# Load real model artifacts
# ------------------------------------------------

MODEL_DIR = "model"

FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


@st.cache_resource
def load_artifacts():
    try:
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return feature_cols, model, scaler
    except:
        return None, None, None


feature_columns, model, scaler = load_artifacts()


# ------------------------------------------------
# Synthetic shipment generator — 12 features
# ------------------------------------------------
def generate_synthetic_shipment():
    return pd.DataFrame([{
        "distance_km": np.random.uniform(100, 2000),
        "duration_hrs": np.random.uniform(5, 72),
        "num_scans": np.random.randint(1, 20),
        "avg_temp": np.random.uniform(-5, 25),
        "min_temp": np.random.uniform(-10, 10),
        "max_temp": np.random.uniform(10, 35),
        "avg_humidity": np.random.uniform(30, 90),
        "shock_events": np.random.randint(0, 8),
        "route_risk_score": np.random.uniform(0, 1),
        "carrier_score": np.random.uniform(0, 1),
        "cargo_value_usd": np.random.uniform(5000, 500000),
        "season_risk": np.random.uniform(0, 1)
    }])


# ------------------------------------------------
# Prediction
# ------------------------------------------------
def predict_risk(df):
    if model is None or scaler is None or feature_columns is None:
        return None, None

    df = df[feature_columns]
    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled)[0][1]
    return round(proba * 100, 2), ("Decline" if proba > 0.5 else "Approve")


# ------------------------------------------------
# UI
# ------------------------------------------------

st.title("📦 Novira.ai — Underwriter Risk Scorecard (Enhanced)")

st.write("Click to score a synthetic shipment using the REAL model.")

if st.button("⚡ Score test shipment"):
    df = generate_synthetic_shipment()
    risk, decision = predict_risk(df)

    if risk is None:
        st.error("Model artifacts not found. Upload them to the /model folder.")
    else:
        st.subheader("Results")

        # Progress bar
        st.progress(min(risk/100, 1.0))

        # Risk Score Display
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; margin-top:10px;">
                {risk} / 100
            </div>
        """, unsafe_allow_html=True)

        # Decision Pill
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
