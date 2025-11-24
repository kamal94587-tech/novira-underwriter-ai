import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -----------------------------
# Load Model + Scaler + Features
# -----------------------------

MODEL_DIR = os.path.join(os.getcwd(), "model")

RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COL_PATH = os.path.join(os.getcwd(), "feature_columns.pkl")

# Load model files
risk_model = joblib.load(RISK_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COL_PATH)

# Ensure exactly 12 features expected by model
EXPECTED_FEATURE_COUNT = len(feature_columns)

# -----------------------------
# Realistic 12-Feature Generator
# -----------------------------

def generate_realistic_shipment():
    return {
        "distance_km": np.random.uniform(50, 2000),                 # transport distance
        "avg_temp_c": np.random.uniform(-5, 30),                    # avg temperature exposure
        "max_temp_c": np.random.uniform(0, 40),                     # peak temp
        "min_temp_c": np.random.uniform(-20, 10),                   # min temp
        "humidity_avg": np.random.uniform(20, 90),                  # avg humidity
        "shock_events": np.random.randint(0, 7),                    # excessive movement
        "route_risk_score": np.random.uniform(0, 1),                # AI route risk score
        "handling_score": np.random.uniform(0, 1),                  # warehouse handling quality
        "delay_hours": np.random.uniform(0, 72),                    # delays along route
        "package_type_score": np.random.uniform(0, 1),              # protective packaging
        "weather_risk_score": np.random.uniform(0, 1),              # weather risk
        "historical_incident_rate": np.random.uniform(0, 0.3)       # carrier risk profile
    }

# -----------------------------
# Risk Score Prediction Function
# -----------------------------

def predict_risk(input_dict):

    # Convert to model feature order
    row = [input_dict[feat] for feat in feature_columns]

    X = np.array(row).reshape(1, -1)

    # Confirm feature count matches model
    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Feature mismatch: model expects {EXPECTED_FEATURE_COUNT}, "
            f"but got {X.shape[1]}."
        )

    # Scale data
    X_scaled = scaler.transform(X)

    # Predict risk (0–1)
    risk_raw = risk_model.predict_proba(X_scaled)[0][1]
    risk_percent = round(risk_raw * 100, 2)

    # Premium suggestion (simple multiplier)
    suggested_premium = round(500 + (risk_percent * 15), 2)

    return risk_percent, suggested_premium

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Novira.ai — Underwriter AI", layout="wide")

st.title("🧠 Novira.ai — Underwriter Risk Scorecard (Enhanced)")

st.write("This version uses **12 features**, real model files, and realistic shipment generation.")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Generate Realistic Shipment")

    if st.button("⚡ Generate Test Shipment"):
        shipment = generate_realistic_shipment()
        st.session_state["shipment"] = shipment

    if "shipment" not in st.session_state:
        st.info("Click the button to generate a test shipment.")
    else:
        st.json(st.session_state["shipment"])

with col2:
    st.subheader("Results")

    if "shipment" in st.session_state:
        try:
            risk, premium = predict_risk(st.session_state["shipment"])

            st.success("Scoring complete.")

            st.metric("📊 Risk Index", f"{risk} / 100")
            st.metric("💰 Suggested Premium", f"${premium}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.divider()

st.write("### Model Feature Order")
st.code(feature_columns)
