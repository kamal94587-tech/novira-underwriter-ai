# novira_underwriter_ai_enhanced_v2/streamlit_min.py
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Novira.ai — Underwriter (Minimal)", layout="wide")
st.title("Novira.ai — Underwriter (Minimal Loader)")
st.caption("This screen renders first. Click the button to load models and score.")

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

# Quick environment checks
cols = st.columns(3)
cols[0].metric("Python", f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}")
cols[1].metric("App folder", BASE.name)
cols[2].metric("Model files", ", ".join(p.name for p in MODEL_DIR.glob("*.pkl")) or "None")

# User inputs kept simple
st.subheader("Shipment Inputs")
c1, c2, c3 = st.columns(3)
declared_value = c1.number_input("Declared value (USD)", min_value=0, value=25000, step=1000)
distance_km    = c2.number_input("Distance (km)", min_value=0, value=1200, step=50)
carrier_score  = c3.slider("Carrier reliability (0–100)", 0, 100, 70)
c4, c5 = st.columns(2)
route_risk     = c4.slider("Route risk (0–1)", 0.0, 1.0, 0.35, step=0.05)
temperature    = c5.selectbox("Temp control", ["none", "ambient", "refrigerated", "frozen"])

# Map to basic feature vector
def build_features(feature_columns):
    row = {col: 0.0 for col in feature_columns}
    # Try common names; fall back if columns differ
    mappings = {
        "declared_value": declared_value,
        "distance_km": distance_km,
        "carrier_score": carrier_score,
        "route_risk": route_risk,
        "temp_ambient": 1.0 if temperature == "ambient" else 0.0,
        "temp_refrigerated": 1.0 if temperature == "refrigerated" else 0.0,
        "temp_frozen": 1.0 if temperature == "frozen" else 0.0,
    }
    for k, v in mappings.items():
        if k in row:
            row[k] = v
    # Reasonable fallbacks
    if "value_usd" in row: row["value_usd"] = declared_value
    if "distance"  in row: row["distance"]  = distance_km
    return pd.DataFrame([row])

st.divider()
if st.button("▶️ Load model & Score"):
    with st.status("Loading model assets…", expanded=True) as status:
        try:
            import joblib
            feature_cols_path = MODEL_DIR / "feature_columns.pkl"
            scaler_path       = MODEL_DIR / "scaler.pkl"
            model_path        = MODEL_DIR / "risk_model.pkl"

            st.write("feature_columns.pkl:", feature_cols_path.exists())
            st.write("scaler.pkl:", scaler_path.exists())
            st.write("risk_model.pkl:", model_path.exists())

            feature_columns = joblib.load(feature_cols_path)
            scaler          = joblib.load(scaler_path)
            model           = joblib.load(model_path)

            status.update(label="Preparing features…")
            X = build_features(feature_columns)

            # Scale if available
            try:
                X_scaled = scaler.transform(X)
            except Exception:
                X_scaled = X.values

            status.update(label="Scoring…")
            # Works for clf or reg; fall back to predict
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_scaled)[0,1])
                risk_index = int(round(proba * 100))
            else:
                pred = float(model.predict(X_scaled)[0])
                # clamp 0–100
                risk_index = int(round(max(0, min(100, pred))))

            # Simple eligibility + premium suggestion
            if risk_index < 35: elig = "Bind";    load = 0.85
            elif risk_index < 65: elig = "Review"; load = 1.0
            else: elig = "Decline";                load = 1.25
            base_rate = 0.0075  # 0.75%
            premium = declared_value * base_rate * load

            status.update(label="Done", state="complete")

        except Exception as e:
            st.error("Failed to load or score.")
            st.exception(e)
            st.code("".join(traceback.format_exc()))
        else:
            st.subheader("Result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Index (0–100)", risk_index)
            m2.metric("Eligibility", elig)
            m3.metric("Suggested Premium (USD)", f"{premium:,.2f}")

            with st.expander("Debug — features sent to model"):
                st.dataframe(X)
