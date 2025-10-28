# Novira.ai — Underwriter Risk Scorecard (Minimal with Dummy Bootstrap)
# - If model pickles are missing, auto-generate small dummy artifacts so the app runs.
# - Keeps robust path resolution and on-page error visibility.

import os, sys, json, traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Novira.ai — Underwriter Risk Scorecard", page_icon="📊", layout="wide")
st.title("📊 Novira.ai — Underwriter Risk Scorecard (Minimal)")

# ---------------------------
# Path resolver (root or enhanced_v2/model)
# ---------------------------
BASE = Path(__file__).resolve().parent
MODEL_DIR1 = BASE / "novira_underwriter_ai_enhanced_v2" / "model"

def find_first(*rel_paths: str):
    for rel in rel_paths:
        p = (BASE / rel).resolve()
        if p.exists():
            return p
    return None

RISK_PKL   = find_first("risk_model.pkl", "novira_underwriter_ai_enhanced_v2/model/risk_model.pkl")
SCALER_PKL = find_first("scaler.pkl", "novira_underwriter_ai_enhanced_v2/model/scaler.pkl")
FEATS_PKL  = find_first("feature_columns.pkl", "novira_underwriter_ai_enhanced_v2/model/feature_columns.pkl")

with st.expander("🔍 Resolved paths & environment", expanded=False):
    st.json({
        "python": sys.version,
        "resolved_paths": {
            "risk_model.pkl": str(RISK_PKL) if RISK_PKL else None,
            "scaler.pkl": str(SCALER_PKL) if SCALER_PKL else None,
            "feature_columns.pkl": str(FEATS_PKL) if FEATS_PKL else None,
        }
    })

# ---------------------------
# Bootstrap dummy artifacts ONCE if missing
# ---------------------------
def bootstrap_dummy_models_if_missing():
    # If any of the three is missing, we create all three into enhanced_v2/model/
    needs_bootstrap = not (RISK_PKL and SCALER_PKL and FEATS_PKL)
    if not needs_bootstrap:
        return False, "All artifacts present — no bootstrap needed."

    try:
        MODEL_DIR1.mkdir(parents=True, exist_ok=True)

        # Deterministic synthetic training set
        rng = np.random.default_rng(42)
        feature_columns = [
            "shipment_value_usd", "distance_km", "days_in_transit",
            "is_international", "is_perishable", "temperature_req_c",
            "route_risk_score", "carrier_reliability", "past_claims_count",
            "package_fragility", "weather_severity_index", "port_congestion_index",
        ]
        n = 800
        X = pd.DataFrame({
            "shipment_value_usd": rng.normal(50000, 15000, n).clip(5000, 120000),
            "distance_km": rng.normal(1200, 600, n).clip(50, 6000),
            "days_in_transit": rng.normal(6, 2, n).clip(1, 30),
            "is_international": rng.integers(0, 2, n),
            "is_perishable": rng.integers(0, 2, n),
            "temperature_req_c": rng.normal(5, 8, n).clip(-40, 25),
            "route_risk_score": rng.normal(45, 20, n).clip(0, 100),
            "carrier_reliability": rng.normal(80, 10, n).clip(40, 100),
            "past_claims_count": rng.poisson(0.6, n).clip(0, 8),
            "package_fragility": rng.integers(0, 4, n),  # 0=low .. 3=high
            "weather_severity_index": rng.normal(20, 15, n).clip(0, 100),
            "port_congestion_index": rng.normal(30, 20, n).clip(0, 100),
        })

        # A simple synthetic risk label
        base = (
            0.002 * X["shipment_value_usd"]
            + 0.01  * X["days_in_transit"]
            + 0.4   * X["is_international"]
            + 0.6   * X["is_perishable"]
            + 0.02  * X["route_risk_score"]
            - 0.02  * X["carrier_reliability"]
            + 0.15  * X["past_claims_count"]
            + 0.1   * X["package_fragility"]
            + 0.01  * X["weather_severity_index"]
            + 0.01  * X["port_congestion_index"]
        )
        p = 1 / (1 + np.exp(-(base - 30) / 8))
        y = (p > 0.5).astype(int)  # binary risk

        # Scale + train a tiny classifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)

        # Save artifacts (always to enhanced_v2/model/)
        joblib.dump(feature_columns, MODEL_DIR1 / "feature_columns.pkl")
        joblib.dump(scaler,           MODEL_DIR1 / "scaler.pkl")
        joblib.dump(model,            MODEL_DIR1 / "risk_model.pkl")

        return True, f"Dummy artifacts created in {MODEL_DIR1}"
    except Exception as e:
        st.error("Bootstrap of dummy models failed:")
        st.exception(e)
        return False, "Bootstrap failed"

created, msg = bootstrap_dummy_models_if_missing()
if created:
    # Refresh our resolvers after creating files
    RISK_PKL   = find_first("risk_model.pkl", "novira_underwriter_ai_enhanced_v2/model/risk_model.pkl")
    SCALER_PKL = find_first("scaler.pkl", "novira_underwriter_ai_enhanced_v2/model/scaler.pkl")
    FEATS_PKL  = find_first("feature_columns.pkl", "novira_underwriter_ai_enhanced_v2/model/feature_columns.pkl")
    st.success(msg)

# ---------------------------
# Loading with visible errors
# ---------------------------
@st.cache_resource(show_spinner=False)
def _load(path: Path):
    return joblib.load(path)

def _require(path: Path, label: str):
    if not path:
        raise FileNotFoundError(f"{label} not found. Upload to repo root OR novira_underwriter_ai_enhanced_v2/model/")
    try:
        return _load(path)
    except Exception as e:
        raise RuntimeError(f"Failed loading {label} from {path}") from e

load_ok, model, scaler, feature_columns = True, None, None, None
for label, path_var in [("feature_columns.pkl", FEATS_PKL), ("scaler.pkl", SCALER_PKL), ("risk_model.pkl", RISK_PKL)]:
    try:
        obj = _require(path_var, label)
