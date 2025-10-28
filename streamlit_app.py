# Novira.ai — Underwriter Risk Scorecard (Minimal, Production-Safe)
# - Robust model resolution (root OR novira_underwriter_ai_enhanced_v2/model/)
# - Clear on-page diagnostics and exceptions (no hidden failures)
# - Simple "Score test shipment" button meeting acceptance criteria

import sys, json, traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Novira.ai — Underwriter Risk Scorecard", page_icon="📊", layout="wide")
st.title("📊 Novira.ai — Underwriter Risk Scorecard (Minimal)")

# ---------------------------
# Path resolver (non-negotiable)
# ---------------------------
BASE = Path(__file__).resolve().parent

def find_first(*rel_paths: str):
    """Return the first existing absolute Path for any of the given relative paths (relative to BASE)."""
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
# Load artifacts with visible errors
# ---------------------------
@st.cache_resource(show_spinner=False)
def _load_artifact(path: Path):
    return joblib.load(path)

def _fail_box(title: str, err: Exception):
    st.error(title)
    st.code("".join(traceback.format_exception_only(type(err), err)))

def _require(path: Path, label: str) -> object:
    if not path:
        raise FileNotFoundError(f"{label} not found. Upload to repo root OR novira_underwriter_ai_enhanced_v2/model/")
    try:
        return _load_artifact(path)
    except Exception as e:
        raise RuntimeError(f"Failed loading {label} from {path}") from e

# Try to load; surface any error inline
model = scaler = feature_columns = None
load_ok = True
try:
    feature_columns = _require(FEATS_PKL, "feature_columns.pkl")
except Exception as e:
    load_ok = False
    _fail_box("feature_columns.pkl load error", e)

try:
    scaler = _require(SCALER_PKL, "scaler.pkl")
except Exception as e:
    load_ok = False
    _fail_box("scaler.pkl load error", e)

try:
    model = _require(RISK_PKL, "risk_model.pkl")
except Exception as e:
    load_ok = False
    _fail_box("risk_model.pkl load error", e)

if not load_ok:
    st.warning("Upload the missing files (see instructions below) and use ⋮ → Clear cache → Reboot.", icon="⚠️")

# ---------------------------
# Feature row builder (safe default)
# ---------------------------
def build_test_row(cols):
    """
    Construct a single-row DataFrame matching feature_columns with safe defaults.
    - numeric-like columns -> 0
    - otherwise -> 0 as well (assumes one-hot or encoded; keeps dtype numeric)
    This is intentionally minimal so the pipeline accepts the shape.
    """
    if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
        names = list(cols)
    else:
        # Sometimes feature_columns is a dict with 'columns' key
        names = list(cols.get("columns", [])) if isinstance(cols, dict) else []

    data = {c: 0 for c in names}
    # Helpful nudges for common names (optional tweaks that won’t break unknown schemas)
    for key in data.keys():
        lk = key.lower()
        if "value" in lk or "amount" in lk or "weight" in lk or "volume" in lk:
            data[key] = 1
        if "international" in lk or "hazard" in lk or "perishable" in lk:
            data[key] = 0
        if "days" in lk or "duration" in lk or "transit" in lk:
            data[key] = 3
        if "temperature" in lk or "temp" in lk:
            data[key] = 5
    return pd.DataFrame([data], columns=names)

# ---------------------------
# Scoring helpers
# ---------------------------
def to_0_100(x: float) -> float:
    try:
        return float(np.clip(x, 0, 100))
    except Exception:
        return 0.0

def predict_risk_index(model, X):
    """Return a 0–100 risk index using predict_proba if available; else predict; clamp to [0,100]."""
    # Prefer predict_proba positive class
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # handle binary [n, 2] or [n] shapes
        if isinstance(proba, (list, tuple, np.ndarray)):
            proba = np.array(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return to_0_100(proba[0, 1] * 100.0)
        elif proba.ndim == 1:
            return to_0_100(proba[0] * 100.0)

    # Fallback: raw prediction scaled/clamped
    y = model.predict(X)
    if isinstance(y, (list, tuple, np.ndarray)):
        y = float(np.array(y).ravel()[0])
    return to_0_100(y)

def eligibility_from_risk(risk_index: float) -> str:
    # Simple underwriting policy: lower risk is better
    if risk_index < 40:    return "Bind"
    if risk_index < 70:    return "Review"
    return "Decline"

def suggested_premium_usd(risk_index: float, base_usd: float = 1200.0) -> float:
    # Minimal, monotonic pricing curve — tweak later as needed
    # Example: 0% risk -> 1200; 100% risk -> 1200 * (1 + 100/150) ≈ 2000
    factor = 1.0 + (risk_index / 150.0)
    return float(round(base_usd * factor, 2))

# ---------------------------
# UI — Score a test shipment
# ---------------------------
left, right = st.columns([0.6, 0.4])

with left:
    st.subheader("Test Shipment")
    st.caption("Loads artifacts via robust paths. Click to score a single synthetic row shaped by `feature_columns`.")
    score_btn = st.button("⚡ Score test shipment", type="primary", disabled=not load_ok)

with right:
    st.subheader("Results")

if score_btn and load_ok:
    try:
        # Build feature row
        X = build_test_row(feature_columns)

        # Optional scaling
        X_in = X
        if hasattr(scaler, "transform"):
            try:
                X_in = scaler.transform(X)
            except Exception as e:
                # Surface but continue with unscaled
                st.warning("Scaler.transform failed — using unscaled features for prediction.")
                st.code("".join(traceback.format_exception_only(type(e), e)))

        # Predict risk
        risk_index = predict_risk_index(model, X_in)
        eligibility = eligibility_from_risk(risk_index)
        premium = suggested_premium_usd(risk_index)

        # Display the three acceptance metrics
        m1, m2, m3 = right.columns(3)
        m1.metric("Risk Index", f"{risk_index:.1f} / 100")
        m2.metric("Eligibility", eligibility)
        m3.metric("Suggested Premium", f"${premium:,.2f}")

        with st.expander("🧪 Debug: feature row preview", expanded=False):
            st.write("Top of feature row used for this score:")
            st.dataframe(X)

    except Exception as e:
        st.error("Scoring failed with the following error:")
        st.exception(e)

# ---------------------------
# Missing model guidance (Task D)
# ---------------------------
if not (RISK_PKL and SCALER_PKL and FEATS_PKL):
    st.markdown("---")
    st.subheader("📦 Model placement instructions")
    st.write("Place your trained artifacts in either location:")
    st.code(
        "novira_underwriter_ai_enhanced_v2/model/\n"
        "  ├─ risk_model.pkl\n"
        "  ├─ scaler.pkl\n"
        "  └─ feature_columns.pkl\n\n"
        "# OR (repo root)\n"
        "risk_model.pkl\nscaler.pkl\nfeature_columns.pkl",
        language="bash",
    )
    st.info("After uploading: use ⋮ → Clear cache → Reboot.", icon="ℹ️")
