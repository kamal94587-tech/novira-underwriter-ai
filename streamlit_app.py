# Novira.ai — Underwriter Risk Scorecard (Dummy bootstrap with fallback on load failure)

import sys, traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Novira.ai — Underwriter Risk Scorecard", page_icon="📊", layout="wide")
st.title("🧠 Novira.ai — Underwriter Risk Scorecard (Minimal)")

# Safety: avoid NameError if any early code references score_btn
if "score_btn" not in globals():
    score_btn = False

BASE = Path(__file__).resolve().parent
MODEL_DIR1 = BASE / "novira_underwriter_ai_enhanced_v2" / "model"

def find_first(*rel_paths: str):
    for rel in rel_paths:
        p = (BASE / rel).resolve()
        if p.exists():
            return p
    return None
def resolve_paths():
    return (
        find_first("novira_underwriter_ai_enhanced_v2/model/risk_model.pkl", "risk_model.pkl"),
        find_first("novira_underwriter_ai_enhanced_v2/model/scaler.pkl", "scaler.pkl"),
        find_first("novira_underwriter_ai_enhanced_v2/model/feature_columns.pkl", "feature_columns.pkl"),
    )

RISK_PKL, SCALER_PKL, FEATS_PKL = resolve_paths()

with st.expander("🔍 Resolved paths & environment", expanded=False):
    st.json({
        "python": sys.version,
        "resolved_paths": {
            "risk_model.pkl": str(RISK_PKL) if RISK_PKL else None,
            "scaler.pkl": str(SCALER_PKL) if SCALER_PKL else None,
            "feature_columns.pkl": str(FEATS_PKL) if FEATS_PKL else None,
        }
    })

# ------------------------------------------------------------------
# Bootstrap dummy artifacts (can be called on missing OR on load-fail)
# ------------------------------------------------------------------
def bootstrap_dummy_models() -> str:
    MODEL_DIR1.mkdir(parents=True, exist_ok=True)

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
        "package_fragility": rng.integers(0, 4, n),
        "weather_severity_index": rng.normal(20, 15, n).clip(0, 100),
        "port_congestion_index": rng.normal(30, 20, n).clip(0, 100),
    })

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
    y = (p > 0.5).astype(int)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    joblib.dump(feature_columns, MODEL_DIR1 / "feature_columns.pkl")
    joblib.dump(scaler,           MODEL_DIR1 / "scaler.pkl")
    joblib.dump(model,            MODEL_DIR1 / "risk_model.pkl")

    return f"Dummy artifacts created in {MODEL_DIR1}"

@st.cache_resource(show_spinner=False)
def _load(path: Path):
    return joblib.load(path)

def _try_load_all():
    """Try to load all three; return (ok, model, scaler, feats, errors:list[str])."""
    ok, model, scaler, feats = True, None, None, None
    errors = []
    def _require(path: Path, label: str):
        nonlocal ok
        if not path:
            ok = False
            errors.append(f"{label}: NOT FOUND")
            return None
        try:
            return _load(path)
        except Exception as e:
            ok = False
            errors.append(f"{label}: load failed from {path}\n" + "".join(traceback.format_exception_only(type(e), e)))
            return None

    feats = _require(FEATS_PKL, "feature_columns.pkl")
    scaler = _require(SCALER_PKL, "scaler.pkl")
    model = _require(RISK_PKL, "risk_model.pkl")
    return ok, model, scaler, feats, errors

# First attempt — if any fail, bootstrap, then resolve paths and retry
ok, model, scaler, feature_columns, load_errors = _try_load_all()
if not ok:
    st.warning("Artifacts missing or unreadable — creating dummy artifacts now.", icon="⚠️")
    try:
        msg = bootstrap_dummy_models()
        st.success(msg)
        # prefer the newly created ones under enhanced_v2/model
        RISK_PKL, SCALER_PKL, FEATS_PKL = resolve_paths()
        ok, model, scaler, feature_columns, load_errors = _try_load_all()
    except Exception as e:
        st.error("Bootstrap failed:")
        st.exception(e)

if not ok:
    for err in load_errors:
        st.error(err)
    st.stop()

# ---------------------------
# Feature row & scoring
# ---------------------------
def build_test_row(cols):
    names = list(cols) if isinstance(cols, (list, tuple, np.ndarray, pd.Index)) else (
        list(cols.get("columns", [])) if isinstance(cols, dict) else []
    )
    data = {c: 0 for c in names}
    preset = {
        "shipment_value_usd": 45000, "distance_km": 1200, "days_in_transit": 6,
        "is_international": 0, "is_perishable": 0, "temperature_req_c": 5,
        "route_risk_score": 30, "carrier_reliability": 85, "past_claims_count": 0,
        "package_fragility": 1, "weather_severity_index": 10, "port_congestion_index": 15,
    }
    for k, v in preset.items():
        if k in data: data[k] = v
    return pd.DataFrame([data], columns=names)


def predict_risk_index(model, X):
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2: return to_0_100(proba[0,1]*100)
        if proba.ndim == 1: return to_0_100(proba[0]*100)
    y = float(np.asarray(model.predict(X)).ravel()[0])
    return to_0_100(y)

def eligibility_from_risk(r):
    return "Bind" if r < 40 else ("Review" if r < 70 else "Decline")

def suggested_premium_usd(r, base=1200.0):
    return float(round(base*(1+r/150.0), 2))

left, right = st.columns([0.6, 0.4])
with left:
    st.subheader("Test Shipment")
    st.caption("Click to score a synthetic shipment. Uses dummy model if real artifacts aren’t uploaded yet.")
    score_btn = st.button("⚡ Score test shipment", type="primary")

with right:
    st.subheader("Results")
if score_btn:
    try:
        X = build_test_row(feature_columns)
        try:
            X_in = scaler.transform(X)
        except Exception:
            X_in = X

        risk = predict_risk_index(model, X_in)
        elig = eligibility_from_risk(risk)
        prem = suggested_premium_usd(risk)

        c1, c2, c3 = right.columns(3)
        c1.metric("Risk Index", f"{risk:.1f} / 100")
        c2.metric("Eligibility", elig)
        c3.metric("Suggested Premium", f"${prem:,.2f}")

        with st.expander("Debug: feature row", expanded=False):
            st.dataframe(X)

    except Exception as e:
        st.error("Scoring failed:")
        st.exception(e)
# ---------------------------
# UI — layout + score button (define BEFORE using it)
# ---------------------------
left, right = st.columns([0.6, 0.4])

with right:
    st.subheader("Results")

# Safety: if this file ever gets rearranged, avoid NameError
if "score_btn" not in locals():
    score_btn = False

# ---------------------------
# Score on click
# ---------------------------
if score_btn:
    try:
        X = build_test_row(feature_columns)
        try:
            X_in = scaler.transform(X)
        except Exception:
            X_in = X

        risk = predict_risk_index(model, X_in)
        elig = eligibility_from_risk(risk)
        prem = suggested_premium_usd(risk)

        c1, c2, c3 = right.columns(3)
        c1.metric("Risk Index", f"{risk:.1f} / 100")
        c2.metric("Eligibility", elig)
        c3.metric("Suggested Premium", f"${prem:,.2f}")

        # Optional: Debug view to inspect the feature row
        with st.expander("Debug: feature row", expanded=False):
            st.dataframe(X)

    except Exception as e:
        st.error("Scoring failed:")
        st.exception(e)

st.subheader("📦 Model placement (when you have real artifacts)")
st.code(
    "novira_underwriter_ai_enhanced_v2/model/\n"
    "  ├─ risk_model.pkl\n"
    "  ├─ scaler.pkl\n"
    "  └─ feature_columns.pkl\n\n"
    "# OR (repo root)\n"
    "risk_model.pkl\nscaler.pkl\nfeature_columns.pkl",
    language="bash",
)
st.info("After uploading real artifacts: ⋮ → Clear cache → Reboot.", icon="ℹ️")


