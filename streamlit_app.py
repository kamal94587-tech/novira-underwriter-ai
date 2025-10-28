# Novira.ai — Underwriter Risk Scorecard (Dummy bootstrap with fallback on load failure)

import sys, traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- UI helpers (styles + small formatters) ----
st.markdown("""
<style>
.result-card { padding: 1.0rem 1.2rem; border: 1px solid rgba(255,255,255,0.08); background: #121418; border-radius: 16px; }
.kpi-title { font-size: 0.9rem; opacity: 0.7; margin-bottom: 0.25rem; }
.kpi-value { font-size: 2.0rem; font-weight: 700; line-height: 1.1; }
.badge { display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
.badge-green { background: rgba(34,197,94,0.15); color: #34d399; border: 1px solid rgba(34,197,94,0.25); }
.badge-amber { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }
.badge-red   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
.progress-wrap { width:100%; background:#1b1f26; border-radius: 10px; height: 14px; overflow: hidden; }
.progress-bar  { height: 100%; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

def _format_usd(x: float) -> str:
    return f"${x:,.2f}"

def _elig_badge(elig_text: str) -> str:
    text = elig_text.strip().lower()
    if text == "bind":
        cls = "badge badge-green"; label = "Bind"
    elif text == "review":
        cls = "badge badge-amber"; label = "Review"
    else:
        cls = "badge badge-red"; label = "Decline"
    return f'<span class="{cls}">{label}</span>'

def _risk_color(risk_0_to_100: float) -> str:
    r = max(0, min(100, risk_0_to_100))
    if r <= 50:
        t = r / 50.0
        start = (52, 211, 153)   # green
        end   = (245, 158, 11)   # amber
    else:
        t = (r - 50) / 50.0
        start = (245, 158, 11)   # amber
        end   = (239, 68, 68)    # red
    c = tuple(int(s + (e - s) * t) for s, e in zip(start, end))
    return f"rgb({c[0]},{c[1]},{c[2]})"

def render_result_card(risk: float, elig_text: str, premium_usd: float, feature_row_df):
    with st.container(border=False):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([0.9, 0.6, 0.8])

        # Risk Index
        with c1:
            st.markdown('<div class="kpi-title">Risk Index</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{risk:.1f} / 100</div>', unsafe_allow_html=True)
            bar_color = _risk_color(risk)
            st.markdown(
                f"""
                <div class="progress-wrap">
                    <div class="progress-bar" style="width:{risk:.1f}%; background: {bar_color};"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Eligibility
        with c2:
            st.markdown('<div class="kpi-title">Eligibility</div>', unsafe_allow_html=True)
            st.markdown(_elig_badge(elig_text), unsafe_allow_html=True)

        # Premium
        with c3:
            st.markdown('<div class="kpi-title">Suggested Premium</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{_format_usd(premium_usd)}</div>', unsafe_allow_html=True)

        with st.expander("Debug: feature row", expanded=False):
            st.dataframe(feature_row_df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

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
def render_result_card(risk, elig, prem, X):
    """Pretty results card with gradient bar + colored eligibility badge + premium."""
    try:
        risk_val = float(risk)
    except Exception:
        risk_val = 0.0

    # Badge color by eligibility
    if str(elig).lower().startswith("bind"):
        badge_color = "#22c55e"   # green
        badge_fg    = "#0b0f16"
    elif str(elig).lower().startswith("review"):
        badge_color = "#f59e0b"   # amber
        badge_fg    = "#0b0f16"
    else:
        badge_color = "#ef4444"   # red
        badge_fg    = "white"

    card_html = f"""
    <div style="background:#111827;border:1px solid #2d2f36;border-radius:16px;padding:18px 18px 12px;margin:6px 0;">
      <div style="font-weight:700;font-size:18px;margin:0 0 10px 0;">Results</div>

      <div style="height:10px;background:#27272a;border-radius:8px;overflow:hidden;">
        <div style="height:100%;width:{risk_val:.1f}%;background:linear-gradient(90deg,#22d3ee,#a78bfa);"></div>
      </div>

      <div style="display:flex;gap:28px;align-items:baseline;margin-top:10px;flex-wrap:wrap">
        <div>
          <div style="opacity:.7;font-size:12px;margin-bottom:4px;">Risk Index</div>
          <div style="font-size:26px;font-weight:800;letter-spacing:.3px">{risk_val:.1f} / 100</div>
        </div>

        <div>
          <div style="opacity:.7;font-size:12px;margin-bottom:4px;">Eligibility</div>
          <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{badge_color};color:{badge_fg};
                       font-weight:800;letter-spacing:.3px">{elig}</span>
        </div>

        <div>
          <div style="opacity:.7;font-size:12px;margin-bottom:4px;">Suggested Premium</div>
          <div style="font-size:26px;font-weight:800">${prem:,.2f}</div>
        </div>
      </div>
    </div>
    """
    import streamlit as st
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander("Debug: feature row", expanded=False):
        st.dataframe(X)


# Helper: clamp any value between 0–100
def to_0_100(x):
    try:
        import numpy as np
        arr = np.asarray(x, dtype=float)
        arr = np.clip(arr, 0, 100)
        return arr.item() if arr.size == 1 else arr
    except Exception:
        return 0.0
        
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

        # ✅ Modern visual card
        render_result_card(risk, elig, prem, X)

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


