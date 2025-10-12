
import streamlit as st, joblib, numpy as np, pandas as pd, os

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

st.set_page_config(page_title="Novira.ai — Underwriter Risk Score Card (Enhanced v3)", layout="centered")
st.title("Novira.ai — Underwriter Risk Score Card (Enhanced v3)")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(APP_DIR), "model")

model = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

def to_vector(row: dict):
    base = {
        "avg_temp_c": row["avg_temp_c"],
        "temp_variance": row["temp_variance"],
        "humidity_change": row["humidity_change"],
        "shock_g": row["shock_g"],
        "light_events": row["light_events"],
        "route_deviation_km": row["route_deviation_km"],
        "theft_zone_risk": row["theft_zone_risk"],
        "weather_risk": row["weather_risk"],
        "carrier_score": row["carrier_score"],
        "device_uptime": row["device_uptime"],
        "transit_time_hr": row["transit_time_hr"],
        "declared_value_usd": row["declared_value_usd"],
        "cargo_type_electronics_high_value": 1.0 if row["cargo_type"]=="electronics_high_value" else 0.0,
        "cargo_type_fresh_produce": 1.0 if row["cargo_type"]=="fresh_produce" else 0.0,
        "cargo_type_general_merchandise": 1.0 if row["cargo_type"]=="general_merchandise" else 0.0,
        "cargo_type_pharma_cold_chain": 1.0 if row["cargo_type"]=="pharma_cold_chain" else 0.0,
        "packaging_type_insulated": 1.0 if row["packaging_type"]=="insulated" else 0.0,
        "packaging_type_palletized": 1.0 if row["packaging_type"]=="palletized" else 0.0,
        "packaging_type_refrigerated": 1.0 if row["packaging_type"]=="refrigerated" else 0.0,
    }
    return np.array([base[c] for c in feature_columns], dtype=float).reshape(1,-1)

def eligibility_and_pricing(prob, risk_index, declared_value_usd, carrier_score):
    if prob >= 0.60:
        decision, reason, multiplier = "decline", "Predicted loss probability above 60%.", 1.6
        min_deductible = int(max(2500, 0.03 * declared_value_usd))
    elif risk_index >= 55 or (prob >= 0.4 and carrier_score < 0.7):
        decision, reason, multiplier = "review", "Moderate-high risk or weak carrier reliability.", 1.25
        min_deductible = int(max(2000, 0.02 * declared_value_usd))
    else:
        decision, reason, multiplier = "bind", "Acceptable risk profile.", 1.0
        min_deductible = int(max(1000, 0.01 * declared_value_usd))
    base_rate = 0.010
    premium = base_rate * declared_value_usd * (0.7 + prob) * multiplier
    return decision, reason, round(premium,2), min_deductible, multiplier

def score_one(row: dict):
    X = to_vector(row)
    Xs = scaler.transform(X)
    prob = float(model.predict_proba(Xs)[0,1])
    risk_index = int(round(prob*100))
    band = "Low" if risk_index<=30 else "Moderate" if risk_index<=60 else "High"
    decision, reason, premium, min_deductible, multiplier = eligibility_and_pricing(prob, risk_index, row["declared_value_usd"], row["carrier_score"])
    top_factors = []
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xs)
            sv_row = sv[0] if isinstance(sv, np.ndarray) else sv.values[0]
            idxs = np.argsort(-np.abs(sv_row))[:5]
            for i in idxs:
                top_factors.append((feature_columns[i], float(sv_row[i])))
        except Exception:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                vals = Xs[0] * importances
                idxs = np.argsort(-np.abs(vals))[:5]
                for i in idxs:
                    top_factors.append((feature_columns[i], float(vals[i])))
    else:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            vals = Xs[0] * importances
            idxs = np.argsort(-np.abs(vals))[:5]
            for i in idxs:
                top_factors.append((feature_columns[i], float(vals[i])))
    return {
        "risk_index": risk_index, "prob_loss": prob, "band": band,
        "decision": decision, "reason": reason, "premium": premium,
        "min_deductible": min_deductible, "multiplier": multiplier,
        "top_factors": top_factors
    }

st.subheader("Single Shipment — Score & Explain")
example = {"avg_temp_c":9.0,"temp_variance":5.0,"humidity_change":12.0,"shock_g":2.1,"light_events":1,"route_deviation_km":3.0,
           "theft_zone_risk":6.0,"weather_risk":5.0,"carrier_score":0.78,"device_uptime":0.95,"transit_time_hr":72.0,
           "declared_value_usd":250000.0,"cargo_type":"pharma_cold_chain","packaging_type":"refrigerated"}

with st.form("single"):
    c1,c2 = st.columns(2)
    with c1:
        avg_temp_c = st.number_input("avg_temp_c", value=example["avg_temp_c"])
        temp_variance = st.number_input("temp_variance", value=example["temp_variance"])
        humidity_change = st.number_input("humidity_change", value=example["humidity_change"])
        shock_g = st.number_input("shock_g", value=example["shock_g"])
        light_events = st.number_input("light_events", value=example["light_events"], step=1)
        route_deviation_km = st.number_input("route_deviation_km", value=example["route_deviation_km"])
        theft_zone_risk = st.number_input("theft_zone_risk", value=example["theft_zone_risk"])
        weather_risk = st.number_input("weather_risk", value=example["weather_risk"])
    with c2:
        carrier_score = st.number_input("carrier_score (0-1)", value=example["carrier_score"])
        device_uptime = st.number_input("device_uptime (0-1)", value=example["device_uptime"])
        transit_time_hr = st.number_input("transit_time_hr", value=example["transit_time_hr"])
        declared_value_usd = st.number_input("declared_value_usd", value=example["declared_value_usd"], step=1000.0)
        cargo_type = st.selectbox("cargo_type", ["pharma_cold_chain","electronics_high_value","general_merchandise","fresh_produce","chemicals"], index=0)
        packaging_type = st.selectbox("packaging_type", ["insulated","standard","refrigerated","palletized"], index=2)
    submitted = st.form_submit_button("Score Shipment")
    if submitted:
        row = dict(avg_temp_c=avg_temp_c,temp_variance=temp_variance,humidity_change=humidity_change,shock_g=shock_g,
                   light_events=int(light_events),route_deviation_km=route_deviation_km,theft_zone_risk=theft_zone_risk,
                   weather_risk=weather_risk,carrier_score=carrier_score,device_uptime=device_uptime,
                   transit_time_hr=transit_time_hr,declared_value_usd=declared_value_usd,cargo_type=cargo_type,
                   packaging_type=packaging_type)
        res = score_one(row)
        st.metric("Risk Index (0–100)", res["risk_index"])
        st.write(f"Probability of Loss: **{res['prob_loss']:.3f}**  |  Band: **{res['band']}**")
        st.write(f"Eligibility: **{res['decision'].upper()}** — {res['reason']}")
        st.write(f"Premium suggestion: **${res['premium']:,}**  |  Min deductible: **${res['min_deductible']:,}**  |  Multiplier: **{res['multiplier']}x**")
        st.subheader("Top drivers")
        for f, v in res["top_factors"]:
            st.write(f"- {f}: {v:+.4f}")

st.subheader("Batch scoring (CSV upload)")
uploaded = st.file_uploader("Upload CSV like data/synthetic_shipments.csv", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    outs=[]
    for _, r in df.iterrows():
        outs.append(score_one(r.to_dict()))
    out_df = df.copy()
    out_df["risk_index"] = [o["risk_index"] for o in outs]
    out_df["prob_loss"] = [o["prob_loss"] for o in outs]
    out_df["eligibility"] = [o["decision"] for o in outs]
    st.dataframe(out_df.head(25))
    st.download_button("Download Scored CSV", out_df.to_csv(index=False).encode("utf-8"), file_name="scored_shipments.csv", mime="text/csv")
