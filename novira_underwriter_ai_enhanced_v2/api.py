
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, os

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(APP_DIR), "model")

model = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

app = FastAPI(title="Novira.ai Underwriting Risk Scorer — Enhanced v3")

class ShipmentFeatures(BaseModel):
    avg_temp_c: float
    temp_variance: float
    humidity_change: float
    shock_g: float
    light_events: int
    route_deviation_km: float
    theft_zone_risk: float
    weather_risk: float
    carrier_score: float
    device_uptime: float
    transit_time_hr: float
    declared_value_usd: float
    cargo_type: str
    packaging_type: str

def to_vector(p):
    base = {
        "avg_temp_c": p["avg_temp_c"],
        "temp_variance": p["temp_variance"],
        "humidity_change": p["humidity_change"],
        "shock_g": p["shock_g"],
        "light_events": p["light_events"],
        "route_deviation_km": p["route_deviation_km"],
        "theft_zone_risk": p["theft_zone_risk"],
        "weather_risk": p["weather_risk"],
        "carrier_score": p["carrier_score"],
        "device_uptime": p["device_uptime"],
        "transit_time_hr": p["transit_time_hr"],
        "declared_value_usd": p["declared_value_usd"],
        "cargo_type_electronics_high_value": 1.0 if p["cargo_type"]=="electronics_high_value" else 0.0,
        "cargo_type_fresh_produce": 1.0 if p["cargo_type"]=="fresh_produce" else 0.0,
        "cargo_type_general_merchandise": 1.0 if p["cargo_type"]=="general_merchandise" else 0.0,
        "cargo_type_pharma_cold_chain": 1.0 if p["cargo_type"]=="pharma_cold_chain" else 0.0,
        "packaging_type_insulated": 1.0 if p["packaging_type"]=="insulated" else 0.0,
        "packaging_type_palletized": 1.0 if p["packaging_type"]=="palletized" else 0.0,
        "packaging_type_refrigerated": 1.0 if p["packaging_type"]=="refrigerated" else 0.0,
    }
    return np.array([base[c] for c in feature_columns], dtype=float).reshape(1,-1)

def eligibility_and_pricing(prob, risk_index, declared_value_usd, carrier_score):
    if prob >= 0.60:
        decision = "decline"
        reason = "Predicted loss probability above 60%."
        min_deductible = int(max(2500, 0.03 * declared_value_usd))
        multiplier = 1.6
    elif risk_index >= 55 or (prob >= 0.4 and carrier_score < 0.7):
        decision = "review"
        reason = "Moderate-high risk or weak carrier reliability."
        min_deductible = int(max(2000, 0.02 * declared_value_usd))
        multiplier = 1.25
    else:
        decision = "bind"
        reason = "Acceptable risk profile."
        min_deductible = int(max(1000, 0.01 * declared_value_usd))
        multiplier = 1.0

    base_rate = 0.010
    premium = base_rate * declared_value_usd * (0.7 + prob) * multiplier
    return decision, reason, round(premium,2), min_deductible, multiplier

@app.post("/score")
def score(f: ShipmentFeatures):
    p = f.model_dump()
    X = to_vector(p)
    Xs = scaler.transform(X)
    prob = float(model.predict_proba(Xs)[0,1])
    risk_index = int(round(prob*100))
    band = "low" if risk_index<=30 else "moderate" if risk_index<=60 else "high"

    decision, reason, premium, min_deductible, multiplier = eligibility_and_pricing(
        prob, risk_index, p["declared_value_usd"], p["carrier_score"]
    )

    top_factors = []
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xs)
            sv_row = sv[0] if isinstance(sv, np.ndarray) else sv.values[0]
            idxs = np.argsort(-np.abs(sv_row))[:5]
            for i in idxs:
                top_factors.append({"feature": feature_columns[i], "shap_contribution": float(sv_row[i])})
        except Exception:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                vals = Xs[0] * importances
                idxs = np.argsort(-np.abs(vals))[:5]
                for i in idxs:
                    top_factors.append({"feature": feature_columns[i], "importance_proxy": float(vals[i])})
    else:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            vals = Xs[0] * importances
            idxs = np.argsort(-np.abs(vals))[:5]
            for i in idxs:
                top_factors.append({"feature": feature_columns[i], "importance_proxy": float(vals[i])})

    return {
        "risk_index": risk_index,
        "prob_loss": round(prob,4),
        "band": band,
        "eligibility": {"decision": decision, "reason": reason},
        "pricing": {"premium_suggestion_usd": premium, "min_deductible_usd": min_deductible, "multiplier": multiplier},
        "explainability": {"top_factors": top_factors},
        "model_version": "enhanced-gbc-0.3"
    }
