import streamlit as st
import sys, os, json, platform, traceback
from pathlib import Path

st.set_page_config(page_title="Novira.ai — Diagnostics", page_icon="🛠", layout="wide")
st.title("🛠 Novira.ai — Diagnostics")

def section(t): st.markdown(f"### {t}")

section("Environment")
st.json({"python": sys.version, "platform": platform.platform(), "streamlit": st.__version__, "cwd": os.getcwd(), "__file__": __file__})

section("Top-level files")
try:
    st.write(sorted(os.listdir("."))[:200])
except Exception as e:
    st.exception(e)

BASE = Path(__file__).resolve().parent
ENH = BASE / "novira_underwriter_ai_enhanced_v2"

section("Enhanced v2 present?")
st.write({"novira_underwriter_ai_enhanced_v2_exists": ENH.exists()})

section("Model paths")
candidates = [
    Path("risk_model.pkl"),
    Path("scaler.pkl"),
    Path("feature_columns.pkl"),
    ENH / "model" / "risk_model.pkl",
    ENH / "model" / "scaler.pkl",
    ENH / "model" / "feature_columns.pkl",
]
st.json({"exists": {str(p): p.exists() for p in candidates}})

section("Imports")
def try_import(name):
    try:
        m = __import__(name)
        st.success(f"Imported: {name} ({getattr(m, '__version__', 'no __version__')})")
    except Exception as e:
        st.error(f"Import failed: {name}")
        st.code("".join(traceback.format_exception_only(type(e), e)))
for lib in ["pandas", "numpy", "scikit_learn", "shap", "joblib"]:
    try_import(lib)

section("Load pickles (first match)")
def load_pickle(p):
    import pickle
    with open(p, "rb") as f: return pickle.load(f)

results, errors = {}, []
for label, group in {
    "risk_model": ["risk_model.pkl", str(ENH/"model"/"risk_model.pkl")],
    "scaler": ["scaler.pkl", str(ENH/"model"/"scaler.pkl")],
    "feature_columns": ["feature_columns.pkl", str(ENH/"model"/"feature_columns.pkl")],
}.items():
    ok = False
    for g in group:
        p = Path(g)
        if p.exists():
            try:
                _ = load_pickle(p)
                results[label] = f"Loaded from {p}"
                ok = True
                break
            except Exception as e:
                errors.append({"label": label, "path": str(p), "error": repr(e)})
    if not ok:
        results[label] = "NOT FOUND in any candidate path"

st.json({"loaded": results})
if errors:
    st.warning("Errors while loading some pickles:")
    st.code(json.dumps(errors, indent=2))

st.success("Diagnostics complete. If all green, I’ll swap to the real UI next.")
