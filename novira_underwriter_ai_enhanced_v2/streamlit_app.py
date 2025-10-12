import streamlit as st
import os, sys, traceback, runpy

st.set_page_config(page_title="Novira.ai — Underwriter", layout="wide")

st.title("🔧 Novira.ai — Underwriter Launcher")
st.write("Python:", sys.version)
st.write("CWD:", os.getcwd())

target = "novira_underwriter_ai_enhanced_v2/streamlit_app.py"
st.write("Target file:", target, "→", "FOUND ✅" if os.path.exists(target) else "MISSING ❌")

# Render the page immediately (so we avoid a spinner)
st.success("Launcher UI is rendering. Click the button below to run the full app.")

if st.button("▶️ Run full Underwriter app"):
    try:
        st.info("Starting full app…")
        runpy.run_path(target, run_name="__main__")
    except Exception as e:
        st.error("🚨 App failed to start")
        st.exception(e)
        st.code("".join(traceback.format_exc()))
