import streamlit as st
import traceback, runpy, sys, os

st.set_page_config(page_title="Novira.ai — Underwriter", layout="wide")

st.title("🚀 Novira.ai — Underwriter App Launcher")
st.write("Python version:", sys.version)
st.write("Current directory:", os.getcwd())

try:
    st.info("✅ Launching Novira Underwriter AI (Enhanced v2)…")
    # Make sure file exists before running
    path = "novira_underwriter_ai_enhanced_v2/streamlit_app.py"
    if os.path.exists(path):
        st.success(f"Found file: {path}")
        runpy.run_path(path, run_name="__main__")
    else:
        st.error(f"❌ Could not find {path}")
        st.write("Available files:", os.listdir("novira_underwriter_ai_enhanced_v2"))
except Exception as e:
    st.error("🚨 App failed to start")
    st.exception(e)
    st.code("".join(traceback.format_exc()))
