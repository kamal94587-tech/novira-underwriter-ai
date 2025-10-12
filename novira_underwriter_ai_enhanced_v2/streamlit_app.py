import streamlit as st
import traceback, runpy, sys

st.set_page_config(page_title="Novira.ai — Underwriter", layout="wide")

try:
    st.info("✅ Launching Novira Underwriter UI…")
    # IMPORTANT: call the top-level app file (NOT the nested /app version)
    runpy.run_path("novira_underwriter_ai_enhanced_v2/streamlit_app.py", run_name="__main__")
except Exception as e:
    st.error("🚨 App failed to start")
    st.write("Here’s the exact error so we can fix it fast:")
    st.exception(e)
    st.code("".join(traceback.format_exc()))
    st.caption(f"Python {sys.version}")
