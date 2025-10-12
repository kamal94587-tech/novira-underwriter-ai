import streamlit as st
import traceback, runpy

st.set_page_config(page_title="Novira.ai — Underwriter", layout="wide")

try:
    st.info("✅ Loading Novira.ai Underwriter AI (Enhanced v2)...")
    runpy.run_path("novira_underwriter_ai_enhanced_v2/app/api.py", run_name="__main__")
except Exception as e:
    st.error("🚨 App failed to start")
    st.write("Here’s the exact error so we can fix it fast:")
    st.exception(e)
    st.code("".join(traceback.format_exc()))
