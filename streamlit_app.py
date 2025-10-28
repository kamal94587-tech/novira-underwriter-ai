import streamlit as st, sys, platform

st.set_page_config(page_title="Novira.ai – Hello", page_icon="✅", layout="wide")
st.title("✅ Streamlit is running")
st.write({"python": sys.version, "platform": platform.platform(), "streamlit": st.__version__})
st.success("You’re live. Next: diagnostics → full app.")
