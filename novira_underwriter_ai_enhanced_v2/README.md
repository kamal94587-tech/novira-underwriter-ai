
# Novira.ai — Underwriter AI (Enhanced v3)

- Streamlit dashboard with **SHAP explainability** + **eligibility (bind/review/decline)**.
- FastAPI scorer at `/score` with pricing & top-factors.
- Trained model included.

## Quickstart (Streamlit Cloud)
1) Push this folder to a **public GitHub repo**.
2) Go to https://share.streamlit.io → New app
3) Repo: yourusername/yourrepo • Branch: main • Main file: `app/streamlit_app.py`
4) Deploy → share your URL with underwriters.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## FastAPI (optional)
```bash
uvicorn app.api:app --reload
```

Model AUC on synthetic test split: **0.568**
