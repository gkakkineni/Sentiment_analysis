import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from pathlib import Path
import joblib
from src.data_utils import batch_clean

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🔍")

MODEL_PATH = Path("models/sentiment_model.joblib")

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        st.error(f"Model not found at {path}. Please train it first: `python src/train.py`")
        st.stop()
    return joblib.load(path)

st.title("🔍 Sentiment Analyzer")
st.write("Type text below and get a **positive / negative / neutral** prediction.")

model = load_model(MODEL_PATH)

mode = st.radio("Mode", ["Single Text", "Batch (one per line)"], horizontal=True)

if mode == "Single Text":
    text = st.text_area("Enter text", height=160, placeholder="e.g., I love this product!")
    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            clean = batch_clean([text])
            label = model.predict(clean)[0]
            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    conf = float(max(model.predict_proba(clean)[0]))
                except Exception:
                    conf = None
            st.success(f"**Prediction:** {label}")
            if conf is not None:
                st.write(f"**Confidence:** {conf:.3f}")
else:
    lines = st.text_area("Enter one text per line", height=200, placeholder="good job!\nthis is terrible\nit's okay")
    if st.button("Predict Batch"):
        rows = [ln.strip() for ln in lines.splitlines() if ln.strip()]
        if not rows:
            st.warning("Enter at least one line.")
        else:
            clean = batch_clean(rows)
            labels = model.predict(clean).tolist()
            st.write("**Results:**")
            for t, l in zip(rows, labels):
                st.write(f"- `{t}` → **{l}**")