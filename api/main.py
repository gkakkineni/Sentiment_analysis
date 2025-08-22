from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
from src.data_utils import batch_clean

MODEL_PATH = Path("models/sentiment_model.joblib")

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Lazy-loaded global model
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train it first (python src/train.py).")
        _model = joblib.load(MODEL_PATH)
    return _model

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    label: str
    confidence: float | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    model = get_model()
    clean = batch_clean([payload.text])
    label = model.predict(clean)[0]
    conf = None
    if hasattr(model, "predict_proba"):
        try:
            conf = float(max(model.predict_proba(clean)[0]))
        except Exception:
            conf = None
    return {"label": label, "confidence": conf}