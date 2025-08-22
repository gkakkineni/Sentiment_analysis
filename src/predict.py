import argparse, sys, json
from pathlib import Path
import joblib
from data_utils import batch_clean

def load_model(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found at {model_path}. Train it first with: python src/train.py", file=sys.stderr)
        sys.exit(1)
    return joblib.load(model_path)

def predict_texts(model, texts):
    clean = batch_clean(texts)
    preds = model.predict(clean).tolist()
    out = []
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(clean)
        except Exception:
            proba = None
    for i, t in enumerate(texts):
        rec = {"text": t, "label": preds[i]}
        if proba is not None:
            rec["confidence"] = float(max(proba[i]))
        out.append(rec)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for input text(s).")
    parser.add_argument("--model", type=str, default="models/sentiment_model.joblib")
    parser.add_argument("--text", type=str, help="Single text to classify.")
    parser.add_argument("--file", type=str, help="Path to a text file with one example per line.")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.text:
        results = predict_texts(model, [args.text])
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        results = predict_texts(model, lines)
    else:
        print("Provide --text or --file", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(results, indent=2, ensure_ascii=False))
