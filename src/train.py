import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from data_utils import batch_clean

def load_data(train_csv: Path, test_csv: Path | None):
    df_train = pd.read_csv(train_csv)
    if test_csv and Path(test_csv).exists():
        df_test = pd.read_csv(test_csv)
    else:
        # Split from train if test not provided
        df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])
    return df_train, df_test

def build_pipeline():
    # Strong classical baseline for sentiment
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=None,
            tokenizer=None,
            lowercase=False,   # we clean manually
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    return pipe

def main(args):
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_test = load_data(args.train_csv, args.test_csv)

    # Clean text
    df_train['clean_text'] = batch_clean(df_train['text'].astype(str).tolist())
    df_test['clean_text']  = batch_clean(df_test['text'].astype(str).tolist())

    X_train = df_train['clean_text']
    y_train = df_train['label'].astype(str)

    X_test = df_test['clean_text']
    y_test = df_test['label'].astype(str)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print("\n=== Evaluation on test set ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    model_path = models_dir / "sentiment_model.joblib"
    joblib.dump(pipe, model_path)
    print(f"\nSaved trained pipeline to: {model_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment classifier.")
    parser.add_argument("--train-csv", type=str, default="data/sample_train.csv", help="Path to training CSV with columns: text,label")
    parser.add_argument("--test-csv", type=str, default="data/sample_test.csv", help="(Optional) Path to test CSV with columns: text,label")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save trained model pipeline.")
    args = parser.parse_args()
    main(args)