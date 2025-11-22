import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import argparse
import numpy as np

from utils import extract_url_features  # uses the safer version you fixed

p = Path(__file__).parent

# Keep feature column names in one place
FEATURE_COLS = [
    "url_len", "host_len", "path_len", "count_dots", "count_hyphens",
    "has_at", "has_ip", "has_query", "num_subdirs", "is_https",
    "sld_len", "has_non_ascii", "count_double_slash", "has_punycode",
    "host_entropy", "has_port",
]


def get_numeric_features(X):
    """
    Custom transformer to extract numeric features from a series of URLs.
    Uses extract_url_features() from utils.py and returns a 2D numpy array.
    """
    # Build a DataFrame from list of dicts (one per URL)
    df_features = pd.DataFrame([extract_url_features(url) for url in X])

    # Make sure all expected columns exist (fill missing with 0)
    for col in FEATURE_COLS:
        if col not in df_features.columns:
            df_features[col] = 0.0

    # Replace any NaN with 0 so StandardScaler doesn't choke
    df_features = df_features[FEATURE_COLS].fillna(0.0)

    return df_features.values


def load_dataset(dataset):
    """
    Load and lightly clean the dataset.
    Ensures 'url' and 'label' columns exist and removes empty rows.
    """
    if dataset is None:
        path = p / "sample_phishing.csv"
    else:
        path = Path(dataset)

    # engine="python" is a bit more forgiving with messy CSVs
    df = pd.read_csv(path, engine="python")

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'url' and 'label' columns")

    # Drop rows where url or label is missing
    df = df.dropna(subset=["url", "label"])

    # Clean basic formatting
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    return df


def train(dataset=None, out="phish_model.joblib"):
    # ---- Load data safely ----
    df = load_dataset(dataset)

    X = df["url"]
    y = df["label"]

    # ---- Numeric feature pipeline ----
    numeric_features = Pipeline([
        ("extractor", FunctionTransformer(get_numeric_features, validate=False)),
        ("scaler", StandardScaler()),
    ])

    # ---- Character-level TF-IDF (text) pipeline ----
    text_features = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6))),
    ])

    # ---- Combine numeric + text features ----
    preprocessor = FeatureUnion([
        ("text", text_features),
        ("numeric", numeric_features),
    ])

    # ---- Final pipeline ----
    pipe = Pipeline([
        ("union", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
    ])

    # ---- Train/test split + training ----
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(Xtrain, ytrain)
    pred = pipe.predict(Xtest)

    print("Accuracy:", accuracy_score(ytest, pred))
    print(classification_report(ytest, pred))

    # ---- Save model ----
    out_path = p / out
    joblib.dump(pipe, out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV dataset path (must have columns: url,label)")
    ap.add_argument("--out", default="phish_model.joblib")
    args = ap.parse_args()

    train(args.data, args.out)
