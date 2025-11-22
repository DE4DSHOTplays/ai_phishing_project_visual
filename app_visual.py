import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import io

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import matplotlib.pyplot as plt
import numpy as np

from utils import extract_url_features
from train_and_save import train as train_model  # uses robust loader & features

# ---- Keep feature column names in one place (must match train_and_save.py) ----
FEATURE_COLS = [
    "url_len", "host_len", "path_len", "count_dots", "count_hyphens",
    "has_at", "has_ip", "has_query", "num_subdirs", "is_https",
    "sld_len", "has_non_ascii", "count_double_slash", "has_punycode",
    "host_entropy", "has_port",
]


# --- Numeric feature transformer (aligned with train_and_save.get_numeric_features) ---
def get_numeric_features(X):
    """
    Custom transformer to extract numeric features from a series of URLs.
    Uses extract_url_features() from utils.py and returns a 2D numpy array.
    """
    df_features = pd.DataFrame([extract_url_features(url) for url in X])

    # Make sure all expected columns exist (fill missing with 0)
    for col in FEATURE_COLS:
        if col not in df_features.columns:
            df_features[col] = 0.0

    df_features = df_features[FEATURE_COLS].fillna(0.0)
    return df_features.values


# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Phishing URL Detector")
st.markdown(
    "Enter a URL to check, or upload a dataset to train the model. "
    "This model uses **Hybrid Features** (character n-grams + URL metrics) "
    "for better accuracy against lookalike and obfuscated URLs."
)

p = Path(__file__).parent
model_path = p / "phish_model.joblib"

# Left column: single-url check
col1, col2 = st.columns([1, 1])

# =========================
# 🔗 Single URL prediction
# =========================
with col1:
    st.header("🔗 Check a single URL")
    url = st.text_input("Enter URL", "http://micros0ft.com/login")

    if st.button("Check URL"):
        # Train a quick demo model if none exists yet
        if not model_path.exists():
            st.info(
                "Model not found. Training a small demo model with the Hybrid Pipeline..."
            )
            train_model(None)  # uses sample_phishing.csv internally

        model = joblib.load(model_path)
        pred = model.predict([url])[0]
        prob = model.predict_proba([url])[0]
        label = "Phishing" if pred == 1 else "Legitimate"
        confidence = float(np.max(prob))

        if pred == 1:
            st.error(f"⚠️ Likely PHISHING ({confidence * 100:.1f}% confidence)")
        else:
            st.success(f"✅ Likely legitimate ({confidence * 100:.1f}% confidence)")

        st.subheader("Why the model thinks so (Key URL Features)")
        features_dict = extract_url_features(url)
        # Hide TLD string just to keep the display compact
        st.json({k: v for k, v in features_dict.items() if k != "tld"})


# =========================
# 📁 Upload / train section
# =========================
with col2:
    st.header("📁 Upload dataset to train (optional)")
    st.markdown("CSV format: two columns — `url`, `label` (0 = legit, 1 = phishing)")

    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
    use_sample = st.checkbox(
        "Use example dataset (recommended for quick demo)", value=True
    )

    if uploaded is not None:
        # Only wrap the CSV read in this try/except so error message is accurate
        try:
            df = pd.read_csv(uploaded, engine="python")
        except Exception as e:
            st.error(
                "Failed to read CSV file. Ensure it is a valid CSV. Error: " + str(e)
            )
        else:
            if "url" not in df.columns or "label" not in df.columns:
                st.error("CSV must contain `url` and `label` columns.")
            else:
                st.success(f"Dataset loaded: {len(df)} rows")
                st.dataframe(df.head(10))

                if st.button("Train model on uploaded dataset"):
                    tmp = p / "uploaded_dataset.csv"
                    df.to_csv(tmp, index=False)

                    with st.spinner("Training model on uploaded dataset..."):
                        # train_model will do its own robust loading + cleaning
                        train_model(str(tmp))

                    st.success("Model trained and saved as phish_model.joblib.")
                    model_path = p / "phish_model.joblib"
    else:
        if use_sample:
            df = pd.read_csv(p / "sample_phishing.csv", engine="python")
            st.write("Using built-in demo dataset. Sample:")
            st.dataframe(df.head(10))


st.markdown("---")

# =========================
# 📊 Evaluation section
# =========================
st.header("📊 Model Evaluation (on demo test split)")

if st.button("Show evaluation on demo dataset"):
    df = pd.read_csv(p / "sample_phishing.csv", engine="python")
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    X = df["url"]
    y = df["label"]

    # Hybrid pipeline (must match train_and_save.py)
    numeric_features_pipe = Pipeline(
        [
            ("extractor", FunctionTransformer(get_numeric_features, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    text_features_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6))),
        ]
    )

    preprocessor = FeatureUnion(
        [
            ("text", text_features_pipe),
            ("numeric", numeric_features_pipe),
        ]
    )

    pipe = Pipeline(
        [
            ("union", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )

    # Train/Test Split and Evaluation
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(Xtrain, ytrain)
    ypred = pipe.predict(Xtest)
    yprob = pipe.predict_proba(Xtest)[:, 1]
    acc = accuracy_score(ytest, ypred)

    st.metric("Accuracy on demo test split (Hybrid Model)", f"{acc * 100:.2f}%")

    st.subheader("Classification report")
    st.text(classification_report(ytest, ypred))

    # Confusion matrix
    cm = confusion_matrix(ytest, ypred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix (rows = true, cols = pred)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    st.pyplot(fig, use_container_width=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(ytest, yprob)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.set_title(f"ROC curve (AUC = {roc_auc:.2f})")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    st.pyplot(fig2, use_container_width=True)

st.markdown("---")
st.info(
    "Tips: The **Host Entropy** feature highlights randomly generated domains. "
    "The **Has Non-ASCII** and **Has Punycode** features help catch homograph/lookalike attacks."
)
