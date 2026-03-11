"""TF-IDF + Linear SVM pipeline for Rakuten product classification.

Trains a TfidfVectorizer followed by a LinearSVC classifier. Supports
training on product title only or title + description. Evaluation uses
F1-macro and F1-weighted scores.

Usage:
    python tfidf_pipeline.py --data_path dataset/train.jsonl --n_samples 50000
    python tfidf_pipeline.py --data_path dataset/train.jsonl --use_description
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import joblib
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Cross-directory import from shared/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.data_loader import load_random_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_stopwords() -> list[str]:
    """Return combined French + English stopword list."""
    for resource in ("corpora/stopwords",):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    return list(set(stopwords.words("french") + stopwords.words("english")))


def _prepare_text(
    df: pd.DataFrame,
    use_description: bool = False,
) -> pd.Series:
    """Build the text feature column from designation and optionally description.

    Args:
        df: DataFrame with at least 'designation' column. May also have
            'description' and 'producttype'.
        use_description: If True, concatenate description to the text.

    Returns:
        Series of text strings ready for vectorisation.
    """
    df = df.copy()
    df["designation"] = df["designation"].fillna("")

    parts = [df["designation"]]

    if "producttype" in df.columns:
        df["_clean_type"] = df["producttype"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else (str(x) if x else "")
        )
        parts.append(df["_clean_type"])

    if use_description and "description" in df.columns:
        df["description"] = df["description"].fillna("")
        parts.append(df["description"])

    return pd.Series(
        [" ".join(t) for t in zip(*[p.astype(str) for p in parts])],
        index=df.index,
    )


def _filter_rare_classes(
    df: pd.DataFrame,
    label_column: str,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Remove classes with fewer than *min_samples* samples."""
    counts = df[label_column].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if rare:
        n_before = len(df)
        df = df[~df[label_column].isin(rare)].copy()
        print(f"[TFIDF] Removed {len(rare)} rare classes ({n_before - len(df)} samples)")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def train_tfidf_pipeline(
    data_path: str,
    n_samples: int = 50_000,
    label_column: str = "typelabel",
    use_description: bool = False,
    max_features: int = 50_000,
    test_size: float = 0.2,
    save_dir: str = "models_tfidf",
    classifier: str = "linearsvc",
    seed: int = 42,
) -> dict:
    """Train a TF-IDF + linear classifier pipeline.

    Args:
        data_path: Path to the JSONL training file.
        n_samples: Number of samples to load.
        label_column: Column name for labels.
        use_description: Whether to include product description in text.
        max_features: Maximum vocabulary size for TF-IDF.
        test_size: Fraction of data for the test set.
        save_dir: Directory to save trained artefacts.
        classifier: One of 'linearsvc', 'logreg', 'sgd'.
        seed: Random seed.

    Returns:
        Dictionary with model, vectorizer, metrics, and predictions.
    """
    print("=" * 60)
    print("TF-IDF + Linear Classifier Pipeline")
    print("=" * 60)

    # --- Load data ---
    df = load_random_sample(data_path, n_samples=n_samples, seed=seed)
    df = _filter_rare_classes(df, label_column)

    # --- Build text feature ---
    text_col = _prepare_text(df, use_description=use_description)
    labels = df[label_column]
    num_classes = labels.nunique()
    print(f"[TFIDF] Classes: {num_classes} | Samples: {len(df):,}")
    print(f"[TFIDF] Use description: {use_description}")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        text_col, labels, test_size=test_size, random_state=seed, stratify=labels,
    )
    print(f"[TFIDF] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # --- TF-IDF vectorisation ---
    stop_words = _get_stopwords()
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )

    print(f"[TFIDF] Vectorising (max_features={max_features})...")
    t0 = time.time()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    print(f"[TFIDF] Vectorised in {time.time() - t0:.1f}s  |  vocab={len(tfidf.vocabulary_):,}")

    # --- Classifier ---
    classifiers = {
        "linearsvc": lambda: LinearSVC(
            max_iter=200, C=1.0, class_weight="balanced",
            dual="auto", random_state=seed,
        ),
        "logreg": lambda: LogisticRegression(
            max_iter=200, solver="saga", n_jobs=-1,
            class_weight="balanced", random_state=seed,
        ),
        "sgd": lambda: SGDClassifier(
            loss="log_loss", max_iter=200, alpha=1e-4,
            class_weight="balanced", n_jobs=-1, random_state=seed,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
        ),
    }

    if classifier not in classifiers:
        raise ValueError(f"Unknown classifier '{classifier}'. Choose from {list(classifiers)}")

    model = classifiers[classifier]()
    print(f"[TFIDF] Training {classifier}...")

    t0 = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test_vec)
    pred_time = time.time() - t0

    # --- Metrics ---
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"[TFIDF] Training time:    {train_time:.1f}s")
    print(f"[TFIDF] Prediction time:  {pred_time * 1000:.1f}ms")
    print(f"[TFIDF] F1-Macro:         {f1_macro:.4f}")
    print(f"[TFIDF] F1-Weighted:      {f1_weighted:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    # --- Save artefacts ---
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "model.joblib"))
    joblib.dump(tfidf, os.path.join(save_dir, "vectorizer.joblib"))

    config = {
        "classifier": classifier,
        "max_features": max_features,
        "use_description": use_description,
        "vocab_size": len(tfidf.vocabulary_),
        "num_classes": num_classes,
        "n_samples": n_samples,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "train_time_s": float(train_time),
        "pred_time_s": float(pred_time),
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Save predictions for later comparison
    preds_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    preds_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    print(f"[TFIDF] Artefacts saved to {save_dir}/")

    return {
        "model": model,
        "vectorizer": tfidf,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time": train_time,
        "pred_time": pred_time,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict(text: str, model_dir: str = "models_tfidf") -> str:
    """Load saved model and predict the category for a single text string.

    Args:
        text: Input product text.
        model_dir: Path to the directory containing saved artefacts.

    Returns:
        Predicted label string.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    tfidf = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    vec = tfidf.transform([text])
    return model.predict(vec)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TF-IDF + Linear Classifier for Rakuten product classification",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to load")
    parser.add_argument("--use_description", action="store_true", help="Include product description in text features")
    parser.add_argument("--max_features", type=int, default=50_000, help="TF-IDF max vocabulary size")
    parser.add_argument("--classifier", type=str, default="linearsvc", choices=["linearsvc", "logreg", "sgd"])
    parser.add_argument("--save_dir", type=str, default="models_tfidf", help="Output directory for artefacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Predict mode
    parser.add_argument("--predict", type=str, default=None, help="Text to classify (inference mode)")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory of saved model (for --predict)")

    args = parser.parse_args()

    if args.predict is not None:
        model_dir = args.model_dir or args.save_dir
        label = predict(args.predict, model_dir=model_dir)
        print(f"Prediction: {label}")
        return

    train_tfidf_pipeline(
        data_path=args.data_path,
        n_samples=args.n_samples,
        use_description=args.use_description,
        max_features=args.max_features,
        classifier=args.classifier,
        save_dir=args.save_dir,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
