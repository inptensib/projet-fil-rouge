"""Word2Vec + classifier pipeline for Rakuten product classification.

Trains (or loads) Word2Vec embeddings, computes document vectors by averaging
word vectors, then trains a linear classifier (LinearSVC, LogReg, or SGD).

Usage:
    python word2vec_pipeline.py --data_path dataset/train.jsonl --n_samples 50000
    python word2vec_pipeline.py --data_path dataset/train.jsonl --classifier sgd
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
from gensim.models import Word2Vec
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

from rakuten.common.data_loader import load_random_sample  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_stopwords() -> set[str]:
    """Return combined French + English stopword set."""
    from .common import get_stopwords
    return get_stopwords()


def _tokenize(text: str, stop_words: set[str]) -> list[str]:
    """Lowercase, split, remove stopwords and short tokens.

    Args:
        text: Raw input string.
        stop_words: Set of stopwords to exclude.

    Returns:
        List of cleaned tokens.
    """
    if not isinstance(text, str):
        text = str(text) if text else ""
    tokens = text.lower().split()
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def _prepare_text(df: pd.DataFrame) -> pd.Series:
    """Build the text feature column from designation + producttype.

    Args:
        df: DataFrame with 'designation' and optionally 'producttype'.

    Returns:
        Series of combined text strings.
    """
    df = df.copy()
    df["designation"] = df["designation"].fillna("")
    clean_type = df["producttype"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else (str(x) if x else "")
    ) if "producttype" in df.columns else ""
    return df["designation"] + " " + clean_type


def _document_vector(
    tokens: list[str],
    w2v_model: Word2Vec,
    vector_size: int,
) -> np.ndarray:
    """Compute a document vector by averaging its word vectors.

    Args:
        tokens: List of word tokens.
        w2v_model: Trained Word2Vec model.
        vector_size: Embedding dimensionality.

    Returns:
        Numpy array of shape (vector_size,).
    """
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vector_size)


def _filter_rare_classes(
    df: pd.DataFrame,
    label_column: str,
    min_samples: int = 5,
) -> pd.DataFrame:
    from .common import filter_rare_classes
    return filter_rare_classes(df, label_column, min_samples, tag="W2V")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def train_word2vec_pipeline(
    data_path: str,
    n_samples: int = 50_000,
    label_column: str = "typelabel",
    vector_size: int = 200,
    window: int = 5,
    min_count: int = 2,
    w2v_epochs: int = 10,
    test_size: float = 0.2,
    save_dir: str = "models_word2vec",
    classifier: str = "linearsvc",
    w2v_path: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Train a Word2Vec + linear classifier pipeline.

    Args:
        data_path: Path to the JSONL training file.
        n_samples: Number of samples to load.
        label_column: Column name for labels.
        vector_size: Dimensionality of Word2Vec embeddings.
        window: Context window size for Word2Vec.
        min_count: Minimum word frequency for Word2Vec vocabulary.
        w2v_epochs: Number of Word2Vec training epochs.
        test_size: Fraction of data for the test set.
        save_dir: Directory to save trained artefacts.
        classifier: One of 'linearsvc', 'logreg', 'sgd'.
        w2v_path: Optional path to a pre-trained Word2Vec model to load instead.
        seed: Random seed.

    Returns:
        Dictionary with model, w2v_model, metrics, and predictions.
    """
    print("=" * 60)
    print("Word2Vec + Linear Classifier Pipeline")
    print("=" * 60)

    # --- Load data ---
    df = load_random_sample(data_path, n_samples=n_samples, seed=seed)
    df = _filter_rare_classes(df, label_column)

    # --- Build text feature and tokenize ---
    stop_words = _get_stopwords()
    text_col = _prepare_text(df)
    df["tokens"] = text_col.apply(lambda t: _tokenize(t, stop_words))
    labels = df[label_column]
    num_classes = labels.nunique()
    print(f"[W2V] Classes: {num_classes} | Samples: {len(df):,}")

    # --- Word2Vec ---
    if w2v_path and os.path.exists(w2v_path):
        print(f"[W2V] Loading pre-trained Word2Vec from {w2v_path}")
        w2v_model = Word2Vec.load(w2v_path)
        w2v_train_time = 0.0
    else:
        print(f"[W2V] Training Word2Vec (dim={vector_size}, window={window}, "
              f"min_count={min_count}, epochs={w2v_epochs})...")
        t0 = time.time()
        w2v_model = Word2Vec(
            sentences=df["tokens"].tolist(),
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=os.cpu_count() or 4,
            sg=1,  # Skip-gram
            epochs=w2v_epochs,
            seed=seed,
        )
        w2v_train_time = time.time() - t0
        print(f"[W2V] Word2Vec trained in {w2v_train_time:.1f}s  |  "
              f"vocab={len(w2v_model.wv):,}")

    # --- Train/test split ---
    X_train_tok, X_test_tok, y_train, y_test = train_test_split(
        df["tokens"], labels, test_size=test_size, random_state=seed, stratify=labels,
    )
    print(f"[W2V] Train: {len(X_train_tok):,} | Test: {len(X_test_tok):,}")

    # --- Document vectorisation ---
    print("[W2V] Computing document vectors...")
    t0 = time.time()
    X_train_vec = np.array([
        _document_vector(tokens, w2v_model, vector_size)
        for tokens in X_train_tok
    ])
    X_test_vec = np.array([
        _document_vector(tokens, w2v_model, vector_size)
        for tokens in X_test_tok
    ])
    vec_time = time.time() - t0
    print(f"[W2V] Vectorised in {vec_time:.1f}s  |  shape={X_train_vec.shape}")

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
    print(f"[W2V] Training {classifier}...")

    t0 = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test_vec)
    pred_time = time.time() - t0

    # --- Metrics ---
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"[W2V] W2V training time:  {w2v_train_time:.1f}s")
    print(f"[W2V] Clf training time:  {train_time:.1f}s")
    print(f"[W2V] Prediction time:    {pred_time * 1000:.1f}ms")
    print(f"[W2V] F1-Macro:           {f1_macro:.4f}")
    print(f"[W2V] F1-Weighted:        {f1_weighted:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    # --- Save artefacts ---
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "model.joblib"))
    w2v_model.save(os.path.join(save_dir, "word2vec_model.bin"))

    config = {
        "classifier": classifier,
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "w2v_epochs": w2v_epochs,
        "w2v_vocab_size": len(w2v_model.wv),
        "num_classes": num_classes,
        "n_samples": n_samples,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "w2v_train_time_s": float(w2v_train_time),
        "clf_train_time_s": float(train_time),
        "pred_time_s": float(pred_time),
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    preds_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    preds_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    print(f"[W2V] Artefacts saved to {save_dir}/")

    return {
        "model": model,
        "w2v_model": w2v_model,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "w2v_train_time": w2v_train_time,
        "train_time": train_time,
        "pred_time": pred_time,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict(text: str, model_dir: str = "models_word2vec") -> str:
    """Load saved model and predict the category for a single text string.

    Args:
        text: Input product text.
        model_dir: Path to the directory containing saved artefacts.

    Returns:
        Predicted label string.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    w2v_model = Word2Vec.load(os.path.join(model_dir, "word2vec_model.bin"))

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    vector_size = config["vector_size"]

    stop_words = _get_stopwords()
    tokens = _tokenize(text, stop_words)
    vec = _document_vector(tokens, w2v_model, vector_size).reshape(1, -1)
    return model.predict(vec)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Word2Vec + Linear Classifier for Rakuten product classification",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--n_samples", type=int, default=50_000)
    parser.add_argument("--vector_size", type=int, default=200, help="Word2Vec embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec context window")
    parser.add_argument("--min_count", type=int, default=2, help="Min word frequency in Word2Vec")
    parser.add_argument("--w2v_epochs", type=int, default=10, help="Word2Vec training epochs")
    parser.add_argument("--w2v_path", type=str, default=None, help="Path to pre-trained Word2Vec model")
    parser.add_argument("--classifier", type=str, default="linearsvc", choices=["linearsvc", "logreg", "sgd"])
    parser.add_argument("--save_dir", type=str, default="models_word2vec")
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

    train_word2vec_pipeline(
        data_path=args.data_path,
        n_samples=args.n_samples,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        w2v_epochs=args.w2v_epochs,
        w2v_path=args.w2v_path,
        classifier=args.classifier,
        save_dir=args.save_dir,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
