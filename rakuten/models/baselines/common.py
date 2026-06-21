"""Utilitaires partagés entre les pipelines baselines (TF-IDF, Word2Vec, BERT)."""

from __future__ import annotations

import nltk
import pandas as pd


def get_stopwords() -> set[str]:
    """Return combined French + English stopword set."""
    for resource in ("corpora/stopwords",):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("french") + stopwords.words("english"))


def filter_rare_classes(
    df: pd.DataFrame,
    label_column: str,
    min_samples: int = 5,
    tag: str = "",
) -> pd.DataFrame:
    """Remove classes with fewer than *min_samples* samples."""
    counts = df[label_column].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if rare:
        n_before = len(df)
        df = df[~df[label_column].isin(rare)].copy()
        prefix = f"[{tag}] " if tag else ""
        print(f"{prefix}Removed {len(rare)} rare classes ({n_before - len(df)} samples)")
    return df
