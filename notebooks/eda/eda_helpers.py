"""Fonctions utilitaires pour l'EDA du dataset Rakuten.

Fournit des fonctions statistiques, de streaming et de visualisation
pour l'analyse exploratoire du dataset de classification hierarchique.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def gini_coefficient(values: np.ndarray) -> float:
    """Calcule le coefficient de Gini (0 = égalité parfaite, 1 = inegalite max)."""
    sorted_v = np.sort(np.asarray(values, dtype=float))
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_v) / (n * np.sum(sorted_v))) - (n + 1) / n)


def lorenz_curve(values: np.ndarray, ax: plt.Axes, label: str, color: str = "#3498db") -> None:
    """Trace la courbe de Lorenz sur un axe matplotlib."""
    sorted_v = np.sort(np.asarray(values, dtype=float))
    cumulative = np.cumsum(sorted_v) / np.sum(sorted_v)
    x = np.linspace(0, 1, len(cumulative))
    ax.plot(x, cumulative, label=label, color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Égalité parfaite")
    ax.fill_between(x, cumulative, x, alpha=0.1, color=color)
    ax.set_xlabel("Proportion cumulée des classes")
    ax.set_ylabel("Proportion cumulée des échantillons")
    ax.legend()


def head_medium_tail(counts: pd.Series, thresholds: tuple = (1000, 100)) -> pd.DataFrame:
    """Classifie les catégories en Head / Medium / Tail."""
    high, low = thresholds
    data = {
        f"Head (>={high})": [(counts >= high).sum(), counts[counts >= high].sum()],
        f"Medium ({low}-{high - 1})": [
            ((counts >= low) & (counts < high)).sum(),
            counts[(counts >= low) & (counts < high)].sum(),
        ],
        f"Tail (<{low})": [(counts < low).sum(), counts[counts < low].sum()],
    }
    df = pd.DataFrame(data, index=["Nb classes", "Nb échantillons"]).T
    df["% classes"] = (df["Nb classes"] / df["Nb classes"].sum() * 100).round(1)
    df["% échantillons"] = (df["Nb échantillons"] / df["Nb échantillons"].sum() * 100).round(1)
    return df


# ---------------------------------------------------------------------------
# Streaming (pour le dataset complet sans tout charger en RAM)
# ---------------------------------------------------------------------------

def stream_label_counts(file_path: str) -> tuple[Counter, Counter]:
    """Compte les labels L1 et L2 en streaming sur un fichier JSONL."""
    l1_counts: Counter = Counter()
    l2_counts: Counter = Counter()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                l1 = record.get("typelabel", "Unknown")
                l2_raw = record.get("producttype", [])
                if isinstance(l2_raw, list) and l2_raw:
                    l2 = str(l2_raw[0]).strip()
                elif isinstance(l2_raw, str):
                    l2 = l2_raw.strip()
                else:
                    l2 = ""
                if l1:
                    l1_counts[l1] += 1
                if l2 and l2.lower() not in ("", "none", "nan", "unknown"):
                    l2_counts[l2] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return l1_counts, l2_counts


# ---------------------------------------------------------------------------
# Analyse textuelle
# ---------------------------------------------------------------------------

def get_top_words(
    texts: pd.Series,
    n: int = 20,
    stopwords: Optional[set] = None,
    min_len: int = 3,
) -> list[tuple[str, int]]:
    """Retourne les n mots les plus frequents (hors stopwords)."""
    counter: Counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        for word in re.findall(r"\b\w+\b", text.lower()):
            if len(word) >= min_len and (stopwords is None or word not in stopwords):
                counter[word] += 1
    return counter.most_common(n)


def compute_token_lengths(
    texts: list[str],
    tokenizer,
    batch_size: int = 1000,
) -> np.ndarray:
    """Calcule le nombre de tokens pour chaque texte (par batch pour la vitesse)."""
    lengths = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False, padding=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.array(lengths)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_distribution_grid(
    series_list: list[pd.Series],
    titles: list[str],
    figsize: tuple = (16, 10),
    bins: int = 50,
    color: str = "#3498db",
) -> plt.Figure:
    """Grille d'histogrammes avec lignes mean/median."""
    n = len(series_list)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for ax, s, title in zip(axes, series_list, titles):
        ax.hist(s.dropna(), bins=bins, color=color, alpha=0.7, edgecolor="white")
        mean_val = s.mean()
        med_val = s.median()
        ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Moyenne: {mean_val:.0f}")
        ax.axvline(med_val, color="#2ecc71", linestyle="-", linewidth=1.5, label=f"Médiane: {med_val:.0f}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylabel("Fréquence")

    # Masquer les axes inutilises
    for i in range(n, len(list(axes))):
        axes[i].set_visible(False)

    fig.tight_layout()
    return fig


def plot_category_barh(
    counts: pd.Series,
    title: str,
    top_n: int = 30,
    figsize: tuple = (12, 10),
    color: str = "#3498db",
    log_scale: bool = False,
) -> plt.Figure:
    """Bar chart horizontal des top_n catégories."""
    fig, ax = plt.subplots(figsize=figsize)
    data = counts.head(top_n).iloc[::-1]
    bars = ax.barh(range(len(data)), data.values, color=color, alpha=0.8)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Nombre d'échantillons")
    if log_scale:
        ax.set_xscale("log")

    # Annotations
    for bar, val in zip(bars, data.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {val:,.0f}", va="center", fontsize=8, color="#555")

    fig.tight_layout()
    return fig


def plot_word_clouds(
    texts_by_category: dict[str, str],
    stopwords: set,
    ncols: int = 3,
    figsize: tuple = (18, 10),
) -> plt.Figure:
    """Grille de word clouds par catégorie."""
    from wordcloud import WordCloud

    categories = list(texts_by_category.keys())
    nrows = (len(categories) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    axes_flat = np.array(axes).flat

    for ax, cat in zip(axes_flat, categories):
        wc = WordCloud(
            width=500, height=350, background_color="white",
            stopwords=stopwords, max_words=80,
            colormap="viridis", prefer_horizontal=0.7,
        ).generate(texts_by_category[cat])
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(cat, fontsize=11, fontweight="bold")
        ax.axis("off")

    for i in range(len(categories), nrows * ncols):
        axes_flat[i].set_visible(False)

    fig.suptitle("Nuages de mots par catégorie L1", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_tfidf_top_terms(
    tfidf_scores: dict[str, list[tuple[str, float]]],
    ncols: int = 2,
    figsize: tuple = (16, 20),
) -> plt.Figure:
    """Grille de bar charts des termes TF-IDF les plus discriminants par catégorie."""
    categories = list(tfidf_scores.keys())
    nrows = (len(categories) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flat

    colors = sns.color_palette("husl", len(categories))
    for ax, cat, color in zip(axes_flat, categories, colors):
        terms, scores = zip(*tfidf_scores[cat])
        y_pos = range(len(terms))
        ax.barh(y_pos, scores, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9)
        ax.set_title(cat, fontsize=10, fontweight="bold")
        ax.invert_yaxis()

    for i in range(len(categories), nrows * ncols):
        axes_flat[i].set_visible(False)

    fig.suptitle("Termes les plus discriminants par catégorie (TF-IDF)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def styled_summary_table(df: pd.DataFrame, caption: str = "") -> None:
    """Affiche un DataFrame avec un style professionnel."""
    from IPython.display import display

    styled = (
        df.style
        .set_properties(**{"text-align": "left", "font-size": "12px"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#2c3e50"), ("color", "white"),
                                          ("font-size", "12px"), ("text-align", "left")]},
            {"selector": "td", "props": [("border-bottom", "1px solid #eee")]},
        ])
    )
    if caption:
        styled = styled.set_caption(caption)
    display(styled)
