"""
visualization.py
Generates visualizations (bar chart, pie chart, count plot, word frequency, etc.)
for batch sentiment prediction results, using Matplotlib and Seaborn.
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import ensure_dir

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
SENTIMENT_COLORS = {"Negative": "#E74C3C", "Neutral": "#F1C40F", "Positive": "#2ECC71"}


def plot_bar_chart(df: pd.DataFrame, save_path: str, column: str = "predicted_sentiment"):
    """Bar chart of sentiment counts."""
    counts = df[column].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    plt.figure(figsize=(6, 5))
    sns.barplot(x=counts.index, y=counts.values,
                palette=[SENTIMENT_COLORS[s] for s in counts.index])
    plt.title("Sentiment Distribution (Bar Chart)")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pie_chart(df: pd.DataFrame, save_path: str, column: str = "predicted_sentiment"):
    """Pie chart of sentiment percentage breakdown."""
    counts = df[column].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[SENTIMENT_COLORS[s] for s in counts.index],
        startangle=90,
    )
    plt.title("Sentiment Distribution (Pie Chart)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_count_plot(df: pd.DataFrame, save_path: str, column: str = "predicted_sentiment"):
    """Seaborn count plot of sentiment classes."""
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x=column, order=SENTIMENT_ORDER,
                   palette=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
    plt.title("Sentiment Count Plot")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_review_length_distribution(df: pd.DataFrame, save_path: str,
                                     text_column: str = "review", hue_column: str = "predicted_sentiment"):
    """Distribution graph of review text lengths split by sentiment."""
    df = df.copy()
    df["review_length"] = df[text_column].astype(str).apply(len)
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="review_length", hue=hue_column, hue_order=SENTIMENT_ORDER,
                 kde=True, palette=SENTIMENT_COLORS, element="step")
    plt.title("Review Length Distribution by Sentiment")
    plt.xlabel("Review Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _get_top_words(texts, top_n: int = 15):
    """Helper: return the top N most frequent words from a list of texts."""
    words = []
    for text in texts:
        words.extend(re.findall(r"\b[a-z]+\b", str(text).lower()))
    return Counter(words).most_common(top_n)


def plot_top_words_by_sentiment(df: pd.DataFrame, sentiment: str, save_path: str,
                                 text_column: str = "cleaned_review",
                                 sentiment_column: str = "predicted_sentiment"):
    """Bar chart of the most common words for a specific sentiment class."""
    subset = df[df[sentiment_column] == sentiment]
    if subset.empty:
        return
    top_words = _get_top_words(subset[text_column], top_n=15)
    if not top_words:
        return
    words, counts = zip(*top_words)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=list(counts), y=list(words), color=SENTIMENT_COLORS.get(sentiment, "#3498DB"))
    plt.title(f"Most Common Words — {sentiment} Reviews")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_all_visualizations(df: pd.DataFrame, output_dir: str = "outputs/graphs"):
    """
    Generate the full suite of visualizations for a batch-predicted DataFrame
    and save them as PNG files in `output_dir`.
    Returns a dict mapping graph name -> file path.
    """
    ensure_dir(output_dir)
    paths = {}

    paths["bar_chart"] = os.path.join(output_dir, "bar_chart.png")
    plot_bar_chart(df, paths["bar_chart"])

    paths["pie_chart"] = os.path.join(output_dir, "pie_chart.png")
    plot_pie_chart(df, paths["pie_chart"])

    paths["count_plot"] = os.path.join(output_dir, "count_plot.png")
    plot_count_plot(df, paths["count_plot"])

    paths["length_distribution"] = os.path.join(output_dir, "length_distribution.png")
    plot_review_length_distribution(df, paths["length_distribution"])

    for sentiment in SENTIMENT_ORDER:
        key = f"top_words_{sentiment.lower()}"
        path = os.path.join(output_dir, f"top_words_{sentiment.lower()}.png")
        plot_top_words_by_sentiment(df, sentiment, path)
        if os.path.exists(path):
            paths[key] = path

    print(f"[INFO] All visualizations saved to: {output_dir}")
    return paths