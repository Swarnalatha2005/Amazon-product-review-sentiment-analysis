"""
utils.py
Common utility functions shared across the project.
"""

import os
import joblib


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_object(obj, path: str) -> None:
    """Save any Python object (model, vectorizer, etc.) using joblib."""
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)
    print(f"[INFO] Object saved to: {path}")


def load_object(path: str):
    """Load a Python object previously saved with joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at: {path}")
    return joblib.load(path)


def label_to_text(label) -> str:
    """Convert numeric label to readable sentiment text."""
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    if isinstance(label, str):
        return label
    return mapping.get(int(label), "Unknown")


def text_to_label(text: str) -> int:
    """Convert sentiment text to numeric label."""
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    return mapping.get(str(text).strip().lower(), 1)