"""
train_model.py
Trains the primary Logistic Regression sentiment model using TF-IDF features.
Evaluates the model and saves both the model and vectorizer to disk.

Run this file with:  python train_model.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess_dataframe
from logistic_model import build_vectorizer, train_logistic_model
from utils import save_object, ensure_dir, text_to_label

DATA_PATH = "data/amazon_reviews.csv"
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"
GRAPH_DIR = "outputs/graphs"


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the Amazon reviews dataset. Raises a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please place your CSV file there. "
            f"Required columns: 'review', 'sentiment'."
        )
    df = pd.read_csv(path)
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns.")
    return df


def plot_confusion_matrix(cm, labels, save_path: str):
    """Plot and save a confusion matrix heatmap."""
    ensure_dir(GRAPH_DIR)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")


def main():
    print("[INFO] Loading dataset...")
    df = load_dataset()

    print("[INFO] Preprocessing reviews...")
    df = preprocess_dataframe(df, text_column="review")
    df["label"] = df["sentiment"].apply(text_to_label)

    print("[INFO] Splitting data into train/test sets...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["cleaned_review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("[INFO] Vectorizing text using TF-IDF...")
    vectorizer = build_vectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("[INFO] Training Logistic Regression model...")
    model = train_logistic_model(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=["Negative", "Neutral", "Positive"], zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)
    ensure_dir(GRAPH_DIR)
    plot_confusion_matrix(cm, ["Negative", "Neutral", "Positive"],
                           os.path.join(GRAPH_DIR, "confusion_matrix.png"))

    print("[INFO] Saving model and vectorizer...")
    save_object(model, MODEL_PATH)
    save_object(vectorizer, VECTORIZER_PATH)

    print("[SUCCESS] Training complete. Model is ready for predictions.")


if __name__ == "__main__":
    main()