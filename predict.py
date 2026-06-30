"""
predict.py
Loads the trained Logistic Regression model and vectorizer to make
sentiment predictions on new text or batches of reviews.
"""

import pandas as pd
from preprocess import preprocess_text, preprocess_dataframe
from utils import load_object, label_to_text

MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

_model = None
_vectorizer = None


def _load_artifacts():
    """Lazy-load the trained model and vectorizer (cached after first call)."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = load_object(MODEL_PATH)
        _vectorizer = load_object(VECTORIZER_PATH)
    return _model, _vectorizer


def predict_single_review(review_text: str) -> str:
    """Predict sentiment for one review sentence. Returns 'Positive'/'Negative'/'Neutral'."""
    if not review_text or not review_text.strip():
        raise ValueError("Review text cannot be empty.")

    model, vectorizer = _load_artifacts()
    cleaned = preprocess_text(review_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return label_to_text(prediction)


def predict_batch_reviews(df: pd.DataFrame, text_column: str = "review") -> pd.DataFrame:
    """
    Predict sentiments for an entire DataFrame of reviews.
    Returns the DataFrame with an added 'predicted_sentiment' column.
    """
    model, vectorizer = _load_artifacts()
    df = preprocess_dataframe(df, text_column=text_column)
    vectors = vectorizer.transform(df["cleaned_review"])
    predictions = model.predict(vectors)
    df["predicted_sentiment"] = [label_to_text(p) for p in predictions]
    return df


if __name__ == "__main__":
    sample_review = "The battery backup is excellent."
    try:
        result = predict_single_review(sample_review)
        print(f"Review: {sample_review}")
        print(f"Predicted Sentiment: {result}")
    except FileNotFoundError:
        print("[ERROR] Model not found. Please run 'python train_model.py' first.")