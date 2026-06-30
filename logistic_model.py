"""
logistic_model.py
Logistic Regression model wrapper for sentiment classification.
This is the PRIMARY model used by the application.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Create and return a TF-IDF vectorizer instance."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def build_logistic_model() -> LogisticRegression:
    """Create and return a Logistic Regression classifier."""
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        multi_class="auto",
        solver="lbfgs",
        random_state=42,
    )


def train_logistic_model(X_train, y_train):
    """Train the Logistic Regression model and return the fitted model."""
    model = build_logistic_model()
    model.fit(X_train, y_train)
    return model