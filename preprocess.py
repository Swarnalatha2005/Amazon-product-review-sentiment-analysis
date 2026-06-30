"""
preprocess.py
Text preprocessing utilities using NLTK:
lowercasing, punctuation removal, tokenization,
stopword removal, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (safe to call repeatedly)
NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
]

for resource in NLTK_RESOURCES:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Keep negation words — important for sentiment analysis
NEGATIONS = {"not", "no", "nor", "never", "none"}
STOP_WORDS = STOP_WORDS - NEGATIONS


def clean_text(text: str) -> str:
    """Lowercase and remove punctuation/special characters/numbers."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"<.*?>", "", text)                    # remove HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)                      # remove digits
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text


def tokenize_text(text: str) -> list:
    """Tokenize cleaned text into individual words."""
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def remove_stopwords(tokens: list) -> list:
    """Remove common English stopwords (negations preserved)."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize_tokens(tokens: list) -> list:
    """Lemmatize each token to its base form."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    clean -> tokenize -> remove stopwords -> lemmatize -> rejoin
    Returns a single preprocessed string ready for TF-IDF.
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess_dataframe(df, text_column: str = "review"):
    """Apply preprocessing to an entire DataFrame column."""
    df = df.copy()
    df[text_column] = df[text_column].fillna("")
    df["cleaned_review"] = df[text_column].apply(preprocess_text)
    return df


if __name__ == "__main__":
    sample = "The product quality is AMAZING!!! Delivery was not fast at all, 100% disappointed."
    print("Original:", sample)
    print("Preprocessed:", preprocess_text(sample))