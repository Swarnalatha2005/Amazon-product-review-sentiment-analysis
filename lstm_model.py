"""
lstm_model.py
LSTM-based deep learning model for sentiment classification.
Requires TensorFlow/Keras. This file is independent and runnable
provided tensorflow is installed.
"""

import numpy as np

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

MAX_WORDS = 5000
MAX_LEN = 100
EMBEDDING_DIM = 128


def build_tokenizer(texts):
    """Fit a Keras Tokenizer on the provided text corpus."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(tokenizer, texts):
    """Convert raw texts into padded integer sequences."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")


def build_lstm_model(num_classes: int = 3):
    """Build and compile the LSTM model architecture."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")

    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.3),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train_lstm_model(texts, labels, num_classes: int = 3, epochs: int = 5, batch_size: int = 32):
    """
    Full LSTM training pipeline.
    `labels` should be integer-encoded (0=Negative, 1=Neutral, 2=Positive).
    Returns the trained model and the fitted tokenizer.
    """
    tokenizer = build_tokenizer(texts)
    X = texts_to_padded_sequences(tokenizer, texts)
    y = to_categorical(np.array(labels), num_classes=num_classes)

    model = build_lstm_model(num_classes=num_classes)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return model, tokenizer


def predict_lstm(model, tokenizer, text: str) -> str:
    """Predict sentiment for a single text using a trained LSTM model."""
    from preprocess import preprocess_text
    cleaned = preprocess_text(text)
    seq = texts_to_padded_sequences(tokenizer, [cleaned])
    pred = model.predict(seq, verbose=0)
    label = int(np.argmax(pred, axis=1)[0])
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return mapping[label]


if __name__ == "__main__":
    if TENSORFLOW_AVAILABLE:
        print("TensorFlow is available. LSTM model can be trained via train_model.py")
    else:
        print("TensorFlow is NOT installed. Install it with: pip install tensorflow")