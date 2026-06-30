"""
bert_model.py
BERT-based transformer model for sentiment classification using
HuggingFace Transformers. This file is independent and runnable
provided `transformers` and `torch` are installed.
"""

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 3  # Negative, Neutral, Positive


def load_bert_tokenizer():
    """Load the pretrained BERT tokenizer."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers/torch not installed. Run: pip install transformers torch")
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_bert_model(num_labels: int = NUM_LABELS):
    """Load a pretrained BERT model configured for sequence classification."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers/torch not installed. Run: pip install transformers torch")
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )


class SentimentDataset(torch.utils.data.Dataset if TRANSFORMERS_AVAILABLE else object):
    """PyTorch Dataset wrapper for tokenized text + labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_bert_model(texts, labels, output_dir: str = "model/bert_sentiment", epochs: int = 2):
    """
    Fine-tune BERT on the provided texts and integer labels.
    NOTE: This is computationally expensive and works best with a GPU.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers/torch not installed. Run: pip install transformers torch")

    tokenizer = load_bert_tokenizer()
    encodings = tokenizer(
        list(texts), truncation=True, padding=True, max_length=128
    )
    dataset = SentimentDataset(encodings, list(labels))
    model = load_bert_model()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


def predict_bert(model, tokenizer, text: str) -> str:
    """Predict sentiment for a single text using a fine-tuned BERT model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return mapping[pred]


if __name__ == "__main__":
    if TRANSFORMERS_AVAILABLE:
        print("Transformers/torch available. BERT model can be trained via train_model.py")
    else:
        print("transformers/torch NOT installed. Install with: pip install transformers torch")