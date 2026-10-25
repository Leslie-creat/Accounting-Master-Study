"""DistilBERT models via Hugging Face: zero-shot pipeline and fine-tuning."""

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)

SST2_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def pipeline_positive_probability(texts, model_name=SST2_MODEL, batch_size=32):
    """
    Zero-shot sentiment: a DistilBERT already fine-tuned on movie reviews
    (SST-2) is applied to the tweets without any further training.
    Returns the probability of the positive class for each text.
    """
    classifier = pipeline("sentiment-analysis", model=model_name)
    outputs = classifier(list(texts), truncation=True, batch_size=batch_size)
    return np.array([
        out["score"] if out["label"].upper() == "POSITIVE" else 1 - out["score"]
        for out in outputs
    ])


def _tokenize(tokenizer, texts, labels=None, max_length=128):
    data = {"text": list(texts)}
    if labels is not None:
        data["label"] = [int(label) for label in labels]
    dataset = Dataset.from_dict(data)
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )


def fine_tune_distilbert(
    train_texts,
    train_labels,
    model_name="distilbert-base-uncased",
    num_train_epochs=2,
    train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.0,
    max_length=128,
    seed=42,
    output_dir="./results",
):
    """
    Fine-tune DistilBERT for binary sentiment classification.

    Only training data is passed in; the test set is never seen during
    training. Returns (trainer, tokenizer) for use in predict_positive_probability.
    """
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = _tokenize(tokenizer, train_texts, train_labels, max_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    return trainer, tokenizer


def predict_positive_probability(trainer, tokenizer, texts, max_length=128):
    """Positive-class probabilities from a fine-tuned model."""
    dataset = _tokenize(tokenizer, texts, max_length=max_length)
    logits = trainer.predict(dataset).predictions
    # The model outputs raw logits, not probabilities; softmax converts them
    # so that the 0.5 cutoff in the evaluation is meaningful.
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]
