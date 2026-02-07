"""GPU-accelerated sentiment classifier for crypto news.

Uses FinBERT (ProsusAI/finbert) for financial sentiment classification.
Runs on T4 GPU (~5ms per headline) with batching support.

Outputs: (sentiment_label, score, confidence) per headline.
  - sentiment_label: "positive", "negative", "neutral"
  - score: float in [-1, +1] (negative to positive)
  - confidence: float in [0, 1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"


@dataclass(frozen=True)
class SentimentResult:
    """Sentiment classification result for a single text."""

    text: str
    label: str           # "positive", "negative", "neutral"
    score: float         # -1 to +1
    confidence: float    # 0 to 1


class SentimentClassifier:
    """FinBERT-based financial sentiment classifier.

    Loads model onto GPU if available, falls back to CPU.
    Supports single and batched inference.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading %s on %s...", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # FinBERT label mapping: 0=positive, 1=negative, 2=neutral
        self.labels = ["positive", "negative", "neutral"]
        logger.info("Model loaded: %s", model_name)

    @torch.no_grad()
    def classify(self, texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
        """Classify a batch of texts.

        Returns list of SentimentResult, one per input text.
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            for j, text in enumerate(batch):
                p = probs[j]
                pos, neg, neu = p[0].item(), p[1].item(), p[2].item()

                # Score: +1 = fully positive, -1 = fully negative
                score = pos - neg

                # Label is the argmax
                label_idx = p.argmax().item()
                label = self.labels[label_idx]

                # Confidence is the max probability
                confidence = p.max().item()

                results.append(SentimentResult(
                    text=text,
                    label=label,
                    score=score,
                    confidence=confidence,
                ))

        return results

    def classify_one(self, text: str) -> SentimentResult:
        """Classify a single text."""
        return self.classify([text])[0]


# ---------------------------------------------------------------------------
# Singleton for reuse across the app
# ---------------------------------------------------------------------------

_classifier: SentimentClassifier | None = None


def get_classifier() -> SentimentClassifier:
    """Get or create the global sentiment classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = SentimentClassifier()
    return _classifier
