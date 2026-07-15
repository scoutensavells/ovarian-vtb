from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AbstentionSummary:
    precision: float
    recall: float
    abstention_rate: float
    silent_failure_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int


def abstention_summary(needs_review: BoolArray, abstained: BoolArray) -> AbstentionSummary:
    if needs_review.ndim != 1 or abstained.ndim != 1:
        raise ValueError("inputs must be vectors")
    if needs_review.size == 0 or needs_review.size != abstained.size:
        raise ValueError("inputs must align and be nonempty")
    true_positives = int(np.sum(needs_review & abstained))
    false_positives = int(np.sum(~needs_review & abstained))
    false_negatives = int(np.sum(needs_review & ~abstained))
    predicted_positive = true_positives + false_positives
    actual_positive = true_positives + false_negatives
    precision = true_positives / predicted_positive if predicted_positive else 0.0
    recall = true_positives / actual_positive if actual_positive else 0.0
    high_confidence = ~abstained
    silent_failures = needs_review & high_confidence
    silent_failure_rate = (
        float(np.mean(silent_failures[high_confidence])) if np.any(high_confidence) else 0.0
    )
    return AbstentionSummary(
        precision=precision,
        recall=recall,
        abstention_rate=float(np.mean(abstained)),
        silent_failure_rate=silent_failure_rate,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def abstention_from_set_sizes(sizes: IntArray) -> BoolArray:
    if sizes.ndim != 1 or sizes.size == 0:
        raise ValueError("sizes must be a nonempty vector")
    if np.any(sizes < 1):
        raise ValueError("prediction sets must contain at least one label")
    return sizes >= 2


def disagreement_recall(errors: BoolArray, disagreements: BoolArray) -> float:
    if errors.shape != disagreements.shape or errors.ndim != 1:
        raise ValueError("vectors must align")
    error_count = int(np.sum(errors))
    return float(np.sum(errors & disagreements) / error_count) if error_count else 0.0
