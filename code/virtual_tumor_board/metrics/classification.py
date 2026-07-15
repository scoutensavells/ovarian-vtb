from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ClassificationSummary:
    accuracy: float
    macro_auc: float
    count: int
    correct: int


def accuracy_score(targets: IntArray, predictions: IntArray) -> float:
    _validate_vectors(targets, predictions)
    return float(np.mean(targets == predictions))


def guideline_concordance(reference: NDArray[np.bool_], recommended: NDArray[np.bool_]) -> float:
    _validate_vectors(reference, recommended)
    return float(np.mean(reference == recommended))


def macro_one_vs_rest_auc(targets: IntArray, probabilities: FloatArray) -> float:
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.size:
        raise ValueError("probabilities must align with targets")
    classes = np.arange(probabilities.shape[1])
    observed = np.unique(targets)
    if not np.array_equal(observed, classes):
        raise ValueError("all classes must occur in the target vector")
    return float(roc_auc_score(targets, probabilities, average="macro", multi_class="ovr"))


def classification_summary(
    targets: IntArray,
    predictions: IntArray,
    probabilities: FloatArray,
) -> ClassificationSummary:
    accuracy = accuracy_score(targets, predictions)
    auc = macro_one_vs_rest_auc(targets, probabilities)
    return ClassificationSummary(
        accuracy=accuracy,
        macro_auc=auc,
        count=len(targets),
        correct=int(np.sum(targets == predictions)),
    )


def per_class_accuracy(targets: IntArray, predictions: IntArray) -> dict[int, float]:
    _validate_vectors(targets, predictions)
    result = {}
    for value in np.unique(targets):
        mask = targets == value
        result[int(value)] = float(np.mean(predictions[mask] == value))
    return result


def confusion_counts(targets: IntArray, predictions: IntArray) -> dict[tuple[int, int], int]:
    _validate_vectors(targets, predictions)
    counts: dict[tuple[int, int], int] = {}
    for target, prediction in zip(targets, predictions, strict=True):
        key = (int(target), int(prediction))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _validate_vectors(left: NDArray[np.generic], right: NDArray[np.generic]) -> None:
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("inputs must be vectors")
    if left.size == 0 or left.size != right.size:
        raise ValueError("inputs must align and be nonempty")
