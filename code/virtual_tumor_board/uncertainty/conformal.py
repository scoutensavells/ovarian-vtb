from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    quantile: float
    calibration_size: int
    labels: tuple[str, ...]

    def prediction_set(self, probabilities: FloatArray) -> tuple[str, ...]:
        if probabilities.ndim != 1 or probabilities.size != len(self.labels):
            raise ValueError("probabilities must align with calibration labels")
        selected = [
            label
            for label, probability in zip(self.labels, probabilities, strict=True)
            if 1.0 - probability <= self.quantile
        ]
        return tuple(selected)


def nonconformity_scores(probabilities: FloatArray, true_indices: NDArray[np.int64]) -> FloatArray:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    if true_indices.ndim != 1 or len(true_indices) != len(probabilities):
        raise ValueError("true indices must align with probability rows")
    if np.any(true_indices < 0) or np.any(true_indices >= probabilities.shape[1]):
        raise ValueError("true indices are out of bounds")
    return 1.0 - probabilities[np.arange(len(probabilities)), true_indices]


def finite_sample_quantile(scores: FloatArray, alpha: float) -> float:
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a nonempty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    rank = ceil((scores.size + 1) * (1.0 - alpha))
    rank = min(rank, scores.size)
    return float(np.sort(scores)[rank - 1])


def fit_split_conformal(
    probabilities: FloatArray,
    true_indices: NDArray[np.int64],
    labels: tuple[str, ...],
    alpha: float = 0.10,
) -> ConformalCalibration:
    if probabilities.shape[1] != len(labels):
        raise ValueError("labels must align with probability columns")
    scores = nonconformity_scores(probabilities, true_indices)
    quantile = finite_sample_quantile(scores, alpha)
    return ConformalCalibration(
        alpha=alpha,
        quantile=quantile,
        calibration_size=len(scores),
        labels=labels,
    )


def prediction_sets(
    calibration: ConformalCalibration,
    probabilities: FloatArray,
) -> tuple[tuple[str, ...], ...]:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    return tuple(calibration.prediction_set(row) for row in probabilities)


def empirical_coverage(sets: tuple[tuple[str, ...], ...], true_labels: tuple[str, ...]) -> float:
    if len(sets) != len(true_labels) or not sets:
        raise ValueError("prediction sets and labels must align and be nonempty")
    return sum(label in values for values, label in zip(sets, true_labels, strict=True)) / len(sets)


def set_size_summary(sets: tuple[tuple[str, ...], ...]) -> dict[str, float]:
    if not sets:
        raise ValueError("prediction sets must not be empty")
    sizes = np.asarray([len(values) for values in sets], dtype=np.float64)
    return {
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
        "minimum": float(np.min(sizes)),
        "maximum": float(np.max(sizes)),
        "singleton_rate": float(np.mean(sizes == 1.0)),
        "review_rate": float(np.mean(sizes >= 2.0)),
    }


def coverage_gap(observed: float, alpha: float) -> float:
    if not 0.0 <= observed <= 1.0:
        raise ValueError("observed coverage must be between zero and one")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    return observed - (1.0 - alpha)
