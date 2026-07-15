from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BrierDecomposition:
    score: float
    reliability: float
    resolution: float
    uncertainty: float


def multiclass_brier_score(probabilities: FloatArray, targets: IntArray) -> float:
    _validate(probabilities, targets)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def binary_brier_score(probabilities: FloatArray, targets: IntArray) -> float:
    if probabilities.ndim != 1 or targets.ndim != 1:
        raise ValueError("inputs must be vectors")
    if probabilities.size == 0 or probabilities.size != targets.size:
        raise ValueError("inputs must align and be nonempty")
    if np.any((targets != 0) & (targets != 1)):
        raise ValueError("targets must be binary")
    return float(np.mean((probabilities - targets) ** 2))


def binary_brier_decomposition(
    probabilities: FloatArray,
    targets: IntArray,
    bins: int = 15,
) -> BrierDecomposition:
    score = binary_brier_score(probabilities, targets)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    base_rate = float(np.mean(targets))
    reliability = 0.0
    resolution = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(np.sum(mask))
        if count == 0:
            continue
        weight = count / len(targets)
        mean_probability = float(np.mean(probabilities[mask]))
        observed_rate = float(np.mean(targets[mask]))
        reliability += weight * (mean_probability - observed_rate) ** 2
        resolution += weight * (observed_rate - base_rate) ** 2
    uncertainty = base_rate * (1.0 - base_rate)
    return BrierDecomposition(score, reliability, resolution, uncertainty)


def _validate(probabilities: FloatArray, targets: IntArray) -> None:
    if probabilities.ndim != 2 or targets.ndim != 1:
        raise ValueError("probabilities must be a matrix and targets a vector")
    if probabilities.shape[0] == 0 or probabilities.shape[0] != targets.size:
        raise ValueError("inputs must align and be nonempty")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must be between zero and one")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
