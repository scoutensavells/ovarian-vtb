from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    iterations: int


def percentile_interval(
    values: FloatArray,
    statistic: Callable[[FloatArray], float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> ConfidenceInterval:
    _validate(values, iterations, confidence)
    generator = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample_indices = generator.integers(0, len(values), size=len(values))
        estimates[index] = statistic(values[sample_indices])
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=float(statistic(values)),
        lower=float(np.quantile(estimates, tail)),
        upper=float(np.quantile(estimates, 1.0 - tail)),
        confidence=confidence,
        iterations=iterations,
    )


def paired_difference_interval(
    left: FloatArray,
    right: FloatArray,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> ConfidenceInterval:
    if left.ndim != 1 or right.ndim != 1 or left.size == 0 or left.size != right.size:
        raise ValueError("paired vectors must align and be nonempty")
    differences = left - right
    return percentile_interval(
        differences, lambda values: float(np.mean(values)), iterations, confidence, seed
    )


def stratified_interval(
    values: FloatArray,
    strata: NDArray[np.str_],
    statistic: Callable[[FloatArray], float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> ConfidenceInterval:
    if values.ndim != 1 or strata.ndim != 1 or values.size == 0 or values.size != strata.size:
        raise ValueError("values and strata must align and be nonempty")
    generator = np.random.default_rng(seed)
    unique = np.unique(strata)
    grouped = [np.flatnonzero(strata == value) for value in unique]
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        pieces = [generator.choice(indices, size=len(indices), replace=True) for indices in grouped]
        sample = np.concatenate(pieces)
        estimates[iteration] = statistic(values[sample])
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=float(statistic(values)),
        lower=float(np.quantile(estimates, tail)),
        upper=float(np.quantile(estimates, 1.0 - tail)),
        confidence=confidence,
        iterations=iterations,
    )


def _validate(values: FloatArray, iterations: int, confidence: float) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty vector")
    if iterations < 100:
        raise ValueError("at least one hundred iterations are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
