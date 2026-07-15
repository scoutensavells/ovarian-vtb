from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DeLongResult:
    auc_left: float
    auc_right: float
    difference: float
    variance: float
    z_score: float
    p_value: float


def binary_delong(targets: IntArray, left: FloatArray, right: FloatArray) -> DeLongResult:
    _validate(targets, left, right)
    positives = targets == 1
    negatives = targets == 0
    left_positive = left[positives]
    left_negative = left[negatives]
    right_positive = right[positives]
    right_negative = right[negatives]
    left_components = _components(left_positive, left_negative)
    right_components = _components(right_positive, right_negative)
    auc_left = float(np.mean(left_components[0]))
    auc_right = float(np.mean(right_components[0]))
    positive_covariance = np.cov(np.vstack((left_components[0], right_components[0])), ddof=1)
    negative_covariance = np.cov(np.vstack((left_components[1], right_components[1])), ddof=1)
    covariance = positive_covariance / len(left_positive) + negative_covariance / len(left_negative)
    contrast = np.asarray([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    difference = auc_left - auc_right
    if variance <= 0.0:
        z_score = 0.0 if difference == 0.0 else float(np.sign(difference) * np.inf)
    else:
        z_score = difference / np.sqrt(variance)
    p_value = float(2.0 * norm.sf(abs(z_score)))
    return DeLongResult(auc_left, auc_right, difference, variance, z_score, p_value)


def _components(positive: FloatArray, negative: FloatArray) -> tuple[FloatArray, FloatArray]:
    comparisons = positive[:, None] - negative[None, :]
    kernel = (comparisons > 0.0).astype(np.float64) + 0.5 * (comparisons == 0.0)
    return np.mean(kernel, axis=1), np.mean(kernel, axis=0)


def _validate(targets: IntArray, left: FloatArray, right: FloatArray) -> None:
    if targets.ndim != 1 or left.ndim != 1 or right.ndim != 1:
        raise ValueError("inputs must be vectors")
    if targets.size == 0 or targets.size != left.size or targets.size != right.size:
        raise ValueError("inputs must align and be nonempty")
    if not np.array_equal(np.unique(targets), np.asarray([0, 1])):
        raise ValueError("both binary target classes must occur")
