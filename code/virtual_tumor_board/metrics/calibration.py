from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float
    absolute_gap: float


@dataclass(frozen=True)
class CalibrationSummary:
    expected_calibration_error: float
    maximum_calibration_error: float
    bins: tuple[CalibrationBin, ...]


def calibration_bins(
    confidence: FloatArray,
    correctness: BoolArray,
    count: int = 15,
) -> tuple[CalibrationBin, ...]:
    _validate(confidence, correctness, count)
    boundaries = np.linspace(0.0, 1.0, count + 1)
    result = []
    for index in range(count):
        lower = float(boundaries[index])
        upper = float(boundaries[index + 1])
        if index == count - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        members = int(np.sum(mask))
        if members == 0:
            accuracy = 0.0
            mean_confidence = 0.0
        else:
            accuracy = float(np.mean(correctness[mask]))
            mean_confidence = float(np.mean(confidence[mask]))
        result.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=members,
                accuracy=accuracy,
                confidence=mean_confidence,
                absolute_gap=abs(accuracy - mean_confidence) if members else 0.0,
            )
        )
    return tuple(result)


def expected_calibration_error(
    confidence: FloatArray,
    correctness: BoolArray,
    count: int = 15,
) -> float:
    bins = calibration_bins(confidence, correctness, count)
    total = len(confidence)
    return sum(item.count / total * item.absolute_gap for item in bins)


def maximum_calibration_error(
    confidence: FloatArray,
    correctness: BoolArray,
    count: int = 15,
) -> float:
    bins = calibration_bins(confidence, correctness, count)
    populated = [item.absolute_gap for item in bins if item.count]
    return max(populated, default=0.0)


def calibration_summary(
    confidence: FloatArray,
    correctness: BoolArray,
    count: int = 15,
) -> CalibrationSummary:
    bins = calibration_bins(confidence, correctness, count)
    total = len(confidence)
    ece = sum(item.count / total * item.absolute_gap for item in bins)
    maximum = max((item.absolute_gap for item in bins if item.count), default=0.0)
    return CalibrationSummary(ece, maximum, bins)


def _validate(confidence: FloatArray, correctness: BoolArray, count: int) -> None:
    if confidence.ndim != 1 or correctness.ndim != 1:
        raise ValueError("inputs must be vectors")
    if confidence.size == 0 or confidence.size != correctness.size:
        raise ValueError("inputs must align and be nonempty")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("confidence must be between zero and one")
    if count < 2:
        raise ValueError("at least two bins are required")
