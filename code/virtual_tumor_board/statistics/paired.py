from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class McNemarResult:
    left_only: int
    right_only: int
    discordant: int
    statistic: float
    p_value: float
    adjusted_p_value: float
    adjusted_alpha: float
    significant: bool


def exact_mcnemar(
    left_correct: BoolArray,
    right_correct: BoolArray,
    comparisons: int = 5,
    familywise_alpha: float = 0.05,
) -> McNemarResult:
    if left_correct.ndim != 1 or right_correct.ndim != 1:
        raise ValueError("inputs must be vectors")
    if left_correct.size == 0 or left_correct.size != right_correct.size:
        raise ValueError("inputs must align and be nonempty")
    if comparisons < 1:
        raise ValueError("comparisons must be positive")
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        p_value = float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
        statistic = ((abs(left_only - right_only) - 1.0) ** 2) / discordant
    adjusted_p = min(1.0, p_value * comparisons)
    adjusted_alpha = familywise_alpha / comparisons
    return McNemarResult(
        left_only=left_only,
        right_only=right_only,
        discordant=discordant,
        statistic=statistic,
        p_value=p_value,
        adjusted_p_value=adjusted_p,
        adjusted_alpha=adjusted_alpha,
        significant=p_value < adjusted_alpha,
    )


def bonferroni(values: tuple[float, ...]) -> tuple[float, ...]:
    count = len(values)
    if count == 0:
        raise ValueError("at least one p value is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("p values must be between zero and one")
    return tuple(min(1.0, value * count) for value in values)


def paired_accuracy_difference(left_correct: BoolArray, right_correct: BoolArray) -> float:
    if left_correct.shape != right_correct.shape or left_correct.ndim != 1:
        raise ValueError("paired vectors must align")
    return float(np.mean(left_correct) - np.mean(right_correct))
