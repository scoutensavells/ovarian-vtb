from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from virtual_tumor_board.domain.enums import FailureMode, HarmGrade
from virtual_tumor_board.records.evaluation import (
    FailureReview,
    HarmReview,
    PerturbationRecord,
    TemporalRecord,
)


@dataclass(frozen=True)
class FailureSummary:
    total: int
    counts: dict[FailureMode, int]
    fractions: dict[FailureMode, float]
    uncertainty_capture: float
    reviewer_agreement: float


@dataclass(frozen=True)
class HarmSummary:
    total: int
    counts: dict[HarmGrade, int]
    fractions: dict[HarmGrade, float]
    potential_harm_capture: float


@dataclass(frozen=True)
class PerturbationSummary:
    count: int
    baseline_accuracy: float
    perturbed_accuracy: float
    accuracy_change: float
    mean_confidence_change: float
    label_change_rate: float


@dataclass(frozen=True)
class TemporalSummary:
    quarterly_accuracy: dict[str, float]
    quarterly_guideline_concordance: dict[str, float]
    maximum_accuracy_change: float


@dataclass(frozen=True)
class DifficultyCorrelation:
    coefficient: float
    p_value: float
    count: int


def failure_summary(records: tuple[FailureReview, ...]) -> FailureSummary:
    if not records:
        raise ValueError("failure reviews must not be empty")
    counts = Counter(item.failure_mode for item in records)
    total = len(records)
    fractions = {mode: counts.get(mode, 0) / total for mode in FailureMode}
    return FailureSummary(
        total=total,
        counts={mode: counts.get(mode, 0) for mode in FailureMode},
        fractions=fractions,
        uncertainty_capture=float(np.mean([item.uncertainty_flagged for item in records])),
        reviewer_agreement=float(
            np.mean([item.reviewer_one_mode == item.reviewer_two_mode for item in records])
        ),
    )


def harm_summary(records: tuple[HarmReview, ...]) -> HarmSummary:
    if not records:
        raise ValueError("harm reviews must not be empty")
    counts = Counter(item.harm_grade for item in records)
    total = len(records)
    harmful = [item for item in records if item.harm_grade is not HarmGrade.NONE]
    capture = float(np.mean([item.uncertainty_flagged for item in harmful])) if harmful else 1.0
    return HarmSummary(
        total=total,
        counts={grade: counts.get(grade, 0) for grade in HarmGrade},
        fractions={grade: counts.get(grade, 0) / total for grade in HarmGrade},
        potential_harm_capture=capture,
    )


def perturbation_summary(records: tuple[PerturbationRecord, ...]) -> PerturbationSummary:
    if not records:
        raise ValueError("perturbation records must not be empty")
    baseline = np.asarray([item.original_correct for item in records], dtype=np.float64)
    perturbed = np.asarray([item.perturbed_correct for item in records], dtype=np.float64)
    confidence_change = np.asarray(
        [item.perturbed_confidence - item.original_confidence for item in records]
    )
    label_changed = np.asarray([item.original_label != item.perturbed_label for item in records])
    return PerturbationSummary(
        count=len(records),
        baseline_accuracy=float(np.mean(baseline)),
        perturbed_accuracy=float(np.mean(perturbed)),
        accuracy_change=float(np.mean(perturbed) - np.mean(baseline)),
        mean_confidence_change=float(np.mean(confidence_change)),
        label_change_rate=float(np.mean(label_changed)),
    )


def temporal_summary(records: tuple[TemporalRecord, ...]) -> TemporalSummary:
    if not records:
        raise ValueError("temporal records must not be empty")
    groups: dict[str, list[TemporalRecord]] = defaultdict(list)
    for record in records:
        groups[record.quarter].append(record)
    accuracy = {
        quarter: float(np.mean([item.correct for item in members]))
        for quarter, members in sorted(groups.items())
    }
    guideline = {
        quarter: float(np.mean([item.guideline_concordant for item in members]))
        for quarter, members in sorted(groups.items())
    }
    values = tuple(accuracy.values())
    spread = max(values) - min(values) if values else 0.0
    return TemporalSummary(accuracy, guideline, spread)


def difficulty_correlation(
    disagreement: np.ndarray,
    difficulty: np.ndarray,
) -> DifficultyCorrelation:
    if disagreement.ndim != 1 or difficulty.ndim != 1:
        raise ValueError("inputs must be vectors")
    if disagreement.size < 3 or disagreement.size != difficulty.size:
        raise ValueError("at least three aligned observations are required")
    result = spearmanr(disagreement, difficulty)
    return DifficultyCorrelation(float(result.statistic), float(result.pvalue), len(disagreement))
