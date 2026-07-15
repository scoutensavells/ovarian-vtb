from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from virtual_tumor_board.records.evaluation import ReaderObservation


@dataclass(frozen=True)
class ReaderArmSummary:
    condition: str
    observation_count: int
    case_count: int
    reader_count: int
    diagnostic_accuracy: float
    guideline_concordance: float
    complex_accuracy: float
    median_review_seconds: float
    mean_confidence: float | None


@dataclass(frozen=True)
class WashoutSummary:
    minimum_days: int
    median_days: float
    violations: int


def summarize_reader_arm(records: tuple[ReaderObservation, ...]) -> ReaderArmSummary:
    if not records:
        raise ValueError("reader observations must not be empty")
    condition = records[0].condition
    if any(item.condition != condition for item in records):
        raise ValueError("reader arm summary requires one condition")
    diagnostic = np.asarray([item.diagnostic_correct for item in records], dtype=np.float64)
    guideline = np.asarray([item.guideline_concordant for item in records], dtype=np.float64)
    complex_mask = np.asarray([item.complex_case for item in records], dtype=np.bool_)
    times = np.asarray([item.review_seconds for item in records], dtype=np.float64)
    confidence = [item.confidence for item in records if item.confidence is not None]
    return ReaderArmSummary(
        condition=condition,
        observation_count=len(records),
        case_count=len({item.case_key for item in records}),
        reader_count=len({item.reader_code for item in records}),
        diagnostic_accuracy=float(np.mean(diagnostic)),
        guideline_concordance=float(np.mean(guideline)),
        complex_accuracy=float(np.mean(diagnostic[complex_mask]))
        if np.any(complex_mask)
        else float("nan"),
        median_review_seconds=float(np.median(times)),
        mean_confidence=float(np.mean(confidence)) if confidence else None,
    )


def summarize_all_arms(
    records: tuple[ReaderObservation, ...],
) -> tuple[ReaderArmSummary, ...]:
    grouped: dict[str, list[ReaderObservation]] = defaultdict(list)
    for record in records:
        grouped[record.condition].append(record)
    required = {"clinician", "model", "team"}
    if set(grouped) != required:
        raise ValueError("all three reader conditions are required")
    return tuple(summarize_reader_arm(tuple(grouped[key])) for key in sorted(grouped))


def validate_reader_balance(records: tuple[ReaderObservation, ...]) -> None:
    if not records:
        raise ValueError("reader observations must not be empty")
    by_reader: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        by_reader[record.reader_code][record.condition] += 1
    for conditions in by_reader.values():
        if set(conditions) != {"clinician", "model", "team"}:
            raise ValueError("each reader must complete all conditions")
        counts = tuple(conditions.values())
        if max(counts) - min(counts) > 1:
            raise ValueError("reader condition counts must be balanced")


def summarize_washout(records: tuple[ReaderObservation, ...]) -> WashoutSummary:
    if not records:
        raise ValueError("reader observations must not be empty")
    days = np.asarray([item.washout_days for item in records])
    return WashoutSummary(
        minimum_days=int(np.min(days)),
        median_days=float(np.median(days)),
        violations=int(np.sum(days < 14)),
    )
