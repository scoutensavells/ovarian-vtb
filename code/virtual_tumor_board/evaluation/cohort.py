from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from virtual_tumor_board.records.evaluation import CaseEvaluation


@dataclass(frozen=True)
class CohortSummary:
    method: str
    run_index: int
    count: int
    diagnostic_accuracy: float
    board_concordance: float
    guideline_concordance: float
    conformal_coverage: float
    complex_accuracy: float
    median_processing_seconds: float
    processing_iqr: tuple[float, float]


@dataclass(frozen=True)
class RunAggregate:
    method: str
    runs: int
    diagnostic_accuracy_mean: float
    diagnostic_accuracy_standard_deviation: float
    guideline_concordance_mean: float
    guideline_concordance_standard_deviation: float
    conformal_coverage_mean: float
    conformal_coverage_standard_deviation: float


def summarize_cohort(records: tuple[CaseEvaluation, ...]) -> CohortSummary:
    if not records:
        raise ValueError("cohort records must not be empty")
    method = records[0].prediction.method
    run_index = records[0].prediction.run_index
    if any(item.prediction.method != method for item in records):
        raise ValueError("cohort summary requires a single method")
    if any(item.prediction.run_index != run_index for item in records):
        raise ValueError("cohort summary requires a single run")
    diagnostic = np.asarray([item.diagnostic_correct for item in records], dtype=np.float64)
    board = np.asarray([item.board_concordant for item in records], dtype=np.float64)
    guideline = np.asarray([item.guideline_concordant for item in records], dtype=np.float64)
    coverage = np.asarray([item.conformal_covered for item in records], dtype=np.float64)
    complex_mask = np.asarray([item.complex_case for item in records], dtype=np.bool_)
    processing = np.asarray(
        [item.prediction.processing_seconds for item in records], dtype=np.float64
    )
    complex_accuracy = (
        float(np.mean(diagnostic[complex_mask])) if np.any(complex_mask) else float("nan")
    )
    return CohortSummary(
        method=method,
        run_index=run_index,
        count=len(records),
        diagnostic_accuracy=float(np.mean(diagnostic)),
        board_concordance=float(np.mean(board)),
        guideline_concordance=float(np.mean(guideline)),
        conformal_coverage=float(np.mean(coverage)),
        complex_accuracy=complex_accuracy,
        median_processing_seconds=float(np.median(processing)),
        processing_iqr=(float(np.quantile(processing, 0.25)), float(np.quantile(processing, 0.75))),
    )


def summarize_runs(summaries: tuple[CohortSummary, ...]) -> RunAggregate:
    if not summaries:
        raise ValueError("run summaries must not be empty")
    method = summaries[0].method
    if any(item.method != method for item in summaries):
        raise ValueError("run aggregation requires a single method")
    run_indices = [item.run_index for item in summaries]
    if len(run_indices) != len(set(run_indices)):
        raise ValueError("run indices must be unique")
    diagnostic = np.asarray([item.diagnostic_accuracy for item in summaries])
    guideline = np.asarray([item.guideline_concordance for item in summaries])
    coverage = np.asarray([item.conformal_coverage for item in summaries])
    ddof = 1 if len(summaries) > 1 else 0
    return RunAggregate(
        method=method,
        runs=len(summaries),
        diagnostic_accuracy_mean=float(np.mean(diagnostic)),
        diagnostic_accuracy_standard_deviation=float(np.std(diagnostic, ddof=ddof)),
        guideline_concordance_mean=float(np.mean(guideline)),
        guideline_concordance_standard_deviation=float(np.std(guideline, ddof=ddof)),
        conformal_coverage_mean=float(np.mean(coverage)),
        conformal_coverage_standard_deviation=float(np.std(coverage, ddof=ddof)),
    )


def group_by_method_and_run(
    records: tuple[CaseEvaluation, ...],
) -> dict[tuple[str, int], tuple[CaseEvaluation, ...]]:
    grouped: dict[tuple[str, int], list[CaseEvaluation]] = defaultdict(list)
    for record in records:
        key = (record.prediction.method, record.prediction.run_index)
        grouped[key].append(record)
    return {key: tuple(values) for key, values in grouped.items()}


def aggregate_all_runs(records: tuple[CaseEvaluation, ...]) -> tuple[RunAggregate, ...]:
    grouped = group_by_method_and_run(records)
    per_method: dict[str, list[CohortSummary]] = defaultdict(list)
    for (method, _), values in grouped.items():
        per_method[method].append(summarize_cohort(values))
    return tuple(summarize_runs(tuple(values)) for _, values in sorted(per_method.items()))
