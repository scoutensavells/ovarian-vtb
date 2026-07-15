from datetime import UTC, datetime

from virtual_tumor_board.audit.common import AuditSummary
from virtual_tumor_board.evaluation.cohort import CohortSummary, RunAggregate
from virtual_tumor_board.reporting.schema import (
    IntervalValue,
    MetricEntry,
    ResultTable,
    SafetyEntry,
    StudyReport,
)


def cohort_table(identifier: str, title: str, summaries: tuple[CohortSummary, ...]) -> ResultTable:
    metrics: list[MetricEntry] = []
    for summary in summaries:
        metrics.extend(
            (
                _metric(summary, "diagnostic_accuracy", summary.diagnostic_accuracy, "fraction"),
                _metric(summary, "board_concordance", summary.board_concordance, "fraction"),
                _metric(
                    summary, "guideline_concordance", summary.guideline_concordance, "fraction"
                ),
                _metric(summary, "conformal_coverage", summary.conformal_coverage, "fraction"),
                _metric(summary, "complex_accuracy", summary.complex_accuracy, "fraction"),
                _metric(
                    summary,
                    "median_processing_seconds",
                    summary.median_processing_seconds,
                    "seconds",
                ),
            )
        )
    return ResultTable(identifier=identifier, title=title, metrics=tuple(metrics))


def aggregate_table(
    identifier: str, title: str, aggregates: tuple[RunAggregate, ...]
) -> ResultTable:
    metrics: list[MetricEntry] = []
    for item in aggregates:
        metrics.extend(
            (
                MetricEntry(
                    metric="diagnostic_accuracy",
                    value=IntervalValue(
                        estimate=item.diagnostic_accuracy_mean,
                        lower=None,
                        upper=None,
                        unit="fraction",
                    ),
                    count=item.runs,
                    method=item.method,
                    cohort="run_aggregate",
                ),
                MetricEntry(
                    metric="guideline_concordance",
                    value=IntervalValue(
                        estimate=item.guideline_concordance_mean,
                        lower=None,
                        upper=None,
                        unit="fraction",
                    ),
                    count=item.runs,
                    method=item.method,
                    cohort="run_aggregate",
                ),
                MetricEntry(
                    metric="conformal_coverage",
                    value=IntervalValue(
                        estimate=item.conformal_coverage_mean,
                        lower=None,
                        upper=None,
                        unit="fraction",
                    ),
                    count=item.runs,
                    method=item.method,
                    cohort="run_aggregate",
                ),
            )
        )
    return ResultTable(identifier=identifier, title=title, metrics=tuple(metrics))


def safety_table(identifier: str, title: str, audit: AuditSummary) -> ResultTable:
    entries = tuple(
        SafetyEntry(
            axis=audit.name,
            metric=item.name,
            observed=item.observed,
            threshold=item.threshold,
            comparison=item.comparison,
            passes=item.status.value == "pass",
        )
        for item in audit.criteria
    )
    return ResultTable(identifier=identifier, title=title, safety=entries)


def study_report(
    tables: tuple[ResultTable, ...],
    case_count: int,
    method_count: int,
    issues: tuple[str, ...] = (),
) -> StudyReport:
    return StudyReport(
        report_version="1.0",
        generated_at=datetime.now(UTC),
        tables=tables,
        case_count=case_count,
        method_count=method_count,
        unresolved_data_issues=issues,
    )


def _metric(
    summary: CohortSummary,
    name: str,
    estimate: float,
    unit: str,
) -> MetricEntry:
    return MetricEntry(
        metric=name,
        value=IntervalValue(estimate=estimate, unit=unit),
        count=summary.count,
        method=summary.method,
        cohort="cohort",
        run_index=summary.run_index,
    )
