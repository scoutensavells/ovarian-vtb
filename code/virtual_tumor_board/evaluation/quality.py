from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from virtual_tumor_board.domain.enums import CohortKind, Histotype
from virtual_tumor_board.records.evaluation import CaseEvaluation


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    case_key: str | None = None
    method: str | None = None
    run_index: int | None = None


@dataclass(frozen=True)
class DistributionProfile:
    count: int
    histotype_counts: dict[Histotype, int]
    complex_count: int
    site_counts: dict[str, int]
    region_counts: dict[str, int]
    confidence_minimum: float
    confidence_maximum: float
    confidence_mean: float
    processing_median_seconds: float


@dataclass(frozen=True)
class QualityReport:
    record_count: int
    case_count: int
    methods: tuple[str, ...]
    runs_by_method: dict[str, tuple[int, ...]]
    issues: tuple[QualityIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(item.severity == "error" for item in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "warning" for item in self.issues)

    @property
    def valid(self) -> bool:
        return self.error_count == 0


def distribution_profile(records: tuple[CaseEvaluation, ...]) -> DistributionProfile:
    if not records:
        raise ValueError("evaluation records must not be empty")
    histotypes = Counter(item.reference.confirmed_histotype for item in records)
    sites = Counter(item.site_code for item in records if item.site_code is not None)
    regions = Counter(item.region_code for item in records if item.region_code is not None)
    confidence = np.asarray([item.prediction.confidence for item in records], dtype=np.float64)
    processing = np.asarray(
        [item.prediction.processing_seconds for item in records],
        dtype=np.float64,
    )
    return DistributionProfile(
        count=len(records),
        histotype_counts={value: histotypes.get(value, 0) for value in Histotype},
        complex_count=sum(item.complex_case for item in records),
        site_counts=dict(sites),
        region_counts=dict(regions),
        confidence_minimum=float(np.min(confidence)),
        confidence_maximum=float(np.max(confidence)),
        confidence_mean=float(np.mean(confidence)),
        processing_median_seconds=float(np.median(processing)),
    )


def quality_report(records: tuple[CaseEvaluation, ...]) -> QualityReport:
    if not records:
        raise ValueError("evaluation records must not be empty")
    issues = []
    issues.extend(_duplicate_predictions(records))
    issues.extend(_reference_consistency(records))
    issues.extend(_run_balance(records))
    issues.extend(_cohort_consistency(records))
    issues.extend(_derived_flag_consistency(records))
    issues.extend(_coverage_consistency(records))
    issues.extend(_distribution_warnings(records))
    methods = tuple(sorted({item.prediction.method for item in records}))
    runs_by_method = {
        method: tuple(
            sorted(
                {item.prediction.run_index for item in records if item.prediction.method == method}
            )
        )
        for method in methods
    }
    return QualityReport(
        record_count=len(records),
        case_count=len({item.reference.case_key for item in records}),
        methods=methods,
        runs_by_method=runs_by_method,
        issues=tuple(issues),
    )


def require_quality(records: tuple[CaseEvaluation, ...]) -> QualityReport:
    report = quality_report(records)
    if not report.valid:
        codes = sorted({item.code for item in report.issues if item.severity == "error"})
        raise ValueError("evaluation quality errors: " + ", ".join(codes))
    return report


def _duplicate_predictions(records: tuple[CaseEvaluation, ...]) -> list[QualityIssue]:
    seen: set[tuple[str, str, int]] = set()
    issues = []
    for record in records:
        key = (
            record.reference.case_key,
            record.prediction.method,
            record.prediction.run_index,
        )
        if key in seen:
            issues.append(
                QualityIssue(
                    code="duplicate_prediction",
                    severity="error",
                    message="case, method, and run combination occurs more than once",
                    case_key=record.reference.case_key,
                    method=record.prediction.method,
                    run_index=record.prediction.run_index,
                )
            )
        seen.add(key)
    return issues


def _reference_consistency(records: tuple[CaseEvaluation, ...]) -> list[QualityIssue]:
    signatures: dict[str, tuple[Histotype, str, str]] = {}
    issues = []
    for record in records:
        reference = record.reference
        signature = (
            reference.confirmed_histotype,
            reference.final_board_decision,
            reference.implemented_first_line_treatment,
        )
        previous = signatures.get(reference.case_key)
        if previous is not None and previous != signature:
            issues.append(
                QualityIssue(
                    code="inconsistent_reference",
                    severity="error",
                    message="reference standard differs across method or run records",
                    case_key=reference.case_key,
                )
            )
        signatures[reference.case_key] = signature
    return issues


def _run_balance(records: tuple[CaseEvaluation, ...]) -> list[QualityIssue]:
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for record in records:
        counts[(record.prediction.method, record.prediction.run_index)] += 1
    methods = {key[0] for key in counts}
    issues = []
    for method in methods:
        values = [count for (name, _), count in counts.items() if name == method]
        if values and max(values) != min(values):
            issues.append(
                QualityIssue(
                    code="unbalanced_runs",
                    severity="error",
                    message="run sizes differ within a method",
                    method=method,
                )
            )
    return issues


def _cohort_consistency(records: tuple[CaseEvaluation, ...]) -> list[QualityIssue]:
    cohorts: dict[str, CohortKind] = {}
    issues = []
    for record in records:
        case_key = record.reference.case_key
        previous = cohorts.get(case_key)
        if previous is not None and previous is not record.cohort:
            issues.append(
                QualityIssue(
                    code="cross_cohort_case",
                    severity="error",
                    message="case key occurs in more than one cohort",
                    case_key=case_key,
                )
            )
        cohorts[case_key] = record.cohort
    return issues


def _derived_flag_consistency(records: tuple[CaseEvaluation, ...]) -> list[QualityIssue]:
    issues = []
    for record in records:
        expected_diagnostic = (
            record.prediction.predicted_histotype == record.reference.confirmed_histotype
        )
        expected_board = (
            record.prediction.board_decision_recommendation == record.reference.final_board_decision
        )
        expected_guideline = (
            record.prediction.treatment_recommendation
            in record.reference.guideline_concordant_treatments
        )
        flags = (
            ("diagnostic_correct", record.diagnostic_correct, expected_diagnostic),
            ("board_concordant", record.board_concordant, expected_board),
            ("guideline_concordant", record.guideline_concordant, expected_guideline),
        )
        for name, observed, expected in flags:
            if observed != expected:
                issues.append(
                    QualityIssue(
                        code="inconsistent_derived_flag",
                        severity="error",
                        message=f"{name} does not match record contents",
                        case_key=record.reference.case_key,
                        method=record.prediction.method,
                        run_index=record.prediction.run_index,
                    )
                )
    return issues


def _coverage_consistency(records: tuple[CaseEvaluation, ...]) -> list[QualityIssue]:
    issues = []
    for record in records:
        expected = record.reference.confirmed_histotype in record.prediction.conformal_set
        if record.conformal_covered != expected:
            issues.append(
                QualityIssue(
                    code="inconsistent_conformal_coverage",
                    severity="error",
                    message="coverage flag does not match prediction set",
                    case_key=record.reference.case_key,
                    method=record.prediction.method,
                    run_index=record.prediction.run_index,
                )
            )
    return issues


def _distribution_warnings(records: tuple[CaseEvaluation, ...]) -> list[QualityIssue]:
    issues = []
    profile = distribution_profile(records)
    missing_histotypes = [
        key.value for key, value in profile.histotype_counts.items() if value == 0
    ]
    if missing_histotypes:
        issues.append(
            QualityIssue(
                code="missing_histotypes",
                severity="warning",
                message="no cases observed for: " + ", ".join(missing_histotypes),
            )
        )
    if profile.complex_count == 0:
        issues.append(
            QualityIssue(
                code="missing_complex_cases",
                severity="warning",
                message="the cohort contains no cases marked complex",
            )
        )
    if profile.confidence_minimum == profile.confidence_maximum:
        issues.append(
            QualityIssue(
                code="constant_confidence",
                severity="warning",
                message="all predictions have identical confidence",
            )
        )
    return issues
