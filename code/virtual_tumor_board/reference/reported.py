from dataclasses import dataclass


@dataclass(frozen=True)
class ProspectiveResult:
    method: str
    diagnostic_accuracy_percent: float
    board_concordance_percent: float
    guideline_concordance_percent: float
    complex_accuracy_percent: float
    median_seconds: float


@dataclass(frozen=True)
class HoldoutResult:
    method: str
    diagnostic_accuracy_percent: float
    guideline_concordance_percent: float
    histotype_auc_percent: float
    median_seconds: float


@dataclass(frozen=True)
class AblationResult:
    configuration: str
    accuracy_percent: float
    guideline_concordance_percent: float
    complex_accuracy_percent: float


@dataclass(frozen=True)
class HistotypeResult:
    histotype: str
    count: int
    accuracy_percent: float
    guideline_concordance_percent: float
    auc_percent: float


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    expected_calibration_error_percent: float
    brier_percent: float
    abstention_precision_percent: float
    abstention_recall_percent: float


@dataclass(frozen=True)
class ReaderResult:
    arm: str
    diagnostic_accuracy_percent: float
    guideline_concordance_percent: float
    complex_accuracy_percent: float
    median_seconds: float


PROSPECTIVE_RESULTS: tuple[ProspectiveResult, ...] = (
    ProspectiveResult(
        method="MSDP",
        diagnostic_accuracy_percent=90.1,
        board_concordance_percent=89.4,
        guideline_concordance_percent=88.6,
        complex_accuracy_percent=84.6,
        median_seconds=74,
    ),
    ProspectiveResult(
        method="Single LLM",
        diagnostic_accuracy_percent=75.8,
        board_concordance_percent=73.9,
        guideline_concordance_percent=73.1,
        complex_accuracy_percent=63.2,
        median_seconds=26,
    ),
    ProspectiveResult(
        method="Chain-of-thought agent",
        diagnostic_accuracy_percent=77.6,
        board_concordance_percent=75.8,
        guideline_concordance_percent=75.4,
        complex_accuracy_percent=65.9,
        median_seconds=45,
    ),
    ProspectiveResult(
        method="Traditional CDSS",
        diagnostic_accuracy_percent=80.8,
        board_concordance_percent=79.6,
        guideline_concordance_percent=84.2,
        complex_accuracy_percent=68.1,
        median_seconds=13,
    ),
    ProspectiveResult(
        method="Individual specialist DL",
        diagnostic_accuracy_percent=83.6,
        board_concordance_percent=70.4,
        guideline_concordance_percent=70.1,
        complex_accuracy_percent=74.9,
        median_seconds=38,
    ),
    ProspectiveResult(
        method="Simple multi-agent voting",
        diagnostic_accuracy_percent=81.4,
        board_concordance_percent=79.7,
        guideline_concordance_percent=79.2,
        complex_accuracy_percent=72.3,
        median_seconds=57,
    ),
)


HOLDOUT_RESULTS: tuple[HoldoutResult, ...] = (
    HoldoutResult(
        method="MSDP",
        diagnostic_accuracy_percent=91.2,
        guideline_concordance_percent=89.7,
        histotype_auc_percent=90.4,
        median_seconds=67,
    ),
    HoldoutResult(
        method="Single LLM",
        diagnostic_accuracy_percent=76.4,
        guideline_concordance_percent=73.1,
        histotype_auc_percent=81.2,
        median_seconds=23,
    ),
    HoldoutResult(
        method="Chain-of-thought agent",
        diagnostic_accuracy_percent=78.8,
        guideline_concordance_percent=75.9,
        histotype_auc_percent=83.7,
        median_seconds=41,
    ),
    HoldoutResult(
        method="Traditional CDSS",
        diagnostic_accuracy_percent=81.5,
        guideline_concordance_percent=85.7,
        histotype_auc_percent=76.4,
        median_seconds=12,
    ),
    HoldoutResult(
        method="Individual specialist DL",
        diagnostic_accuracy_percent=84.1,
        guideline_concordance_percent=70.3,
        histotype_auc_percent=91.3,
        median_seconds=35,
    ),
    HoldoutResult(
        method="Simple multi-agent voting",
        diagnostic_accuracy_percent=82.3,
        guideline_concordance_percent=80.6,
        histotype_auc_percent=86.9,
        median_seconds=52,
    ),
)


ABLATION_RESULTS: tuple[AblationResult, ...] = (
    AblationResult(
        configuration="Full MSDP",
        accuracy_percent=91.2,
        guideline_concordance_percent=89.7,
        complex_accuracy_percent=85.3,
    ),
    AblationResult(
        configuration="without Pathologist",
        accuracy_percent=87.4,
        guideline_concordance_percent=85.7,
        complex_accuracy_percent=79.4,
    ),
    AblationResult(
        configuration="without Radiologist",
        accuracy_percent=88.6,
        guideline_concordance_percent=86.9,
        complex_accuracy_percent=80.9,
    ),
    AblationResult(
        configuration="without Medical Oncologist",
        accuracy_percent=89.1,
        guideline_concordance_percent=83.4,
        complex_accuracy_percent=81.6,
    ),
    AblationResult(
        configuration="without Molecular Biologist",
        accuracy_percent=89.4,
        guideline_concordance_percent=87.4,
        complex_accuracy_percent=82.4,
    ),
    AblationResult(
        configuration="without cross-examination",
        accuracy_percent=86.9,
        guideline_concordance_percent=84.6,
        complex_accuracy_percent=76.5,
    ),
    AblationResult(
        configuration="without evidence weighting",
        accuracy_percent=88,
        guideline_concordance_percent=86.3,
        complex_accuracy_percent=79.4,
    ),
    AblationResult(
        configuration="without conflict resolution",
        accuracy_percent=87.4,
        guideline_concordance_percent=85.1,
        complex_accuracy_percent=77.9,
    ),
    AblationResult(
        configuration="without meta-agent orchestration",
        accuracy_percent=84.6,
        guideline_concordance_percent=82.3,
        complex_accuracy_percent=74.3,
    ),
    AblationResult(
        configuration="without evidence weighting and conflict resolution",
        accuracy_percent=84.3,
        guideline_concordance_percent=81.7,
        complex_accuracy_percent=73.5,
    ),
)


HISTOTYPE_RESULTS: tuple[HistotypeResult, ...] = (
    HistotypeResult(
        histotype="HGSOC",
        count=108,
        accuracy_percent=94.4,
        guideline_concordance_percent=92.6,
        auc_percent=96.7,
    ),
    HistotypeResult(
        histotype="Endometrioid",
        count=24,
        accuracy_percent=87.5,
        guideline_concordance_percent=83.3,
        auc_percent=91.2,
    ),
    HistotypeResult(
        histotype="Clear cell",
        count=18,
        accuracy_percent=83.3,
        guideline_concordance_percent=77.8,
        auc_percent=89.1,
    ),
    HistotypeResult(
        histotype="Mucinous",
        count=13,
        accuracy_percent=84.6,
        guideline_concordance_percent=84.6,
        auc_percent=90.3,
    ),
    HistotypeResult(
        histotype="LGSC",
        count=12,
        accuracy_percent=75,
        guideline_concordance_percent=75,
        auc_percent=84.7,
    ),
)


CALIBRATION_RESULTS: tuple[CalibrationResult, ...] = (
    CalibrationResult(
        method="MSDP",
        expected_calibration_error_percent=4.2,
        brier_percent=4.8,
        abstention_precision_percent=78.4,
        abstention_recall_percent=34.2,
    ),
    CalibrationResult(
        method="Single LLM",
        expected_calibration_error_percent=13.4,
        brier_percent=14.2,
        abstention_precision_percent=52.1,
        abstention_recall_percent=13.2,
    ),
    CalibrationResult(
        method="Chain-of-thought agent",
        expected_calibration_error_percent=11.8,
        brier_percent=12.7,
        abstention_precision_percent=56.1,
        abstention_recall_percent=18.6,
    ),
    CalibrationResult(
        method="Specialist DL",
        expected_calibration_error_percent=8.9,
        brier_percent=9.4,
        abstention_precision_percent=60.2,
        abstention_recall_percent=21.8,
    ),
    CalibrationResult(
        method="Simple multi-agent",
        expected_calibration_error_percent=9.7,
        brier_percent=10.8,
        abstention_precision_percent=63.5,
        abstention_recall_percent=26.7,
    ),
)


READER_RESULTS: tuple[ReaderResult, ...] = (
    ReaderResult(
        arm="Clinician alone",
        diagnostic_accuracy_percent=86.7,
        guideline_concordance_percent=85.9,
        complex_accuracy_percent=74.1,
        median_seconds=684,
    ),
    ReaderResult(
        arm="Model alone",
        diagnostic_accuracy_percent=90.1,
        guideline_concordance_percent=88.6,
        complex_accuracy_percent=84.6,
        median_seconds=74,
    ),
    ReaderResult(
        arm="Human-AI team",
        diagnostic_accuracy_percent=93.8,
        guideline_concordance_percent=92.4,
        complex_accuracy_percent=85.3,
        median_seconds=408,
    ),
)


def prospective_result(method: str) -> ProspectiveResult:
    return next(item for item in PROSPECTIVE_RESULTS if item.method == method)


def holdout_result(method: str) -> HoldoutResult:
    return next(item for item in HOLDOUT_RESULTS if item.method == method)


def ablation_result(configuration: str) -> AblationResult:
    return next(item for item in ABLATION_RESULTS if item.configuration == configuration)


def histotype_result(histotype: str) -> HistotypeResult:
    return next(item for item in HISTOTYPE_RESULTS if item.histotype == histotype)


def calibration_result(method: str) -> CalibrationResult:
    return next(item for item in CALIBRATION_RESULTS if item.method == method)


def reader_result(arm: str) -> ReaderResult:
    return next(item for item in READER_RESULTS if item.arm == arm)
