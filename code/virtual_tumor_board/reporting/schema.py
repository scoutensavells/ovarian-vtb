from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IntervalValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    estimate: float
    lower: float | None = None
    upper: float | None = None
    unit: str
    confidence_level: float | None = Field(default=None, gt=0.0, lt=1.0)


class MetricEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    value: IntervalValue
    count: int = Field(gt=0)
    method: str
    cohort: str
    run_index: int | None = Field(default=None, ge=0)


class ComparisonEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left_method: str
    right_method: str
    metric: str
    difference: float
    statistic_name: str
    statistic: float
    raw_p_value: float = Field(ge=0.0, le=1.0)
    adjusted_p_value: float = Field(ge=0.0, le=1.0)
    adjusted_alpha: float = Field(gt=0.0, lt=1.0)
    significant: bool


class CalibrationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    expected_calibration_error: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0)
    conformal_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    median_set_size: float | None = Field(default=None, ge=1.0)
    abstention_precision: float = Field(ge=0.0, le=1.0)
    abstention_recall: float = Field(ge=0.0, le=1.0)


class SafetyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: str
    metric: str
    observed: float
    threshold: float | None = None
    comparison: str | None = None
    passes: bool | None = None


class ResultTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identifier: str
    title: str
    metrics: tuple[MetricEntry, ...] = ()
    comparisons: tuple[ComparisonEntry, ...] = ()
    calibration: tuple[CalibrationEntry, ...] = ()
    safety: tuple[SafetyEntry, ...] = ()


class StudyReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: str
    generated_at: datetime
    protocol_max_rounds: int = Field(default=3, ge=1, le=3)
    conflict_threshold: float = Field(default=0.15, gt=0.0, lt=1.0)
    conformal_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    tables: tuple[ResultTable, ...]
    case_count: int = Field(ge=0)
    method_count: int = Field(ge=0)
    unresolved_data_issues: tuple[str, ...] = ()
