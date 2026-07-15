from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator
from virtual_tumor_board.domain.enums import (
    CohortKind,
    FailureMode,
    HarmGrade,
    Histotype,
    MolecularSubtype,
    SpecialistKind,
)


class ReferenceStandard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    confirmed_histotype: Histotype
    final_board_decision: str = Field(min_length=1)
    implemented_first_line_treatment: str = Field(min_length=1)
    guideline_concordant_treatments: tuple[str, ...]
    confirmation_date: date | None = None


class MethodPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    method: str = Field(min_length=1)
    run_index: int = Field(ge=0)
    predicted_histotype: Histotype
    treatment_recommendation: str = Field(min_length=1)
    board_decision_recommendation: str = Field(min_length=1)
    histotype_probabilities: dict[Histotype, float]
    confidence: float = Field(ge=0.0, le=1.0)
    conformal_set: tuple[Histotype, ...]
    processing_seconds: float = Field(gt=0.0)
    completed_at: datetime

    @model_validator(mode="after")
    def validate_probabilities(self) -> "MethodPrediction":
        if set(self.histotype_probabilities) != set(Histotype):
            raise ValueError("probabilities must cover all histotypes")
        total = sum(self.histotype_probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("histotype probabilities must sum to one")
        if any(value < 0.0 or value > 1.0 for value in self.histotype_probabilities.values()):
            raise ValueError("histotype probabilities must lie between zero and one")
        if not self.conformal_set:
            raise ValueError("conformal set must not be empty")
        return self


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort: CohortKind
    reference: ReferenceStandard
    prediction: MethodPrediction
    board_concordant: bool
    diagnostic_correct: bool
    guideline_concordant: bool
    conformal_covered: bool
    complex_case: bool
    site_code: str | None = None
    region_code: str | None = None
    age_tercile: str | None = None

    @model_validator(mode="after")
    def validate_keys(self) -> "CaseEvaluation":
        if self.reference.case_key != self.prediction.case_key:
            raise ValueError("reference and prediction case keys must match")
        return self


class SpecialistTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    specialist: SpecialistKind
    run_index: int = Field(ge=0)
    initial_label: str
    final_label: str
    initial_confidence: float = Field(ge=0.0, le=1.0)
    final_confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=0)
    uncertainty_count: int = Field(ge=0)
    rounds_participated: int = Field(ge=0, le=3)
    challenge_count: int = Field(ge=0)
    support_count: int = Field(ge=0)


class MolecularEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    true_subtype: MolecularSubtype
    predicted_subtype: MolecularSubtype
    probabilities: dict[MolecularSubtype, float]
    molecular_agent_present: bool

    @model_validator(mode="after")
    def validate_distribution(self) -> "MolecularEvaluation":
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("molecular probabilities must sum to one")
        return self


class FailureReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    failure_mode: FailureMode
    reviewer_one_mode: FailureMode
    reviewer_two_mode: FailureMode
    consensus_reached: bool
    uncertainty_flagged: bool
    conformal_set_size: int = Field(ge=1)
    narrative: str = Field(min_length=1, max_length=4000)


class HarmReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    harm_grade: HarmGrade
    reviewer_code: str = Field(min_length=1, max_length=32)
    uncertainty_flagged: bool
    blinded_to_method: bool = True
    rationale: str = Field(min_length=1, max_length=4000)


class PerturbationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    perturbation_name: str
    removed_modality: str | None = None
    original_correct: bool
    perturbed_correct: bool
    original_confidence: float = Field(ge=0.0, le=1.0)
    perturbed_confidence: float = Field(ge=0.0, le=1.0)
    original_label: str
    perturbed_label: str


class TemporalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    quarter: str = Field(pattern=r"^\d{4}-Q[1-4]$")
    correct: bool
    guideline_concordant: bool
    processing_seconds: float = Field(gt=0.0)


class ReaderObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    reader_code: str = Field(min_length=1, max_length=32)
    condition: str = Field(pattern=r"^(clinician|model|team)$")
    period: int = Field(ge=1, le=3)
    diagnostic_correct: bool
    guideline_concordant: bool
    complex_case: bool
    review_seconds: float = Field(gt=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=10.0)
    washout_days: int = Field(default=14, ge=14)


class WorkflowObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    report_ready_at: datetime
    board_scheduled_at: datetime
    report_consulted: bool
    preparation_minutes: float = Field(gt=0.0)
    board_agreed: bool

    @property
    def ready_before_board(self) -> bool:
        return self.report_ready_at <= self.board_scheduled_at
