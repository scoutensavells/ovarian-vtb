from pydantic import BaseModel, ConfigDict, Field, model_validator
from virtual_tumor_board.domain.enums import (
    DisagreementSource,
    EpistemicAct,
    EvidenceModality,
    RecommendationKind,
    ResolutionMode,
    SpecialistKind,
    TriageLevel,
)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_key: str = Field(min_length=1, max_length=256)
    passage: str = Field(min_length=1, max_length=4000)
    modality: EvidenceModality
    relevance: float = Field(ge=0.0, le=1.0)
    supports_claim: bool = True


class CandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, max_length=256)
    probability: float = Field(ge=0.0, le=1.0)


class SpecialistAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    specialist: SpecialistKind
    recommendation_kind: RecommendationKind
    primary_label: str = Field(min_length=1, max_length=256)
    confidence: float = Field(ge=0.0, le=1.0)
    candidates: tuple[CandidateScore, ...]
    evidence: tuple[EvidenceItem, ...]
    uncertainties: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_distribution(self) -> "SpecialistAssessment":
        labels = [item.label for item in self.candidates]
        if len(labels) != len(set(labels)):
            raise ValueError("candidate labels must be unique")
        total = sum(item.probability for item in self.candidates)
        if self.candidates and abs(total - 1.0) > 1e-6:
            raise ValueError("candidate probabilities must sum to one")
        if self.candidates and self.primary_label not in labels:
            raise ValueError("primary label must occur among candidates")
        return self

    def probability(self, label: str) -> float:
        for candidate in self.candidates:
            if candidate.label == label:
                return candidate.probability
        return 0.0


class DeliberationAct(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_index: int = Field(ge=1, le=3)
    specialist: SpecialistKind
    act: EpistemicAct
    target_specialist: SpecialistKind | None = None
    claim: str = Field(min_length=1, max_length=4000)
    label: str = Field(min_length=1, max_length=256)
    confidence_before: float = Field(ge=0.0, le=1.0)
    confidence_after: float = Field(ge=0.0, le=1.0)
    evidence: tuple[EvidenceItem, ...] = ()


class RankedRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    label: str = Field(min_length=1, max_length=256)
    probability: float = Field(ge=0.0, le=1.0)
    weighted_support: float = Field(ge=0.0)
    supporting_specialists: tuple[SpecialistKind, ...]


class DeliberationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str
    recommendations: tuple[RankedRecommendation, ...]
    resolution_mode: ResolutionMode
    disagreement: float = Field(ge=0.0, le=1.0)
    disagreement_source: DisagreementSource | None = None
    rounds_completed: int = Field(ge=0, le=3)
    domain_weights: dict[SpecialistKind, float]
    acts: tuple[DeliberationAct, ...]
    conformal_labels: tuple[str, ...] = ()
    triage: TriageLevel = TriageLevel.EXPERT_REVIEW

    @model_validator(mode="after")
    def validate_ranking(self) -> "DeliberationResult":
        ranks = [item.rank for item in self.recommendations]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("recommendation ranks must be contiguous")
        probabilities = [item.probability for item in self.recommendations]
        if probabilities and abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("recommendation probabilities must sum to one")
        return self
