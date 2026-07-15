from collections.abc import Callable, Mapping

from virtual_tumor_board.domain.assessment import (
    DeliberationAct,
    DeliberationResult,
    SpecialistAssessment,
)
from virtual_tumor_board.domain.case import OvarianCancerCase
from virtual_tumor_board.domain.enums import (
    DisagreementSource,
    ResolutionMode,
    SpecialistKind,
    TriageLevel,
)
from virtual_tumor_board.protocol.conflict import classify_disagreement, confidence_span
from virtual_tumor_board.protocol.consensus import rank_recommendations
from virtual_tumor_board.protocol.relevance import case_specific_weights
from virtual_tumor_board.protocol.settings import ProtocolSettings

RoundFunction = Callable[
    [int, tuple[SpecialistAssessment, ...], Mapping[SpecialistKind, float]],
    tuple[tuple[SpecialistAssessment, ...], tuple[DeliberationAct, ...]],
]


class DeliberationEngine:
    def __init__(self, settings: ProtocolSettings) -> None:
        self.settings = settings

    def execute(
        self,
        case: OvarianCancerCase,
        assessments: tuple[SpecialistAssessment, ...],
        round_function: RoundFunction | None = None,
    ) -> DeliberationResult:
        self._validate_specialists(assessments)
        current = assessments
        acts: list[DeliberationAct] = []
        rounds_completed = 0
        weights = case_specific_weights(case, current)
        for round_index in range(1, self.settings.max_rounds + 1):
            if confidence_span(current) < self.settings.conflict_threshold:
                break
            if round_function is None:
                break
            current, round_acts = round_function(round_index, current, weights)
            self._validate_specialists(current)
            acts.extend(round_acts)
            rounds_completed = round_index
            weights = case_specific_weights(case, current)
        disagreement = confidence_span(current)
        source: DisagreementSource | None = None
        if disagreement >= self.settings.conflict_threshold:
            source = classify_disagreement(current, self.settings.conflict_threshold)
        if source is DisagreementSource.EQUIPOISE:
            mode = ResolutionMode.RANKED_EQUIPOISE
        elif source is not None:
            mode = ResolutionMode.WEIGHTED
        else:
            mode = ResolutionMode.CONSENSUS
        recommendations = rank_recommendations(current, weights)
        triage = self._triage(disagreement, source, len(recommendations))
        return DeliberationResult(
            case_key=case.case_key,
            recommendations=recommendations,
            resolution_mode=mode,
            disagreement=disagreement,
            disagreement_source=source,
            rounds_completed=rounds_completed,
            domain_weights=weights,
            acts=tuple(acts),
            triage=triage,
        )

    def _validate_specialists(self, assessments: tuple[SpecialistAssessment, ...]) -> None:
        expected = set(SpecialistKind)
        observed = {item.specialist for item in assessments}
        if observed != expected or len(assessments) != len(expected):
            raise ValueError("one assessment from each specialist is required")

    def _triage(
        self,
        disagreement: float,
        source: DisagreementSource | None,
        candidate_count: int,
    ) -> TriageLevel:
        if source is DisagreementSource.EQUIPOISE or candidate_count > 2:
            return TriageLevel.EXPERT_REVIEW
        if disagreement >= self.settings.conflict_threshold:
            return TriageLevel.FOCUSED_REVIEW
        return TriageLevel.PROCEED
