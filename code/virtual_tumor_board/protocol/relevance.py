from collections.abc import Mapping

from virtual_tumor_board.domain.assessment import SpecialistAssessment
from virtual_tumor_board.domain.case import OvarianCancerCase
from virtual_tumor_board.domain.enums import EvidenceModality, SpecialistKind

SPECIALIST_MODALITIES: dict[SpecialistKind, tuple[EvidenceModality, ...]] = {
    SpecialistKind.PATHOLOGIST: (EvidenceModality.PATHOLOGY,),
    SpecialistKind.RADIOLOGIST: (EvidenceModality.IMAGING,),
    SpecialistKind.MEDICAL_ONCOLOGIST: (
        EvidenceModality.CLINICAL_HISTORY,
        EvidenceModality.DEMOGRAPHICS,
    ),
    SpecialistKind.MOLECULAR_BIOLOGIST: (EvidenceModality.GENOMICS,),
}


def normalize_weights(values: Mapping[SpecialistKind, float]) -> dict[SpecialistKind, float]:
    clipped = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clipped.values())
    if total == 0.0:
        uniform = 1.0 / len(clipped)
        return {key: uniform for key in clipped}
    return {key: value / total for key, value in clipped.items()}


def evidence_relevance(assessment: SpecialistAssessment) -> float:
    allowed = set(SPECIALIST_MODALITIES[assessment.specialist])
    relevant = [item.relevance for item in assessment.evidence if item.modality in allowed]
    if not relevant:
        return 0.0
    return sum(relevant) / len(relevant)


def case_specific_weights(
    case: OvarianCancerCase,
    assessments: tuple[SpecialistAssessment, ...],
) -> dict[SpecialistKind, float]:
    completeness = case.modality_completeness()
    scores: dict[SpecialistKind, float] = {}
    for assessment in assessments:
        modalities = SPECIALIST_MODALITIES[assessment.specialist]
        modality_score = sum(completeness[item.value] for item in modalities) / len(modalities)
        evidence_score = evidence_relevance(assessment)
        uncertainty_penalty = 1.0 / (1.0 + len(assessment.uncertainties))
        score = 0.45 * modality_score + 0.35 * evidence_score + 0.20 * uncertainty_penalty
        scores[assessment.specialist] = score
    return normalize_weights(scores)


def log_odds_reliability_weight(reliability: float, classes: int) -> float:
    if classes < 2:
        raise ValueError("classes must be at least two")
    chance = 1.0 / classes
    if reliability <= chance or reliability >= 1.0:
        raise ValueError("reliability must be above chance and below one")
    import math

    return math.log(((classes - 1) * reliability) / (1.0 - reliability))
