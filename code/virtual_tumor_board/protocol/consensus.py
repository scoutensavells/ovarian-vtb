from collections import defaultdict
from collections.abc import Mapping

from virtual_tumor_board.domain.assessment import (
    RankedRecommendation,
    SpecialistAssessment,
)
from virtual_tumor_board.domain.enums import SpecialistKind


def candidate_labels(assessments: tuple[SpecialistAssessment, ...]) -> tuple[str, ...]:
    labels = set()
    for assessment in assessments:
        labels.add(assessment.primary_label)
        labels.update(item.label for item in assessment.candidates)
    return tuple(sorted(labels))


def weighted_scores(
    assessments: tuple[SpecialistAssessment, ...],
    weights: Mapping[SpecialistKind, float],
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for assessment in assessments:
        weight = weights[assessment.specialist]
        if assessment.candidates:
            for candidate in assessment.candidates:
                scores[candidate.label] += weight * candidate.probability
        else:
            scores[assessment.primary_label] += weight * assessment.confidence
    return dict(scores)


def normalized_scores(scores: Mapping[str, float]) -> dict[str, float]:
    nonnegative = {key: max(0.0, float(value)) for key, value in scores.items()}
    total = sum(nonnegative.values())
    if total <= 0.0:
        raise ValueError("scores must include positive support")
    return {key: value / total for key, value in nonnegative.items()}


def rank_recommendations(
    assessments: tuple[SpecialistAssessment, ...],
    weights: Mapping[SpecialistKind, float],
) -> tuple[RankedRecommendation, ...]:
    raw = weighted_scores(assessments, weights)
    probabilities = normalized_scores(raw)
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    ranked = []
    for index, (label, probability) in enumerate(ordered, start=1):
        supporters = tuple(
            assessment.specialist
            for assessment in assessments
            if assessment.primary_label == label or assessment.probability(label) > 0.0
        )
        ranked.append(
            RankedRecommendation(
                rank=index,
                label=label,
                probability=probability,
                weighted_support=raw[label],
                supporting_specialists=supporters,
            )
        )
    return tuple(ranked)


def majority_vote(assessments: tuple[SpecialistAssessment, ...]) -> str:
    counts: dict[str, int] = defaultdict(int)
    confidence: dict[str, float] = defaultdict(float)
    for assessment in assessments:
        counts[assessment.primary_label] += 1
        confidence[assessment.primary_label] += assessment.confidence
    return max(counts, key=lambda label: (counts[label], confidence[label], label))
