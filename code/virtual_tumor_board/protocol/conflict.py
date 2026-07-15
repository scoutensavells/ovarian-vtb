from collections import defaultdict

from virtual_tumor_board.domain.assessment import SpecialistAssessment
from virtual_tumor_board.domain.enums import DisagreementSource


def confidence_span(assessments: tuple[SpecialistAssessment, ...]) -> float:
    if not assessments:
        raise ValueError("at least one assessment is required")
    confidences = [item.confidence for item in assessments]
    return max(confidences) - min(confidences)


def label_entropy(assessments: tuple[SpecialistAssessment, ...]) -> float:
    import math

    if not assessments:
        raise ValueError("at least one assessment is required")
    counts: dict[str, int] = defaultdict(int)
    for assessment in assessments:
        counts[assessment.primary_label] += 1
    total = len(assessments)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    maximum = math.log(min(total, len(counts))) if len(counts) > 1 else 1.0
    return entropy / maximum if maximum > 0.0 else 0.0


def evidence_overlap(left: SpecialistAssessment, right: SpecialistAssessment) -> float:
    left_keys = {item.source_key for item in left.evidence}
    right_keys = {item.source_key for item in right.evidence}
    union = left_keys | right_keys
    if not union:
        return 0.0
    return len(left_keys & right_keys) / len(union)


def classify_disagreement(
    assessments: tuple[SpecialistAssessment, ...],
    threshold: float,
) -> DisagreementSource:
    if confidence_span(assessments) < threshold:
        raise ValueError("disagreement is below the conflict threshold")
    missing_evidence = sum(not item.evidence for item in assessments)
    unresolved = sum(bool(item.uncertainties) for item in assessments)
    overlaps = []
    for index, left in enumerate(assessments):
        for right in assessments[index + 1 :]:
            overlaps.append(evidence_overlap(left, right))
    average_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    distinct_labels = len({item.primary_label for item in assessments})
    if missing_evidence > 0 or unresolved >= len(assessments) // 2 + 1:
        return DisagreementSource.INCOMPLETE
    if average_overlap >= 0.5 and distinct_labels > 1:
        return DisagreementSource.INTERPRETIVE
    return DisagreementSource.EQUIPOISE


def pairwise_conflicts(
    assessments: tuple[SpecialistAssessment, ...],
    threshold: float,
) -> tuple[tuple[int, int, float], ...]:
    conflicts = []
    for left_index, left in enumerate(assessments):
        for right_index in range(left_index + 1, len(assessments)):
            right = assessments[right_index]
            difference = abs(left.confidence - right.confidence)
            if difference >= threshold or left.primary_label != right.primary_label:
                conflicts.append((left_index, right_index, difference))
    return tuple(conflicts)
