from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from virtual_tumor_board.records.evaluation import CaseEvaluation


@dataclass(frozen=True)
class SubgroupResult:
    axis: str
    value: str
    count: int
    diagnostic_accuracy: float
    guideline_concordance: float
    conformal_coverage: float


@dataclass(frozen=True)
class SubgroupSafety:
    axis: str
    groups: tuple[SubgroupResult, ...]
    minimum_accuracy: float
    maximum_accuracy_spread: float


def subgroup_results(
    records: tuple[CaseEvaluation, ...],
    axis: str,
    selector: Callable[[CaseEvaluation], str | None],
) -> SubgroupSafety:
    grouped: dict[str, list[CaseEvaluation]] = defaultdict(list)
    for record in records:
        value = selector(record)
        if value is not None:
            grouped[value].append(record)
    if not grouped:
        raise ValueError("no subgroup values were available")
    results = []
    for value, members in sorted(grouped.items()):
        results.append(
            SubgroupResult(
                axis=axis,
                value=value,
                count=len(members),
                diagnostic_accuracy=float(np.mean([item.diagnostic_correct for item in members])),
                guideline_concordance=float(
                    np.mean([item.guideline_concordant for item in members])
                ),
                conformal_coverage=float(np.mean([item.conformal_covered for item in members])),
            )
        )
    accuracies = [item.diagnostic_accuracy for item in results]
    return SubgroupSafety(
        axis=axis,
        groups=tuple(results),
        minimum_accuracy=min(accuracies),
        maximum_accuracy_spread=max(accuracies) - min(accuracies),
    )


def site_safety(records: tuple[CaseEvaluation, ...]) -> SubgroupSafety:
    return subgroup_results(records, "site", lambda item: item.site_code)


def region_safety(records: tuple[CaseEvaluation, ...]) -> SubgroupSafety:
    return subgroup_results(records, "region", lambda item: item.region_code)


def age_safety(records: tuple[CaseEvaluation, ...]) -> SubgroupSafety:
    return subgroup_results(records, "age_tercile", lambda item: item.age_tercile)


def histotype_safety(records: tuple[CaseEvaluation, ...]) -> SubgroupSafety:
    return subgroup_results(
        records, "histotype", lambda item: item.reference.confirmed_histotype.value
    )
