from dataclasses import dataclass
from enum import StrEnum


class CriterionStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_ASSESSABLE = "not_assessable"


@dataclass(frozen=True)
class Criterion:
    name: str
    observed: float
    threshold: float
    comparison: str
    status: CriterionStatus


@dataclass(frozen=True)
class AuditSummary:
    name: str
    criteria: tuple[Criterion, ...]

    @property
    def passed(self) -> int:
        return sum(item.status is CriterionStatus.PASS for item in self.criteria)

    @property
    def failed(self) -> int:
        return sum(item.status is CriterionStatus.FAIL for item in self.criteria)

    @property
    def assessable(self) -> int:
        return sum(item.status is not CriterionStatus.NOT_ASSESSABLE for item in self.criteria)

    @property
    def success(self) -> bool:
        return self.assessable > 0 and self.failed == 0


def at_least(name: str, observed: float, threshold: float) -> Criterion:
    status = CriterionStatus.PASS if observed >= threshold else CriterionStatus.FAIL
    return Criterion(name, observed, threshold, ">=", status)


def below(name: str, observed: float, threshold: float) -> Criterion:
    status = CriterionStatus.PASS if observed < threshold else CriterionStatus.FAIL
    return Criterion(name, observed, threshold, "<", status)


def at_most(name: str, observed: float, threshold: float) -> Criterion:
    status = CriterionStatus.PASS if observed <= threshold else CriterionStatus.FAIL
    return Criterion(name, observed, threshold, "<=", status)


def not_assessable(name: str, threshold: float, comparison: str) -> Criterion:
    return Criterion(name, float("nan"), threshold, comparison, CriterionStatus.NOT_ASSESSABLE)
