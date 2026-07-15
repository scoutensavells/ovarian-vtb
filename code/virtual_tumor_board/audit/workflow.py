from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowMeasures:
    available_before_board_fraction: float
    report_consultation_fraction: float
    baseline_preparation_minutes: float
    supported_preparation_minutes: float
    consulted_case_agreement: float

    @property
    def absolute_minutes_saved(self) -> float:
        return self.baseline_preparation_minutes - self.supported_preparation_minutes

    @property
    def relative_time_reduction(self) -> float:
        if self.baseline_preparation_minutes <= 0.0:
            raise ValueError("baseline preparation time must be positive")
        return self.absolute_minutes_saved / self.baseline_preparation_minutes


def workflow_measures(
    available_count: int,
    total_count: int,
    consultation_count: int,
    audit_count: int,
    baseline_preparation_minutes: float,
    supported_preparation_minutes: float,
    agreement_count: int,
) -> WorkflowMeasures:
    if min(total_count, audit_count, consultation_count) <= 0:
        raise ValueError("workflow counts must be positive")
    if not 0 <= available_count <= total_count:
        raise ValueError("available count must be within total count")
    if not 0 <= consultation_count <= audit_count:
        raise ValueError("consultation count must be within audit count")
    if not 0 <= agreement_count <= consultation_count:
        raise ValueError("agreement count must be within consultation count")
    return WorkflowMeasures(
        available_before_board_fraction=available_count / total_count,
        report_consultation_fraction=consultation_count / audit_count,
        baseline_preparation_minutes=baseline_preparation_minutes,
        supported_preparation_minutes=supported_preparation_minutes,
        consulted_case_agreement=agreement_count / consultation_count,
    )


def reported_workflow_measures() -> WorkflowMeasures:
    return WorkflowMeasures(
        available_before_board_fraction=0.943,
        report_consultation_fraction=0.712,
        baseline_preparation_minutes=18.6,
        supported_preparation_minutes=12.1,
        consulted_case_agreement=0.885,
    )
