from dataclasses import dataclass


@dataclass(frozen=True)
class ReaderArm:
    name: str
    diagnostic_accuracy: float
    guideline_concordance: float
    complex_accuracy: float
    median_seconds: float
    confidence: float | None


@dataclass(frozen=True)
class ReaderComparison:
    team_gain_over_clinician: float
    team_gain_over_model: float
    guideline_gain_over_clinician: float
    guideline_gain_over_model: float
    time_change_from_clinician: float
    confidence_change_from_clinician: float | None


def compare_reader_arms(
    clinician: ReaderArm,
    model: ReaderArm,
    team: ReaderArm,
) -> ReaderComparison:
    if len({clinician.name, model.name, team.name}) != 3:
        raise ValueError("reader arms must have distinct names")
    confidence_change = None
    if clinician.confidence is not None and team.confidence is not None:
        confidence_change = team.confidence - clinician.confidence
    return ReaderComparison(
        team_gain_over_clinician=team.diagnostic_accuracy - clinician.diagnostic_accuracy,
        team_gain_over_model=team.diagnostic_accuracy - model.diagnostic_accuracy,
        guideline_gain_over_clinician=team.guideline_concordance - clinician.guideline_concordance,
        guideline_gain_over_model=team.guideline_concordance - model.guideline_concordance,
        time_change_from_clinician=team.median_seconds - clinician.median_seconds,
        confidence_change_from_clinician=confidence_change,
    )


def reported_reader_comparison() -> ReaderComparison:
    clinician = ReaderArm("clinician", 0.867, 0.859, 0.741, 684.0, 7.1)
    model = ReaderArm("model", 0.901, 0.886, 0.846, 74.0, None)
    team = ReaderArm("team", 0.938, 0.924, 0.853, 408.0, 8.4)
    return compare_reader_arms(clinician, model, team)
