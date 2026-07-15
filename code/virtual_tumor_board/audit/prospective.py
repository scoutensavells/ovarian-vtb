from dataclasses import dataclass

from virtual_tumor_board.audit.common import AuditSummary, at_least


@dataclass(frozen=True)
class ProspectiveMeasures:
    board_concordance: float
    diagnostic_accuracy: float
    board_first_pass_accuracy: float
    guideline_concordance: float
    conformal_coverage: float
    minimum_subgroup_accuracy: float


def prospective_success(measures: ProspectiveMeasures) -> AuditSummary:
    return AuditSummary(
        name="prospective_silent_mode",
        criteria=(
            at_least("board_concordance", measures.board_concordance, 0.85),
            at_least(
                "diagnostic_noninferiority",
                measures.diagnostic_accuracy,
                measures.board_first_pass_accuracy - 0.05,
            ),
            at_least("guideline_concordance", measures.guideline_concordance, 0.85),
            at_least("conformal_coverage", measures.conformal_coverage, 0.90),
            at_least("minimum_subgroup_accuracy", measures.minimum_subgroup_accuracy, 0.80),
        ),
    )


def reported_prospective_success() -> AuditSummary:
    return prospective_success(
        ProspectiveMeasures(
            board_concordance=0.894,
            diagnostic_accuracy=0.901,
            board_first_pass_accuracy=0.879,
            guideline_concordance=0.886,
            conformal_coverage=0.908,
            minimum_subgroup_accuracy=0.820,
        )
    )
