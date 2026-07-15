from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum

from virtual_tumor_board.domain.assessment import DeliberationAct, SpecialistAssessment
from virtual_tumor_board.domain.enums import DisagreementSource, ResolutionMode, SpecialistKind


class Phase(StrEnum):
    CREATED = "created"
    INDEPENDENT_ASSESSMENT = "independent_assessment"
    EVIDENCE_WEIGHTED_DEBATE = "evidence_weighted_debate"
    CONFLICT_RESOLUTION = "conflict_resolution"
    UNCERTAINTY_CALIBRATION = "uncertainty_calibration"
    COMPLETED = "completed"


@dataclass(frozen=True)
class Transition:
    sequence: int
    previous: Phase
    current: Phase
    recorded_at: datetime
    reason: str


@dataclass(frozen=True)
class RoundTrace:
    index: int
    assessments_before: tuple[SpecialistAssessment, ...]
    assessments_after: tuple[SpecialistAssessment, ...]
    acts: tuple[DeliberationAct, ...]
    weights: dict[SpecialistKind, float]
    disagreement_before: float
    disagreement_after: float


@dataclass(frozen=True)
class ProtocolTrace:
    case_key: str
    phase: Phase
    transitions: tuple[Transition, ...]
    initial_assessments: tuple[SpecialistAssessment, ...]
    rounds: tuple[RoundTrace, ...]
    disagreement_source: DisagreementSource | None
    resolution_mode: ResolutionMode | None
    calibration_quantile: float | None
    conformal_labels: tuple[str, ...]

    @classmethod
    def create(cls, case_key: str) -> "ProtocolTrace":
        if not case_key:
            raise ValueError("case key is required")
        return cls(
            case_key=case_key,
            phase=Phase.CREATED,
            transitions=(),
            initial_assessments=(),
            rounds=(),
            disagreement_source=None,
            resolution_mode=None,
            calibration_quantile=None,
            conformal_labels=(),
        )

    def begin_assessment(
        self,
        assessments: tuple[SpecialistAssessment, ...],
    ) -> "ProtocolTrace":
        self._require_phase(Phase.CREATED)
        specialists = [item.specialist for item in assessments]
        if set(specialists) != set(SpecialistKind) or len(specialists) != len(set(SpecialistKind)):
            raise ValueError("one assessment from every specialist is required")
        updated = replace(self, initial_assessments=assessments)
        return updated._transition(Phase.INDEPENDENT_ASSESSMENT, "specialist assessments recorded")

    def begin_debate(self) -> "ProtocolTrace":
        self._require_phase(Phase.INDEPENDENT_ASSESSMENT)
        return self._transition(Phase.EVIDENCE_WEIGHTED_DEBATE, "independent assessments complete")

    def append_round(self, round_trace: RoundTrace) -> "ProtocolTrace":
        self._require_phase(Phase.EVIDENCE_WEIGHTED_DEBATE)
        expected_index = len(self.rounds) + 1
        if round_trace.index != expected_index:
            raise ValueError("round indices must be contiguous")
        if round_trace.index > 3:
            raise ValueError("the protocol permits at most three rounds")
        if round_trace.disagreement_before < 0.0 or round_trace.disagreement_before > 1.0:
            raise ValueError("disagreement must lie between zero and one")
        if round_trace.disagreement_after < 0.0 or round_trace.disagreement_after > 1.0:
            raise ValueError("disagreement must lie between zero and one")
        return replace(self, rounds=(*self.rounds, round_trace))

    def begin_resolution(
        self,
        source: DisagreementSource | None,
        mode: ResolutionMode,
    ) -> "ProtocolTrace":
        if self.phase not in {Phase.INDEPENDENT_ASSESSMENT, Phase.EVIDENCE_WEIGHTED_DEBATE}:
            raise ValueError("resolution can only follow assessment or debate")
        updated = replace(self, disagreement_source=source, resolution_mode=mode)
        return updated._transition(
            Phase.CONFLICT_RESOLUTION, "debate termination condition reached"
        )

    def begin_calibration(self) -> "ProtocolTrace":
        self._require_phase(Phase.CONFLICT_RESOLUTION)
        if self.resolution_mode is None:
            raise ValueError("resolution mode must be recorded")
        return self._transition(Phase.UNCERTAINTY_CALIBRATION, "ranked recommendations resolved")

    def complete(
        self,
        calibration_quantile: float,
        conformal_labels: tuple[str, ...],
    ) -> "ProtocolTrace":
        self._require_phase(Phase.UNCERTAINTY_CALIBRATION)
        if not 0.0 <= calibration_quantile <= 1.0:
            raise ValueError("calibration quantile must lie between zero and one")
        if not conformal_labels:
            raise ValueError("at least one conformal label is required")
        updated = replace(
            self,
            calibration_quantile=calibration_quantile,
            conformal_labels=conformal_labels,
        )
        return updated._transition(Phase.COMPLETED, "conformal prediction set recorded")

    def _transition(self, target: Phase, reason: str) -> "ProtocolTrace":
        allowed = {
            Phase.CREATED: {Phase.INDEPENDENT_ASSESSMENT},
            Phase.INDEPENDENT_ASSESSMENT: {
                Phase.EVIDENCE_WEIGHTED_DEBATE,
                Phase.CONFLICT_RESOLUTION,
            },
            Phase.EVIDENCE_WEIGHTED_DEBATE: {Phase.CONFLICT_RESOLUTION},
            Phase.CONFLICT_RESOLUTION: {Phase.UNCERTAINTY_CALIBRATION},
            Phase.UNCERTAINTY_CALIBRATION: {Phase.COMPLETED},
            Phase.COMPLETED: set(),
        }
        if target not in allowed[self.phase]:
            raise ValueError(f"invalid phase transition from {self.phase} to {target}")
        transition = Transition(
            sequence=len(self.transitions) + 1,
            previous=self.phase,
            current=target,
            recorded_at=datetime.now(UTC),
            reason=reason,
        )
        return replace(self, phase=target, transitions=(*self.transitions, transition))

    def _require_phase(self, expected: Phase) -> None:
        if self.phase is not expected:
            raise ValueError(f"operation requires phase {expected}, found {self.phase}")


def trace_duration_seconds(trace: ProtocolTrace) -> float:
    if len(trace.transitions) < 2:
        return 0.0
    start = trace.transitions[0].recorded_at
    end = trace.transitions[-1].recorded_at
    return (end - start).total_seconds()


def phase_durations(trace: ProtocolTrace) -> dict[Phase, float]:
    durations: dict[Phase, float] = {}
    for current, following in zip(trace.transitions, trace.transitions[1:], strict=False):
        durations[current.current] = (following.recorded_at - current.recorded_at).total_seconds()
    return durations


def round_disagreement_changes(trace: ProtocolTrace) -> tuple[float, ...]:
    return tuple(item.disagreement_after - item.disagreement_before for item in trace.rounds)


def converged_early(trace: ProtocolTrace, threshold: float = 0.15) -> bool:
    if not trace.rounds:
        return False
    final = trace.rounds[-1]
    return len(trace.rounds) < 3 and final.disagreement_after < threshold
