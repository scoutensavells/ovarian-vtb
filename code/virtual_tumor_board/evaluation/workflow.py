from dataclasses import dataclass

import numpy as np
from virtual_tumor_board.records.evaluation import WorkflowObservation


@dataclass(frozen=True)
class WorkflowSummary:
    count: int
    availability: float
    consultation_rate: float
    consulted_agreement: float
    consulted_median_minutes: float
    unconsulted_median_minutes: float
    median_minutes_saved: float


def summarize_workflow(records: tuple[WorkflowObservation, ...]) -> WorkflowSummary:
    if not records:
        raise ValueError("workflow observations must not be empty")
    ready = np.asarray([item.ready_before_board for item in records], dtype=np.float64)
    consulted = np.asarray([item.report_consulted for item in records], dtype=np.bool_)
    agreement = np.asarray([item.board_agreed for item in records], dtype=np.bool_)
    minutes = np.asarray([item.preparation_minutes for item in records], dtype=np.float64)
    if not np.any(consulted):
        raise ValueError("at least one consulted case is required")
    if np.all(consulted):
        raise ValueError("at least one unconsulted case is required")
    consulted_minutes = float(np.median(minutes[consulted]))
    unconsulted_minutes = float(np.median(minutes[~consulted]))
    return WorkflowSummary(
        count=len(records),
        availability=float(np.mean(ready)),
        consultation_rate=float(np.mean(consulted)),
        consulted_agreement=float(np.mean(agreement[consulted])),
        consulted_median_minutes=consulted_minutes,
        unconsulted_median_minutes=unconsulted_minutes,
        median_minutes_saved=unconsulted_minutes - consulted_minutes,
    )


def preparation_time_distribution(
    records: tuple[WorkflowObservation, ...],
) -> dict[str, float]:
    if not records:
        raise ValueError("workflow observations must not be empty")
    values = np.asarray([item.preparation_minutes for item in records], dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "first_quartile": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "third_quartile": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }
