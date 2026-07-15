from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator
from virtual_tumor_board.domain.enums import CohortKind, Histotype


class CohortDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: CohortKind
    planned_count: int = Field(gt=0)
    enrollment_start: date | None = None
    enrollment_end: date | None = None
    consecutive_enrollment: bool = False
    deidentified: bool = True
    influences_care: bool = False

    @model_validator(mode="after")
    def validate_enrollment(self) -> "CohortDefinition":
        if (
            self.enrollment_start is not None
            and self.enrollment_end is not None
            and self.enrollment_start > self.enrollment_end
        ):
            raise ValueError("enrollment start must precede enrollment end")
        return self


class SplitDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development_count: int = Field(gt=0)
    holdout_count: int = Field(gt=0)
    development_fraction: float = Field(gt=0.0, lt=1.0)
    holdout_fraction: float = Field(gt=0.0, lt=1.0)
    temporal: bool = True

    @model_validator(mode="after")
    def validate_fractions(self) -> "SplitDefinition":
        if abs(self.development_fraction + self.holdout_fraction - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to one")
        return self


class HistotypeComposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    histotype: Histotype
    count: int = Field(ge=0)
    fraction: float = Field(ge=0.0, le=1.0)


class PublicCohortProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_count: int = Field(gt=0)
    split: SplitDefinition
    composition: tuple[HistotypeComposition, ...]

    @model_validator(mode="after")
    def validate_profile(self) -> "PublicCohortProfile":
        if self.split.development_count + self.split.holdout_count != self.total_count:
            raise ValueError("split counts must equal total count")
        if sum(item.count for item in self.composition) != self.total_count:
            raise ValueError("histotype counts must equal total count")
        return self


def reported_cohorts() -> tuple[CohortDefinition, ...]:
    return (
        CohortDefinition(
            name="public_development",
            kind=CohortKind.PUBLIC_DEVELOPMENT,
            planned_count=410,
        ),
        CohortDefinition(
            name="public_temporal_holdout",
            kind=CohortKind.PUBLIC_HOLDOUT,
            planned_count=175,
        ),
        CohortDefinition(
            name="retrospective_supporting",
            kind=CohortKind.RETROSPECTIVE,
            planned_count=3847,
        ),
        CohortDefinition(
            name="prospective_silent_mode",
            kind=CohortKind.PROSPECTIVE,
            planned_count=1462,
            consecutive_enrollment=True,
        ),
        CohortDefinition(
            name="reader_study",
            kind=CohortKind.READER_STUDY,
            planned_count=420,
        ),
    )
