from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator
from virtual_tumor_board.domain.enums import BRCAStatus, FIGOStage, MenopausalStatus


class Demographics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    age_years: int = Field(ge=18, le=120)
    ecog_status: int = Field(ge=0, le=5)
    menopausal_status: MenopausalStatus


class PathologySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: tuple[str, ...] = ()
    cytology: tuple[str, ...] = ()
    nuclear_grade: int | None = Field(default=None, ge=1, le=3)
    mitotic_index: float | None = Field(default=None, ge=0.0)
    immunohistochemistry: dict[str, str] = Field(default_factory=dict)
    free_text: str = Field(min_length=1)


class ImagingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    laterality: str | None = None
    lesion_size_mm: float | None = Field(default=None, gt=0.0)
    peritoneal_involvement: bool | None = None
    ascites: bool | None = None
    lymph_node_involvement: bool | None = None
    distant_metastasis: bool | None = None
    ultrasound_features: tuple[str, ...] = ()
    ct_features: tuple[str, ...] = ()
    free_text: str = Field(min_length=1)


class GenomicProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    brca1_status: BRCAStatus = BRCAStatus.NOT_TESTED
    brca2_status: BRCAStatus = BRCAStatus.NOT_TESTED
    hrd_score: float | None = Field(default=None, ge=0.0)
    tumor_mutation_burden: float | None = Field(default=None, ge=0.0)
    mutations: tuple[str, ...] = ()
    copy_number_alterations: tuple[str, ...] = ()
    expression_scores: dict[str, float] = Field(default_factory=dict)
    ca125_units_ml: float | None = Field(default=None, ge=0.0)


class ClinicalHistory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    figo_stage: FIGOStage = FIGOStage.UNKNOWN
    symptoms: tuple[str, ...] = ()
    surgeries: tuple[str, ...] = ()
    residual_disease_mm: float | None = Field(default=None, ge=0.0)
    prior_therapies: tuple[str, ...] = ()
    treatment_outcomes: tuple[str, ...] = ()
    diagnosis_date: date | None = None
    free_text: str = Field(min_length=1)


class OvarianCancerCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    demographics: Demographics
    pathology: PathologySummary
    imaging: ImagingSummary
    genomics: GenomicProfile
    clinical_history: ClinicalHistory
    accession_date: date | None = None
    site_code: str | None = Field(default=None, max_length=32)
    region_code: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_dates(self) -> "OvarianCancerCase":
        diagnosis_date = self.clinical_history.diagnosis_date
        if (
            diagnosis_date is not None
            and self.accession_date is not None
            and diagnosis_date > self.accession_date
        ):
            raise ValueError("diagnosis_date must not follow accession_date")
        return self

    def is_complex(self, maximum_subtype_probability: float | None = None) -> bool:
        advanced = self.clinical_history.figo_stage in {FIGOStage.STAGE_III, FIGOStage.STAGE_IV}
        molecularly_ambiguous = (
            maximum_subtype_probability is not None and maximum_subtype_probability < 0.7
        )
        return advanced or molecularly_ambiguous

    def modality_completeness(self) -> dict[str, float]:
        pathology_items = len(self.pathology.architecture) + len(self.pathology.cytology)
        imaging_values = (
            self.imaging.peritoneal_involvement,
            self.imaging.ascites,
            self.imaging.lymph_node_involvement,
            self.imaging.distant_metastasis,
        )
        genomics_items = len(self.genomics.mutations) + len(self.genomics.expression_scores)
        clinical_items = len(self.clinical_history.symptoms) + len(self.clinical_history.surgeries)
        return {
            "demographics": 1.0,
            "pathology": min(1.0, pathology_items / 4.0),
            "imaging": sum(item is not None for item in imaging_values) / 4.0,
            "genomics": min(1.0, genomics_items / 4.0),
            "clinical_history": min(1.0, clinical_items / 4.0),
        }
