from dataclasses import dataclass

from virtual_tumor_board.domain.case import OvarianCancerCase


@dataclass(frozen=True)
class VignetteSection:
    title: str
    fields: tuple[tuple[str, str], ...]

    def render(self) -> str:
        values = [self.title]
        values.extend(f"{key}: {value}" for key, value in self.fields)
        return "\n".join(values)


@dataclass(frozen=True)
class ClinicalVignette:
    case_key: str
    sections: tuple[VignetteSection, ...]

    def render(self) -> str:
        return "\n\n".join(section.render() for section in self.sections)


def compose_vignette(case: OvarianCancerCase) -> ClinicalVignette:
    sections = (
        _demographics(case),
        _pathology(case),
        _imaging(case),
        _genomics(case),
        _clinical_history(case),
    )
    return ClinicalVignette(case.case_key, sections)


def _demographics(case: OvarianCancerCase) -> VignetteSection:
    values = case.demographics
    return VignetteSection(
        title="Demographics",
        fields=(
            ("Age", str(values.age_years)),
            ("ECOG performance status", str(values.ecog_status)),
            ("Menopausal status", values.menopausal_status.value),
        ),
    )


def _pathology(case: OvarianCancerCase) -> VignetteSection:
    values = case.pathology
    fields = [
        ("Architecture", _sequence(values.architecture)),
        ("Cytology", _sequence(values.cytology)),
        ("Nuclear grade", _optional(values.nuclear_grade)),
        ("Mitotic index", _optional(values.mitotic_index)),
        ("Narrative", values.free_text),
    ]
    fields.extend(
        (f"IHC {key}", value) for key, value in sorted(values.immunohistochemistry.items())
    )
    return VignetteSection("Pathology summary", tuple(fields))


def _imaging(case: OvarianCancerCase) -> VignetteSection:
    values = case.imaging
    return VignetteSection(
        title="Imaging summary",
        fields=(
            ("Laterality", _optional(values.laterality)),
            ("Lesion size mm", _optional(values.lesion_size_mm)),
            ("Peritoneal involvement", _optional(values.peritoneal_involvement)),
            ("Ascites", _optional(values.ascites)),
            ("Lymph node involvement", _optional(values.lymph_node_involvement)),
            ("Distant metastasis", _optional(values.distant_metastasis)),
            ("Ultrasound features", _sequence(values.ultrasound_features)),
            ("CT features", _sequence(values.ct_features)),
            ("Narrative", values.free_text),
        ),
    )


def _genomics(case: OvarianCancerCase) -> VignetteSection:
    values = case.genomics
    fields = [
        ("BRCA1", values.brca1_status.value),
        ("BRCA2", values.brca2_status.value),
        ("HRD score", _optional(values.hrd_score)),
        ("Tumor mutation burden", _optional(values.tumor_mutation_burden)),
        ("Mutations", _sequence(values.mutations)),
        ("Copy number alterations", _sequence(values.copy_number_alterations)),
        ("CA-125", _optional(values.ca125_units_ml)),
    ]
    fields.extend(
        (f"Expression {key}", str(value)) for key, value in sorted(values.expression_scores.items())
    )
    return VignetteSection("Genomic profile", tuple(fields))


def _clinical_history(case: OvarianCancerCase) -> VignetteSection:
    values = case.clinical_history
    return VignetteSection(
        title="Clinical history",
        fields=(
            ("FIGO stage", values.figo_stage.value),
            ("Symptoms", _sequence(values.symptoms)),
            ("Surgeries", _sequence(values.surgeries)),
            ("Residual disease mm", _optional(values.residual_disease_mm)),
            ("Prior therapies", _sequence(values.prior_therapies)),
            ("Treatment outcomes", _sequence(values.treatment_outcomes)),
            ("Narrative", values.free_text),
        ),
    )


def _optional(value: object | None) -> str:
    return "not available" if value is None else str(value)


def _sequence(values: tuple[str, ...]) -> str:
    return "; ".join(values) if values else "not available"
