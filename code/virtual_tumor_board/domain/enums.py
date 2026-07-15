from enum import StrEnum


class Histotype(StrEnum):
    HIGH_GRADE_SEROUS = "high_grade_serous"
    ENDOMETRIOID = "endometrioid"
    CLEAR_CELL = "clear_cell"
    MUCINOUS = "mucinous"
    LOW_GRADE_SEROUS = "low_grade_serous"


class MolecularSubtype(StrEnum):
    IMMUNOREACTIVE = "immunoreactive"
    MESENCHYMAL = "mesenchymal"
    PROLIFERATIVE = "proliferative"
    DIFFERENTIATED = "differentiated"
    UNRESOLVED = "unresolved"


class SpecialistKind(StrEnum):
    PATHOLOGIST = "pathologist"
    RADIOLOGIST = "radiologist"
    MEDICAL_ONCOLOGIST = "medical_oncologist"
    MOLECULAR_BIOLOGIST = "molecular_biologist"


class EpistemicAct(StrEnum):
    PROPOSE = "propose"
    CHALLENGE = "challenge"
    SUPPORT = "support"
    SYNTHESIZE = "synthesize"


class DisagreementSource(StrEnum):
    INCOMPLETE = "incomplete"
    INTERPRETIVE = "interpretive"
    EQUIPOISE = "equipoise"


class EvidenceModality(StrEnum):
    DEMOGRAPHICS = "demographics"
    PATHOLOGY = "pathology"
    IMAGING = "imaging"
    GENOMICS = "genomics"
    CLINICAL_HISTORY = "clinical_history"


class FIGOStage(StrEnum):
    STAGE_I = "I"
    STAGE_II = "II"
    STAGE_III = "III"
    STAGE_IV = "IV"
    UNKNOWN = "unknown"


class BRCAStatus(StrEnum):
    PATHOGENIC = "pathogenic"
    WILD_TYPE = "wild_type"
    UNCERTAIN = "uncertain"
    NOT_TESTED = "not_tested"


class MenopausalStatus(StrEnum):
    PREMENOPAUSAL = "premenopausal"
    POSTMENOPAUSAL = "postmenopausal"
    UNKNOWN = "unknown"


class RecommendationKind(StrEnum):
    DIAGNOSIS = "diagnosis"
    STAGING = "staging"
    TREATMENT = "treatment"
    MOLECULAR = "molecular"


class ResolutionMode(StrEnum):
    CONSENSUS = "consensus"
    WEIGHTED = "weighted"
    RANKED_EQUIPOISE = "ranked_equipoise"


class TriageLevel(StrEnum):
    PROCEED = "proceed"
    FOCUSED_REVIEW = "focused_review"
    EXPERT_REVIEW = "expert_review"


class HarmGrade(StrEnum):
    NONE = "none"
    MINOR = "minor"
    SIGNIFICANT = "significant"


class FailureMode(StrEnum):
    HISTOTYPE_BOUNDARY = "histotype_boundary"
    GENOMIC_INTEGRATION = "genomic_integration"
    STAGE_TREATMENT_MISMATCH = "stage_treatment_mismatch"
    RARE_HISTOTYPE = "rare_histotype"


class CohortKind(StrEnum):
    PUBLIC_DEVELOPMENT = "public_development"
    PUBLIC_HOLDOUT = "public_holdout"
    RETROSPECTIVE = "retrospective"
    PROSPECTIVE = "prospective"
    READER_STUDY = "reader_study"
