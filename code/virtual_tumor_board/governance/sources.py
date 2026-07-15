from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse


class AccessClass(StrEnum):
    PUBLIC = "public"
    CONTROLLED = "controlled"
    PRIVATE = "private"


class SourceKind(StrEnum):
    GENOMIC = "genomic"
    IMAGING = "imaging"
    CLINICAL = "clinical"
    GUIDELINE = "guideline"


@dataclass(frozen=True)
class DataSource:
    key: str
    name: str
    kind: SourceKind
    access: AccessClass
    url: str | None
    version: str
    license_name: str
    permitted_for_public_pipeline: bool

    def validate(self) -> None:
        if not self.key or not self.name:
            raise ValueError("source key and name are required")
        if self.access is AccessClass.PUBLIC and self.url is None:
            raise ValueError("public sources require a URL")
        if self.url is not None:
            parsed = urlparse(self.url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("source URL must use HTTPS")
        if self.access is not AccessClass.PUBLIC and self.permitted_for_public_pipeline:
            raise ValueError("nonpublic sources cannot enter the public pipeline")


PUBLIC_SOURCES: tuple[DataSource, ...] = (
    DataSource(
        key="gdc_tcga_ov",
        name="TCGA-OV Genomic Data Commons",
        kind=SourceKind.GENOMIC,
        access=AccessClass.PUBLIC,
        url="https://portal.gdc.cancer.gov/projects/TCGA-OV",
        version="PanCancer Atlas",
        license_name="GDC Data Use Certification and project access controls",
        permitted_for_public_pipeline=True,
    ),
    DataSource(
        key="tcia_tcga_ov",
        name="TCGA-OV Cancer Imaging Archive",
        kind=SourceKind.IMAGING,
        access=AccessClass.PUBLIC,
        url="https://www.cancerimagingarchive.net/collection/tcga-ov/",
        version="TCGA-OV collection",
        license_name="TCIA Data Usage Policy and collection terms",
        permitted_for_public_pipeline=True,
    ),
    DataSource(
        key="cbioportal",
        name="cBioPortal for Cancer Genomics",
        kind=SourceKind.GENOMIC,
        access=AccessClass.PUBLIC,
        url="https://www.cbioportal.org/",
        version="study-specific",
        license_name="study-level licenses and portal terms",
        permitted_for_public_pipeline=True,
    ),
)


RESTRICTED_SOURCES: tuple[DataSource, ...] = (
    DataSource(
        key="retrospective_supporting_arm",
        name="Retrospective supporting arm",
        kind=SourceKind.CLINICAL,
        access=AccessClass.PRIVATE,
        url=None,
        version="study cohort",
        license_name="institutional data governance",
        permitted_for_public_pipeline=False,
    ),
    DataSource(
        key="prospective_silent_mode_arm",
        name="Prospective silent-mode arm",
        kind=SourceKind.CLINICAL,
        access=AccessClass.PRIVATE,
        url=None,
        version="study cohort",
        license_name="institutional data governance",
        permitted_for_public_pipeline=False,
    ),
)


def validate_registry() -> None:
    sources = PUBLIC_SOURCES + RESTRICTED_SOURCES
    keys = [item.key for item in sources]
    if len(keys) != len(set(keys)):
        raise ValueError("source registry keys must be unique")
    for source in sources:
        source.validate()


def public_urls() -> tuple[str, ...]:
    validate_registry()
    return tuple(item.url for item in PUBLIC_SOURCES if item.url is not None)


def source_by_key(key: str) -> DataSource:
    for source in PUBLIC_SOURCES + RESTRICTED_SOURCES:
        if source.key == key:
            return source
    raise KeyError(key)
