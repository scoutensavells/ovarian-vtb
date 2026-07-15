from collections.abc import Mapping
from dataclasses import dataclass
from re import compile as compile_pattern


@dataclass(frozen=True)
class PrivacyFinding:
    field: str
    category: str
    matched_text: str


PATTERNS = {
    "email": compile_pattern(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": compile_pattern(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)"),
    "medical_record_number": compile_pattern(
        r"\b(?:MRN|medical record)\s*[:#-]?\s*[A-Za-z0-9-]{5,}\b", 2
    ),
    "absolute_path": compile_pattern(r"(?:/[A-Za-z0-9._-]+){3,}"),
}


def scan_fields(values: Mapping[str, str]) -> tuple[PrivacyFinding, ...]:
    findings = []
    for field, value in values.items():
        for category, pattern in PATTERNS.items():
            for match in pattern.finditer(value):
                findings.append(PrivacyFinding(field, category, match.group(0)))
    return tuple(findings)


def require_deidentified(values: Mapping[str, str]) -> None:
    findings = scan_fields(values)
    if findings:
        categories = sorted({item.category for item in findings})
        raise ValueError("privacy-sensitive content detected: " + ", ".join(categories))


def redact_fields(values: Mapping[str, str]) -> dict[str, str]:
    redacted = {}
    for field, value in values.items():
        current = value
        for category, pattern in PATTERNS.items():
            current = pattern.sub(f"[{category}_removed]", current)
        redacted[field] = current
    return redacted
