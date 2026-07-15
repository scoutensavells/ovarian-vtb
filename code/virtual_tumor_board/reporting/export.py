import csv
from pathlib import Path

import orjson
from virtual_tumor_board.reporting.schema import StudyReport


def export_json(report: StudyReport, path: Path) -> None:
    payload = orjson.dumps(
        report.model_dump(mode="json"),
        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
    )
    _atomic_write(path, payload + b"\n")


def export_metrics_csv(report: StudyReport, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "table",
                "cohort",
                "method",
                "run_index",
                "metric",
                "estimate",
                "lower",
                "upper",
                "unit",
                "count",
            ),
        )
        writer.writeheader()
        for table in report.tables:
            for metric in table.metrics:
                writer.writerow(
                    {
                        "table": table.identifier,
                        "cohort": metric.cohort,
                        "method": metric.method,
                        "run_index": metric.run_index,
                        "metric": metric.metric,
                        "estimate": metric.value.estimate,
                        "lower": metric.value.lower,
                        "upper": metric.value.upper,
                        "unit": metric.value.unit,
                        "count": metric.count,
                    }
                )
                count += 1
        stream.flush()
    temporary.replace(path)
    return count


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
