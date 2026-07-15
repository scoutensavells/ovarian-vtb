from pathlib import Path

import typer

from virtual_tumor_board.audit.prospective import reported_prospective_success
from virtual_tumor_board.audit.safety import reported_deployment_safety
from virtual_tumor_board.data.serialization import load_jsonl
from virtual_tumor_board.evaluation.cohort import (
    aggregate_all_runs,
    group_by_method_and_run,
    summarize_cohort,
)
from virtual_tumor_board.governance.sources import public_urls, validate_registry
from virtual_tumor_board.records.evaluation import CaseEvaluation
from virtual_tumor_board.reporting.builders import (
    aggregate_table,
    cohort_table,
    safety_table,
    study_report,
)
from virtual_tumor_board.reporting.export import export_json

application = typer.Typer(no_args_is_help=True)


@application.command()
def evaluate(input_path: Path, output_path: Path) -> None:
    records = load_jsonl(input_path, CaseEvaluation)
    grouped = group_by_method_and_run(records)
    summaries = tuple(summarize_cohort(values) for _, values in sorted(grouped.items()))
    aggregates = aggregate_all_runs(records)
    tables = (
        cohort_table("case_results", "Case-level evaluation", summaries),
        aggregate_table("run_results", "Independent-run aggregation", aggregates),
    )
    report = study_report(
        tables=tables,
        case_count=len({item.reference.case_key for item in records}),
        method_count=len({item.prediction.method for item in records}),
    )
    export_json(report, output_path)


@application.command()
def audit(output_path: Path) -> None:
    tables = (
        safety_table(
            "prospective_criteria",
            "Prospective success criteria",
            reported_prospective_success(),
        ),
        safety_table(
            "deployment_safety",
            "Deployment safety criteria",
            reported_deployment_safety(),
        ),
    )
    export_json(study_report(tables, 0, 1), output_path)


@application.command()
def calibrate() -> None:
    validate_registry()
    for url in public_urls():
        typer.echo(url)


if __name__ == "__main__":
    application()
