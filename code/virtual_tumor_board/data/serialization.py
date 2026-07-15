from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import orjson
from pydantic import BaseModel

RecordType = TypeVar("RecordType", bound=BaseModel)


def load_jsonl(path: Path, model: type[RecordType]) -> tuple[RecordType, ...]:
    _validate_input_path(path)
    records = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            records.append(model.model_validate(payload))
    return tuple(records)


def iter_jsonl(path: Path, model: type[RecordType]) -> Iterator[RecordType]:
    _validate_input_path(path)
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            yield model.model_validate(payload)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as stream:
        for record in records:
            stream.write(orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
            count += 1
        stream.flush()
    temporary.replace(path)
    return count


def load_json(path: Path, model: type[RecordType]) -> RecordType:
    _validate_input_path(path)
    try:
        payload = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError as error:
        raise ValueError("invalid JSON document") from error
    return model.model_validate(payload)


def write_json(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = orjson.dumps(
        record.model_dump(mode="json"),
        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
    )
    temporary.write_bytes(payload + b"\n")
    temporary.replace(path)


def _validate_input_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")
