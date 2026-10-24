from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from .models import ShotAnnotation


def append_jsonl(records: Iterable[BaseModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def load_annotations(path: Path) -> list[ShotAnnotation]:
    if not path.exists():
        return []
    records: list[ShotAnnotation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(ShotAnnotation.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid annotation on line {line_number}: {error}") from error
    return records


def export_annotation_csv(records: Iterable[ShotAnnotation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "annotation_id",
        "session_id",
        "video_id",
        "shot_id",
        "annotator_id",
        "phase",
        "outcome",
        "shot_start",
        "load",
        "release",
        "rim_interaction",
        "shot_end",
        "diagnosis_code",
        "coach_cue",
        "drill_name",
        "confidence",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            data = record.model_dump(mode="json")
            events = data.pop("events")
            row = {field: data.get(field) for field in fields}
            row.update(events)
            writer.writerow(row)


def atomic_write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
