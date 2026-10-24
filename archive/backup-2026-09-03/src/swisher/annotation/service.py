from __future__ import annotations

from pathlib import Path

from swisher.data.io import export_annotation_csv, load_annotations
from swisher.data.models import ShotAnnotation

DIAGNOSIS_LABELS_HE: dict[str, str] = {
    "balance": "חוסר איזון",
    "foot_base": "בסיס רגליים לא עקבי",
    "leg_use": "שימוש חלש ברגליים",
    "release_timing": "תזמון שחרור לא עקבי",
    "elbow_flare": "המרפק בורח החוצה",
    "follow_through": "המשך תנועה קצר",
    "lateral_drift": "סטייה צדית של הגוף",
    "tempo": "קצב זריקה לא עקבי",
    "release_point_consistency": "נקודת שחרור לא עקבית",
    "landing": "נחיתה לא יציבה",
    "low_trajectory": "מסלול כדור נמוך",
    "excess_tension": "מתח מיותר בתנועה",
}


def next_shot_id(records: list[ShotAnnotation], video_id: str) -> str:
    matching = [record for record in records if record.video_id == video_id]
    return f"shot_{len(matching) + 1:03d}"


class AnnotationStore:
    """Deterministic local store for the pilot annotation workflow."""

    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path
        self.csv_path = jsonl_path.with_suffix(".csv")

    def all(self) -> list[ShotAnnotation]:
        return load_annotations(self.jsonl_path)

    def upsert(self, record: ShotAnnotation) -> None:
        records = self.all()
        for index, existing in enumerate(records):
            same_identity = (
                existing.session_id == record.session_id
                and existing.video_id == record.video_id
                and existing.shot_id == record.shot_id
            )
            if same_identity:
                record = record.model_copy(
                    update={
                        "annotation_id": existing.annotation_id,
                        "annotation_version": existing.annotation_version + 1,
                    }
                )
                records[index] = record
                break
        else:
            records.append(record)
        self._rewrite(records)
        export_annotation_csv(records, self.csv_path)

    def _rewrite(self, records: list[ShotAnnotation]) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.jsonl_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json())
                handle.write("\n")
        temporary.replace(self.jsonl_path)
