from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_MEDIA_TYPES = {"photo", "txt", "pdf"}
VALID_MODALITIES = {"image", "text", "pdf"}


def _ensure_not_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class ManifestRow:
    doc_id: str
    category: str
    modality: str
    file_path: str
    source_tag: str
    language_guess: str
    status: str
    media_type: str
    original_filename: str

    def validate(self, workspace_root: Path) -> None:
        _ensure_not_empty(self.doc_id, "doc_id")
        _ensure_not_empty(self.category, "category")
        _ensure_not_empty(self.file_path, "file_path")
        _ensure_not_empty(self.source_tag, "source_tag")
        if self.media_type not in VALID_MEDIA_TYPES:
            raise ValueError(f"Invalid media_type: {self.media_type}")
        if self.modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality: {self.modality}")

        absolute = workspace_root / self.file_path
        if self.status == "ok" and not absolute.exists():
            raise ValueError(f"Manifest path does not exist: {self.file_path}")


@dataclass(frozen=True)
class QueryRow:
    query_id: str
    query_modality: str
    query_source: str
    query_text_or_path: str
    category_focus: str

    def validate(self, workspace_root: Path) -> None:
        _ensure_not_empty(self.query_id, "query_id")
        if self.query_modality not in {"text", "image"}:
            raise ValueError(f"Invalid query_modality: {self.query_modality}")
        _ensure_not_empty(self.query_source, "query_source")
        _ensure_not_empty(self.query_text_or_path, "query_text_or_path")
        _ensure_not_empty(self.category_focus, "category_focus")

        if self.query_modality == "image":
            absolute = workspace_root / self.query_text_or_path
            if not absolute.exists():
                raise ValueError(f"Image query path does not exist: {self.query_text_or_path}")


@dataclass(frozen=True)
class QrelRow:
    query_id: str
    doc_id: str
    relevance_grade: int | None
    annotation_notes: str
    tie_group: str | None

    def validate(self) -> None:
        _ensure_not_empty(self.query_id, "query_id")
        _ensure_not_empty(self.doc_id, "doc_id")
        if self.relevance_grade is not None and self.relevance_grade not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid relevance_grade: {self.relevance_grade}")