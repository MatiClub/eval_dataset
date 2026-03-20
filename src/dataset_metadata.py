from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from schemas import ManifestRow, QrelRow, QueryRow


TEXT_EXTENSIONS = {".txt"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


TEXT_QUERY_TEMPLATES = {
    "animal": [
        "find photos of animals",
        "show documents with pets",
        "images with cats or dogs",
        "animal picture examples",
    ],
    "car": [
        "photos of cars",
        "vehicle images",
        "automobile pictures",
        "car-related photos",
    ],
    "diploma": [
        "diploma or certificate image",
        "award certificate photo",
        "school diploma document",
        "certificate style page",
    ],
    "food": [
        "food image",
        "dish photo",
        "meal picture",
        "something edible shown in image",
    ],
    "food_recipe": [
        "recipe instructions text",
        "cooking recipe document",
        "ingredients and cooking steps",
        "text about preparing food",
    ],
    "identity": [
        "id card or passport photo",
        "identity document image",
        "personal identification card",
        "passport scan-like photo",
    ],
    "invoice": [
        "invoice document image",
        "billing invoice",
        "payment request document",
        "commercial invoice page",
    ],
    "medical": [
        "medical test or report image",
        "health record document",
        "clinical result picture",
        "medical paperwork",
    ],
    "receipt": [
        "shopping receipt image",
        "point of sale receipt",
        "cash register receipt",
        "purchase receipt paper",
    ],
    "syllabus": [
        "course syllabus document",
        "academic course outline",
        "university syllabus pdf",
        "class plan document",
    ],
    "warranty": [
        "warranty card or paper",
        "guarantee document image",
        "product warranty sheet",
        "proof of warranty image",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset metadata tooling")
    parser.add_argument("--workspace-root", default=".", help="workspace root path")
    parser.add_argument("--data-dir", default="data", help="source data directory")
    parser.add_argument("--output-dir", default="artifacts/metadata", help="output artifact directory")
    parser.add_argument("--query-count", type=int, default=60, help="total number of queries to generate")
    parser.add_argument(
        "--image-query-ratio",
        type=float,
        default=0.2,
        help="fraction of queries that should be image queries",
    )
    return parser.parse_args()


def _safe_token(value: str) -> str:
    output = []
    for char in value.lower():
        if char.isalnum() or char == "_":
            output.append(char)
        else:
            output.append("_")
    token = "".join(output).strip("_")
    return token or "unknown"


def _split_category_media(folder_name: str) -> tuple[str, str]:
    if "-" not in folder_name:
        raise ValueError(f"Folder name must follow <category>-<media_type>: {folder_name}")
    category, media_type = folder_name.rsplit("-", 1)
    if not category or not media_type:
        raise ValueError(f"Invalid folder name format: {folder_name}")
    return category, media_type


def _modality_from_media_type(media_type: str) -> str:
    if media_type == "photo":
        return "image"
    if media_type == "txt":
        return "text"
    if media_type == "pdf":
        return "pdf"
    return "text"


def _language_guess(path: Path, media_type: str) -> str:
    if media_type != "txt":
        return "unknown"

    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:1500].lower()
    except OSError:
        return "unknown"

    polish_hits = sum(token in sample for token in [" i ", " oraz ", " się ", " na ", " z "])
    english_hits = sum(token in sample for token in [" the ", " and ", " with ", " for ", " to "])
    if polish_hits > english_hits and polish_hits >= 2:
        return "pl"
    if english_hits > polish_hits and english_hits >= 2:
        return "en"
    return "unknown"


def _stable_doc_id(category: str, modality: str, relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"doc_{_safe_token(category)}_{_safe_token(modality)}_{digest}"


def _assert_extension_matches(path: Path, media_type: str) -> None:
    suffix = path.suffix.lower()
    if media_type == "photo" and suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unexpected extension for photo asset: {path}")
    if media_type == "txt" and suffix not in TEXT_EXTENSIONS:
        raise ValueError(f"Unexpected extension for txt asset: {path}")
    if media_type == "pdf" and suffix not in PDF_EXTENSIONS:
        raise ValueError(f"Unexpected extension for pdf asset: {path}")


def build_manifest(workspace_root: Path, data_dir: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []

    for category_folder in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        category, media_type = _split_category_media(category_folder.name)
        modality = _modality_from_media_type(media_type)
        for file_path in sorted([p for p in category_folder.iterdir() if p.is_file()], key=lambda p: p.name):
            _assert_extension_matches(file_path, media_type)
            rel_path = file_path.relative_to(workspace_root).as_posix()
            row = ManifestRow(
                doc_id=_stable_doc_id(category, modality, rel_path),
                category=category,
                modality=modality,
                file_path=rel_path,
                source_tag=category_folder.name,
                language_guess=_language_guess(file_path, media_type),
                status="ok" if file_path.exists() else "missing",
                media_type=media_type,
                original_filename=file_path.name,
            )
            row.validate(workspace_root)
            rows.append(row)

    unique_ids = {row.doc_id for row in rows}
    if len(unique_ids) != len(rows):
        raise ValueError("Duplicate doc_id detected in manifest")

    return rows


def _queries_for_category(category: str, count: int) -> list[str]:
    templates = TEXT_QUERY_TEMPLATES.get(category, [f"find items related to {category}"])
    output = []
    for idx in range(count):
        output.append(templates[idx % len(templates)])
    return output


def build_queries(
    workspace_root: Path,
    manifest: list[ManifestRow],
    total_queries: int,
    image_ratio: float,
) -> list[QueryRow]:
    if total_queries <= 0:
        raise ValueError("query-count must be > 0")

    image_docs = [row for row in manifest if row.modality == "image"]
    image_target = max(1, min(int(round(total_queries * image_ratio)), total_queries - 1))
    text_target = total_queries - image_target

    categories = sorted({row.category for row in manifest})
    per_category_text = max(1, text_target // max(1, len(categories)))

    rows: list[QueryRow] = []

    q_idx = 1
    for category in categories:
        for query_text in _queries_for_category(category, per_category_text):
            if len(rows) >= text_target:
                break
            row = QueryRow(
                query_id=f"q_{q_idx:04d}",
                query_modality="text",
                query_source="template_v1",
                query_text_or_path=query_text,
                category_focus=category,
            )
            row.validate(workspace_root)
            rows.append(row)
            q_idx += 1
        if len(rows) >= text_target:
            break

    while len(rows) < text_target:
        category = categories[(len(rows) - 1) % len(categories)]
        row = QueryRow(
            query_id=f"q_{q_idx:04d}",
            query_modality="text",
            query_source="template_v1",
            query_text_or_path=f"find items related to {category}",
            category_focus=category,
        )
        row.validate(workspace_root)
        rows.append(row)
        q_idx += 1

    selected_images = image_docs[:image_target]
    while len(selected_images) < image_target and image_docs:
        selected_images.append(image_docs[len(selected_images) % len(image_docs)])

    for doc in selected_images:
        row = QueryRow(
            query_id=f"q_{q_idx:04d}",
            query_modality="image",
            query_source="existing_doc_image",
            query_text_or_path=doc.file_path,
            category_focus=doc.category,
        )
        row.validate(workspace_root)
        rows.append(row)
        q_idx += 1

    unique_ids = {row.query_id for row in rows}
    if len(unique_ids) != len(rows):
        raise ValueError("Duplicate query_id detected")

    return rows[:total_queries]


def build_qrels_template(queries: list[QueryRow], manifest: list[ManifestRow]) -> list[QrelRow]:
    rows: list[QrelRow] = []
    for query in queries:
        for doc in manifest:
            rows.append(
                QrelRow(
                    query_id=query.query_id,
                    doc_id=doc.doc_id,
                    relevance_grade=None,
                    annotation_notes="",
                    tie_group=None,
                )
            )

    for row in rows:
        row.validate()
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def run_phase1(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    data_dir = (workspace_root / args.data_dir).resolve()
    output_dir = (workspace_root / args.output_dir).resolve()

    manifest = build_manifest(workspace_root=workspace_root, data_dir=data_dir)
    queries = build_queries(
        workspace_root=workspace_root,
        manifest=manifest,
        total_queries=args.query_count,
        image_ratio=args.image_query_ratio,
    )
    qrels = build_qrels_template(queries=queries, manifest=manifest)

    manifest_rows = [asdict(row) for row in manifest]
    query_rows = [asdict(row) for row in queries]
    qrel_rows = [asdict(row) for row in qrels]

    _write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    _write_jsonl(output_dir / "queries.jsonl", query_rows)
    _write_jsonl(output_dir / "qrels_template.jsonl", qrel_rows)

    summary = {
        "phase": "phase1",
        "manifest_docs": len(manifest_rows),
        "queries": len(query_rows),
        "qrels_template_rows": len(qrel_rows),
        "query_image_count": sum(1 for row in query_rows if row["query_modality"] == "image"),
        "query_text_count": sum(1 for row in query_rows if row["query_modality"] == "text"),
        "output_dir": output_dir.relative_to(workspace_root).as_posix(),
        "judgment_policy": {
            "scale": "0-3",
            "meaning": {
                "0": "not relevant",
                "1": "weakly relevant",
                "2": "relevant",
                "3": "highly relevant",
            },
            "tie_handling": "Use tie_group with same identifier for equal confidence labels.",
        },
    }
    _write_json(output_dir / "phase1_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    try:
        run_phase1(args)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())