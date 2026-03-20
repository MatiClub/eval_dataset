from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline_common import (
    FakeModelProvider,
    RetryPolicy,
    RealModelProvider,
    load_text_from_file,
    now_utc_iso,
    truncate_text,
)
from pipeline_runner import BasePhase2Pipeline

DEFAULT_EMBED_MODEL = "Qwen.Qwen3-VL-Embedding-2B"
DEFAULT_IMAGE_PROMPT = "Represent this image for retrieval:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Pipeline A: VL-only embeddings")
    parser.add_argument("--workspace-root", default=".", help="workspace root path")
    parser.add_argument("--manifest", default="artifacts/metadata/manifest.jsonl", help="metadata manifest jsonl")
    parser.add_argument("--queries", default="artifacts/metadata/queries.jsonl", help="metadata queries jsonl")
    parser.add_argument("--run-id", default=None, help="optional run id; generated if omitted")
    parser.add_argument("--base-url", default="http://localhost:8080", help="llama.cpp endpoint")
    parser.add_argument("--api-key", default=None, help="optional bearer token")
    parser.add_argument("--timeout", type=float, default=240.0, help="request timeout in seconds")
    parser.add_argument("--image-prompt", default=DEFAULT_IMAGE_PROMPT, help="prompt prefix for image embeddings")
    parser.add_argument("--retry-attempts", type=int, default=3, help="request retry attempts")
    parser.add_argument("--retry-base-delay", type=float, default=1.0, help="retry base delay seconds")
    parser.add_argument("--max-docs", type=int, default=None, help="optional doc cap for smoke run")
    parser.add_argument("--max-queries", type=int, default=None, help="optional query cap for smoke run")
    parser.add_argument("--fake-run", action="store_true", help="mock model calls and generate synthetic vectors")
    parser.add_argument("--fake-dim", type=int, default=256, help="fake embedding dimension")
    parser.add_argument("--fake-seed", type=int, default=17, help="fake run seed")
    parser.add_argument("--reset", action="store_true", help="ignore checkpoint and rebuild outputs")
    return parser.parse_args()


def _doc_embedding_input(row: dict[str, Any], workspace_root: Path) -> tuple[str, str, Path | None]:
    modality = str(row.get("modality", "")).strip().lower()
    rel_path = str(row.get("file_path", "")).strip()
    abs_path = workspace_root / rel_path

    if modality == "image":
        return modality, "", abs_path

    if modality == "text":
        text = load_text_from_file(abs_path, fallback_label=f"text document: {rel_path}")
        return modality, text, None

    # MVP path for PDFs without OCR: stable surrogate text from file metadata.
    pdf_text = f"pdf document path {rel_path} filename {abs_path.name}"
    return modality, truncate_text(pdf_text), None


def _query_embedding_input(row: dict[str, Any], workspace_root: Path) -> tuple[str, str, Path | None]:
    q_modality = str(row.get("query_modality", "")).strip().lower()
    payload = str(row.get("query_text_or_path", "")).strip()

    if q_modality == "image":
        return q_modality, "", workspace_root / payload

    return q_modality, truncate_text(payload), None


class VLEmbeddingPipeline(BasePhase2Pipeline):
    run_id_prefix = "vl"
    pipeline_name = "A_vl_only"

    @property
    def doc_jsonl_suffix(self) -> str:
        return "doc_vectors"

    @property
    def query_jsonl_suffix(self) -> str:
        return "query_vectors"

    @property
    def doc_parquet_suffix(self) -> str:
        return "doc_vectors"

    @property
    def query_parquet_suffix(self) -> str:
        return "query_vectors"

    def build_provider(self, args: argparse.Namespace) -> Any:
        if args.fake_run:
            return FakeModelProvider(vector_dim=args.fake_dim, seed=args.fake_seed)
        return RealModelProvider(
            base_url=args.base_url,
            embedding_model=DEFAULT_EMBED_MODEL,
            api_key=args.api_key,
            timeout=args.timeout,
            retry_policy=RetryPolicy(max_attempts=args.retry_attempts, base_delay_sec=args.retry_base_delay),
        )

    def build_run_manifest(self, args: argparse.Namespace, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "pipeline": self.pipeline_name,
            "fake_run": bool(args.fake_run),
            "embedding_model": DEFAULT_EMBED_MODEL,
            "endpoint": args.base_url,
            "image_prompt": args.image_prompt,
            "manifest_path": args.manifest,
            "queries_path": args.queries,
            "started_at": now_utc_iso(),
            "finished_at": None,
            "doc_rows_written": 0,
            "query_rows_written": 0,
        }

    def build_doc_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        modality, text_payload, image_path = _doc_embedding_input(row, workspace_root)
        if image_path is not None:
            vector = provider.embed_image(image_path=image_path, prompt_prefix=args.image_prompt)
            content_ref = row.get("file_path")
        else:
            vector = provider.embed_text(text_payload)
            content_ref = text_payload[:300]

        return {
            "run_id": run_id,
            "item_id": row["doc_id"],
            "item_type": "document",
            "category": row.get("category"),
            "modality": modality,
            "source_tag": row.get("source_tag"),
            "file_path": row.get("file_path"),
            "content_ref": content_ref,
            "vector_dim": len(vector),
            "vector": vector,
        }

    def build_query_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        q_modality, text_payload, image_path = _query_embedding_input(row, workspace_root)
        if image_path is not None:
            vector = provider.embed_image(image_path=image_path, prompt_prefix=args.image_prompt)
            content_ref = row.get("query_text_or_path")
        else:
            vector = provider.embed_text(text_payload)
            content_ref = text_payload[:300]

        return {
            "run_id": run_id,
            "item_id": row["query_id"],
            "item_type": "query",
            "category_focus": row.get("category_focus"),
            "modality": q_modality,
            "query_source": row.get("query_source"),
            "query_text_or_path": row.get("query_text_or_path"),
            "content_ref": content_ref,
            "vector_dim": len(vector),
            "vector": vector,
        }

    def output_paths_for_manifest(
        self,
        workspace_root: Path,
        docs_jsonl: Path,
        queries_jsonl: Path,
        docs_parquet: Path,
        queries_parquet: Path,
        checkpoint_path: Path,
    ) -> dict[str, str]:
        return {
            "doc_vectors_jsonl": docs_jsonl.relative_to(workspace_root).as_posix(),
            "query_vectors_jsonl": queries_jsonl.relative_to(workspace_root).as_posix(),
            "doc_vectors_parquet": docs_parquet.relative_to(workspace_root).as_posix(),
            "query_vectors_parquet": queries_parquet.relative_to(workspace_root).as_posix(),
            "checkpoint": checkpoint_path.relative_to(workspace_root).as_posix(),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    pipeline = VLEmbeddingPipeline()
    return pipeline.run(args)


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
        print(summary)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
