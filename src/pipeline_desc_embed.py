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

DEFAULT_TEXT_EMBED_MODEL = "Qwen.Qwen3-Embedding-4B"
DEFAULT_VISION_MODEL = "Qwen.Qwen3-VL-4B-Instruct"
DEFAULT_DOC_PROMPT = (
    "Describe this document image for retrieval. Include document type, key entities, "
    "layout cues, and likely topic in one concise paragraph."
)
DEFAULT_QUERY_PROMPT = (
    "Describe this query image for retrieval intent. Include visible objects, document type, "
    "and likely search target in one concise paragraph."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Pipeline B: VL description plus text embedding")
    parser.add_argument("--workspace-root", default=".", help="workspace root path")
    parser.add_argument("--manifest", default="artifacts/metadata/manifest.jsonl", help="metadata manifest jsonl")
    parser.add_argument("--queries", default="artifacts/metadata/queries.jsonl", help="metadata queries jsonl")
    parser.add_argument("--run-id", default=None, help="optional run id; generated if omitted")
    parser.add_argument(
        "--mode",
        choices=["full", "descriptions-only", "embeddings-only"],
        default="full",
        help="execution mode: full pipeline, description cache only, or embeddings from cached descriptions",
    )
    parser.add_argument("--base-url", default="http://localhost:8080", help="llama.cpp endpoint")
    parser.add_argument("--api-key", default=None, help="optional bearer token")
    parser.add_argument("--timeout", type=float, default=240.0, help="request timeout in seconds")
    parser.add_argument("--doc-description-prompt", default=DEFAULT_DOC_PROMPT, help="prompt for document images")
    parser.add_argument("--query-description-prompt", default=DEFAULT_QUERY_PROMPT, help="prompt for query images")
    parser.add_argument("--retry-attempts", type=int, default=3, help="request retry attempts")
    parser.add_argument("--retry-base-delay", type=float, default=1.0, help="retry base delay seconds")
    parser.add_argument("--max-docs", type=int, default=None, help="optional doc cap for smoke run")
    parser.add_argument("--max-queries", type=int, default=None, help="optional query cap for smoke run")
    parser.add_argument("--fake-run", action="store_true", help="mock model calls and generate synthetic outputs")
    parser.add_argument("--fake-dim", type=int, default=256, help="fake embedding dimension")
    parser.add_argument("--fake-seed", type=int, default=19, help="fake run seed")
    parser.add_argument("--reset", action="store_true", help="ignore checkpoint and rebuild outputs")
    return parser.parse_args()


def _description_for_doc(
    provider: RealModelProvider | FakeModelProvider,
    row: dict[str, Any],
    workspace_root: Path,
    vision_model: str,
    doc_prompt: str,
) -> tuple[str, str]:
    modality = str(row.get("modality", "")).strip().lower()
    rel_path = str(row.get("file_path", "")).strip()
    abs_path = workspace_root / rel_path

    if modality == "image":
        return provider.describe_image(image_path=abs_path, vision_model=vision_model, prompt_text=doc_prompt), "model"

    if modality == "text":
        text = load_text_from_file(abs_path, fallback_label=f"text document: {rel_path}")
        return truncate_text(text), "passthrough_text"

    # MVP for PDF without OCR.
    description = f"PDF document named {abs_path.name} from path {rel_path}."
    return description, "pdf_surrogate"


def _description_for_query(
    provider: RealModelProvider | FakeModelProvider,
    row: dict[str, Any],
    workspace_root: Path,
    vision_model: str,
    query_prompt: str,
) -> tuple[str, str]:
    q_modality = str(row.get("query_modality", "")).strip().lower()
    payload = str(row.get("query_text_or_path", "")).strip()

    if q_modality == "image":
        abs_path = workspace_root / payload
        return provider.describe_image(image_path=abs_path, vision_model=vision_model, prompt_text=query_prompt), "model"

    return truncate_text(payload), "passthrough_text"


class DescriptionEmbeddingPipeline(BasePhase2Pipeline):
    run_id_prefix = "desc"
    pipeline_name = "B_vl_description_plus_text_embedding"

    @property
    def supports_cached_rows(self) -> bool:
        return True

    @property
    def doc_jsonl_suffix(self) -> str:
        return "doc_desc_vectors"

    @property
    def query_jsonl_suffix(self) -> str:
        return "query_desc_vectors"

    @property
    def doc_parquet_suffix(self) -> str:
        return "doc_desc_vectors"

    @property
    def query_parquet_suffix(self) -> str:
        return "query_desc_vectors"

    @property
    def doc_cache_jsonl_suffix(self) -> str:
        return "doc_descriptions"

    @property
    def query_cache_jsonl_suffix(self) -> str:
        return "query_descriptions"

    @property
    def doc_cache_parquet_suffix(self) -> str:
        return "doc_descriptions"

    @property
    def query_cache_parquet_suffix(self) -> str:
        return "query_descriptions"

    def build_provider(self, args: argparse.Namespace) -> Any:
        if args.fake_run:
            return FakeModelProvider(vector_dim=args.fake_dim, seed=args.fake_seed)
        return RealModelProvider(
            base_url=args.base_url,
            embedding_model=DEFAULT_TEXT_EMBED_MODEL,
            api_key=args.api_key,
            timeout=args.timeout,
            retry_policy=RetryPolicy(max_attempts=args.retry_attempts, base_delay_sec=args.retry_base_delay),
        )

    def build_run_manifest(self, args: argparse.Namespace, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "pipeline": self.pipeline_name,
            "fake_run": bool(args.fake_run),
            "vision_model": DEFAULT_VISION_MODEL,
            "text_embedding_model": DEFAULT_TEXT_EMBED_MODEL,
            "endpoint": args.base_url,
            "prompt_version": "v1",
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
        cache_row = self.build_doc_cache_row(
            provider=provider,
            row=row,
            workspace_root=workspace_root,
            run_id=run_id,
            args=args,
        )
        return self.build_doc_row_from_cache(
            row=cache_row,
            provider=provider,
            workspace_root=workspace_root,
            run_id=run_id,
            args=args,
        )

    def build_query_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        cache_row = self.build_query_cache_row(
            provider=provider,
            row=row,
            workspace_root=workspace_root,
            run_id=run_id,
            args=args,
        )
        return self.build_query_row_from_cache(
            row=cache_row,
            provider=provider,
            workspace_root=workspace_root,
            run_id=run_id,
            args=args,
        )

    def build_doc_cache_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        description, source_kind = _description_for_doc(
            provider=provider,
            row=row,
            workspace_root=workspace_root,
            vision_model=DEFAULT_VISION_MODEL,
            doc_prompt=args.doc_description_prompt,
        )
        return {
            "run_id": run_id,
            "item_id": row["doc_id"],
            "item_type": "document",
            "category": row.get("category"),
            "modality": row.get("modality"),
            "source_tag": row.get("source_tag"),
            "file_path": row.get("file_path"),
            "description_source": source_kind,
            "description_text": description,
        }

    def build_query_cache_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        description, source_kind = _description_for_query(
            provider=provider,
            row=row,
            workspace_root=workspace_root,
            vision_model=DEFAULT_VISION_MODEL,
            query_prompt=args.query_description_prompt,
        )
        return {
            "run_id": run_id,
            "item_id": row["query_id"],
            "item_type": "query",
            "category_focus": row.get("category_focus"),
            "modality": row.get("query_modality"),
            "query_source": row.get("query_source"),
            "query_text_or_path": row.get("query_text_or_path"),
            "description_source": source_kind,
            "description_text": description,
        }

    def _build_embedding_row_from_description(
        self,
        description_row: dict[str, Any],
        provider: Any,
        run_id: str,
    ) -> dict[str, Any]:
        description = str(description_row.get("description_text", "")).strip()
        if not description:
            item_id = str(description_row.get("item_id", "")).strip() or "<unknown>"
            raise ValueError(f"missing description_text for item_id '{item_id}'")
        vector = provider.embed_text(description)
        output = dict(description_row)
        output["run_id"] = run_id
        output["vector_dim"] = len(vector)
        output["vector"] = vector
        return output

    def build_doc_row_from_cache(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        return self._build_embedding_row_from_description(
            description_row=row,
            provider=provider,
            run_id=run_id,
        )

    def build_query_row_from_cache(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        return self._build_embedding_row_from_description(
            description_row=row,
            provider=provider,
            run_id=run_id,
        )

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
            "doc_desc_vectors_jsonl": docs_jsonl.relative_to(workspace_root).as_posix(),
            "query_desc_vectors_jsonl": queries_jsonl.relative_to(workspace_root).as_posix(),
            "doc_desc_vectors_parquet": docs_parquet.relative_to(workspace_root).as_posix(),
            "query_desc_vectors_parquet": queries_parquet.relative_to(workspace_root).as_posix(),
            "embedding_checkpoint": checkpoint_path.relative_to(workspace_root).as_posix(),
        }

    def extra_output_paths_for_manifest(
        self,
        workspace_root: Path,
        docs_jsonl: Path,
        queries_jsonl: Path,
        docs_parquet: Path,
        queries_parquet: Path,
        checkpoint_path: Path,
        cache_docs_jsonl: Path | None,
        cache_queries_jsonl: Path | None,
        cache_docs_parquet: Path | None,
        cache_queries_parquet: Path | None,
        cache_checkpoint_path: Path | None,
    ) -> dict[str, str]:
        outputs: dict[str, str] = {}
        if cache_docs_jsonl is not None:
            outputs["doc_descriptions_jsonl"] = cache_docs_jsonl.relative_to(workspace_root).as_posix()
        if cache_queries_jsonl is not None:
            outputs["query_descriptions_jsonl"] = cache_queries_jsonl.relative_to(workspace_root).as_posix()
        if cache_docs_parquet is not None:
            outputs["doc_descriptions_parquet"] = cache_docs_parquet.relative_to(workspace_root).as_posix()
        if cache_queries_parquet is not None:
            outputs["query_descriptions_parquet"] = cache_queries_parquet.relative_to(workspace_root).as_posix()
        if cache_checkpoint_path is not None:
            outputs["description_checkpoint"] = cache_checkpoint_path.relative_to(workspace_root).as_posix()
        return outputs

    def update_run_manifest(self, run_manifest: dict[str, Any], context: dict[str, Any]) -> None:
        run_manifest["doc_description_rows_written"] = int(context.get("cache_doc_rows_written", 0))
        run_manifest["query_description_rows_written"] = int(context.get("cache_query_rows_written", 0))
        run_manifest["doc_description_rows_reused"] = int(context.get("cache_doc_rows_reused", 0))
        run_manifest["query_description_rows_reused"] = int(context.get("cache_query_rows_reused", 0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    pipeline = DescriptionEmbeddingPipeline()
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
