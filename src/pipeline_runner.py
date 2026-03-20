from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pipeline_common import (
    CheckpointStore,
    ensure_dir,
    jsonl_row_count,
    jsonl_to_parquet,
    JsonlAppender,
    make_run_id,
    now_utc_iso,
    read_jsonl,
    validate_unique_ids,
    write_json,
)


def progress(iterable: Any, **kwargs: Any) -> Any:
    try:
        tqdm_module = __import__("tqdm.auto", fromlist=["tqdm"])
        return tqdm_module.tqdm(iterable, **kwargs)
    except Exception:
        return iterable


class BasePhase2Pipeline(ABC):
    run_id_prefix: str
    pipeline_name: str

    @property
    @abstractmethod
    def doc_jsonl_suffix(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def query_jsonl_suffix(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def doc_parquet_suffix(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def query_parquet_suffix(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_provider(self, args: argparse.Namespace) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_run_manifest(self, args: argparse.Namespace, run_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_doc_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_query_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def output_paths_for_manifest(
        self,
        workspace_root: Path,
        docs_jsonl: Path,
        queries_jsonl: Path,
        docs_parquet: Path,
        queries_parquet: Path,
        checkpoint_path: Path,
    ) -> dict[str, str]:
        raise NotImplementedError

    @property
    def supports_cached_rows(self) -> bool:
        return False

    @property
    def doc_cache_jsonl_suffix(self) -> str:
        return "doc_cache"

    @property
    def query_cache_jsonl_suffix(self) -> str:
        return "query_cache"

    @property
    def doc_cache_parquet_suffix(self) -> str:
        return "doc_cache"

    @property
    def query_cache_parquet_suffix(self) -> str:
        return "query_cache"

    def build_doc_cache_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        raise NotImplementedError("build_doc_cache_row is required when supports_cached_rows=True")

    def build_query_cache_row(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        raise NotImplementedError("build_query_cache_row is required when supports_cached_rows=True")

    def build_doc_row_from_cache(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        raise NotImplementedError("build_doc_row_from_cache is required when supports_cached_rows=True")

    def build_query_row_from_cache(
        self,
        row: dict[str, Any],
        provider: Any,
        workspace_root: Path,
        run_id: str,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        raise NotImplementedError("build_query_row_from_cache is required when supports_cached_rows=True")

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
        return {}

    def update_run_manifest(
        self,
        run_manifest: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        return None

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        workspace_root = Path(args.workspace_root).resolve()
        manifest_path = (workspace_root / args.manifest).resolve()
        queries_path = (workspace_root / args.queries).resolve()
        run_id = args.run_id or make_run_id(self.run_id_prefix)
        mode = str(getattr(args, "mode", "full"))
        if mode not in {"full", "descriptions-only", "embeddings-only"}:
            raise ValueError(f"unsupported mode: {mode}")
        if not self.supports_cached_rows and mode != "full":
            raise ValueError(f"pipeline '{self.pipeline_name}' does not support mode '{mode}'")

        output_dir = (workspace_root / "artifacts" / "embeddings" / run_id).resolve()

        ensure_dir(output_dir)
        ensure_dir(output_dir / "checkpoints")

        docs_jsonl = output_dir / f"{run_id}_{self.doc_jsonl_suffix}.jsonl"
        queries_jsonl = output_dir / f"{run_id}_{self.query_jsonl_suffix}.jsonl"
        docs_parquet = output_dir / f"{run_id}_{self.doc_parquet_suffix}.parquet"
        queries_parquet = output_dir / f"{run_id}_{self.query_parquet_suffix}.parquet"
        checkpoint_path = output_dir / "checkpoints" / f"{run_id}.json"
        run_manifest_path = output_dir / f"{run_id}_run_manifest.json"

        cache_docs_jsonl: Path | None = None
        cache_queries_jsonl: Path | None = None
        cache_docs_parquet: Path | None = None
        cache_queries_parquet: Path | None = None
        cache_checkpoint_path: Path | None = None
        if self.supports_cached_rows:
            cache_docs_jsonl = output_dir / f"{run_id}_{self.doc_cache_jsonl_suffix}.jsonl"
            cache_queries_jsonl = output_dir / f"{run_id}_{self.query_cache_jsonl_suffix}.jsonl"
            cache_docs_parquet = output_dir / f"{run_id}_{self.doc_cache_parquet_suffix}.parquet"
            cache_queries_parquet = output_dir / f"{run_id}_{self.query_cache_parquet_suffix}.parquet"
            cache_checkpoint_path = output_dir / "checkpoints" / f"{run_id}_cache.json"

        if args.reset:
            reset_paths = [docs_jsonl, queries_jsonl, docs_parquet, queries_parquet, checkpoint_path, run_manifest_path]
            if cache_docs_jsonl is not None and cache_queries_jsonl is not None:
                reset_paths.extend([cache_docs_jsonl, cache_queries_jsonl])
            if cache_docs_parquet is not None and cache_queries_parquet is not None:
                reset_paths.extend([cache_docs_parquet, cache_queries_parquet])
            if cache_checkpoint_path is not None:
                reset_paths.append(cache_checkpoint_path)
            for path in reset_paths:
                if path.exists():
                    path.unlink()

        manifest_rows = [row for row in read_jsonl(manifest_path) if row.get("status") == "ok"]
        query_rows = read_jsonl(queries_path)
        validate_unique_ids(manifest_rows, "doc_id")
        validate_unique_ids(query_rows, "query_id")

        if args.max_docs is not None:
            manifest_rows = manifest_rows[: max(args.max_docs, 0)]
        if args.max_queries is not None:
            query_rows = query_rows[: max(args.max_queries, 0)]

        provider = self.build_provider(args)
        final_checkpoint = CheckpointStore(checkpoint_path)
        run_manifest = self.build_run_manifest(args, run_id)
        run_manifest["mode"] = mode

        cache_doc_rows_reused = 0
        cache_query_rows_reused = 0

        if self.supports_cached_rows:
            if (
                cache_docs_jsonl is None
                or cache_queries_jsonl is None
                or cache_docs_parquet is None
                or cache_queries_parquet is None
                or cache_checkpoint_path is None
            ):
                raise RuntimeError("cached-row paths were not initialized")

            if mode in {"full", "descriptions-only"}:
                cache_checkpoint = CheckpointStore(cache_checkpoint_path)

                with JsonlAppender(cache_docs_jsonl) as cache_docs_writer:
                    for row in progress(
                        manifest_rows,
                        desc="Processing docs",
                        unit="doc",
                        total=len(manifest_rows),
                    ):
                        doc_id = str(row["doc_id"])
                        if doc_id in cache_checkpoint.processed_doc_ids:
                            continue
                        cache_docs_writer.write_row(
                            self.build_doc_cache_row(
                                row=row,
                                provider=provider,
                                workspace_root=workspace_root,
                                run_id=run_id,
                                args=args,
                            )
                        )
                        cache_checkpoint.add_doc_id(doc_id)
                        cache_checkpoint.save()

                with JsonlAppender(cache_queries_jsonl) as cache_queries_writer:
                    for row in progress(
                        query_rows,
                        desc="Processing queries",
                        unit="query",
                        total=len(query_rows),
                    ):
                        query_id = str(row["query_id"])
                        if query_id in cache_checkpoint.processed_query_ids:
                            continue
                        cache_queries_writer.write_row(
                            self.build_query_cache_row(
                                row=row,
                                provider=provider,
                                workspace_root=workspace_root,
                                run_id=run_id,
                                args=args,
                            )
                        )
                        cache_checkpoint.add_query_id(query_id)
                        cache_checkpoint.save()

                jsonl_to_parquet(cache_docs_jsonl, cache_docs_parquet)
                jsonl_to_parquet(cache_queries_jsonl, cache_queries_parquet)

            if mode in {"full", "embeddings-only"}:
                if not cache_docs_jsonl.exists():
                    raise FileNotFoundError(
                        f"missing document cache for run-id '{run_id}': {cache_docs_jsonl}"
                    )
                if not cache_queries_jsonl.exists():
                    raise FileNotFoundError(
                        f"missing query cache for run-id '{run_id}': {cache_queries_jsonl}"
                    )

                doc_cached_rows = read_jsonl(cache_docs_jsonl)
                query_cached_rows = read_jsonl(cache_queries_jsonl)

                if args.max_docs is not None:
                    doc_cached_rows = doc_cached_rows[: max(args.max_docs, 0)]
                if args.max_queries is not None:
                    query_cached_rows = query_cached_rows[: max(args.max_queries, 0)]

                if mode == "embeddings-only":
                    cache_doc_rows_reused = len(doc_cached_rows)
                    cache_query_rows_reused = len(query_cached_rows)

                with JsonlAppender(docs_jsonl) as docs_writer:
                    for row in progress(
                        doc_cached_rows,
                        desc="Embedding docs",
                        unit="doc",
                        total=len(doc_cached_rows),
                    ):
                        doc_id = str(row.get("item_id", "")).strip()
                        if not doc_id:
                            raise ValueError("missing item_id in document cache row")
                        if doc_id in final_checkpoint.processed_doc_ids:
                            continue
                        docs_writer.write_row(
                            self.build_doc_row_from_cache(
                                row=row,
                                provider=provider,
                                workspace_root=workspace_root,
                                run_id=run_id,
                                args=args,
                            )
                        )
                        final_checkpoint.add_doc_id(doc_id)
                        final_checkpoint.save()

                with JsonlAppender(queries_jsonl) as queries_writer:
                    for row in progress(
                        query_cached_rows,
                        desc="Embedding queries",
                        unit="query",
                        total=len(query_cached_rows),
                    ):
                        query_id = str(row.get("item_id", "")).strip()
                        if not query_id:
                            raise ValueError("missing item_id in query cache row")
                        if query_id in final_checkpoint.processed_query_ids:
                            continue
                        queries_writer.write_row(
                            self.build_query_row_from_cache(
                                row=row,
                                provider=provider,
                                workspace_root=workspace_root,
                                run_id=run_id,
                                args=args,
                            )
                        )
                        final_checkpoint.add_query_id(query_id)
                        final_checkpoint.save()

                jsonl_to_parquet(docs_jsonl, docs_parquet)
                jsonl_to_parquet(queries_jsonl, queries_parquet)

        else:
            with JsonlAppender(docs_jsonl) as docs_writer:
                for row in progress(
                    manifest_rows,
                    desc="Processing docs",
                    unit="doc",
                    total=len(manifest_rows),
                ):
                    doc_id = str(row["doc_id"])
                    if doc_id in final_checkpoint.processed_doc_ids:
                        continue
                    docs_writer.write_row(
                        self.build_doc_row(
                            row=row,
                            provider=provider,
                            workspace_root=workspace_root,
                            run_id=run_id,
                            args=args,
                        )
                    )
                    final_checkpoint.add_doc_id(doc_id)
                    final_checkpoint.save()

            with JsonlAppender(queries_jsonl) as queries_writer:
                for row in progress(
                    query_rows,
                    desc="Processing queries",
                    unit="query",
                    total=len(query_rows),
                ):
                    query_id = str(row["query_id"])
                    if query_id in final_checkpoint.processed_query_ids:
                        continue
                    queries_writer.write_row(
                        self.build_query_row(
                            row=row,
                            provider=provider,
                            workspace_root=workspace_root,
                            run_id=run_id,
                            args=args,
                        )
                    )
                    final_checkpoint.add_query_id(query_id)
                    final_checkpoint.save()

            jsonl_to_parquet(docs_jsonl, docs_parquet)
            jsonl_to_parquet(queries_jsonl, queries_parquet)

        run_manifest["finished_at"] = now_utc_iso()
        run_manifest["doc_rows_written"] = jsonl_row_count(docs_jsonl)
        run_manifest["query_rows_written"] = jsonl_row_count(queries_jsonl)
        outputs = self.output_paths_for_manifest(
            workspace_root=workspace_root,
            docs_jsonl=docs_jsonl,
            queries_jsonl=queries_jsonl,
            docs_parquet=docs_parquet,
            queries_parquet=queries_parquet,
            checkpoint_path=checkpoint_path,
        )
        outputs.update(
            self.extra_output_paths_for_manifest(
                workspace_root=workspace_root,
                docs_jsonl=docs_jsonl,
                queries_jsonl=queries_jsonl,
                docs_parquet=docs_parquet,
                queries_parquet=queries_parquet,
                checkpoint_path=checkpoint_path,
                cache_docs_jsonl=cache_docs_jsonl,
                cache_queries_jsonl=cache_queries_jsonl,
                cache_docs_parquet=cache_docs_parquet,
                cache_queries_parquet=cache_queries_parquet,
                cache_checkpoint_path=cache_checkpoint_path,
            )
        )
        run_manifest["outputs"] = outputs
        self.update_run_manifest(
            run_manifest=run_manifest,
            context={
                "mode": mode,
                "supports_cached_rows": self.supports_cached_rows,
                "cache_docs_jsonl": cache_docs_jsonl,
                "cache_queries_jsonl": cache_queries_jsonl,
                "cache_doc_rows_written": jsonl_row_count(cache_docs_jsonl) if cache_docs_jsonl else 0,
                "cache_query_rows_written": jsonl_row_count(cache_queries_jsonl) if cache_queries_jsonl else 0,
                "cache_doc_rows_reused": cache_doc_rows_reused,
                "cache_query_rows_reused": cache_query_rows_reused,
            },
        )
        write_json(run_manifest_path, run_manifest)
        return run_manifest
