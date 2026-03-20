from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cosine-distance HDBSCAN over document vectors and generate an HTML cluster report."
        )
    )
    parser.add_argument("--workspace-root", default=".", help="Workspace root used for artifact resolution.")
    parser.add_argument("--run-id", required=True, help="Run id used to resolve embedding and clustering paths.")
    parser.add_argument(
        "--report-name",
        default=None,
        help="Optional output base name (without extension). Defaults to <run_id>_hdbscan.",
    )
    parser.add_argument("--vector-col", default="vector", help="Column containing embedding arrays.")
    parser.add_argument("--item-id-col", default="item_id", help="Column containing item id.")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--min-samples", type=int, default=None, help="HDBSCAN min_samples.")
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon.",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include cluster -1 rows in the detailed report section.",
    )
    parser.add_argument(
        "--max-preview-chars",
        type=int,
        default=120,
        help="Max characters shown for long text preview columns.",
    )
    parser.add_argument(
        "--csv-mode",
        choices=["concise", "full"],
        default="concise",
        help="CSV verbosity: concise omits heavy/raw columns, full writes all columns.",
    )
    parser.add_argument(
        "--csv-include-preview",
        action="store_true",
        help="Include preview text column in concise CSV mode.",
    )
    return parser.parse_args()


def _resolve_path(raw: str, workspace_root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (workspace_root / p).resolve()


def _input_path_for_run(workspace_root: Path, run_id: str) -> Path:
    run_dir = (workspace_root / "artifacts" / "embeddings" / run_id).resolve()
    manifest_path = run_dir / f"{run_id}_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run manifest not found for run-id '{run_id}': {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}

    candidate_keys = [
        "doc_vectors_parquet",
        "doc_desc_vectors_parquet",
        "doc_vectors_jsonl",
        "doc_desc_vectors_jsonl",
    ]
    for key in candidate_keys:
        raw = outputs.get(key)
        if not raw:
            continue
        path = _resolve_path(str(raw), workspace_root)
        if path.exists():
            return path

    raise FileNotFoundError(
        f"no supported document vectors found for run-id '{run_id}'. "
        f"Checked keys: {', '.join(candidate_keys)}"
    )


def _read_rows(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return pd.DataFrame(records)
    raise ValueError(f"unsupported input format: {path.suffix}")


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        vec = value.astype(float)
    elif isinstance(value, (list, tuple)):
        vec = np.asarray(value, dtype=float)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty vector string")
        parsed = json.loads(text)
        vec = np.asarray(parsed, dtype=float)
    else:
        raise ValueError(f"unsupported vector type: {type(value).__name__}")

    if vec.ndim != 1 or vec.size == 0:
        raise ValueError("vector must be a non-empty 1D sequence")
    return vec


def _normalize_rows(vectors: list[np.ndarray]) -> np.ndarray:
    matrix = np.vstack(vectors).astype(float)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("at least one vector has zero norm; cannot use cosine distance")
    return matrix / norms[:, None]


def _pick_preview_column(frame: pd.DataFrame) -> str | None:
    for candidate in [
        "description_text",
        "content_ref",
        "query_text_or_path",
        "file_path",
    ]:
        if candidate in frame.columns:
            return candidate
    return None


def _truncate(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 3)]}..."


def _cluster_rows_html(frame: pd.DataFrame, preview_col: str | None, max_preview_chars: int) -> str:
    columns = ["item_id", "category", "modality", "file_path", "membership_prob", "cosine_to_centroid"]
    if preview_col and preview_col not in columns:
        columns.append(preview_col)

    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows: list[str] = []
    for _, row in frame.iterrows():
        cells: list[str] = []
        for col in columns:
            value = row.get(col, "")
            if col in {"membership_prob", "cosine_to_centroid"}:
                try:
                    text = f"{float(value):.4f}"
                except (TypeError, ValueError):
                    text = ""
                klass = "num"
            else:
                text = _truncate(value, max_preview_chars)
                klass = "txt"
            cells.append(f"<td class=\"{klass}\">{html.escape(text)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        "<table class=\"cluster-table\">"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _render_html(
    input_path: Path,
    output_path: Path,
    clustered: pd.DataFrame,
    cluster_sizes: pd.DataFrame,
    include_noise: bool,
    preview_col: str | None,
    max_preview_chars: int,
) -> None:
    total = len(clustered)
    noise_count = int((clustered["cluster_id"] == -1).sum())
    non_noise_clusters = int((cluster_sizes["cluster_id"] != -1).sum())

    summary_rows: list[str] = []
    for _, row in cluster_sizes.iterrows():
        cid = int(row["cluster_id"])
        size = int(row["size"])
        pct = (size / total * 100.0) if total else 0.0
        label = "noise" if cid == -1 else str(cid)
        summary_rows.append(
            "<tr>"
            f"<td class=\"txt\">{label}</td>"
            f"<td class=\"num\">{size}</td>"
            f"<td class=\"num\">{pct:.2f}%</td>"
            "</tr>"
        )

    detail_blocks: list[str] = []
    grouped = clustered.groupby("cluster_id", sort=True)
    for cluster_id, sub in grouped:
        cid = int(cluster_id)
        if cid == -1 and not include_noise:
            continue
        label = "noise" if cid == -1 else str(cid)
        size = len(sub)
        avg_prob = float(sub["membership_prob"].mean()) if size else float("nan")
        avg_sim = float(sub["cosine_to_centroid"].mean()) if size else float("nan")
        table_html = _cluster_rows_html(
            frame=sub.sort_values(by=["cosine_to_centroid", "membership_prob"], ascending=False),
            preview_col=preview_col,
            max_preview_chars=max_preview_chars,
        )
        detail_blocks.append(
            "<details open>"
            f"<summary>Cluster {html.escape(label)} | size={size} | avg_prob={avg_prob:.4f} | avg_sim={avg_sim:.4f}</summary>"
            f"{table_html}"
            "</details>"
        )

    generated_at = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()

    page = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Vector Cluster Report</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --panel: #fffef8;
      --text: #1f2937;
      --muted: #5b6470;
      --line: #d9d2c5;
      --accent: #0f766e;
      --accent-soft: #d6f2ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 20% 10%, #f1ede4 0%, transparent 35%),
        radial-gradient(circle at 80% 0%, #ece6d8 0%, transparent 30%),
        var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      line-height: 1.45;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .meta {{ color: var(--muted); margin-bottom: 20px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px 14px;
    }}
    .card .label {{ color: var(--muted); font-size: 13px; }}
    .card .value {{ font-size: 22px; font-weight: 700; color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); }}
    th, td {{ border: 1px solid var(--line); padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f1ece1; text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.txt {{ text-align: left; }}
    details {{
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      background: var(--panel);
    }}
    summary {{
      cursor: pointer;
      list-style: none;
      padding: 10px 12px;
      font-weight: 600;
      background: var(--accent-soft);
      border-bottom: 1px solid var(--line);
    }}
    summary::-webkit-details-marker {{ display: none; }}
    .cluster-table {{ border: none; margin: 0; }}
    .cluster-table th, .cluster-table td {{ font-size: 13px; }}
    .note {{ color: var(--muted); margin-top: 14px; font-size: 13px; }}
  </style>
</head>
<body>
  <main class=\"container\">
    <h1>Cosine HDBSCAN Cluster Report</h1>
    <div class=\"meta\">input: {html.escape(input_path.as_posix())} | generated: {html.escape(generated_at)}</div>
    <section class=\"cards\">
      <div class=\"card\"><div class=\"label\">Rows</div><div class=\"value\">{total}</div></div>
      <div class=\"card\"><div class=\"label\">Clusters (excl. noise)</div><div class=\"value\">{non_noise_clusters}</div></div>
      <div class=\"card\"><div class=\"label\">Noise points</div><div class=\"value\">{noise_count}</div></div>
      <div class=\"card\"><div class=\"label\">Preview Column</div><div class=\"value\">{html.escape(preview_col or 'none')}</div></div>
    </section>

    <h2>Cluster Summary</h2>
    <table>
      <thead><tr><th>cluster_id</th><th>size</th><th>share</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>

    <h2>Cluster Details</h2>
    {''.join(detail_blocks)}
    <p class=\"note\">membership_prob is HDBSCAN membership strength (0 to 1). cosine_to_centroid is higher when an item is closer to the cluster center.</p>
  </main>
</body>
</html>
"""

    output_path.write_text(page, encoding="utf-8")


def _compute_centroid_similarities(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    similarities = np.full(shape=(matrix.shape[0],), fill_value=np.nan, dtype=float)
    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id == -1:
            continue
        idx = np.where(labels == cluster_id)[0]
        if idx.size == 0:
            continue
        centroid = matrix[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        centroid = centroid / norm
        similarities[idx] = np.einsum("ij,j->i", matrix[idx], centroid)
    return similarities


def _select_csv_columns(
    frame: pd.DataFrame,
    preview_col: str | None,
    csv_mode: str,
    include_preview: bool,
) -> list[str]:
    if csv_mode == "full":
        return frame.columns.tolist()

    ordered = [
        "item_id",
        "cluster_id",
        "membership_prob",
        "cosine_to_centroid",
        "category",
        "modality",
        "source_tag",
        "file_path",
    ]
    if include_preview and preview_col and preview_col not in ordered:
        ordered.append(preview_col)

    return [col for col in ordered if col in frame.columns]


def run(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = Path(args.workspace_root).resolve()
    input_path = _input_path_for_run(workspace_root=workspace_root, run_id=args.run_id)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    output_dir = (workspace_root / "artifacts" / "clustering" / args.run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_name = args.report_name or f"{args.run_id}_hdbscan"
    html_path = output_dir / f"{report_name}.html"
    csv_path = output_dir / f"{report_name}.csv"
    json_path = output_dir / f"{report_name}.summary.json"

    frame = _read_rows(input_path)
    if args.vector_col not in frame.columns:
        raise ValueError(f"vector column '{args.vector_col}' not found")
    if args.item_id_col not in frame.columns:
        raise ValueError(f"item id column '{args.item_id_col}' not found")

    if frame.empty:
        raise ValueError("input contains no rows")

    vectors: list[np.ndarray] = [_parse_vector(v) for v in frame[args.vector_col].tolist()]
    matrix = _normalize_rows(vectors)

    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="cosine",
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        algorithm="generic",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(matrix)
    probabilities = clusterer.probabilities_
    centroid_sim = _compute_centroid_similarities(matrix, labels)

    clustered = frame.copy()
    clustered["item_id"] = clustered[args.item_id_col].astype(str)
    clustered["cluster_id"] = labels.astype(int)
    clustered["membership_prob"] = probabilities.astype(float)
    clustered["cosine_to_centroid"] = centroid_sim.astype(float)

    if "category" not in clustered.columns:
        clustered["category"] = ""
    if "modality" not in clustered.columns:
        clustered["modality"] = ""
    if "file_path" not in clustered.columns:
        clustered["file_path"] = ""

    cluster_sizes = (
        clustered.groupby("cluster_id", as_index=False)
        .size()
        .sort_values(by=["size", "cluster_id"], ascending=[False, True])
    )

    preview_col = _pick_preview_column(clustered)

    clustered_sorted = clustered.sort_values(
        by=["cluster_id", "cosine_to_centroid", "membership_prob", "item_id"],
        ascending=[True, False, False, True],
    )
    csv_columns = _select_csv_columns(
        clustered_sorted,
        preview_col=preview_col,
        csv_mode=args.csv_mode,
        include_preview=bool(args.csv_include_preview),
    )
    clustered_sorted.loc[:, csv_columns].to_csv(csv_path, index=False)

    _render_html(
        input_path=input_path,
        output_path=html_path,
        clustered=clustered,
        cluster_sizes=cluster_sizes,
        include_noise=args.include_noise,
        preview_col=preview_col,
        max_preview_chars=args.max_preview_chars,
    )

    summary = {
        "run_id": args.run_id,
        "input": input_path.as_posix(),
        "rows": int(len(clustered)),
        "vector_dim": int(matrix.shape[1]),
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": None if args.min_samples is None else int(args.min_samples),
        "cluster_selection_epsilon": float(args.cluster_selection_epsilon),
        "num_clusters_excluding_noise": int((cluster_sizes["cluster_id"] != -1).sum()),
        "noise_points": int((clustered["cluster_id"] == -1).sum()),
        "outputs": {
            "html_report": html_path.as_posix(),
            "csv_rows": csv_path.as_posix(),
            "summary_json": json_path.as_posix(),
        },
        "csv_mode": args.csv_mode,
        "csv_include_preview": bool(args.csv_include_preview),
        "csv_columns": csv_columns,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
