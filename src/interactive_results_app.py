from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from html import escape

import numpy as np
import pandas as pd
import streamlit as st


EMBEDDINGS_DIR = Path("artifacts/embeddings")
CLUSTERING_DIR = Path("artifacts/clustering")


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_workspace(raw: str, workspace_root: Path) -> Path | None:
    if not raw.strip():
        return None
    p = Path(raw.strip())
    if p.is_absolute():
        return p
    return (workspace_root / p).resolve()


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return path.as_posix()


def _detect_run_ids(workspace_root: Path) -> list[str]:
    root = (workspace_root / EMBEDDINGS_DIR).resolve()
    if not root.exists():
        return []

    run_ids: list[str] = []
    for child in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        run_id = child.name
        manifest_path = child / f"{run_id}_run_manifest.json"
        if manifest_path.exists():
            run_ids.append(run_id)
    return run_ids


def _manifest_path_for_run(workspace_root: Path, run_id: str) -> Path:
    return (workspace_root / EMBEDDINGS_DIR / run_id / f"{run_id}_run_manifest.json").resolve()


def _cluster_csv_for_run(workspace_root: Path, run_id: str) -> Path | None:
    cluster_dir = (workspace_root / CLUSTERING_DIR / run_id).resolve()
    if not cluster_dir.exists():
        return None

    preferred = sorted(cluster_dir.glob("*_hdbscan.csv"))
    if preferred:
        return preferred[0]

    fallback = sorted(cluster_dir.glob("*.csv"))
    if fallback:
        return fallback[0]
    return None


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
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        vec = value.astype(float)
    elif isinstance(value, (list, tuple)):
        vec = np.asarray(value, dtype=float)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty vector string")
        vec = np.asarray(json.loads(text), dtype=float)
    else:
        raise ValueError(f"Unsupported vector type: {type(value).__name__}")

    if vec.ndim != 1 or vec.size == 0:
        raise ValueError("Vector must be a non-empty 1D array")
    return vec


def _normalize_rows(vectors: list[np.ndarray]) -> np.ndarray:
    matrix = np.vstack(vectors).astype(float)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("At least one vector has zero norm")
    return matrix / norms[:, None]


def _load_manifest(manifest_path: Path, workspace_root: Path) -> tuple[Path, Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs", {}) if isinstance(payload.get("outputs"), dict) else {}

    docs = outputs.get("doc_vectors_parquet") or outputs.get("doc_vectors_jsonl")
    queries = outputs.get("query_vectors_parquet") or outputs.get("query_vectors_jsonl")

    if not docs or not queries:
        raise ValueError(
            "Run manifest is missing required vector outputs. "
            "Expected outputs.doc_vectors_parquet/jsonl and outputs.query_vectors_parquet/jsonl."
        )

    docs_path = _resolve_from_workspace(str(docs), workspace_root)
    queries_path = _resolve_from_workspace(str(queries), workspace_root)
    if docs_path is None or queries_path is None:
        raise ValueError("Failed to resolve docs/queries paths from run manifest")

    if not docs_path.exists() or not queries_path.exists():
        raise FileNotFoundError(
            f"Resolved vectors not found. docs={docs_path}, queries={queries_path}"
        )
    return docs_path, queries_path


def _build_display_text(row: pd.Series) -> str:
    item_id = str(row.get("item_id", ""))
    item_type = str(row.get("item_type", ""))
    modality = str(row.get("modality", ""))
    focus = str(row.get("category_focus") or row.get("category") or "")
    if item_type == "query":
        text = str(row.get("query_text_or_path") or row.get("content_ref") or "")
        return f"{item_id} | query | {modality} | {focus} | {text[:80]}"
    path_or_ref = str(row.get("file_path") or row.get("content_ref") or "")
    return f"{item_id} | doc | {modality} | {focus} | {path_or_ref}"


def _prepare_items(doc_df: pd.DataFrame, query_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    for name, frame in [("docs", doc_df), ("queries", query_df)]:
        if "vector" not in frame.columns:
            raise ValueError(f"{name} vectors input is missing 'vector' column")
        if "item_id" not in frame.columns:
            raise ValueError(f"{name} vectors input is missing 'item_id' column")

    doc_df = doc_df.copy()
    query_df = query_df.copy()
    doc_df["item_type"] = "document"
    query_df["item_type"] = "query"

    if "category_focus" not in doc_df.columns:
        doc_df["category_focus"] = doc_df.get("category", "")
    if "category" not in query_df.columns:
        query_df["category"] = query_df.get("category_focus", "")

    combined = pd.concat([doc_df, query_df], ignore_index=True)
    vectors = [_parse_vector(v) for v in combined["vector"].tolist()]
    norm = _normalize_rows(vectors)
    combined = combined.reset_index(drop=True)
    combined["display_text"] = combined.apply(_build_display_text, axis=1)
    return combined, norm


def _row_preview(
    row: pd.Series,
    workspace_root: Path,
    show_meta: bool = True,
    image_width: int = 180,
) -> None:
    item_type = str(row.get("item_type", ""))
    modality = str(row.get("modality", ""))

    if show_meta:
        st.caption(
            " | ".join(
                [
                    f"id={row.get('item_id')}",
                    f"type={item_type}",
                    f"modality={modality}",
                    f"category={row.get('category') or row.get('category_focus')}",
                ]
            )
        )

    if modality == "image":
        rel = str(row.get("file_path") or row.get("query_text_or_path") or row.get("content_ref") or "")
        img_path = (workspace_root / rel).resolve()
        if img_path.exists():
            st.image(str(img_path), width=image_width)
        else:
            st.warning(f"Image file not found: {img_path}")
    elif modality == "text":
        preview, details = _extract_text_preview(row, workspace_root)
        if show_meta:
            st.text_area("Document text preview", value=preview, height=120)
            with st.expander("Show more text", expanded=False):
                st.write(details)
        else:
            st.write(details)
    elif item_type == "query":
        text = str(row.get("query_text_or_path") or row.get("content_ref") or "")
        if show_meta:
            st.text_area("Query text", value=text, height=100)
        else:
            st.write(text)
    else:
        text = str(row.get("content_ref") or row.get("file_path") or "")
        if show_meta:
            st.text_area("Document content ref", value=text, height=100)
        else:
            st.write(text)


def _resolve_image_path(row: pd.Series, workspace_root: Path) -> tuple[str, Path] | tuple[None, None]:
    rel = str(
        row.get("file_path")
        or row.get("query_text_or_path")
        or row.get("content_ref")
        or ""
    ).strip()
    if not rel:
        return None, None

    candidate = Path(rel)
    if candidate.is_absolute():
        img_path = candidate
    else:
        img_path = (workspace_root / candidate).resolve()
    if not img_path.exists():
        return None, None
    return rel, img_path


def _truncate_text(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _extract_text_preview(row: pd.Series, workspace_root: Path) -> tuple[str, str]:
    content_ref = str(row.get("content_ref") or "")
    file_path_raw = str(row.get("file_path") or "").strip()

    full_text = ""
    if file_path_raw:
        candidate = Path(file_path_raw)
        file_path = candidate if candidate.is_absolute() else (workspace_root / candidate).resolve()
        if file_path.exists() and file_path.is_file() and file_path.suffix.lower() in {".txt", ".md"}:
            try:
                full_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                full_text = ""

    if not full_text:
        full_text = content_ref

    preview = _truncate_text(full_text, 240)
    details = _truncate_text(full_text, 1400)
    return preview, details


def _render_doc_gallery(
    df: pd.DataFrame,
    workspace_root: Path,
    similarity_col: str | None,
    max_cols: int = 6,
    key_prefix: str = "gallery",
) -> None:
    if df.empty:
        return

    cols = st.columns(max_cols)
    shown = 0

    for _, row in df.iterrows():
        cosine_text = "N/A"
        if similarity_col:
            cosine_raw = row.get(similarity_col)
            if pd.notna(cosine_raw):
                try:
                    cosine_text = f"{float(cosine_raw):.4f}"
                except (TypeError, ValueError):
                    cosine_text = str(cosine_raw)
        category_text = str(row.get("category") or row.get("category_focus") or "")
        modality = str(row.get("modality") or "")
        item_id = str(row.get("item_id") or "")

        col = cols[shown % max_cols]
        with col:
            label_text = escape(f"{category_text}  {cosine_text}")
            #st.caption(f"{category_text}  {cosine_text}")
            st.markdown(
                (
                    "<div style='font-size:0.95rem; line-height:1.2; "
                    "font-weight:500; margin-bottom:0.15rem;'>"
                    f"{label_text}</div>"
                ),
                unsafe_allow_html=True,
            )
            rel, img_path = _resolve_image_path(row, workspace_root)
            if modality == "image" and img_path is not None:
                st.image(str(img_path), width=180)
            else:
                preview, details = _extract_text_preview(row, workspace_root)
                st.text_area(
                    "Text snippet",
                    value=preview,
                    height=140,
                    disabled=True,
                    help=details,
                    key=f"{key_prefix}_text_{item_id}_{shown}",
                )
        shown += 1

    if shown == 0:
        st.caption("No previewable items found for this cluster.")


def _compute_topk(
    items: pd.DataFrame,
    normalized: np.ndarray,
    anchor_idx: int,
    k: int,
    scope: str,
    include_self: bool,
) -> pd.DataFrame:
    sims = normalized @ normalized[anchor_idx]
    out = items.copy()
    out["cosine_similarity"] = sims

    if not include_self:
        out = out[out.index != anchor_idx]

    if scope == "documents":
        out = out[out["item_type"] == "document"]
    elif scope == "queries":
        out = out[out["item_type"] == "query"]

    out = out.sort_values("cosine_similarity", ascending=False)
    return out.head(k)


def _load_clusters(cluster_csv_path: Path | None) -> pd.DataFrame | None:
    if cluster_csv_path is None:
        return None
    if not cluster_csv_path.exists():
        return None
    try:
        return pd.read_csv(cluster_csv_path)
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Interactive Multimodal Results Analyzer", layout="wide")

    workspace_root = _workspace_root()

    if "artifacts_loaded" not in st.session_state:
        st.session_state["artifacts_loaded"] = False
    if "selected_run_id" not in st.session_state:
        st.session_state["selected_run_id"] = ""

    detected_run_ids = _detect_run_ids(workspace_root)

    if not st.session_state["artifacts_loaded"]:
        st.markdown("### Load Data Artifacts")
        with st.form("artifact_loader_inline", width=400):
            if detected_run_ids:
                selected_run_id = st.selectbox(
                    "Detected run ids",
                    options=detected_run_ids,
                    index=0,
                )
            else:
                selected_run_id = ""
                st.warning("No runs found under artifacts/embeddings.")

            load_button = st.form_submit_button("Load Artifacts", type="primary")

        if load_button and selected_run_id:
            st.session_state["selected_run_id"] = selected_run_id
            st.session_state["artifacts_loaded"] = True
            st.rerun()
        st.stop()


    try:
        run_id = str(st.session_state["selected_run_id"])
        if not run_id:
            raise ValueError("No run-id selected")

        manifest_path = _manifest_path_for_run(workspace_root, run_id)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found for run-id '{run_id}': {manifest_path}")

        docs_path, queries_path = _load_manifest(manifest_path, workspace_root)

        if docs_path is None or queries_path is None:
            raise ValueError("Both docs and queries vector paths are required")
        if not docs_path.exists():
            raise FileNotFoundError(f"Docs vectors path not found: {docs_path}")
        if not queries_path.exists():
            raise FileNotFoundError(f"Query vectors path not found: {queries_path}")

        doc_df = _read_rows(docs_path)
        query_df = _read_rows(queries_path)

        items, normalized = _prepare_items(doc_df, query_df)
        cluster_df = _load_clusters(_cluster_csv_for_run(workspace_root, run_id))

    except Exception as exc:
        st.error(f"Failed to load artifacts: {exc}")
        return

    header_left, header_right = st.columns([4, 3])
    with header_left:
        st.markdown("### Interactive Multimodal Results Analyzer")
        st.caption(
            " | ".join(
                [
                    f"run-id: {run_id}",
                    f"manifest: {_display_path(manifest_path, workspace_root)}",
                    f"docs: {_display_path(docs_path, workspace_root)}",
                    f"queries: {_display_path(queries_path, workspace_root)}",
                ]
            )
        )
    with header_right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Documents", int((items["item_type"] == "document").sum()))
        m2.metric("Queries", int((items["item_type"] == "query").sum()))
        category_count = int(
            items["category_focus"]
            .fillna(items.get("category", ""))
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .nunique()
        )
        m3.metric("Categories", category_count)
        with st.expander("Change data source", expanded=False, width=250):
            with st.form("artifact_loader_change", border=False):
                if detected_run_ids:
                    current = st.session_state["selected_run_id"]
                    default_idx = detected_run_ids.index(current) if current in detected_run_ids else 0
                    selected_run_id = st.selectbox(
                        "Detected run ids",
                        options=detected_run_ids,
                        index=default_idx,
                        key="run_id_change_select",
                    )
                    reload_button = st.form_submit_button("Reload Artifacts")
                    if reload_button:
                        st.session_state["selected_run_id"] = selected_run_id
                        st.session_state["artifacts_loaded"] = True
                        st.rerun()
                else:
                    st.warning("No runs found under artifacts/embeddings.")

    tabs = st.tabs(["Neighbor Search", "Pairwise Cosine", "Cluster Browser"])

    with tabs[0]:
        st.subheader("Top-k Similar Items")

        filter_col_a, filter_col_b = st.columns(2)
        with filter_col_a:
            item_type_filter = st.multiselect(
                "Filter source items by type",
                options=sorted(items["item_type"].dropna().unique().tolist()),
                default=sorted(items["item_type"].dropna().unique().tolist()),
            )
        with filter_col_b:
            modality_filter = st.multiselect(
                "Filter source items by modality",
                options=sorted(items["modality"].dropna().unique().tolist()),
                default=sorted(items["modality"].dropna().unique().tolist()),
            )
        focus_filter = st.multiselect(
            "Filter source items by category focus",
            options=sorted(items["category_focus"].fillna("").unique().tolist()),
            default=sorted(items["category_focus"].fillna("").unique().tolist()),
        )

        pool = items[
            items["item_type"].isin(item_type_filter)
            & items["modality"].isin(modality_filter)
            & items["category_focus"].fillna("").isin(focus_filter)
        ]

        if pool.empty:
            st.warning("No source items match the current filters.")
        else:
            display_options = pool["display_text"].tolist()
            source_col_a, source_col_b = st.columns([2, 3], vertical_alignment="center")
            with source_col_a:
                selected = st.selectbox("Choose source item", options=display_options)
            anchor_idx = int(pool[pool["display_text"] == selected].index[0])
            with source_col_b:
                st.markdown("Source preview")
                _row_preview(items.loc[anchor_idx], workspace_root, show_meta=False, image_width=180)

            col_a, col_b = st.columns(2)
            k = col_a.slider("Top-k", min_value=1, max_value=50, value=10)
            scope = col_b.selectbox("Search scope", options=["all", "documents", "queries"], index=1)

            results = _compute_topk(
                items=items,
                normalized=normalized,
                anchor_idx=anchor_idx,
                k=k,
                scope=scope,
                include_self=False,
            )

            st.markdown("Nearest neighbors")
            _render_doc_gallery(
                results,
                workspace_root,
                similarity_col="cosine_similarity",
                key_prefix="neighbor",
            )

    with tabs[1]:
        st.subheader("Pairwise Cosine Similarity")
        options = items["display_text"].tolist()

        col_left, col_right = st.columns(2)
        a_text = col_left.selectbox("Item A", options=options, index=0)
        b_text = col_right.selectbox("Item B", options=options, index=min(1, len(options) - 1))

        a_idx = int(items[items["display_text"] == a_text].index[0])
        b_idx = int(items[items["display_text"] == b_text].index[0])

        cosine = float(np.dot(normalized[a_idx], normalized[b_idx]))
        with st.container(horizontal_alignment="center"):
            st.metric("Cosine similarity", f"{cosine:.6f}", width="content")

        pa, pb = st.columns(2)
        with pa:
            st.markdown("Item A")
            _row_preview(items.loc[a_idx], workspace_root)
        with pb:
            st.markdown("Item B")
            _row_preview(items.loc[b_idx], workspace_root)

    with tabs[2]:
        st.subheader("Cluster Browser (Documents)")
        if cluster_df is None:
            st.info("No cluster CSV loaded. Provide one in the sidebar to enable this tab.")
        else:
            required_cols = {"item_id", "cluster_id"}
            if not required_cols.issubset(set(cluster_df.columns)):
                st.warning("Cluster CSV is missing required columns: item_id, cluster_id")
            else:
                cluster_df = cluster_df.copy()
                cluster_df["cluster_id"] = cluster_df["cluster_id"].astype(int)

                summary = (
                    cluster_df.groupby("cluster_id", as_index=False)
                    .size()
                    .rename(columns={"size": "doc_count"})
                )
                # Keep HDBSCAN noise cluster (-1) at the end for easier browsing.
                summary = summary.assign(
                    _noise_last=summary["cluster_id"].eq(-1).astype(int)
                ).sort_values(["_noise_last", "cluster_id"], ascending=[True, True]).drop(
                    columns=["_noise_last"]
                )
                st.markdown("Cluster summary")
                st.dataframe(summary, width='stretch', hide_index=True)

                cluster_ids = summary["cluster_id"].tolist()
                for idx, cid in enumerate(cluster_ids):
                    if idx > 0:
                        st.divider()
                    rows = cluster_df[cluster_df["cluster_id"] == int(cid)]
                    merged = rows.merge(items, on="item_id", how="left", suffixes=("_cluster", ""))
                    merged = merged.sort_values(
                        by=[
                            "cosine_to_centroid" if "cosine_to_centroid" in merged.columns else "item_id"
                        ],
                        ascending=False,
                    )

                    st.markdown(f"Cluster {int(cid)} items")
                    _render_doc_gallery(
                        merged,
                        workspace_root,
                        similarity_col="cosine_to_centroid",
                        key_prefix=f"cluster_{int(cid)}",
                    )


if __name__ == "__main__":
    main()
