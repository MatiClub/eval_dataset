"""Model Arena: blinded side-by-side comparison of embedding runs.

For each query, two runs' top-k retrieval results are shown as anonymous
"Model A" / "Model B" columns (left/right order randomized). The judge votes
A / B / tie / both bad; votes are appended to artifacts/arena/votes.jsonl and
aggregated into a leaderboard (win rate + Elo), a per-category breakdown, and
a cost panel (vector dims, run duration, per-call latency from run manifests).

Queries are served most-disagreeing-first: ballots where the runs retrieve
nearly identical top-k carry no decision signal and go last.

Run with:
    streamlit run src/arena_app.py
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from interactive_results_app import (
    _detect_run_ids,
    _extract_text_preview,
    _load_manifest,
    _manifest_path_for_run,
    _normalize_rows,
    _parse_vector,
    _read_rows,
    _resolve_image_path,
    _workspace_root,
)

VOTES_PATH = Path("artifacts/arena/votes.jsonl")
ELO_K = 32.0
ELO_START = 1000.0


# ------------------------------------------------------------------ loading


@dataclass
class RunData:
    run_id: str
    doc_df: pd.DataFrame
    doc_matrix: np.ndarray
    query_df: pd.DataFrame
    query_matrix: np.ndarray
    manifest: dict[str, Any]

    @property
    def model_label(self) -> str:
        embedder = str(
            self.manifest.get("text_embedding_model")
            or self.manifest.get("embedding_model")
            or "?"
        )
        descriptor = self.manifest.get("vision_model")
        if descriptor:
            return f"{embedder} (desc: {descriptor})"
        return embedder

    @property
    def vector_dim(self) -> int:
        return int(self.doc_matrix.shape[1])

    @property
    def duration_sec(self) -> float | None:
        try:
            started = datetime.fromisoformat(str(self.manifest["started_at"]))
            finished = datetime.fromisoformat(str(self.manifest["finished_at"]))
            return (finished - started).total_seconds()
        except (KeyError, ValueError, TypeError):
            return None


@st.cache_data(show_spinner="Loading run vectors...")
def _load_run(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    workspace_root = _workspace_root()
    manifest_path = _manifest_path_for_run(workspace_root, run_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs_path, queries_path = _load_manifest(manifest_path, workspace_root)
    return _read_rows(docs_path), _read_rows(queries_path), manifest


def _build_run_data(run_id: str) -> RunData:
    doc_df, query_df, manifest = _load_run(run_id)
    doc_df = doc_df.reset_index(drop=True)
    query_df = query_df.reset_index(drop=True)
    doc_matrix = _normalize_rows([_parse_vector(v) for v in doc_df["vector"]])
    query_matrix = _normalize_rows([_parse_vector(v) for v in query_df["vector"]])
    return RunData(
        run_id=run_id,
        doc_df=doc_df,
        doc_matrix=doc_matrix,
        query_df=query_df,
        query_matrix=query_matrix,
        manifest=manifest,
    )


def _topk_doc_ids(run: RunData, query_id: str, k: int) -> list[str]:
    matches = run.query_df.index[run.query_df["item_id"] == query_id]
    if len(matches) == 0:
        return []
    sims = run.doc_matrix @ run.query_matrix[int(matches[0])]
    order = np.argsort(-sims)[:k]
    return [str(run.doc_df.at[int(i), "item_id"]) for i in order]


def _doc_row(runs: list[RunData], doc_id: str) -> pd.Series | None:
    for run in runs:
        matches = run.doc_df.index[run.doc_df["item_id"] == doc_id]
        if len(matches):
            return run.doc_df.loc[int(matches[0])]
    return None


# ------------------------------------------------------------------- votes


def _read_votes() -> pd.DataFrame:
    if not VOTES_PATH.exists():
        return pd.DataFrame(
            columns=["ts", "annotator", "query_id", "query_category", "run_a", "run_b", "winner", "k"]
        )
    rows = [json.loads(line) for line in VOTES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _append_vote(row: dict[str, Any]) -> None:
    VOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VOTES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def _elo_ratings(votes: pd.DataFrame) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for _, vote in votes.iterrows():
        winner = str(vote.get("winner", ""))
        if winner not in {"a", "b", "tie"}:
            continue
        run_a, run_b = str(vote["run_a"]), str(vote["run_b"])
        ra = ratings.setdefault(run_a, ELO_START)
        rb = ratings.setdefault(run_b, ELO_START)
        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        score_a = {"a": 1.0, "b": 0.0, "tie": 0.5}[winner]
        ratings[run_a] = ra + ELO_K * (score_a - expected_a)
        ratings[run_b] = rb + ELO_K * ((1.0 - score_a) - (1.0 - expected_a))
    return ratings


def _leaderboard(votes: pd.DataFrame) -> pd.DataFrame:
    counted = votes[votes["winner"].isin(["a", "b", "tie"])]
    stats: dict[str, dict[str, float]] = {}
    for _, vote in counted.iterrows():
        for side, run in [("a", str(vote["run_a"])), ("b", str(vote["run_b"]))]:
            entry = stats.setdefault(run, {"battles": 0, "wins": 0, "ties": 0})
            entry["battles"] += 1
            if vote["winner"] == "tie":
                entry["ties"] += 1
            elif vote["winner"] == side:
                entry["wins"] += 1
    ratings = _elo_ratings(counted)
    rows = []
    for run, entry in stats.items():
        battles = entry["battles"]
        score = entry["wins"] + 0.5 * entry["ties"]
        rows.append(
            {
                "run": run,
                "elo": round(ratings.get(run, ELO_START), 1),
                "battles": int(battles),
                "wins": int(entry["wins"]),
                "ties": int(entry["ties"]),
                "losses": int(battles - entry["wins"] - entry["ties"]),
                "win_rate": round(score / battles, 3) if battles else None,
            }
        )
    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def _category_breakdown(votes: pd.DataFrame) -> pd.DataFrame | None:
    counted = votes[votes["winner"].isin(["a", "b", "tie"])]
    if counted.empty or "query_category" not in counted.columns:
        return None
    records = []
    for _, vote in counted.iterrows():
        category = str(vote.get("query_category") or "?")
        for side, run in [("a", str(vote["run_a"])), ("b", str(vote["run_b"]))]:
            score = 0.5 if vote["winner"] == "tie" else (1.0 if vote["winner"] == side else 0.0)
            records.append({"run": run, "category": category, "score": score})
    frame = pd.DataFrame(records)
    pivot = frame.pivot_table(index="run", columns="category", values="score", aggfunc="mean")
    counts = frame.pivot_table(index="run", columns="category", values="score", aggfunc="count")
    return pivot.round(2).where(counts >= 1)


# -------------------------------------------------------------- ballot flow


def _disagreement_order(
    runs: list[RunData], query_ids: list[str], k: int
) -> list[tuple[str, float]]:
    """Return (query_id, mean pairwise Jaccard overlap), lowest overlap first."""
    scored = []
    for query_id in query_ids:
        topks = [set(_topk_doc_ids(run, query_id, k)) for run in runs]
        overlaps = [
            len(x & y) / len(x | y)
            for x, y in combinations(topks, 2)
            if x or y
        ]
        scored.append((query_id, float(np.mean(overlaps)) if overlaps else 1.0))
    scored.sort(key=lambda pair: pair[1])
    return scored


def _make_ballot(query_id: str, overlap: float, run_ids: list[str], salt: int) -> dict[str, Any]:
    rng = random.Random(f"{query_id}:{salt}:{time.time_ns()}")
    pair = rng.sample(run_ids, 2)
    return {"query_id": query_id, "overlap": overlap, "left": pair[0], "right": pair[1]}


def _render_query(row: pd.Series, workspace_root: Path) -> None:
    modality = str(row.get("modality", ""))
    if modality == "image":
        _, img_path = _resolve_image_path(row, workspace_root)
        if img_path is not None:
            st.image(str(img_path), width=260)
            return
    st.markdown(f"#### \u201c{row.get('query_text_or_path') or row.get('content_ref')}\u201d")


def _render_result_column(
    runs: list[RunData],
    doc_ids: list[str],
    workspace_root: Path,
    key_prefix: str,
) -> None:
    cols = st.columns(2)
    for rank, doc_id in enumerate(doc_ids, 1):
        row = _doc_row(runs, doc_id)
        with cols[(rank - 1) % 2]:
            if row is None:
                st.caption(f"{rank}. {doc_id} (missing)")
                continue
            st.markdown(f"**{rank}.** `{row.get('category')}`")
            if str(row.get("modality")) == "image":
                _, img_path = _resolve_image_path(row, workspace_root)
                if img_path is not None:
                    st.image(str(img_path), width=190)
                else:
                    st.caption("image unavailable")
            else:
                preview, details = _extract_text_preview(row, workspace_root)
                st.text_area(
                    "snippet",
                    value=preview,
                    height=110,
                    disabled=True,
                    help=details,
                    label_visibility="collapsed",
                    key=f"{key_prefix}_{rank}_{doc_id}",
                )


def _cast_vote(ballot: dict[str, Any], winner_side: str, extra: dict[str, Any]) -> None:
    """winner_side is relative to screen ('left'/'right'/'tie'/'both_bad'/'skip')."""
    if winner_side == "left":
        winner = "a"
    elif winner_side == "right":
        winner = "b"
    else:
        winner = winner_side
    if winner != "skip":
        _append_vote(
            {
                "ts": datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat(),
                "annotator": extra["annotator"],
                "query_id": ballot["query_id"],
                "query_category": extra["query_category"],
                "query_modality": extra["query_modality"],
                "run_a": ballot["left"],
                "run_b": ballot["right"],
                "winner": winner,
                "k": extra["k"],
                "topk_overlap": round(float(ballot["overlap"]), 3),
            }
        )
        st.session_state["last_reveal"] = (
            f"Model A was `{ballot['left']}`, Model B was `{ballot['right']}`"
        )
    st.session_state["ballot_idx"] = int(st.session_state.get("ballot_idx", 0)) + 1
    st.session_state.pop("current_ballot", None)


# --------------------------------------------------------------------- app


def main() -> None:
    st.set_page_config(page_title="Embedding Model Arena", layout="wide")
    workspace_root = _workspace_root()
    run_ids = _detect_run_ids(workspace_root)

    if len(run_ids) < 2:
        st.error("The arena needs at least two runs under artifacts/embeddings.")
        st.stop()

    with st.sidebar:
        st.markdown("### Arena setup")
        selected = st.multiselect("Runs to compare", options=run_ids, default=run_ids)
        k = st.slider("Top-k shown per side", min_value=4, max_value=20, value=8)
        annotator = st.text_input("Judge name (stored with votes)", value="anon").strip() or "anon"
        include_voted = st.checkbox("Include already-voted ballots", value=False)
        st.caption(f"Votes file: `{VOTES_PATH.as_posix()}`")

    if len(selected) < 2:
        st.warning("Select at least two runs.")
        st.stop()

    runs = [_build_run_data(run_id) for run_id in selected]

    # queries common to every selected run
    common_query_ids = set(runs[0].query_df["item_id"].astype(str))
    for run in runs[1:]:
        common_query_ids &= set(run.query_df["item_id"].astype(str))
    if not common_query_ids:
        st.error("Selected runs share no query ids; were they produced from the same metadata?")
        st.stop()

    query_meta = runs[0].query_df[runs[0].query_df["item_id"].astype(str).isin(common_query_ids)]
    query_meta = query_meta.set_index(query_meta["item_id"].astype(str))

    battle_tab, board_tab = st.tabs(["Battle", "Leaderboard"])

    with battle_tab:
        votes = _read_votes()
        ordered = _disagreement_order(runs, sorted(common_query_ids), k)

        if not include_voted and not votes.empty:
            voted_queries = set(votes[votes["annotator"] == annotator]["query_id"].astype(str))
            pending = [pair for pair in ordered if pair[0] not in voted_queries]
        else:
            pending = ordered

        done = len(ordered) - len(pending)
        st.progress(
            done / len(ordered) if ordered else 0.0,
            text=f"{done}/{len(ordered)} queries voted (judge: {annotator})",
        )

        reveal = st.session_state.pop("last_reveal", None)
        if reveal:
            st.info(f"Previous ballot: {reveal}")

        if not pending:
            st.success("All ballots voted. Check the Leaderboard tab, or enable "
                       "'Include already-voted ballots' to revisit.")
        else:
            idx = int(st.session_state.get("ballot_idx", 0)) % len(pending)
            query_id, overlap = pending[idx]

            ballot = st.session_state.get("current_ballot")
            if not ballot or ballot["query_id"] != query_id:
                ballot = _make_ballot(query_id, overlap, selected, salt=idx)
                st.session_state["current_ballot"] = ballot

            qrow = query_meta.loc[query_id]
            head_left, head_right = st.columns([3, 1])
            with head_left:
                _render_query(qrow, workspace_root)
                st.caption(
                    f"query {query_id} | category: {qrow.get('category_focus')} | "
                    f"modality: {qrow.get('modality')} | top-{k} overlap between models: {overlap:.2f}"
                )
            with head_right:
                extra = {
                    "annotator": annotator,
                    "query_category": str(qrow.get("category_focus") or ""),
                    "query_modality": str(qrow.get("modality") or ""),
                    "k": k,
                }
                vote_cols = st.columns(2)
                vote_cols[0].button(
                    "A is better", type="primary", width="stretch",
                    on_click=_cast_vote, args=(ballot, "left", extra),
                )
                vote_cols[1].button(
                    "B is better", type="primary", width="stretch",
                    on_click=_cast_vote, args=(ballot, "right", extra),
                )
                tie_cols = st.columns(3)
                tie_cols[0].button("Tie", width="stretch",
                                   on_click=_cast_vote, args=(ballot, "tie", extra))
                tie_cols[1].button("Both bad", width="stretch",
                                   on_click=_cast_vote, args=(ballot, "both_bad", extra))
                tie_cols[2].button("Skip", width="stretch",
                                   on_click=_cast_vote, args=(ballot, "skip", extra))

            st.divider()
            left_col, right_col = st.columns(2)
            run_by_id = {run.run_id: run for run in runs}
            with left_col:
                st.markdown("### Model A")
                _render_result_column(
                    runs,
                    _topk_doc_ids(run_by_id[ballot["left"]], query_id, k),
                    workspace_root,
                    key_prefix=f"A_{query_id}",
                )
            with right_col:
                st.markdown("### Model B")
                _render_result_column(
                    runs,
                    _topk_doc_ids(run_by_id[ballot["right"]], query_id, k),
                    workspace_root,
                    key_prefix=f"B_{query_id}",
                )

    with board_tab:
        votes = _read_votes()
        counted = votes[votes["winner"].isin(["a", "b", "tie"])] if not votes.empty else votes
        if counted.empty:
            st.info("No votes yet. Cast some in the Battle tab.")
        else:
            st.subheader("Leaderboard")
            st.caption(
                f"{len(counted)} counted votes "
                f"({int((votes['winner'] == 'both_bad').sum())} 'both bad' excluded). "
                "win_rate counts a tie as half a win."
            )
            st.dataframe(_leaderboard(counted), hide_index=True, width="stretch")

            breakdown = _category_breakdown(counted)
            if breakdown is not None:
                st.subheader("Win rate by query category")
                st.caption("Where each model is strong or weak. 1.0 = wins every battle in that category.")
                try:
                    import matplotlib  # noqa: F401  (Styler.background_gradient needs it)

                    styled: Any = breakdown.style.background_gradient(
                        cmap="RdYlGn", vmin=0.0, vmax=1.0
                    )
                except ImportError:
                    styled = breakdown  # matplotlib not installed; plain table
                st.dataframe(styled, width="stretch")

        st.subheader("Cost")
        st.caption("Quality is only half the decision; latency and vector size drive serving cost.")
        cost_rows = []
        for run in runs:
            stats = run.manifest.get("cost_stats") or {}
            cost_rows.append(
                {
                    "run": run.run_id,
                    "models": run.model_label,
                    "vector_dim": run.vector_dim,
                    "run_duration_sec": round(run.duration_sec, 1) if run.duration_sec else None,
                    "embed_text_avg_sec": stats.get("embed_text_avg_sec"),
                    "embed_image_avg_sec": stats.get("embed_image_avg_sec"),
                    "describe_image_avg_sec": stats.get("describe_image_avg_sec"),
                }
            )
        st.dataframe(pd.DataFrame(cost_rows), hide_index=True, width="stretch")
        st.caption(
            "Per-call latencies come from cost_stats in the run manifest and are "
            "recorded automatically by the pipelines; older runs show only totals."
        )


if __name__ == "__main__":
    main()
