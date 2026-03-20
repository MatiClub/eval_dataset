from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from image_utils import image_to_data_uri_for_model, prepare_image_bytes_for_model

MULTIMODAL_MARKER = "<__media__>"


def now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(prefix: str) -> str:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = hashlib.sha1(str(time.time_ns()).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{stamp}_{token}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class JsonlAppender:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> "JsonlAppender":
        ensure_dir(self.path.parent)
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write_row(self, row: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlAppender is not open")
        self._handle.write(json.dumps(row, ensure_ascii=False))
        self._handle.write("\n")


def jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                count += 1
    return count


def normalize_l2(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0.0:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]


def truncate_text(value: str, max_chars: int = 4000) -> str:
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


def load_text_from_file(path: Path, fallback_label: str) -> str:
    try:
        return truncate_text(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return fallback_label


def pooled_embedding(embedding: list[Any]) -> list[float]:
    if not embedding:
        raise ValueError("received empty embedding")
    if isinstance(embedding[0], list):
        width = len(embedding[0])
        acc = [0.0] * width
        for token_vec in embedding:
            if len(token_vec) != width:
                raise ValueError("inconsistent token embedding width")
            for idx, value in enumerate(token_vec):
                acc[idx] += float(value)
        return [value / len(embedding) for value in acc]
    return [float(v) for v in embedding]


def response_data_to_vectors(data: dict[str, Any] | list[Any]) -> list[list[float]]:
    if isinstance(data, dict) and "data" in data:
        items = sorted(data["data"], key=lambda x: int(x.get("index", 0)))
    elif isinstance(data, list):
        items = sorted(data, key=lambda x: int(x.get("index", 0)))
    else:
        raise ValueError(f"unexpected embedding response shape: {type(data).__name__}")

    vectors: list[list[float]] = []
    for item in items:
        emb = item.get("embedding")
        if emb is None:
            raise ValueError("missing embedding field")
        vectors.append(normalize_l2(pooled_embedding(emb)))
    return vectors


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_sec: float = 1.0


class LlamaHttpClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: float,
        retry_policy: RetryPolicy,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_policy = retry_policy

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | list[Any]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            request = urllib.request.Request(
                url=f"{self.base_url}{path}",
                data=body,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code} from {path}: {raw}")
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"failed to reach {self.base_url}{path}: {exc.reason}")

            if attempt < self.retry_policy.max_attempts:
                sleep_for = self.retry_policy.base_delay_sec * attempt
                time.sleep(sleep_for)

        if last_error is None:
            raise RuntimeError("request failed for unknown reason")
        raise last_error


class RealModelProvider:
    def __init__(
        self,
        base_url: str,
        embedding_model: str,
        api_key: str | None,
        timeout: float,
        retry_policy: RetryPolicy,
    ) -> None:
        self.embedding_model = embedding_model
        self.http = LlamaHttpClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            retry_policy=retry_policy,
        )

    def embed_text(self, text: str) -> list[float]:
        payload = {
            "model": self.embedding_model,
            "input": [text],
            "encoding_format": "float",
        }
        response = self.http.post_json("/v1/embeddings", payload)
        vectors = response_data_to_vectors(response)
        if len(vectors) != 1:
            raise ValueError("expected exactly one embedding for text input")
        return vectors[0]

    def embed_image(self, image_path: Path, prompt_prefix: str) -> list[float]:
        binary = prepare_image_bytes_for_model(image_path=image_path, model=self.embedding_model)
        b64 = base64.b64encode(binary).decode("ascii")
        payload = {
            "model": self.embedding_model,
            "input": [
                {
                    "prompt_string": f"{prompt_prefix} {MULTIMODAL_MARKER}",
                    "multimodal_data": [b64],
                }
            ],
            "encoding_format": "float",
        }
        response = self.http.post_json("/v1/embeddings", payload)
        vectors = response_data_to_vectors(response)
        if len(vectors) != 1:
            raise ValueError("expected exactly one embedding for image input")
        return vectors[0]

    def describe_image(self, image_path: Path, vision_model: str, prompt_text: str) -> str:
        data_uri = image_to_data_uri_for_model(image_path=image_path, model=vision_model)
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 180,
        }
        response = self.http.post_json("/v1/chat/completions", payload)
        if not isinstance(response, dict):
            raise ValueError("unexpected chat response shape")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing choices in chat response")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            return truncate_text(content, max_chars=1200)
        raise ValueError("missing text content in chat response")


class FakeModelProvider:
    def __init__(self, vector_dim: int, seed: int) -> None:
        if vector_dim <= 0:
            raise ValueError("vector_dim must be > 0")
        self.vector_dim = vector_dim
        self.seed = seed

    def _rng(self, key: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _make_vector(self, key: str) -> list[float]:
        rng = self._rng(key)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.vector_dim)]
        return normalize_l2(vec)

    def _make_words(self, key: str, min_len: int = 18, max_len: int = 34) -> str:
        lexicon = [
            "document",
            "contains",
            "photo",
            "layout",
            "scene",
            "object",
            "text",
            "summary",
            "invoice",
            "receipt",
            "syllabus",
            "medical",
            "identity",
            "warranty",
            "animal",
            "vehicle",
            "certificate",
            "recipe",
            "details",
            "context",
            "classification",
            "retrieval",
            "signal",
            "semantic",
            "attributes",
            "visible",
            "structured",
            "sample",
            "tokens",
            "evidence",
        ]
        rng = self._rng(f"desc:{key}")
        size = rng.randint(min_len, max_len)
        words = [lexicon[rng.randrange(0, len(lexicon))] for _ in range(size)]
        words[0] = words[0].capitalize()
        return " ".join(words) + "."

    def embed_text(self, text: str) -> list[float]:
        return self._make_vector(f"text:{text}")

    def embed_image(self, image_path: Path, prompt_prefix: str) -> list[float]:
        key = f"image:{image_path.as_posix()}:{prompt_prefix}"
        return self._make_vector(key)

    def describe_image(self, image_path: Path, vision_model: str, prompt_text: str) -> str:
        key = f"{vision_model}:{image_path.as_posix()}:{prompt_text}"
        return self._make_words(key)


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._state = {
            "processed_doc_ids": [],
            "processed_query_ids": [],
        }
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self._state["processed_doc_ids"] = list(data.get("processed_doc_ids", []))
                self._state["processed_query_ids"] = list(data.get("processed_query_ids", []))

    @property
    def processed_doc_ids(self) -> set[str]:
        return set(self._state["processed_doc_ids"])

    @property
    def processed_query_ids(self) -> set[str]:
        return set(self._state["processed_query_ids"])

    def add_doc_id(self, doc_id: str) -> None:
        if doc_id not in self._state["processed_doc_ids"]:
            self._state["processed_doc_ids"].append(doc_id)

    def add_query_id(self, query_id: str) -> None:
        if query_id not in self._state["processed_query_ids"]:
            self._state["processed_query_ids"].append(query_id)

    def save(self) -> None:
        write_json(self.path, self._state)


def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    try:
        pd = __import__("pandas")
    except Exception as exc:
        raise RuntimeError("pandas is required to build parquet outputs") from exc

    rows = read_jsonl(jsonl_path)
    frame = pd.DataFrame(rows)
    ensure_dir(parquet_path.parent)
    frame.to_parquet(parquet_path, index=False)


def validate_unique_ids(rows: Iterable[dict[str, Any]], field_name: str) -> None:
    seen: set[str] = set()
    for row in rows:
        value = str(row.get(field_name, "")).strip()
        if not value:
            raise ValueError(f"missing required id field: {field_name}")
        if value in seen:
            raise ValueError(f"duplicate {field_name}: {value}")
        seen.add(value)
