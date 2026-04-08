# Multimodal Benchmark Pipelines

This workspace contains two pipelines for generating embeddings on a multimodal document/query dataset.

## Prerequisites

- Python 3.11+
- llama.cpp server running with vision-language and/or text embedding models
- Dependencies installed:

```bash
# Option 1: venv + pip
python3 -m venv .venv
# ACTIVATE venv
pip3 install -e .

# Option 2: uv
uv sync
```

## Dataset metadata gen

[dataset_metadata.py](src\dataset_metadata.py) script generates metadata based on [data](data) dir in [artifacts/metadata](artifacts/metadata).


## Pipeline A: VL-Only Embeddings

Directly generates embeddings from images and text using a vision-language embedding model.

**Run in full mode:**

```bash
python3 -m src/pipeline_vl_embed \
    --run-id YOUR_RUN_ID \
    --workspace-root . \
    --base-url http://localhost:8080
```

**Key options:**
- `--base-url` — llama.cpp server endpoint (default: `http://localhost:8080`)
- `--max-docs N` — limit to first N documents (for testing)
- `--max-queries N` — limit to first N queries (for testing)
- `--fake-run` — use mock model outputs instead of real API calls
- `--reset` — ignore checkpoints and rebuild from scratch

**Output:** `artifacts/embeddings/YOUR_RUN_ID/`

---

## Pipeline B: Description + Text Embedding

First generates textual descriptions of images using a vision-language model, then produces text embeddings from those descriptions.

**Run in full mode:**

```bash
python3 -m src/pipeline_desc_embed \
    --run-id YOUR_RUN_ID \
    --workspace-root . \
    --mode full \
  --description-base-url http://localhost:8080 \
  --embedding-base-url http://localhost:8081
```

**Key options:**
- `--mode` — execution mode:
  - `full` — generate descriptions then embeddings (default)
  - `descriptions-only` — generate descriptions only (can be run separately from embeddings)
  - `embeddings-only` — generate embeddings from cached descriptions
- `--description-base-url` — llama.cpp endpoint used for `/v1/chat/completions` in description generation
- `--embedding-base-url` — llama.cpp endpoint used for `/v1/embeddings` in text embedding generation
- `--base-url` — fallback endpoint used when the endpoint-specific flags are omitted
- `--doc-description-prompt` — custom prompt for document image descriptions
- `--query-description-prompt` — custom prompt for query image descriptions
- `--max-docs N` — limit to first N documents
- `--max-queries N` — limit to first N queries
- `--fake-run` — use mock model outputs
- `--reset` — ignore checkpoints and rebuild

`full` mode requires different description and embedding endpoints, because a single llama.cpp instance cannot serve both models at once.

**Output:** `artifacts/embeddings/YOUR_RUN_ID/`

---

## Checkpoints

Both pipelines save progress automatically. Use `--reset` to ignore existing checkpoints and rebuild outputs from scratch.

## llama.cpp example commands

Vision Embedding
`llama-server -hf DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF:Q8_0 --port 8080 --embeddings -c 1024 --pooling last --image-min-tokens 256 -ub 1024 -np 1 --cache-ram 0`

Text Embedding
`llama-server -hf Gwriiuuu/Qwen3-Embedding-0.6B-Q8_0-GGUF:Q8_0 --port 8080 --embeddings -c 1024 --pooling last -ub 1024 -np 1 --cache-ram 0`

Description Gen
`llama-server -hf Qwen/Qwen3-VL-2B-Instruct-GGUF:Q8_0 --port 8080 -c 8192 -ub 2048 --image-min-tokens 512 -np 1 --cache-ram 0`

Remember to adjust ports when running two models at once.

## Clustering

```bash
python3 src/analyze_vector_clusters.py  --run-id YOUR_RUN_ID
```
