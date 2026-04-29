# ChatPLVis

Interactive proteome visualisation and AI chatbot for *Mycobacterium tuberculosis* and related species. Select proteins on a 2D/3D scatter plot and ask the built-in RAG-powered assistant anything about them.

---

## Features

| Feature | Details |
|---|---|
| WebGL scatter plot | `scattergl` renders 50 k+ protein nodes smoothly |
| Multi-selection | Box and lasso selection tools; click to toggle individual nodes |
| AI chatbot | Hybrid FAISS + BM25 retrieval, streamed token-by-token via SSE |
| Two LLM backends | OpenAI (cloud) or any locally-running Ollama model |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` with Apple MPS acceleration |
| Responsive UI | Apple-HIG design; resizable sidebar; works on mobile and tablet |
| Dark / light mode | System-preference aware, toggle in toolbar |

---

## Quick start

### 1 — Clone and create environment

```bash
git clone <repo-url>
cd chatplvis_prod
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure

```bash
cp .env.example .env
```

Edit `.env` and choose one LLM backend:

**Option A — OpenAI (cloud)**
```env
OPENAI_API_KEY=sk-...
```

**Option B — Ollama (free, local)**
1. Install Ollama: <https://ollama.com>
2. Pull any model: `ollama pull llama3.2` (or whichever you prefer)
3. Leave `OPENAI_API_KEY` unset — ChatPLVis auto-detects Ollama and picks the first available model.

### 3 — Add data files

Place these files in the repository (they are excluded from git due to size):

| File | Default path | Description |
|---|---|---|
| Proteome CSV | `uploads/mycobacterium_proteome_df.csv` | Main dataset with UMAP coordinates, gene names, organisms |
| Counts Excel | `data/counts_all_stages_MAGECK_with_ES.xlsx` | Experimental screen hit counts (optional) |

Override paths with `CSV_PATH` and `COUNTS_PATH` in `.env`.

### 4 — Run

```bash
python run.py
```

Open <http://localhost:5000>. The embedding model and FAISS index are built in the background on first launch (this takes ~30 s; subsequent launches use the cached index).

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Enables OpenAI backend (uses `gpt-4o-mini`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Preferred Ollama model (falls back to first available) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace sentence encoder |
| `CSV_PATH` | `uploads/mycobacterium_proteome_df.csv` | Proteome data file |
| `COUNTS_PATH` | `data/counts_all_stages_MAGECK_with_ES.xlsx` | Counts data file |
| `PORT` | `5000` | HTTP port |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

---

## Project structure

```
chatplvis_prod/
├── run.py                   # Entry point
├── requirements.txt
├── .env.example
│
├── app/
│   ├── __init__.py          # Flask app factory + service wiring
│   ├── api/
│   │   ├── routes.py        # GET / POST / and /chatbot/stream endpoints
│   │   └── schemas.py       # Pydantic-like dataclasses (PlotState, ChatRequest, ProteinNode)
│   ├── core/
│   │   ├── config.py        # Settings dataclass (all env-var defaults)
│   │   └── exceptions.py    # DataError, EmbeddingError, LLMError, RateLimitedError
│   ├── services/
│   │   ├── data_service.py      # CSV loading, LRU cache, plot DataFrame building
│   │   ├── embedding_service.py # Sentence encoder, FAISS index, hybrid BM25 search
│   │   ├── llm_service.py       # OpenAI / Ollama backend detection, streaming
│   │   └── rag_service.py       # Retrieval-augmented generation pipeline
│   └── utils/
│       ├── cache.py         # TTL LRU response cache
│       └── text.py          # Organism name cleaning helpers
│
├── templates/
│   └── index.html           # Single-page UI (Jinja2 + Plotly.js + vanilla JS)
│
├── static/
│   └── css/style.css
│
├── data/
│   └── counts_all_stages_MAGECK_with_ES.xlsx
│
└── uploads/
    └── mycobacterium_proteome_df.csv
```

---

## How the RAG pipeline works

1. **Selection** — User clicks or box/lasso-selects proteins on the plot.
2. **Retrieval query** — Selected proteins' annotation text is combined with the user's question to form a rich retrieval query.
3. **Hybrid search** — FAISS (semantic) and BM25 (keyword) search the full proteome corpus; results are merged with Reciprocal Rank Fusion (RRF). Selected proteins are excluded from the retrieved set to avoid duplication.
4. **Relevance ranking** — Selected proteins are re-ranked by cosine similarity to the retrieval query; top 20 are kept.
5. **Context assembly** — Selected proteins (ranked) + retrieved similar proteins are formatted with gene names, organisms, counts, and annotations, capped at 16 000 chars.
6. **LLM generation** — A system prompt + context + user question are sent to the LLM. The response streams back token-by-token via SSE and is rendered as Markdown in the chat panel.

---

## macOS / Apple Silicon notes

To prevent segfaults caused by OpenMP + PyTorch forking on macOS, `run.py` sets these env vars before any import:

```python
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

The embedding model automatically uses **Apple MPS** (GPU) when available, falling back to CPU.

---

## Using the UI

### Selecting proteins
- **Pan mode** (default) — click a node to toggle selection.
- **Box / Lasso** — use the segmented control in the toolbar to switch; drag to select a region.
- Selected proteins appear highlighted; a floating bar shows the count with **Chat** and **Clear** buttons.

### Chatting
1. Select one or more proteins.
2. Switch to the **Chat** tab or click the **Chat** button in the floating bar.
3. Type a question and press Enter or click Send.
4. The answer streams in real time. Toggle **Include similar proteins** to broaden the RAG context.

### Plot settings
All settings are in the **Settings** tab and submitted via the form:

| Setting | Options |
|---|---|
| Column | Counts_1st / 2nd / 3rd, Total_Counts, Rank (raw or normalised) |
| Comparison | All proteomes, Mycobacterium tuberculosis, vs smegmatis, vs marinum, … |
| Plot Type | 2D UMAP Based, 3D PCA Based |
| Info Source | Function [CC] (gene ontology annotations), Abstracts (PubMed-style) |
