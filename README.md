# 🇩🇪↔🇬🇧 German-English RAG Tutor

A **Speech-Driven Retrieval-Augmented Generation (RAG)** application for German and English language learning.

Speak a sentence in German or English → the system transcribes it via Whisper, retrieves grounded factual context from Wikipedia via Qdrant, and generates a structured language lesson — complete with translation, grammar breakdown, and vocabulary analysis — streamed token-by-token from a local LLM.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
5. [Services Deep Dive](#services-deep-dive)
6. [Configuration](#configuration)
7. [Setup & Running](#setup--running)
8. [API Endpoints](#api-endpoints)
9. [Streamlit UI](#streamlit-ui)
10. [Data Models](#data-models)
11. [Performance & Expectations](#performance--expectations)

---

## Overview

| Component | Technology |
|---|---|
| **Web Framework** | FastAPI |
| **Speech-to-Text** | `faster-whisper` (`base` model, CPU int8) |
| **Embedding Model** | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual) |
| **Vector Database** | Qdrant (self-hosted via Docker) |
| **Query Cache** | Redis (SHA-256 keyed, 1-hour TTL) |
| **Knowledge Source** | Wikipedia (via `wikipedia-api`) |
| **LLM / Lesson Generator** | `Qwen/Qwen2.5-1.5B-Instruct` (HuggingFace Transformers, greedy decoding) |
| **Streaming** | Server-Sent Events (SSE) via FastAPI `StreamingResponse` |
| **UI** | Streamlit (`ui.py`) |
| **Containerization** | Docker + Docker Compose |
| **Package Manager (Docker)** | `uv` (fast pip replacement from Astral) |

---

## Architecture

```
+-------------------------------------------------------------------+
|                           CLIENT                                  |
|         Streamlit UI  -OR-  direct HTTP / curl                   |
|   Sends: audio file (.wav/.mp3/etc.) + optional wiki_topic        |
+-----------------------------+-------------------------------------+
                              |  POST /process-audio
                              |  POST /process-audio-stream  (SSE)
                              v
+-------------------------------------------------------------------+
|                    FastAPI  (port 8000)                           |
|                       app/main.py                                 |
|    asyncio.to_thread() offloads all blocking work                 |
+----------+---------------------------------------+-----------------+
           |                                       |
           v                                       v
+--------------------+              +---------------------------------+
|   RAGPipeline      |              |   POST /index-wiki              |
|   pipeline.py      |              |   (standalone Wikipedia         |
+----+---------------+              |    article ingestion)           |
     |                              +---------------------------------+
     |  Step 1: Speech-to-Text
     v
+---------------------------+
|   SpeechToTextService     |
|   faster-whisper (base)   |
|   -> transcribed_text     |
|   -> detected_language    |
+-------------+-------------+
              |
              |  Step 2: Context Retrieval
              v
+-------------------------+    +-----------+    +-------------------+
| WikipediaRetrieverSvc   |<-->|   Redis   |    |  Qdrant Vector DB |
| embed query (384-dim)   |    | (1-hr TTL |<-->|  wikipedia_chunks |
| -> top-3 relevant chunks|    |  cache)   |    |  cosine similarity|
+-------------+-----------+    +-----------+    +-------------------+
              |
              |  Step 3: Lesson Generation
              v
+---------------------------+
|   LLMGeneratorService     |
|   Qwen2.5-1.5B-Instruct  |
|   -> translation          |
|   -> grammar breakdown    |
|   -> vocabulary analysis  |
|   -> context explanation  |
+-------------+-------------+
              |
              v
   JSON response  -OR-  SSE token stream  -> CLIENT
```

---

## Project Structure

```
de-en-rag-app/
|
+-- app/                          # Main application package
|   +-- main.py                   # FastAPI app, routes, lifespan
|   +-- db/
|   |   +-- qdrant_client.py      # Qdrant connection + collection init
|   +-- schemas/
|   |   +-- rag.py                # Pydantic request/response models
|   +-- services/
|   |   +-- stt_service.py        # Whisper speech-to-text
|   |   +-- retriever.py          # Wikipedia fetch + Qdrant upsert/search
|   |   +-- generator.py          # Qwen LLM prompt builder + lesson generator
|   |   +-- pipeline.py           # Orchestrates STT -> Retrieval -> Generation
|   |   +-- cache.py              # Redis-backed query result cache
|   +-- utils/                    # Reserved for future helpers
|
+-- config/
|   +-- settings.py               # Pydantic settings (env-driven config)
|
+-- data/
|   +-- audio_cache/              # Temp storage for uploaded audio files
|   +-- qdrant_storage/           # Persistent Qdrant vector DB data (Docker volume)
|
+-- docker/
|   +-- Dockerfile.api            # Dockerfile for the FastAPI service
|
+-- ui.py                         # Streamlit frontend UI
+-- docker-compose.yml            # Orchestrates API + Qdrant + Redis services
+-- requirements.txt              # Python dependencies
+-- .env                          # Environment variable overrides
```

---

## Full Pipeline Walkthrough

### Step 0 — Startup (`lifespan` in `main.py`)

On boot, the app:
1. Creates `data/audio_cache/` if it doesn't exist.
2. Connects to Qdrant and creates the **`wikipedia_chunks`** collection if absent:
   - Vector size: **384 dimensions**
   - Distance metric: **Cosine similarity**
   - HNSW index: `m=16`, `ef_construct=100`
   - Vectors pinned in RAM (`on_disk=False`) for fast ANN search

---

### Step 1 — Speech-to-Text (`stt_service.py`)

The uploaded audio is saved to `data/audio_cache/` under a UUID filename (deleted in the `finally` block after processing).

`faster-whisper` (CTranslate2-optimised Whisper) transcribes the audio:
- **Model**: `base` (configurable via `WHISPER_MODEL`)
- **Device**: CPU with `int8` quantization
- **Threads**: 4 CPU threads
- **Language detection**: automatic — returns ISO 639-1 code (`"de"`, `"en"`, etc.)
- **Returns**: `(transcribed_text: str, detected_language: str)`

---

### Step 2 — Context Retrieval (`retriever.py` + `cache.py`)

The transcribed text is embedded and searched against the Qdrant vector DB.

**Redis cache check:**
1. Normalise query → SHA-256 hash → check Redis
2. **Cache HIT** → return stored chunks immediately (skips embedding + Qdrant)
3. **Cache MISS** → encode query, run Qdrant ANN search, store result in Redis (TTL: 1 hour), return

**Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` maps German and English into the same 384-dim vector space, enabling cross-lingual retrieval (a German query can match an English Wikipedia article and vice versa).

**Qdrant search**: cosine similarity, returns **top 3** most relevant paragraph chunks.

Each stored chunk payload:
```json
{
  "topic": "German grammar",
  "language": "en",
  "chunk_index": 0,
  "text": "German is a West Germanic language...",
  "url": "https://en.wikipedia.org/wiki/German_grammar"
}
```

> **Note:** If `wiki_topic` is provided in the request, the pipeline first indexes that article before searching.

---

### Step 3 — Wikipedia Indexing (`retriever.py`)

When `fetch_and_index_topic()` is called (via `/index-wiki` or inline from `/process-audio`):

1. Fetches the article via `wikipedia-api`
2. Splits the full text by `\n\n`, filters paragraphs shorter than 50 characters
3. Takes the **first 15 paragraphs** to keep indexing time bounded
4. Batch-encodes all paragraphs with the multilingual embedding model
5. Upserts `PointStruct` objects into Qdrant with UUID IDs (safe to re-run)

---

### Step 4 — Lesson Generation (`generator.py`)

`LLMGeneratorService` loads `Qwen/Qwen2.5-1.5B-Instruct`:

1. **Direction detection**: `detected_lang == "de"` → translate to English; else → translate to German.
2. **Prompt construction** via `tokenizer.apply_chat_template()`:
   - **System**: Bilingual tutor persona — must rely only on provided context for facts.
   - **User**: Wikipedia context chunks + transcribed sentence + four explicit tasks:
     - Translation
     - Grammar & sentence structure (word order, verb tenses, grammatical cases, genders)
     - Vocabulary breakdown
     - Context explanation linking the sentence to retrieved Wikipedia facts
3. **Decoding**: Greedy (`do_sample=False`) — deterministic, max 768 new tokens.

Two generation modes:
- **`generate_lesson()`** — blocking, returns full string (used by `/process-audio`)
- **`generate_lesson_stream()`** — uses `TextIteratorStreamer` + background thread, yields tokens one-by-one (used by `/process-audio-stream` SSE endpoint)

---

## Services Deep Dive

### `SpeechToTextService`
| Property | Value |
|---|---|
| Model | `faster-whisper` base |
| Device | CPU, `int8` quantization, 4 threads |
| Language detection | Automatic (returns ISO code: `"de"`, `"en"`, …) |
| Input | Absolute path to audio file |
| Output | `(text: str, language_code: str)` |

### `WikipediaRetrieverService`
| Property | Value |
|---|---|
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector dimensions | 384 |
| Search results | Top 3 by cosine similarity |
| Chunk limit | First 15 paragraphs per article |
| Min chunk length | 50 characters |
| Cache backend | Redis (SHA-256 key, 1-hour TTL) |

### `LLMGeneratorService`
| Property | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (env configurable) |
| Max new tokens | 768 |
| Decoding | Greedy (`do_sample=False`) — deterministic |
| Precision | `bfloat16` (CPU memory-efficient) |
| Device | Auto (CUDA if available, else CPU) |
| Streaming | `TextIteratorStreamer` + background thread |

### `RAGPipeline`
Thin orchestrator that sequences STT → optional wiki-index → Qdrant retrieval → LLM generation.
Single public method: `process_audio_learning_session(audio_path, wiki_topic)`.

### Redis Cache (`cache.py`)
| Property | Value |
|---|---|
| Key format | `rag:ctx:<sha256(query.lower())>` |
| TTL | 3600 seconds (1 hour) |
| Graceful degradation | Falls back to live Qdrant search if Redis is unreachable |
| Failure behaviour | Logged at `WARNING` level; never raises to caller |

---

## Configuration

All settings live in `config/settings.py` and are overridable via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `German-English RAG Tutor` | Application display name |
| `DEBUG` | `True` | Debug mode flag |
| `QDRANT_HOST` | `qdrant` | Qdrant service hostname (Docker service name) |
| `QDRANT_PORT` | `6333` | Qdrant REST API port |
| `REDIS_HOST` | `redis` | Redis service hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace causal LM for lesson generation |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `AUDIO_CACHE_DIR` | `data/audio_cache` | Temp directory for uploaded audio files |

> **Tip**: For better lesson quality, switch `LLM_MODEL` to `Qwen/Qwen2.5-7B-Instruct` if you have sufficient RAM/VRAM.

---

## Setup & Running

### Prerequisites
- Docker & Docker Compose
- (Optional) NVIDIA GPU with CUDA drivers for faster LLM inference

### 1. Clone & Configure

```bash
git clone <repo-url>
cd de-en-rag-app
```

Edit `.env` if needed (the defaults work out of the box with Docker Compose):

```env
APP_NAME="German-English RAG Tutor"
QDRANT_HOST=qdrant
QDRANT_PORT=6333
REDIS_HOST=redis
REDIS_PORT=6379
EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WHISPER_MODEL=base
```

### 2. Start All Services

```bash
docker compose up --build
```

This starts three containers:

| Container | Service | Port |
|---|---|---|
| `rag_qdrant` | Qdrant vector DB | `6333` (REST), `6334` (gRPC) |
| `rag_redis` | Redis cache | `6379` |
| `rag_app_api` | FastAPI application | `8000` |

> **First run** downloads HuggingFace models (~1–2 GB). Subsequent restarts are instant thanks to the `hf_cache` Docker volume.

### 3. Verify the API

```bash
curl http://localhost:8000/
```

Expected:
```json
{
  "status": "online",
  "app_name": "German-English RAG Tutor",
  "qdrant_host": "qdrant"
}
```

### 4. Launch the Streamlit UI

In a separate terminal (with Python + Streamlit installed):

```bash
pip install streamlit requests
streamlit run ui.py
```

Opens at **http://localhost:8501**.

### 5. Access Interactive API Docs

Swagger UI: **http://localhost:8000/docs**

---

## API Endpoints

### `GET /`
**Health check** — confirms the API is online.

**Response `200 OK`:**
```json
{
  "status": "online",
  "app_name": "German-English RAG Tutor",
  "qdrant_host": "qdrant"
}
```

---

### `POST /index-wiki`

Pre-load a Wikipedia article into Qdrant so its content is available as context during audio processing.

**Content-Type:** `application/json`

**Request body:**
```json
{
  "topic": "German grammar",
  "language": "en"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `topic` | `string` | Yes | Wikipedia article title |
| `language` | `string` | No (default `"en"`) | Wikipedia language edition: `"en"` or `"de"` |

**Success `201 Created`:**
```json
{
  "status": "success",
  "message": "Successfully indexed Wikipedia article 'German grammar' into Qdrant!"
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `404 Not Found` | Wikipedia article title doesn't exist |
| `500 Internal Server Error` | Qdrant connection failure or embedding model error |

**curl example:**
```bash
curl -X POST http://localhost:8000/index-wiki \
  -H "Content-Type: application/json" \
  -d '{"topic": "German grammar", "language": "en"}'
```

---

### `POST /process-audio`

The core endpoint. Accepts an audio recording, runs it through the full RAG pipeline, and returns a complete structured lesson as JSON.

**Content-Type:** `multipart/form-data`

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Audio file — `.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`, `.mp4` |
| `wiki_topic` | `string` | No | Topic fetched and indexed into Qdrant **before** the vector search |

**Internal flow:**
1. Save audio to `data/audio_cache/` with a UUID filename
2. Whisper transcribes + auto-detects language
3. If `wiki_topic` given → fetch & index that Wikipedia article
4. Embed transcription → Qdrant search → top-3 chunks (Redis-cached)
5. Qwen LLM generates the full lesson
6. Temp audio file deleted
7. Full JSON result returned

**Success `200 OK`:**
```json
{
  "status": "success",
  "transcription": "Ich lerne gerne neue Sprachen",
  "detected_language": "de",
  "retrieved_context": [
    {
      "score": 0.87,
      "topic": "German language",
      "text": "German is a West Germanic language in the Indo-European language family...",
      "url": "https://en.wikipedia.org/wiki/German_language"
    }
  ],
  "lesson": "### Translation\n\"I enjoy learning new languages.\"\n\n### Grammar..."
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `400 Bad Request` | Unsupported audio file format |
| `422 Unprocessable Entity` | Audio was silent or transcription returned empty |
| `500 Internal Server Error` | Whisper, Qdrant, or LLM failure |

**curl:**
```bash
curl -X POST http://localhost:8000/process-audio \
  -F "file=@my_audio.wav" \
  -F "wiki_topic=German grammar"
```

**Python:**
```python
import requests

with open("my_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process-audio",
        files={"file": ("my_audio.wav", f, "audio/wav")},
        data={"wiki_topic": "German grammar"},
    )

data = response.json()
print("Transcription:", data["transcription"])
print("Detected language:", data["detected_language"])
print("\nLesson:\n", data["lesson"])
```

---

### `POST /process-audio-stream`

Identical inputs to `/process-audio`, but streams the LLM output token-by-token via **Server-Sent Events (SSE)** for real-time lesson delivery.

**Content-Type:** `multipart/form-data` (same fields as `/process-audio`)

**SSE event format:**

```
data: {"token": "###"}

data: {"token": " Translation"}

...

data: {"event": "done", "transcription": "...", "detected_language": "de", "retrieved_context": [...]}
```

- Every `{"token": "..."}` event is one piece of generated text.
- The final `{"event": "done"}` event carries transcription + retrieved context metadata.

**Python streaming example:**
```python
import requests, json

with requests.post(
    "http://localhost:8000/process-audio-stream",
    files={"file": ("audio.wav", open("audio.wav", "rb"), "audio/wav")},
    stream=True,
    timeout=300,
) as r:
    for raw_line in r.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8")
        if not line.startswith("data:"):
            continue
        payload = json.loads(line[5:].strip())
        if "token" in payload:
            print(payload["token"], end="", flush=True)
        elif payload.get("event") == "done":
            print("\n\nTranscription:", payload["transcription"])
```

---

## Streamlit UI

`ui.py` provides a browser-based frontend that wraps the SSE streaming endpoint.

**Features:**
- **Sidebar** — Index any Wikipedia topic into Qdrant with a single click.
- **Live microphone recording** — Record directly in the browser using Streamlit's `audio_input` widget.
- **File upload** — Upload `.wav`, `.mp3`, `.m4a`, or `.ogg` files.
- **Optional dynamic topic** — Inject a Wikipedia topic on-the-fly alongside your audio.
- **Real-time streaming** — Lesson text streams word-by-word via `st.write_stream()`.
- **Post-stream metadata** — Transcription, detected language, and Qdrant context chunks displayed after completion (with Wikipedia links and similarity scores).

**Launch:**
```bash
streamlit run ui.py
```

Opens at **http://localhost:8501**. Requires the FastAPI backend (`docker compose up`) to be running at `http://localhost:8000`.

---

## Data Models

### `WikiIndexRequestSchema`
```python
topic: str       # Wikipedia article title (e.g., "German grammar")
language: str    # "en" or "de" — default "en"
```

### `WikiIndexResponseSchema`
```python
status: str      # "success"
message: str     # Human-readable result description
```

### `ContextChunkSchema`
A single retrieved Wikipedia paragraph chunk:
```python
score: float          # Cosine similarity score (0.0-1.0, higher = more relevant)
topic: Optional[str]  # Wikipedia article title
text: str             # The retrieved paragraph content
url: Optional[str]    # Full Wikipedia article URL
```

### `RAGResponseSchema`
Full output from `/process-audio`:
```python
status: str                                # "success" or "error"
transcription: str                         # Text transcribed from audio
detected_language: str                     # ISO 639-1 code: "de", "en", etc.
retrieved_context: List[ContextChunkSchema]# Top-3 relevant Wikipedia chunks
lesson: str                                # Full generated lesson (markdown)
```

---

## Performance & Expectations

### First Run
Models are downloaded from HuggingFace on first startup (~1-2 GB total). Subsequent restarts are instant thanks to the `hf_cache` Docker volume.

### Response Times

| Condition | `/process-audio` (batch) | `/process-audio-stream` (first token) |
|---|---|---|
| CPU-only | 30-120 seconds | ~5-15 seconds |
| CUDA GPU | 5-20 seconds | < 5 seconds |
| Redis cache HIT | Saves ~1-3 seconds on retrieval | Same savings |

> The Qdrant collection must have indexed articles for context-rich lessons. An empty DB still works — the LLM just has no factual grounding.

### Lesson Quality

- **With Wikipedia context indexed**: Lessons include factual, topic-relevant explanations.
- **Without context (empty DB)**: Grammar and translation still work; "Context Explanation" will be generic.
- **Language direction is automatic**: Speak German → get English lesson. Speak English → get German lesson.

### Recommended Workflow

1. **Index topics** — Use the Streamlit sidebar or `POST /index-wiki` for 2-3 relevant topics (e.g., "German grammar", "Berlin", "Goethe").
2. **Record** — Use the Streamlit mic widget or upload an audio file.
3. **Get your lesson** — Tokens stream in real-time.
4. **Iterate** — The Qdrant DB persists across restarts via the Docker volume.

### Qdrant Dashboard

Qdrant ships with a built-in web UI at **http://localhost:6333/dashboard** — browse collections, inspect vector payloads, and run test searches.
