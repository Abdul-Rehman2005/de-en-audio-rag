import asyncio
import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config.settings import settings
from app.db.qdrant_client import init_qdrant_db
from app.services.retriever import retriever_service
from app.services.pipeline import rag_pipeline
from app.services.generator import generator_service
from app.schemas.rag import RAGResponseSchema, WikiIndexRequestSchema, WikiIndexResponseSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up German-English RAG Tutor API...")
    try:
        # Ensure temporary audio cache directory exists on boot
        os.makedirs(settings.AUDIO_CACHE_DIR, exist_ok=True)
        init_qdrant_db()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Speech-driven RAG Application for German and English Language Learning",
    version="1.0.0",
    lifespan=lifespan
)

# Allow React Dev Servers (Vite on 5173 / CRA on 3000) and general clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Health Check")
def read_root():
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "qdrant_host": settings.QDRANT_HOST
    }


# ── Phase 3: async def + asyncio.to_thread() ─────────────────────────────────
# Heavy CPU-bound work (embedding, Qdrant I/O, Wikipedia fetch) is dispatched to
# a worker thread via asyncio.to_thread() so the event loop stays unblocked.
@app.post(
    "/index-wiki",
    response_model=WikiIndexResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Index a Wikipedia Topic into Qdrant"
)
async def index_wikipedia_topic(payload: WikiIndexRequestSchema):
    try:
        success = await asyncio.to_thread(
            retriever_service.fetch_and_index_topic,
            payload.topic,
            payload.language,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Wikipedia page '{payload.topic}' could not be found or processed."
            )
        return {
            "status": "success",
            "message": f"Successfully indexed Wikipedia article '{payload.topic}' into Qdrant!"
        }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Failed to index topic: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/process-audio",
    response_model=RAGResponseSchema,
    summary="Process Audio to Transcribe, Retrieve Context, and Generate Lesson"
)
async def process_audio(
    file: UploadFile = File(...),
    wiki_topic: Optional[str] = Form(None)
):
    """
    Phase 3: all blocking calls dispatched to thread-pool via asyncio.to_thread()
    so the FastAPI event loop is never stalled.
    """
    allowed_extensions = [".wav", ".mp3", ".ogg", ".m4a", ".flac", ".mp4"]
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )

    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_file_path = os.path.join(settings.AUDIO_CACHE_DIR, temp_filename)

    try:
        # Offload file-write to thread (avoids blocking event loop on large uploads)
        def _save_upload():
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        await asyncio.to_thread(_save_upload)

        # Offload the entire blocking pipeline (STT + embed + Qdrant + LLM)
        result = await asyncio.to_thread(
            rag_pipeline.process_audio_learning_session,
            temp_file_path,
            wiki_topic,
        )

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.get("message")
            )

        return result

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process audio: {str(e)}"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ── Phase 3: Server-Sent Events streaming endpoint ───────────────────────────
async def _sse_token_generator(
    audio_path: str,
    wiki_topic: Optional[str],
) -> AsyncGenerator[str, None]:
    """
    Async generator that:
      1. Runs Whisper STT + optional wiki-index + Redis/Qdrant retrieval in a thread.
      2. Streams LLM tokens one-by-one as SSE events.
      3. Emits a final [DONE] sentinel event with transcription metadata.
    """
    # ── Step 1: Transcribe (blocking) ─────────────────────────────────────────
    transcribed_text, detected_lang = await asyncio.to_thread(
        rag_pipeline.stt.transcribe, audio_path
    )

    if not transcribed_text:
        yield "data: " + json.dumps({"error": "Transcription failed or audio was silent."}) + "\n\n"
        return

    # ── Step 2: Optional wiki index (blocking) ────────────────────────────────
    if wiki_topic:
        await asyncio.to_thread(
            rag_pipeline.retriever.fetch_and_index_topic,
            wiki_topic,
            detected_lang,
        )

    # ── Step 3: Context retrieval (Redis cache → Qdrant) ─────────────────────
    context_chunks = await asyncio.to_thread(
        rag_pipeline.retriever.search_context,
        transcribed_text,
        3,
    )

    # ── Step 4: Stream LLM tokens ─────────────────────────────────────────────
    # generate_lesson_stream() is a synchronous generator that blocks between
    # tokens; wrap the whole loop in to_thread via a thread-safe queue pattern.
    token_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _generate_tokens():
        """Run in a thread; put tokens into the asyncio queue."""
        for token in generator_service.generate_lesson_stream(
            transcribed_text, detected_lang, context_chunks
        ):
            loop.call_soon_threadsafe(token_queue.put_nowait, token)
        loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel

    gen_thread = asyncio.to_thread(_generate_tokens)
    asyncio.ensure_future(gen_thread)

    while True:
        token = await token_queue.get()
        if token is None:
            break
        yield "data: " + json.dumps({"token": token}) + "\n\n"

    # ── Step 5: Metadata sentinel event ───────────────────────────────────────
    yield "data: " + json.dumps({
        "event": "done",
        "transcription": transcribed_text,
        "detected_language": detected_lang,
        "retrieved_context": context_chunks,
    }) + "\n\n"


@app.post(
    "/process-audio-stream",
    summary="Stream LLM tokens via SSE for real-time lesson delivery",
    response_description="Server-Sent Events stream of generated tokens",
)
async def process_audio_stream(
    file: UploadFile = File(...),
    wiki_topic: Optional[str] = Form(None)
):
    """
    Phase 3 — SSE streaming endpoint.

    Yields events in the following format::

        data: {"token": "..."}\n\n          ← one per LLM token
        data: {"event":"done", "transcription":"...", ...}\n\n

    The client should use the EventSource API or fetch() with a ReadableStream.
    """
    allowed_extensions = [".wav", ".mp3", ".ogg", ".m4a", ".flac", ".mp4"]
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )

    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_file_path = os.path.join(settings.AUDIO_CACHE_DIR, temp_filename)

    def _save_upload():
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    await asyncio.to_thread(_save_upload)

    async def _stream_with_cleanup():
        try:
            async for chunk in _sse_token_generator(temp_file_path, wiki_topic):
                yield chunk
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return StreamingResponse(
        _stream_with_cleanup(),
        media_type="text/event-stream",
        headers={
            # Disable buffering at every proxy layer so tokens arrive immediately
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )