import logging
import os
from typing import Dict, Any

from app.services.stt_service import stt_service
from app.services.retriever import retriever_service
from app.services.generator import generator_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self) -> None:
        self.stt = stt_service
        self.retriever = retriever_service
        self.generator = generator_service

    def process_audio_learning_session(self, audio_path: str, wiki_topic: str = None) -> Dict[str, Any]:
        logger.info("--- Starting RAG Audio Processing Session ---")

        transcribed_text, detected_lang = self.stt.transcribe(audio_path)

        if not transcribed_text:
            return {
                "status": "error",
                "message": "Speech transcription failed or audio was silent."
            }

        if wiki_topic:
            logger.info(f"Indexing requested topic: '{wiki_topic}'...")
            self.retriever.fetch_and_index_topic(topic=wiki_topic, language=detected_lang)

        context_chunks = self.retriever.search_context(query=transcribed_text, top_k=3)
        lesson_output = self.generator.generate_lesson(
            user_text=transcribed_text,
            detected_lang=detected_lang,
            context_chunks=context_chunks
        )
        logger.info("--- RAG Session Completed Successfully ---")
        return {
            "status": "success",
            "transcription": transcribed_text,
            "detected_language": detected_lang,
            "retrieved_context": context_chunks,
            "lesson": lesson_output
        }

rag_pipeline = RAGPipeline()