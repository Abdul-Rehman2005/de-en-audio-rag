import logging
import os
from typing import Tuple
from faster_whisper import WhisperModel
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextService:
    def __init__(self):
        logger.info(f"Loading Whisper model '{settings.WHISPER_MODEL}'...")

        self.model = WhisperModel(
            settings.WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=4
        )
        logger.info("Whisper model loaded successfully!")

    def transcribe(self, audio_path: str) -> Tuple[str, str]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        logger.info(f"Transcribing audio from: {audio_path}")

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language=None
        )
        transcribed_text = " ".join([segment.text for segment in segments]).strip()
        detected_language = info.language

        logger.info(f"Detected language: {detected_language} (Probability: {info.language_probability:.2f})")
        logger.info(f"Transcription result: '{transcribed_text}'")

        return transcribed_text, detected_language       

stt_service = SpeechToTextService()

