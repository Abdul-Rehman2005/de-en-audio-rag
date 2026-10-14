from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ContextChunkSchema(BaseModel):
    """Schema representing a retrieved Wikipedia context chunk."""
    score: float = Field(..., description="Cosine similarity score from Qdrant")
    topic: Optional[str] = Field(None, description="Title of the Wikipedia article")
    text: str = Field(..., description="The context paragraph snippet")
    url: Optional[str] = Field(None, description="URL of the Wikipedia article source")


class RAGResponseSchema(BaseModel):
    """Schema for the final lesson output returned to the client."""
    status: str = Field(..., example="success")
    transcription: str = Field(..., description="Transcribed text from user audio")
    detected_language: str = Field(..., description="Language code detected by Whisper (e.g., 'de' or 'en')")
    retrieved_context: List[ContextChunkSchema] = Field(
        default_factory=list, 
        description="List of Wikipedia chunks retrieved from Qdrant"
    )
    lesson: str = Field(..., description="Generated translation, grammar breakdown, and explanation")


class WikiIndexRequestSchema(BaseModel):
    """Schema for manually triggering Wikipedia article indexing."""
    topic: str = Field(
        ..., 
        description="Wikipedia topic/title to index",
        json_schema_extra={"example": "German grammar"}
    )
    language: str = Field(
        default="en", 
        description="Language version of Wikipedia ('en' or 'de')",
        json_schema_extra={"example": "en"}
    )

class WikiIndexResponseSchema(BaseModel):
    """Response schema following a Wikipedia indexing operation."""
    status: str = Field(..., example="success")
    message: str = Field(..., description="Details regarding the indexing result")