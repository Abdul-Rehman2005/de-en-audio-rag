import logging
import uuid 
from typing import List, Dict, Any
import wikipediaapi
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models

from config.settings import settings
from app.db.qdrant_client import qdrant_client, WIKI_COLLECTION_NAME
from app.services.cache import get_cached_context, set_cached_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaRetrieverService:
    def __init__(self) -> None:
        logger.info(f"Loading embedding model '{settings.EMBEDDING_MODEL}'...")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully!")

    def _get_wiki_client(self, language: str = "en") -> wikipediaapi.Wikipedia:
        return wikipediaapi.Wikipedia(
            user_agent="GermanEnglishRAGTutor/1.0 (learning_project)",
            language=language
        )

    def fetch_and_index_topic(self, topic: str, language: str = "en") -> bool:
        wiki = self._get_wiki_client(language=language)
        page = wiki.page(topic)
        if not page.exists():
            logger.warning(f"Wikipedia page '{topic}' in language '{language}' does not exist.")
            return False

        logger.info(f"Fetched Wikipedia page: '{page.title}'. Splitting into chunks...")
                
        text = page.text
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        if not raw_paragraphs:
            return False

        target_paragraphs = raw_paragraphs[:15]
        
        vectors = self.embedding_model.encode(target_paragraphs).tolist()

        points = []
        for index, paragraph in enumerate(target_paragraphs):
            point_id = str(uuid.uuid4())
            payload = {
                "topic": page.title,
                "language": language,
                "chunk_index": index,
                "text": paragraph,
                "url": page.fullurl
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors[index],
                    payload=payload
                )
            )

        qdrant_client.upsert(
            collection_name=WIKI_COLLECTION_NAME,
            points=points
        )
        logger.info(f"Successfully indexed {len(points)} chunks for topic '{page.title}' in Qdrant!")
        return True

    def search_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Return top-k Wikipedia chunks for *query*.

        Cache flow (Phase 4):
          1. Check Redis — if HIT, return cached chunks immediately (skips embed + Qdrant).
          2. On MISS: encode query → Qdrant ANN search → store result in Redis → return.
        """
        # ── Phase 4: Redis cache check ──────────────────────────────────────────
        cached = get_cached_context(query)
        if cached is not None:
            return cached

        # ── Cache miss: run embedding + vector search ───────────────────────────
        logger.info(f"Generating query embedding for search: '{query}'")
        query_vector = self.embedding_model.encode(query).tolist()

        search_results = qdrant_client.search(
            collection_name=WIKI_COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        results = []
        for hit in search_results:
            results.append({
                "score": hit.score,
                "topic": hit.payload.get("topic"),
                "text": hit.payload.get("text"),
                "url": hit.payload.get("url")
            })
        logger.info(f"Retrieved {len(results)} relevant Wikipedia context chunks.")

        # ── Store in Redis for subsequent identical queries ─────────────────────
        set_cached_context(query, results)

        return results

retriever_service = WikipediaRetrieverService()